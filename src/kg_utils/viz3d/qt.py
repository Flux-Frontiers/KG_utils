"""Qt glue for light-field renders — the machinery, not the wiring.

Requires the ``viz3d-qt`` extra::

    pip install 'kgmodule-utils[viz3d-qt]'

A KG viewer that can cast to a Looking Glass ends up owning a surprising
amount of code that has nothing to do with its graph: a thread to keep an
external ray-tracer off the GUI loop, a progress bar fed by counting files, a
temp directory to clean up, a preview window, and the four-step
render-write-cast dance.  None of that expresses a domain's claim about its
data, and two repos writing it twice is how the two copies drift.

What stays in the consuming repo is the *wiring* — which node becomes a trunk,
which button is called what, where the output goes.  This module takes those
as arguments rather than assuming attribute names, so a viewer adapts by
constructing :class:`PovRenderSession` with its own widgets instead of
inheriting a mixin that has opinions about them.

Qt is imported at module scope because :class:`PovRenderWorker` and friends
subclass ``QThread`` / ``QDialog`` / ``QObject``, and a base class must exist
when the class body runs.  That is why this is a separate module rather than
lazy imports inside :mod:`kg_utils.viz3d.organic`-style functions, and why
:mod:`kg_utils.viz3d` does not re-export it: importing the layouts must not
require PyQt.

:mod:`quiltwright` and :mod:`pyvista` *are* imported lazily, inside the calls
that need them, so constructing a session costs nothing.

Author: Eric G. Suchanek, PhD

License: Elastic 2.0
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

__all__ = [
    "DEFAULT_CAST_SCALE",
    "DEFAULT_QUILT_PRESET",
    "CastResult",
    "ImagePopup",
    "PovRenderSession",
    "PovRenderWorker",
    "cast_scene_to_looking_glass",
]

#: Quilt preset cast to by default.  Which panel is physically plugged in is
#: deployment config, not a claim about anyone's data — hence a default with an
#: override rather than a constant each consumer redeclares.
DEFAULT_QUILT_PRESET: str = "16-landscape"

#: Fraction of the preset's pixel size the default cast renders at.  The local
#: render costs about a second at full size; the wait is Bridge loading the
#: resulting PNG, and its decode time scales with the image's area.  Half-size
#: is roughly a quarter of the pixels for a difference the panel does not show.
DEFAULT_CAST_SCALE: float = 0.5

#: Workers that outlived their window.  A ``QThread`` destroyed while running
#: takes the process with it, so a session that cannot stop one in time parks
#: it here — Python keeps the reference alive until the thread ends on its own.
_ORPHANED_WORKERS: set[PovRenderWorker] = set()


class PovRenderWorker(QThread):
    """Ray-trace a written ``.pov`` off the GUI thread.

    POV-Ray is an external process and a slow one — roughly 18 s for a single
    900x675 view of a mid-sized tree, and a 48-view quilt is that again per
    tile.  Calling it inline would freeze the window for the whole render,
    including the status label meant to report progress, so the work happens
    here and the window learns about it through signals.

    :param pov_path: The scene file to trace.
    :param spec: Quilt spec; a 1x1 spec renders a single image.
    :param camera: Camera in POV-Ray coordinates.
    :param views_dir: Directory the worker writes ``viewNNN.png`` into as each
        trace finishes; the caller owns it and is expected to clean it up.
    :param jobs: Parallel POV-Ray processes.
    :param antialias: POV-Ray ``+A`` threshold; ``None`` omits the flag.
    """

    #: Emitted with the assembled image once the render succeeds.
    finished_ok: pyqtSignal = pyqtSignal(object)
    #: Emitted with a human-readable reason when it does not.
    failed: pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        pov_path: Path,
        spec: Any,
        camera: Any,
        views_dir: Path,
        jobs: int = 1,
        antialias: float | None = 0.3,
    ) -> None:
        """Store the render inputs; nothing is traced until :meth:`run`."""
        super().__init__()
        self._pov_path = pov_path
        self._spec = spec
        self._camera = camera
        self._views_dir = views_dir
        self._jobs = jobs
        self._antialias = antialias

    def run(self) -> None:
        """Trace the scene, emitting :attr:`finished_ok` or :attr:`failed`."""
        try:
            import numpy as np
            from PIL import Image
            from quiltwright import assemble_quilt
            from quiltwright.povray import render_pov_views

            # render_pov_views rather than render_pov_quilt: it writes one
            # viewNNN.png into a directory the caller owns as each trace
            # finishes, which is what makes progress observable at all. The
            # quilt path keeps its views in a private temp dir, so a caster
            # waiting minutes has nothing to watch.
            paths = render_pov_views(
                self._pov_path,
                self._spec,
                self._camera,
                self._views_dir,
                jobs=self._jobs,
                antialias=self._antialias,
                progress=False,
            )
            image = assemble_quilt(
                [np.asarray(Image.open(p).convert("RGB")) for p in paths], self._spec
            )
        except FileNotFoundError as exc:
            self.failed.emit(f"No povray binary on PATH — install POV-Ray to ray-trace. ({exc})")
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller's status line
            self.failed.emit(str(exc))
        else:
            self.finished_ok.emit(image)


class PovRenderSession(QObject):
    """One ray-trace, from start to cleanup, driving a caller's widgets.

    Owns the parts a viewer would otherwise scatter across half a dozen
    methods: the worker thread, the temp directory it writes views into, the
    poll timer that turns those files into a progress bar, and the teardown
    that has to happen whether the render succeeds, fails, or is still going
    when the window closes.

    The contract is deliberately narrow — a progress bar, a status setter, and
    an optional busy setter — because the two repos that need this call their
    widgets different things (``cast_btn`` here, ``cast_button`` there).  A
    mixin would bake one repo's names into the SDK.

    **Closing the window during a render is the case worth knowing about.** A
    ``QThread`` destroyed while running aborts the process, and a queued signal
    delivered after its widgets are gone raises from deep inside Qt.  Call
    :meth:`shutdown` from the window's ``closeEvent``; it disconnects first, so
    nothing lands on a deleted widget, and parks a thread it cannot stop in
    time rather than letting Qt destroy it.

    :param parent: Widget that owns this session; also the popup's parent.
    :param progress_bar: Bar to drive.  Ranged to the view count on
        :meth:`start` and hidden when the render ends.
    :param set_status: Called with a human-readable line at each transition.
    :param set_busy: Called ``True`` when a render starts and ``False`` when it
        ends — for disabling the buttons that would start another.
    :param poll_ms: How often to count finished views.
    """

    def __init__(
        self,
        parent: QWidget,
        *,
        progress_bar: QProgressBar,
        set_status: Callable[[str], None],
        set_busy: Callable[[bool], None] | None = None,
        poll_ms: int = 400,
    ) -> None:
        """Wire the session to its widgets; nothing runs until :meth:`start`."""
        super().__init__(parent)
        self._parent = parent
        self._bar = progress_bar
        self._set_status = set_status
        self._set_busy = set_busy
        self._worker: PovRenderWorker | None = None
        self._views_dir: Path | None = None
        self._started: float | None = None
        self._on_image: Callable[[Any, Any, str], None] | None = None
        self._label = "render"
        self._spec: Any = None

        # One timer for the session's whole life. Creating it per render leaks
        # a stopped QTimer per trace: it is parented to the window, so Qt keeps
        # every one of them until the window dies.
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self._poll)

    @property
    def is_running(self) -> bool:
        """:return: Whether a trace is in flight."""
        return self._worker is not None and self._worker.isRunning()

    def start(
        self,
        pov_path: Path,
        spec: Any,
        camera: Any,
        *,
        on_image: Callable[[Any, Any, str], None],
        label: str = "render",
        jobs: int | None = None,
        antialias: float | None = 0.3,
    ) -> bool:
        """Trace *pov_path* in the background and route the result.

        :param pov_path: Scene to trace.
        :param spec: Quilt spec; 1x1 for a single-view preview.
        :param camera: Camera in POV-Ray coordinates.
        :param on_image: Called on the GUI thread as ``(image, spec, label)``
            when the trace succeeds.  What to do with a finished quilt is the
            consumer's decision, not this module's.
        :param label: Word for the status line, e.g. ``"preview"``.
        :param jobs: Parallel POV-Ray processes; defaults to all cores but one.
        :param antialias: POV-Ray ``+A`` threshold; ``None`` omits the flag.
        :return: ``False`` if a render was already in flight, else ``True``.
        """
        if self.is_running:
            self._set_status("A render is already running.")
            return False

        self._on_image = on_image
        self._label = label
        self._spec = spec
        self._set_status(
            f"POV-Ray: tracing {spec.n_views} view(s) at "
            f"{spec.tile_width}x{spec.tile_height} — this is not fast..."
        )
        if self._set_busy:
            self._set_busy(True)

        # The worker writes views here as it finishes them; the GUI thread
        # counts the files. Polling the filesystem beats threading a callback
        # out of an external process pool, and it cannot wedge the render.
        self._views_dir = Path(tempfile.mkdtemp(prefix="kgviz-pov-"))
        self._started = time.perf_counter()
        self._bar.setRange(0, spec.n_views)
        self._bar.setValue(0)
        self._bar.setFormat("%v / %m views")
        self._bar.show()
        self._timer.start()

        worker = PovRenderWorker(
            pov_path,
            spec,
            camera,
            self._views_dir,
            jobs=jobs if jobs is not None else max(1, (os.cpu_count() or 2) - 1),
            antialias=antialias,
        )
        worker.finished_ok.connect(self._on_ok)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker  # keep a reference; a GC'd QThread is a crash
        worker.start()
        return True

    def shutdown(self, *, timeout_ms: int = 5000) -> bool:
        """Stop polling and detach from a running trace, safely.

        Signals are disconnected *before* waiting, so a render that lands mid
        teardown cannot deliver into widgets that are being destroyed.  A
        worker still running after *timeout_ms* is parked in a module-level set
        rather than left for Qt to destroy, which would abort the process.

        Safe to call when nothing is running, and safe to call twice.

        :param timeout_ms: How long to wait for the trace to notice.
        :return: ``True`` if no worker is still running when this returns.
        """
        self._timer.stop()
        worker = self._worker
        self._worker = None

        if worker is not None:
            for signal in (worker.finished_ok, worker.failed, worker.finished):
                try:
                    signal.disconnect()
                except TypeError:
                    pass  # nothing was connected; Qt raises rather than no-ops
            if worker.isRunning():
                worker.requestInterruption()
                if not worker.wait(timeout_ms):
                    # POV-Ray is an external process and will not notice the
                    # interruption request; hold the reference so Python does
                    # not collect a live QThread out from under Qt.
                    _ORPHANED_WORKERS.add(worker)
                    worker.finished.connect(lambda w=worker: _ORPHANED_WORKERS.discard(w))

        # Dropping the views directory out from under a parked worker makes its
        # remaining traces fail — which is fine and deliberate: its signals are
        # already disconnected, so nothing is listening for the result, and
        # ``run`` swallows the error. Leaving the directory to be tidy would
        # leak a temp tree per abandoned render instead.
        self._cleanup_views()
        if self._set_busy:
            self._set_busy(False)
        return not (worker is not None and worker.isRunning())

    def _cleanup_views(self) -> None:
        """Delete the temp directory this session created, if it still exists."""
        views_dir = self._views_dir
        self._views_dir = None
        if views_dir is not None:
            # Only ever a directory this session created via mkdtemp.
            shutil.rmtree(views_dir, ignore_errors=True)

    def _poll(self) -> None:
        """Count finished views on disk and advance the bar.

        Cheap enough at the default interval: a directory listing against a
        render whose views take seconds each.
        """
        views_dir = self._views_dir
        if views_dir is None or not views_dir.exists():
            return
        done = len(list(views_dir.glob("view*.png")))
        self._bar.setValue(done)

        # Elapsed and ETA rather than a frame number. With jobs > 1 there is no
        # single "current frame" — roughly one trace per core is in flight, and
        # `done` is how many have landed, not which one is being worked on.
        # Extrapolating from the completed rate is the honest reading of that.
        if self._started is None or done == 0:
            return
        elapsed = time.perf_counter() - self._started
        total = self._bar.maximum()
        eta = elapsed / done * (total - done)
        self._bar.setFormat(f"%v / %m views  ·  {elapsed:.0f}s, ~{eta:.0f}s left")

    def _on_ok(self, image: Any) -> None:
        """Hand a finished image to the consumer's callback."""
        if self._on_image is not None:
            self._on_image(image, self._spec, self._label)

    def _on_failed(self, message: str) -> None:
        """Report a failed trace.

        :param message: Human-readable reason.
        """
        self._set_status(f"POV-Ray failed: {message}")

    def _on_finished(self) -> None:
        """Stop polling, hide the bar, release the buttons, drop the views."""
        self._timer.stop()
        self._bar.hide()
        self._cleanup_views()
        if self._set_busy:
            self._set_busy(False)


class ImagePopup(QDialog):
    """A simple viewer for a rendered image.

    :param title: Window title.
    :param path: Image file to display.
    :param parent: Parent widget.
    :param max_size: Bounding box the image is scaled into, aspect preserved.
    :param path_color: CSS colour for the file-location line beneath it.
    """

    def __init__(
        self,
        title: str,
        path: Path,
        parent: QWidget | None = None,
        *,
        max_size: tuple[int, int] = (1100, 800),
        path_color: str = "#90EE90",
    ) -> None:
        """Show *path* scaled to fit, with the file location under it."""
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)

        label = QLabel(self)
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            # Scoped enum names (Qt.AlignmentFlag.AlignCenter) rather than the
            # unscoped PyQt5 aliases (Qt.AlignCenter): both work here, but only
            # these resolve for a type checker, and PyQt6 dropped the aliases.
            label.setPixmap(
                pixmap.scaled(
                    min(pixmap.width(), max_size[0]),
                    min(pixmap.height(), max_size[1]),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        where = QLabel(str(path), self)
        where.setStyleSheet(f"color:{path_color}; font-size:11px;")
        where.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(where)

        close_btn = QPushButton("Close", self)
        # reject() over close(): the dialog's own dismissal slot, and it returns
        # None where close() returns bool, which a signal connection rejects.
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)


@dataclass(frozen=True)
class CastResult:
    """What came of one cast, including the line to put on a status bar.

    A cast has three outcomes, not two: it can fail before anything is written,
    write a quilt that Bridge then refuses to display, or succeed.  Both viewers
    branched on ``(path, error)`` to say the same three things in the same
    words, so the wording lives here and :attr:`message` is what a caller shows.

    :param path: The quilt written, or ``None`` if the render itself failed.
    :param error: Why it failed; ``None`` on a clean cast.  Set alongside a
        *path* when the file was written but the display did not happen.
    :param elapsed: Wall-clock seconds the whole cast took.
    :param message: Human-readable summary of the three cases above.
    """

    path: Path | None
    error: str | None
    elapsed: float
    message: str


def cast_scene_to_looking_glass(
    build_scene: Callable[[Any], object],
    camera_position: Any,
    out_stem: str | Path,
    spec: Any | None = None,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> CastResult:
    """Render a scene off-screen as a quilt and push it to the Looking Glass.

    The PyVista path, as distinct from the ray-traced one: composes into a
    fresh off-screen plotter, copies the caller's camera so what is cast is the
    view being looked at, then writes and casts.  Fast enough to run inline on
    the GUI thread, which is why this is a function and not a
    :class:`PovRenderSession`.

    What is left to the caller is what is genuinely its own — which nodes to
    draw, where the file lands, which button to grey out while it happens.

    :param build_scene: Called with the off-screen ``pv.Plotter`` to compose
        into.  Which nodes become what is the consumer's business.  Its return
        value is ignored, so a builder that returns its plotter can be passed
        directly rather than wrapped.
    :param camera_position: ``camera_position`` copied from the live viewport.
    :param out_stem: Output path stem; the quilt suffix is appended.
    :param spec: Quilt spec to render at.  Defaults to
        :data:`DEFAULT_QUILT_PRESET` scaled by :data:`DEFAULT_CAST_SCALE`.
    :param progress: Called as ``(step, total, message)`` before each stage,
        for a status bar.  A Qt caller should pump its event loop here.
    :return: A :class:`CastResult`; nothing here raises, because a dark panel
        must not take the viewer with it.
    """
    import pyvista as pv
    from quiltwright import QUILT_PRESETS, render_quilt, save_and_cast_quilt

    if spec is None:
        spec = QUILT_PRESETS[DEFAULT_QUILT_PRESET].scaled(DEFAULT_CAST_SCALE)

    def _step(n: int, message: str) -> None:
        if progress:
            progress(n, 4, message)

    started = time.perf_counter()
    offscreen = pv.Plotter(off_screen=True)
    try:
        _step(1, "building scene...")
        build_scene(offscreen)
        offscreen.camera_position = camera_position

        _step(2, f"rendering {spec.n_views} views at {spec.tile_width}x{spec.tile_height}...")
        quilt = render_quilt(offscreen, spec)

        _step(3, f"writing {spec.quilt_width}x{spec.quilt_height} quilt...")
        _step(4, "handing to Bridge...")
        path, error = save_and_cast_quilt(quilt, out_stem, spec)
    except Exception as exc:  # noqa: BLE001 - a dark panel must not kill the viewer
        path, error = None, str(exc)
    finally:
        offscreen.close()

    elapsed = time.perf_counter() - started
    if path is None:
        message = f"Cast failed (is Bridge running?): {error}"
    elif error:
        # The quilt is on disk; only the display is missing.
        message = f"Wrote {path.name}, casting failed: {error}"
    else:
        message = f"Cast {path.name} in {elapsed:.1f}s"
    return CastResult(path=path, error=error, elapsed=elapsed, message=message)
