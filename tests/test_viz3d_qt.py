"""Tests for kg_utils.viz3d.qt — the Qt render lifecycle.

These never invoke POV-Ray.  What is worth pinning here is not that a trace
produces pixels — quiltwright covers that — but that the *lifecycle* around it
behaves: that a session cleans up after itself, that closing a window during a
render cannot deliver a signal into destroyed widgets, and that a worker which
will not stop in time is held rather than dropped.

The last one is the reason this module exists.  A ``QThread`` garbage-collected
or destroyed while running aborts the interpreter, and that failure only shows
up when someone closes the viewer during a multi-minute ray-trace.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="kg_utils.viz3d.qt needs the viz3d-qt extra")

from PyQt5.QtWidgets import QApplication, QLabel, QProgressBar, QWidget  # noqa: E402

from kg_utils.viz3d.qt import (  # noqa: E402
    _ORPHANED_WORKERS,
    ImagePopup,
    PovRenderSession,
    PovRenderWorker,
    cast_scene_to_looking_glass,
)


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the session; Qt allows only one."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def harness(qapp, tmp_path):
    """A session wired to throwaway widgets, plus the calls it made."""

    class Harness:
        def __init__(self):
            self.parent = QWidget()
            self.bar = QProgressBar(self.parent)
            self.statuses: list[str] = []
            self.busy: list[bool] = []
            self.images: list[tuple] = []
            self.session = PovRenderSession(
                self.parent,
                progress_bar=self.bar,
                set_status=self.statuses.append,
                set_busy=self.busy.append,
                poll_ms=10,
            )

    h = Harness()
    yield h
    h.session.shutdown()
    h.parent.deleteLater()


class TestSessionLifecycle:
    def test_starts_idle(self, harness):
        assert harness.session.is_running is False

    def test_shutdown_on_an_idle_session_is_safe(self, harness):
        assert harness.session.shutdown() is True

    def test_shutdown_is_idempotent(self, harness):
        harness.session.shutdown()
        assert harness.session.shutdown() is True

    def test_shutdown_stops_the_poll_timer(self, harness):
        harness.session._timer.start()
        harness.session.shutdown()
        assert harness.session._timer.isActive() is False

    def test_one_timer_for_the_session(self, harness):
        """A timer per render leaks one stopped QTimer per trace."""
        timer = harness.session._timer
        harness.session.shutdown()
        assert harness.session._timer is timer


class TestProgressPolling:
    def test_counts_finished_views(self, harness, tmp_path):
        harness.session._views_dir = tmp_path
        harness.bar.setRange(0, 4)
        for i in range(3):
            (tmp_path / f"view{i:03d}.png").touch()
        harness.session._poll()
        assert harness.bar.value() == 3

    def test_ignores_non_view_files(self, harness, tmp_path):
        harness.session._views_dir = tmp_path
        harness.bar.setRange(0, 4)
        (tmp_path / "view000.png").touch()
        (tmp_path / "scene.pov").touch()
        (tmp_path / "notes.txt").touch()
        harness.session._poll()
        assert harness.bar.value() == 1

    def test_a_missing_views_dir_is_not_an_error(self, harness, tmp_path):
        harness.session._views_dir = tmp_path / "gone"
        harness.session._poll()  # must not raise

    def test_eta_appears_only_after_a_view_lands(self, harness, tmp_path):
        harness.session._views_dir = tmp_path
        harness.session._started = 0.0
        harness.bar.setRange(0, 4)
        harness.session._poll()
        assert "left" not in harness.bar.format()
        (tmp_path / "view000.png").touch()
        harness.session._poll()
        assert "left" in harness.bar.format()


class TestCleanup:
    def test_finish_removes_the_views_dir(self, harness, tmp_path):
        views = tmp_path / "views"
        views.mkdir()
        harness.session._views_dir = views
        harness.session._on_finished()
        assert not views.exists()

    def test_shutdown_removes_the_views_dir(self, harness, tmp_path):
        views = tmp_path / "views"
        views.mkdir()
        harness.session._views_dir = views
        harness.session.shutdown()
        assert not views.exists()

    def test_finish_releases_the_busy_flag(self, harness):
        harness.session._on_finished()
        assert harness.busy[-1] is False


class TestShutdownDetachesFromAWorker:
    """The crash-on-close case: a queued signal must not outlive the window."""

    def test_a_late_success_does_not_reach_the_callback(self, harness, qapp):
        worker = PovRenderWorker(None, None, None, None)  # never started
        harness.session._worker = worker
        harness.session._on_image = lambda *a: harness.images.append(a)
        worker.finished_ok.connect(harness.session._on_ok)

        harness.session.shutdown()
        worker.finished_ok.emit(object())  # the render lands after teardown
        qapp.processEvents()

        assert harness.images == [], "a detached session must not be called back"

    def test_a_late_failure_does_not_reach_the_status_line(self, harness, qapp):
        worker = PovRenderWorker(None, None, None, None)
        harness.session._worker = worker
        worker.failed.connect(harness.session._on_failed)

        harness.session.shutdown()
        before = list(harness.statuses)
        worker.failed.emit("too late")
        qapp.processEvents()

        assert harness.statuses == before

    def test_shutdown_clears_the_worker_reference(self, harness):
        harness.session._worker = PovRenderWorker(None, None, None, None)
        harness.session.shutdown()
        assert harness.session._worker is None

    def test_a_stopped_worker_is_not_orphaned(self, harness):
        """Parking is for threads still running; a finished one just goes."""
        worker = PovRenderWorker(None, None, None, None)
        harness.session._worker = worker
        harness.session.shutdown()
        assert worker not in _ORPHANED_WORKERS


class TestStartGuards:
    def test_refuses_a_second_concurrent_render(self, harness, monkeypatch):
        monkeypatch.setattr(
            PovRenderSession, "is_running", property(lambda self: True), raising=True
        )
        started = harness.session.start(None, None, None, on_image=lambda *a: None, label="preview")
        assert started is False
        assert "already running" in harness.statuses[-1]


class TestImagePopup:
    def test_a_missing_image_still_opens(self, qapp, tmp_path):
        """A popup that raises on a bad path takes the viewer with it."""
        popup = ImagePopup("gone", tmp_path / "nope.png")
        assert popup.windowTitle() == "gone"
        popup.deleteLater()

    def test_shows_the_path_it_loaded(self, qapp, tmp_path):
        target = tmp_path / "quilt.png"
        popup = ImagePopup("q", target)
        labels = [label.text() for label in popup.findChildren(QLabel)]
        assert any(str(target) in text for text in labels)
        popup.deleteLater()


def _needs_cast_support() -> None:
    """Skip unless the installed quiltwright has ``save_and_cast_quilt``.

    Added in quiltwright 0.6.0.  The ``viz3d-qt`` extra floors it there, but a
    dev environment pinned to 0.5.x should skip this rather than fail: the
    import is deferred to call time precisely so the rest of the module still
    works against an older release.
    """
    pytest.importorskip("pyvista")
    quiltwright = pytest.importorskip("quiltwright")
    if not hasattr(quiltwright, "save_and_cast_quilt"):
        pytest.skip("quiltwright < 0.6.0 has no save_and_cast_quilt")


class TestCastSceneToLookingGlass:
    def test_a_failed_scene_build_is_returned_not_raised(self, qapp):
        _needs_cast_support()

        def explode(_plotter):
            raise RuntimeError("scene build failed")

        result = cast_scene_to_looking_glass(explode, None, "unused", _tiny_spec(), progress=None)
        assert result.path is None
        assert result.error is not None and "scene build failed" in result.error

    def test_a_failure_carries_a_status_line(self, qapp):
        """The caller shows ``message``; branching on the tuple is this module's job."""
        _needs_cast_support()

        def explode(_plotter):
            raise RuntimeError("scene build failed")

        result = cast_scene_to_looking_glass(explode, None, "unused", _tiny_spec())
        assert "Bridge" in result.message and "scene build failed" in result.message
        assert result.elapsed >= 0.0

    def test_a_builder_may_return_a_value(self, qapp):
        """Both consumers' scene builders return a plotter; wrapping them is noise."""
        _needs_cast_support()
        built: list[object] = []

        def build(plotter):
            built.append(plotter)
            raise RuntimeError("stop before rendering")

        cast_scene_to_looking_glass(build, None, "unused", _tiny_spec())
        assert built

    def test_spec_defaults_to_the_shared_preset(self, qapp):
        """Omitting the spec must not raise; it resolves the default preset."""
        _needs_cast_support()
        seen: list[tuple[int, int, str]] = []

        def explode(_plotter):
            raise RuntimeError("stop here")

        result = cast_scene_to_looking_glass(
            explode, None, "unused", progress=lambda *a: seen.append(a)
        )
        assert result.path is None
        assert seen and seen[0][0] == 1

    def test_progress_is_reported_before_each_stage(self, qapp):
        _needs_cast_support()
        seen: list[tuple[int, int, str]] = []

        def explode(_plotter):
            raise RuntimeError("stop here")

        cast_scene_to_looking_glass(
            explode, None, "unused", _tiny_spec(), progress=lambda *a: seen.append(a)
        )
        assert seen and seen[0][0] == 1 and seen[0][1] == 4


def _tiny_spec():
    """:return: A minimal 2x2 quilt spec."""
    from quiltwright import QuiltSpec

    return QuiltSpec(columns=2, rows=2, quilt_width=128, quilt_height=128, aspect=1.0)
