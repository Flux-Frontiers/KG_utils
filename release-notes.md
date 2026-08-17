# Release Notes — v0.15.0

> Released: 2026-08-16

This release adds `kg_utils.viz3d.qt`, a shared Qt render lifecycle for
light-field output, behind a new `viz3d-qt` extra. It also completes the
release pipeline: pushing a tag now publishes the package to PyPI.

## What changed

**Qt render lifecycle.** A viewer that casts a knowledge graph to a Looking
Glass display needs a worker thread to keep POV-Ray off the GUI loop, a
progress bar fed by counting rendered views, a temporary directory, a preview
window, and the build → render → write → cast sequence. `pycode_kg` and
`gutenberg_kg` each carried a near-identical copy of that machinery.
`kg_utils.viz3d.qt` replaces both copies with one module: `PovRenderWorker`,
`PovRenderSession`, `ImagePopup`, and `cast_scene_to_looking_glass`. Domain
decisions — which node becomes a trunk, what the window looks like — stay in
each repo; the session takes its progress bar and status callback as
constructor arguments.

The shared version also fixes a crash both copies had: closing the window
during a render left a running `QThread` for Qt to destroy, which aborts the
process. `PovRenderSession.shutdown()` disconnects the worker's signals before
waiting, and parks a worker that does not stop in time instead of dropping it.

The module is not re-exported from `kg_utils.viz3d`, because its classes
subclass `QThread`, `QDialog`, and `QObject` — importing them requires PyQt5,
and importing a layout must not. Install the extra to use it:
`pip install 'kgmodule-utils[viz3d-qt]'`.

**Automated PyPI publishing.** The Release workflow now uploads the wheel and
sdist to PyPI through trusted publishing (OIDC) after it creates the GitHub
Release, using the same built files for both. No API token lives in the
repository secrets, and releases no longer end with a manual upload.

**Headless Qt tests.** The test suite forces Qt onto the offscreen platform,
in CI and locally, so the Qt lifecycle tests run without a display and without
flashing windows during a local run. Set `QT_QPA_PLATFORM` yourself to
override this while debugging a widget.

## Upgrading

No action required. Existing APIs are unchanged; the `viz3d-qt` extra is new
and opt-in. Downstream viewers that carry their own copy of the render
machinery (`pycode_kg`, `gutenberg_kg`) can delete it and depend on
`kg_utils.viz3d.qt` instead.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
