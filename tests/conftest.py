"""Shared test configuration.

Force Qt onto the offscreen platform before any test imports PyQt5, so the
viz3d-qt suite runs headless everywhere — locally it stops widget windows
flashing up and stealing focus mid-run, and it matches what CI does (the
Test job exports the same variable).  ``setdefault`` keeps it overridable:
``QT_QPA_PLATFORM=cocoa pytest tests/test_viz3d_qt.py`` shows real windows
when debugging a widget.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
