"""Registering the standard ``<tool> mcp`` command must not import ``mcp``.

A module's CLI registers the command at import time, so every one of its
commands pays for whatever that import pulls in. ``mcp`` costs about half a
second against pycode_kg's whole CLI startup of about 0.17 s (measured
2026-10-25), so ``kg_utils.mcp`` loads it only when a server is built.

Run in a subprocess: this test process may already have imported ``mcp``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import kg_utils

pytest.importorskip("click")

_SRC = str(Path(kg_utils.__file__).resolve().parents[1])


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": _SRC, "PATH": ""},
    )
    return result.stdout.strip()


def test_registering_the_command_does_not_import_mcp() -> None:
    out = _run(
        """
        import sys, click
        from kg_utils.mcp import CANONICAL_TOOLS, TRANSPORTS, add_mcp_command
        @click.group()
        def cli(): pass
        add_mcp_command(cli, lambda *a: None)
        print('mcp' in sys.modules)
        """
    )
    assert out == "False"


def test_server_names_still_load_on_first_use() -> None:
    pytest.importorskip("mcp")
    out = _run(
        """
        import sys
        from kg_utils.mcp import make_server, Context
        print('mcp' in sys.modules, callable(make_server))
        """
    )
    assert out == "True True"


def test_an_unknown_name_is_an_attribute_error() -> None:
    import kg_utils.mcp as package

    with pytest.raises(AttributeError, match="no_such_name"):
        package.no_such_name  # noqa: B018
