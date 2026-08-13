#!/usr/bin/env python3
"""
Pre-publish gate: prove the artifact contains what the release claims.

Run this before ``poetry publish``. It exists because 0.12.0 went to PyPI
documented as introducing the ``viz3d`` organic engine and containing no such
thing: the promotion sat on a branch the ``v0.12.0`` tag did not include, so the
tagged tree had neither the engine nor the tests that would have missed it. The
version string said 0.12.0, the changelog described the engine, and the wheel
had two modules in ``viz3d`` where it should have had three.

Nothing in a source checkout can catch that, because the checkout is not what
users install. So this script throws the checkout away: it builds a wheel,
installs it into a throwaway virtual environment with no source directory on the
path, and imports the public API from *there*.

It checks three things:

1. The wheel's version matches ``pyproject.toml``.
2. The newest released heading in ``CHANGELOG.md`` matches that version, so a
   release cannot ship while the changelog still describes it as unreleased or
   describes some other version entirely.
3. Every name in :data:`REQUIRED_API` imports from the installed wheel.

Exit status is 0 when the artifact is publishable and 1 otherwise, so CI can
gate a tag on it.

Usage::

    python scripts/verify_release.py            # build, then verify
    python scripts/verify_release.py --wheel dist/kgmodule_utils-0.12.1-py3-none-any.whl
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The surface downstream repos import. ``gutenberg_kg.scene`` alone takes five
#: of these from ``kg_utils.viz3d``; every one was missing from 0.12.0.
REQUIRED_API: dict[str, tuple[str, ...]] = {
    "kg_utils.viz3d": (
        "AlliumLayout",
        "FunnelLayout",
        "Layout3D",
        "LayoutEdge",
        "LayoutNode",
        "Skeleton",
        "colonize",
        "crown_spacing",
        "fibonacci_annulus",
        "fibonacci_sphere",
        "golden_spiral_2d",
        "grow_tree",
        "leaf_glyphs",
        "pipe_radii",
        "root_to_tip_paths",
        "seed_from_key",
        "smooth_paths",
        "tree_mesh",
    ),
    "kg_utils.viz3d.organic": ("colonize", "grow_tree", "seed_from_key"),
}


def _fail(message: str) -> None:
    """Print a failure and exit non-zero."""
    print(f"FAIL  {message}")
    sys.exit(1)


def declared_version() -> str:
    """
    Read the version from ``pyproject.toml``.

    :return: Version string.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        _fail("no version in pyproject.toml")
    return match.group(1)  # type: ignore[union-attr]


def newest_changelog_version() -> str | None:
    """
    Read the newest *released* version heading from ``CHANGELOG.md``.

    ``[Unreleased]`` is skipped, since it by definition describes nothing that
    has shipped.

    :return: Version string, or ``None`` if no released heading exists.
    """
    for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^##\s*\[([^\]]+)\]", line)
        if match and match.group(1).lower() != "unreleased":
            return match.group(1)
    return None


def build_wheel() -> Path:
    """
    Build a wheel into ``dist/``.

    :return: Path to the built wheel.
    """
    poetry = shutil.which("poetry")
    if poetry is None:
        _fail("poetry not found on PATH; cannot build a wheel to verify")
    subprocess.run([str(poetry), "build", "--format", "wheel"], cwd=ROOT, check=True)
    wheels = sorted((ROOT / "dist").glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        _fail("poetry build produced no wheel")
    return wheels[-1]


def verify_wheel(wheel: Path) -> None:
    """
    Install *wheel* into a throwaway environment and import the public API.

    The probe runs with ``cwd`` set outside the repository so the source tree
    cannot satisfy an import that the wheel does not.

    :param wheel: Wheel to install.
    """
    probe = "\n".join(
        [
            "import importlib, sys",
            f"required = {REQUIRED_API!r}",
            "missing = []",
            "for mod_name, symbols in required.items():",
            "    try:",
            "        module = importlib.import_module(mod_name)",
            "    except Exception as exc:",
            "        missing.append(f'{mod_name}: {exc}')",
            "        continue",
            "    for symbol in symbols:",
            "        if not hasattr(module, symbol):",
            "            missing.append(f'{mod_name}.{symbol}')",
            "print('MISSING:' + ','.join(missing))",
        ]
    )

    with tempfile.TemporaryDirectory() as tmp:
        env_dir = Path(tmp) / "venv"
        venv.create(env_dir, with_pip=True)
        python = env_dir / "bin" / "python"

        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", f"{wheel}[viz3d]"],
            check=True,
        )
        result = subprocess.run(
            [str(python), "-c", probe],
            cwd=tmp,  # never import from the source tree
            capture_output=True,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        _fail(f"probe crashed: {result.stderr.strip()[:400]}")

    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("MISSING:")), None)
    if line is None:
        _fail(f"probe produced no verdict: {result.stdout.strip()[:200]}")
    missing = [item for item in line[len("MISSING:") :].split(",") if item]
    if missing:
        _fail("installed wheel is missing: " + ", ".join(missing))


def main() -> None:
    """Run every gate and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="Verify this wheel instead of building one.")
    args = parser.parse_args()

    version = declared_version()
    changelog = newest_changelog_version()
    if changelog != version:
        _fail(f"CHANGELOG newest release is {changelog!r}, pyproject says {version!r}")
    print(f"ok    changelog and pyproject agree on {version}")

    wheel = args.wheel or build_wheel()
    if version not in wheel.name:
        _fail(f"wheel {wheel.name} does not carry version {version}")
    print(f"ok    wheel is {wheel.name}")

    verify_wheel(wheel)
    print(f"ok    installed wheel exports the full public API")
    print(f"\n{version} is publishable.")


if __name__ == "__main__":
    main()
