"""
The public API is a contract, and these tests are what make it one.

Written after 0.12.0 shipped without the ``viz3d`` organic engine it was
documented as introducing. The promotion lived on a branch the release tag did
not contain, so the tagged tree had neither the feature nor its tests — and a
test that is not in the tree you release cannot fail. Every check here is
therefore designed to fail on a tree that is *missing* something, not merely on
one that is wrong:

- :func:`test_all_names_resolve` walks each module's ``__all__`` and imports
  every name. A partially promoted module — ``__init__`` advertising symbols
  that no longer have a home — fails here rather than in a caller's traceback.
- :func:`test_every_source_module_is_packaged` builds a real wheel and compares
  it against the source tree, which is the only check that sees what users will
  actually install rather than what the checkout happens to contain.

For the release gate that runs against the built artifact in a clean
environment, see ``scripts/verify_release.py``.
"""

from __future__ import annotations

import importlib
import pkgutil
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src" / "kg_utils"

#: Modules whose import pulls a heavy or optional dependency. They are still
#: checked for *existence*; only the symbol walk skips them.
_OPTIONAL_IMPORT_MODULES = frozenset({"kg_utils.synthesis"})


def _public_modules() -> list[str]:
    """
    Every importable module under ``kg_utils``.

    :return: Dotted module names, parents before children.
    """
    names = ["kg_utils"]
    for info in pkgutil.walk_packages([str(_SRC)], prefix="kg_utils."):
        names.append(info.name)
    return names


def test_viz3d_organic_engine_is_importable() -> None:
    """
    The specific regression: 0.12.0 published a ``viz3d`` with no engine.

    Named explicitly rather than folded into the generic walk so that a failure
    says what broke instead of leaving someone to read a parametrised ID.
    """
    organic = importlib.import_module("kg_utils.viz3d.organic")

    for symbol in (
        "colonize",
        "crown_spacing",
        "grow_tree",
        "leaf_glyphs",
        "pipe_radii",
        "root_to_tip_paths",
        "seed_from_key",
        "smooth_paths",
        "tree_mesh",
    ):
        assert hasattr(organic, symbol), f"kg_utils.viz3d.organic is missing {symbol}"


def test_all_names_resolve() -> None:
    """Every name a module advertises in ``__all__`` must actually exist."""
    missing: list[str] = []

    for mod_name in _public_modules():
        if mod_name in _OPTIONAL_IMPORT_MODULES:
            continue
        try:
            module = importlib.import_module(mod_name)
        except ImportError as exc:  # optional extra absent in this environment
            pytest.skip(f"{mod_name} not importable here: {exc}")
        for name in getattr(module, "__all__", ()):
            if not hasattr(module, name):
                missing.append(f"{mod_name}.{name}")

    assert not missing, "__all__ advertises names that do not exist: " + ", ".join(missing)


def test_viz3d_reexports_the_whole_engine() -> None:
    """
    ``kg_utils.viz3d`` must re-export the engine, not just define the modules.

    Consumers import from the package root — ``gutenberg_kg.scene`` takes five
    of these names that way — so a module present on disk but absent from
    ``__init__`` is still a broken release from their point of view.
    """
    viz3d = importlib.import_module("kg_utils.viz3d")

    for symbol in ("Skeleton", "colonize", "crown_spacing", "grow_tree", "seed_from_key"):
        assert symbol in viz3d.__all__, f"kg_utils.viz3d.__all__ is missing {symbol}"
        assert hasattr(viz3d, symbol), f"kg_utils.viz3d does not export {symbol}"


@pytest.mark.integration
def test_every_source_module_is_packaged(tmp_path: Path) -> None:
    """
    Build a wheel and assert it carries every module in the source tree.

    This is the check that sees what users install. A module can be present in
    the checkout and still be missing from the artifact — excluded by packaging
    configuration, or simply absent from the tree a release was cut from — and
    a version string reveals neither.
    """
    poetry = shutil.which("poetry")
    if poetry is None:
        pytest.skip("poetry not on PATH")
    result = subprocess.run(
        [str(poetry), "build", "--format", "wheel", "--output", str(tmp_path)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # A build that runs and fails is a real failure, not a reason to skip.
    assert result.returncode == 0, f"poetry build failed: {result.stderr.strip()[:400]}"

    wheels = list(tmp_path.glob("*.whl"))
    assert wheels, "poetry build produced no wheel"
    packaged = {
        name
        for name in zipfile.ZipFile(wheels[0]).namelist()
        if name.startswith("kg_utils/") and name.endswith(".py")
    }

    expected = {
        f"kg_utils/{p.relative_to(_SRC).as_posix()}"
        for p in _SRC.rglob("*.py")
        if "__pycache__" not in p.parts
    }

    assert not (expected - packaged), (
        "source modules absent from the built wheel: " + ", ".join(sorted(expected - packaged))
    )
