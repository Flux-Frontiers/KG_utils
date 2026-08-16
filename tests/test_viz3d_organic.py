"""
test_viz3d_organic.py

Tests for :mod:`kg_utils.viz3d.organic` — space-colonization tree skeletons.

Two design claims are under test. First, that the engine is domain-neutral: it
takes crown attractors and a root, and the hierarchy that produced those points
is the caller's business, so a document-shaped cloud and a diary-shaped one both
grow trees without the module knowing which is which.

Second, that the geometry is importable without PyVista. The ``viz3d`` extra
installs NumPy only, and thirteen repos depend on ``kgmodule-utils``; acquiring
VTK to import a layout would be a regression. Only the three mesh builders need
PyVista, and they must say so clearly rather than fail obscurely.

Author: Eric G. Suchanek, PhD
License: Elastic 2.0
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

import subprocess
import sys

import numpy as np

from kg_utils.viz3d import (
    Skeleton,
    colonize,
    crown_spacing,
    grow_tree,
    pipe_radii,
    root_to_tip_paths,
    seed_from_key,
)


def _document_cloud(n: int = 240) -> np.ndarray:
    """A compact crown, as a book's chunks would form."""
    rng = np.random.default_rng(11)
    pts = rng.normal(scale=1.5, size=(n, 3))
    pts[:, 2] += 6.0
    return pts


def _diary_cloud(n_years: int = 8, per_year: int = 40) -> np.ndarray:
    """
    A hollow, columnar crown, as a diary's dated entries form.

    Entries ring the trunk axis and stack by year, leaving the middle empty —
    the case the ``step`` docstring calls out, where too short an internode
    marches the leader up the hole as one unbranched chain.
    """
    rng = np.random.default_rng(7)
    out = []
    for year in range(n_years):
        theta = rng.uniform(0, 2 * np.pi, per_year)
        r = 2.0 + rng.normal(scale=0.2, size=per_year)
        z = 2.0 + year * 1.5 + rng.normal(scale=0.1, size=per_year)
        out.append(np.column_stack([r * np.cos(theta), r * np.sin(theta), z]))
    return np.vstack(out)


class TestSeeding:
    """A tree must be identical between sessions, renders, and printed figures."""

    def test_seed_is_stable_across_calls(self) -> None:
        assert seed_from_key("pepys") == seed_from_key("pepys")

    def test_seed_differs_by_key(self) -> None:
        assert seed_from_key("pepys") != seed_from_key("evelyn")

    def test_seed_in_uint32_range(self) -> None:
        assert 0 <= seed_from_key("anything") < 2**32

    def test_seed_survives_process_restart(self) -> None:
        """
        ``hash()`` is salted per process, which is why this function exists.

        Pinning the value to an independently computed blake2b digest is what
        makes "identical between sessions" testable inside one process.
        """
        import hashlib

        expected = int.from_bytes(hashlib.blake2b(b"pepys", digest_size=4).digest(), "big")
        assert seed_from_key("pepys") == expected


class TestGrowth:
    """The skeleton must reach the data and branch while doing it."""

    def test_grow_tree_produces_a_rooted_skeleton(self) -> None:
        pts = _document_cloud()
        sk = grow_tree(pts, np.zeros(3), key="book")
        assert isinstance(sk, Skeleton)
        assert sk.n_nodes > 1
        assert sk.parents[0] == -1, "index 0 must be the root"
        assert (sk.parents[1:] >= 0).all(), "every non-root node needs a parent"

    def test_growth_is_deterministic_for_a_key(self) -> None:
        pts = _document_cloud()
        a = grow_tree(pts, np.zeros(3), key="book")
        b = grow_tree(pts, np.zeros(3), key="book")
        assert np.array_equal(a.points, b.points)
        assert np.array_equal(a.parents, b.parents)

    def test_different_keys_grow_different_wood(self) -> None:
        pts = _document_cloud()
        a = grow_tree(pts, np.zeros(3), key="book")
        c = grow_tree(pts, np.zeros(3), key="other")
        assert not (a.points.shape == c.points.shape and np.array_equal(a.points, c.points)), (
            "jitter is what keeps two similar crowns from growing identical wood"
        )

    def test_it_actually_branches(self) -> None:
        """A lollipop — one chain to a blob — would defeat the purpose."""
        sk = grow_tree(_document_cloud(), np.zeros(3), key="book")
        assert len(sk.tips) > 3, "a tree with <4 tips has not branched"

    def test_hollow_columnar_crown_still_branches(self) -> None:
        """The diary case: entries ring the axis, leaving the middle empty."""
        sk = grow_tree(_diary_cloud(), np.zeros(3), key="pepys")
        assert len(sk.tips) > 3, "leader marched up the hole without branching"

    def test_grows_toward_the_crown(self) -> None:
        pts = _document_cloud()
        sk = grow_tree(pts, np.zeros(3), key="book")
        assert sk.points[:, 2].max() > pts[:, 2].mean() * 0.5

    def test_attractor_accounting_is_reported(self) -> None:
        """A cap must not pass silently."""
        pts = _document_cloud(n=100)
        sk = colonize(pts, np.zeros(3), max_attractors=20, seed=1)
        assert sk.attractors_total == 100
        assert sk.attractors_used <= 20


class TestPipeModel:
    """Limb thickness must follow what the limb carries — da Vinci's rule."""

    def test_grow_tree_assigns_radii(self) -> None:
        sk = grow_tree(_document_cloud(), np.zeros(3), key="book")
        assert sk.radii is not None
        assert sk.radii.shape[0] == sk.n_nodes

    def test_trunk_is_thicker_than_tips(self) -> None:
        sk = grow_tree(_document_cloud(), np.zeros(3), key="book")
        assert sk.radii is not None
        assert sk.radii[0] > sk.radii[sk.tips].max(), "trunk must carry the most"

    def test_radii_are_positive(self) -> None:
        sk = grow_tree(_document_cloud(), np.zeros(3), key="book")
        assert sk.radii is not None
        assert (sk.radii > 0).all()

    def test_pipe_radii_is_idempotent(self) -> None:
        sk = grow_tree(_document_cloud(), np.zeros(3), key="book")
        assert sk.radii is not None
        first = sk.radii.copy()
        pipe_radii(sk)
        assert np.allclose(first, sk.radii)


class TestPaths:
    """Root-to-tip paths are what the mesh sweeps along."""

    def test_one_path_per_tip(self) -> None:
        sk = grow_tree(_document_cloud(), np.zeros(3), key="book")
        assert len(root_to_tip_paths(sk)) == len(sk.tips)

    def test_every_path_starts_at_the_root(self) -> None:
        sk = grow_tree(_document_cloud(), np.zeros(3), key="book")
        assert all(p[0] == 0 for p in root_to_tip_paths(sk))


class TestCrownSpacing:
    """The natural length scale of a cloud."""

    def test_spacing_is_positive(self) -> None:
        assert crown_spacing(_document_cloud()) > 0

    def test_degenerate_cloud_does_not_divide_by_zero(self) -> None:
        assert crown_spacing(np.zeros((1, 3))) == 1.0

    def test_denser_cloud_has_smaller_spacing(self) -> None:
        rng = np.random.default_rng(3)
        sparse = rng.normal(scale=5.0, size=(200, 3))
        dense = rng.normal(scale=0.5, size=(200, 3))
        assert crown_spacing(dense) < crown_spacing(sparse)


class TestPyVistaIsOptional:
    """
    The ``viz3d`` extra installs NumPy only.

    Thirteen repos depend on ``kgmodule-utils``; importing a layout must not
    drag in VTK. These tests run in both worlds — where PyVista is absent the
    error must name the fix, and where it is present the builders must work.
    """

    def test_geometry_imports_without_pyvista(self) -> None:
        """This module imported at collection time, which is the assertion."""
        assert grow_tree is not None and seed_from_key is not None

    def test_mesh_builders_explain_themselves_when_pyvista_is_absent(self) -> None:
        """
        The message a caller without the render extra actually sees.

        This used to skip whenever pyvista *was* installed — which is to say it
        skipped on every machine that could run the rest of the suite, and on
        CI once the ``viz3d-render`` extra was added. A test of the absence
        path cannot be gated on absence; it has to manufacture it. A subprocess
        with the import blocked does that, and runs everywhere.
        """
        probe = """
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("pyvista", "vtk", "vtkmodules"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _Block())

import numpy as np
from kg_utils.viz3d import grow_tree, leaf_glyphs, tree_mesh

rng = np.random.default_rng(0)
cloud = rng.normal(0, 1, (80, 3)) * np.array([3.0, 3.0, 6.0]) + np.array([0, 0, 20.0])
sk = grow_tree(cloud, np.zeros(3), key="book")

for call in (lambda: tree_mesh(sk), lambda: leaf_glyphs(cloud, sk)):
    try:
        call()
    except ModuleNotFoundError as exc:
        assert "pip install pyvista" in str(exc), str(exc)
    else:
        raise AssertionError("mesh builder did not raise without pyvista")
print("OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout

    def test_mesh_builders_work_when_pyvista_is_present(self) -> None:
        pytest.importorskip("pyvista")

        from kg_utils.viz3d import leaf_glyphs, smooth_paths, tree_mesh

        sk = grow_tree(_document_cloud(), np.zeros(3), key="book")
        assert len(smooth_paths(sk)) == len(sk.tips)
        assert tree_mesh(sk).n_points > 0
        assert leaf_glyphs(_document_cloud(), sk).n_points > 0


# ---------------------------------------------------------------------------
# Ported from gutenberg_kg's tests/test_layout_organic.py, which covered this
# engine while it lived there.  Kept verbatim in substance so the promotion
# loses no coverage; only the slug -> key rename and the import path changed.
# ---------------------------------------------------------------------------


def _ellipsoid_crown(n: int, seed: int = 1) -> np.ndarray:
    """A filled ellipsoidal crown of *n* points, centred above the origin."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 3)) * np.array([4.0, 4.0, 5.0]) + np.array([0.0, 0.0, 30.0])


def _hollow_crown(n: int, seed: int = 2) -> np.ndarray:
    """An annular crown with an empty core — the shape a diary's entries make."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    radius = rng.uniform(30.0, 45.0, n)
    z = rng.uniform(13.0, 44.0, n)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])


class TestColonizeEdges:
    """Boundary and control-parameter behaviour of the growth loop."""

    def test_empty_attractors_yield_bare_root(self) -> None:
        sk = colonize(np.empty((0, 3)), np.zeros(3))
        assert sk.n_nodes == 1
        assert sk.parents[0] == -1

    def test_growth_branches_rather_than_chaining(self) -> None:
        sk = colonize(_ellipsoid_crown(400), np.zeros(3), seed=3)
        assert sk.n_nodes > 100
        # A chain has exactly one tip; a tree has many.
        assert len(sk.tips) > 10

    def test_large_hollow_crown_still_branches(self) -> None:
        # Regression: a step derived from inter-chunk spacing is far too fine to
        # span an empty core, so the leader marched up the hole as one chain.
        sk = colonize(_hollow_crown(3000), np.zeros(3), seed=4)
        assert len(sk.tips) > 5

    def test_parents_always_precede_their_children(self) -> None:
        sk = colonize(_ellipsoid_crown(200), np.zeros(3), seed=5)
        assert sk.parents[0] == -1
        assert (sk.parents[1:] >= 0).all()
        # colonize only ever appends children, which pipe_radii relies on.
        assert (sk.parents[1:] < np.arange(1, sk.n_nodes)).all()

    def test_growth_is_deterministic_for_a_seed(self) -> None:
        crown = _ellipsoid_crown(300)
        a = colonize(crown, np.zeros(3), seed=7)
        b = colonize(crown, np.zeros(3), seed=7)
        assert np.array_equal(a.points, b.points)

    def test_uncapped_growth_uses_every_attractor(self) -> None:
        sk = colonize(_ellipsoid_crown(200), np.zeros(3), max_attractors=None, seed=9)
        assert sk.attractors_used == sk.attractors_total == 200

    def test_cap_reports_both_counts(self) -> None:
        sk = colonize(_ellipsoid_crown(5000), np.zeros(3), max_attractors=1000, seed=8)
        assert sk.attractors_used == 1000
        assert sk.attractors_total == 5000

    def test_upward_tropism_lifts_the_crown(self) -> None:
        crown = _ellipsoid_crown(300)
        up = colonize(crown, np.zeros(3), tropism=(0, 0, 0.6), seed=11)
        down = colonize(crown, np.zeros(3), tropism=(0, 0, -0.3), seed=11)
        assert up.points[:, 2].max() > down.points[:, 2].max()


class TestPipeRadiiDetail:
    """The pipe model in detail — monotonicity, tips, and scale."""

    def test_tips_get_the_tip_radius(self) -> None:
        sk = colonize(_ellipsoid_crown(300), np.zeros(3), seed=12)
        radii = pipe_radii(sk, tip_radius=0.05)
        assert np.allclose(radii[sk.tips], 0.05)

    def test_trunk_is_the_thickest_limb(self) -> None:
        sk = colonize(_ellipsoid_crown(300), np.zeros(3), seed=13)
        radii = pipe_radii(sk)
        assert radii[0] == pytest.approx(radii.max())

    def test_a_parent_is_never_thinner_than_its_child(self) -> None:
        sk = colonize(_ellipsoid_crown(400), np.zeros(3), seed=14)
        radii = pipe_radii(sk)
        for child, parent in enumerate(sk.parents):
            if parent >= 0:
                assert radii[parent] >= radii[child] - 1e-9

    def test_a_bigger_corpus_grows_a_thicker_trunk(self) -> None:
        """This is what makes a prolific year read as heavier wood."""
        small = pipe_radii(colonize(_ellipsoid_crown(120), np.zeros(3), seed=15))
        large = pipe_radii(colonize(_ellipsoid_crown(2000), np.zeros(3), seed=15))
        assert large[0] > small[0]

    def test_radii_start_unset_and_are_stored(self) -> None:
        sk = colonize(_ellipsoid_crown(100), np.zeros(3), seed=16)
        assert sk.radii is None
        pipe_radii(sk)
        assert sk.radii is not None


class TestSmoothing:
    """Spline sweeps — the only geometry step that needs PyVista."""

    def test_smoothing_adds_points_and_keeps_radii_aligned(self) -> None:
        pytest.importorskip("pyvista")
        from kg_utils.viz3d import smooth_paths

        sk = grow_tree(_ellipsoid_crown(300), np.zeros(3), key="book")
        raw = root_to_tip_paths(sk)
        for (points, radii), path in zip(
            smooth_paths(sk), (p for p in raw if len(p) >= 2), strict=False
        ):
            assert points.shape[0] >= len(path)
            assert points.shape[0] == radii.shape[0]

    def test_smoothing_fills_radii_when_missing(self) -> None:
        pytest.importorskip("pyvista")
        from kg_utils.viz3d import smooth_paths

        sk = colonize(_ellipsoid_crown(200), np.zeros(3), seed=18)
        assert sk.radii is None
        assert smooth_paths(sk)
        assert sk.radii is not None
