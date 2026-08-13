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
        try:
            import pyvista  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            pytest.skip("pyvista installed; absence path not exercisable here")

        from kg_utils.viz3d import leaf_glyphs, tree_mesh

        sk = grow_tree(_document_cloud(), np.zeros(3), key="book")
        for call in (lambda: tree_mesh(sk), lambda: leaf_glyphs(_document_cloud(), sk)):
            with pytest.raises(ModuleNotFoundError, match="pip install pyvista"):
                call()

    def test_mesh_builders_work_when_pyvista_is_present(self) -> None:
        pytest.importorskip("pyvista")

        from kg_utils.viz3d import leaf_glyphs, smooth_paths, tree_mesh

        sk = grow_tree(_document_cloud(), np.zeros(3), key="book")
        assert len(smooth_paths(sk)) == len(sk.tips)
        assert tree_mesh(sk).n_points > 0
        assert leaf_glyphs(_document_cloud(), sk).n_points > 0
