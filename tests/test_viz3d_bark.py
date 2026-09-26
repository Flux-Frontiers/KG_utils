"""
Tests for the textured bark sweep (bark_sweep / bark_mesh).

``tree_mesh`` sweeps with ``tube()``, which writes no texture coordinates, so
no bark image could ever be applied.  ``bark_sweep`` is the sweep that can: its
contract is watertight-enough, outward-facing wood whose ``u`` wraps in whole
tiles and whose ``v`` keeps the image's aspect.
"""

from __future__ import annotations

import numpy as np
import pytest

from kg_utils.viz3d import (
    SPECIES,
    BarkSweep,
    Skeleton,
    bark_sweep,
    crown_sections,
    grow_tree,
    pipe_radii,
    section_cluster,
)
from kg_utils.viz3d.organic import _chains


@pytest.fixture(scope="module")
def skeleton() -> Skeleton:
    habit = SPECIES["oak"]
    tips = crown_sections(12, 20.0, 6.0, habit)
    crown = np.vstack(
        [section_cluster(15, t, np.array([0.0, 0.0, t[2]]), 1.5, habit) for t in tips]
    )
    return grow_tree(crown, np.zeros(3), key="bark", habit=habit)


def test_triangles_face_outward(skeleton: Skeleton) -> None:
    sweep = bark_sweep(skeleton)
    a, b, c = (sweep.points[sweep.faces[:, i]] for i in range(3))
    face_n = np.cross(b - a, c - a)
    vert_n = sweep.normals[sweep.faces].sum(axis=1)
    area = np.linalg.norm(face_n, axis=1)
    keep = area > 1e-12
    agree = (np.einsum("ij,ij->i", face_n[keep], vert_n[keep]) > 0).mean()
    # Counter-clockwise from outside, or the bark renders inside-out.
    assert agree > 0.99


def test_indices_and_normals_are_sound(skeleton: Skeleton) -> None:
    sweep = bark_sweep(skeleton)
    assert isinstance(sweep, BarkSweep)
    assert sweep.faces.min() >= 0 and sweep.faces.max() < sweep.n_points
    assert sweep.uv.shape == (sweep.n_points, 2) and np.isfinite(sweep.uv).all()
    np.testing.assert_allclose(np.linalg.norm(sweep.normals, axis=1), 1.0, atol=1e-9)


def test_u_wraps_in_whole_tiles(skeleton: Skeleton) -> None:
    sweep = bark_sweep(skeleton, subdivisions=1)
    # Rows of one ring share v; the seam vertex closes each ring at a whole tile count.
    rings = np.split(sweep.uv, np.flatnonzero(np.diff(sweep.uv[:, 0]) < 0) + 1)
    for ring in rings:
        assert ring[0, 0] == 0.0
        assert ring[-1, 0] == pytest.approx(round(ring[-1, 0]))
        assert ring[-1, 0] >= 1


def test_v_keeps_the_image_aspect(skeleton: Skeleton) -> None:
    square = bark_sweep(skeleton, aspect=1.0)
    tall = bark_sweep(skeleton, aspect=2.0)
    # A taller image covers twice the length per tile, so v advances half as fast.
    np.testing.assert_allclose(tall.uv[:, 1], square.uv[:, 1] / 2.0)
    np.testing.assert_array_equal(tall.uv[:, 0], square.uv[:, 0])


def test_every_edge_is_swept_once(skeleton: Skeleton) -> None:
    assert skeleton.radii is not None
    edges = []
    for chain in _chains(skeleton, skeleton.radii):
        edges.extend(zip(chain[:-1], chain[1:]))
    expected = {(int(p), i) for i, p in enumerate(skeleton.parents) if p >= 0}
    assert len(edges) == len(set(edges))
    assert set(edges) == expected


def test_min_radius_thickens_only_the_drawn_tube(skeleton: Skeleton) -> None:
    thin = bark_sweep(skeleton, subdivisions=1)
    thick = bark_sweep(skeleton, subdivisions=1, min_radius=0.2)
    assert thick.points.shape == thin.points.shape
    np.testing.assert_array_equal(thick.uv, thin.uv)
    assert skeleton.radii is not None and skeleton.radii.min() < 0.2
    # Every drawn ring is at least min_radius across; each ring starts at u = 0
    # and ends on its duplicated seam vertex.
    starts = np.flatnonzero(thick.uv[:, 0] == 0.0)
    for lo, hi in zip(starts, [*starts[1:], thick.n_points]):
        ring = thick.points[lo : hi - 1]
        radius = np.linalg.norm(ring - ring.mean(axis=0), axis=1)
        assert radius.min() >= 0.2 - 1e-6


def test_a_bare_root_sweeps_nothing() -> None:
    sk = Skeleton(points=np.zeros((1, 3)), parents=np.array([-1]))
    pipe_radii(sk)
    sweep = bark_sweep(sk)
    assert sweep.n_points == 0 and sweep.faces.shape == (0, 3)


def test_bark_mesh_carries_texture_coordinates(skeleton: Skeleton) -> None:
    pv = pytest.importorskip("pyvista")
    from kg_utils.viz3d import bark_mesh

    mesh = bark_mesh(skeleton, aspect=2.0)
    assert isinstance(mesh, pv.PolyData)
    sweep = bark_sweep(skeleton, aspect=2.0)
    assert mesh.n_points == sweep.n_points and mesh.n_cells == len(sweep.faces)
    np.testing.assert_allclose(mesh.active_texture_coordinates, sweep.uv)
    assert "Normals" in mesh.point_data
