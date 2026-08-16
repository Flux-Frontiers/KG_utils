"""Tests for the renderer-agnostic geometry accessors used by POV-Ray export.

``leaf_frames`` and ``limb_paths`` exist so a non-PyVista renderer can place
the same geometry ``leaf_glyphs`` and ``tree_mesh`` build.  These tests pin
the two properties that matter for that: they run without PyVista installed,
and they agree with the PyVista path they were factored out of.
"""

import numpy as np
import pytest

from kg_utils.viz3d.organic import (
    LEAF_ASPECT,
    Skeleton,
    colonize,
    grow_tree,
    leaf_facing,
    leaf_frames,
    limb_paths,
    oriented_cluster,
    pipe_radii,
    root_to_tip_paths,
    seed_from_key,
)

pv = pytest.importorskip("pyvista", reason="PyVista comparisons need the render extra")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def crown() -> np.ndarray:
    """A reproducible blob of attractors above the origin."""
    rng = np.random.default_rng(20260816)
    return rng.normal(0.0, 1.5, (120, 3)) + np.array([0.0, 6.0, 0.0])


@pytest.fixture
def skeleton(crown: np.ndarray) -> Skeleton:
    """A grown skeleton with radii assigned."""
    skel = grow_tree(crown, root=np.array([0.0, 0.0, 0.0]), key="test-tree")
    if skel.radii is None:
        pipe_radii(skel)
    return skel


# ---------------------------------------------------------------------------
# leaf_frames
# ---------------------------------------------------------------------------


def test_leaf_frames_returns_one_frame_per_leaf(crown, skeleton):
    points, directions = leaf_frames(crown, skeleton)
    assert points.shape == crown.shape
    assert directions.shape == crown.shape


def test_leaf_frames_directions_are_unit_length(crown, skeleton):
    _, directions = leaf_frames(crown, skeleton)
    assert np.linalg.norm(directions, axis=1) == pytest.approx(np.ones(crown.shape[0]))


def test_leaf_frames_is_deterministic_for_a_seed(crown, skeleton):
    a = leaf_frames(crown, skeleton, seed=7)
    b = leaf_frames(crown, skeleton, seed=7)
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])


def test_leaf_frames_seed_changes_the_roll_not_the_placement(crown, skeleton):
    a = leaf_frames(crown, skeleton, seed=1)
    b = leaf_frames(crown, skeleton, seed=2)
    assert a[0] == pytest.approx(b[0])  # cling is deterministic
    assert not np.allclose(a[1], b[1])  # jitter is not


def _anchors(crown: np.ndarray, skeleton: Skeleton) -> np.ndarray:
    """The skeleton node each leaf is measured against, chosen before any cling."""
    distances = np.linalg.norm(crown[:, None, :] - skeleton.points[None, :, :], axis=2)
    return skeleton.points[distances.argmin(axis=1)]


def test_leaf_frames_cling_zero_leaves_the_crown_untouched(crown, skeleton):
    """``cling=0`` skips the adjustment entirely — the documented "leave it" case."""
    points, _ = leaf_frames(crown, skeleton, cling=0.0)
    assert points == pytest.approx(crown)


def test_leaf_frames_cling_pulls_leaves_toward_their_anchor(crown, skeleton):
    """Reach shrinks as cling rises, measured against the pre-move anchor.

    It cannot be measured against whichever node is nearest *afterwards*:
    pulling a leaf onto its own twig can leave it nearer some other branch
    that happens to pass close by.
    """
    anchors = _anchors(crown, skeleton)
    reaches = [
        np.linalg.norm(leaf_frames(crown, skeleton, cling=c)[0] - anchors, axis=1)
        for c in (0.25, 0.5, 1.0)
    ]
    for looser, tighter in zip(reaches, reaches[1:]):
        assert np.all(tighter <= looser + 1e-9)
    assert reaches[-1].mean() < reaches[0].mean()


def test_leaf_frames_never_puts_leaves_inside_the_wood(crown, skeleton):
    """The pull stops at the branch radius plus the leaf's own size."""
    size = 0.35
    anchors = _anchors(crown, skeleton)
    distances = np.linalg.norm(crown[:, None, :] - skeleton.points[None, :, :], axis=2)
    clearance = skeleton.radii[distances.argmin(axis=1)] + size
    points, _ = leaf_frames(crown, skeleton, size=size, cling=1.0)
    assert np.all(np.linalg.norm(points - anchors, axis=1) >= clearance - 1e-6)


def test_leaf_frames_cling_pushes_buried_leaves_back_out(crown, skeleton):
    """A leaf that starts inside the clearance is moved out to it, not left there."""
    size = 0.35
    anchors = _anchors(crown, skeleton)
    distances = np.linalg.norm(crown[:, None, :] - skeleton.points[None, :, :], axis=2)
    clearance = skeleton.radii[distances.argmin(axis=1)] + size
    buried = distances.min(axis=1) < clearance
    assert buried.any(), "fixture no longer exercises the clamp"
    points, _ = leaf_frames(crown, skeleton, size=size, cling=0.5)
    reach = np.linalg.norm(points - anchors, axis=1)
    assert reach[buried] == pytest.approx(clearance[buried], abs=1e-9)


def test_leaf_frames_handles_an_empty_crown(skeleton):
    points, directions = leaf_frames(np.zeros((0, 3)), skeleton)
    assert points.shape == (0, 3)
    assert directions.shape == (0, 3)


def test_leaf_glyphs_places_glyphs_at_leaf_frame_points(crown, skeleton):
    """The refactor must not have moved the foliage."""
    from kg_utils.viz3d.organic import leaf_glyphs

    points, _ = leaf_frames(crown, skeleton, seed=3)
    glyphs = leaf_glyphs(crown, skeleton, seed=3)
    # Every glyph vertex lies within one prototype radius of a frame point.
    reach = 0.35 * max(LEAF_ASPECT) + 1e-6
    verts = np.asarray(glyphs.points)
    distances = np.linalg.norm(verts[:, None, :] - points[None, :, :], axis=2).min(axis=1)
    assert distances.max() <= reach


# ---------------------------------------------------------------------------
# limb_paths
# ---------------------------------------------------------------------------


def test_limb_paths_returns_one_path_per_tip(skeleton):
    paths = limb_paths(skeleton)
    usable = [p for p in root_to_tip_paths(skeleton) if len(p) >= 2]
    assert len(paths) == len(usable)


def test_limb_paths_radii_match_point_counts(skeleton):
    for points, radii in limb_paths(skeleton):
        assert points.shape[0] == radii.shape[0]
        assert points.shape[1] == 3


def test_limb_paths_start_at_the_root(skeleton):
    for points, _ in limb_paths(skeleton):
        assert points[0] == pytest.approx(skeleton.points[0], abs=1e-9)


def test_limb_paths_end_at_a_skeleton_node(skeleton):
    """The spline interpolates its control points, so tips are not pulled in."""
    for points, _ in limb_paths(skeleton):
        distances = np.linalg.norm(skeleton.points - points[-1], axis=1)
        assert distances.min() == pytest.approx(0.0, abs=1e-9)


def test_limb_paths_subdivision_increases_sample_count(skeleton):
    coarse = limb_paths(skeleton, subdivisions=2)
    fine = limb_paths(skeleton, subdivisions=8)
    assert sum(p.shape[0] for p, _ in fine) > sum(p.shape[0] for p, _ in coarse)


def test_limb_paths_radii_taper_outward(skeleton):
    """The pipe model makes limbs thinner toward the tip."""
    for _, radii in limb_paths(skeleton):
        assert radii[0] >= radii[-1]


def test_limb_paths_assigns_radii_when_missing(crown):
    bare = colonize(crown, root=np.array([0.0, 0.0, 0.0]), seed=seed_from_key("bare"))
    assert bare.radii is None
    assert limb_paths(bare)
    assert bare.radii is not None


def test_limb_paths_tracks_smooth_paths_closely(skeleton):
    """The NumPy spline is not VTK's, but it must not wander off the limb."""
    from kg_utils.viz3d.organic import smooth_paths

    scale = float(np.linalg.norm(skeleton.points.max(axis=0) - skeleton.points.min(axis=0)))
    for (mine, _), (theirs, _) in zip(limb_paths(skeleton), smooth_paths(skeleton), strict=True):
        assert mine.shape == theirs.shape
        deviation = np.linalg.norm(mine - theirs, axis=1).max()
        assert deviation < 0.02 * scale


def test_a_two_node_path_is_the_one_place_the_sample_counts_differ():
    """
    Catmull-Rom needs three control points to curve, so a two-node path comes
    back unchanged where ``smooth_paths`` resamples it along the same straight
    line.  Both describe that line and share both endpoints, so nothing renders
    differently — but the test above zips the two outputs, and a caller doing
    the same on a skeleton with a two-node limb would be surprised.  Pinned so
    the asymmetry is a documented property rather than a discovery.
    """
    from kg_utils.viz3d.organic import smooth_paths

    straight = Skeleton(
        points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        parents=np.array([-1, 0]),
    )
    (mine, _), (theirs, _) = limb_paths(straight)[0], smooth_paths(straight)[0]

    assert mine.shape[0] == 2
    assert theirs.shape[0] == 5  # subdivisions + 1
    assert np.allclose(mine[[0, -1]], theirs[[0, -1]])
    # Same segment: every one of the resampled points lies on it.
    direction = mine[-1] - mine[0]
    offsets = theirs - mine[0]
    cross = np.cross(np.broadcast_to(direction, offsets.shape), offsets)
    assert np.allclose(cross, 0.0)


# ---------------------------------------------------------------------------
# leaf_facing / oriented_cluster — promoted from gutenberg_kg and pycode_kg
# ---------------------------------------------------------------------------


def test_leaf_facing_is_unit_length():
    for outward in ([1.0, 0.0, 0.0], [0.3, -0.9, 0.2], [-2.0, 5.0, -1.0]):
        assert np.linalg.norm(leaf_facing(np.array(outward))) == pytest.approx(1.0)


def test_leaf_facing_tilts_upward_but_keeps_the_limb_direction():
    """Foliage runs out along the branch, then reaches for light."""
    facing = leaf_facing(np.array([1.0, 0.0, 0.0]))
    assert facing[0] > 0.0, "must keep running outward"
    assert facing[2] > 0.0, "must reach upward"


def test_leaf_facing_up_bias_controls_the_reach():
    flat = leaf_facing(np.array([1.0, 0.0, 0.0]), up_bias=0.0)
    steep = leaf_facing(np.array([1.0, 0.0, 0.0]), up_bias=4.0)
    assert flat[2] == pytest.approx(0.0)
    assert steep[2] > flat[2]


def test_leaf_facing_ignores_the_vertical_component_of_outward():
    """Only the horizontal run matters; the upward part comes from up_bias."""
    a = leaf_facing(np.array([1.0, 0.0, 0.0]))
    b = leaf_facing(np.array([1.0, 0.0, 9.0]))
    assert a == pytest.approx(b)


def test_leaf_facing_falls_back_to_up_for_a_vertical_limb():
    assert leaf_facing(np.array([0.0, 0.0, 5.0])) == pytest.approx([0.0, 0.0, 1.0])


def test_oriented_cluster_places_every_point_on_the_facing_side():
    """Far-side points are reflected, not discarded, so none end up behind."""
    centre = np.array([1.0, 2.0, 3.0])
    facing = np.array([0.0, 0.0, 1.0])
    points = np.asarray(oriented_cluster(40, centre, facing, 2.0))
    assert points.shape == (40, 3)
    assert np.all((points - centre) @ facing >= -1e-9)


def test_oriented_cluster_keeps_every_point_on_the_sphere():
    centre = np.array([0.0, 0.0, 0.0])
    facing = np.array([0.0, 1.0, 0.0])
    points = np.asarray(oriented_cluster(30, centre, facing, 1.5))
    assert np.linalg.norm(points - centre, axis=1) == pytest.approx(np.full(30, 1.5))


def test_oriented_cluster_returns_the_requested_count():
    """Reflection rather than rejection is why the count is exact."""
    for n in (1, 2, 7, 50):
        assert len(oriented_cluster(n, np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0)) == n


def test_oriented_cluster_handles_an_empty_cluster():
    """The guard gutenberg_kg's copy lacked: it raised a broadcast ValueError."""
    assert oriented_cluster(0, np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0) == []
    assert oriented_cluster(-3, np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0) == []


def test_oriented_cluster_follows_facing():
    """Rotating the facing vector carries the whole cluster with it."""
    up = np.asarray(oriented_cluster(30, np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0))
    side = np.asarray(oriented_cluster(30, np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.0))
    assert up[:, 2].mean() > 0.3
    assert side[:, 0].mean() > 0.3
