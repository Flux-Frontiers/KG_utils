"""
organic.py — botanically credible tree skeletons for KG graphs.

Where :class:`~kg_utils.viz3d.layout.FunnelLayout` and its siblings place nodes
on a golden-angle spiral — right for an interactive explorer at full scale —
this module *grows* a skeleton that reaches the data.  Leaf-level node positions
become attraction points and the branch structure is produced by space
colonization (Runions, Lane & Prusinkiewicz 2007), so every limb is a real
structural path through the graph's hierarchy and the canopy's shape is the
graph's shape.

The hierarchy itself is the caller's business.  A document corpus grows
document → section → chunk; a diary grows trunk → period limb → entry cluster →
chunk leaves, one limb per year.  This module only needs the crown attractors
and a root.

Pipeline::

    colonize()      attractors + root  → node/parent skeleton
    pipe_radii()    skeleton           → per-node radius (da Vinci's rule)
    smooth_paths()  skeleton           → Catmull-Rom root-to-tip polylines
    tree_mesh()     skeleton           → one swept-tube PolyData per tree

Everything is deterministic given a seed; :func:`seed_from_key` derives one from
any stable string key so a tree is identical between sessions, renders, and any
printed figure.

Geometry here is pure NumPy.  Only the three mesh builders — :func:`smooth_paths`,
:func:`tree_mesh` and :func:`leaf_glyphs` — need PyVista, and they import it
lazily so that ``kgmodule-utils[viz3d]`` stays a NumPy-only dependency for
callers that just want skeletons and positions.

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from kg_utils.viz3d.layout import fibonacci_sphere

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyvista as pv

__author__ = "Eric G. Suchanek, PhD"

#: Pipe-model exponent (da Vinci's rule).  n = 2 is the classical statement
#: (cross-sections sum exactly); 2–2.5 reads better on real trees.
PIPE_EXPONENT: float = 2.2

#: Default ceiling on attractors used for *growth*.  Past a few thousand the
#: silhouette stops changing and the O(tips x attractors) inner loop dominates.
MAX_ATTRACTORS: int = 3000

#: Per-axis scale flattening the leaf prototype from a ball into a blade.
#: Shared so a non-PyVista renderer can build the same shape from
#: :func:`leaf_frames` rather than re-deriving it.
LEAF_ASPECT: tuple[float, float, float] = (1.0, 0.55, 0.2)


def _pyvista():
    """
    Import PyVista on demand, with an actionable message when it is absent.

    The geometry in this module is pure NumPy; only the mesh builders need
    PyVista.  Importing it lazily keeps ``kgmodule-utils[viz3d]`` installable
    without VTK for callers that only want skeletons and positions.

    :return: The ``pyvista`` module.
    :raises ModuleNotFoundError: If PyVista is not installed.
    """
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise ModuleNotFoundError(
            "Building tree meshes needs PyVista, which kgmodule-utils[viz3d] "
            "does not install (the layout engine is NumPy-only). "
            "Install it with: pip install pyvista"
        ) from exc
    return pv


def seed_from_key(key: str) -> int:
    """Stable 32-bit seed for any identifying string.

    Python's builtin ``hash()`` is salted per process, so it cannot be used
    for geometry that must reproduce between runs.

    :param key: Identifying string — a book slug, diary name, or repo path.
    :return: Seed in ``[0, 2**32)``.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def crown_spacing(
    attractors: np.ndarray,
    *,
    sample: int = 512,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Typical nearest-neighbour distance within an attractor cloud.

    This is the natural length scale for growth: internodes shorter than the
    spacing between chunks produce twigs, longer ones produce spokes.  The
    estimate is taken from a random subsample so a 5,000-chunk book does not
    pay for a full ``M × M`` distance matrix.

    :param attractors: ``(M, 3)`` points.
    :param sample: Maximum number of probe points.
    :param rng: Generator for the subsample; a fixed default keeps callers
        that pass no RNG reproducible.
    :return: Median nearest-neighbour distance (never zero).
    """
    pts = np.atleast_2d(np.asarray(attractors, dtype=float))
    if pts.shape[0] < 2:
        return 1.0
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = (
        np.arange(pts.shape[0])
        if pts.shape[0] <= sample
        else rng.choice(pts.shape[0], size=sample, replace=False)
    )
    dist = np.linalg.norm(pts[idx, None, :] - pts[None, :, :], axis=2)
    dist[np.arange(idx.size), idx] = np.inf  # ignore self-distance
    nn = dist.min(axis=1)
    nn = nn[np.isfinite(nn)]
    return float(np.median(nn)) if nn.size else 1.0


@dataclass
class Skeleton:
    """A grown tree skeleton: points plus a parent index per point.

    :param points: ``(N, 3)`` node positions; index 0 is the root.
    :param parents: ``(N,)`` parent index per node, ``-1`` for the root.
    :param radii: ``(N,)`` branch radius per node, filled by :func:`pipe_radii`.
    """

    points: np.ndarray
    parents: np.ndarray
    radii: np.ndarray | None = None
    #: Indices of childless nodes — the leaf-bearing tips.
    tips: list[int] = field(default_factory=list)
    #: Attractors actually grown toward, and how many the crown held.  These
    #: differ when ``colonize(max_attractors=...)`` subsamples a dense crown;
    #: report both rather than let a cap pass silently.
    attractors_used: int = 0
    attractors_total: int = 0

    @property
    def n_nodes(self) -> int:
        """Number of skeleton nodes."""
        return int(self.points.shape[0])

    def children(self) -> dict[int, list[int]]:
        """``{parent index: [child indices]}`` for the whole skeleton."""
        kids: dict[int, list[int]] = {}
        for i, p in enumerate(self.parents):
            if p >= 0:
                kids.setdefault(int(p), []).append(i)
        return kids


def colonize(
    attractors: np.ndarray,
    root: np.ndarray,
    *,
    step: float | None = None,
    influence: float | None = None,
    kill: float | None = None,
    tropism: tuple[float, float, float] = (0.0, 0.0, 0.18),
    jitter: float = 0.12,
    max_attractors: int | None = MAX_ATTRACTORS,
    max_iter: int = 800,
    seed: int = 0,
) -> Skeleton:
    """
    Grow a branching skeleton from *root* toward *attractors*.

    Space colonization: each surviving attractor pulls the single nearest
    skeleton node that lies within *influence*; every node with at least one
    attractor takes one *step* along the averaged pull direction (plus
    *tropism*), and attractors closer than *kill* to any node are consumed.
    Growth stops when the attractors run out or nothing moves.

    Distances default to the attractor cloud's own scale, so the same call
    works for a 40-chunk pamphlet and a 5,000-chunk epic.

    :param attractors: ``(M, 3)`` crown points to reach — the chunk positions.
    :param root: ``(3,)`` trunk base.
    :param step: Internode length.  Default: a fixed fraction of the crown's
        diagonal, floored by half the inter-attractor spacing.  Internode
        length has to scale with the *tree*, not with chunk density: a
        hollow crown (a diary's entries ring the trunk axis, leaving the
        middle empty) needs steps long enough for the influence radius to
        span the gap, or the leader marches up the hole as one unbranched
        chain.
    :param influence: Attraction radius.  Default: ``12 * step``.
    :param kill: Consumption radius.  Default: ``2 * step`` — larger and one
        node swallows a whole cluster, collapsing the branching.
    :param tropism: Constant bias added to every growth direction — upward
        for conifer-like genres, negative Z for weeping ones.
    :param jitter: Random direction noise as a fraction of the step, which is
        what keeps two books with similar crowns from growing identical wood.
    :param max_attractors: Grow toward at most this many attractors, sampled
        deterministically; ``None`` uses every one.  The per-iteration cost is
        ``O(tips x attractors)``, and a 19,000-chunk diary is well past the
        point where extra attractors change the silhouette.  Leaves still
        render at *every* chunk position — only the skeleton is subsampled.
    :param max_iter: Hard iteration cap.
    :param seed: RNG seed; see :func:`seed_from_key`.
    :return: The grown :class:`Skeleton` (radii not yet assigned).
    """
    all_pts = np.atleast_2d(np.asarray(attractors, dtype=float))
    root = np.asarray(root, dtype=float).reshape(3)
    if all_pts.size == 0:
        return Skeleton(points=root[None, :], parents=np.array([-1]))

    rng = np.random.default_rng(seed)
    if max_attractors is not None and all_pts.shape[0] > max_attractors:
        pts = all_pts[rng.choice(all_pts.shape[0], size=max_attractors, replace=False)]
    else:
        pts = all_pts

    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    step = step if step is not None else max(extent / 40.0, 0.5 * crown_spacing(pts, rng=rng))
    influence = influence if influence is not None else 12.0 * step
    kill = kill if kill is not None else 2.0 * step

    tropism_v = np.asarray(tropism, dtype=float)

    nodes: list[np.ndarray] = [root]
    parents: list[int] = [-1]
    alive = np.ones(len(pts), dtype=bool)

    def bridge(from_index: int, target: np.ndarray, stop_at: float) -> None:
        """March a chain of internodes from a node until *target* is within *stop_at*."""
        current = from_index
        for _ in range(max_iter):
            gap = target - nodes[current]
            distance = float(np.linalg.norm(gap))
            if distance <= stop_at:
                return
            nodes.append(nodes[current] + gap / distance * min(step, distance))
            parents.append(current)
            current = len(nodes) - 1

    # The root starts outside every influence sphere, so nothing would ever
    # grow: lead a trunk up to the nearest attractor first.
    bridge(0, pts[int(np.argmin(np.linalg.norm(pts - root, axis=1)))], influence)

    for _ in range(max_iter):
        live_idx = np.flatnonzero(alive)
        if live_idx.size == 0:
            break

        node_arr = np.asarray(nodes)
        # (live attractors, nodes) distance matrix — the O(tips × attractors)
        # core.  See the plan's open question 1 if a big book gets slow.
        dist = np.linalg.norm(pts[live_idx, None, :] - node_arr[None, :, :], axis=2)
        nearest = dist.argmin(axis=1)
        nearest_d = dist[np.arange(live_idx.size), nearest]

        in_range = nearest_d <= influence
        if not in_range.any():
            # Attractors remain but none is reachable: the tree has consumed a
            # nearby cluster and the rest of the crown lies across a gap wider
            # than the influence radius.  Textbook colonization simply stops
            # here, which strands the whole rest of the canopy — a tree with a
            # stump and a cloud of unattached leaves.  Grow a limb across the
            # gap and carry on.
            closest = int(np.argmin(nearest_d))
            bridge(int(nearest[closest]), pts[live_idx[closest]], influence)
            continue

        pulls: dict[int, np.ndarray] = {}
        for a_i, n_i in zip(live_idx[in_range], nearest[in_range]):
            v = pts[a_i] - node_arr[n_i]
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                continue
            pulls[int(n_i)] = pulls.get(int(n_i), np.zeros(3)) + v / n

        if not pulls:
            break

        for n_i, pull in pulls.items():
            direction = pull / max(float(np.linalg.norm(pull)), 1e-9) + tropism_v
            if jitter:
                direction = direction + rng.normal(0.0, jitter, 3)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                continue
            nodes.append(node_arr[n_i] + direction / norm * step)
            parents.append(n_i)

        # Consume attractors reached by the nodes just added
        new_arr = np.asarray(nodes[len(node_arr) :])
        if new_arr.size:
            reached = np.linalg.norm(pts[live_idx, None, :] - new_arr[None, :, :], axis=2)
            alive[live_idx[reached.min(axis=1) <= kill]] = False

    # Every chunk must hang on wood.  An isolated attractor can stay in range
    # for the whole run and still never win a node, because it is never the
    # sole claimant of one — the averaged pull always goes to the crowd.  A
    # book's five one-chunk front-matter sections are exactly that case, and
    # they show up as leaves floating with no branch under them.  Give each
    # survivor its own twig.
    for a_i in np.flatnonzero(alive):
        node_arr = np.asarray(nodes)
        gaps = np.linalg.norm(node_arr - pts[a_i], axis=1)
        if gaps.min() <= kill:
            continue
        bridge(int(np.argmin(gaps)), pts[a_i], kill)

    parent_arr = np.asarray(parents, dtype=int)
    has_child = np.zeros(len(nodes), dtype=bool)
    has_child[parent_arr[parent_arr >= 0]] = True
    return Skeleton(
        points=np.asarray(nodes, dtype=float),
        parents=parent_arr,
        tips=np.flatnonzero(~has_child).tolist(),
        attractors_used=int(pts.shape[0]),
        attractors_total=int(all_pts.shape[0]),
    )


def pipe_radii(
    skeleton: Skeleton,
    *,
    tip_radius: float = 0.05,
    exponent: float = PIPE_EXPONENT,
) -> np.ndarray:
    """
    Assign a radius to every skeleton node by the pipe model.

    Tips get *tip_radius*; every junction's radius is
    ``(Σ rᵢ**exponent) ** (1/exponent)`` over its children, so a limb's
    thickness states exactly how much of the book it carries — including the
    trunk, whose final radius is a function of total chunk count.

    :param skeleton: Skeleton from :func:`colonize`; mutated in place
        (``skeleton.radii`` is set) and also returned.
    :param tip_radius: Radius of a leaf-bearing tip.
    :param exponent: Pipe-model exponent; 2 is exact, 2–2.5 looks like wood.
    :return: ``(N,)`` radius per node.
    """
    kids = skeleton.children()
    radii = np.full(skeleton.n_nodes, tip_radius, dtype=float)

    # Postorder by construction: a node's parent index is always lower than
    # its own (colonize only ever appends children), so one reverse sweep
    # accumulates children before their parent is read.
    accum = np.zeros(skeleton.n_nodes, dtype=float)
    for i in range(skeleton.n_nodes - 1, -1, -1):
        if kids.get(i):
            radii[i] = accum[i] ** (1.0 / exponent)
        p = int(skeleton.parents[i])
        if p >= 0:
            accum[p] += radii[i] ** exponent

    skeleton.radii = radii
    return radii


def root_to_tip_paths(skeleton: Skeleton) -> list[list[int]]:
    """
    Decompose the skeleton into paths, one per tip, for sweeping.

    Each path runs from the root to a tip; every edge appears in at least one
    path so the swept tubes cover the whole skeleton.

    :param skeleton: Skeleton from :func:`colonize`.
    :return: List of index paths, root-first.
    """
    kids = skeleton.children()
    leaves = [i for i in range(skeleton.n_nodes) if not kids.get(i)]
    paths: list[list[int]] = []
    for leaf in leaves:
        path = [leaf]
        node = leaf
        while True:
            parent = int(skeleton.parents[node])
            if parent < 0:
                break
            path.append(parent)
            node = parent
        paths.append(path[::-1])
    return paths


def smooth_paths(
    skeleton: Skeleton,
    *,
    subdivisions: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Turn skeleton paths into smooth polylines with per-point radii.

    The colonization output is piecewise-linear at the step length, which
    reads as faceted wire.  Splining each root-to-tip path (Catmull-Rom via
    ``pv.Spline``) and resampling the radii along it gives limbs that curve.

    :param skeleton: Skeleton with radii assigned by :func:`pipe_radii`.
    :param subdivisions: Spline samples per skeleton segment.
    :return: ``[(points (K, 3), radii (K,)), ...]`` per path.
    """
    pv = _pyvista()
    if skeleton.radii is None:
        pipe_radii(skeleton)
    radii = skeleton.radii
    assert radii is not None  # set above; keeps the type checker honest

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for path in root_to_tip_paths(skeleton):
        if len(path) < 2:
            continue
        raw = skeleton.points[path]
        n_out = max(len(path), (len(path) - 1) * subdivisions + 1)
        spline = pv.Spline(raw, n_out)
        smoothed = np.asarray(spline.points, dtype=float)
        # Resample radii on the same normalised arc parameter as the spline.
        src_t = np.linspace(0.0, 1.0, len(path))
        dst_t = np.linspace(0.0, 1.0, smoothed.shape[0])
        out.append((smoothed, np.interp(dst_t, src_t, radii[path])))
    return out


def limb_paths(
    skeleton: Skeleton,
    *,
    subdivisions: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Smooth root-to-tip limb paths with per-point radii, without PyVista.

    The NumPy counterpart of :func:`smooth_paths`, for renderers that describe
    a limb analytically — a POV-Ray ``sphere_sweep``, say — and so never need
    a tessellated tube or the VTK stack that builds one.

    The smoothing is a uniform Catmull-Rom through the same control points,
    which interpolates them exactly as ``pv.Spline`` does but is **not
    bit-identical** to VTK's parametric spline.  When two backends must agree
    to the pixel, run :func:`smooth_paths` once and give both the same points
    rather than letting each smooth its own.

    The sample *count* matches :func:`smooth_paths` for every path of three or
    more nodes.  It does not for a two-node path: Catmull-Rom needs three
    control points to curve, so this returns the two points unchanged where
    ``smooth_paths`` returns ``subdivisions + 1`` of them along the same
    straight segment.  Both describe the same line and share both endpoints,
    so nothing renders differently — but do not zip the two functions' outputs
    together, or assert that they agree in length, without allowing for it.

    :param skeleton: Skeleton with radii assigned by :func:`pipe_radii`;
        :func:`pipe_radii` is called on it here if they are missing, which
        sets ``skeleton.radii`` as a side effect.
    :param subdivisions: Spline samples per skeleton segment.
    :return: ``[(points (K, 3), radii (K,)), ...]`` per path.
    """
    if skeleton.radii is None:
        pipe_radii(skeleton)
    radii = skeleton.radii
    assert radii is not None  # set above; keeps the type checker honest

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for path in root_to_tip_paths(skeleton):
        if len(path) < 2:
            continue
        raw = skeleton.points[path]
        smoothed = _catmull_rom(raw, subdivisions)
        src_t = np.linspace(0.0, 1.0, len(path))
        dst_t = np.linspace(0.0, 1.0, smoothed.shape[0])
        out.append((smoothed, np.interp(dst_t, src_t, radii[path])))
    return out


def _catmull_rom(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """
    Resample a polyline through a uniform Catmull-Rom spline.

    Endpoints are duplicated so the first and last segments have the
    neighbours the basis needs, which keeps the curve pinned to the original
    ends rather than shrinking away from them.

    :param points: ``(K, 3)`` control points.
    :param samples_per_segment: Output samples per input segment.
    :return: ``(K', 3)`` resampled polyline.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[0] < 3 or samples_per_segment < 2:
        return pts
    padded = np.vstack([pts[0], pts, pts[-1]])
    out = [pts[0]]
    for i in range(pts.shape[0] - 1):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        for step in range(1, samples_per_segment + 1):
            t = step / samples_per_segment
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * (
                    (2 * p1)
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
            )
    return np.asarray(out, dtype=float)


def tree_mesh(
    skeleton: Skeleton,
    *,
    subdivisions: int = 4,
    n_sides: int = 8,
) -> pv.PolyData:
    """
    Sweep the skeleton into a single wood mesh.

    One merged mesh per tree, one actor per tree — the same batching
    discipline the schematic renderer applies to trunks and branch lines.

    :param skeleton: Skeleton from :func:`colonize`.
    :param subdivisions: Spline samples per skeleton segment.
    :param n_sides: Tube cross-section resolution.
    :return: Merged ``PolyData`` of all limbs (empty if nothing grew).
    """
    pv = _pyvista()
    tubes: list[pv.PolyData] = []
    for points, radii in smooth_paths(skeleton, subdivisions=subdivisions):
        spline = pv.Spline(points, points.shape[0])
        spline["radius"] = radii
        tubes.append(spline.tube(scalars="radius", absolute=True, n_sides=n_sides))
    if not tubes:
        return pv.PolyData()
    return tubes[0] if len(tubes) == 1 else pv.merge(tubes)


def _unit(vector: np.ndarray) -> np.ndarray:
    """
    Unit vector, falling back to ``+z`` for a degenerate input.

    :param vector: Any 3-vector.
    :return: Unit-length vector, or ``+z`` if the input is near zero.
    """
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=float) / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])


def leaf_facing(outward: np.ndarray, up_bias: float = 0.6) -> np.ndarray:
    """
    Direction a limb's foliage cluster should face.

    Foliage runs out along the branch and then reaches for light, so the
    cluster axis is the limb's outward direction tilted upward — not world
    ``+z``.  A cluster that always points straight up is the single clearest
    tell that a tree was assembled rather than grown, and it is far more
    obvious in parallax on a light-field panel than in a flat projection.

    Assumes a ``+z``-up world, as the rest of this module does.

    :param outward: Vector from the trunk axis to the branch tip.
    :param up_bias: How strongly foliage reaches upward relative to running
        outward; ``0`` follows the limb exactly, large values return to
        vertical.
    :return: Unit facing vector.
    """
    horizontal = np.asarray(outward, dtype=float).copy()
    horizontal[2] = 0.0
    if float(np.linalg.norm(horizontal)) < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return _unit(_unit(horizontal) + np.array([0.0, 0.0, up_bias]))


def oriented_cluster(
    n_points: int,
    center: np.ndarray,
    facing: np.ndarray,
    radius: float,
) -> list[np.ndarray]:
    """
    A hemispherical cluster of *n_points* around *center*, opening along *facing*.

    Points on the far side are **reflected** across the facing plane rather
    than discarded, so a cluster of any size fills its hemisphere evenly
    instead of thinning out as half the samples are thrown away.

    :param n_points: Number of positions to return.
    :param center: Cluster centre, typically a branch tip.
    :param facing: Unit direction the hemisphere opens toward, from
        :func:`leaf_facing`.
    :param radius: Cluster radius in scene units.
    :return: List of ``(3,)`` positions; empty when *n_points* is not positive.
    """
    if n_points <= 0:
        return []
    sphere = np.asarray(fibonacci_sphere(n_points, radius=radius), dtype=float)
    centre = np.asarray(center, dtype=float)
    behind = np.minimum(sphere @ np.asarray(facing, dtype=float), 0.0)
    return list(centre + sphere - 2.0 * behind[:, None] * np.asarray(facing, dtype=float))


def leaf_frames(
    positions: np.ndarray,
    skeleton: Skeleton,
    *,
    size: float = 0.35,
    cling: float = 0.7,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Leaf positions and aim vectors, without building any mesh.

    This is the placement half of :func:`leaf_glyphs` — the clinging that
    pulls each leaf in to its nearest twig, and the local branch direction
    that orients it — separated out so it can be used by renderers other than
    PyVista.  ``leaf_glyphs`` calls it and then glyphs the result; a POV-Ray
    emitter calls it and instances an analytic ellipsoid per leaf instead.

    Pure NumPy, so it stays available under the ``viz3d`` extra alone.

    :param positions: ``(M, 3)`` leaf positions (the chunk attractors).
    :param skeleton: Grown skeleton, used for local branch direction.
    :param size: Leaf glyph radius, which sets the clearance a clung leaf
        keeps from the wood.
    :param cling: How far each leaf is pulled toward its nearest skeleton
        node, from ``0`` to ``1``.  See :func:`leaf_glyphs`.
    :param seed: RNG seed for the roll jitter.
    :return: ``(points (M, 3), directions (M, 3))``, directions unit-length.
        Both are empty arrays when *positions* is empty.
    """
    pts = np.atleast_2d(np.asarray(positions, dtype=float))
    if pts.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    rng = np.random.default_rng(seed)

    # Local branch direction: the segment ending at the nearest skeleton node.
    dists = np.linalg.norm(pts[:, None, :] - skeleton.points[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)
    parents = skeleton.parents[nearest]

    if cling > 0.0:
        anchor = skeleton.points[nearest]
        offset = pts - anchor
        distance = np.linalg.norm(offset, axis=1, keepdims=True)
        direction = offset / np.maximum(distance, 1e-9)
        radii = skeleton.radii if skeleton.radii is not None else pipe_radii(skeleton)
        clearance = (radii[nearest] + size)[:, None]
        pts = anchor + direction * np.maximum(distance * (1.0 - cling), clearance)

    vecs = np.where(
        parents[:, None] >= 0,
        skeleton.points[nearest] - skeleton.points[np.clip(parents, 0, None)],
        np.array([0.0, 0.0, 1.0]),
    )
    vecs = vecs + rng.normal(0.0, 0.3, vecs.shape)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return pts, vecs / np.maximum(norms, 1e-9)


def leaf_glyphs(
    positions: np.ndarray,
    skeleton: Skeleton,
    *,
    size: float = 0.35,
    tint: np.ndarray | None = None,
    cling: float = 0.7,
    seed: int = 0,
) -> pv.PolyData:
    """
    Leaves as oriented flattened glyphs rather than spheres.

    Each leaf is oriented by the direction of the nearest skeleton segment
    with random roll, so the canopy silhouette is irregular; one glyph call
    covers the whole crown.  At light-field viewing distance the cluster
    shape carries the read, not the individual leaf.

    Leaves are also drawn in toward the twig they belong to.  The crown
    positions come from the schematic layout, and colonization consumes an
    attractor as soon as a branch gets within its kill radius, so left where
    they are a good fraction of leaves hang in mid-air with no wood near them
    — the single most artificial thing about the canopy up close.

    :param positions: ``(M, 3)`` leaf positions (the chunk attractors).
    :param skeleton: Grown skeleton, used for local branch direction.
    :param size: Leaf glyph radius in scene units.
    :param tint: Optional ``(M,)`` scalar per leaf, carried onto the glyphs as
        a ``"tint"`` array so the caller can colour foliage by a lookup table
        (seasonal colour, query illumination) in the same single draw call.
    :param cling: How far each leaf is pulled toward its nearest skeleton
        node, from ``0`` (leave it in the crown volume) to ``1`` (sit on the
        wood).  Leaves never enter the wood: the pull stops at the branch
        radius plus the leaf's own size.
    :param seed: RNG seed for the roll jitter.
    :return: Glyphed ``PolyData``, one actor's worth.
    """
    pv = _pyvista()
    pts, vecs = leaf_frames(positions, skeleton, size=size, cling=cling, seed=seed)
    if pts.size == 0:
        return pv.PolyData()

    cloud = pv.PolyData(pts)
    if tint is not None:
        cloud["tint"] = np.asarray(tint, dtype=float)
    cloud["direction"] = vecs

    # A flattened ellipsoid: cheap, and it silhouettes like foliage.
    proto = pv.Sphere(radius=size, theta_resolution=8, phi_resolution=6)
    proto.scale(LEAF_ASPECT, inplace=True)
    return cloud.glyph(geom=proto, orient="direction", scale=False)


def grow_tree(
    chunk_positions: np.ndarray,
    root: np.ndarray,
    *,
    key: str = "",
    tip_radius: float = 0.05,
    tropism: tuple[float, float, float] = (0.0, 0.0, 0.18),
    **colonize_kwargs,
) -> Skeleton:
    """
    Convenience: colonize then apply pipe radii, seeded from a stable key.

    :param chunk_positions: ``(M, 3)`` crown attractors.
    :param root: ``(3,)`` trunk base.
    :param key: Stable identifier; seeds the RNG so the tree is reproducible.
    :param tip_radius: Radius of leaf-bearing tips, passed to
        :func:`pipe_radii`.
    :param tropism: Growth bias, per genre silhouette.
    :param colonize_kwargs: Forwarded to :func:`colonize`.
    :return: Skeleton with radii assigned.
    """
    skeleton = colonize(
        chunk_positions,
        root,
        tropism=tropism,
        seed=seed_from_key(key),
        **colonize_kwargs,
    )
    pipe_radii(skeleton, tip_radius=tip_radius)
    return skeleton


@dataclass(frozen=True)
class CameraFrame:
    """Where to stand to photograph a tree, in world coordinates.

    Renderer-independent on purpose.  A PyVista caller assigns the three fields
    onto ``plotter.camera``; a POV-Ray caller converts them into a
    ``PovCamera``.  Neither rule is written twice.

    :param position: Eye position.
    :param focal_point: Point looked at, which a light-field renderer also
        takes as the focal plane — the depth that lands on the glass.
    :param up: Up vector.  ``+z``, as everything in this module assumes.
    """

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)


def frame_tree(
    points: np.ndarray,
    *,
    fov: float | None = None,
    standoff: float = 1.5,
    include_root: bool = True,
) -> CameraFrame:
    """
    Frame a grown tree for a hero shot: level view, ``+z`` up, looking along ``+y``.

    The camera stands off along ``-y`` by *standoff* times the subject's
    vertical extent and looks at the centre of it, so the crown straddles the
    focal plane rather than sitting entirely behind it.  On a light-field panel
    that placement is what decides which half of the tree floats out of the
    glass and which half recedes, so it is worth having exactly once.

    **Frame the subject, not the scene.**  Pass the crown — the points the
    skeleton grew toward.  Framing from a renderer's own bounds instead means
    framing whatever happens to be in it: a ground plane three crown-widths
    across drags the centre down and the camera back, and the tree ends up
    small and high in the tile.

    :param points: ``(N, 3)`` subject points, typically the crown attractors.
    :param fov: Vertical field of view in degrees.  Given one, the camera is
        placed at the distance that fits the subject's bounding sphere in it —
        the answer ``plotter.reset_camera()`` computes, which a renderer
        without one has to compute for itself.  ``None`` falls back to
        *standoff*, which is what a PyVista caller wants: it sets a direction
        and lets ``reset_camera()`` do the fitting.
    :param standoff: Camera distance as a multiple of the subject's ``z``
        extent, used only when *fov* is ``None``.  The default matches the
        framing ``gutenkg quilt`` and ``pycodekg quilt`` have used since they
        were written.
    :param include_root: Extend the bounds to the origin, where a grown
        skeleton's root node sits.  The trunk carries no attractors, so without
        this the frame covers the canopy and cuts the tree off at the ankles.
    :return: The :class:`CameraFrame`.
    :raises ValueError: If *points* is empty.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.size == 0:
        raise ValueError("cannot frame an empty point set")

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    if include_root:
        lo, hi = np.minimum(lo, 0.0), np.maximum(hi, 0.0)
    centre = (lo + hi) / 2.0
    depth = float(hi[2] - lo[2]) or 1.0

    if fov is None:
        # Historical rule: stand off from the near face, so the subject
        # straddles the focal plane.  reset_camera() fixes up the distance.
        eye_y = lo[1] - depth * standoff
    else:
        # A fit is a camera-to-centre distance, so it is measured from the
        # focal point — measuring it from the near face would stand the
        # camera a half-depth too far back and undersize the subject.
        radius = float(np.linalg.norm(hi - lo)) / 2.0 or 1.0
        eye_y = centre[1] - radius / max(np.tan(np.radians(float(fov) / 2.0)), 1e-6)

    return CameraFrame(
        position=(float(centre[0]), float(eye_y), float(centre[2])),
        focal_point=(float(centre[0]), float(centre[1]), float(centre[2])),
        up=(0.0, 0.0, 1.0),
    )
