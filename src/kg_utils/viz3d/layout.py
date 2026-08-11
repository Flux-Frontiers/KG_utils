"""
Pluggable 3-D layout engine for knowledge graphs.

A layout answers one question: where in space does each node go?  It takes a
node list and an edge list and returns ``{node_id: np.ndarray([x, y, z])}``.
Nothing here draws anything, so the same layouts serve a PyVista desktop
viewer, an off-screen light-field renderer, or a plain matplotlib scatter.

Provides an abstract :class:`Layout3D` base class and two concrete
implementations:

- :class:`AlliumLayout`: each root node is rendered as a Giant Allium plant
  (a vertical stem with a Fibonacci-sphere "head" of its children).  Roots are
  arranged in a Fibonacci annulus in the XY plane.

- :class:`FunnelLayout`: node kind determines the Z level (roots at the bottom,
  their children above, and so on).  XY positions are spread via a golden-angle
  spiral within each layer.

Both are domain-neutral: which kind counts as a root, which relation expresses
containment, and which kind sits on which Z level are all constructor
parameters, so a code graph, a document corpus, and a metabolic network can
share one implementation.

The Fibonacci utilities (:func:`fibonacci_sphere`, :func:`fibonacci_annulus`)
are adapted from *repo_vis* ``pkg_visualizer/utility.py``
(Eric G. Suchanek, PhD — https://github.com/Suchanek/repo_vis).

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_LEVEL_SIZE",
    "AlliumLayout",
    "FunnelLayout",
    "Layout3D",
    "LayoutEdge",
    "LayoutNode",
    "fibonacci_annulus",
    "fibonacci_sphere",
    "golden_spiral_2d",
]

#: Node radius assumed for a Z level absent from a :class:`FunnelLayout`'s
#: ``level_sizes`` map.
DEFAULT_LEVEL_SIZE: float = 0.7


# ---------------------------------------------------------------------------
# Fibonacci spatial utilities  (adapted from repo_vis/pkg_visualizer/utility.py)
# ---------------------------------------------------------------------------


def fibonacci_sphere(
    samples: int,
    radius: float = 1.0,
    center: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Distribute *samples* points uniformly on a sphere using the Fibonacci spiral.

    Adapted from ``utility.fibonacci_sphere`` in *repo_vis*.

    :param samples: Number of points to generate.
    :param radius: Sphere radius.
    :param center: Centre of the sphere (default: origin).
    :return: List of 3-D coordinate arrays.
    """
    if center is None:
        center = np.zeros(3)
    if samples <= 0:
        return []
    if samples == 1:
        return [center + radius * np.array([0.0, 0.0, 1.0])]

    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    points: list[np.ndarray] = []
    for i in range(samples):
        y = 1.0 - (i / float(samples - 1)) * 2.0
        r_at_y = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = np.cos(theta) * r_at_y
        z = np.sin(theta) * r_at_y
        points.append(center + radius * np.array([x, y, z]))
    return points


def fibonacci_annulus(
    samples: int,
    inner_radius: float = 1.0,
    outer_radius: float = 2.0,
    center: np.ndarray | None = None,
    z_thickness: float = 0.2,
) -> list[np.ndarray]:
    """
    Distribute *samples* points in a flat annular ring in the XY plane.

    A small Z jitter (``z_thickness``) adds visual depth when non-zero.  The
    jitter is drawn from a fixed seed, so repeated calls with the same arguments
    return the same points — layouts must be stable across renders.
    Adapted from ``utility.fibonacci_annulus`` in *repo_vis*.

    :param samples: Number of points to generate.
    :param inner_radius: Inner radius of the annulus.
    :param outer_radius: Outer radius of the annulus.
    :param center: Centre of the annulus (default: origin).
    :param z_thickness: Half-range of Z jitter applied to each point.
    :return: List of 3-D coordinate arrays.
    """
    if center is None:
        center = np.zeros(3)
    if samples <= 0:
        return []
    if samples == 1:
        mid = (inner_radius + outer_radius) / 2.0
        return [center + np.array([mid, 0.0, 0.0])]

    phi = np.pi * (3.0 - np.sqrt(5.0))
    r_range = outer_radius - inner_radius
    r_step = r_range / max(samples - 1, 1)
    rng = np.random.default_rng(42)  # deterministic jitter seed

    points: list[np.ndarray] = []
    for i in range(samples):
        r = inner_radius + i * r_step
        theta = phi * i
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        z = (rng.random() * 2.0 - 1.0) * z_thickness
        points.append(center + np.array([x, y, z]))
    return points


def golden_spiral_2d(
    samples: int,
    radius: float = 1.0,
    center: np.ndarray | None = None,
    z: float = 0.0,
) -> list[np.ndarray]:
    """
    Place *samples* points in the XY plane using a golden-angle disc spiral.

    :param samples: Number of points.
    :param radius: Outer radius of the disc.
    :param center: XY centre (Z component ignored; overridden by *z*).
    :param z: Fixed Z coordinate for all output points.
    :return: List of 3-D coordinate arrays.
    """
    if center is None:
        center = np.zeros(3)
    if samples <= 0:
        return []

    phi = np.pi * (3.0 - np.sqrt(5.0))
    points: list[np.ndarray] = []
    for i in range(samples):
        r = radius * np.sqrt(i / max(samples - 1, 1))
        theta = phi * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append(center + np.array([x, y, z]))
    return points


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class LayoutNode:
    """
    Thin, domain-neutral wrapper around a node dict from a graph store.

    Only ``id``, ``kind`` and ``name`` are common to every knowledge-graph
    domain; the remaining fields are optional provenance that renderers use for
    tooltips and size cues.  Their names come from the code-graph origin of this
    module, but they are read generically: *any* source path, *any* summary
    text, *any* line span.  A document corpus puts the file path in
    ``module_path`` and the chunk text in ``docstring``.

    :param id: Stable node identifier (e.g. ``mod:src/foo.py``).
    :param kind: Node kind — the domain's own vocabulary, e.g. ``module`` /
        ``class`` / ``function`` for code, ``document`` / ``section`` /
        ``chunk`` for documents.
    :param name: Short display name of the node.
    :param module_path: Source path this node came from (may be ``None``).
    :param docstring: Summary or body text (may be ``None``).
    :param lineno: First source line number (may be ``None``).
    :param end_lineno: Last source line number (may be ``None``).
    """

    id: str
    kind: str
    name: str
    module_path: str | None = None
    docstring: str | None = None
    lineno: int | None = None
    end_lineno: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> LayoutNode:
        """Construct from a graph-store node dict.

        :param d: Dict with keys ``id``, ``kind``, ``name``, etc.
        :return: New :class:`LayoutNode`.
        """
        return cls(
            id=d["id"],
            kind=d["kind"],
            name=d["name"],
            module_path=d.get("module_path"),
            docstring=d.get("docstring"),
            lineno=d.get("lineno"),
            end_lineno=d.get("end_lineno"),
        )

    @property
    def line_count(self) -> int:
        """Approximate source size in lines (0 if unknown).

        :return: ``end_lineno - lineno`` or 0.
        """
        if self.lineno and self.end_lineno:
            return max(0, self.end_lineno - self.lineno)
        return 0


@dataclass
class LayoutEdge:
    """
    Thin, domain-neutral wrapper around an edge dict from a graph store.

    :param src: Source node ID.
    :param rel: Relation type in the domain's own vocabulary — ``CONTAINS``,
        ``CALLS``, ``IMPORTS``, ``INHERITS`` for code; ``CONTAINS``,
        ``SIMILAR_TO``, ``MENTIONS`` for documents.
    :param dst: Destination node ID.
    """

    src: str
    rel: str
    dst: str

    @classmethod
    def from_dict(cls, d: dict) -> LayoutEdge:
        """Construct from a graph-store edge dict.

        :param d: Dict with keys ``src``, ``rel``, ``dst``.
        :return: New :class:`LayoutEdge`.
        """
        return cls(src=d["src"], rel=d["rel"], dst=d["dst"])


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Layout3D(ABC):
    """
    Abstract base class for 3-D graph layout strategies.

    Subclasses implement :meth:`compute` to assign a 3-D position to every
    node, returning a ``{node_id: np.ndarray([x, y, z])}`` mapping that a
    renderer consumes.
    """

    @abstractmethod
    def compute(
        self,
        nodes: list[LayoutNode],
        edges: list[LayoutEdge],
    ) -> dict[str, np.ndarray]:
        """
        Compute 3-D positions for all *nodes*.

        :param nodes: All nodes in the graph.
        :param edges: All edges in the graph (used to derive hierarchy).
        :return: Mapping from node ID to ``[x, y, z]`` position.
        """
        ...


# ---------------------------------------------------------------------------
# AlliumLayout
# ---------------------------------------------------------------------------


class AlliumLayout(Layout3D):
    """
    Allium-plant layout: each root node is visualised as a Giant Allium flower.

    Spatial structure:

    - **Stem base** — the root node sits at an XY position in a Fibonacci
      annulus at ``Z = 0``.
    - **Head** — its direct children are distributed on a Fibonacci sphere
      centred at the stem apex (``Z = stem_height``).  Head radius scales with
      ``sqrt(n_children)``.
    - **Florets** — grandchildren orbit their parent on a smaller Fibonacci
      sphere, slightly above the head.
    - **Orphans** — nodes with no containment parent cluster on a small sphere
      at the origin.

    Multiple alliums are arranged in a Fibonacci annulus in the XY plane so
    they are evenly spaced regardless of count.  Roots take their slots in the
    order they appear in the node list, so **callers must pass a stable order**
    — a store whose ordering varies between rebuilds will make the whole scene
    shuffle even when the graph has not changed.

    Inspired by :class:`GiantAllium` in *repo_vis/pkg_visualizer/plants3d.py*.

    :param stem_height: Height of each allium stem (Z offset of the head).
    :param base_head_radius: Minimum radius for the Fibonacci sphere head.
    :param method_orbit_radius: Base radius for grandchild sub-spheres.
    :param annulus_inner_radius: Inner radius of the root placement ring.
    :param annulus_outer_radius: Minimum outer radius (auto-scaled for large graphs).
    :param root_kind: Node kind that forms the stems — ``module`` for a code
        graph, ``document`` for a corpus.  Nodes with no containment parent
        stand in when no node has this kind.
    :param contains_rel: Relation expressing containment, used to derive the
        parent/child hierarchy.
    """

    def __init__(
        self,
        stem_height: float = 8.0,
        base_head_radius: float = 2.0,
        method_orbit_radius: float = 0.8,
        annulus_inner_radius: float = 8.0,
        annulus_outer_radius: float = 20.0,
        root_kind: str = "module",
        contains_rel: str = "CONTAINS",
    ) -> None:
        """Initialise layout parameters.

        :param stem_height: Vertical height of each allium stem.
        :param base_head_radius: Minimum allium head sphere radius.
        :param method_orbit_radius: Base orbit radius for grandchildren.
        :param annulus_inner_radius: Inner radius for root ring placement.
        :param annulus_outer_radius: Minimum outer radius for root ring.
        :param root_kind: Node kind forming the stems.
        :param contains_rel: Relation expressing containment.
        """
        self.stem_height = stem_height
        self.base_head_radius = base_head_radius
        self.method_orbit_radius = method_orbit_radius
        self.annulus_inner_radius = annulus_inner_radius
        self.annulus_outer_radius = annulus_outer_radius
        self.root_kind = root_kind
        self.contains_rel = contains_rel

    def compute(
        self,
        nodes: list[LayoutNode],
        edges: list[LayoutEdge],
    ) -> dict[str, np.ndarray]:
        """
        Compute allium-plant 3-D positions for all nodes.

        :param nodes: All nodes in the graph.
        :param edges: All edges (:attr:`contains_rel` used to derive hierarchy).
        :return: Mapping from node ID to ``[x, y, z]`` position.
        """
        # Build the containment hierarchy: child_id -> parent_id,
        # parent_id -> [child_ids]
        parent: dict[str, str] = {}
        children: dict[str, list[str]] = {}
        for e in edges:
            if e.rel == self.contains_rel:
                parent[e.dst] = e.src
                children.setdefault(e.src, []).append(e.dst)

        node_by_id: dict[str, LayoutNode] = {n.id: n for n in nodes}
        positions: dict[str, np.ndarray] = {}

        # Root nodes form the allium stems
        roots = [n for n in nodes if n.kind == self.root_kind]
        if not roots:
            # Fallback: treat nodes without a containment parent as pseudo-roots
            roots = [n for n in nodes if n.id not in parent]

        n_roots = len(roots)
        inner = self.annulus_inner_radius
        # Scale outer radius so stems don't crowd each other
        outer = max(self.annulus_outer_radius, inner + np.sqrt(n_roots) * 4.0)

        root_positions = fibonacci_annulus(
            n_roots,
            inner_radius=inner,
            outer_radius=outer,
            center=np.zeros(3),
            z_thickness=0.0,  # flat ring — alliums stand vertically
        )

        for root_node, root_pos in zip(roots, root_positions):
            positions[root_node.id] = np.array(root_pos)
            stem_apex = np.array([root_pos[0], root_pos[1], self.stem_height])

            direct_ids = children.get(root_node.id, [])
            direct = [node_by_id[cid] for cid in direct_ids if cid in node_by_id]
            n_direct = len(direct)
            if not direct:
                continue

            # Head radius scales with child count
            head_r = self.base_head_radius + np.sqrt(n_direct) * 0.4
            head_positions = fibonacci_sphere(n_direct, radius=head_r, center=stem_apex)

            for child, child_pos in zip(direct, head_positions):
                positions[child.id] = np.array(child_pos)

                # Grandchildren orbit their parent
                grand_ids = children.get(child.id, [])
                grand = [node_by_id[gid] for gid in grand_ids if gid in node_by_id]
                n_grand = len(grand)
                if not grand:
                    continue

                orbit_r = self.method_orbit_radius + np.sqrt(n_grand) * 0.15
                orbit_positions = fibonacci_sphere(
                    n_grand, radius=orbit_r, center=np.array(child_pos)
                )
                for gc, gc_pos in zip(grand, orbit_positions):
                    positions[gc.id] = np.array(gc_pos)

        # Orphan nodes: anything not yet placed (stubs, unrooted nodes)
        orphans = [n for n in nodes if n.id not in positions]
        if orphans:
            orphan_r = 3.0
            orphan_positions = fibonacci_sphere(
                len(orphans), radius=orphan_r, center=np.array([0.0, 0.0, orphan_r])
            )
            for n, pos in zip(orphans, orphan_positions):
                positions[n.id] = np.array(pos)

        return positions


# ---------------------------------------------------------------------------
# FunnelLayout
# ---------------------------------------------------------------------------


class FunnelLayout(Layout3D):
    """
    Stratified layout: node *kind* determines the Z layer; XY positions use a
    golden-angle disc spiral within each layer.

    For a code graph the layers run modules → classes → functions/methods →
    symbol stubs, bottom to top; a document corpus might run documents →
    sections → chunks.  The mapping is supplied by the caller via *zlevels*,
    since only the domain knows its own hierarchy.

    Cross-cutting edges arc between layers, making structural coupling
    immediately visible from any angle.

    Disc radius is derived algorithmically:
    ``r = node_spacing * node_size * sqrt(n)``
    so the layout scales correctly for graphs of any size without hand-tuning.

    :param layer_gap: Vertical distance between adjacent layers.
    :param node_spacing: Spacing multiplier — larger spreads layers out more.
    :param zlevels: Node kind to Z level.  When this is ``None`` every node
        lands on *default_level* — a flat disc rather than a funnel, which is
        the honest rendering of "no hierarchy was declared".
    :param level_sizes: Representative node radius per Z level, used to derive
        each disc's radius.  Levels absent from the map use
        :data:`DEFAULT_LEVEL_SIZE`.
    :param default_level: Z level for kinds absent from *zlevels*.  Domains
        that reserve a top layer for unrecognised stubs pass that layer's
        index here.
    """

    def __init__(
        self,
        layer_gap: float = 12.0,
        node_spacing: float = 2.0,
        zlevels: Mapping[str, int] | None = None,
        level_sizes: Mapping[int, float] | None = None,
        default_level: int = 0,
    ) -> None:
        """Initialise layout parameters.

        :param layer_gap: Vertical separation between layers.
        :param node_spacing: Controls minimum gap between node surfaces.
        :param zlevels: Node kind to Z level.
        :param level_sizes: Node radius per Z level.
        :param default_level: Z level for kinds absent from *zlevels*.
        """
        self.layer_gap = layer_gap
        self.node_spacing = node_spacing
        self.zlevels: dict[str, int] = dict(zlevels or {})
        self.level_sizes: dict[int, float] = dict(level_sizes or {})
        self.default_level = default_level

    def compute(
        self,
        nodes: list[LayoutNode],
        edges: list[LayoutEdge],
    ) -> dict[str, np.ndarray]:
        """
        Compute funnel 3-D positions for all nodes.

        :param nodes: All nodes in the graph.
        :param edges: Unused by this layout (present for API compatibility).
        :return: Mapping from node ID to ``[x, y, z]`` position.
        """
        # Group nodes by Z layer
        layers: dict[int, list[LayoutNode]] = {}
        for n in nodes:
            level = self.zlevels.get(n.kind, self.default_level)
            layers.setdefault(level, []).append(n)

        positions: dict[str, np.ndarray] = {}

        for level, layer_nodes in layers.items():
            z = level * self.layer_gap
            node_size = self.level_sizes.get(level, DEFAULT_LEVEL_SIZE)
            # Derived radius: scales with sqrt(n) and node size so no manual
            # tuning is needed as the graph grows
            r = self.node_spacing * node_size * np.sqrt(len(layer_nodes))
            r = max(r, 4.0)
            pts = golden_spiral_2d(len(layer_nodes), radius=r, z=z)
            for n, pt in zip(layer_nodes, pts):
                positions[n.id] = pt

        return positions
