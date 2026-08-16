"""Shared 3-D graph layout for KG modules.

Requires the ``viz3d`` extra::

    pip install 'kgmodule-utils[viz3d]'

A layout maps a node list and an edge list onto ``{node_id: [x, y, z]}``.  It
draws nothing, so the same layout feeds a PyVista desktop viewer, an off-screen
light-field renderer, or a plain scatter plot — each KG module keeps its own
renderer and shares the spatial reasoning.

Domain differences are supplied as constructor arguments — which kind is a root,
which relation means containment, which kind sits on which Z level — so a code
graph, a document corpus, and a metabolic network share one implementation.

:mod:`~kg_utils.viz3d.organic` is the other half: instead of placing nodes on a
lattice it *grows* a tree skeleton toward them by space colonization, so a
corpus reads as wood rather than as a scatter plot.  Its geometry is NumPy-only
like the layouts; only its three mesh builders need PyVista, which they import
lazily, so this extra stays light for callers that just want positions.
"""

from kg_utils.viz3d.layout import (
    DEFAULT_LEVEL_SIZE,
    AlliumLayout,
    FunnelLayout,
    Layout3D,
    LayoutEdge,
    LayoutNode,
    fibonacci_annulus,
    fibonacci_sphere,
    golden_spiral_2d,
)
from kg_utils.viz3d.organic import (
    LEAF_ASPECT,
    MAX_ATTRACTORS,
    PIPE_EXPONENT,
    CameraFrame,
    Skeleton,
    colonize,
    crown_spacing,
    frame_tree,
    grow_tree,
    leaf_facing,
    leaf_frames,
    leaf_glyphs,
    limb_paths,
    oriented_cluster,
    pipe_radii,
    root_to_tip_paths,
    seed_from_key,
    smooth_paths,
    tree_mesh,
)

__all__ = [
    "DEFAULT_LEVEL_SIZE",
    "LEAF_ASPECT",
    "MAX_ATTRACTORS",
    "PIPE_EXPONENT",
    "AlliumLayout",
    "CameraFrame",
    "FunnelLayout",
    "Layout3D",
    "LayoutEdge",
    "LayoutNode",
    "Skeleton",
    "colonize",
    "crown_spacing",
    "fibonacci_annulus",
    "fibonacci_sphere",
    "frame_tree",
    "golden_spiral_2d",
    "grow_tree",
    "leaf_facing",
    "leaf_frames",
    "leaf_glyphs",
    "limb_paths",
    "oriented_cluster",
    "pipe_radii",
    "root_to_tip_paths",
    "seed_from_key",
    "smooth_paths",
    "tree_mesh",
]
