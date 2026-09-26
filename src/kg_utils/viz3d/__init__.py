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

:mod:`~kg_utils.viz3d.species` makes those trees look like species: a
:class:`Habit` shapes the crown envelope and how the wood grows toward it, and
:data:`SPECIES` holds nine tuned presets (oak, chestnut, fir, plane,
blackthorn, pine, birch, willow, poplar).
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
    BARK_TILE,
    DROOP_PER_NODE,
    DROOP_STIFF,
    LEAF_ASPECT,
    MAX_ATTRACTORS,
    PIPE_EXPONENT,
    BarkSweep,
    CameraFrame,
    Skeleton,
    bark_mesh,
    bark_sweep,
    colonize,
    crown_spacing,
    droop_skeleton,
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
from kg_utils.viz3d.species import (
    CROWN_TOP,
    DEFAULT_HABIT,
    ENVELOPES,
    SPECIES,
    Habit,
    crown_sections,
    envelope_width,
    section_cluster,
    species_table,
    vary_habit,
)

__all__ = [
    "BARK_TILE",
    "CROWN_TOP",
    "DEFAULT_HABIT",
    "DEFAULT_LEVEL_SIZE",
    "DROOP_PER_NODE",
    "DROOP_STIFF",
    "ENVELOPES",
    "LEAF_ASPECT",
    "MAX_ATTRACTORS",
    "PIPE_EXPONENT",
    "SPECIES",
    "AlliumLayout",
    "BarkSweep",
    "CameraFrame",
    "FunnelLayout",
    "Habit",
    "Layout3D",
    "LayoutEdge",
    "LayoutNode",
    "Skeleton",
    "bark_mesh",
    "bark_sweep",
    "colonize",
    "crown_sections",
    "crown_spacing",
    "droop_skeleton",
    "envelope_width",
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
    "section_cluster",
    "seed_from_key",
    "smooth_paths",
    "species_table",
    "tree_mesh",
    "vary_habit",
]
