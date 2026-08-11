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
