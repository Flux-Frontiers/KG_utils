"""Shared graph visualisation for KG modules.

Requires the ``viz`` extra::

    pip install 'kgmodule-utils[viz]'

Domain differences are supplied as data — a :class:`GraphTheme` naming the
domain's kinds and relations, and a :class:`TooltipSpec` naming the fields worth
showing — so every module shares one rendering implementation.
"""

from kg_utils.viz.graph_html import (
    CENTRALITY_MIN_OPACITY,
    CENTRALITY_SIZE_RANGE,
    SEED_FRACTION,
    build_graph_html,
    select_nodes,
)
from kg_utils.viz.theme import GraphTheme, KindStyle, with_alpha
from kg_utils.viz.tooltip import TooltipRow, TooltipSpec

__all__ = [
    "CENTRALITY_MIN_OPACITY",
    "CENTRALITY_SIZE_RANGE",
    "SEED_FRACTION",
    "GraphTheme",
    "KindStyle",
    "TooltipRow",
    "TooltipSpec",
    "build_graph_html",
    "select_nodes",
    "with_alpha",
]
