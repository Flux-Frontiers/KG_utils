"""
Declarative visual vocabulary for graph renderers.

A :class:`GraphTheme` says how one domain's node kinds and edge relations should
look.  It is data, not code: each KG module constructs one and hands it to the
renderer, so all modules share a single rendering implementation while naming
their own kinds.  ``code`` has modules and functions, ``doc`` has sections and
chunks, ``meta`` has compounds and reactions — only ``id``, ``kind`` and
``name`` are common to all of them.

Typical use::

    THEME = GraphTheme(
        kinds={
            "document": KindStyle("#4A90D9", shape="box", size=18),
            "chunk": KindStyle("#27AE60", shape="ellipse"),
        },
        fallback=KindStyle("#95A5A6", shape="triangle"),
        relations={"CONTAINS": "#BDC3C7", "SIMILAR_TO": "#E74C3C"},
    )
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

__all__ = ["GraphTheme", "KindStyle", "with_alpha"]

#: Used when a theme names no relation colour and declares no fallback.
DEFAULT_RELATION_COLOR: Final[str] = "#888888"


@dataclass(frozen=True)
class KindStyle:
    """How one node kind is drawn.

    :param color: Fill colour as ``#RRGGBB``.  The renderer converts this to
        ``rgba()`` when a centrality metric drives opacity, so six-digit hex is
        required for fading to work.
    :param shape: vis.js node shape — ``dot``, ``box``, ``ellipse``,
        ``diamond``, ``triangle``, ``star``, and so on.
    :param size: Node diameter in pixels when no metric is driving size.
    :param radius: Node radius in world units for 3-D renderers.  Ignored by the
        HTML renderer; carried here so one theme can serve both.
    """

    color: str
    shape: str = "dot"
    size: int = 12
    radius: float = 0.7


@dataclass(frozen=True)
class GraphTheme:
    """The full visual vocabulary for one knowledge-graph domain.

    :param kinds: Node kind to :class:`KindStyle`.
    :param fallback: Style for kinds absent from *kinds*.  Every graph
        eventually meets a kind nobody planned for, and silently dropping those
        nodes is worse than drawing them grey.
    :param relations: Edge relation to colour.
    :param relation_fallback: Colour for relations absent from *relations*.
    :param resolve_kind: Optional hook mapping a node to a *render* kind that
        need not exist in the store — for example drawing underscore-prefixed
        functions differently.  Receives the whole node dict.  Must return a key
        present in *kinds*, or anything else to get *fallback*.
    """

    kinds: Mapping[str, KindStyle]
    fallback: KindStyle
    relations: Mapping[str, str] = field(default_factory=dict)
    relation_fallback: str = DEFAULT_RELATION_COLOR
    resolve_kind: Callable[[Mapping[str, Any]], str] | None = None

    def kind_of(self, node: Mapping[str, Any]) -> str:
        """Return the render kind for *node*.

        :param node: Node attribute mapping.
        :return: A kind name, which may be a render-only kind from
            :attr:`resolve_kind` rather than one the store knows about.
        """
        if self.resolve_kind is not None:
            return self.resolve_kind(node)
        return str(node.get("kind", ""))

    def style_of(self, node: Mapping[str, Any]) -> KindStyle:
        """Return the style for *node*, falling back rather than raising.

        :param node: Node attribute mapping.
        :return: The matching :class:`KindStyle`, or :attr:`fallback`.
        """
        return self.kinds.get(self.kind_of(node), self.fallback)

    def style_for_kind(self, kind: str) -> KindStyle:
        """Return the style for a kind name.

        :param kind: Kind name.
        :return: The matching :class:`KindStyle`, or :attr:`fallback`.
        """
        return self.kinds.get(kind, self.fallback)

    def relation_color(self, relation: str) -> str:
        """Return the colour for an edge relation.

        :param relation: Relation name.
        :return: Hex colour string.
        """
        return self.relations.get(relation, self.relation_fallback)


def with_alpha(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` to an ``rgba()`` string at the given opacity.

    vis.js accepts CSS colour strings for node backgrounds, so opacity is
    expressed by converting the palette hex rather than by compositing against
    the canvas background.

    :param hex_color: Colour in ``#RRGGBB`` form, with or without the leading
        ``#``.
    :param alpha: Opacity in ``[0, 1]``; values outside the range are clamped.
    :return: An ``rgba(r, g, b, a)`` string, or *hex_color* unchanged when it is
        not a six-digit hex colour — a CSS keyword should degrade to an opaque
        colour rather than crash a render.
    """
    raw = hex_color.lstrip("#")
    if len(raw) != 6:
        return hex_color
    try:
        r, g, b = (int(raw[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return hex_color
    a = min(1.0, max(0.0, alpha))
    return f"rgba({r}, {g}, {b}, {a:.3f})"
