"""
kg_utils/stats.py

The fleet's one shape for a knowledge graph's statistics.

Every module's public ``stats()``, and so every server's canonical ``stats``
MCP tool, returns::

    {
        "kind": "doc",
        "nodes": {"total": 1234, "by_kind": {"chunk": 900, "section": 334}},
        "edges": {"total": 5678, "by_rel": {"CONTAINS": 1234, "NEXT": 4444}},
        ...domain extras, at top level...
    }

``total`` counts every row, and ``by_kind`` / ``by_rel`` are always present and
sum to it, so two KGs' counts mean the same thing. A domain's own measure (a
code graph's ``meaningful_nodes``, a diary's ``temporal_span``) is an extra
under its own name, never a stand-in for ``total``.

This is the *public* shape. A store's own ``stats()`` keeps its flat
``total_nodes`` / ``node_counts`` keys, which snapshots persist; nothing here
changes what a snapshot records.

Settled 2026-10-25, kgrag_priv FLEET_STANDARDS "KG stats shape".

Author: Eric G. Suchanek, PhD
License: Elastic 2.0
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["RESERVED_KEYS", "flat_stats", "kg_stats", "stats_from_store", "stats_problems"]

#: The keys the shape owns. An extra may not use them.
RESERVED_KEYS: frozenset[str] = frozenset({"kind", "nodes", "edges"})

#: A store's flat keys that the shape replaces; :func:`stats_from_store` drops them.
_FLAT_KEYS: frozenset[str] = frozenset({"total_nodes", "total_edges", "node_counts", "edge_counts"})


def kg_stats(
    kind: str,
    node_counts: Mapping[str, int],
    edge_counts: Mapping[str, int],
    /,
    **extras: Any,
) -> dict[str, Any]:
    """Build a KG's statistics in the fleet shape.

    Totals are summed from the breakdowns, so the two cannot disagree.

    :param kind: The KG kind, as the server advertises it (``"doc"``, ``"code"``).
    :param node_counts: Node count per node kind.
    :param edge_counts: Edge count per relation.
    :param extras: Domain statistics, kept at top level under their own names.
    :return: ``{"kind", "nodes": {"total", "by_kind"}, "edges": {"total", "by_rel"}, **extras}``.
    :raises ValueError: If an extra uses a reserved key.
    """
    clash = RESERVED_KEYS & extras.keys()
    if clash:
        raise ValueError(f"stats extras may not use the reserved keys {sorted(clash)}")
    by_kind = {str(k): int(v) for k, v in node_counts.items()}
    by_rel = {str(k): int(v) for k, v in edge_counts.items()}
    return {
        "kind": kind,
        "nodes": {"total": sum(by_kind.values()), "by_kind": by_kind},
        "edges": {"total": sum(by_rel.values()), "by_rel": by_rel},
        **extras,
    }


def stats_from_store(kind: str, store_stats: Mapping[str, Any], /, **extras: Any) -> dict[str, Any]:
    """Reshape a store's flat ``stats()`` into the fleet shape.

    Takes the ``node_counts`` / ``edge_counts`` breakdowns every fleet store
    reports, drops the flat totals the shape replaces, and keeps every other
    key (``db_path``, ``meaningful_nodes``, ...) as an extra.

    :param kind: The KG kind.
    :param store_stats: A store's ``stats()`` dict, with ``node_counts`` and ``edge_counts``.
    :param extras: More domain statistics; they win over the store's own keys.
    :return: The fleet shape.
    :raises KeyError: If the store reports no ``node_counts`` or ``edge_counts``.
    """
    rest = {k: v for k, v in store_stats.items() if k not in _FLAT_KEYS}
    return kg_stats(
        kind, store_stats["node_counts"], store_stats["edge_counts"], **{**rest, **extras}
    )


def flat_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Return *stats* in a store's flat keys, the form snapshots persist.

    The inverse of :func:`stats_from_store`: ``nodes``/``edges`` become
    ``total_nodes``, ``node_counts``, ``total_edges`` and ``edge_counts``, and
    every extra is kept. A dict already in the flat form comes back as a copy,
    so a snapshot records the same keys whichever ``stats()`` it was handed.

    :param stats: A fleet-shape or flat statistics dict.
    :return: The flat form.
    """
    nodes, edges = stats.get("nodes"), stats.get("edges")
    if not (isinstance(nodes, Mapping) and isinstance(edges, Mapping)):
        return dict(stats)
    flat = {k: v for k, v in stats.items() if k not in RESERVED_KEYS}
    flat.update(
        total_nodes=nodes.get("total", 0),
        node_counts=dict(nodes.get("by_kind", {})),
        total_edges=edges.get("total", 0),
        edge_counts=dict(edges.get("by_rel", {})),
    )
    return flat


def stats_problems(stats: Any) -> list[str]:
    """Every way *stats* departs from the fleet shape; empty if it conforms.

    :param stats: What a ``stats()`` returned.
    :return: Human-readable problems, one per departure.
    """
    if not isinstance(stats, Mapping):
        return [f"stats must be a mapping, got {type(stats).__name__}"]
    problems: list[str] = []
    if not isinstance(stats.get("kind"), str) or not stats.get("kind"):
        problems.append("`kind` must be a non-empty string")
    for section, breakdown in (("nodes", "by_kind"), ("edges", "by_rel")):
        part = stats.get(section)
        if not isinstance(part, Mapping):
            problems.append(f"`{section}` must be an object with `total` and `{breakdown}`")
            continue
        total, counts = part.get("total"), part.get(breakdown)
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            problems.append(f"`{section}.total` must be a non-negative integer")
        if not isinstance(counts, Mapping) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in counts.values()
        ):
            problems.append(f"`{section}.{breakdown}` must map names to integers")
        elif isinstance(total, int) and sum(counts.values()) != total:
            problems.append(
                f"`{section}.{breakdown}` sums to {sum(counts.values())}, not `total` {total}"
            )
    return problems
