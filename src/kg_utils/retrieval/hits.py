# © 2026 Eric G. Suchanek, PhD — Flux-Frontiers · SPDX-License-Identifier: Elastic-2.0
"""Hit serialization and content hydration helpers for KG retrieval responses."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

__all__ = ["hit_to_dict", "attach_content_by_sqlite"]


def _is_diary_kind(kind_value: Any) -> bool:
    kind_str = str(kind_value)
    return kind_str == "KGKind.DIARY" or kind_str.lower().endswith("diary")


def hit_to_dict(hit: Any, include_diary_timestamp: bool = False) -> dict:
    """Serialize a KGRAG hit object into a plain dictionary.

    :param hit: Hit-like object with standard retrieval attributes.
    :param include_diary_timestamp: Include ``timestamp`` field for diary hits.
    :returns: Serialized hit dictionary.
    """
    out = {
        "kg_name": hit.kg_name,
        "kg_kind": str(hit.kg_kind),
        "node_id": hit.node_id,
        "name": hit.name,
        "kind": hit.kind,
        "score": round(float(hit.score), 4),
        "summary": hit.summary,
        "source_path": hit.source_path,
    }
    if include_diary_timestamp:
        out["timestamp"] = hit.name if _is_diary_kind(hit.kg_kind) else None
    return out


def attach_content_by_sqlite(hits: list[dict], kg_sqlite_map: dict[str, Path]) -> None:
    """Attach full node text under ``content`` via batched SQLite lookups.

    Missing or unreadable databases are ignored to preserve permissive behavior.

    :param hits: Mutable hit dictionaries. Each hit should include ``kg_name`` and ``node_id``.
    :param kg_sqlite_map: Mapping of KG name to sqlite database path.
    """
    by_kg: dict[str, list[dict]] = defaultdict(list)
    for hit in hits:
        by_kg[hit.get("kg_name", "")].append(hit)

    for kg_name, kg_hits in by_kg.items():
        db_path = kg_sqlite_map.get(kg_name)
        if not db_path or not Path(db_path).exists():
            continue

        ids = [h.get("node_id") for h in kg_hits if h.get("node_id")]
        if not ids:
            continue

        text_by_id: dict[str, str] = {}
        try:
            with sqlite3.connect(str(db_path)) as con:
                placeholders = ",".join("?" * len(ids))
                query = f"SELECT id, text FROM nodes WHERE id IN ({placeholders})"
                for node_id, text in con.execute(query, ids):
                    text_by_id[node_id] = text or ""
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            continue

        for hit in kg_hits:
            node_id = hit.get("node_id")
            if node_id:
                hit["content"] = text_by_id.get(node_id, "")
