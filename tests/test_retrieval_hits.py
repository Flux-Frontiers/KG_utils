"""Tests for kg_utils.retrieval.hits."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from kg_utils.retrieval.hits import attach_content_by_sqlite, hit_to_dict


def _mk_hit(**kwargs):
    defaults = {
        "kg_name": "gutenberg",
        "kg_kind": "KGKind.GUTENBERG",
        "node_id": "n1",
        "name": "Sample",
        "kind": "chunk",
        "score": 0.91234,
        "summary": "summary",
        "source_path": "genre/book.txt",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_hit_to_dict_basics() -> None:
    hit = _mk_hit(score=0.98765)
    out = hit_to_dict(hit)
    assert out["kg_name"] == "gutenberg"
    assert out["kg_kind"] == "KGKind.GUTENBERG"
    assert out["score"] == 0.9877
    assert "timestamp" not in out


def test_hit_to_dict_includes_diary_timestamp() -> None:
    hit = _mk_hit(kg_kind="KGKind.DIARY", name="1666-09-03")
    out = hit_to_dict(hit, include_diary_timestamp=True)
    assert out["timestamp"] == "1666-09-03"


def test_hit_to_dict_non_diary_timestamp_none() -> None:
    hit = _mk_hit(kg_kind="KGKind.GUTENBERG", name="x")
    out = hit_to_dict(hit, include_diary_timestamp=True)
    assert out["timestamp"] is None


def test_attach_content_by_sqlite_hydrates_matching_nodes() -> None:
    with TemporaryDirectory() as td:
        db_path = Path(td) / "graph.sqlite"
        with sqlite3.connect(db_path) as con:
            con.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, text TEXT)")
            con.execute("INSERT INTO nodes (id, text) VALUES ('n1', 'alpha text')")
            con.execute("INSERT INTO nodes (id, text) VALUES ('n2', 'beta text')")
            con.commit()

        hits = [
            {"kg_name": "gutenberg", "node_id": "n1"},
            {"kg_name": "gutenberg", "node_id": "n2"},
            {"kg_name": "gutenberg", "node_id": "n3"},
        ]
        attach_content_by_sqlite(hits, {"gutenberg": db_path})

        assert hits[0]["content"] == "alpha text"
        assert hits[1]["content"] == "beta text"
        assert hits[2]["content"] == ""


def test_attach_content_by_sqlite_noop_on_missing_db() -> None:
    hits = [{"kg_name": "gutenberg", "node_id": "n1"}]
    attach_content_by_sqlite(hits, {"gutenberg": Path("/no/such/file.sqlite")})
    assert "content" not in hits[0]


def test_attach_content_by_sqlite_noop_on_empty_ids() -> None:
    with TemporaryDirectory() as td:
        db_path = Path(td) / "graph.sqlite"
        with sqlite3.connect(db_path) as con:
            con.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, text TEXT)")
            con.commit()

        hits = [{"kg_name": "gutenberg", "node_id": ""}, {"kg_name": "gutenberg"}]
        attach_content_by_sqlite(hits, {"gutenberg": db_path})
        assert "content" not in hits[0]
        assert "content" not in hits[1]
