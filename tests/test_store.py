"""Tests for kg_utils.store — GraphStore SQLite persistence layer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from kg_utils.specs import EdgeSpec, NodeSpec
from kg_utils.store import GraphStore, ProvMeta, _module_to_dotted_variants

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _node(nid: str, kind: str = "function", name: str | None = None, **kw) -> NodeSpec:
    return NodeSpec(
        node_id=nid,
        kind=kind,
        name=name or nid.rsplit(":", maxsplit=1)[-1],
        qualname=kw.pop("qualname", nid),
        source_path=kw.pop("source_path", "src/mod.py"),
        **kw,
    )


def _edge(src: str, dst: str, rel: str = "CALLS") -> EdgeSpec:
    return EdgeSpec(source_id=src, target_id=dst, relation=rel)


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    s = GraphStore(tmp_path / "graph.sqlite")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# _module_to_dotted_variants
# ---------------------------------------------------------------------------


def test_dotted_simple() -> None:
    assert _module_to_dotted_variants("src/pkg/mod.py") == ("src.pkg.mod", "pkg.mod")


def test_dotted_no_src_prefix() -> None:
    assert _module_to_dotted_variants("pkg/mod.py") == ("pkg.mod",)


def test_dotted_none() -> None:
    assert _module_to_dotted_variants(None) == ()


def test_dotted_empty() -> None:
    assert _module_to_dotted_variants("") == ()


# ---------------------------------------------------------------------------
# ProvMeta
# ---------------------------------------------------------------------------


def test_provmeta_repr() -> None:
    p = ProvMeta(best_hop=2, via_seed="seed:x")
    assert "best_hop=2" in repr(p)
    assert "seed:x" in repr(p)


# ---------------------------------------------------------------------------
# GraphStore — connection and context manager
# ---------------------------------------------------------------------------


def test_store_creates_db_on_first_access(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "graph.sqlite"
    s = GraphStore(db)
    _ = s.con  # triggers creation
    assert db.exists()
    s.close()


def test_store_context_manager(tmp_path: Path) -> None:
    with GraphStore(tmp_path / "g.sqlite") as s:
        s.write([_node("n:a")], [])
    # no exception; connection closed cleanly


def test_store_close_idempotent(store: GraphStore) -> None:
    store.close()
    store.close()  # should not raise


# ---------------------------------------------------------------------------
# Write and basic reads
# ---------------------------------------------------------------------------


def test_write_nodes_and_read_back(store: GraphStore) -> None:
    nodes = [_node("func:src/a.py:foo", "function", "foo", docstring="does foo")]
    store.write(nodes, [])
    n = store.node("func:src/a.py:foo")
    assert n is not None
    assert n["kind"] == "function"
    assert n["name"] == "foo"
    assert n["docstring"] == "does foo"


def test_write_edges_and_edges_within(store: GraphStore) -> None:
    nodes = [_node("a"), _node("b")]
    edges = [_edge("a", "b")]
    store.write(nodes, edges)
    result = store.edges_within({"a", "b"})
    assert len(result) == 1
    assert result[0]["src"] == "a"
    assert result[0]["dst"] == "b"
    assert result[0]["rel"] == "CALLS"


def test_edges_within_empty_set(store: GraphStore) -> None:
    assert store.edges_within(set()) == []


def test_edges_within_excludes_external(store: GraphStore) -> None:
    nodes = [_node("a"), _node("b"), _node("c")]
    store.write(nodes, [_edge("a", "b"), _edge("b", "c")])
    result = store.edges_within({"a", "b"})
    assert len(result) == 1  # a→b only, not b→c (c excluded)


def test_write_wipe(store: GraphStore) -> None:
    store.write([_node("x")], [])
    store.write([_node("y")], [], wipe=True)
    assert store.node("x") is None
    assert store.node("y") is not None


def test_upsert_node_updates_existing(store: GraphStore) -> None:
    store.write([_node("n:a", docstring="v1")], [])
    store.write([_node("n:a", docstring="v2")], [])
    n = store.node("n:a")
    assert n["docstring"] == "v2"


def test_upsert_edge_with_metadata(store: GraphStore) -> None:
    nodes = [_node("a"), _node("b")]
    edge = EdgeSpec(source_id="a", target_id="b", relation="CALLS", metadata={"lineno": 42})
    store.write(nodes, [edge])
    rows = store.edges_within({"a", "b"})
    assert len(rows) == 1
    ev = json.loads(rows[0]["evidence"])
    assert ev["lineno"] == 42


def test_node_missing_returns_none(store: GraphStore) -> None:
    assert store.node("nonexistent") is None


def test_clear(store: GraphStore) -> None:
    store.write([_node("a"), _node("b")], [_edge("a", "b")])
    store.clear()
    s = store.stats()
    assert s["total_nodes"] == 0
    assert s["total_edges"] == 0


# ---------------------------------------------------------------------------
# query_nodes
# ---------------------------------------------------------------------------


def test_query_nodes_all(store: GraphStore) -> None:
    store.write([_node("a", "function"), _node("b", "class"), _node("c", "module")], [])
    assert len(store.query_nodes()) == 3


def test_query_nodes_by_kind(store: GraphStore) -> None:
    store.write([_node("a", "function"), _node("b", "class"), _node("c", "function")], [])
    fns = store.query_nodes(kinds=["function"])
    assert len(fns) == 2
    assert all(n["kind"] == "function" for n in fns)


def test_query_nodes_by_module(store: GraphStore) -> None:
    store.write(
        [
            _node("a", source_path="src/a.py"),
            _node("b", source_path="src/b.py"),
        ],
        [],
    )
    result = store.query_nodes(module="src/a.py")
    assert len(result) == 1
    assert result[0]["id"] == "a"


# ---------------------------------------------------------------------------
# expand — BFS graph traversal
# ---------------------------------------------------------------------------


def test_expand_hop0_returns_seeds(store: GraphStore) -> None:
    store.write([_node("a"), _node("b"), _node("c")], [_edge("a", "b"), _edge("b", "c")])
    meta = store.expand({"a"}, hop=0)
    assert set(meta.keys()) == {"a"}


def test_expand_hop1(store: GraphStore) -> None:
    store.write([_node("a"), _node("b"), _node("c")], [_edge("a", "b"), _edge("b", "c")])
    meta = store.expand({"a"}, hop=1)
    assert "a" in meta
    assert "b" in meta
    assert "c" not in meta


def test_expand_hop2(store: GraphStore) -> None:
    store.write([_node("a"), _node("b"), _node("c")], [_edge("a", "b"), _edge("b", "c")])
    meta = store.expand({"a"}, hop=2)
    assert {"a", "b", "c"} == set(meta.keys())


def test_expand_provenance(store: GraphStore) -> None:
    store.write([_node("a"), _node("b")], [_edge("a", "b")])
    meta = store.expand({"a"}, hop=1)
    assert meta["a"].best_hop == 0
    assert meta["b"].best_hop == 1
    assert meta["b"].via_seed == "a"


def test_expand_respects_rel_filter(store: GraphStore) -> None:
    store.write(
        [_node("a"), _node("b"), _node("c")],
        [_edge("a", "b", "CALLS"), _edge("a", "c", "IMPORTS")],
    )
    meta = store.expand({"a"}, hop=1, rels=("CALLS",))
    assert "b" in meta
    assert "c" not in meta


def test_expand_empty_seeds(store: GraphStore) -> None:
    store.write([_node("a")], [])
    meta = store.expand(set(), hop=2)
    assert meta == {}


# ---------------------------------------------------------------------------
# resolve_symbols
# ---------------------------------------------------------------------------


def test_resolve_symbols_exact_qualname(store: GraphStore) -> None:
    store.write(
        [
            _node("func:src/a.py:foo", "function", "foo", qualname="foo"),
            _node("sym:foo", "symbol", "foo", qualname="pkg.mod.foo"),
        ],
        [],
    )
    count = store.resolve_symbols()
    edges = store.edges_within({"func:src/a.py:foo", "sym:foo"})
    rels = {e["rel"] for e in edges}
    assert count > 0
    assert "RESOLVES_TO" in rels


def test_resolve_symbols_idempotent(store: GraphStore) -> None:
    store.write(
        [_node("f:a", "function", "foo"), _node("sym:foo", "symbol", "foo")],
        [],
    )
    c1 = store.resolve_symbols()
    c2 = store.resolve_symbols()
    assert c1 == c2  # second call finds same edges already present


def test_resolve_symbols_receiver_typed_scopes_to_class(store: GraphStore) -> None:
    # Two classes define the same method name; only the one matching the
    # stub's receiver_class metadata should get a RESOLVES_TO edge.
    store.write(
        [
            _node("m:a.py:Log.render", "method", "render", qualname="Log.render"),
            _node("m:a.py:Plotter.render", "method", "render", qualname="Plotter.render"),
            _node(
                "sym:plotter.render",
                "symbol",
                "render",
                qualname="plotter.render",
                metadata={"receiver_class": "Log"},
            ),
        ],
        [],
    )
    store.resolve_symbols()
    edges = store.edges_within({"sym:plotter.render", "m:a.py:Log.render", "m:a.py:Plotter.render"})
    resolved = [e for e in edges if e["rel"] == "RESOLVES_TO"]
    assert len(resolved) == 1
    assert resolved[0]["dst"] == "m:a.py:Log.render"
    evidence = json.loads(resolved[0]["evidence"])
    assert evidence["resolution_mode"] == "receiver_typed"


def test_resolve_symbols_receiver_typed_no_match_stays_unresolved(
    store: GraphStore,
) -> None:
    # receiver_class names a class with no matching method in the graph —
    # must not fall back to a same-named method on an unrelated class.
    store.write(
        [
            _node("m:a.py:Other.render", "method", "render", qualname="Other.render"),
            _node(
                "sym:plotter.render",
                "symbol",
                "render",
                qualname="plotter.render",
                metadata={"receiver_class": "Log"},
            ),
        ],
        [],
    )
    count = store.resolve_symbols()
    assert count == 0


# ---------------------------------------------------------------------------
# callers_of and edges_from
# ---------------------------------------------------------------------------


def test_callers_of_direct(store: GraphStore) -> None:
    store.write([_node("caller"), _node("callee")], [_edge("caller", "callee", "CALLS")])
    callers = store.callers_of("callee", rel="CALLS")
    assert len(callers) == 1
    assert callers[0]["id"] == "caller"


def test_callers_of_no_match(store: GraphStore) -> None:
    store.write([_node("a"), _node("b")], [])
    assert store.callers_of("b") == []


def test_edges_from(store: GraphStore) -> None:
    store.write([_node("a"), _node("b"), _node("c")], [_edge("a", "b"), _edge("a", "c")])
    rows = store.edges_from("a", rel="CALLS")
    assert len(rows) == 2
    dsts = {r["dst"] for r in rows}
    assert dsts == {"b", "c"}


def test_edges_from_limit(store: GraphStore) -> None:
    nodes = [_node(f"n{i}") for i in range(5)]
    edges = [_edge("n0", f"n{i}") for i in range(1, 5)]
    store.write(nodes, edges)
    rows = store.edges_from("n0", rel="CALLS", limit=2)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_empty(store: GraphStore) -> None:
    s = store.stats()
    assert s["total_nodes"] == 0
    assert s["total_edges"] == 0


def test_stats_counts(store: GraphStore) -> None:
    store.write(
        [
            _node("a", "function", docstring="doc"),
            _node("b", "method", docstring=""),
            _node("c", "class"),
        ],
        [_edge("a", "b")],
    )
    s = store.stats()
    assert s["total_nodes"] == 3
    assert s["total_edges"] == 1
    assert s["node_counts"]["function"] == 1
    assert s["node_counts"]["class"] == 1
    assert s["edge_counts"]["CALLS"] == 1
    # docstring coverage: 1 fn/method has doc out of 2
    assert s["docstring_coverage"] == pytest.approx(0.5)


def test_store_repr(store: GraphStore) -> None:
    assert "GraphStore" in repr(store)


# ---------------------------------------------------------------------------
# Node metadata persistence + additive migration
# ---------------------------------------------------------------------------


class TestNodeMetadata:
    """NodeSpec.metadata must survive a write/read round trip.

    It did not until 0.18.0: the spec carried the field, the schema had no
    column for it, and it was dropped silently on write. Any consumer reading
    node metadata back — the temporal contract most of all — got nothing.
    """

    def _node(self, node_id="n:1", **kw):
        return NodeSpec(
            node_id=node_id,
            kind="entry",
            name="n",
            qualname="n",
            source_path="p.md",
            **kw,
        )

    def test_metadata_round_trips(self, tmp_path):
        store = GraphStore(tmp_path / "g.sqlite")
        store.write([self._node(metadata={"occurred_start": "2026-04-15"})], [])
        node = store.node("n:1")
        assert node["metadata"] == {"occurred_start": "2026-04-15"}
        store.close()

    def test_absent_metadata_reads_as_empty_dict(self, tmp_path):
        store = GraphStore(tmp_path / "g.sqlite")
        store.write([self._node()], [])
        assert store.node("n:1")["metadata"] == {}
        store.close()

    def test_metadata_survives_list_nodes(self, tmp_path):
        store = GraphStore(tmp_path / "g.sqlite")
        store.write([self._node(metadata={"k": "v"})], [])
        nodes = store.query_nodes()
        assert nodes[0]["metadata"] == {"k": "v"}
        store.close()

    def test_upsert_updates_metadata(self, tmp_path):
        store = GraphStore(tmp_path / "g.sqlite")
        store.write([self._node(metadata={"occurred_start": "2026-04-15"})], [])
        store.write([self._node(metadata={"occurred_start": "2026-05-01"})], [])
        assert store.node("n:1")["metadata"]["occurred_start"] == "2026-05-01"
        store.close()

    def test_non_ascii_metadata_preserved(self, tmp_path):
        store = GraphStore(tmp_path / "g.sqlite")
        store.write([self._node(metadata={"title": "Café Ontwerp — 1876"})], [])
        assert store.node("n:1")["metadata"]["title"] == "Café Ontwerp — 1876"
        store.close()

    def test_corrupt_metadata_does_not_break_the_node(self, tmp_path):
        """Extension data is not worth making a node unreadable over."""
        db = tmp_path / "g.sqlite"
        store = GraphStore(db)
        store.write([self._node()], [])
        store.con.execute("UPDATE nodes SET metadata = ? WHERE id = ?", ("{not json", "n:1"))
        store.con.commit()
        node = store.node("n:1")
        assert node["metadata"] == {}
        assert node["name"] == "n"
        store.close()

    def test_non_object_metadata_reads_as_empty(self, tmp_path):
        db = tmp_path / "g.sqlite"
        store = GraphStore(db)
        store.write([self._node()], [])
        store.con.execute("UPDATE nodes SET metadata = ? WHERE id = ?", ("[1,2,3]", "n:1"))
        store.con.commit()
        assert store.node("n:1")["metadata"] == {}
        store.close()


class TestMetadataMigration:
    """An existing database predating the column must still open and read.

    CREATE TABLE IF NOT EXISTS is a no-op against an old database, so without
    an explicit ALTER every KG built before 0.18.0 would raise
    "no such column: metadata" on its next query — before any rebuild.
    """

    def _legacy_db(self, path):
        """Build a database with the pre-0.18.0 nodes schema."""
        con = sqlite3.connect(str(path))
        con.executescript(
            """
            CREATE TABLE nodes (
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
              qualname TEXT, module_path TEXT, lineno INTEGER,
              end_lineno INTEGER, docstring TEXT
            );
            CREATE TABLE edges (
              src TEXT NOT NULL, rel TEXT NOT NULL, dst TEXT NOT NULL,
              evidence TEXT, PRIMARY KEY (src, rel, dst)
            );
            """
        )
        con.execute(
            "INSERT INTO nodes (id, kind, name, qualname, module_path) VALUES (?,?,?,?,?)",
            ("old:1", "entry", "old", "old", "p.md"),
        )
        con.commit()
        con.close()

    def test_legacy_db_is_migrated_on_open(self, tmp_path):
        db = tmp_path / "legacy.sqlite"
        self._legacy_db(db)
        store = GraphStore(db)
        node = store.node("old:1")
        assert node is not None
        assert node["name"] == "old"
        assert node["metadata"] == {}
        store.close()

    def test_legacy_db_accepts_new_writes_after_migration(self, tmp_path):
        db = tmp_path / "legacy.sqlite"
        self._legacy_db(db)
        store = GraphStore(db)
        store.write(
            [
                NodeSpec(
                    node_id="new:1",
                    kind="entry",
                    name="new",
                    qualname="new",
                    source_path="p.md",
                    metadata={"occurred_start": "2026-04-15"},
                )
            ],
            [],
        )
        assert store.node("new:1")["metadata"]["occurred_start"] == "2026-04-15"
        assert store.node("old:1")["metadata"] == {}
        store.close()

    def test_migration_is_idempotent(self, tmp_path):
        db = tmp_path / "legacy.sqlite"
        self._legacy_db(db)
        for _ in range(3):
            store = GraphStore(db)
            assert store.node("old:1") is not None
            store.close()
