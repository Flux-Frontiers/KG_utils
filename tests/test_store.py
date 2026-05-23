"""Tests for kg_utils.store — GraphStore SQLite persistence layer."""

from __future__ import annotations

import json
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
        name=name or nid.split(":")[-1],
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
