"""Tests for KGModule.neighbors() and GraphStore.edges_touching().

Graph-only: every test builds with ``build_graph()`` and never touches the
vector index, which is the point -- ``neighbors`` must work on a KG whose
index was never built, without loading the embedding model.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import kg_utils.pipeline as pipeline_mod
from kg_utils.extractor import KGExtractor
from kg_utils.pipeline import KGModule
from kg_utils.specs import EdgeSpec, NodeSpec


class _Extractor(KGExtractor):
    """mod CONTAINS a, b; a CALLS b; c CALLS a; a CALLS a stub that is never defined."""

    def node_kinds(self) -> list[str]:
        return ["module", "function"]

    def edge_kinds(self) -> list[str]:
        return ["CONTAINS", "CALLS"]

    def meaningful_node_kinds(self) -> list[str]:
        return ["function"]

    def extract(self) -> Iterator[NodeSpec | EdgeSpec]:
        yield NodeSpec(node_id="mod", kind="module", name="mod", qualname="mod", source_path="m.py")
        for n in ("a", "b", "c"):
            yield NodeSpec(
                node_id=n, kind="function", name=n, qualname=n, source_path="m.py", lineno=1
            )
        yield EdgeSpec(source_id="mod", target_id="a", relation="CONTAINS")
        yield EdgeSpec(source_id="mod", target_id="b", relation="CONTAINS")
        yield EdgeSpec(source_id="a", target_id="b", relation="CALLS")
        yield EdgeSpec(source_id="c", target_id="a", relation="CALLS")
        yield EdgeSpec(source_id="a", target_id="ghost", relation="CALLS")


class _KG(KGModule):
    _default_dir = ".nkg"

    def make_extractor(self) -> KGExtractor:
        return _Extractor(self.repo_root)

    def kind(self) -> str:
        return "test"

    def analyze(self) -> str:
        return "# test"


@pytest.fixture
def kg(tmp_path: Path) -> Iterator[_KG]:
    kg = _KG(tmp_path)
    kg.build_graph(wipe=True)
    yield kg
    kg.close()


def _pairs(result: dict) -> list[tuple[str, str, str]]:
    return [(e["src"], e["rel"], e["dst"]) for e in result["edges"]]


def test_both_directions_by_default(kg: _KG) -> None:
    result = kg.neighbors("a")
    assert result is not None
    assert result["node"]["id"] == "a"
    assert _pairs(result) == [
        ("a", "CALLS", "b"),
        ("a", "CALLS", "ghost"),
        ("c", "CALLS", "a"),
        ("mod", "CONTAINS", "a"),
    ]


def test_far_ends_are_distinct_existing_nodes_in_edge_order(kg: _KG) -> None:
    result = kg.neighbors("a")
    assert result is not None
    # "ghost" is an edge endpoint with no node row: the edge is kept, the node is not invented.
    assert [n["id"] for n in result["nodes"]] == ["b", "c", "mod"]


def test_direction_out_and_in(kg: _KG) -> None:
    out = kg.neighbors("a", direction="out")
    into = kg.neighbors("a", direction="in")
    assert out is not None and into is not None
    assert _pairs(out) == [("a", "CALLS", "b"), ("a", "CALLS", "ghost")]
    assert _pairs(into) == [("c", "CALLS", "a"), ("mod", "CONTAINS", "a")]


def test_rels_filter(kg: _KG) -> None:
    result = kg.neighbors("a", rels=["CONTAINS"])
    assert result is not None
    assert _pairs(result) == [("mod", "CONTAINS", "a")]


def test_limit_caps_edges_deterministically(kg: _KG) -> None:
    result = kg.neighbors("a", limit=2)
    assert result is not None
    assert _pairs(result) == [("a", "CALLS", "b"), ("a", "CALLS", "ghost")]


def test_unknown_node_is_none_like_node(kg: _KG) -> None:
    assert kg.node("nope") is None
    assert kg.neighbors("nope") is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"node_id": ""}, "node_id"),
        ({"node_id": "   "}, "node_id"),
        ({"node_id": "a", "direction": "sideways"}, "direction"),
        ({"node_id": "a", "limit": 0}, "limit"),
        ({"node_id": "a", "limit": 501}, "limit"),
        ({"node_id": "a", "limit": "8"}, "limit"),
        ({"node_id": "a", "limit": True}, "limit"),
    ],
)
def test_bad_arguments_name_the_parameter(kg: _KG, kwargs: dict, message: str) -> None:
    node_id = kwargs.pop("node_id")
    with pytest.raises(ValueError, match=message):
        kg.neighbors(node_id, **kwargs)


def test_limit_bound_follows_the_class_attribute(tmp_path: Path) -> None:
    class Tight(_KG):
        max_max_nodes = 2

    kg = Tight(tmp_path)
    kg.build_graph(wipe=True)
    try:
        assert kg.neighbors("a", limit=2) is not None
        with pytest.raises(ValueError, match="limit"):
            kg.neighbors("a", limit=3)
    finally:
        kg.close()


def test_never_loads_the_embedding_model(kg: _KG, monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("neighbors() loaded the embedding model")

    monkeypatch.setattr(pipeline_mod, "SentenceTransformerEmbedder", _refuse)
    assert kg.neighbors("a") is not None


def test_store_rejects_unknown_direction(kg: _KG) -> None:
    with pytest.raises(ValueError, match="direction"):
        kg.store.edges_touching("a", direction="up")
