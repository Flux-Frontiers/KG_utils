"""Tests for kg_utils.types."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from kg_utils.types import EdgeSpec, KGExtractor, KGModule, NodeSpec, QueryResult, SnippetPack


# -- Spec dataclasses --------------------------------------------------------


def test_nodespec_defaults() -> None:
    n = NodeSpec(
        node_id="file:src/main.py:main.py",
        kind="file",
        name="main.py",
        qualname="src/main.py",
        source_path="src/main.py",
    )
    assert n.docstring == ""
    assert n.node_id == "file:src/main.py:main.py"


def test_nodespec_with_docstring() -> None:
    n = NodeSpec(
        node_id="func:src/main.py:main",
        kind="function",
        name="main",
        qualname="src/main.py:main",
        source_path="src/main.py",
        docstring="Entry point.",
    )
    assert n.docstring == "Entry point."


def test_edgespec() -> None:
    e = EdgeSpec(source_id="a", target_id="b", relation="CALLS")
    assert e.source_id == "a"
    assert e.relation == "CALLS"


def test_queryresult_defaults() -> None:
    q = QueryResult()
    assert q.nodes == []
    assert q.seeds == 0


def test_snippetpack_defaults() -> None:
    s = SnippetPack(query="test")
    assert s.query == "test"
    assert s.nodes == []
    assert s.snippets == []


# -- KGExtractor -------------------------------------------------------------


class DummyExtractor(KGExtractor):
    def node_kinds(self) -> list[str]:
        return ["file", "directory"]

    def edge_kinds(self) -> list[str]:
        return ["CONTAINS"]

    def extract(self) -> Iterator[NodeSpec | EdgeSpec]:
        yield NodeSpec(
            node_id="file:a.txt:a.txt",
            kind="file",
            name="a.txt",
            qualname="a.txt",
            source_path="a.txt",
            docstring="A text file.",
        )
        yield NodeSpec(
            node_id="file:b.txt:b.txt",
            kind="file",
            name="b.txt",
            qualname="b.txt",
            source_path="b.txt",
        )
        yield EdgeSpec(source_id="directory:.:.", target_id="file:a.txt:a.txt", relation="CONTAINS")


def test_extractor_node_kinds(tmp_path: Path) -> None:
    ext = DummyExtractor(tmp_path)
    assert ext.node_kinds() == ["file", "directory"]


def test_extractor_meaningful_defaults(tmp_path: Path) -> None:
    ext = DummyExtractor(tmp_path)
    assert ext.meaningful_node_kinds() == ext.node_kinds()


def test_extractor_extract(tmp_path: Path) -> None:
    items = list(DummyExtractor(tmp_path).extract())
    nodes = [i for i in items if isinstance(i, NodeSpec)]
    edges = [i for i in items if isinstance(i, EdgeSpec)]
    assert len(nodes) == 2
    assert len(edges) == 1


def test_extractor_coverage(tmp_path: Path) -> None:
    ext = DummyExtractor(tmp_path)
    nodes = [i for i in ext.extract() if isinstance(i, NodeSpec)]
    assert ext.coverage_metric(nodes) == pytest.approx(0.5)


def test_base_extractor_raises() -> None:
    base = KGExtractor(Path("/tmp"))
    with pytest.raises(NotImplementedError):
        base.node_kinds()
    with pytest.raises(NotImplementedError):
        list(base.extract())


# -- KGModule ----------------------------------------------------------------


def test_base_module_raises() -> None:
    m = KGModule(repo_root=Path("/tmp"))
    with pytest.raises(NotImplementedError):
        m.make_extractor()
    with pytest.raises(NotImplementedError):
        m.kind()
    with pytest.raises(NotImplementedError):
        m.build()
    with pytest.raises(NotImplementedError):
        m.query("test")
    with pytest.raises(NotImplementedError):
        m.stats()
    with pytest.raises(NotImplementedError):
        m.pack("test")
    with pytest.raises(NotImplementedError):
        m.analyze()


def test_module_init_defaults() -> None:
    m = KGModule(repo_root=Path("/tmp"))
    assert m.db_path is None
    assert m.lancedb_dir is None
    assert m.config == {}


def test_module_init_with_all_params(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    lance = tmp_path / "lance"
    cfg = {"key": "value"}
    m = KGModule(repo_root=tmp_path, db_path=db, lancedb_dir=lance, config=cfg)
    assert m.repo_root == tmp_path
    assert m.db_path == db
    assert m.lancedb_dir == lance
    assert m.config == {"key": "value"}


# -- KGExtractor additional cases --------------------------------------------


def test_base_extractor_edge_kinds_raises() -> None:
    base = KGExtractor(Path("/tmp"))
    with pytest.raises(NotImplementedError):
        base.edge_kinds()


def test_extractor_meaningful_node_kinds_override(tmp_path: Path) -> None:
    class _Selective(DummyExtractor):
        def meaningful_node_kinds(self) -> list[str]:
            return ["file"]  # exclude "directory"

    ext = _Selective(tmp_path)
    assert ext.meaningful_node_kinds() == ["file"]


def test_extractor_coverage_empty_nodes(tmp_path: Path) -> None:
    ext = DummyExtractor(tmp_path)
    assert ext.coverage_metric([]) == 0.0


def test_extractor_coverage_all_covered(tmp_path: Path) -> None:
    ext = DummyExtractor(tmp_path)
    nodes = [
        NodeSpec("a", "file", "a", "a", "a", docstring="doc a"),
        NodeSpec("b", "file", "b", "b", "b", docstring="doc b"),
    ]
    assert ext.coverage_metric(nodes) == pytest.approx(1.0)


def test_extractor_coverage_only_non_meaningful_kinds(tmp_path: Path) -> None:
    class _SelectiveExt(DummyExtractor):
        def meaningful_node_kinds(self) -> list[str]:
            return ["function"]  # none of the nodes below are functions

    ext = _SelectiveExt(tmp_path)
    nodes = [
        NodeSpec("a", "file", "a", "a", "a"),
        NodeSpec("b", "directory", "b", "b", "b"),
    ]
    # No meaningful nodes → coverage is 0.0
    assert ext.coverage_metric(nodes) == 0.0


def test_extractor_config_default(tmp_path: Path) -> None:
    ext = KGExtractor.__new__(KGExtractor)
    KGExtractor.__init__(ext, tmp_path)
    assert ext.config == {}


def test_extractor_config_passed(tmp_path: Path) -> None:
    ext = KGExtractor.__new__(KGExtractor)
    KGExtractor.__init__(ext, tmp_path, config={"max_depth": 3})
    assert ext.config == {"max_depth": 3}


# -- QueryResult and SnippetPack with actual data ----------------------------


def test_queryresult_with_data() -> None:
    q = QueryResult(
        nodes=[{"id": "a"}],
        edges=[{"rel": "CALLS"}],
        seeds=1,
        expanded_nodes=3,
        returned_nodes=1,
        hop=2,
        rels=["CALLS", "CONTAINS"],
    )
    assert len(q.nodes) == 1
    assert q.seeds == 1
    assert q.hop == 2
    assert "CALLS" in q.rels


def test_snippetpack_with_data() -> None:
    s = SnippetPack(
        query="find embedders",
        seeds=2,
        expanded_nodes=5,
        returned_nodes=2,
        hop=1,
        rels=["CONTAINS"],
        model="BAAI/bge-small-en-v1.5",
        nodes=[{"id": "n1"}],
        edges=[{"rel": "CONTAINS"}],
        snippets=["def foo(): pass"],
    )
    assert s.query == "find embedders"
    assert s.model == "BAAI/bge-small-en-v1.5"
    assert len(s.snippets) == 1
    assert s.returned_nodes == 2
