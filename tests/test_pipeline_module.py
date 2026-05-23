"""Integration tests for the concrete KGModule pipeline (build → query → pack).

These tests require sentence-transformers and lancedb (the 'semantic' extra).
They are marked with pytest.mark.integration so they can be skipped in lean CI.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from kg_utils.extractor import KGExtractor
from kg_utils.pipeline import KGModule
from kg_utils.specs import BuildStats, EdgeSpec, NodeSpec, QueryResult, SnippetPack

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Minimal domain implementation used across all tests
# ---------------------------------------------------------------------------


class _TextExtractor(KGExtractor):
    """Indexes .txt files in repo_path as function nodes."""

    def node_kinds(self) -> list[str]:
        return ["module", "function"]

    def edge_kinds(self) -> list[str]:
        return ["CONTAINS"]

    def meaningful_node_kinds(self) -> list[str]:
        return ["function"]

    def extract(self) -> Iterator[NodeSpec | EdgeSpec]:
        mod_id = "module:src/corpus.py:corpus"
        yield NodeSpec(
            node_id=mod_id,
            kind="module",
            name="corpus",
            qualname="corpus",
            source_path="src/corpus.py",
        )
        for f in sorted(self.repo_path.glob("*.txt")):
            nid = f"function:src/corpus.py:{f.stem}"
            content = f.read_text(encoding="utf-8").strip()
            yield NodeSpec(
                node_id=nid,
                kind="function",
                name=f.stem,
                qualname=f.stem,
                source_path="src/corpus.py",
                lineno=1,
                end_lineno=len(content.splitlines()) or 1,
                docstring=content,
            )
            yield EdgeSpec(source_id=mod_id, target_id=nid, relation="CONTAINS")


class _TextKG(KGModule):
    _default_dir = ".textkg"

    def make_extractor(self) -> KGExtractor:
        return _TextExtractor(self.repo_root)

    def kind(self) -> str:
        return "text"

    def analyze(self) -> str:
        s = self.stats()
        return f"# TextKG\nnodes={s['total_nodes']}"

    def _kind_priority(self, kind: str) -> int:
        return {"function": 0, "module": 1}.get(kind, 99)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    (root / "src").mkdir()
    (root / "embedder.txt").write_text(
        "The embedder converts text to dense float32 vectors using sentence-transformers.",
        encoding="utf-8",
    )
    (root / "store.txt").write_text(
        "The graph store persists nodes and edges in a SQLite database.",
        encoding="utf-8",
    )
    (root / "pipeline.txt").write_text(
        "The pipeline builds the knowledge graph from extraction through vector indexing.",
        encoding="utf-8",
    )
    (root / "semantic.txt").write_text(
        "Semantic search uses LanceDB to find similar nodes via cosine distance.",
        encoding="utf-8",
    )
    (root / "src" / "corpus.py").write_text("# placeholder source file\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def kg(corpus: Path) -> _TextKG:
    kg = _TextKG(corpus)
    kg.build(wipe=True)
    return kg


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_returns_build_stats(self, corpus: Path) -> None:
        kg = _TextKG(corpus)
        stats = kg.build_graph(wipe=True)
        assert isinstance(stats, BuildStats)

    def test_node_counts_correct(self, corpus: Path) -> None:
        kg = _TextKG(corpus)
        stats = kg.build_graph(wipe=True)
        assert stats.total_nodes == 5  # 1 module + 4 txt files
        assert stats.node_counts["function"] == 4
        assert stats.node_counts["module"] == 1

    def test_edge_counts_correct(self, corpus: Path) -> None:
        kg = _TextKG(corpus)
        stats = kg.build_graph(wipe=True)
        assert stats.total_edges == 4  # one CONTAINS per function

    def test_wipe_clears_previous(self, corpus: Path) -> None:
        kg = _TextKG(corpus)
        kg.build_graph(wipe=True)
        kg.build_graph(wipe=True)
        s = kg.stats()
        assert s["total_nodes"] == 5  # not doubled


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_indexed_rows_equals_meaningful_nodes(self, kg: _TextKG) -> None:
        stats = kg.build_index(wipe=True)
        assert stats.indexed_rows == 4  # only function nodes indexed

    def test_index_dim_correct(self, kg: _TextKG) -> None:
        stats = kg.build_index(wipe=True)
        assert stats.index_dim == 384


# ---------------------------------------------------------------------------
# KGModule.stats / node / callers
# ---------------------------------------------------------------------------


class TestKGModuleConvenience:
    def test_stats_returns_dict(self, kg: _TextKG) -> None:
        s = kg.stats()
        assert "total_nodes" in s
        assert s["total_nodes"] == 5

    def test_node_fetches_by_id(self, kg: _TextKG) -> None:
        n = kg.node("function:src/corpus.py:embedder")
        assert n is not None
        assert n["kind"] == "function"

    def test_node_missing_returns_none(self, kg: _TextKG) -> None:
        assert kg.node("nonexistent:x") is None

    def test_analyze_returns_markdown(self, kg: _TextKG) -> None:
        md = kg.analyze()
        assert "TextKG" in md

    def test_kind(self, kg: _TextKG) -> None:
        assert kg.kind() == "text"


# ---------------------------------------------------------------------------
# KGModule.query
# ---------------------------------------------------------------------------


class TestKGModuleQuery:
    def test_returns_query_result(self, kg: _TextKG) -> None:
        result = kg.query("vector embeddings")
        assert isinstance(result, QueryResult)

    def test_query_returns_nodes(self, kg: _TextKG) -> None:
        result = kg.query("vector embeddings")
        assert len(result.nodes) > 0

    def test_query_seeds_nonzero(self, kg: _TextKG) -> None:
        result = kg.query("graph store SQLite")
        assert result.seeds > 0

    def test_query_embedder_semantic_match(self, kg: _TextKG) -> None:
        result = kg.query("sentence transformer vectors")
        node_names = {n["name"] for n in result.nodes}
        assert "embedder" in node_names

    def test_query_store_semantic_match(self, kg: _TextKG) -> None:
        result = kg.query("SQLite database nodes edges")
        node_names = {n["name"] for n in result.nodes}
        assert "store" in node_names

    def test_query_relevance_attached(self, kg: _TextKG) -> None:
        result = kg.query("pipeline build")
        assert all("relevance" in n for n in result.nodes)

    def test_query_min_score_filters(self, kg: _TextKG) -> None:
        result_all = kg.query("build graph", min_score=0.0)
        result_filtered = kg.query("build graph", min_score=0.99)
        assert len(result_filtered.nodes) <= len(result_all.nodes)

    def test_query_hop0_pure_semantic(self, kg: _TextKG) -> None:
        result = kg.query("embeddings float32", hop=0)
        # hop=0 means no graph expansion — only semantic seeds
        assert result.hop == 0
        assert result.expanded_nodes == result.seeds

    def test_query_rerank_hybrid(self, kg: _TextKG) -> None:
        result = kg.query("pipeline knowledge graph", rerank_mode="hybrid")
        assert len(result.nodes) > 0
        modes = {n["relevance"]["mode"] for n in result.nodes}
        assert "hybrid" in modes

    def test_query_max_nodes_respected(self, kg: _TextKG) -> None:
        result = kg.query("graph", max_nodes=2)
        assert len(result.nodes) <= 2


# ---------------------------------------------------------------------------
# KGModule.pack
# ---------------------------------------------------------------------------


class TestKGModulePack:
    def test_returns_snippet_pack(self, kg: _TextKG) -> None:
        pack = kg.pack("vector store")
        assert isinstance(pack, SnippetPack)

    def test_pack_has_nodes(self, kg: _TextKG) -> None:
        pack = kg.pack("semantic search LanceDB")
        assert len(pack.nodes) > 0

    def test_pack_snippets_have_text(self, kg: _TextKG) -> None:
        pack = kg.pack("embedding pipeline")
        nodes_with_snippets = [n for n in pack.nodes if n.get("snippet")]
        assert len(nodes_with_snippets) > 0

    def test_pack_to_markdown(self, kg: _TextKG) -> None:
        pack = kg.pack("graph nodes")
        md = pack.to_markdown()
        assert "KGModule Snippet Pack" in md

    def test_pack_to_json(self, kg: _TextKG) -> None:
        import json

        pack = kg.pack("graph nodes")
        d = json.loads(pack.to_json())
        assert "nodes" in d
        assert "edges" in d

    def test_pack_max_nodes_respected(self, kg: _TextKG) -> None:
        pack = kg.pack("build index", max_nodes=2)
        assert len(pack.nodes) <= 2

    def test_pack_internal_keys_stripped(self, kg: _TextKG) -> None:
        pack = kg.pack("store pipeline")
        for n in pack.nodes:
            assert not any(k.startswith("_") for k in n)


# ---------------------------------------------------------------------------
# KGModule context manager
# ---------------------------------------------------------------------------


def test_context_manager_closes_store(corpus: Path) -> None:
    with _TextKG(corpus) as kg:
        s = kg.stats()
        assert s["total_nodes"] > 0
    # connection closed — further access should reopen or be None
    assert kg._store is None or kg._store._con is None


# ---------------------------------------------------------------------------
# Lazy property initialisation
# ---------------------------------------------------------------------------


def test_lazy_store_created_on_access(corpus: Path) -> None:
    kg = _TextKG(corpus)
    assert kg._store is None
    _ = kg.store
    assert kg._store is not None


def test_lazy_embedder_created_on_access(corpus: Path) -> None:
    kg = _TextKG(corpus)
    assert kg._embedder is None
    _ = kg.embedder
    assert kg._embedder is not None
