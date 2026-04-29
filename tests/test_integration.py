"""Integration tests — cross-module behaviour for kg_utils.

These tests exercise multiple components working together rather than
isolating individual units.  They run with real file I/O, real subprocess
calls, and (for the embedder group) a real SentenceTransformer model.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kg_utils.embed import DEFAULT_MODEL, Embedder as EmbedderProtocol, resolve_model_path
from kg_utils.embedder import (
    Embedder,
    SentenceTransformerEmbedder,
    load_sentence_transformer,
    wrap_embedder,
)
from kg_utils.snapshots import SnapshotManager
from kg_utils.snapshots.models import Snapshot
from kg_utils.types import EdgeSpec, KGExtractor, KGModule, NodeSpec

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_model() -> Any:
    return load_sentence_transformer(DEFAULT_MODEL)


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(DEFAULT_MODEL)


@pytest.fixture
def mgr(tmp_path: Path) -> SnapshotManager:
    return SnapshotManager(tmp_path / "snapshots", package_name="kg-utils")


# ---------------------------------------------------------------------------
# embed ↔ embedder: protocol compliance
# ---------------------------------------------------------------------------


class TestEmbedderProtocolCompliance:
    """SentenceTransformerEmbedder and wrap_embedder both satisfy the
    Embedder protocol defined in kg_utils.embed."""

    def test_sentence_transformer_embedder_satisfies_protocol(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        assert isinstance(real_embedder, EmbedderProtocol)

    def test_wrapped_model_satisfies_protocol(self, real_model: Any) -> None:
        wrapped = wrap_embedder(real_model, DEFAULT_MODEL)
        assert isinstance(wrapped, EmbedderProtocol)

    def test_embedded_base_satisfies_protocol(self) -> None:
        class _Stub(Embedder):
            dim = 4

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 4 for _ in texts]

        assert isinstance(_Stub(), EmbedderProtocol)


# ---------------------------------------------------------------------------
# embed ↔ embedder: resolve_model_path feeds load_sentence_transformer
# ---------------------------------------------------------------------------


class TestPathResolutionToModelLoad:
    """resolve_model_path() produces a path that load_sentence_transformer
    actually uses to find the cached model."""

    def test_resolved_path_exists_for_default_model(self) -> None:
        path = resolve_model_path(DEFAULT_MODEL)
        assert path.exists(), f"Expected cached model at {path}"

    def test_loaded_model_dimension_matches_protocol_expectation(self, real_model: Any) -> None:
        dim_fn = getattr(real_model, "get_embedding_dimension", None) or getattr(
            real_model, "get_sentence_embedding_dimension", None
        )
        assert dim_fn is not None
        assert dim_fn() == 384

    def test_embedder_dim_matches_model_dim(
        self, real_model: Any, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        dim_fn = getattr(real_model, "get_embedding_dimension", None) or getattr(
            real_model, "get_sentence_embedding_dimension", None
        )
        assert real_embedder.dim == dim_fn()


# ---------------------------------------------------------------------------
# embed ↔ embedder: wrap_embedder ↔ SentenceTransformerEmbedder output parity
# ---------------------------------------------------------------------------


class TestWrapEmbedderParity:
    """wrap_embedder and SentenceTransformerEmbedder must produce identical
    vectors for the same input when sharing the same underlying model."""

    def test_embed_query_identical(
        self, real_model: Any, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        wrapped = wrap_embedder(real_model, DEFAULT_MODEL)
        text = "The Great Fire of London, 1666."
        assert wrapped.embed_query(text) == real_embedder.embed_query(text)

    def test_embed_texts_identical(
        self, real_model: Any, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        wrapped = wrap_embedder(real_model, DEFAULT_MODEL)
        texts = ["Pepys at the theatre.", "The King walks in the park."]
        assert wrapped.embed_texts(texts) == real_embedder.embed_texts(texts)

    def test_vectors_are_normalised(self, real_embedder: SentenceTransformerEmbedder) -> None:
        for text in ["short", "a much longer sentence about history and science"]:
            vec = real_embedder.embed_query(text)
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 1e-4, f"vector not unit-normalised for: {text!r}"


# ---------------------------------------------------------------------------
# Snapshot lifecycle: capture → save → load → diff → prune
# ---------------------------------------------------------------------------


class TestSnapshotLifecycle:
    """Full snapshot workflow exercised end-to-end with real file I/O."""

    def _snap(self, mgr: SnapshotManager, key: str, nodes: int, edges: int) -> Snapshot:
        return mgr.capture(
            version="1.0.0",
            branch="main",
            graph_stats_dict={"total_nodes": nodes, "total_edges": edges},
            tree_hash=key,
        )

    def test_save_then_load_round_trip(self, mgr: SnapshotManager) -> None:
        snap = self._snap(mgr, "rt-hash", 42, 18)
        path = mgr.save_snapshot(snap)
        assert path is not None and path.exists()

        loaded = mgr.load_snapshot("rt-hash")
        assert loaded is not None
        assert loaded.metrics["total_nodes"] == 42
        assert loaded.metrics["total_edges"] == 18

    def test_delta_chain_vs_previous_and_baseline(self, mgr: SnapshotManager) -> None:
        s1 = self._snap(mgr, "chain-1", 10, 5)
        mgr.save_snapshot(s1, force=True)

        s2 = self._snap(mgr, "chain-2", 20, 10)
        mgr.save_snapshot(s2, force=True)

        s3 = self._snap(mgr, "chain-3", 30, 15)
        mgr.save_snapshot(s3, force=True)

        loaded3 = mgr.load_snapshot("chain-3")
        assert loaded3 is not None
        # vs_previous: chain-3 minus chain-2
        assert loaded3.vs_previous is not None
        assert loaded3.vs_previous["nodes"] == 10
        # vs_baseline: chain-3 minus chain-1
        assert loaded3.vs_baseline is not None
        assert loaded3.vs_baseline["nodes"] == 20

    def test_list_returns_reverse_chronological_order(self, mgr: SnapshotManager) -> None:
        for i, key in enumerate(["ord-a", "ord-b", "ord-c"], start=1):
            snap = self._snap(mgr, key, i * 5, i * 2)
            mgr.save_snapshot(snap, force=True)

        listed = mgr.list_snapshots()
        timestamps = [s["timestamp"] for s in listed]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_diff_reports_correct_deltas(self, mgr: SnapshotManager) -> None:
        s_a = self._snap(mgr, "diff-a", 10, 4)
        s_b = self._snap(mgr, "diff-b", 25, 12)
        mgr.save_snapshot(s_a, force=True)
        mgr.save_snapshot(s_b, force=True)

        result = mgr.diff_snapshots("diff-a", "diff-b")
        assert "error" not in result
        assert result["delta"]["nodes"] == 15
        assert result["delta"]["edges"] == 8

    def test_prune_removes_only_interior_duplicates(self, mgr: SnapshotManager) -> None:
        for key, n in [("keep-1", 10), ("dup-mid", 10), ("keep-last", 20)]:
            snap = self._snap(mgr, key, n, 5)
            mgr.save_snapshot(snap, force=True)

        result = mgr.prune_snapshots(dry_run=False)
        assert result.dry_run is False
        assert "dup-mid" in result.removed

        remaining_keys = {s["key"] for s in mgr.list_snapshots()}
        assert "keep-1" in remaining_keys
        assert "keep-last" in remaining_keys
        assert "dup-mid" not in remaining_keys

    def test_load_latest_tracks_most_recent_save(self, mgr: SnapshotManager) -> None:
        mgr.save_snapshot(self._snap(mgr, "early", 5, 2), force=True)
        mgr.save_snapshot(self._snap(mgr, "later", 15, 7), force=True)

        latest = mgr.load_snapshot("latest")
        assert latest is not None
        assert latest.key == "later"


# ---------------------------------------------------------------------------
# Snapshot Manager subclassing — domain-specific delta fields
# ---------------------------------------------------------------------------


class TestSnapshotManagerSubclass:
    """Verifies that _compute_delta and _compute_delta_from_metrics can be
    overridden to add domain-specific delta fields."""

    def test_custom_delta_fields_propagate_to_snapshot(self, tmp_path: Path) -> None:
        class _DocMgr(SnapshotManager):
            def _compute_delta_from_metrics(self, new_m: dict, old_m: dict) -> dict:
                base = super()._compute_delta_from_metrics(new_m, old_m)
                base["coverage_delta"] = round(
                    new_m.get("coverage", 0.0) - old_m.get("coverage", 0.0), 4
                )
                return base

        mgr = _DocMgr(tmp_path / "snaps", package_name="doc-kg")

        s1 = mgr.capture(
            version="1.0",
            branch="main",
            graph_stats_dict={"total_nodes": 10, "total_edges": 5, "coverage": 0.50},
            tree_hash="doc-s1",
        )
        mgr.save_snapshot(s1, force=True)

        s2 = mgr.capture(
            version="1.0",
            branch="main",
            graph_stats_dict={"total_nodes": 20, "total_edges": 10, "coverage": 0.75},
            tree_hash="doc-s2",
        )
        mgr.save_snapshot(s2, force=True)

        # Custom fields appear in list_snapshots (which calls _compute_delta_from_metrics).
        # load_snapshot backfills only nodes/edges from the manifest.
        listed = mgr.list_snapshots()
        latest = next(s for s in listed if s["key"] == "doc-s2")
        delta = latest["deltas"]["vs_previous"]
        assert delta["nodes"] == 10
        assert abs(delta["coverage_delta"] - 0.25) < 1e-6


# ---------------------------------------------------------------------------
# KGExtractor + KGModule concrete subclass wired together
# ---------------------------------------------------------------------------


class _FileTreeExtractor(KGExtractor):
    """Minimal extractor that indexes .txt files under repo_path."""

    def node_kinds(self) -> list[str]:
        return ["directory", "file"]

    def edge_kinds(self) -> list[str]:
        return ["CONTAINS"]

    def meaningful_node_kinds(self) -> list[str]:
        return ["file"]

    def extract(self) -> Iterator[NodeSpec | EdgeSpec]:
        dir_id = f"directory:{self.repo_path}:{self.repo_path.name}"
        yield NodeSpec(
            node_id=dir_id,
            kind="directory",
            name=self.repo_path.name,
            qualname=str(self.repo_path),
            source_path=str(self.repo_path),
        )
        for f in sorted(self.repo_path.glob("*.txt")):
            file_id = f"file:{f}:{f.name}"
            yield NodeSpec(
                node_id=file_id,
                kind="file",
                name=f.name,
                qualname=str(f),
                source_path=str(f),
                docstring=f.read_text(encoding="utf-8").strip(),
            )
            yield EdgeSpec(source_id=dir_id, target_id=file_id, relation="CONTAINS")


class _FileTreeModule(KGModule):
    def make_extractor(self) -> KGExtractor:
        return _FileTreeExtractor(self.repo_root, self.config)

    def kind(self) -> str:
        return "filetree"

    def stats(self) -> dict[str, Any]:
        ext = self.make_extractor()
        items = list(ext.extract())
        nodes = [i for i in items if isinstance(i, NodeSpec)]
        edges = [i for i in items if isinstance(i, EdgeSpec)]
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "coverage": ext.coverage_metric(nodes),
        }


class TestConcreteKGModuleAndExtractor:
    @pytest.fixture
    def corpus(self, tmp_path: Path) -> Path:
        (tmp_path / "a.txt").write_text("A document about rivers.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("A document about mountains.", encoding="utf-8")
        (tmp_path / "c.txt").write_text("", encoding="utf-8")  # empty → uncovered
        return tmp_path

    def test_extractor_yields_correct_counts(self, corpus: Path) -> None:
        ext = _FileTreeExtractor(corpus)
        items = list(ext.extract())
        nodes = [i for i in items if isinstance(i, NodeSpec)]
        edges = [i for i in items if isinstance(i, EdgeSpec)]
        assert len(nodes) == 4  # 1 dir + 3 files
        assert len(edges) == 3  # one CONTAINS per file

    def test_coverage_metric_reflects_docstring_presence(self, corpus: Path) -> None:
        ext = _FileTreeExtractor(corpus)
        nodes = [i for i in ext.extract() if isinstance(i, NodeSpec)]
        # 2 of 3 files have non-empty docstrings
        cov = ext.coverage_metric(nodes)
        assert abs(cov - 2 / 3) < 1e-6

    def test_module_stats_integrates_extractor(self, corpus: Path) -> None:
        mod = _FileTreeModule(repo_root=corpus)
        stats = mod.stats()
        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 3
        assert abs(stats["coverage"] - 2 / 3) < 1e-6

    def test_snapshot_manager_captures_module_stats(self, corpus: Path, tmp_path: Path) -> None:
        mod = _FileTreeModule(repo_root=corpus)
        mgr = SnapshotManager(tmp_path / "snaps", package_name="filetree-kg")

        snap = mgr.capture(
            version="0.1.0",
            branch="main",
            graph_stats_dict=mod.stats(),
            tree_hash="corpus-v1",
        )
        path = mgr.save_snapshot(snap)
        assert path is not None and path.exists()

        loaded = mgr.load_snapshot("corpus-v1")
        assert loaded is not None
        assert loaded.metrics["total_nodes"] == 4
        assert abs(loaded.metrics["coverage"] - 2 / 3) < 1e-6


# ---------------------------------------------------------------------------
# SnapshotManager git subprocess integration
# ---------------------------------------------------------------------------


class TestSnapshotManagerGitIntegration:
    """capture() without explicit branch/tree_hash uses subprocess git calls.
    We're inside a git repo, so these should return real non-empty values."""

    def test_auto_detected_branch_is_non_empty(self, mgr: SnapshotManager) -> None:
        snap = mgr.capture(
            version="0.1.0",
            graph_stats_dict={"total_nodes": 5, "total_edges": 2},
            tree_hash="git-test",
        )
        assert snap.branch != "" and snap.branch != "unknown"

    def test_auto_detected_tree_hash_is_hex(self, tmp_path: Path) -> None:
        mgr2 = SnapshotManager(tmp_path / "snaps", package_name="test")
        snap = mgr2.capture(
            version="0.1.0",
            graph_stats_dict={"total_nodes": 5, "total_edges": 2},
        )
        assert len(snap.tree_hash) == 40
        assert all(c in "0123456789abcdef" for c in snap.tree_hash)
