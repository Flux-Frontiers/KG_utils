"""KGModule.sync: rewrite the graph, embed only what changed (sweep item 79).

A counting hashing embedder stands in for the model, so these run without the
``semantic`` extra's torch stack and can assert exactly how many texts each
sync embedded. They need numpy and sqlite-vec.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sqlite_vec")

from kg_utils.embedder import Embedder  # noqa: E402
from kg_utils.extractor import KGExtractor  # noqa: E402
from kg_utils.pipeline import KGModule  # noqa: E402
from kg_utils.semantic import SemanticIndex  # noqa: E402
from kg_utils.specs import EdgeSpec, NodeSpec  # noqa: E402
from kg_utils.vector_backend import SYNC_HASH_TABLE, SYNC_INFO_TABLE  # noqa: E402


class _CountingEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder that counts its inputs."""

    dim = 32

    def __init__(self) -> None:
        self.embedded = 0

    def embed_texts(self, texts: list[str], encode_batch_size: int = 32) -> list[list[float]]:
        self.embedded += len(texts)
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"[a-z]+", text.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class _Extractor(KGExtractor):
    """One function node per ``*.txt`` file."""

    def node_kinds(self) -> list[str]:
        return ["function"]

    def edge_kinds(self) -> list[str]:
        return []

    def meaningful_node_kinds(self) -> list[str]:
        return ["function"]

    def extract(self) -> Iterator[NodeSpec | EdgeSpec]:
        for f in sorted(self.repo_path.glob("*.txt")):
            yield NodeSpec(
                node_id=f"function:{f.stem}",
                kind="function",
                name=f.stem,
                qualname=f.stem,
                source_path=f.name,
                docstring=f.read_text(encoding="utf-8"),
            )


class _KG(KGModule):
    _default_dir = ".synckg"

    def make_extractor(self) -> KGExtractor:
        return _Extractor(self.repo_root)

    def kind(self) -> str:
        return "sync"

    def analyze(self) -> str:
        return "# sync"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "alpha.txt").write_text("graph store persists nodes", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("vector index seeds queries", encoding="utf-8")
    (tmp_path / "gamma.txt").write_text("snapshots record metrics", encoding="utf-8")
    return tmp_path


def _kg(repo: Path, embedder: _CountingEmbedder | None = None) -> _KG:
    kg = _KG(repo)
    kg._embedder = embedder or _CountingEmbedder()
    return kg


def _embedded(kg: _KG) -> int:
    emb = kg._embedder
    assert isinstance(emb, _CountingEmbedder)
    return emb.embedded


def _ids(kg: _KG, q: str) -> list[str]:
    return [n["id"] for n in kg.query(q, k=3, hop=0).nodes]


def test_first_sync_embeds_everything_then_nothing(repo: Path) -> None:
    with _kg(repo) as kg:
        first = kg.sync()
        assert (first.embedded_rows, first.deleted_rows, first.indexed_rows) == (3, 0, 3)
        before = _embedded(kg)
        second = kg.sync()
        assert (second.embedded_rows, second.deleted_rows, second.indexed_rows) == (0, 0, 3)
        assert _embedded(kg) == before  # the model was not called at all


def test_sync_embeds_only_a_changed_node(repo: Path) -> None:
    with _kg(repo) as kg:
        kg.sync()
        (repo / "beta.txt").write_text("vector index seeds queries quickly", encoding="utf-8")
        stats = kg.sync()
        assert (stats.embedded_rows, stats.deleted_rows, stats.indexed_rows) == (1, 0, 3)


def test_sync_deletes_a_removed_node_and_adds_a_new_one(repo: Path) -> None:
    with _kg(repo) as kg:
        kg.sync()
        (repo / "gamma.txt").unlink()
        (repo / "delta.txt").write_text("lint finds orphans", encoding="utf-8")
        stats = kg.sync()
        assert (stats.embedded_rows, stats.deleted_rows, stats.indexed_rows) == (1, 1, 3)
        assert "function:gamma" not in _ids(kg, "snapshots record metrics")


def test_sync_matches_a_full_build(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    with _kg(repo) as kg:
        kg.sync()
        (repo / "beta.txt").write_text("vector index seeds many queries", encoding="utf-8")
        (repo / "gamma.txt").unlink()
        kg.sync()
        synced = {q: _ids(kg, q) for q in ("vector index", "graph store", "metrics")}
        synced_nodes = kg.store.stats()["total_nodes"]
    with _kg(repo) as kg:
        kg.build(wipe=True)
        built = {q: _ids(kg, q) for q in synced}
        assert kg.store.stats()["total_nodes"] == synced_nodes
    assert synced == built


def test_a_new_signature_wipes_and_reembeds(repo: Path) -> None:
    with _kg(repo) as kg:
        kg.sync()
        idx = kg.index.sync(kg.store, signature="another-model:32")
        assert (idx["embedded_rows"], idx["indexed_rows"]) == (3, 3)
        again = kg.index.sync(kg.store, signature="another-model:32")
        assert again["embedded_rows"] == 0


def test_a_full_build_resets_sync_state(repo: Path) -> None:
    with _kg(repo) as kg:
        kg.sync()
        kg.build(wipe=True)  # drops the store, sidecars included
        stats = kg.sync()
        assert stats.embedded_rows == 3


def test_sync_state_lives_outside_vec_meta(repo: Path) -> None:
    with _kg(repo) as kg:
        kg.build(wipe=True)
        path = kg.vectors_path
    con = sqlite3.connect(path)
    built_cols = [r[1] for r in con.execute("PRAGMA table_info(vec_meta)")]
    built_tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert SYNC_HASH_TABLE not in built_tables and SYNC_INFO_TABLE not in built_tables

    with _kg(repo) as kg:
        kg.sync()
    con = sqlite3.connect(path)
    synced_cols = [r[1] for r in con.execute("PRAGMA table_info(vec_meta)")]
    synced_tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert synced_cols == built_cols
    assert {SYNC_HASH_TABLE, SYNC_INFO_TABLE} <= synced_tables


def test_build_stats_dict_is_unchanged_for_a_full_build(repo: Path) -> None:
    with _kg(repo) as kg:
        built = kg.build(wipe=True).to_dict()
        synced = kg.sync().to_dict()
    assert "embedded_rows" not in built and "deleted_rows" not in built
    assert synced["embedded_rows"] == 3 and synced["deleted_rows"] == 0
    assert "synced" in str(kg.sync())


class _MemoryBackend:
    """Minimal non-sqlite backend: sync must fall back to a full build."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.wiped = 0

    def open(self, *, wipe: bool = False) -> None:
        if wipe:
            self.rows.clear()
            self.wiped += 1

    def upsert(self, rows: list[dict[str, Any]], *, batch_size: int = 1000) -> int:
        for r in rows:
            self.rows[r["id"]] = r
        return len(rows)

    def delete_ids(self, ids: Any) -> int:
        return sum(1 for i in list(ids) if self.rows.pop(i, None) is not None)

    def existing_ids(self) -> set[str]:
        return set(self.rows)

    def search(self, qvec: Any, k: int, *, where: str | None = None) -> list[dict[str, Any]]:
        return []

    def count(self) -> int:
        return len(self.rows)


def test_a_non_sqlite_backend_falls_back_to_a_full_build(repo: Path) -> None:
    with _kg(repo) as kg:
        kg.build_graph(wipe=True)
        backend = _MemoryBackend()
        idx = SemanticIndex(
            kg.vectors_path,
            embedder=_CountingEmbedder(),
            index_kinds=["function"],
            backend=backend,
        )
        stats = idx.sync(kg.store)
    assert backend.wiped == 1
    assert (stats["embedded_rows"], stats["deleted_rows"], stats["indexed_rows"]) == (3, 0, 3)
