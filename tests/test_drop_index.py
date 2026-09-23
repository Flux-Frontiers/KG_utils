"""A wiped graph drops the vector index it invalidates (sweep item 54).

``build_graph(wipe=True)`` used to leave the previous build's vectors in
place, so a graph-only rebuild followed by ``query()`` seeded from nodes the
new graph might no longer hold -- found in vault_kg, whose CLI deleted the
stale index itself, while connectome_kg only warned. The base class now
drops it. A hashing embedder stands in for the model, so these run without
the ``semantic`` extra's torch stack; they need numpy and sqlite-vec.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sqlite_vec")

from kg_utils.embedder import Embedder  # noqa: E402
from kg_utils.extractor import KGExtractor  # noqa: E402
from kg_utils.pipeline import KGModule  # noqa: E402
from kg_utils.semantic import SemanticIndex  # noqa: E402
from kg_utils.specs import EdgeSpec, NodeSpec  # noqa: E402
from kg_utils.vector_backend import VectorStoreNotFoundError  # noqa: E402


class _HashEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder: no model download."""

    dim = 32

    def embed_texts(self, texts: list[str], encode_batch_size: int = 32) -> list[list[float]]:
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
    _default_dir = ".dropkg"

    def make_extractor(self) -> KGExtractor:
        return _Extractor(self.repo_root)

    def kind(self) -> str:
        return "drop"

    def analyze(self) -> str:
        return "# drop"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "alpha.txt").write_text("graph store persists nodes", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("vector index seeds queries", encoding="utf-8")
    return tmp_path


def _kg(repo: Path) -> _KG:
    kg = _KG(repo)
    kg._embedder = _HashEmbedder()
    return kg


def test_graph_only_rebuild_drops_the_index(repo: Path) -> None:
    with _kg(repo) as kg:
        kg.build(wipe=True)
        assert kg.query("vector index", k=1).nodes
        assert kg.vectors_path.exists()

        (repo / "beta.txt").unlink()  # the note the index still describes
        kg.build_graph(wipe=True)

        assert not kg.vectors_path.exists()
        with pytest.raises(VectorStoreNotFoundError):
            kg.query("vector index", k=1)


def test_full_rebuild_ends_with_a_fresh_index(repo: Path) -> None:
    with _kg(repo) as kg:
        kg.build(wipe=True)
        (repo / "beta.txt").unlink()
        stats = kg.build(wipe=True)
        assert stats.indexed_rows == 1
        ids = {n["id"] for n in kg.query("vector index seeds", k=5, hop=0).nodes}
        assert "function:beta" not in ids


def test_an_unwiped_graph_build_keeps_the_index(repo: Path) -> None:
    with _kg(repo) as kg:
        kg.build(wipe=True)
        kg.build_graph(wipe=False)
        assert kg.vectors_path.exists()
        assert kg.query("vector index", k=1).nodes


def test_drop_index_removes_sidecars_and_a_legacy_store(repo: Path) -> None:
    with _kg(repo) as kg:
        kg.build(wipe=True)
        kg.query("graph", k=1)  # opens the index, and with it a connection
        sidecar = kg.vectors_path.parent / f"{kg.vectors_path.name}-wal"
        sidecar.write_bytes(b"")
        legacy = kg._legacy_store_dir
        (legacy / "vectors.lance").mkdir(parents=True)

        removed = kg.drop_index()

        # WAL mode may also have left a real -shm file; it goes too.
        assert {kg.vectors_path, sidecar, legacy} <= set(removed)
        leftovers = [p for p in kg.vectors_path.parent.iterdir() if p.name.startswith("vectors")]
        assert not leftovers and not legacy.exists()
        assert kg.store.node("function:alpha") is not None  # the graph is untouched
        with pytest.raises(VectorStoreNotFoundError):
            kg.query("graph", k=1)


def test_drop_index_keeps_a_caller_supplied_index(repo: Path) -> None:
    """A module or test that sets its own index keeps it across a wiped build."""
    with _kg(repo) as kg:
        index = kg.index
        kg.build(wipe=True)
        assert kg.index is index
        assert kg.query("vector index", k=1).nodes


def test_drop_index_with_nothing_to_drop(repo: Path) -> None:
    with _kg(repo) as kg:
        assert kg.drop_index() == []


def test_semantic_index_close_tolerates_a_backend_without_close(tmp_path: Path) -> None:
    class _NoClose:
        pass

    index = SemanticIndex(tmp_path / "v.sqlite", embedder=_HashEmbedder(), backend=_NoClose())  # type: ignore[arg-type]
    index.close()  # must not raise
    SemanticIndex(tmp_path / "v.sqlite", embedder=_HashEmbedder()).close()  # no backend yet
