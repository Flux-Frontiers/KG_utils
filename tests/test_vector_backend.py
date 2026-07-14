"""Tests for kg_utils.vector_backend — LanceDBBackend and SqliteVecBackend.

Both backends are exercised through the same VectorBackend protocol against
real LanceDB and real sqlite-vec (no mocks), plus a parity test asserting they
return identical top-k ids on the same tiny corpus, an int8 round-trip
(sqlite-vec gotcha: raw blobs must be wrapped with vec_int8), and a
SemanticIndex integration test over each backend with a deterministic fake
embedder (no model download).
"""

# pylint: disable=redefined-outer-name,missing-function-docstring

from __future__ import annotations

import numpy as np
import pytest

from unittest.mock import MagicMock

from kg_utils.semantic import SeedHit, SemanticIndex
from kg_utils.vector_backend import (
    LanceDBBackend,
    SqliteVecBackend,
    VectorBackend,
    _pq_subvectors,
)

sqlite_vec = pytest.importorskip("sqlite_vec")
_ = pytest.importorskip("lancedb")

DIM = 16
META = ("kind", "name", "qualname", "module_path")


def _corpus(n: int = 8) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(42)
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rows = [
        {
            "id": f"n{i}",
            "kind": "class" if i % 2 else "function",
            "name": f"name{i}",
            "qualname": f"q{i}",
            "module_path": f"m{i}.py",
            "text": f"text {i}",
            "vector": v[i].tolist(),
        }
        for i in range(n)
    ]
    return rows, v


def _make_lance(tmp_path, **kw) -> LanceDBBackend:
    return LanceDBBackend(tmp_path / "lancedb", table="t", dim=DIM, meta_columns=META, **kw)


def _make_sqlite(tmp_path, **kw) -> SqliteVecBackend:
    return SqliteVecBackend(tmp_path / "vectors.sqlite", dim=DIM, meta_columns=META, **kw)


@pytest.fixture(params=["lancedb", "sqlite"])
def backend(request, tmp_path) -> VectorBackend:
    b = _make_lance(tmp_path) if request.param == "lancedb" else _make_sqlite(tmp_path)
    b.open(wipe=True)
    return b


# ---------------------------------------------------------------------------
# Protocol conformance — both backends
# ---------------------------------------------------------------------------


def test_is_vector_backend(backend):
    assert isinstance(backend, VectorBackend)


def test_upsert_count_and_existing_ids(backend):
    rows, _ = _corpus()
    written = backend.upsert(rows, batch_size=3)
    assert written == 8
    assert backend.count() == 8
    assert backend.existing_ids() == {f"n{i}" for i in range(8)}


def test_search_returns_self_first(backend):
    rows, v = _corpus()
    backend.upsert(rows)
    hits = backend.search(v[3].tolist(), 4)
    assert hits[0]["id"] == "n3"
    assert hits[0]["_distance"] == pytest.approx(0.0, abs=1e-5)
    assert len(hits) == 4
    # returned rows carry the declared meta columns
    for col in META:
        assert col in hits[0]


def test_filtered_search_is_prefilter(backend):
    rows, v = _corpus()
    backend.upsert(rows)
    hits = backend.search(v[3].tolist(), 4, where="kind = 'class'")
    assert hits, "expected some class hits"
    assert all(h["kind"] == "class" for h in hits)
    assert all(h["id"] in {"n1", "n3", "n5", "n7"} for h in hits)


def test_delete_ids(backend):
    rows, _ = _corpus()
    backend.upsert(rows)
    removed = backend.delete_ids(["n0", "n1", "missing"])
    assert removed == 2
    assert backend.count() == 6
    assert "n0" not in backend.existing_ids()


def test_upsert_replaces_existing_id_incremental(backend):
    # Fresh build skips dedup for speed; replace semantics apply on a
    # subsequent non-fresh (incremental) open — the real re-embed path.
    rows, v = _corpus()
    backend.upsert(rows)
    backend.open(wipe=False)
    rows[2] = {**rows[2], "vector": v[5].tolist()}
    backend.upsert([rows[2]])
    assert backend.count() == 8
    ids = [h["id"] for h in backend.search(v[5].tolist(), 8)]
    assert ids.count("n2") == 1


# ---------------------------------------------------------------------------
# Cross-backend parity — the Phase 1 verify gate
# ---------------------------------------------------------------------------


def test_backends_agree_on_topk(tmp_path):
    rows, v = _corpus(12)
    lb = _make_lance(tmp_path)
    sb = _make_sqlite(tmp_path)
    for b in (lb, sb):
        b.open(wipe=True)
        b.upsert(rows)
    for qi in (0, 5, 9):  # 3 sample queries
        lb_ids = [h["id"] for h in lb.search(v[qi].tolist(), 5)]
        sb_ids = [h["id"] for h in sb.search(v[qi].tolist(), 5)]
        assert lb_ids == sb_ids, f"query {qi}: {lb_ids} != {sb_ids}"


# ---------------------------------------------------------------------------
# sqlite-vec specifics
# ---------------------------------------------------------------------------


def test_sqlite_int8_round_trip(tmp_path):
    """int8 blobs must be wrapped with vec_int8() on insert AND match."""
    rows, v = _corpus()
    sb = SqliteVecBackend(
        tmp_path / "vectors_int8.sqlite", dim=DIM, meta_columns=META, dtype="int8"
    )
    sb.open(wipe=True)
    sb.upsert(rows)
    assert sb.count() == 8
    hits = sb.search(v[4].tolist(), 3)
    # quantization preserves the self-hit at rank 0
    assert hits[0]["id"] == "n4"


def test_sqlite_rejects_bad_dtype(tmp_path):
    with pytest.raises(ValueError):
        SqliteVecBackend(tmp_path / "x.sqlite", dim=DIM, dtype="float16")


def test_sqlite_wipe_resets(tmp_path):
    rows, _ = _corpus()
    sb = _make_sqlite(tmp_path)
    sb.open(wipe=True)
    sb.upsert(rows)
    assert sb.count() == 8
    sb.open(wipe=True)
    assert sb.count() == 0


# ---------------------------------------------------------------------------
# SemanticIndex over both backends (deterministic fake embedder)
# ---------------------------------------------------------------------------

_KEYWORDS = ["alpha", "beta", "gamma", "delta"]


class _FakeEmbedder:
    """Maps any text to a basis vector chosen by the keyword it contains."""

    dim = len(_KEYWORDS)

    def embed_texts(self, texts, encode_batch_size: int = 128):
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            low = t.lower()
            for i, kw in enumerate(_KEYWORDS):
                if kw in low:
                    vec[i] = 1.0
            if not any(vec):
                vec[0] = 1e-3
            out.append(vec)
        return out

    def embed_query(self, query: str):
        return self.embed_texts([query])[0]


class _FakeStore:
    def query_nodes(self, *, kinds=None, module=None):
        return [
            {
                "id": f"{kw}:1",
                "kind": "function",
                "name": kw,
                "qualname": f"{kw}_fn",
                "module_path": f"{kw}.py",
                "lineno": 1,
                "docstring": f"does {kw} things",
            }
            for kw in _KEYWORDS
        ]


# ---------------------------------------------------------------------------
# LanceDB ANN gating (moved here from doc_kg; mock table, no real LanceDB)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [384, 768, 100, 1024, 17])
def test_pq_subvectors_divides_dim(dim):
    m = _pq_subvectors(dim)
    assert m >= 1
    assert dim % m == 0


def test_pq_subvectors_targets_about_16_dims():
    assert _pq_subvectors(384) == 24  # 384 / 16
    assert _pq_subvectors(768) == 48


def _lance_with_mock(tmp_path, mock_tbl, **kw):
    be = LanceDBBackend(tmp_path / "ld", table="t", dim=384, meta_columns=META, **kw)
    be._tbl = mock_tbl
    return be


class TestAnnGate:
    def test_below_threshold_skips_index(self, tmp_path):
        tbl = MagicMock()
        tbl.count_rows.return_value = 3_278
        be = _lance_with_mock(tmp_path, tbl, ann_threshold=50_000)
        assert be.maybe_create_ann_index(quiet=True) is False
        tbl.create_index.assert_not_called()

    def test_at_or_above_threshold_builds_index(self, tmp_path):
        tbl = MagicMock()
        tbl.count_rows.return_value = 683_001
        be = _lance_with_mock(tmp_path, tbl, ann_threshold=50_000, ann_index_type="IVF_FLAT")
        assert be.maybe_create_ann_index(quiet=True) is True
        tbl.create_index.assert_called_once()
        _, kwargs = tbl.create_index.call_args
        assert kwargs["metric"] == "cosine"
        assert kwargs["vector_column_name"] == "vector"
        assert kwargs["num_partitions"] >= 1

    def test_ivf_pq_passes_num_sub_vectors(self, tmp_path):
        tbl = MagicMock()
        tbl.count_rows.return_value = 100_000
        be = _lance_with_mock(tmp_path, tbl, ann_threshold=1, ann_index_type="IVF_PQ")
        be.maybe_create_ann_index(quiet=True)
        _, kwargs = tbl.create_index.call_args
        assert kwargs["num_sub_vectors"] == 24

    def test_ivf_flat_omits_num_sub_vectors(self, tmp_path):
        tbl = MagicMock()
        tbl.count_rows.return_value = 100_000
        be = _lance_with_mock(tmp_path, tbl, ann_threshold=1, ann_index_type="IVF_FLAT")
        be.maybe_create_ann_index(quiet=True)
        _, kwargs = tbl.create_index.call_args
        assert "num_sub_vectors" not in kwargs

    def test_threshold_zero_disables_index(self, tmp_path):
        tbl = MagicMock()
        tbl.count_rows.return_value = 1_000_000
        be = _lance_with_mock(tmp_path, tbl, ann_threshold=0)
        assert be.maybe_create_ann_index(quiet=True) is False
        tbl.create_index.assert_not_called()

    def test_num_partitions_sqrt_heuristic(self, tmp_path):
        tbl = MagicMock()
        tbl.count_rows.return_value = 1_000_000  # sqrt = 1000
        be = _lance_with_mock(tmp_path, tbl, ann_threshold=1)
        be.maybe_create_ann_index(quiet=True)
        _, kwargs = tbl.create_index.call_args
        assert kwargs["num_partitions"] == 1000

    def test_create_index_failure_falls_back_to_flat(self, tmp_path):
        tbl = MagicMock()
        tbl.count_rows.return_value = 100_000
        tbl.create_index.side_effect = RuntimeError("lancedb said no")
        be = _lance_with_mock(tmp_path, tbl, ann_threshold=1)
        assert be.maybe_create_ann_index(quiet=True) is False

    def test_legacy_lancedb_without_index_type_kwarg(self, tmp_path):
        tbl = MagicMock()
        tbl.count_rows.return_value = 100_000

        def _create(*_args, **kwargs):
            if "index_type" in kwargs:
                raise TypeError("unexpected keyword 'index_type'")
            return None

        tbl.create_index.side_effect = _create
        be = _lance_with_mock(tmp_path, tbl, ann_threshold=1)
        assert be.maybe_create_ann_index(quiet=True) is True
        assert tbl.create_index.call_count == 2


@pytest.mark.parametrize("which", ["lancedb", "sqlite"])
def test_semantic_index_over_backend(tmp_path, which):
    if which == "lancedb":
        be = LanceDBBackend(tmp_path / "ld", table="t", dim=_FakeEmbedder.dim, meta_columns=META)
    else:
        be = SqliteVecBackend(tmp_path / "v.sqlite", dim=_FakeEmbedder.dim, meta_columns=META)

    idx = SemanticIndex(tmp_path / "ld", embedder=_FakeEmbedder(), backend=be)
    stats = idx.build(_FakeStore(), wipe=True, batch_size=2)
    assert stats["indexed_rows"] == 4
    assert stats["dim"] == 4

    hits = idx.search("tell me about gamma", k=2)
    assert isinstance(hits[0], SeedHit)
    assert hits[0].name == "gamma"
    assert hits[0].qualname == "gamma_fn"

    # filtered search reaches the backend's where path
    hits2 = idx.search("alpha", k=4, where="kind = 'function'")
    assert all(h.kind == "function" for h in hits2)
