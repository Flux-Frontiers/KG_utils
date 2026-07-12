"""Tests for kg_utils.corpus_embedder — CorpusEmbedder, EmbeddingCache, resolve_device."""

# pylint: disable=redefined-outer-name,missing-function-docstring,import-outside-toplevel

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kg_utils.corpus_embedder import CorpusEmbedder, EmbeddingCache, _embed_shard, _InlineProgress
from kg_utils.embedder import resolve_device

# ---------------------------------------------------------------------------
# EmbeddingCache creation
# ---------------------------------------------------------------------------


def test_embedding_cache_basic_creation():
    cache = EmbeddingCache(
        model="test-model",
        dim=32,
        texts=["hello", "world"],
        vectors=[[0.1] * 32, [0.2] * 32],
    )
    assert cache.model == "test-model"
    assert cache.dim == 32
    assert cache.texts == ["hello", "world"]
    assert len(cache.vectors) == 2


def test_embedding_cache_n_vectors_property():
    vectors = [[float(i)] * 8 for i in range(5)]
    cache = EmbeddingCache(model="m", dim=8, texts=[f"t{i}" for i in range(5)], vectors=vectors)
    assert cache.n_vectors == 5


def test_embedding_cache_auto_generates_created_at():
    cache = EmbeddingCache(model="m", dim=4, texts=[], vectors=[])
    assert cache.created_at != ""


def test_embedding_cache_auto_created_at_is_iso_format():
    from datetime import datetime

    cache = EmbeddingCache(model="m", dim=4, texts=[], vectors=[])
    assert datetime.fromisoformat(cache.created_at) is not None


def test_embedding_cache_explicit_created_at_preserved():
    ts = "2025-01-15T12:00:00+00:00"
    cache = EmbeddingCache(model="m", dim=4, texts=[], vectors=[], created_at=ts)
    assert cache.created_at == ts


def test_embedding_cache_default_metadata_is_empty_list():
    cache = EmbeddingCache(model="m", dim=4, texts=["t"], vectors=[[0.0] * 4])
    assert cache.metadata == []


# ---------------------------------------------------------------------------
# save_cache / load_cache roundtrip
# ---------------------------------------------------------------------------


def _make_cache(n: int = 5, dim: int = 8) -> EmbeddingCache:
    texts = [f"text number {i}" for i in range(n)]
    vectors = [[float(i * 0.1 + j * 0.01) for j in range(dim)] for i in range(n)]
    metadata = [{"index": i, "source": f"doc{i}.md"} for i in range(n)]
    return EmbeddingCache(
        model="test-model-v1",
        dim=dim,
        texts=texts,
        vectors=vectors,
        metadata=metadata,
        created_at="2025-04-01T00:00:00+00:00",
    )


def test_save_cache_creates_file(tmp_path):
    cache = _make_cache()
    out = tmp_path / "embeddings.json"
    CorpusEmbedder.save_cache(cache, out)
    assert out.exists()


def test_save_cache_creates_parent_dirs(tmp_path):
    cache = _make_cache()
    out = tmp_path / "nested" / "dir" / "embeddings.json"
    CorpusEmbedder.save_cache(cache, out)
    assert out.exists()


def test_load_cache_roundtrip_vectors(tmp_path):
    cache = _make_cache(n=4, dim=8)
    out = tmp_path / "embeddings.json"
    CorpusEmbedder.save_cache(cache, out)
    loaded = CorpusEmbedder.load_cache(out)

    assert len(loaded.vectors) == 4
    for orig, loaded_v in zip(cache.vectors, loaded.vectors):
        assert len(loaded_v) == 8
        for a, b in zip(orig, loaded_v):
            assert abs(a - b) < 1e-6


def test_load_cache_roundtrip_metadata(tmp_path):
    cache = _make_cache(n=3)
    out = tmp_path / "embeddings.json"
    CorpusEmbedder.save_cache(cache, out)
    loaded = CorpusEmbedder.load_cache(out)
    assert loaded.metadata == cache.metadata


def test_save_load_cache_gzip_roundtrip(tmp_path):
    """A .gz suffix writes and reads a gzip-compressed cache."""
    cache = _make_cache(n=4, dim=8)
    out = tmp_path / "embeddings.json.gz"
    CorpusEmbedder.save_cache(cache, out)
    assert out.exists()

    loaded = CorpusEmbedder.load_cache(out)
    assert loaded.model == cache.model
    assert loaded.texts == cache.texts
    assert loaded.n_vectors == cache.n_vectors


def test_save_cache_json_structure(tmp_path):
    cache = _make_cache(n=2)
    out = tmp_path / "embeddings.json"
    CorpusEmbedder.save_cache(cache, out)

    with open(out, encoding="utf-8") as f:
        data = json.load(f)

    for key in ("model", "dim", "n_vectors", "created_at", "texts", "metadata", "embeddings"):
        assert key in data, f"Missing key in saved JSON: {key}"


# ---------------------------------------------------------------------------
# resolve_device (re-exported from kg_utils.embedder; exercised through the
# CorpusEmbedder call surface too, below)
# ---------------------------------------------------------------------------


def test_resolve_device_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("KG_EMBED_DEVICE", "cuda")
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_env_var_used_when_no_explicit_arg(monkeypatch):
    monkeypatch.setenv("KG_EMBED_DEVICE", "mps")
    assert resolve_device(None) == "mps"


def test_resolve_device_normalizes_case_and_whitespace(monkeypatch):
    monkeypatch.delenv("KG_EMBED_DEVICE", raising=False)
    assert resolve_device("  CPU  ") == "cpu"


def test_corpus_embedder_device_defaults_to_resolved_value(monkeypatch):
    monkeypatch.setenv("KG_EMBED_DEVICE", "cpu")
    embedder = CorpusEmbedder()
    assert embedder.device == "cpu"


# ---------------------------------------------------------------------------
# Default n_workers is capped, not unbounded cpu_count // 2 — each CPU worker
# loads a full model copy, so high-core-count machines risk OOM otherwise.
# ---------------------------------------------------------------------------


def test_default_n_workers_capped_at_four_on_high_core_machines(monkeypatch):
    monkeypatch.setattr("kg_utils.corpus_embedder.os.cpu_count", lambda: 20)
    embedder = CorpusEmbedder()
    assert embedder.n_workers == 4


def test_default_n_workers_below_cap_on_low_core_machines(monkeypatch):
    monkeypatch.setattr("kg_utils.corpus_embedder.os.cpu_count", lambda: 4)
    embedder = CorpusEmbedder()
    assert embedder.n_workers == 2


def test_default_n_workers_floors_at_one(monkeypatch):
    monkeypatch.setattr("kg_utils.corpus_embedder.os.cpu_count", lambda: None)
    embedder = CorpusEmbedder()
    assert embedder.n_workers == 1


def test_explicit_n_workers_not_capped(monkeypatch):
    monkeypatch.setattr("kg_utils.corpus_embedder.os.cpu_count", lambda: 20)
    embedder = CorpusEmbedder(n_workers=8)
    assert embedder.n_workers == 8


# ---------------------------------------------------------------------------
# _InlineProgress — put() protocol adapter used by the sequential/GPU path
# ---------------------------------------------------------------------------


def test_inline_progress_put_advances_the_bar():
    progress = MagicMock()
    adapter = _InlineProgress(progress, task_id="task-1")
    adapter.put(5)
    progress.advance.assert_called_once_with("task-1", 5)


def test_inline_progress_put_none_sentinel_is_noop():
    progress = MagicMock()
    adapter = _InlineProgress(progress, task_id="task-1")
    adapter.put(None)
    progress.advance.assert_not_called()


# ---------------------------------------------------------------------------
# GPU devices force single-process embedding (the OOM-prevention guard)
# ---------------------------------------------------------------------------


def test_embed_forces_sequential_on_mps_even_with_many_texts_and_workers():
    """A GPU device can't be shared across spawn workers; embed() must not
    fan out into _embed_parallel regardless of corpus size or n_workers."""
    embedder = CorpusEmbedder(n_workers=4, device="mps")
    texts = [f"text {i}" for i in range(200)]

    with (
        patch.object(embedder, "_embed_sequential", return_value=[[0.0]] * 200) as seq,
        patch.object(embedder, "_embed_parallel") as par,
    ):
        embedder.embed(texts)

    seq.assert_called_once()
    par.assert_not_called()


def test_embed_forces_sequential_on_cuda():
    embedder = CorpusEmbedder(n_workers=4, device="cuda")
    texts = [f"text {i}" for i in range(200)]

    with (
        patch.object(embedder, "_embed_sequential", return_value=[[0.0]] * 200) as seq,
        patch.object(embedder, "_embed_parallel") as par,
    ):
        embedder.embed(texts)

    seq.assert_called_once()
    par.assert_not_called()


def test_embed_uses_parallel_on_cpu_with_enough_texts_and_workers():
    embedder = CorpusEmbedder(n_workers=4, device="cpu")
    texts = [f"text {i}" for i in range(200)]

    with (
        patch.object(embedder, "_embed_sequential") as seq,
        patch.object(embedder, "_embed_parallel", return_value=[[0.0]] * 200) as par,
    ):
        embedder.embed(texts)

    par.assert_called_once()
    seq.assert_not_called()


# ---------------------------------------------------------------------------
# _embed_shard — model resolution, device pinning, progress (mocked)
# ---------------------------------------------------------------------------


def _make_fake_st(dim: int = 4):
    """Return a mock SentenceTransformer that produces deterministic vectors."""
    fake = MagicMock()
    fake.encode.return_value = np.zeros((1, dim), dtype="float32")
    # load_sentence_transformer() ends with `model = model.to(device)`; the
    # mock must return itself so the configured `.encode` survives the move.
    fake.to.return_value = fake
    return fake


@pytest.mark.integration
def test_embed_shard_returns_correct_shape(tmp_path):
    """_embed_shard returns (worker_id, list_of_vectors) of expected length."""
    texts = ["a", "b", "c"]
    model_name = "BAAI/bge-small-en-v1.5"
    fake_st = MagicMock()
    fake_st.encode.side_effect = lambda batch, **kw: np.zeros((len(batch), 4), dtype="float32")
    fake_st.to.return_value = fake_st
    missing = tmp_path / "nonexistent"

    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=missing),
        patch("sentence_transformers.SentenceTransformer", return_value=fake_st),
    ):
        worker_id, vectors = _embed_shard((texts, model_name, 8, 7, None, None))

    assert worker_id == 7
    assert len(vectors) == 3
    assert len(vectors[0]) == 4


@pytest.mark.integration
def test_embed_shard_pins_device_when_given(tmp_path):
    """A concrete device pins the loaded model via model.to(device) —
    what keeps N parallel CPU workers from each auto-selecting MPS."""
    texts = ["a"]
    model_name = "BAAI/bge-small-en-v1.5"
    fake_st = _make_fake_st()
    fake_st.encode.side_effect = lambda batch, **kw: np.zeros((len(batch), 4), dtype="float32")
    missing = tmp_path / "nonexistent"

    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=missing),
        patch("sentence_transformers.SentenceTransformer", return_value=fake_st),
    ):
        _embed_shard((texts, model_name, 8, 0, None, "cpu"))

    fake_st.to.assert_called_with("cpu")


@pytest.mark.integration
def test_embed_shard_reports_progress(tmp_path):
    """A non-None progress_queue receives per-batch counts and a None sentinel."""
    texts = ["a", "b", "c", "d", "e"]
    model_name = "BAAI/bge-small-en-v1.5"
    fake_st = _make_fake_st()
    fake_st.encode.side_effect = lambda batch, **kw: np.zeros((len(batch), 4), dtype="float32")
    missing = tmp_path / "nonexistent"
    queue = MagicMock()

    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=missing),
        patch("sentence_transformers.SentenceTransformer", return_value=fake_st),
    ):
        _embed_shard((texts, model_name, 2, 0, queue, None))

    # 3 batches of size <= 2 (2, 2, 1) plus a trailing None sentinel.
    assert queue.put.call_count == 4
    assert queue.put.call_args_list[-1].args == (None,)


@pytest.mark.integration
def test_embed_sequential_wires_inline_progress_and_returns_vectors(tmp_path):
    """_embed_sequential (the single-process/GPU path) shows a live progress
    bar via _InlineProgress instead of running silently — same _embed_shard,
    just handed an in-process adapter instead of None."""
    texts = ["a", "b", "c"]
    embedder = CorpusEmbedder(model_name="BAAI/bge-small-en-v1.5", device="cpu")
    fake_st = _make_fake_st()
    fake_st.encode.side_effect = lambda batch, **kw: np.zeros((len(batch), 4), dtype="float32")
    missing = tmp_path / "nonexistent"

    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=missing),
        patch("sentence_transformers.SentenceTransformer", return_value=fake_st),
    ):
        vectors = embedder._embed_sequential(texts)

    assert len(vectors) == 3
    assert len(vectors[0]) == 4
