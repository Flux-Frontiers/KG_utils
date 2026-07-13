"""Tests for kg_utils.corpus_embedder — CorpusEmbedder, EmbeddingCache, resolve_device."""

# pylint: disable=redefined-outer-name,missing-function-docstring,import-outside-toplevel

from __future__ import annotations

import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kg_utils.corpus_embedder import (
    CorpusEmbedder,
    EmbeddingCache,
    _embed_shard,
    _embed_shard_to_file,
    _InlineProgress,
)
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


# ---------------------------------------------------------------------------
# embed_to_cache — streaming JSONL cache (bounded peak memory)
# ---------------------------------------------------------------------------


def _make_indexed_st():
    """Fake SentenceTransformer whose vectors encode each text's trailing index,
    so tests can assert id↔vector alignment survives sharding and merge."""
    fake = _make_fake_st()
    fake.encode.side_effect = lambda batch, **kw: np.array(
        [[float(t.rsplit(" ", 1)[-1])] * 4 for t in batch], dtype="float32"
    )
    return fake


def _write_empty_and_return_dim(_texts, _rows, out_path):
    """Routing-test stand-in for _stream_*: touch out_path, report dim 4."""
    out_path.write_text("")
    return 4


def test_embed_to_cache_forces_sequential_on_gpu(tmp_path):
    """The GPU fan-out guard applies to the streaming path too."""
    embedder = CorpusEmbedder(n_workers=4, device="mps")
    texts = [f"text {i}" for i in range(200)]
    out = tmp_path / "cache.jsonl"

    with (
        patch.object(
            embedder, "_stream_sequential", side_effect=_write_empty_and_return_dim
        ) as seq,
        patch.object(embedder, "_stream_parallel") as par,
    ):
        result = embedder.embed_to_cache(texts, out_path=out)

    seq.assert_called_once()
    par.assert_not_called()
    assert result == out


def test_embed_to_cache_uses_parallel_on_cpu_with_enough_texts(tmp_path):
    embedder = CorpusEmbedder(n_workers=4, device="cpu")
    texts = [f"text {i}" for i in range(200)]
    out = tmp_path / "cache.jsonl"

    with (
        patch.object(embedder, "_stream_sequential") as seq,
        patch.object(embedder, "_stream_parallel", side_effect=_write_empty_and_return_dim) as par,
    ):
        embedder.embed_to_cache(texts, out_path=out)

    par.assert_called_once()
    seq.assert_not_called()


def test_embed_to_cache_empty_texts_writes_header_only(tmp_path):
    embedder = CorpusEmbedder(n_workers=1, device="cpu")
    out = embedder.embed_to_cache([], out_path=tmp_path / "cache.jsonl")

    lines = out.read_text().splitlines()
    assert len(lines) == 1
    meta = json.loads(lines[0])["__meta__"]
    assert meta["dim"] == 0
    assert meta["n_vectors"] == 0


def test_finalize_cache_merges_parts_in_order_and_deletes_them(tmp_path):
    """Parts are concatenated in shard order (= original input order) and
    removed as consumed — a silent reorder would corrupt id↔vector alignment."""
    embedder = CorpusEmbedder(n_workers=1, device="cpu")
    parts = []
    for i in range(3):
        p = tmp_path / f"cache.jsonl.part-{i:05d}"
        p.write_text(f'{{"id":"{i}"}}\n')
        parts.append(p)
    out = tmp_path / "cache.jsonl"

    embedder._finalize_cache(out, parts, dim=4, n_vectors=3)

    lines = out.read_text().splitlines()
    meta = json.loads(lines[0])["__meta__"]
    assert meta["version"] == 1
    assert meta["dim"] == 4
    assert [json.loads(line)["id"] for line in lines[1:]] == ["0", "1", "2"]
    assert not any(p.exists() for p in parts)


def test_stream_sequential_cleans_up_part_on_failure(tmp_path):
    embedder = CorpusEmbedder(n_workers=1, device="cpu")

    def boom(args):
        Path(args[7]).write_text("partial\n")
        raise RuntimeError("boom")

    with (
        patch("kg_utils.corpus_embedder._embed_shard_to_file", side_effect=boom),
        pytest.raises(RuntimeError),
    ):
        embedder._stream_sequential(["a"], [{"id": "0"}], tmp_path / "cache.jsonl")

    assert not list(tmp_path.glob("*.part-*"))


def test_stream_parallel_falls_back_to_sequential_on_pool_failure(tmp_path):
    embedder = CorpusEmbedder(n_workers=2, device="cpu")
    texts = [f"t {i}" for i in range(4)]
    rows = [{"id": str(i)} for i in range(4)]

    with (
        patch(
            "kg_utils.corpus_embedder.multiprocessing.Manager",
            side_effect=RuntimeError("boom"),
        ),
        patch.object(embedder, "_stream_sequential", return_value=4) as seq,
    ):
        dim = embedder._stream_parallel(texts, rows, tmp_path / "cache.jsonl")

    assert dim == 4
    seq.assert_called_once()


@pytest.mark.integration
def test_embed_shard_to_file_streams_rows_and_returns_path(tmp_path):
    """The streaming worker writes header-less JSONL rows in input order and
    returns the part path (not the vectors)."""
    texts = ["text 0", "text 1", "text 2"]
    rows = [{"id": f"node:{i}", "kind": "chunk", "name": f"n{i}"} for i in range(3)]
    part = tmp_path / "cache.jsonl.part-00000"
    fake_st = _make_indexed_st()
    missing = tmp_path / "nonexistent"
    queue = MagicMock()

    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=missing),
        patch("sentence_transformers.SentenceTransformer", return_value=fake_st),
    ):
        worker_id, path, dim = _embed_shard_to_file(
            (texts, rows, "BAAI/bge-small-en-v1.5", 2, 3, queue, "cpu", str(part))
        )

    assert (worker_id, path, dim) == (3, str(part), 4)
    parsed = [json.loads(line) for line in part.read_text().splitlines()]
    assert len(parsed) == 3  # rows only — the parent writes the header
    assert [r["id"] for r in parsed] == ["node:0", "node:1", "node:2"]
    assert [r["vector"][0] for r in parsed] == [0.0, 1.0, 2.0]
    assert parsed[0]["text"] == "text 0"
    # per-batch progress counts plus the end-of-shard sentinel
    assert queue.put.call_args_list[-1].args == (None,)


@pytest.mark.integration
def test_embed_shard_to_file_nomic_prefix_only_for_encoding(tmp_path):
    """Nomic's task prefix is applied to the encoded text but rows keep the original."""
    seen: list[str] = []
    fake_st = _make_fake_st()

    def encode(batch, **kw):
        seen.extend(batch)
        return np.zeros((len(batch), 4), dtype="float32")

    fake_st.encode.side_effect = encode
    part = tmp_path / "cache.jsonl.part-00000"
    missing = tmp_path / "nonexistent"

    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=missing),
        patch("sentence_transformers.SentenceTransformer", return_value=fake_st),
    ):
        _embed_shard_to_file(
            (["hello"], [{"id": "0"}], "nomic-ai/nomic-embed-text-v1", 8, 0, None, "cpu", str(part))
        )

    assert seen == ["search_document: hello"]
    assert json.loads(part.read_text().splitlines()[0])["text"] == "hello"


@pytest.mark.integration
def test_embed_to_cache_sequential_end_to_end(tmp_path):
    """Known sequence in → same sequence out: header, canonical row fields,
    extra metadata keys preserved, no leftover part files."""
    texts = [f"text {i}" for i in range(5)]
    metadata = [{"id": f"node:{i}", "kind": "chunk", "extra": i} for i in range(5)]
    embedder = CorpusEmbedder(model_name="BAAI/bge-small-en-v1.5", n_workers=1, device="cpu")
    fake_st = _make_indexed_st()
    missing = tmp_path / "nonexistent"

    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=missing),
        patch("sentence_transformers.SentenceTransformer", return_value=fake_st),
    ):
        out = embedder.embed_to_cache(texts, metadata, out_path=tmp_path / "cache.jsonl")

    lines = out.read_text().splitlines()
    meta = json.loads(lines[0])["__meta__"]
    assert meta["version"] == 1
    assert meta["model"] == "BAAI/bge-small-en-v1.5"
    assert meta["dim"] == 4
    assert meta["n_vectors"] == 5

    rows_out = [json.loads(line) for line in lines[1:]]
    assert [r["id"] for r in rows_out] == [f"node:{i}" for i in range(5)]
    assert [r["vector"][0] for r in rows_out] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert rows_out[0]["title"] == ""
    assert rows_out[0]["file_path"] == ""
    assert rows_out[2]["extra"] == 2
    assert not list(tmp_path.glob("*.part-*"))


@pytest.mark.integration
def test_embed_to_cache_default_ids_are_input_indices(tmp_path):
    texts = [f"text {i}" for i in range(3)]
    embedder = CorpusEmbedder(model_name="BAAI/bge-small-en-v1.5", n_workers=1, device="cpu")
    fake_st = _make_indexed_st()
    missing = tmp_path / "nonexistent"

    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=missing),
        patch("sentence_transformers.SentenceTransformer", return_value=fake_st),
    ):
        out = embedder.embed_to_cache(texts, out_path=tmp_path / "cache.jsonl")

    rows_out = [json.loads(line) for line in out.read_text().splitlines()[1:]]
    assert [r["id"] for r in rows_out] == ["0", "1", "2"]


@pytest.mark.integration
def test_embed_to_cache_gzip_output(tmp_path):
    texts = [f"text {i}" for i in range(4)]
    embedder = CorpusEmbedder(model_name="BAAI/bge-small-en-v1.5", n_workers=1, device="cpu")
    fake_st = _make_indexed_st()
    missing = tmp_path / "nonexistent"

    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=missing),
        patch("sentence_transformers.SentenceTransformer", return_value=fake_st),
    ):
        out = embedder.embed_to_cache(texts, out_path=tmp_path / "cache.jsonl.gz")

    with gzip.open(out, "rt", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert json.loads(lines[0])["__meta__"]["dim"] == 4
    assert [json.loads(line)["vector"][0] for line in lines[1:]] == [0.0, 1.0, 2.0, 3.0]
