"""Tests for kg_utils.embedder — Embedder base class, load helpers, and factories."""

# pylint: disable=redefined-outer-name,missing-function-docstring,too-few-public-methods,import-outside-toplevel

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kg_utils.embed import DEFAULT_MODEL, KNOWN_MODELS
from kg_utils.embedder import (
    Embedder,
    SentenceTransformerEmbedder,
    get_embedder,
    load_sentence_transformer,
    wrap_embedder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def st_model() -> Any:
    """Load the default SentenceTransformer once per test session."""
    return load_sentence_transformer(DEFAULT_MODEL)


@pytest.fixture(scope="session")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Embedder abstract base
# ---------------------------------------------------------------------------


def test_embedder_embed_texts_raises() -> None:
    emb = Embedder()
    with pytest.raises(NotImplementedError):
        emb.embed_texts(["hello"])


def test_embedder_embed_query_delegates_to_embed_texts() -> None:
    class _Stub(Embedder):
        dim = 4
        calls: list[list[str]] = []

        def embed_texts(self, texts):
            self.calls.append(texts)
            return [[0.1, 0.2, 0.3, 0.4]]

    stub = _Stub()
    result = stub.embed_query("test query")
    assert result == [0.1, 0.2, 0.3, 0.4]
    assert stub.calls == [["test query"]]


# ---------------------------------------------------------------------------
# KNOWN_MODELS alias resolution
# ---------------------------------------------------------------------------


def test_known_models_default_alias() -> None:
    assert KNOWN_MODELS["bge-small"] == "BAAI/bge-small-en-v1.5"
    assert KNOWN_MODELS["default"] == DEFAULT_MODEL


def test_known_models_full_id_passthrough() -> None:
    resolved = KNOWN_MODELS.get("BAAI/bge-small-en-v1.5", "BAAI/bge-small-en-v1.5")
    assert resolved == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# load_sentence_transformer — path logic (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_load_uses_local_files_only_when_path_exists(tmp_path) -> None:
    """When the resolved local path exists, local_files_only=True must be passed."""
    fake_model_dir = tmp_path / "BAAI" / "bge-small-en-v1.5"
    fake_model_dir.mkdir(parents=True)

    # SentenceTransformer is imported lazily inside the function, so patch
    # it at the sentence_transformers module level.
    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=fake_model_dir),
        patch("sentence_transformers.SentenceTransformer", return_value=MagicMock()) as mock_st,
    ):
        load_sentence_transformer("bge-small")
    assert mock_st.call_args.kwargs.get("local_files_only") is True


@pytest.mark.integration
def test_load_falls_back_to_network_on_os_error(tmp_path) -> None:
    """Falls back to network fetch when local_files_only raises OSError."""
    missing = tmp_path / "nonexistent"  # does not exist

    with (
        patch("kg_utils.embedder.resolve_model_path", return_value=missing),
        patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=[OSError("not cached"), MagicMock()],
        ) as mock_st,
    ):
        result = load_sentence_transformer("bge-small")

    assert result is not None
    assert mock_st.call_count == 2
    assert mock_st.call_args_list[1].kwargs.get("local_files_only") is not True


# ---------------------------------------------------------------------------
# load_sentence_transformer — real model
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_load_returns_sentence_transformer(st_model: Any) -> None:
    assert st_model is not None
    dim_fn = getattr(st_model, "get_embedding_dimension", None) or getattr(
        st_model, "get_sentence_embedding_dimension", None
    )
    assert dim_fn is not None and dim_fn() == 384


@pytest.mark.integration
def test_load_uses_mps_or_cpu(st_model: Any) -> None:
    device = str(st_model.device)
    assert device in ("mps:0", "cpu", "cuda:0"), f"unexpected device: {device}"


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_embedder_dim(embedder: SentenceTransformerEmbedder) -> None:
    assert embedder.dim == 384


@pytest.mark.integration
def test_embedder_model_name_resolved(embedder: SentenceTransformerEmbedder) -> None:
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"


@pytest.mark.integration
def test_embedder_repr(embedder: SentenceTransformerEmbedder) -> None:
    r = repr(embedder)
    assert "SentenceTransformerEmbedder" in r
    assert "bge-small-en-v1.5" in r
    assert "384" in r


@pytest.mark.integration
def test_embed_query_length(embedder: SentenceTransformerEmbedder) -> None:
    vec = embedder.embed_query("Samuel Pepys walked to Westminster Hall.")
    assert len(vec) == 384


@pytest.mark.integration
def test_embed_query_normalized(embedder: SentenceTransformerEmbedder) -> None:
    vec = embedder.embed_query("test normalization")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.integration
def test_embed_texts_batch(embedder: SentenceTransformerEmbedder) -> None:
    texts = [
        "Pepys dined at the tavern.",
        "The King walked through St James's Park.",
        "Plague spreads through the city.",
    ]
    vecs = embedder.embed_texts(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


@pytest.mark.integration
def test_embed_texts_normalized(embedder: SentenceTransformerEmbedder) -> None:
    vecs = embedder.embed_texts(["normalization check"])
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.integration
def test_embed_texts_different_inputs_differ(embedder: SentenceTransformerEmbedder) -> None:
    a = embedder.embed_texts(["fire destroys the city"])
    b = embedder.embed_texts(["Pepys plays the lute"])
    # Semantically unrelated — cosine similarity should be well below 1
    dot = sum(x * y for x, y in zip(a[0], b[0]))
    assert dot < 0.99


@pytest.mark.integration
def test_embed_query_consistent_with_embed_texts(embedder: SentenceTransformerEmbedder) -> None:
    text = "The diary entry for 1 January 1660."
    via_query = embedder.embed_query(text)
    via_texts = embedder.embed_texts([text])[0]
    # Should be identical (same encode call)
    assert via_query == via_texts


# ---------------------------------------------------------------------------
# get_embedder factory
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_get_embedder_returns_sentence_transformer_embedder() -> None:
    emb = get_embedder(DEFAULT_MODEL)
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert emb.dim == 384


@pytest.mark.integration
def test_get_embedder_alias() -> None:
    emb = get_embedder("bge-small")
    assert emb.model_name == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# wrap_embedder
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_wrap_embedder_is_embedder(st_model: Any) -> None:
    wrapped = wrap_embedder(st_model, DEFAULT_MODEL)
    assert isinstance(wrapped, Embedder)


@pytest.mark.integration
def test_wrap_embedder_dim(st_model: Any) -> None:
    wrapped = wrap_embedder(st_model, DEFAULT_MODEL)
    assert wrapped.dim == 384


@pytest.mark.integration
def test_wrap_embedder_model_name(st_model: Any) -> None:
    wrapped = wrap_embedder(st_model, "bge-small")
    assert wrapped.model_name == "BAAI/bge-small-en-v1.5"  # type: ignore[attr-defined]


@pytest.mark.integration
def test_wrap_embedder_embed_query(st_model: Any) -> None:
    wrapped = wrap_embedder(st_model, DEFAULT_MODEL)
    vec = wrapped.embed_query("Pepys goes to the theatre.")
    assert len(vec) == 384
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.integration
def test_wrap_embedder_embed_texts(st_model: Any) -> None:
    wrapped = wrap_embedder(st_model, DEFAULT_MODEL)
    vecs = wrapped.embed_texts(["entry one", "entry two"])
    assert len(vecs) == 2
    assert all(len(v) == 384 for v in vecs)


@pytest.mark.integration
def test_wrap_embedder_matches_direct_encode(
    st_model: Any, embedder: SentenceTransformerEmbedder
) -> None:
    """Wrapped model and direct SentenceTransformerEmbedder produce identical vectors."""
    text = "consistent encoding test"
    wrapped = wrap_embedder(st_model, DEFAULT_MODEL)
    from_wrapped = wrapped.embed_query(text)
    from_embedder = embedder.embed_query(text)
    assert from_wrapped == from_embedder


# ---------------------------------------------------------------------------
# doc_kg backward-compat re-export
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec("doc_kg") is None,
    reason="doc_kg not installed",
)
def test_doc_kg_re_exports_embedder_classes() -> None:
    from doc_kg.index import Embedder as DocEmbedder  # pylint: disable=import-error
    from doc_kg.index import SentenceTransformerEmbedder as DocSTE  # pylint: disable=import-error

    assert DocEmbedder is Embedder
    assert DocSTE is SentenceTransformerEmbedder
