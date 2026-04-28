"""Tests for kg_utils.embed — Embedder protocol, constants, and path helpers."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


from kg_utils.embed import (
    DEFAULT_MODEL,
    KNOWN_MODELS,
    Embedder,
    kg_model_cache_dir,
    resolve_model_path,
)


# ---------------------------------------------------------------------------
# Embedder protocol (runtime_checkable)
# ---------------------------------------------------------------------------


def test_embedder_protocol_satisfied_by_conforming_object() -> None:
    class _Good:
        def embed_query(self, text: str) -> list[float]:
            return [0.0]

    assert isinstance(_Good(), Embedder)


def test_embedder_protocol_not_satisfied_without_method() -> None:
    class _Bad:
        pass

    assert not isinstance(_Bad(), Embedder)


def test_embedder_protocol_not_satisfied_with_wrong_method_name() -> None:
    class _Wrong:
        def embed(self, text: str) -> list[float]:
            return [0.0]

    assert not isinstance(_Wrong(), Embedder)


# ---------------------------------------------------------------------------
# DEFAULT_MODEL and KNOWN_MODELS
# ---------------------------------------------------------------------------


def test_default_model_is_bge_small() -> None:
    assert DEFAULT_MODEL == "BAAI/bge-small-en-v1.5"


def test_known_models_contains_expected_aliases() -> None:
    expected = {
        "default",
        "bge-small",
        "bge-small-en-v1.5",
        "bge-large",
        "bge-large-en-v1.5",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "nomic",
        "nomic-v1.5",
    }
    assert expected.issubset(KNOWN_MODELS.keys())


def test_known_models_nomic_alias() -> None:
    assert KNOWN_MODELS["nomic"] == "nomic-ai/nomic-embed-text-v1.5"
    assert KNOWN_MODELS["nomic-v1.5"] == KNOWN_MODELS["nomic"]


def test_known_models_bge_large_alias() -> None:
    assert KNOWN_MODELS["bge-large"] == "BAAI/bge-large-en-v1.5"


def test_known_models_sentence_transformers_aliases() -> None:
    assert KNOWN_MODELS["all-MiniLM-L6-v2"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert KNOWN_MODELS["all-mpnet-base-v2"] == "sentence-transformers/all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# kg_model_cache_dir()
# ---------------------------------------------------------------------------


def test_cache_dir_default() -> None:
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("KGRAG_MODEL_DIR", None)
        result = kg_model_cache_dir()
    assert result == Path.home() / ".kgrag" / "models"


def test_cache_dir_env_override(tmp_path: Path) -> None:
    with patch.dict(os.environ, {"KGRAG_MODEL_DIR": str(tmp_path)}):
        result = kg_model_cache_dir()
    assert result == tmp_path.resolve()


def test_cache_dir_env_relative_resolved(tmp_path: Path) -> None:
    relative = "some/relative/path"
    with patch.dict(os.environ, {"KGRAG_MODEL_DIR": relative}):
        result = kg_model_cache_dir()
    assert result.is_absolute()


def test_cache_dir_returns_path_object() -> None:
    result = kg_model_cache_dir()
    assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# resolve_model_path()
# ---------------------------------------------------------------------------


def test_resolve_known_alias_no_fallback_no_env() -> None:
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("KGRAG_MODEL_DIR", None)
        result = resolve_model_path("bge-small")
    expected = kg_model_cache_dir() / "BAAI" / "bge-small-en-v1.5"
    assert result == expected


def test_resolve_unknown_model_passthrough_no_env() -> None:
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("KGRAG_MODEL_DIR", None)
        result = resolve_model_path("org/custom-model")
    expected = kg_model_cache_dir() / "org" / "custom-model"
    assert result == expected


def test_resolve_with_env_var_ignores_local_fallback(tmp_path: Path) -> None:
    fallback = tmp_path / "local_cache"
    with patch.dict(os.environ, {"KGRAG_MODEL_DIR": str(tmp_path / "global")}):
        result = resolve_model_path("bge-small", local_fallback=fallback)
    # KGRAG_MODEL_DIR must win; result should be under the env-var path
    assert str(tmp_path / "global") in str(result)
    assert str(fallback) not in str(result)


def test_resolve_with_local_fallback_uses_double_dash_separator(tmp_path: Path) -> None:
    fallback = tmp_path / "local_cache"
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("KGRAG_MODEL_DIR", None)
        result = resolve_model_path("BAAI/bge-small-en-v1.5", local_fallback=fallback)
    # Local fallback convention: org/model → org--model
    assert result == fallback / "BAAI--bge-small-en-v1.5"


def test_resolve_with_no_fallback_uses_os_sep() -> None:
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("KGRAG_MODEL_DIR", None)
        result = resolve_model_path("BAAI/bge-small-en-v1.5", local_fallback=None)
    # System cache convention: org/model → org<sep>model
    parts = result.parts
    assert "BAAI" in parts
    assert "bge-small-en-v1.5" in parts


def test_resolve_alias_resolved_before_path_construction(tmp_path: Path) -> None:
    fallback = tmp_path / "cache"
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("KGRAG_MODEL_DIR", None)
        result = resolve_model_path("bge-small", local_fallback=fallback)
    # "bge-small" → "BAAI/bge-small-en-v1.5", then double-dash for local
    assert result == fallback / "BAAI--bge-small-en-v1.5"


def test_resolve_env_var_uses_os_sep_not_double_dash(tmp_path: Path) -> None:
    global_dir = tmp_path / "global"
    with patch.dict(os.environ, {"KGRAG_MODEL_DIR": str(global_dir)}):
        result = resolve_model_path("BAAI/bge-small-en-v1.5")
    parts = result.parts
    assert "BAAI" in parts
    assert "bge-small-en-v1.5" in parts
