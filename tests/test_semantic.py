"""Tests for kg_utils.semantic — re-export identity and _local_model_path.

SemanticIndex's own build/search behavior is exercised end-to-end by
test_pipeline_module.py (network-dependent, marked integration). This module
covers what changed when Embedder/SentenceTransformerEmbedder/DEFAULT_MODEL/
resolve_model_path stopped being defined locally and started being
re-exported from kg_utils.embed / kg_utils.embedder.
"""

# pylint: disable=redefined-outer-name,missing-function-docstring

from __future__ import annotations

from pathlib import Path

import kg_utils.embed as _embed
import kg_utils.embedder as _embedder
from kg_utils.semantic import (
    DEFAULT_MODEL,
    Embedder,
    SentenceTransformerEmbedder,
    _local_model_path,
    resolve_model_path,
)

# ---------------------------------------------------------------------------
# Re-export identity — semantic.py must not carry a diverged local copy.
# ---------------------------------------------------------------------------


def test_embedder_is_kg_utils_embedder_class():
    assert Embedder is _embedder.Embedder


def test_sentence_transformer_embedder_is_kg_utils_embedder_class():
    assert SentenceTransformerEmbedder is _embedder.SentenceTransformerEmbedder


def test_default_model_matches_kg_utils_embed():
    assert DEFAULT_MODEL == _embed.DEFAULT_MODEL


def test_resolve_model_path_is_kg_utils_embed_function():
    assert resolve_model_path is _embed.resolve_model_path


# ---------------------------------------------------------------------------
# _local_model_path — backward-compat CWD-relative fallback for pycode_kg's
# download-model command (kg_utils.semantic._local_model_path is imported
# directly by pycode_kg/cli/cmd_model.py and cmd_init.py).
# ---------------------------------------------------------------------------


def test_local_model_path_uses_cwd_relative_kgcache_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("KGRAG_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    result = _local_model_path("BAAI/bge-small-en-v1.5")

    assert result == tmp_path / ".kgcache" / "models" / "BAAI--bge-small-en-v1.5"


def test_local_model_path_respects_known_alias(monkeypatch, tmp_path):
    monkeypatch.delenv("KGRAG_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    result = _local_model_path("bge-small")

    assert result == tmp_path / ".kgcache" / "models" / "BAAI--bge-small-en-v1.5"


def test_local_model_path_kgrag_model_dir_env_wins(monkeypatch, tmp_path):
    override = tmp_path / "shared-cache"
    monkeypatch.setenv("KGRAG_MODEL_DIR", str(override))

    result = _local_model_path("BAAI/bge-small-en-v1.5")

    assert result == override / "BAAI" / "bge-small-en-v1.5"


def test_local_model_path_is_absolute():
    assert Path(_local_model_path("bge-small")).is_absolute()
