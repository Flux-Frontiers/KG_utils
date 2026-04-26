"""
kg_utils.embed — Shared embedding protocol and model-cache convention.

Zero external dependencies (stdlib only).  Concrete implementations
(SentenceTransformerEmbedder, LlamaCppEmbedder) live in each KG module and in
kgrag; this module provides only the shared contract they all implement.

Contents
--------
Embedder
    Structural protocol: any object with ``embed_query(text) -> list[float]``
    satisfies it.  KG modules, kgrag adapters, and tests can type-hint against
    this without coupling to any specific implementation.

DEFAULT_MODEL
    Canonical default embedding model for the KGModule stack.
    ``BAAI/bge-small-en-v1.5`` (384-dim, ~24 MB, no licence restrictions).

KNOWN_MODELS
    Short alias → HuggingFace repo ID mapping shared by all modules.
    Lets users write ``"bge-small"`` instead of ``"BAAI/bge-small-en-v1.5"``.

kg_model_cache_dir()
    Return the system-wide model cache root (``~/.kgrag/models/`` by default).
    Override with the ``KGRAG_MODEL_DIR`` environment variable.  All KG modules
    should resolve their model paths through this function so that a single
    ``KGRAG_MODEL_DIR`` setting redirects every module at once.

    Local fallback convention for standalone use::

        path = kg_model_cache_dir() / model_name.replace("/", "--")

Author: Eric G. Suchanek, PhD
License: Elastic 2.0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Shared protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding protocol for the KGModule stack.

    Any object with an ``embed_query`` method satisfies this protocol and can
    be injected into any KGModule-based KG backend (DocKG, MemoryKG, etc.).

    :method embed_query: Embed a single query string into a float vector.
    """

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a dense float vector.

        :param text: The query string to embed.
        :return: Dense float32 vector as a plain Python list.
        """
        ...


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "BAAI/bge-small-en-v1.5"
"""Canonical default embedding model for the KGModule stack (384-dim)."""

KNOWN_MODELS: dict[str, str] = {
    "default": "BAAI/bge-small-en-v1.5",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-v1.5": "nomic-ai/nomic-embed-text-v1.5",
}
"""Short alias → HuggingFace repo ID.  Shared by all KG modules and kgrag."""


# ---------------------------------------------------------------------------
# Shared cache path convention
# ---------------------------------------------------------------------------


def kg_model_cache_dir() -> Path:
    """Return the system-wide embedding model cache root.

    Default: ``~/.kgrag/models/``
    Override: set ``KGRAG_MODEL_DIR`` environment variable.

    All KG modules should resolve model paths through this function so that a
    single env-var change redirects every module's cache at once.

    :return: Absolute :class:`~pathlib.Path` to the model cache directory.
    """
    env = os.environ.get("KGRAG_MODEL_DIR")
    if env:
        return Path(env).resolve()
    return Path.home() / ".kgrag" / "models"


def resolve_model_path(model_name: str, local_fallback: Path | None = None) -> Path:
    """Return the local cache path for *model_name*.

    Checks the system-wide cache (``kg_model_cache_dir()``) first.  If
    *local_fallback* is provided and the system cache env var is not set, uses
    that instead — allowing standalone modules to keep their own local cache
    while respecting a global override.

    The model name is stored as ``<org>/<model>`` directory structure (matching
    HuggingFace layout), e.g. ``BAAI/bge-small-en-v1.5`` →
    ``~/.kgrag/models/BAAI/bge-small-en-v1.5/``.

    :param model_name: HuggingFace model identifier or known alias.
    :param local_fallback: Per-module fallback directory (used when
        ``KGRAG_MODEL_DIR`` is not set).
    :return: Absolute :class:`~pathlib.Path` to the model directory.
    """
    resolved = KNOWN_MODELS.get(model_name, model_name)
    if os.environ.get("KGRAG_MODEL_DIR") or local_fallback is None:
        return kg_model_cache_dir() / resolved.replace("/", os.sep)
    return local_fallback / resolved.replace("/", "--")
