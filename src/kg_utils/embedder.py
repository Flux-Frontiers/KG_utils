"""kg_utils.embedder — Concrete SentenceTransformer embedding for the KGModule stack.

All model-loading logic lives here so that the ``local_files_only`` guard,
KNOWN_MODELS alias resolution, and path convention are defined exactly once.
Every KG module (doc_kg, diary_kg, code_kg, …) imports from here instead of
reimplementing the load sequence.

Contents
--------
Embedder
    Abstract base class with ``embed_texts`` + ``embed_query`` + ``dim``.

SentenceTransformerEmbedder
    Concrete implementation.  Always uses ``local_files_only=True`` when the
    model is cached locally — prevents HuggingFace HEAD requests that leave
    stale thread/network state and cause SIGBUS on MPS.

load_sentence_transformer(model_name)
    Raw ``SentenceTransformer`` factory with the canonical safe-load sequence.
    Use when you need the bare model object (e.g. multi-process workers that
    each load their own copy by name).

get_embedder(model_name)
    High-level factory returning a ready-to-use ``SentenceTransformerEmbedder``.

wrap_embedder(st_model, model_name)
    Wrap an already-loaded ``SentenceTransformer`` as an ``Embedder``.  Use
    this to share a live model between pipeline stages (e.g. DiaryTransformer
    → DocKG) without loading a second copy on MPS/CUDA.

Author: Eric G. Suchanek, PhD
License: Elastic 2.0
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from kg_utils.embed import DEFAULT_MODEL, KNOWN_MODELS, resolve_model_path

#: Default per-call encode batch fed to ``model.encode(batch_size=...)``.
#: Transformer attention memory scales with ``batch x seq^2``, so a large batch
#: on long (near-max-sequence) chunks can allocate many GB per call and OOM /
#: stall MPS.  Throughput is flat above ~128 on both CPU and MPS for the models
#: in use, so 128 is the safe default; raise it only for a large-VRAM CUDA GPU
#: with short sequences.
DEFAULT_ENCODE_BATCH: int = 128

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Embedder:
    """Abstract embedding backend for the KGModule stack.

    :param dim: Embedding dimension — set by concrete ``__init__``.
    """

    dim: int

    def embed_texts(
        self, texts: list[str], encode_batch_size: int = DEFAULT_ENCODE_BATCH
    ) -> list[list[float]]:
        """Embed a list of strings into float32 vectors.

        :param texts: Input strings.
        :param encode_batch_size: Per-call ``model.encode`` batch (default
            :data:`DEFAULT_ENCODE_BATCH`); memory scales with ``batch x seq^2``.
        :return: One float32 vector per input.
        """
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        :param query: Query string.
        :return: Float32 vector.
        """
        return self.embed_texts([query])[0]


# ---------------------------------------------------------------------------
# Canonical model loader
# ---------------------------------------------------------------------------


def load_sentence_transformer(model_name: str = DEFAULT_MODEL, device: str | None = None) -> Any:
    """Load a ``SentenceTransformer`` with the canonical safe-load sequence.

    Resolution order:

    1. Resolve KNOWN_MODELS alias → HuggingFace repo ID.
    2. If ``resolve_model_path()`` returns an existing directory, load from
       the local path with ``local_files_only=True`` — no HF HEAD requests.
    3. Otherwise try ``local_files_only=True`` (hits HF's own cache layout).
    4. Fall back to a live network fetch only if the model is genuinely absent.

    The ``local_files_only=True`` guard on step 2 is critical on MPS: HF HEAD
    retry loops leave stale thread state that causes SIGBUS on the first
    ``encode()`` call.

    Device precedence: explicit *device* arg > ``KG_EMBED_DEVICE`` env >
    auto-detect.  The env var exists because ``spawn``-based embedding workers
    inherit ``os.environ`` but can't easily receive a Python arg — without a way
    to pin the device, each worker auto-selects MPS and N parallel workers stack
    N GPU allocations into an OOM.  So CPU multiprocessing embedding on Apple
    Silicon is only safe with this knob.

    :param model_name: HuggingFace model ID or KNOWN_MODELS alias.
    :param device: Explicit device (``"cpu"``/``"mps"``/``"cuda"``).  ``None``
        falls back to ``KG_EMBED_DEVICE`` then CUDA→MPS→CPU auto-detect.
    :return: Loaded ``SentenceTransformer`` instance.
    """
    SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer

    try:
        hf_logging = importlib.import_module("transformers.logging")

        hf_logging.set_verbosity_error()
        # TQDM_DISABLE alone misses transformers' _tqdm_active gate
        hf_logging.disable_progress_bar()
    except (ImportError, ValueError):
        pass

    os.environ["TQDM_DISABLE"] = "1"

    torch = importlib.import_module("torch")

    # Device precedence: explicit arg > KG_EMBED_DEVICE env > auto-detect.
    # The env var lets spawn-based embedding workers (which inherit os.environ
    # but can't easily receive a Python arg) pin to e.g. CPU — without it each
    # worker auto-selects MPS and N parallel workers stack N GPU allocations →
    # MPS OOM. The override is why CPU multiprocessing embedding is safe on
    # Apple Silicon.
    sel = (device or os.environ.get("KG_EMBED_DEVICE", "")).strip().lower()
    if sel:
        device = sel
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except AttributeError:
            device = "cpu"

    resolved = KNOWN_MODELS.get(model_name, model_name)
    trust_remote = "nomic-ai/" in resolved
    local_path = resolve_model_path(resolved)

    if local_path.exists():
        model = SentenceTransformer(
            str(local_path),
            local_files_only=True,
            trust_remote_code=trust_remote,
            device=device,
        )
    else:
        try:
            model = SentenceTransformer(
                resolved,
                local_files_only=True,
                trust_remote_code=trust_remote,
                device=device,
            )
        except OSError:
            model = SentenceTransformer(resolved, trust_remote_code=trust_remote, device=device)

    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Concrete embedder
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder(Embedder):
    """Concrete embedder backed by ``sentence-transformers``.

    Delegates model loading to :func:`load_sentence_transformer` so the
    ``local_files_only`` guard is always in effect.

    :param model_name: HuggingFace model ID or KNOWN_MODELS alias.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            hf_logging = importlib.import_module("transformers.logging")

            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
        except (ImportError, ValueError):
            pass

        _prev = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:
            self.model = load_sentence_transformer(model_name)
        finally:
            if _prev is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = _prev

        self.model_name: str = KNOWN_MODELS.get(model_name, model_name)
        # ST ≥5.4 renamed get_embedding_dimension; ≤5.3 had get_sentence_embedding_dimension.
        _dim_fn = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension", None
        )
        self.dim: int = (_dim_fn() if _dim_fn is not None else None) or 384

    def embed_texts(
        self, texts: list[str], encode_batch_size: int = DEFAULT_ENCODE_BATCH
    ) -> list[list[float]]:
        """Embed a list of strings into float32 vectors.

        :param texts: Input strings.
        :param encode_batch_size: Per-call ``model.encode`` batch (default
            :data:`DEFAULT_ENCODE_BATCH`).  Memory scales with ``batch x seq^2``;
            tune down further if OOM on MPS, up only for large-VRAM CUDA.
        """
        np = importlib.import_module("numpy")

        vecs = self.model.encode(
            texts,
            batch_size=encode_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [np.asarray(v, dtype="float32").tolist() for v in vecs]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a float32 vector."""
        np = importlib.import_module("numpy")

        vec = self.model.encode([query], normalize_embeddings=True)[0]
        return list(np.asarray(vec, dtype="float32").tolist())

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedder(model={self.model_name!r}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def get_embedder(model_name: str = DEFAULT_MODEL) -> SentenceTransformerEmbedder:
    """Return a ready-to-use :class:`SentenceTransformerEmbedder`.

    :param model_name: HuggingFace model ID or KNOWN_MODELS alias.
    :return: Configured embedder instance.
    """
    return SentenceTransformerEmbedder(model_name)


def wrap_embedder(st_model: Any, model_name: str = DEFAULT_MODEL) -> Embedder:
    """Wrap an already-loaded ``SentenceTransformer`` as an :class:`Embedder`.

    Use this when a live model is already on the GPU (e.g. DiaryTransformer →
    DocKG handoff) to avoid loading a second copy on MPS/CUDA.

    :param st_model: Live ``SentenceTransformer`` instance.
    :param model_name: Model name stored as metadata on the wrapper.
    :return: An :class:`Embedder` that delegates all calls to *st_model*.
    """
    np = importlib.import_module("numpy")

    resolved = KNOWN_MODELS.get(model_name, model_name)
    _dim_fn = getattr(st_model, "get_embedding_dimension", None) or getattr(
        st_model, "get_sentence_embedding_dimension", None
    )
    _dim = (_dim_fn() if _dim_fn is not None else None) or 384

    class _WrappedEmbedder(Embedder):
        model_name: str = resolved
        dim: int = _dim

        def embed_texts(
            self, texts: list[str], encode_batch_size: int = DEFAULT_ENCODE_BATCH
        ) -> list[list[float]]:
            vecs = st_model.encode(
                texts,
                batch_size=encode_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [np.asarray(v, dtype="float32").tolist() for v in vecs]

        def embed_query(self, query: str) -> list[float]:
            vec = st_model.encode([query], normalize_embeddings=True)[0]
            return list(np.asarray(vec, dtype="float32").tolist())

    return _WrappedEmbedder()
