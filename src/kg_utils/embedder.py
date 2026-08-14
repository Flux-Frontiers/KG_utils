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

TEIEmbedder
    Concrete implementation backed by a remote HuggingFace Text Embeddings
    Inference server.  Stdlib HTTP only — no torch, no numpy — so it is the
    one embedder usable from a core (zero-dependency) install.

resolve_device(device)
    Resolve the embedding device: explicit arg > ``KG_EMBED_DEVICE`` env >
    auto-detect. Public so callers can gate parallelism decisions (e.g.
    ``kg_utils.corpus_embedder.CorpusEmbedder``'s GPU-can't-fan-out guard) on
    the resolved device *before* loading a model.

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
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(device: str | None = None) -> str | None:
    """Resolve the embedding device: explicit arg > ``KG_EMBED_DEVICE`` env > auto-detect.

    The env channel lets ``spawn``-based embedding workers (which inherit
    ``os.environ`` but can't easily receive a Python arg) be pinned to e.g.
    CPU — without it, N parallel workers each auto-select MPS and stack N
    GPU allocations into an OOM. This is what makes CPU multiprocessing
    embedding safe on Apple Silicon, and it's why callers that need to gate
    fan-out decisions (parallel vs. single-process) resolve the device via
    this function *before* constructing a model.

    :param device: Explicit device override (``"cpu"``/``"mps"``/``"cuda"``),
        or ``None``.
    :return: Resolved device string, or ``None`` if torch is unavailable (in
        which case callers should treat it as "let the loader decide").
    """
    sel = (device or os.environ.get("KG_EMBED_DEVICE", "")).strip().lower()
    if sel:
        return sel
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    try:
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except AttributeError:
        return "cpu"


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
        # transformers >=5 dropped the ``transformers.logging`` submodule alias;
        # ``transformers.utils.logging`` is importable on both 4.x and 5.x.
        hf_logging = importlib.import_module("transformers.utils.logging")

        hf_logging.set_verbosity_error()
        # TQDM_DISABLE alone misses transformers' _tqdm_active gate
        hf_logging.disable_progress_bar()
    except (ImportError, ValueError):
        pass

    os.environ["TQDM_DISABLE"] = "1"

    # torch is a hard requirement of this function (SentenceTransformer needs
    # it below); resolve_device()'s own ImportError guard never fires here.
    device = resolve_device(device) or "cpu"

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
            hf_logging = importlib.import_module("transformers.utils.logging")

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
# Remote embedder — HuggingFace Text Embeddings Inference (TEI)
# ---------------------------------------------------------------------------

#: Default TEI base URL.  TEI listens on port 80 inside its container; the
#: fleet convention is to publish that on 8080.
DEFAULT_TEI_ENDPOINT: str = "http://localhost:8080"

#: TEI's own default ``--max-client-batch-size``.  Used as the assumed server
#: limit when :class:`TEIEmbedder` is constructed without probing: a request
#: carrying more items than the server allows is rejected outright with HTTP
#: 422, so the safe assumption is TEI's conservative default rather than the
#: fleet's :data:`DEFAULT_ENCODE_BATCH` of 128.
TEI_DEFAULT_CLIENT_BATCH: int = 32

#: Statuses worth retrying.  429 is not hypothetical: TEI sheds load with it
#: (rather than queueing) as soon as in-flight requests exceed its capacity,
#: so any client fanning out concurrent batches will meet it.
_TEI_RETRY_STATUS: frozenset[int] = frozenset({429, 502, 503, 504})


class TEIEmbedder(Embedder):
    """Embedder backed by a remote HuggingFace Text Embeddings Inference server.

    Speaks TEI's native ``POST /embed`` API over stdlib HTTP.  Nothing here
    imports torch, sentence-transformers or numpy, so this is the only embedder
    in the package that works from a **core (zero-dependency) install** — the
    model runs in the server process, not this one.

    Vectors are requested with ``normalize=True`` to match the fleet contract
    (:class:`SentenceTransformerEmbedder` always passes
    ``normalize_embeddings=True``), and with ``truncate=True`` so
    over-long inputs are clipped at the model's max sequence length the way
    sentence-transformers does silently — without it TEI rejects the whole
    batch.

    Two failure modes are handled explicitly because both were observed while
    benchmarking this backend:

    * **Server batch ceiling.**  TEI refuses a request carrying more than
      ``max_client_batch_size`` items (HTTP 422), whose default is 32 — well
      below the fleet's 128-item convention.  The limit is read from ``/info``
      when probing and every call is re-chunked to respect it, so a caller
      passing ``encode_batch_size=128`` is split transparently rather than
      failing.
    * **Load shedding.**  A saturated TEI returns 429 instead of queueing.
      Those, plus 502/503/504 and transport errors, are retried with bounded
      exponential backoff; everything else fails loudly.

    Nothing is silently substituted on failure.  A wrong-dimension or
    partially-written result would corrupt a vector store far more expensively
    than an exception, so both are raised.

    :param endpoint: TEI base URL.  Falls back to ``KG_EMBED_ENDPOINT`` then
        :data:`DEFAULT_TEI_ENDPOINT`.  A trailing ``/`` is trimmed, as is a
        trailing ``/v1`` (TEI's native routes live at the root; ``/v1`` is its
        OpenAI-compatible alias).
    :param dim: Embedding dimension.  When given, construction performs **no
        network I/O at all** — useful for offline construction and for tests.
        When ``None`` the server is probed once (see :meth:`probe`).
    :param api_key: Bearer token.  Falls back to ``KG_EMBED_API_KEY``.  Not
        required by a stock TEI, which is unauthenticated.
    :param model_name: Informational label for :attr:`model_name` / ``repr``.
        The served model is whatever the server was started with; this cannot
        change it.
    :param timeout: Per-request timeout in seconds.
    :param max_retries: Retry attempts for retryable statuses and transport
        errors.  ``0`` disables retrying.
    :param max_batch: Override the server's reported ``max_client_batch_size``.
        Defaults to the probed value, or :data:`TEI_DEFAULT_CLIENT_BATCH` when
        not probing.
    :raises RuntimeError: If probing is required and the server is unreachable
        or answers unusably.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        dim: int | None = None,
        api_key: str | None = None,
        model_name: str = "",
        timeout: float = 120.0,
        max_retries: int = 3,
        max_batch: int | None = None,
    ) -> None:
        raw = (endpoint or os.environ.get("KG_EMBED_ENDPOINT") or DEFAULT_TEI_ENDPOINT).strip()
        self.endpoint: str = raw.rstrip("/").removesuffix("/v1")

        env_key = os.environ.get("KG_EMBED_API_KEY")
        self.api_key: str = (api_key if api_key is not None else env_key) or ""
        self.model_name: str = model_name
        self.timeout: float = timeout
        self.max_retries: int = max_retries
        self.max_batch: int = max_batch or TEI_DEFAULT_CLIENT_BATCH

        if dim is None:
            info = self.probe()
            if max_batch is None:
                self.max_batch = int(info.get("max_client_batch_size") or TEI_DEFAULT_CLIENT_BATCH)
            if not self.model_name:
                self.model_name = str(info.get("model_id") or "")
            self.dim = int(info["dim"])
        else:
            self.dim = int(dim)

    # -- HTTP -------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        """POST *payload* as JSON to *path* and return the decoded response.

        Retries :data:`_TEI_RETRY_STATUS` and transport errors with exponential
        backoff; any other error is raised immediately.

        :param path: Route below the base URL, e.g. ``"/embed"``.
        :param payload: JSON-serialisable request body.
        :return: Decoded JSON response.
        :raises RuntimeError: On a non-retryable error, or once retries are
            exhausted.
        """
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last = ""
        for attempt in range(self.max_retries + 1):
            try:
                req = Request(self.endpoint + path, data=body, headers=headers)
                with urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - fixed http(s) base
                    return json.loads(resp.read())
            except HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")[:200]
                except (OSError, ValueError):
                    detail = ""
                last = f"HTTP {exc.code} from {path}: {detail or exc.reason}"
                if exc.code not in _TEI_RETRY_STATUS:
                    raise RuntimeError(f"TEI request failed — {last}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                last = f"{type(exc).__name__} contacting {self.endpoint}{path}: {exc}"

            if attempt < self.max_retries:
                time.sleep(2.0**attempt)

        raise RuntimeError(f"TEI request failed after {self.max_retries + 1} attempts — {last}")

    def probe(self) -> dict[str, Any]:
        """Query the server for its limits and embedding dimension.

        ``GET /info`` reports ``max_client_batch_size`` and ``max_input_length``
        but **not** the embedding dimension, so the dimension is measured by
        embedding one short string.  Both facts are cached by ``__init__``;
        nothing on the hot path re-probes, because
        :class:`~kg_utils.vector_backend.VectorBackend` needs ``dim`` at
        table-creation time and must never wait on a network round trip.

        :return: ``{"dim": int, "max_client_batch_size": int|None,
            "model_id": str|None, "max_input_length": int|None}``.
        :raises RuntimeError: If the server is unreachable or returns no vector.
        """
        info: dict[str, Any] = {}
        try:
            req = Request(self.endpoint + "/info", headers={"Accept": "application/json"})
            with urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                info = json.loads(resp.read())
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            # /info is advisory: an older or proxied deployment may not expose
            # it.  The dimension probe below is the part that must succeed.
            info = {}

        vecs = self._post(
            "/embed", {"inputs": ["dimension probe"], "normalize": True, "truncate": True}
        )
        if not isinstance(vecs, list) or not vecs or not isinstance(vecs[0], list):
            raise RuntimeError(
                f"TEI at {self.endpoint} returned no usable vector when probing "
                f"the embedding dimension; got {type(vecs).__name__}."
            )
        return {
            "dim": len(vecs[0]),
            "max_client_batch_size": info.get("max_client_batch_size"),
            "model_id": info.get("model_id"),
            "max_input_length": info.get("max_input_length"),
        }

    # -- Embedder contract ------------------------------------------------

    def embed_texts(
        self, texts: list[str], encode_batch_size: int = DEFAULT_ENCODE_BATCH
    ) -> list[list[float]]:
        """Embed a list of strings into float32 vectors.

        The request batch is ``min(encode_batch_size, max_batch)`` — the server
        enforces its own ceiling and rejects anything above it, so the caller's
        value is treated as an upper bound rather than an instruction.  Unlike
        the in-process embedders there is no ``batch x seq^2`` memory cliff on
        this side of the wire; batching here is purely about request size.

        :param texts: Input strings.
        :param encode_batch_size: Requested per-request batch (default
            :data:`DEFAULT_ENCODE_BATCH`), clamped to the server's limit.
        :return: One float32 vector per input, in input order.
        :raises RuntimeError: If the server errors, or returns the wrong number
            of vectors or a vector of unexpected width.
        """
        if not texts:
            return []

        step = max(1, min(encode_batch_size, self.max_batch))
        out: list[list[float]] = []
        for start in range(0, len(texts), step):
            chunk = texts[start : start + step]
            vecs = self._post("/embed", {"inputs": chunk, "normalize": True, "truncate": True})
            if not isinstance(vecs, list) or len(vecs) != len(chunk):
                raise RuntimeError(
                    f"TEI returned {len(vecs) if isinstance(vecs, list) else '?'} vectors "
                    f"for {len(chunk)} inputs at offset {start}."
                )
            for vec in vecs:
                if len(vec) != self.dim:
                    raise RuntimeError(
                        f"TEI returned a {len(vec)}-dim vector but this embedder is "
                        f"configured for {self.dim}. Refusing to write mixed-dimension "
                        f"vectors — check that {self.endpoint} serves the expected model."
                    )
                out.append([float(x) for x in vec])
        return out

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a float32 vector.

        :param query: Query string.
        :return: Float32 vector.
        """
        return self.embed_texts([query])[0]

    def __repr__(self) -> str:
        label = f", model={self.model_name!r}" if self.model_name else ""
        return f"TEIEmbedder(endpoint={self.endpoint!r}{label}, dim={self.dim})"


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
