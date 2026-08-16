"""kg_utils/semantic.py — sqlite-vec vector index for knowledge graph modules.

SemanticIndex is a derived, disposable layer built from GraphStore nodes.
SQLite (GraphStore) remains the authoritative source of truth.

Model loading, the model registry, and the ``Embedder``/``SentenceTransformerEmbedder``
class hierarchy are NOT defined here — they live in :mod:`kg_utils.embed` (registry,
zero-dependency) and :mod:`kg_utils.embedder` (concrete, device-aware) and are
re-exported below. This module used to carry its own independent copy of both
(pre-dating those two modules); unifying onto them means every consumer of
``SemanticIndex`` now gets ``KG_EMBED_DEVICE`` support and the safer
``trust_remote_code`` gating (only for known ``nomic-ai/`` models, not
unconditionally) for free.

Optional dependencies
---------------------
  sqlite-vec         — vector store backend (the default)
  numpy              — array operations
  sentence-transformers — local embedding model

Install with: pip install 'kgmodule-utils[semantic]'

:class:`LanceDBBackend` is re-exported here for the one caller that still reads
a pre-migration store (``dockg convert-index``), but nothing in this module
constructs it any more — see :meth:`SemanticIndex._get_backend`.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kg_utils.embed import DEFAULT_MODEL, resolve_model_path
from kg_utils.embedder import Embedder, SentenceTransformerEmbedder
from kg_utils.vector_backend import LanceDBBackend, SqliteVecBackend, VectorBackend

__all__ = [
    "DEFAULT_MODEL",
    "Embedder",
    "LanceDBBackend",
    "META_COLUMNS",
    "SeedHit",
    "SemanticIndex",
    "SentenceTransformerEmbedder",
    "SqliteVecBackend",
    "VectorBackend",
    "resolve_model_path",
    "suppress_ingestion_logging",
]

#: Metadata columns persisted alongside each vector for code-shaped KGs.
META_COLUMNS = ("kind", "name", "qualname", "module_path")
_META_COLUMNS = META_COLUMNS

if TYPE_CHECKING:
    from kg_utils.store import GraphStore


def _local_model_path(model_name: str) -> Path:
    """Backward-compat alias for pycode_kg's pre-unification cache convention.

    Resolves *model_name* the same way :func:`kg_utils.embed.resolve_model_path`
    does, but with the CWD-relative ``./.kgcache/models`` fallback this module
    used before it existed — still overridable via ``KGRAG_MODEL_DIR``. New
    code should call :func:`kg_utils.embed.resolve_model_path` directly.

    :param model_name: HuggingFace model identifier or short alias.
    :return: Absolute path to the model directory.
    """
    return resolve_model_path(model_name, local_fallback=Path.cwd() / ".kgcache" / "models")


# ---------------------------------------------------------------------------
# Logging / progress suppression
# ---------------------------------------------------------------------------


def suppress_ingestion_logging() -> None:
    """Suppress verbose progress output during model loading and ingestion."""
    for name in ("sentence_transformers", "transformers", "huggingface_hub", "lancedb", "pylance"):
        logging.getLogger(name).setLevel(logging.WARNING)

    try:
        import transformers  # pylint: disable=import-outside-toplevel

        transformers.logging.set_verbosity_error()  # type: ignore[no-untyped-call]
        transformers.logging.disable_progress_bar()  # type: ignore[no-untyped-call]
    except (ImportError, AttributeError):
        pass

    os.environ["TQDM_DISABLE"] = "1"


# ---------------------------------------------------------------------------
# Seed hit
# ---------------------------------------------------------------------------


@dataclass
class SeedHit:
    """A single result from a semantic vector search.

    :param id: Node ID.
    :param kind: Node kind.
    :param name: Short name.
    :param qualname: Qualified name.
    :param module_path: Repo-relative module path.
    :param distance: Vector distance (lower = more similar).
    :param rank: Zero-based rank in the result list.
    """

    id: str
    kind: str
    name: str
    qualname: str
    module_path: str
    distance: float
    rank: int


# ---------------------------------------------------------------------------
# SemanticIndex
# ---------------------------------------------------------------------------

_DEFAULT_TABLE = "kg_nodes"
_DEFAULT_KINDS = ("module", "class", "function", "method")


class SemanticIndex:
    """sqlite-vec-backed semantic vector index for a knowledge graph.

    Reads nodes from a :class:`~kg_utils.store.GraphStore`, embeds them, and
    stores the vectors in a ``vectors.sqlite`` store.  The index is derived and
    disposable — it can be rebuilt from SQLite at any time without data loss.

    Changed in 0.14.0: the first parameter was ``lancedb_dir``, a *directory*.
    It is now ``vectors_path``, a *file*, and the default backend is
    :class:`SqliteVecBackend` rather than :class:`LanceDBBackend`.  Passing
    ``lancedb_dir=`` raises :exc:`TypeError`, matching how memory-kg 0.7.0,
    Metabo_kg 0.10.0 and diary-kg 0.95.0 each dropped the same parameter.

    :param vectors_path: Path to the ``vectors.sqlite`` store (used only when
        the default backend is constructed; ignored if *backend* is given).
    :param embedder: Embedding backend. Defaults to :class:`SentenceTransformerEmbedder`.
    :param table: Vector table name.
    :param index_kinds: Node kinds to embed.
    :param backend: Vector store backend. Defaults to a :class:`SqliteVecBackend`
        over *vectors_path*.  To read a pre-migration LanceDB store, construct a
        :class:`LanceDBBackend` explicitly and pass it here.
    """

    def __init__(
        self,
        vectors_path: str | Path,
        *,
        embedder: Embedder | None = None,
        table: str = _DEFAULT_TABLE,
        index_kinds: Sequence[str] = _DEFAULT_KINDS,
        backend: VectorBackend | None = None,
    ) -> None:
        self.vectors_path = Path(vectors_path)
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder()
        self.table_name = table
        self.index_kinds = tuple(index_kinds)
        self._backend: VectorBackend | None = backend

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        store: GraphStore,
        *,
        wipe: bool = False,
        batch_size: int = 256,
        quiet: bool = True,
    ) -> dict[str, Any]:
        """Build (or rebuild) the vector index from *store*.

        :param store: Authoritative :class:`~kg_utils.store.GraphStore`.
        :param wipe: If ``True``, delete all existing vectors first.
        :param batch_size: Number of nodes to embed per batch.
        :param quiet: Suppress progress output during ingestion.
        :return: Stats dict with ``indexed_rows``, ``dim``, ``table``, ``vectors_path``, ``kinds``.
        """
        if quiet:
            suppress_ingestion_logging()

        nodes = self._read_nodes(store)
        backend = self._get_backend()
        backend.open(wipe=wipe)

        indexed = 0
        for i in range(0, len(nodes), batch_size):
            chunk = nodes[i : i + batch_size]
            texts = [_build_index_text(n) for n in chunk]
            vecs = self.embedder.embed_texts(texts)

            rows = [
                {
                    "id": n["id"],
                    "kind": n["kind"],
                    "name": n["name"],
                    "qualname": n["qualname"] or "",
                    "module_path": n["module_path"] or "",
                    "text": text,
                    "vector": vec,
                }
                for n, text, vec in zip(chunk, texts, vecs)
            ]
            indexed += backend.upsert(rows, batch_size=len(rows))

        # No-op unless the backend is a LanceDBBackend with ANN enabled.
        if isinstance(backend, LanceDBBackend):
            backend.maybe_create_ann_index(quiet=quiet)

        return {
            "indexed_rows": indexed,
            "dim": self.embedder.dim,
            "table": self.table_name,
            "vectors_path": str(self.vectors_path),
            "kinds": list(self.index_kinds),
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 8, *, where: str | None = None) -> list[SeedHit]:
        """Semantic vector search.

        :param query: Natural-language query string.
        :param k: Number of results to return.
        :param where: Optional SQL predicate over the metadata columns, applied
            as a true prefilter (the ``k`` nearest are drawn from the matching
            subset). Unifies this signature with doc_kg's ``SemanticIndex``.
        :return: List of :class:`SeedHit` ordered by ascending distance.
        """
        backend = self._get_backend()
        qvec = self.embedder.embed_query(query)
        raw = backend.search(qvec, k, where=where)

        hits: list[SeedHit] = []
        for rank, row in enumerate(raw):
            dist = _extract_distance(row, rank)
            hits.append(
                SeedHit(
                    id=row["id"],
                    kind=row.get("kind", ""),
                    name=row.get("name", ""),
                    qualname=row.get("qualname", ""),
                    module_path=row.get("module_path", ""),
                    distance=dist,
                    rank=rank,
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_nodes(self, store: GraphStore) -> list[dict[str, Any]]:
        return store.query_nodes(kinds=list(self.index_kinds))

    def _get_backend(self) -> VectorBackend:
        """Return the vector backend, constructing the default sqlite-vec one lazily.

        Deferred so the embedder's ``dim`` (which may load the model) is only
        touched when the index is actually built or searched.

        Changed in 0.14.0: this used to default to :class:`LanceDBBackend`.
        ``KGModule`` always passes an explicit backend, so the fleet never hit
        that default — but bare use of this class silently created a LanceDB
        store long after the fleet had migrated off it.
        """
        if self._backend is None:
            self._backend = SqliteVecBackend(
                self.vectors_path,
                dim=self.embedder.dim,
                meta_columns=_META_COLUMNS,
            )
        return self._backend

    def __repr__(self) -> str:
        return (
            f"SemanticIndex(vectors_path={self.vectors_path!r}, "
            f"table={self.table_name!r}, embedder={self.embedder!r})"
        )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _build_index_text(n: dict[str, Any]) -> str:
    """Build the canonical text document used for embedding a node.

    Includes KIND, NAME, QUALNAME, MODULE, LINE, DOCSTRING sections plus a
    KEYWORDS section of de-duplicated word tokens from name/qualname/module
    to improve recall for abstract queries.

    :param n: Node dict with keys ``kind``, ``name``, ``qualname``, ``module_path``,
              ``lineno``, and optionally ``docstring``.
    :return: Newline-joined string suitable for embedding.
    """
    parts = [f"KIND: {n['kind']}", f"NAME: {n['name']}"]
    if n.get("qualname"):
        parts.append(f"QUALNAME: {n['qualname']}")
    if n.get("module_path"):
        parts.append(f"MODULE: {n['module_path']}")
    if n.get("lineno") is not None:
        parts.append(f"LINE: {n['lineno']}")
    if n.get("docstring"):
        parts.append("DOCSTRING:\n" + n["docstring"].strip())

    raw = " ".join(filter(None, [n.get("name"), n.get("qualname"), n.get("module_path")]))
    tokens = [w.lower() for w in re.findall(r"[a-zA-Z]+", raw) if len(w) > 2]
    seen_in_doc = set(re.findall(r"[a-zA-Z]+", (n.get("docstring") or "").lower()))
    extra = [t for t in dict.fromkeys(tokens) if t not in seen_in_doc]
    if extra:
        parts.append("KEYWORDS: " + " ".join(extra))

    return "\n".join(parts)


def _extract_distance(row: dict[str, Any], fallback_rank: int) -> float:
    for key in ("_distance", "distance"):
        if key in row and row[key] is not None:
            return float(row[key])
    if "score" in row and row["score"] is not None:
        return 1.0 / (1.0 + float(row["score"]))
    return float(fallback_rank)


def _escape(s: str) -> str:
    return s.replace("'", "''")
