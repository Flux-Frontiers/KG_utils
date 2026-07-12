"""kg_utils/semantic.py — LanceDB vector index for knowledge graph modules.

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
  lancedb            — vector store backend
  numpy              — array operations
  sentence-transformers — local embedding model

Install with: pip install 'kgmodule-utils[semantic]'
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

__all__ = [
    "DEFAULT_MODEL",
    "Embedder",
    "SeedHit",
    "SemanticIndex",
    "SentenceTransformerEmbedder",
    "resolve_model_path",
    "suppress_ingestion_logging",
]

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
    """LanceDB-backed semantic vector index for a knowledge graph.

    Reads nodes from a :class:`~kg_utils.store.GraphStore`, embeds them, and
    stores the vectors in LanceDB.  The index is derived and disposable — it
    can be rebuilt from SQLite at any time without data loss.

    :param lancedb_dir: Directory for the LanceDB database.
    :param embedder: Embedding backend. Defaults to :class:`SentenceTransformerEmbedder`.
    :param table: LanceDB table name.
    :param index_kinds: Node kinds to embed.
    """

    def __init__(
        self,
        lancedb_dir: str | Path,
        *,
        embedder: Embedder | None = None,
        table: str = _DEFAULT_TABLE,
        index_kinds: Sequence[str] = _DEFAULT_KINDS,
    ) -> None:
        self.lancedb_dir = Path(lancedb_dir)
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder()
        self.table_name = table
        self.index_kinds = tuple(index_kinds)
        self._tbl = None

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
        :return: Stats dict with ``indexed_rows``, ``dim``, ``table``, ``lancedb_dir``, ``kinds``.
        """
        if quiet:
            suppress_ingestion_logging()

        nodes = self._read_nodes(store)
        tbl = self._open_table(wipe=wipe)

        indexed = 0
        for i in range(0, len(nodes), batch_size):
            chunk = nodes[i : i + batch_size]
            texts = [_build_index_text(n) for n in chunk]
            vecs = self.embedder.embed_texts(texts)

            ids = [n["id"] for n in chunk]
            if ids:
                pred = " OR ".join([f"id = '{_escape(nid)}'" for nid in ids])
                tbl.delete(pred)

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
            tbl.add(rows)
            indexed += len(rows)

        self._tbl = tbl
        return {
            "indexed_rows": indexed,
            "dim": self.embedder.dim,
            "table": self.table_name,
            "lancedb_dir": str(self.lancedb_dir),
            "kinds": list(self.index_kinds),
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 8) -> list[SeedHit]:
        """Semantic vector search.

        :param query: Natural-language query string.
        :param k: Number of results to return.
        :return: List of :class:`SeedHit` ordered by ascending distance.
        """
        tbl = self._get_table()
        qvec = self.embedder.embed_query(query)
        raw = tbl.search(qvec).limit(k).to_list()

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

    def _open_table(self, *, wipe: bool = False) -> Any:
        import lancedb  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        self.lancedb_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(self.lancedb_dir))

        if self.table_name in db.list_tables().tables:
            if wipe:
                db.drop_table(self.table_name)
            else:
                try:
                    return db.open_table(self.table_name)
                except Exception as exc:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "LanceDB table %r appears corrupt (%s); dropping and recreating.",
                        self.table_name,
                        exc,
                    )
                    db.drop_table(self.table_name)

        dummy = {
            "id": "__dummy__",
            "kind": "dummy",
            "name": "__dummy__",
            "qualname": "",
            "module_path": "",
            "text": "__dummy__",
            "vector": np.zeros((self.embedder.dim,), dtype="float32").tolist(),
        }
        tbl = db.create_table(self.table_name, data=[dummy])
        tbl.delete("id = '__dummy__'")
        return tbl

    def _get_table(self) -> Any:
        if self._tbl is None:
            import lancedb  # pylint: disable=import-outside-toplevel

            db = lancedb.connect(str(self.lancedb_dir))
            self._tbl = db.open_table(self.table_name)
        return self._tbl

    def __repr__(self) -> str:
        return (
            f"SemanticIndex(lancedb_dir={self.lancedb_dir!r}, "
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
