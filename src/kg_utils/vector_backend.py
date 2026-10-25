"""kg_utils/vector_backend.py — the vector store under SemanticIndex.

The :class:`VectorBackend` protocol is the storage seam under
:class:`kg_utils.semantic.SemanticIndex` (and doc_kg's heavier subclass), and
:class:`SqliteVecBackend` is its one implementation: an exact brute-force store
built on ``sqlite-vec`` (``vec0``).  Vectors live in a ``vec0`` virtual table
row-aligned by ``rowid`` to a plain ``vec_meta`` table that carries the
filterable metadata columns, so a SQL ``where`` clause compiles to a true
prefilter.  Exact (recall 1.0) at interactive latency on the corpora we serve.

The backend hardcodes no domain columns: the owning ``SemanticIndex`` declares
its ``meta_columns`` (``("kind","name","qualname","module_path")`` for code,
``("kind","name","title","file_path")`` for documents) and the backend persists
exactly those.  A row is a dict ``{id, <meta_columns...>, text, vector}``;
``text`` is dropped, because hydration comes from the authoritative SQLite
GraphStore.

Removed in 0.29.0: ``LanceDBBackend``, ``resolve_backend_name()`` and
``make_backend()``.  Nothing in the fleet had built a LanceDB index since the
sqlite-vec migration finished; the backend survived only as a reader for
un-migrated stores, and ``dockg convert-index`` was its last caller.  The
protocol stays, so a future backend has the same seam to fill.

Optional dependencies
---------------------
  sqlite-vec — SqliteVecBackend (``pip install 'kgmodule-utils[sqlite-vec]'``)
  numpy      — vector arrays

Author: Eric G. Suchanek, PhD
License: Elastic-2.0
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_LOG = logging.getLogger(__name__)


class VectorStoreNotFoundError(FileNotFoundError):
    """Raised when a vector store is read before anything has built it.

    Reading a store that is not there used to surface as a bare
    ``sqlite3.OperationalError("unable to open database file")`` when the
    parent directory was missing, and as ``no such table: vec_meta`` further
    downstream when the directory existed but the file did not. Neither names
    the file or says what creates it.

    Subclasses :class:`FileNotFoundError` so callers already catching
    :class:`OSError` are unaffected, and so a caller that would rather degrade
    -- capture the graph metrics and skip the vector-derived ones -- can catch
    this specific case instead of guessing at a driver error.
    """


@runtime_checkable
class VectorBackend(Protocol):
    """Storage seam for a semantic vector index.

    A backend owns the vector store lifecycle: table creation/wipe, batched
    upserts, id-scoped deletes, membership listing, kNN search with an optional
    SQL prefilter, and a row count.  It is agnostic to what the metadata columns
    *mean* — the owning index declares them.
    """

    def open(self, *, wipe: bool = False) -> None:
        """Open (creating if needed) the store; ``wipe`` drops existing vectors first."""
        ...

    def upsert(self, rows: Sequence[dict[str, Any]], *, batch_size: int = 1000) -> int:
        """Insert or replace rows; returns the number written."""
        ...

    def delete_ids(self, ids: Iterable[str]) -> int:
        """Delete rows by node id; returns the number removed."""
        ...

    def existing_ids(self) -> set[str]:
        """Return the set of node ids currently stored (empty if the store is absent)."""
        ...

    def search(
        self, qvec: Sequence[float], k: int, *, where: str | None = None
    ) -> list[dict[str, Any]]:
        """Cosine kNN; each result dict carries id, the meta columns, and ``_distance``.

        ``where`` is a SQL predicate over the metadata columns, applied as a true
        prefilter (the k nearest are drawn from the matching subset).
        """
        ...

    def count(self) -> int:
        """Return the number of stored vectors."""
        ...


# ---------------------------------------------------------------------------
# sqlite-vec
# ---------------------------------------------------------------------------


class SqliteVecBackend:
    """Exact brute-force vector store on ``sqlite-vec`` (``vec0``).

    Layout (a sidecar ``vectors.sqlite``):

      * ``vec_meta(id TEXT PRIMARY KEY, <meta_columns...>)`` — one row per
        vector, carrying the filterable metadata; SQLite assigns the integer
        ``rowid``.
      * ``vec_nodes USING vec0(embedding <dtype>[dim] distance_metric=cosine)``
        — row-aligned to ``vec_meta`` by the *same* ``rowid``.

    A ``search(where=...)`` compiles to
    ``... WHERE embedding MATCH ? AND k = ? AND rowid IN
    (SELECT rowid FROM vec_meta WHERE <where>) ORDER BY distance`` — a true
    prefilter.  Cosine distance is scale-invariant, so vectors are stored as
    given (no re-normalisation); ``_distance`` is the cosine distance, the
    convention every consumer's score conversion already assumes.

    :param db_path: Path to the sqlite file (e.g. ``.dockg/vectors.sqlite``).
    :param dim: Embedding dimensionality.
    :param meta_columns: Metadata columns to persist (all stored as ``TEXT``).
    :param dtype: ``"float"`` (fp32) or ``"int8"`` (3× smaller; assumes unit-norm
        inputs scaled by 127 — see :meth:`upsert`).
    :param check_same_thread: Passed to :func:`sqlite3.connect`. Set ``False``
        for a read-mostly store shared across worker threads (a serving handler);
        SQLite's default serialized mode then makes concurrent reads safe.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        dim: int = 384,
        meta_columns: Sequence[str] = ("kind", "name", "qualname", "module_path"),
        dtype: str = "float",
        check_same_thread: bool = True,
    ) -> None:
        if dtype not in ("float", "int8"):
            raise ValueError(f"dtype must be 'float' or 'int8', got {dtype!r}")
        self.db_path = Path(db_path)
        self.dim = int(dim)
        self.meta_columns = tuple(meta_columns)
        self.dtype = dtype
        self.check_same_thread = check_same_thread
        self._conn: Any = None
        self._fresh = False

    # -- lifecycle ------------------------------------------------------

    def _connect(self) -> Any:
        import importlib  # pylint: disable=import-outside-toplevel
        import sqlite3  # pylint: disable=import-outside-toplevel

        try:
            sqlite_vec = importlib.import_module("sqlite_vec")
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SqliteVecBackend requires sqlite-vec: pip install 'kgmodule-utils[sqlite-vec]'"
            ) from exc

        conn = sqlite3.connect(str(self.db_path), check_same_thread=self.check_same_thread)
        try:
            conn.enable_load_extension(True)
        except AttributeError as exc:  # pragma: no cover — stock system Python
            raise RuntimeError(
                "This Python's sqlite3 was built without extension loading; "
                "SqliteVecBackend needs it (a Poetry/pyenv build works)."
            ) from exc
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def open(self, *, wipe: bool = False) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        if wipe:
            conn.execute("DROP TABLE IF EXISTS vec_nodes")
            conn.execute("DROP TABLE IF EXISTS vec_meta")
            # sync() state describes the vectors just dropped.
            conn.execute(f"DROP TABLE IF EXISTS {SYNC_HASH_TABLE}")
            conn.execute(f"DROP TABLE IF EXISTS {SYNC_INFO_TABLE}")
        existed = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_meta'"
            ).fetchone()
        )
        cols = ", ".join(f"{c} TEXT" for c in self.meta_columns)
        conn.execute(f"CREATE TABLE IF NOT EXISTS vec_meta(id TEXT PRIMARY KEY, {cols})")
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0("
            f"embedding {self.dtype}[{self.dim}] distance_metric=cosine)"
        )
        conn.commit()
        self._conn = conn
        # A freshly created (or wiped) store has no rows to replace — skip the
        # per-batch dedup delete during the build.
        self._fresh = wipe or not existed

    def _c(self) -> Any:
        """Return the lazy read connection, refusing to invent a missing store.

        ``sqlite3.connect`` creates whatever file it is pointed at, so without
        this check a mistyped path yields an empty database and a confusing
        failure later. :meth:`open` is the path that legitimately creates the
        store and does not come through here.

        :return: The open SQLite connection.
        :raises VectorStoreNotFoundError: If the store file does not exist.
        """
        if self._conn is None:
            if not self.db_path.exists():
                raise VectorStoreNotFoundError(
                    f"Vector store not found: {self.db_path}. Nothing has written it -- "
                    "build the knowledge graph first, or call open() to create an "
                    "empty store."
                )
            self._conn = self._connect()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- encoding -------------------------------------------------------

    def _encode(self, vec: Sequence[float]) -> bytes:
        import numpy as np  # pylint: disable=import-outside-toplevel

        arr = np.asarray(vec, dtype=np.float32)
        if self.dtype == "int8":
            return np.clip(np.round(arr * 127.0), -128, 127).astype(np.int8).tobytes()
        return arr.tobytes()

    @property
    def _match_expr(self) -> str:
        # int8 blobs MUST be wrapped or sqlite-vec parses them as float32.
        return "vec_int8(?)" if self.dtype == "int8" else "?"

    # -- writes ---------------------------------------------------------

    def upsert(self, rows: Sequence[dict[str, Any]], *, batch_size: int = 1000) -> int:
        conn = self._c()
        meta_cols = ("id", *self.meta_columns)
        meta_sql = (
            f"INSERT INTO vec_meta({', '.join(meta_cols)}) "
            f"VALUES ({', '.join('?' for _ in meta_cols)})"
        )
        vec_sql = f"INSERT INTO vec_nodes(rowid, embedding) VALUES (?, {self._match_expr})"
        written = 0
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            if not chunk:
                continue
            ids = [r["id"] for r in chunk]
            with conn:
                # upsert = delete any prior rows for these ids, then insert.
                # Skipped on a freshly wiped/created store (nothing to replace).
                if not self._fresh:
                    self._delete_ids_conn(conn, ids)
                meta_rows = [
                    tuple(r.get(c) if c != "id" else r["id"] for c in meta_cols) for r in chunk
                ]
                conn.executemany(meta_sql, meta_rows)
                # map id -> freshly assigned rowid, preserving chunk order
                placeholders = ", ".join("?" for _ in ids)
                rowmap = {
                    rid: rw
                    for rid, rw in conn.execute(
                        f"SELECT id, rowid FROM vec_meta WHERE id IN ({placeholders})", ids
                    ).fetchall()
                }
                vec_rows = [(rowmap[r["id"]], self._encode(r["vector"])) for r in chunk]
                conn.executemany(vec_sql, vec_rows)
            written += len(chunk)
        return written

    def _delete_ids_conn(self, conn: Any, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        old = [
            r[0]
            for r in conn.execute(
                f"SELECT rowid FROM vec_meta WHERE id IN ({placeholders})", list(ids)
            ).fetchall()
        ]
        if old:
            rp = ", ".join("?" for _ in old)
            conn.execute(f"DELETE FROM vec_nodes WHERE rowid IN ({rp})", old)
            conn.execute(f"DELETE FROM vec_meta WHERE id IN ({placeholders})", list(ids))
        return len(old)

    def delete_ids(self, ids: Iterable[str]) -> int:
        conn = self._c()
        ids = [i for i in ids if i]
        removed = 0
        with conn:
            for start in range(0, len(ids), 1000):
                removed += self._delete_ids_conn(conn, ids[start : start + 1000])
        return removed

    # -- sync state -----------------------------------------------------
    #
    # Two sidecar tables that only :meth:`SemanticIndex.sync` reads and
    # writes. They are separate from ``vec_meta`` on purpose: every consumer
    # shares that table and some read it directly, so a full build stays
    # exactly what it was. A store that has never been synced has neither.

    def load_sync_state(self) -> tuple[str, dict[str, str]]:
        """Return the stored embedder signature and per-id index-text hashes.

        :return: ``(signature, {node_id: text_sha256})``; ``("", {})`` when
            the store has never been synced.
        """
        conn = self._c()
        have = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?)",
                (SYNC_HASH_TABLE, SYNC_INFO_TABLE),
            ).fetchall()
        }
        if have != {SYNC_HASH_TABLE, SYNC_INFO_TABLE}:
            return "", {}
        row = conn.execute(
            f"SELECT value FROM {SYNC_INFO_TABLE} WHERE key = 'signature'"
        ).fetchone()
        hashes = dict(conn.execute(f"SELECT id, sha FROM {SYNC_HASH_TABLE}").fetchall())
        return (row[0] if row else ""), hashes

    def save_sync_state(
        self, signature: str, written: dict[str, str], removed: Iterable[str]
    ) -> None:
        """Record the embedder signature and update the per-id hashes.

        :param signature: Embedder identity the stored vectors were made with.
        :param written: ``{node_id: text_sha256}`` for ids embedded this run.
        :param removed: Ids whose vectors were deleted this run.
        """
        conn = self._c()
        with conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {SYNC_HASH_TABLE}(id TEXT PRIMARY KEY, sha TEXT NOT NULL)"
            )
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {SYNC_INFO_TABLE}(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                f"INSERT OR REPLACE INTO {SYNC_INFO_TABLE}(key, value) VALUES ('signature', ?)",
                (signature,),
            )
            gone = [i for i in removed if i]
            for start in range(0, len(gone), 1000):
                chunk = gone[start : start + 1000]
                conn.execute(
                    f"DELETE FROM {SYNC_HASH_TABLE} WHERE id IN ({', '.join('?' for _ in chunk)})",
                    chunk,
                )
            conn.executemany(
                f"INSERT OR REPLACE INTO {SYNC_HASH_TABLE}(id, sha) VALUES (?, ?)",
                list(written.items()),
            )

    # -- reads ----------------------------------------------------------

    def existing_ids(self) -> set[str]:
        if not self.db_path.exists():
            return set()
        try:
            conn = self._c()
            return {r[0] for r in conn.execute("SELECT id FROM vec_meta").fetchall()}
        except Exception:  # noqa: BLE001
            return set()

    def iter_vectors(self) -> Iterator[tuple[str, Any]]:
        """Yield ``(id, vector)`` for every stored vector, as float32 arrays.

        An int8 store yields its values divided by 127, the inverse of
        :meth:`_encode`; cosine is scale-invariant, so rankings are the same
        either way. For exporters that copy a store's vectors elsewhere, such
        as :mod:`kg_utils.devicepack`.

        :return: An iterator in ``vec_meta`` rowid order; empty when the
            store does not exist.
        """
        import numpy as np  # pylint: disable=import-outside-toplevel

        if self._conn is None and not self.db_path.exists():
            return
        dtype = np.int8 if self.dtype == "int8" else np.float32
        rows = self._c().execute(
            "SELECT m.id, v.embedding FROM vec_meta m "
            "JOIN vec_nodes v ON v.rowid = m.rowid ORDER BY m.rowid"
        )
        for node_id, blob in rows:
            vector = np.frombuffer(blob, dtype=dtype).astype(np.float32)
            yield node_id, vector / 127.0 if self.dtype == "int8" else vector

    def count(self) -> int:
        # Don't create a fresh db file just to count a store that isn't there.
        if self._conn is None and not self.db_path.exists():
            return 0
        try:
            (n,) = self._c().execute("SELECT COUNT(*) FROM vec_nodes").fetchone()
            return int(n)
        except Exception:  # noqa: BLE001
            return 0

    def search(
        self, qvec: Sequence[float], k: int, *, where: str | None = None
    ) -> list[dict[str, Any]]:
        conn = self._c()
        meta_select = ", ".join(f"vec_meta.{c}" for c in self.meta_columns)
        sql = (
            f"SELECT vec_meta.id, {meta_select}, vec_nodes.distance "
            "FROM vec_nodes JOIN vec_meta ON vec_meta.rowid = vec_nodes.rowid "
            f"WHERE embedding MATCH {self._match_expr} AND k = ? "
        )
        params: list[Any] = [self._encode(qvec), int(k)]
        if where:
            sql += f"AND vec_nodes.rowid IN (SELECT rowid FROM vec_meta WHERE {where}) "
        sql += "ORDER BY distance"
        out: list[dict[str, Any]] = []
        for row in conn.execute(sql, params).fetchall():
            rec: dict[str, Any] = {"id": row[0]}
            for i, col in enumerate(self.meta_columns, start=1):
                rec[col] = row[i]
            rec["_distance"] = float(row[-1])
            out.append(rec)
        return out


#: Sidecar tables holding :meth:`SemanticIndex.sync` state (see
#: :meth:`SqliteVecBackend.load_sync_state`).
SYNC_HASH_TABLE = "vec_sync_hash"
SYNC_INFO_TABLE = "vec_sync_info"


def _escape(s: str) -> str:
    return s.replace("'", "''")
