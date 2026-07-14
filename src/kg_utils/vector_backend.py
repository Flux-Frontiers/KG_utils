"""kg_utils/vector_backend.py — pluggable vector store backends for SemanticIndex.

The :class:`VectorBackend` protocol is the storage seam under
:class:`kg_utils.semantic.SemanticIndex` (and doc_kg's heavier subclass).  Two
implementations ship here:

  * :class:`LanceDBBackend` — wraps the historical LanceDB table plumbing
    (dummy-row table creation, ``delete``-then-``add`` upsert, optional IVF ANN
    index gated on row count).  Behaviour is unchanged from the pre-seam code.
  * :class:`SqliteVecBackend` — an exact brute-force store built on
    ``sqlite-vec`` (``vec0``).  Vectors live in a ``vec0`` virtual table
    row-aligned by ``rowid`` to a plain ``vec_meta`` table that carries the
    filterable metadata columns, so a SQL ``where`` clause compiles to a true
    prefilter.  9–11× smaller than LanceDB and exact (recall 1.0) at comparable
    latency on the corpora we serve.

Neither backend hardcodes domain columns: the owning ``SemanticIndex`` declares
its ``meta_columns`` (``("kind","name","qualname","module_path")`` for code,
``("kind","name","title","file_path")`` for documents) and the backend persists
exactly those.  A row is a dict ``{id, <meta_columns...>, text, vector}``; the
sqlite backend drops ``text`` (hydration comes from the authoritative SQLite
GraphStore), LanceDB keeps it for schema compatibility.

Optional dependencies
---------------------
  lancedb    — LanceDBBackend only
  sqlite-vec — SqliteVecBackend only (``pip install 'kgmodule-utils[sqlite-vec]'``)
  numpy      — both

Author: Eric G. Suchanek, PhD
License: Elastic-2.0
Last Revision: 2026-07-14
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_LOG = logging.getLogger(__name__)


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
# LanceDB
# ---------------------------------------------------------------------------


class LanceDBBackend:
    """LanceDB-backed vector store — the historical default.

    Preserves the pre-seam behaviour byte-for-byte: a dummy-row bootstrap for
    table creation, ``delete``-then-``add`` upserts (skipped on a freshly wiped
    table to avoid scanning a growing fragment list), and an optional IVF index
    created once the row count crosses :paramref:`ann_threshold`.  When no index
    exists, ``search`` is an exact flat scan.

    :param lancedb_dir: Directory for the LanceDB database.
    :param table: Table name.
    :param dim: Embedding dimensionality (for the bootstrap dummy row).
    :param meta_columns: Metadata columns to persist alongside ``id``/``vector``.
    :param ann_threshold: Row count at/above which an IVF index is built
        (``0`` disables ANN — always a flat scan).
    :param ann_nprobes: IVF probes at search time (recall/latency knob).
    :param ann_refine_factor: IVF refine factor at search time (``0`` = off).
    :param ann_index_type: ``"IVF_FLAT"`` or ``"IVF_PQ"``.
    :param store_text: Persist the ``text`` column (LanceDB schema compatibility).
    """

    def __init__(
        self,
        lancedb_dir: str | Path,
        *,
        table: str = "kg_nodes",
        dim: int = 384,
        meta_columns: Sequence[str] = ("kind", "name", "qualname", "module_path"),
        ann_threshold: int = 0,
        ann_nprobes: int = 64,
        ann_refine_factor: int = 0,
        ann_index_type: str = "IVF_FLAT",
        store_text: bool = True,
    ) -> None:
        self.lancedb_dir = Path(lancedb_dir)
        self.table_name = table
        self.dim = int(dim)
        self.meta_columns = tuple(meta_columns)
        self.ann_threshold = int(ann_threshold)
        self.ann_nprobes = int(ann_nprobes)
        self.ann_refine_factor = int(ann_refine_factor)
        self.ann_index_type = ann_index_type
        self.store_text = store_text
        self._tbl: Any = None
        self._fresh = False
        self._has_ann = False

    # -- lifecycle ------------------------------------------------------

    def open(self, *, wipe: bool = False) -> None:
        import lancedb  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        self.lancedb_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(self.lancedb_dir))

        if self.table_name in db.list_tables().tables:
            if wipe:
                db.drop_table(self.table_name)
            else:
                try:
                    self._tbl = db.open_table(self.table_name)
                    self._fresh = False
                    return
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning(
                        "LanceDB table %r appears corrupt (%s); dropping and recreating.",
                        self.table_name,
                        exc,
                    )
                    db.drop_table(self.table_name)

        dummy: dict[str, Any] = {"id": "__dummy__"}
        for col in self.meta_columns:
            dummy[col] = "dummy" if col == "kind" else ""
        if self.store_text:
            dummy["text"] = "__dummy__"
        dummy["vector"] = np.zeros((self.dim,), dtype="float32").tolist()

        tbl = db.create_table(self.table_name, data=[dummy])
        tbl.delete("id = '__dummy__'")
        self._tbl = tbl
        self._fresh = True

    def _table(self) -> Any:
        if self._tbl is None:
            import lancedb  # pylint: disable=import-outside-toplevel

            db = lancedb.connect(str(self.lancedb_dir))
            self._tbl = db.open_table(self.table_name)
        return self._tbl

    # -- writes ---------------------------------------------------------

    def upsert(self, rows: Sequence[dict[str, Any]], *, batch_size: int = 1000) -> int:
        tbl = self._table()
        written = 0
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            if not chunk:
                continue
            ids = [r["id"] for r in chunk]
            # On a freshly wiped/created table there is nothing to delete, and
            # scanning a growing fragment list is O(n^2); skip it.
            if not self._fresh:
                pred = " OR ".join(f"id = '{_escape(i)}'" for i in ids)
                tbl.delete(pred)
            payload = [self._row_for_lance(r) for r in chunk]
            tbl.add(payload)
            written += len(payload)
        return written

    def _row_for_lance(self, r: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"id": r["id"]}
        for col in self.meta_columns:
            out[col] = r.get(col) or ""
        if self.store_text:
            out["text"] = r.get("text", "")
        out["vector"] = r["vector"]
        return out

    def delete_ids(self, ids: Iterable[str]) -> int:
        ids = [i for i in ids if i]
        if not ids:
            return 0
        tbl = self._table()
        before = self.count()
        for start in range(0, len(ids), 1000):
            chunk = ids[start : start + 1000]
            id_list = ", ".join("'" + _escape(x) + "'" for x in chunk)
            tbl.delete(f"id IN ({id_list})")
        # LanceDB's delete() returns no count; report actual rows removed.
        return before - self.count()

    # -- reads ----------------------------------------------------------

    def existing_ids(self) -> set[str]:
        import lancedb  # pylint: disable=import-outside-toplevel

        if not self.lancedb_dir.exists():
            return set()
        try:
            db = lancedb.connect(str(self.lancedb_dir))
            if self.table_name not in db.list_tables().tables:
                return set()
            tbl = db.open_table(self.table_name)
            n = tbl.count_rows()
            if not n:
                return set()
            try:
                arrow = tbl.search().select(["id"]).limit(n).to_arrow()
            except Exception:  # noqa: BLE001
                arrow = tbl.to_arrow()
            return {i for i in arrow.column("id").to_pylist() if i and i != "__dummy__"}
        except Exception:  # noqa: BLE001
            return set()

    def count(self) -> int:
        try:
            return int(self._table().count_rows())
        except Exception:  # noqa: BLE001
            return 0

    def search(
        self, qvec: Sequence[float], k: int, *, where: str | None = None
    ) -> list[dict[str, Any]]:
        tbl = self._table()
        builder = tbl.search(qvec).metric("cosine")
        if self._has_ann_index(tbl):
            import contextlib  # pylint: disable=import-outside-toplevel

            with contextlib.suppress(Exception):
                builder = builder.nprobes(self.ann_nprobes)
            if self.ann_refine_factor and self.ann_refine_factor > 0:
                with contextlib.suppress(Exception):
                    builder = builder.refine_factor(self.ann_refine_factor)
        if where:
            builder = builder.where(where, prefilter=True)
        return builder.limit(k).to_list()

    # -- ANN ------------------------------------------------------------

    def maybe_create_ann_index(self, *, quiet: bool = False) -> bool:
        """Build an IVF index when the table crosses :attr:`ann_threshold` rows.

        The flat scan is always correct, so any failure here is logged and
        swallowed — the index is never load-bearing.

        :param quiet: Suppress the summary log line.
        :return: ``True`` if an index was created.
        """
        tbl = self._table()
        if self.ann_threshold <= 0:
            self._has_ann = False
            return False
        try:
            n = int(tbl.count_rows())
        except Exception:  # noqa: BLE001
            return False
        if n < self.ann_threshold:
            self._has_ann = False
            return False

        num_partitions = max(1, min(round(math.sqrt(n)), max(1, n // 100)))
        index_type = (self.ann_index_type or "IVF_FLAT").upper()
        kwargs: dict[str, Any] = {
            "metric": "cosine",
            "vector_column_name": "vector",
            "replace": True,
            "num_partitions": num_partitions,
        }
        if index_type == "IVF_PQ":
            kwargs["num_sub_vectors"] = _pq_subvectors(self.dim)
        try:
            try:
                tbl.create_index(index_type=index_type, **kwargs)
            except TypeError:
                tbl.create_index(**kwargs)
            self._has_ann = True
            if not quiet:
                _LOG.info("Built %s index (n=%d, partitions=%d)", index_type, n, num_partitions)
            return True
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("ANN index build failed (%s); staying flat.", exc)
            self._has_ann = False
            return False

    def _has_ann_index(self, tbl: Any) -> bool:
        if self._has_ann:
            return True
        try:
            indices = tbl.list_indices()
            self._has_ann = bool(indices)
        except Exception:  # noqa: BLE001
            self._has_ann = False
        return self._has_ann


def _pq_subvectors(dim: int) -> int:
    """Pick a PQ sub-vector count that divides *dim* (≈16 dimensions each).

    Product quantization splits each vector into ``num_sub_vectors`` contiguous
    sub-vectors, so the count must divide the embedding dimension exactly.
    Targets ~16 dims per sub-vector (e.g. 24 for a 384-d model) and walks down
    to the nearest divisor.

    :param dim: Embedding dimension.
    :return: A divisor of *dim* suitable for ``num_sub_vectors`` (``1`` worst case).
    """
    target = max(1, dim // 16)
    for m in range(target, 0, -1):
        if dim % m == 0:
            return m
    return 1


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
    given (no re-normalisation) and the returned ``_distance`` matches LanceDB's
    cosine ``_distance`` exactly.

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
        # per-batch dedup delete during the build, matching LanceDBBackend.
        self._fresh = wipe or not existed

    def _c(self) -> Any:
        if self._conn is None:
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

    # -- reads ----------------------------------------------------------

    def existing_ids(self) -> set[str]:
        if not self.db_path.exists():
            return set()
        try:
            conn = self._c()
            return {r[0] for r in conn.execute("SELECT id FROM vec_meta").fetchall()}
        except Exception:  # noqa: BLE001
            return set()

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


def _escape(s: str) -> str:
    return s.replace("'", "''")
