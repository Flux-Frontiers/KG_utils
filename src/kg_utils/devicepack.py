"""kg_utils.devicepack -- on-device search packs, the format the Knowledge Press apps read.

A device pack is a corpus cut down to what an app needs to search it with no
server: a SQLite file of passages with an FTS5 index, a sidecar of int8 (or
float32) embeddings beside it, a ``manifest.json`` naming the embedder the
vectors require, and a ``golden.json`` of reference rankings the app's own
search must reproduce. The Swift kit in ``knowledge_press`` and
``kpress_family`` reads exactly this; ``gutenberg_kg``'s ``export-swift`` wrote
it first, and this module is that format lifted out so other KGs can write it.
Nothing here knows what a passage *is*: a book chunk, a diary entry and a
person's summary are all a :class:`Passage`.

Layout::

    <out>/
      manifest.json     pack version, the embedder, every pack and its checksum
      <name>.pack       passages + FTS5 over their text
      <name>.vectors    their embeddings, row-major, memory-mappable
      golden.json       reference top-k per query, as the packs answer it

Why the vectors sit beside the pack
-----------------------------------
A ``vec0`` table cannot be read without the sqlite-vec C extension compiled into
the reader, and iOS ships stock SQLite. The sidecar is a 32-byte header (magic,
dtype, dimension, count) and then row-major vectors; ``passages.vector_index``
is a dense row number into it, so a passage with no vector leaves no hole. The
device memory-maps the file and takes dot products straight out of it.

Search
------
:func:`search_pack` is the reference the app must match: a dense channel over
the sidecar, a BM25 channel over FTS5 (exact phrase first, any term as the
fallback), fused by reciprocal rank at :data:`RRF_K`. It filters on ``genre``
only, as the Swift engine does; what a pack holds is the exporter's choice.

Needs NumPy for anything touching vectors (the ``semantic`` or ``viz3d``
extra); writing and reading passages does not.

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

#: The pack format. The app refuses a manifest newer than it understands.
PACK_VERSION: Final[int] = 1

#: First eight bytes of every vector sidecar.
VECTOR_MAGIC: Final[bytes] = b"GKGVEC01"

#: Header bytes before the vector rows. 32 keeps the rows 16-byte aligned, so
#: the device can map them and hand them straight to SIMD.
VECTOR_HEADER_BYTES: Final[int] = 32

#: Reciprocal-rank-fusion constant, shared by the worker, the packs and the app.
RRF_K: Final[int] = 60

DTYPES: Final[tuple[str, ...]] = ("int8", "float")
_DTYPE_CODES: Final[dict[str, int]] = {"int8": 0, "float": 1}
_DTYPE_ITEMSIZE: Final[dict[str, int]] = {"int8": 1, "float": 4}

#: Default parity tolerances written into ``golden.json``. ``max_rank_drift``
#: is not zero because the reference and the device differ in their last bits
#: (numpy over dequantised int8 against Accelerate), so near-ties may swap.
DEFAULT_TOLERANCE: Final[dict[str, float | int]] = {
    "rank_overlap": 0.9,
    "score_delta": 0.02,
    "max_rank_drift": 2,
}

PASSAGE_SCHEMA: Final[str] = """
CREATE TABLE pack_meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE passages (
  id         TEXT PRIMARY KEY,
  kg_name    TEXT NOT NULL,
  kg_kind    TEXT NOT NULL,
  kind       TEXT NOT NULL,
  name       TEXT,
  title      TEXT,
  node_title TEXT,
  author     TEXT,
  genre      TEXT,
  book       TEXT,
  file_path  TEXT,
  char_start INTEGER,
  chapter    INTEGER,
  timestamp  TEXT,
  vector_index INTEGER,
  content    TEXT NOT NULL
);
CREATE INDEX idx_passages_scope ON passages(genre, kind);
CREATE INDEX idx_passages_vec   ON passages(vector_index);
CREATE INDEX idx_passages_read  ON passages(file_path, kind, char_start);
CREATE INDEX idx_passages_kg    ON passages(kg_name);
"""

FTS_SCHEMA: Final[str] = """
CREATE VIRTUAL TABLE passages_fts USING fts5(
  content,
  content='passages',
  content_rowid='rowid',
  tokenize='porter unicode61 remove_diacritics 2'
);
"""


class PackError(RuntimeError):
    """A pack could not be written or read, or NumPy is missing."""


@dataclass(frozen=True)
class Passage:
    """One searchable row of a pack.

    :param id: Unique within the pack; ``<kg_name>:<node_id>`` by convention.
    :param kg_name: The KG the passage came from.
    :param kg_kind: Its KG kind, as a hit reports it.
    :param kind: The node's kind; the app reads ``section`` and ``chunk`` for
        a book's chapters, and nothing else by kind.
    :param content: The text searched and shown.
    :param name: Short label.
    :param title: The work's title, for hit cards.
    :param node_title: This node's own title.
    :param author: Author, or any person the hit card names.
    :param genre: The scope the app filters on.
    :param book: Book or collection slug.
    :param file_path: Source document path.
    :param char_start: Offset in the source document.
    :param chapter: Chapter number, for chapter-less books.
    :param timestamp: ISO date, for dated passages.
    """

    id: str
    kg_name: str
    kg_kind: str
    kind: str
    content: str
    name: str | None = None
    title: str | None = None
    node_title: str | None = None
    author: str | None = None
    genre: str | None = None
    book: str | None = None
    file_path: str | None = None
    char_start: int | None = None
    chapter: int | None = None
    timestamp: str | None = None


_PASSAGE_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "kg_name",
    "kg_kind",
    "kind",
    "name",
    "title",
    "node_title",
    "author",
    "genre",
    "book",
    "file_path",
    "char_start",
    "chapter",
    "timestamp",
    "content",
)


@dataclass
class PackStats:
    """What a finished pack contains.

    :param path: The pack file.
    :param passages: Rows written.
    :param vectors: Vectors written to the sidecar.
    :param missing_vectors: Passages with no vector.
    :param bytes: Pack size.
    :param sha256: Pack checksum.
    :param sidecar: The ``.vectors`` sidecar, if any.
    :param sidecar_bytes: Its size.
    :param sidecar_sha256: Its checksum.
    """

    path: Path
    passages: int = 0
    vectors: int = 0
    missing_vectors: int = 0
    bytes: int = 0
    sha256: str = ""
    sidecar: Path | None = None
    sidecar_bytes: int = 0
    sidecar_sha256: str = ""

    @property
    def name(self) -> str:
        """The pack's file name."""
        return self.path.name

    @property
    def total_bytes(self) -> int:
        """Pack plus sidecar: what has to reach the device."""
        return self.bytes + self.sidecar_bytes


def _numpy() -> Any:
    try:
        import numpy  # noqa: PLC0415 - optional, only for vectors
    except ImportError as exc:
        raise PackError("vectors need numpy: install kgmodule-utils[semantic]") from exc
    return numpy


def sha256_file(path: Path) -> str:
    """Streamed SHA-256 of a file.

    :param path: The file.
    :return: Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------


def encode_vector(vector: Any, dtype: str) -> bytes:
    """Encode one vector for the sidecar.

    int8 vectors are L2-normalised first. Cosine is scale-invariant, so no
    ranking changes, and the ``* 127`` scale then lands inside int8 range.

    :param vector: A float vector.
    :param dtype: ``"int8"`` or ``"float"``.
    :return: The encoded bytes.
    """
    numpy = _numpy()
    array = numpy.asarray(vector, dtype=numpy.float32)
    if dtype == "float":
        return array.tobytes()
    if dtype != "int8":
        raise PackError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    norm = float(numpy.linalg.norm(array))
    if norm > 0:
        array = array / norm
    return numpy.clip(numpy.round(array * 127.0), -128, 127).astype(numpy.int8).tobytes()


def vector_header(dtype: str, dim: int, count: int) -> bytes:
    """Build the sidecar's 32-byte header.

    :param dtype: ``"int8"`` or ``"float"``.
    :param dim: Embedding dimension.
    :param count: Vectors in the file.
    :return: Exactly :data:`VECTOR_HEADER_BYTES` bytes.
    """
    header = bytearray(VECTOR_HEADER_BYTES)
    header[0:8] = VECTOR_MAGIC
    header[8] = _DTYPE_CODES[dtype]
    header[12:16] = int(dim).to_bytes(4, "little")
    header[16:24] = int(count).to_bytes(8, "little")
    return bytes(header)


def read_vector_sidecar(sidecar: Path) -> tuple[Any, str]:
    """Memory-map a sidecar.

    :param sidecar: A ``.vectors`` file.
    :return: ``(matrix, dtype)``: a ``(count, dim)`` array viewing the file,
        and its dtype name.
    :raises PackError: If the magic, dtype or length disagree, as a truncated
        download would.
    """
    numpy = _numpy()
    with open(sidecar, "rb") as handle:
        header = handle.read(VECTOR_HEADER_BYTES)
    if len(header) < VECTOR_HEADER_BYTES or header[0:8] != VECTOR_MAGIC:
        raise PackError(f"{sidecar} is not a vector sidecar")
    dtype = {code: name for name, code in _DTYPE_CODES.items()}.get(header[8])
    if dtype is None:
        raise PackError(f"{sidecar} has an unknown vector dtype ({header[8]})")
    dim = int.from_bytes(header[12:16], "little")
    count = int.from_bytes(header[16:24], "little")
    expected = VECTOR_HEADER_BYTES + count * dim * _DTYPE_ITEMSIZE[dtype]
    actual = sidecar.stat().st_size
    if actual != expected:
        raise PackError(
            f"{sidecar} is {actual} bytes, expected {expected} for {count} x {dim} {dtype}"
        )
    matrix = numpy.memmap(
        sidecar,
        dtype=numpy.int8 if dtype == "int8" else numpy.float32,
        mode="r",
        offset=VECTOR_HEADER_BYTES,
        shape=(count, dim),
    )
    return matrix, dtype


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_passage_pack(
    dest: Path,
    passages: Iterable[Passage],
    vectors: Mapping[str, Any] | None = None,
    *,
    dim: int,
    embed_model: str,
    dtype: str = "int8",
    meta: Mapping[str, object] | None = None,
) -> PackStats:
    """Write a passage pack and, given vectors, its sidecar.

    A passage with no vector stays in the pack, findable lexically, and is
    counted in ``missing_vectors``. Ids must be unique: a collision fails the
    write rather than shrinking the pack.

    :param dest: The ``.pack`` path; replaced if it exists. The sidecar is
        written beside it as ``<stem>.vectors``.
    :param passages: The rows, in the order to store them.
    :param vectors: Passage id -> vector, or ``None`` for a lexical-only pack.
    :param dim: Embedding dimension every vector must have.
    :param embed_model: The model the vectors came from, stamped into the pack.
    :param dtype: ``"int8"`` (default) or ``"float"``.
    :param meta: Extra ``pack_meta`` rows.
    :return: Stats for the finished pack.
    :raises PackError: On a bad dtype or a vector of the wrong dimension.
    """
    if dtype not in DTYPES:
        raise PackError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    sidecar = dest.with_suffix(".vectors")
    sidecar.unlink(missing_ok=True)

    stats = PackStats(path=dest)
    con = sqlite3.connect(str(dest))
    try:
        con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
        con.executescript(PASSAGE_SCHEMA)
        insert = (
            f"INSERT INTO passages({', '.join(_PASSAGE_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _PASSAGE_COLUMNS)})"
        )
        con.executemany(insert, ([getattr(p, c) for c in _PASSAGE_COLUMNS] for p in passages))
        stats.passages = con.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        con.executescript(FTS_SCHEMA)
        con.execute("INSERT INTO passages_fts(passages_fts) VALUES('rebuild')")

        if vectors:
            itemsize = _DTYPE_ITEMSIZE[dtype]
            assignments: list[tuple[int, str]] = []
            with open(sidecar, "wb") as handle:
                handle.write(b"\0" * VECTOR_HEADER_BYTES)
                for (passage_id,) in con.execute("SELECT id FROM passages ORDER BY rowid"):
                    vector = vectors.get(passage_id)
                    if vector is None:
                        continue
                    encoded = encode_vector(vector, dtype)
                    if len(encoded) != dim * itemsize:
                        raise PackError(
                            f"{passage_id} has {len(encoded) // itemsize} dimensions, "
                            f"expected {dim}"
                        )
                    handle.write(encoded)
                    assignments.append((len(assignments), passage_id))
                handle.seek(0)
                handle.write(vector_header(dtype, dim, len(assignments)))
            con.executemany("UPDATE passages SET vector_index = ? WHERE id = ?", assignments)
            stats.vectors = len(assignments)
        stats.missing_vectors = stats.passages - stats.vectors

        stamps: dict[str, object] = {
            "pack": dest.stem,
            "pack_version": PACK_VERSION,
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "embed_model": embed_model,
            "embed_dim": dim,
            "vector_dtype": dtype if stats.vectors else "none",
            "rrf_k": RRF_K,
            **(meta or {}),
        }
        con.executemany(
            "INSERT OR REPLACE INTO pack_meta(key, value) VALUES (?, ?)",
            [(key, str(value)) for key, value in stamps.items()],
        )
        con.commit()
        con.execute("VACUUM")
    except sqlite3.IntegrityError as exc:
        raise PackError(f"duplicate passage id in {dest.name}: {exc}") from exc
    finally:
        con.close()

    if stats.vectors == 0:
        sidecar.unlink(missing_ok=True)
    else:
        stats.sidecar = sidecar
        stats.sidecar_bytes = sidecar.stat().st_size
        stats.sidecar_sha256 = sha256_file(sidecar)
    stats.bytes = dest.stat().st_size
    stats.sha256 = sha256_file(dest)
    return stats


def build_manifest(
    packs: Sequence[PackStats],
    *,
    embed_model: str,
    dim: int,
    dtype: str,
    golden: str | None = "golden.json",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the packs for the app to check itself against.

    The embedder block matters most: vectors are meaningless to a query
    embedded by another model, so the app refuses the corpus until they agree.

    :param packs: The finished packs.
    :param embed_model: The embedder's model name.
    :param dim: Its dimension.
    :param dtype: The vectors' dtype.
    :param golden: The golden file's name, or ``None``.
    :param extra: Further top-level keys, such as privacy settings.
    :return: The manifest document.
    """
    return {
        "pack_version": PACK_VERSION,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedder": {
            "model": embed_model,
            "dim": dim,
            "normalized": True,
            "note": "Query vectors must come from this model; the packs are unusable without it.",
        },
        "vector_dtype": dtype if any(p.vectors for p in packs) else "none",
        "rrf_k": RRF_K,
        "vector_sidecar": {
            "magic": VECTOR_MAGIC.decode("ascii"),
            "header_bytes": VECTOR_HEADER_BYTES,
            "layout": "row-major, one vector per row, indexed by passages.vector_index",
        },
        "packs": [
            {
                "name": p.name,
                "bytes": p.bytes,
                "sha256": p.sha256,
                "passages": p.passages,
                "vectors": p.vectors,
                "passages_without_vectors": p.missing_vectors,
                "sidecar": (
                    {"name": p.sidecar.name, "bytes": p.sidecar_bytes, "sha256": p.sidecar_sha256}
                    if p.sidecar is not None
                    else None
                ),
            }
            for p in packs
        ],
        "total_bytes": sum(p.total_bytes for p in packs),
        "golden": golden,
        **(extra or {}),
    }


# ---------------------------------------------------------------------------
# Search: the reference the app must reproduce
# ---------------------------------------------------------------------------


def fts_terms(query: str) -> list[str]:
    """Split a query into bare alphanumeric terms, so none is read as FTS5 syntax.

    :param query: The query text.
    :return: The terms, in order.
    """
    return [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]


def fts_phrase_expression(query: str) -> str:
    """The exact-phrase FTS5 expression, tried before the any-term fallback.

    :param query: The query text.
    :return: A quoted phrase, or ``""`` when nothing is searchable.
    """
    terms = fts_terms(query)
    return '"' + " ".join(terms) + '"' if terms else ""


def fts_match_expression(query: str) -> str:
    """The any-term FTS5 expression: every term quoted, OR-ed.

    :param query: The query text.
    :return: An FTS5 MATCH expression, or ``""``.
    """
    return " OR ".join(f'"{term}"' for term in fts_terms(query))


def rrf_fuse(
    dense_ids: Sequence[str], lexical_ids: Sequence[str], k: int, *, rrf_k: int = RRF_K
) -> list[str]:
    """Blend two rank lists by reciprocal rank fusion.

    Lexical ranks are added first, so a cross-channel tie goes to the exact
    match.

    :param dense_ids: Ids best-first by cosine.
    :param lexical_ids: Ids best-first by BM25.
    :param k: How many fused ids to return.
    :param rrf_k: The fusion constant.
    :return: Fused ids, best-first.
    """
    scores: dict[str, float] = {}
    for rank, node_id in enumerate(lexical_ids):
        scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, node_id in enumerate(dense_ids):
        scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda i: -scores[i])[:k]


def search_pack(
    pack: Path,
    query_vector: Any,
    query_text: str,
    *,
    k: int = 10,
    genre: str | None = None,
    lexical: bool = True,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Run a pack's hybrid search: dense + BM25, fused by RRF.

    Both channels oversample to ``k * 3``. A hit found only lexically gets its
    cosine from the sidecar, so every fused hit carries an honest score.

    :param pack: A passage pack, its sidecar beside it.
    :param query_vector: The query embedding.
    :param query_text: The query, for the lexical channel.
    :param k: Hits to return.
    :param genre: Restrict to one genre, as the app's scope picker does.
    :param lexical: Include the BM25 channel.
    :return: Hits best-first (``node_id``, ``score``, ``kind``, ``genre``,
        ``title``, ``content`` and the rest of the row), and the ids the
        lexical channel rescued from outside the dense top ``k``.
    """
    numpy = _numpy()
    con = sqlite3.connect(f"file:{pack}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        scope = " AND genre = ?" if genre else ""
        params: list[object] = [genre] if genre else []

        dense_ids: list[str] = []
        all_scores: dict[str, float] = {}
        sidecar = pack.with_suffix(".vectors")
        if sidecar.exists():
            matrix, _dtype = read_vector_sidecar(sidecar)
            rows = con.execute(
                f"SELECT id, vector_index FROM passages WHERE vector_index IS NOT NULL{scope}",
                params,
            ).fetchall()
            if rows:
                ids = [row["id"] for row in rows]
                indices = numpy.fromiter(
                    (row["vector_index"] for row in rows), dtype=numpy.int64, count=len(rows)
                )
                candidates = numpy.asarray(matrix[indices], dtype=numpy.float32)
                norms = numpy.linalg.norm(candidates, axis=1).clip(min=1e-12)
                query = numpy.asarray(query_vector, dtype=numpy.float32)
                query = query / max(float(numpy.linalg.norm(query)), 1e-12)
                similarity = (candidates @ query) / norms
                dense_ids = [ids[i] for i in numpy.argsort(-similarity)[: k * 3]]
                all_scores = {ids[i]: float(similarity[i]) for i in range(len(ids))}

        lexical_ids: list[str] = []
        if lexical:
            sql = (
                "SELECT p.id AS id FROM passages_fts f JOIN passages p ON p.rowid = f.rowid "
                f"WHERE passages_fts MATCH ?{scope.replace('genre', 'p.genre')} "
                "ORDER BY bm25(passages_fts) LIMIT ?"
            )

            def run(expression: str) -> list[str]:
                if not expression:
                    return []
                try:
                    return [r["id"] for r in con.execute(sql, [expression, *params, k * 3])]
                except sqlite3.Error:
                    return []

            lexical_ids = run(fts_phrase_expression(query_text)) or run(
                fts_match_expression(query_text)
            )
            lexical_ids = [i for i in lexical_ids if i in all_scores]

        ordered = rrf_fuse(dense_ids, lexical_ids, k) if lexical_ids else dense_ids[:k]
        rescued = set(lexical_ids) - set(dense_ids[:k])
        if not ordered:
            return [], set()

        marks = ", ".join("?" for _ in ordered)
        by_id = {
            row["id"]: dict(row)
            for row in con.execute(f"SELECT * FROM passages WHERE id IN ({marks})", ordered)
        }
        hits = []
        for node_id in ordered:
            row = by_id.get(node_id)
            if row is None:
                continue
            hits.append(
                {
                    **row,
                    "node_id": node_id,
                    "score": round(all_scores.get(node_id, 0.0), 4),
                }
            )
        return hits, rescued & {h["node_id"] for h in hits}
    finally:
        con.close()


def build_golden(
    packs: Mapping[str, Path],
    queries: Sequence[str],
    query_vectors: Sequence[Any],
    *,
    k: int,
    embed_model: str,
    dim: int,
    dtype: str,
    tolerance: Mapping[str, float | int] | None = None,
    rescue_tolerance: float | None = None,
) -> dict[str, Any]:
    """Record each query's top-k as the packs answer it: the app's parity gate.

    The caller embeds the queries with the same model the packs were built
    with, so this module never loads one.

    :param packs: Pack name -> path; each is searched on its own.
    :param queries: The golden queries.
    :param query_vectors: Their embeddings, in the same order.
    :param k: Hits to record per query.
    :param embed_model: The embedder's model name.
    :param dim: Its dimension.
    :param dtype: The packs' vector dtype.
    :param tolerance: Parity tolerances; :data:`DEFAULT_TOLERANCE` if omitted.
    :param rescue_tolerance: Recorded when the app checks its own constant
        against it.
    :return: The golden document. Each query entry has ``packs``; a caller
        that merges packs adds its own ``all`` ranking.
    """
    if len(queries) != len(query_vectors):
        raise PackError("queries and query_vectors differ in length")
    entries = []
    for query, vector in zip(queries, query_vectors, strict=True):
        ranked: dict[str, list[dict[str, Any]]] = {}
        for name, path in packs.items():
            hits, _rescued = search_pack(path, vector, query, k=k)
            ranked[name] = [
                {"rank": rank, "node_id": hit["node_id"], "score": hit["score"]}
                for rank, hit in enumerate(hits)
            ]
        entries.append({"query": query, "packs": ranked})
    document: dict[str, Any] = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "embed_model": embed_model,
        "embed_dim": dim,
        "vector_dtype": dtype,
        "rrf_k": RRF_K,
        "k": k,
        "tolerance": dict(tolerance or DEFAULT_TOLERANCE),
        "queries": entries,
    }
    if rescue_tolerance is not None:
        document["rescue_tolerance"] = rescue_tolerance
    return document


__all__ = [
    "DEFAULT_TOLERANCE",
    "DTYPES",
    "FTS_SCHEMA",
    "PACK_VERSION",
    "PASSAGE_SCHEMA",
    "RRF_K",
    "VECTOR_HEADER_BYTES",
    "VECTOR_MAGIC",
    "PackError",
    "PackStats",
    "Passage",
    "build_golden",
    "build_manifest",
    "encode_vector",
    "fts_match_expression",
    "fts_phrase_expression",
    "fts_terms",
    "read_vector_sidecar",
    "rrf_fuse",
    "search_pack",
    "sha256_file",
    "vector_header",
    "write_passage_pack",
]
