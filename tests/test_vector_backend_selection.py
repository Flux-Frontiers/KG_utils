"""Tests for kg_utils.vector_backend.resolve_backend_name / make_backend.

These are pure path-based selection helpers — no sqlite-vec or lancedb
import is required to exercise them (constructing a backend object is cheap;
only *opening* it touches the optional dependency), so this module carries
no ``importorskip`` guard.
"""

from __future__ import annotations

import pytest

from kg_utils.vector_backend import (
    LanceDBBackend,
    SqliteVecBackend,
    make_backend,
    resolve_backend_name,
)


# ---------------------------------------------------------------------------
# resolve_backend_name
# ---------------------------------------------------------------------------


def test_auto_resolves_sqlite_vec_when_neither_store_exists(tmp_path):
    resolved = resolve_backend_name(
        "auto",
        lancedb_dir=tmp_path / "lancedb",
        sqlite_path=tmp_path / "vectors.sqlite",
    )
    assert resolved == "sqlite-vec"


def test_auto_resolves_sqlite_vec_when_sidecar_exists(tmp_path):
    (tmp_path / "vectors.sqlite").write_bytes(b"")
    resolved = resolve_backend_name(
        "auto",
        lancedb_dir=tmp_path / "lancedb",
        sqlite_path=tmp_path / "vectors.sqlite",
    )
    assert resolved == "sqlite-vec"


def test_auto_resolves_lancedb_when_unmigrated_store_exists(tmp_path):
    lancedb_dir = tmp_path / "lancedb"
    lancedb_dir.mkdir()
    (lancedb_dir / "table.lance").write_bytes(b"")
    resolved = resolve_backend_name(
        "auto",
        lancedb_dir=lancedb_dir,
        sqlite_path=tmp_path / "vectors.sqlite",
    )
    assert resolved == "lancedb"


def test_auto_resolves_sqlite_vec_when_lancedb_dir_empty(tmp_path):
    lancedb_dir = tmp_path / "lancedb"
    lancedb_dir.mkdir()
    resolved = resolve_backend_name(
        "auto",
        lancedb_dir=lancedb_dir,
        sqlite_path=tmp_path / "vectors.sqlite",
    )
    assert resolved == "sqlite-vec"


@pytest.mark.parametrize("explicit", ["lancedb", "sqlite-vec"])
def test_explicit_names_pass_through(tmp_path, explicit):
    resolved = resolve_backend_name(
        explicit,
        lancedb_dir=tmp_path / "lancedb",
        sqlite_path=tmp_path / "vectors.sqlite",
    )
    assert resolved == explicit


# ---------------------------------------------------------------------------
# make_backend
# ---------------------------------------------------------------------------


def test_make_backend_sqlite_vec(tmp_path):
    backend = make_backend(
        "sqlite-vec",
        lancedb_dir=tmp_path / "lancedb",
        sqlite_path=tmp_path / "vectors.sqlite",
        table="t",
        dim=8,
        meta_columns=("kind", "name"),
    )
    assert isinstance(backend, SqliteVecBackend)
    assert backend.dim == 8


def test_make_backend_lancedb(tmp_path):
    backend = make_backend(
        "lancedb",
        lancedb_dir=tmp_path / "lancedb",
        sqlite_path=tmp_path / "vectors.sqlite",
        table="t",
        dim=8,
        meta_columns=("kind", "name"),
    )
    assert isinstance(backend, LanceDBBackend)
    assert backend.table_name == "t"


def test_make_backend_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="Unknown vector backend"):
        make_backend(
            "faiss",
            lancedb_dir=tmp_path / "lancedb",
            sqlite_path=tmp_path / "vectors.sqlite",
            table="t",
            dim=8,
            meta_columns=("kind",),
        )
