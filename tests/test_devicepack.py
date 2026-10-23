"""Tests for kg_utils.devicepack -- the on-device pack format.

The format is a contract with a Swift reader, so the tests pin the parts that
reader depends on: the sidecar header and its rejection of a truncated file,
the dense ``vector_index`` with no holes, the manifest's keys, and the hybrid
search's two channels (a passage only BM25 finds must still be ranked).
Cross-checked once against gutenberg_kg's real packs, where all 24 golden
rankings came back identical.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from kg_utils.devicepack import (  # noqa: E402 - after the numpy skip
    PACK_VERSION,
    VECTOR_HEADER_BYTES,
    PackError,
    Passage,
    build_golden,
    build_manifest,
    encode_vector,
    fts_match_expression,
    fts_phrase_expression,
    read_vector_sidecar,
    rrf_fuse,
    search_pack,
    write_passage_pack,
)

DIM = 4


def _unit(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _passage(pid: str, content: str, genre: str = "a", kind: str = "person") -> Passage:
    return Passage(id=pid, kg_name="t", kg_kind="test", kind=kind, content=content, genre=genre)


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    passages = [
        _passage("t:north", "the cold north wind over the moor"),
        _passage("t:east", "sunrise in the east over the harbour"),
        _passage("t:south", "warm south shore and quiet orchards", genre="b"),
        _passage("t:pillar", "she looked back and became a pillar of salt"),
        _passage("t:novec", "a passage nobody embedded"),
    ]
    vectors = {
        "t:north": _unit(1, 0, 0, 0),
        "t:east": _unit(0, 1, 0, 0),
        "t:south": _unit(0, 0, 1, 0),
        "t:pillar": _unit(0, 0, 0, 1),
    }
    dest = tmp_path / "family.pack"
    write_passage_pack(dest, passages, vectors, dim=DIM, embed_model="test-model")
    return dest


# ---------------------------------------------------------------------------
# Vectors and the sidecar
# ---------------------------------------------------------------------------


def test_int8_encoding_normalises_then_scales() -> None:
    encoded = np.frombuffer(encode_vector([3.0, 4.0, 0.0, 0.0], "int8"), dtype=np.int8)
    assert encoded.tolist() == [76, 102, 0, 0]  # 0.6 and 0.8 of 127


def test_float_encoding_is_raw_float32() -> None:
    assert np.frombuffer(encode_vector([1.5, -2.0], "float"), dtype=np.float32).tolist() == [
        1.5,
        -2.0,
    ]


def test_sidecar_rows_follow_vector_index_with_no_holes(pack: Path) -> None:
    matrix, dtype = read_vector_sidecar(pack.with_suffix(".vectors"))
    assert dtype == "int8" and matrix.shape == (4, DIM)
    con = sqlite3.connect(pack)
    rows = dict(con.execute("SELECT id, vector_index FROM passages").fetchall())
    con.close()
    assert rows["t:novec"] is None
    assert sorted(i for i in rows.values() if i is not None) == [0, 1, 2, 3]
    assert int(np.argmax(matrix[rows["t:south"]])) == 2


def test_a_truncated_sidecar_is_rejected(pack: Path) -> None:
    sidecar = pack.with_suffix(".vectors")
    sidecar.write_bytes(sidecar.read_bytes()[:-1])
    with pytest.raises(PackError, match="expected"):
        read_vector_sidecar(sidecar)


def test_a_foreign_file_is_not_a_sidecar(tmp_path: Path) -> None:
    bogus = tmp_path / "x.vectors"
    bogus.write_bytes(b"\0" * (VECTOR_HEADER_BYTES + 8))
    with pytest.raises(PackError, match="not a vector sidecar"):
        read_vector_sidecar(bogus)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_stats_count_passages_vectors_and_misses(tmp_path: Path) -> None:
    stats = write_passage_pack(
        tmp_path / "p.pack",
        [_passage("t:1", "one"), _passage("t:2", "two")],
        {"t:1": _unit(1, 0, 0, 0)},
        dim=DIM,
        embed_model="m",
    )
    assert (stats.passages, stats.vectors, stats.missing_vectors) == (2, 1, 1)
    assert stats.sidecar is not None and stats.total_bytes == stats.bytes + stats.sidecar_bytes


def test_a_pack_without_vectors_has_no_sidecar(tmp_path: Path) -> None:
    stats = write_passage_pack(
        tmp_path / "p.pack", [_passage("t:1", "one")], dim=DIM, embed_model="m"
    )
    assert stats.sidecar is None and stats.vectors == 0
    assert not (tmp_path / "p.vectors").exists()


def test_duplicate_ids_fail_rather_than_shrink(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="duplicate"):
        write_passage_pack(
            tmp_path / "p.pack",
            [_passage("t:1", "one"), _passage("t:1", "again")],
            dim=DIM,
            embed_model="m",
        )


def test_a_vector_of_the_wrong_dimension_fails(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="dimensions"):
        write_passage_pack(
            tmp_path / "p.pack",
            [_passage("t:1", "one")],
            {"t:1": [1.0, 0.0]},
            dim=DIM,
            embed_model="m",
        )


def test_pack_meta_says_what_built_it(pack: Path) -> None:
    con = sqlite3.connect(pack)
    meta = dict(con.execute("SELECT key, value FROM pack_meta").fetchall())
    con.close()
    assert meta["pack_version"] == str(PACK_VERSION)
    assert meta["embed_model"] == "test-model"
    assert meta["vector_dtype"] == "int8"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_fts_expressions_neutralise_query_syntax() -> None:
    assert fts_phrase_expression("Lot's wife?") == '"Lot s wife"'
    assert fts_match_expression("Lot's wife?") == '"Lot" OR "s" OR "wife"'
    assert fts_phrase_expression("?!") == ""


def test_rrf_prefers_a_hit_both_channels_found_and_breaks_ties_lexically() -> None:
    assert rrf_fuse(["a", "b"], ["b", "c"], 3)[0] == "b"
    assert rrf_fuse(["d"], ["l"], 2) == ["l", "d"]


def test_dense_search_ranks_the_nearest_first(pack: Path) -> None:
    hits, _ = search_pack(pack, _unit(0.1, 1, 0, 0), "", k=2, lexical=False)
    assert [h["node_id"] for h in hits] == ["t:east", "t:north"]
    assert hits[0]["score"] == pytest.approx(0.995, abs=0.01)


def test_a_phrase_only_bm25_finds_is_rescued(pack: Path) -> None:
    hits, rescued = search_pack(pack, _unit(1, 0, 0, 0), "pillar of salt", k=1)
    assert hits[0]["node_id"] == "t:pillar"
    assert rescued == {"t:pillar"}


def test_genre_scopes_both_channels(pack: Path) -> None:
    hits, _ = search_pack(pack, _unit(1, 0, 0, 0), "north", k=5, genre="b")
    assert [h["node_id"] for h in hits] == ["t:south"]


def test_a_lexical_hit_with_no_vector_is_not_ranked(pack: Path) -> None:
    hits, _ = search_pack(pack, _unit(1, 0, 0, 0), "nobody embedded", k=5)
    assert "t:novec" not in {h["node_id"] for h in hits}


# ---------------------------------------------------------------------------
# Manifest and golden
# ---------------------------------------------------------------------------


def test_manifest_carries_every_key_the_app_decodes(pack: Path) -> None:
    stats = write_passage_pack(
        pack, [_passage("t:1", "one")], {"t:1": _unit(1, 0, 0, 0)}, dim=DIM, embed_model="m"
    )
    manifest = build_manifest([stats], embed_model="m", dim=DIM, dtype="int8", extra={"x": 1})
    # The Swift PackManifest's non-optional fields.
    for key in (
        "pack_version",
        "generated",
        "embedder",
        "vector_dtype",
        "rrf_k",
        "packs",
        "total_bytes",
    ):
        assert key in manifest
    assert manifest["embedder"] == {
        "model": "m",
        "dim": DIM,
        "normalized": True,
        "note": manifest["embedder"]["note"],
    }
    entry = manifest["packs"][0]
    assert {"name", "bytes", "sha256", "passages", "vectors", "sidecar"} <= set(entry)
    assert entry["sidecar"]["name"] == "family.vectors"
    assert manifest["x"] == 1


def test_golden_records_what_the_pack_answers(pack: Path) -> None:
    golden = build_golden(
        {"family": pack},
        ["pillar of salt"],
        [_unit(1, 0, 0, 0)],
        k=2,
        embed_model="m",
        dim=DIM,
        dtype="int8",
        rescue_tolerance=0.15,
    )
    ranked = golden["queries"][0]["packs"]["family"]
    assert ranked[0] == {"rank": 0, "node_id": "t:pillar", "score": pytest.approx(0.0)}
    assert golden["tolerance"]["max_rank_drift"] == 2
    assert golden["rescue_tolerance"] == 0.15
