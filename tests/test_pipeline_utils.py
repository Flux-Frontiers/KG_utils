"""Tests for pure utility functions in kg_utils.pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from kg_utils.pipeline import (
    compute_span,
    docstring_signal,
    lexical_overlap_score,
    make_module_summary,
    make_snippet,
    normalize_query_text,
    query_tokens,
    read_lines,
    safe_join,
    semantic_score_from_distance,
    spans_overlap,
)


# ---------------------------------------------------------------------------
# semantic_score_from_distance
# ---------------------------------------------------------------------------


def test_score_zero_distance() -> None:
    assert semantic_score_from_distance(0.0) == pytest.approx(1.0)


def test_score_large_distance() -> None:
    s = semantic_score_from_distance(1e9)
    assert 0.0 < s < 1e-6


def test_score_negative_clamped() -> None:
    assert semantic_score_from_distance(-5.0) == pytest.approx(1.0)


def test_score_monotone_decreasing() -> None:
    scores = [semantic_score_from_distance(d) for d in [0.0, 0.5, 1.0, 2.0, 10.0]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# query_tokens
# ---------------------------------------------------------------------------


def test_query_tokens_basic() -> None:
    toks = query_tokens("embed texts batch")
    assert "embed" in toks
    assert "texts" in toks
    assert "batch" in toks


def test_query_tokens_splits_underscore() -> None:
    toks = query_tokens("embed_query")
    assert "embed" in toks
    assert "query" in toks
    assert "embed_query" in toks


def test_query_tokens_filters_single_chars() -> None:
    toks = query_tokens("a b ok hi go")
    assert "a" not in toks
    assert "b" not in toks
    assert "ok" in toks
    assert "hi" in toks


def test_query_tokens_empty() -> None:
    assert query_tokens("") == set()


# ---------------------------------------------------------------------------
# normalize_query_text
# ---------------------------------------------------------------------------


def test_normalize_replaces_underscores() -> None:
    assert normalize_query_text("embed_query") == "embed query"


def test_normalize_replaces_hyphens() -> None:
    assert normalize_query_text("bge-small-en") == "bge small en"


def test_normalize_strips_whitespace() -> None:
    assert normalize_query_text("  foo  ") == "foo"


# ---------------------------------------------------------------------------
# docstring_signal
# ---------------------------------------------------------------------------


def test_docstring_signal_none() -> None:
    assert docstring_signal(None) == 0.0


def test_docstring_signal_empty() -> None:
    assert docstring_signal("") == 0.0


def test_docstring_signal_short() -> None:
    s = docstring_signal("ok")
    assert 0.0 <= s <= 1.0


def test_docstring_signal_rich() -> None:
    s = docstring_signal(
        "Embed a list of strings into dense float32 vectors using the local sentence transformer."
    )
    assert s > 0.3


def test_docstring_signal_bounded() -> None:
    for text in ["x", "short doc", "word " * 100]:
        assert 0.0 <= docstring_signal(text) <= 1.0


# ---------------------------------------------------------------------------
# lexical_overlap_score
# ---------------------------------------------------------------------------


def test_lexical_overlap_full_match() -> None:
    # haystack tokenises with r"[A-Za-z0-9_]+" so "embed_query" is one token
    node = {"name": "embed_query", "qualname": "", "module_path": "", "docstring": ""}
    score = lexical_overlap_score({"embed_query"}, node)
    assert score == pytest.approx(1.0)


def test_lexical_overlap_no_match() -> None:
    node = {"name": "foo", "qualname": "foo", "module_path": "", "docstring": ""}
    score = lexical_overlap_score({"bar", "baz"}, node)
    assert score == 0.0


def test_lexical_overlap_empty_tokens() -> None:
    node = {"name": "foo", "qualname": "", "module_path": "", "docstring": ""}
    assert lexical_overlap_score(set(), node) == 0.0


def test_lexical_overlap_partial() -> None:
    # query has 3 tokens; node name contains 1 of them
    node = {"name": "embedder", "qualname": "", "module_path": "", "docstring": ""}
    score = lexical_overlap_score({"embedder", "batch", "pipeline"}, node)
    assert pytest.approx(1 / 3, abs=1e-6) == score


# ---------------------------------------------------------------------------
# safe_join
# ---------------------------------------------------------------------------


def test_safe_join_valid(tmp_path: Path) -> None:
    result = safe_join(tmp_path, "src/main.py")
    assert result == (tmp_path / "src" / "main.py").resolve()


def test_safe_join_escape_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        safe_join(tmp_path, "../../etc/passwd")


def test_safe_join_exact_root(tmp_path: Path) -> None:
    result = safe_join(tmp_path, ".")
    assert result == tmp_path.resolve()


# ---------------------------------------------------------------------------
# read_lines
# ---------------------------------------------------------------------------


def test_read_lines_normal(tmp_path: Path) -> None:
    f = tmp_path / "f.py"
    f.write_text("line1\nline2\nline3", encoding="utf-8")
    lines = read_lines(f)
    assert lines == ["line1", "line2", "line3"]


def test_read_lines_missing(tmp_path: Path) -> None:
    assert read_lines(tmp_path / "nonexistent.py") == []


# ---------------------------------------------------------------------------
# compute_span
# ---------------------------------------------------------------------------


def test_compute_span_module_kind() -> None:
    start, end = compute_span("module", 1, 200, context=5, max_lines=60, file_nlines=500)
    assert start == 1
    assert end == 60


def test_compute_span_no_lineno() -> None:
    start, end = compute_span("function", None, None, context=5, max_lines=60, file_nlines=100)
    assert start == 1
    assert end == 60


def test_compute_span_with_lineno() -> None:
    start, end = compute_span("function", 20, 30, context=3, max_lines=60, file_nlines=100)
    assert start == 17
    assert end == 33


def test_compute_span_capped_by_max_lines() -> None:
    start, end = compute_span("function", 5, 200, context=2, max_lines=20, file_nlines=300)
    assert (end - start + 1) <= 20


def test_compute_span_empty_file() -> None:
    assert compute_span("function", 1, 10, context=2, max_lines=60, file_nlines=0) == (1, 0)


def test_compute_span_clamps_to_file(tmp_path: Path) -> None:
    start, end = compute_span("function", 95, 98, context=5, max_lines=60, file_nlines=100)
    assert end <= 100


# ---------------------------------------------------------------------------
# make_snippet
# ---------------------------------------------------------------------------


def test_make_snippet_basic() -> None:
    lines = [f"line {i}" for i in range(1, 11)]
    sn = make_snippet("src/a.py", lines, start=2, end=4)
    assert sn["path"] == "src/a.py"
    assert sn["start"] == 2
    assert sn["end"] == 4
    assert "line 2" in sn["text"]
    assert "line 4" in sn["text"]


def test_make_snippet_line_numbers_in_text() -> None:
    lines = ["alpha", "beta", "gamma"]
    sn = make_snippet("f.py", lines, start=1, end=3)
    assert "1:" in sn["text"]
    assert "3:" in sn["text"]


# ---------------------------------------------------------------------------
# make_module_summary
# ---------------------------------------------------------------------------


def test_make_module_summary_with_nodes() -> None:
    lines = ["# module header"] * 5
    nodes = [
        {
            "kind": "function",
            "name": "foo",
            "qualname": "foo",
            "lineno": 2,
            "docstring": "does foo",
        },
        {"kind": "class", "name": "Bar", "qualname": "Bar", "lineno": 3, "docstring": ""},
    ]
    sn = make_module_summary("mod.py", lines, "A module.", nodes, max_lines=50)
    assert sn["is_summary"] is True
    assert sn["path"] == "mod.py"
    assert "foo" in sn["text"]
    assert "Bar" in sn["text"]


def test_make_module_summary_no_nodes() -> None:
    lines = ["x"] * 3
    sn = make_module_summary("m.py", lines, None, [], max_lines=20)
    assert sn["is_summary"] is True


# ---------------------------------------------------------------------------
# spans_overlap
# ---------------------------------------------------------------------------


def test_spans_overlap_overlapping() -> None:
    assert spans_overlap((1, 10), (5, 15)) is True


def test_spans_overlap_adjacent_within_gap() -> None:
    assert spans_overlap((1, 5), (6, 10), gap=2) is True


def test_spans_overlap_clearly_separate() -> None:
    assert spans_overlap((1, 5), (20, 30)) is False


def test_spans_overlap_exact_gap_boundary() -> None:
    # a1=5, gap=2 → threshold=7; b0=7 means 7 < 7 is False → overlap
    assert spans_overlap((1, 5), (7, 15), gap=2) is True
    # b0=8 means 7 < 8 is True → no overlap
    assert spans_overlap((1, 5), (8, 15), gap=2) is False
