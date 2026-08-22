"""Tests for kg_utils.temporal — the shared temporal contract.

The cases that matter most here are the ones where a plausible-looking
implementation would be silently wrong: precision widening (a node dated
``"1876"`` covers the whole year, not its first instant), the occurred/recorded
distinction, and open-ended query windows.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from kg_utils.temporal import (
    OCCURRED_END,
    OCCURRED_START,
    RECORDED_AT,
    TEMPORAL_KEYS,
    TemporalSpan,
    parse_temporal,
    read_span,
    spine_chain,
    spine_id,
    temporal_metadata,
)

# -- parse_temporal ----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected_precision"),
    [
        ("2026", "year"),
        ("2026-08", "month"),
        ("2026-08-17", "day"),
        ("2026-08-17T14:30:00", "time"),
        ("2026-08-17T14:30:00Z", "time"),
        ("2026-08-17T14:30:00+02:00", "time"),
    ],
)
def test_parse_precision(value: str, expected_precision: str) -> None:
    parsed = parse_temporal(value)
    assert parsed is not None
    assert parsed[1] == expected_precision


def test_parse_returns_start_of_period() -> None:
    assert parse_temporal("2026")[0] == datetime(2026, 1, 1, tzinfo=UTC)
    assert parse_temporal("2026-08")[0] == datetime(2026, 8, 1, tzinfo=UTC)


def test_parse_accepts_date_and_datetime_objects() -> None:
    assert parse_temporal(date(2026, 8, 17))[1] == "day"
    assert parse_temporal(datetime(2026, 8, 17, 14, 30))[1] == "time"


def test_naive_datetime_is_assumed_utc() -> None:
    """Naive input is accepted, not rejected — most corpora carry no offset."""
    parsed = parse_temporal(datetime(2026, 8, 17, 14, 30))
    assert parsed[0].tzinfo is not None
    assert parsed[0] == datetime(2026, 8, 17, 14, 30, tzinfo=UTC)


def test_aware_datetime_is_converted_to_utc() -> None:
    east = timezone(timedelta(hours=2))
    parsed = parse_temporal(datetime(2026, 8, 17, 14, 30, tzinfo=east))
    assert parsed[0] == datetime(2026, 8, 17, 12, 30, tzinfo=UTC)


def test_empty_values_return_none() -> None:
    assert parse_temporal(None) is None
    assert parse_temporal("") is None
    assert parse_temporal("   ") is None


def test_unparseable_string_raises() -> None:
    with pytest.raises(ValueError):
        parse_temporal("last Tuesday")


# -- temporal_metadata -------------------------------------------------------


def test_metadata_omits_absent_keys() -> None:
    md = temporal_metadata(occurred_start="2026-08-17")
    assert md == {OCCURRED_START: "2026-08-17"}
    assert OCCURRED_END not in md
    assert RECORDED_AT not in md


def test_metadata_preserves_precision() -> None:
    """A year stays a year — never silently widened to a January day."""
    assert temporal_metadata(occurred_start="1876")[OCCURRED_START] == "1876"
    assert temporal_metadata(occurred_start="1876-03")[OCCURRED_START] == "1876-03"


def test_metadata_keys_are_the_contract() -> None:
    md = temporal_metadata(
        occurred_start="2026-08-17", occurred_end="2026-08-18", recorded_at="2026-08-19"
    )
    assert set(md) == TEMPORAL_KEYS


def test_metadata_merges_without_clobbering() -> None:
    merged = {"kind": "entry", **temporal_metadata(occurred_start="2026-08-17")}
    assert merged["kind"] == "entry"
    assert merged[OCCURRED_START] == "2026-08-17"


# -- read_span ---------------------------------------------------------------


def test_read_span_none_when_no_temporal_keys() -> None:
    assert read_span(None) is None
    assert read_span({}) is None
    assert read_span({"kind": "file"}) is None


def test_absent_end_widens_to_precision() -> None:
    """The central rule: no occurred_end means 'as wide as the precision'."""
    span = read_span({OCCURRED_START: "1876"})
    assert span is not None
    assert span.start == datetime(1876, 1, 1, tzinfo=UTC)
    assert span.end.year == 1876
    assert (span.end.month, span.end.day) == (12, 31)


def test_month_precision_widens_to_month_end() -> None:
    span = read_span({OCCURRED_START: "2026-02"})
    assert span.end.day == 28  # 2026 is not a leap year
    span_leap = read_span({OCCURRED_START: "2024-02"})
    assert span_leap.end.day == 29


def test_explicit_end_wins_over_precision() -> None:
    span = read_span({OCCURRED_START: "2026-08-17", OCCURRED_END: "2026-09-02"})
    assert span.end.month == 9
    assert span.end.day == 2


def test_occurred_and_recorded_are_distinct() -> None:
    """A diary entry written tonight about last Tuesday occurred on Tuesday."""
    span = read_span({OCCURRED_START: "2026-08-11", RECORDED_AT: "2026-08-17"})
    assert span.start.day == 11
    assert span.recorded.day == 17
    assert span.sort_key.day == 11


def test_recorded_only_span_still_sorts() -> None:
    span = read_span({RECORDED_AT: "2026-08-17"})
    assert span.start is None
    assert span.sort_key == datetime(2026, 8, 17, tzinfo=UTC)


def test_malformed_dates_are_ignored_not_raised() -> None:
    """One bad date in one corpus must not abort a federated query."""
    span = read_span({OCCURRED_START: "sometime in the 90s", RECORDED_AT: "2026-08-17"})
    assert span is not None
    assert span.start is None
    assert span.recorded == datetime(2026, 8, 17, tzinfo=UTC)


def test_all_malformed_returns_none() -> None:
    assert read_span({OCCURRED_START: "nonsense"}) is None


# -- overlaps ----------------------------------------------------------------


def test_year_precision_overlaps_any_query_in_that_year() -> None:
    span = read_span({OCCURRED_START: "1876"})
    assert span.overlaps("1876-03-04", "1876-03-04")
    assert span.overlaps("1876-12-31", "1877-01-01")
    assert not span.overlaps("1877-01-01", "1877-12-31")


def test_day_precision_does_not_overlap_other_days() -> None:
    span = read_span({OCCURRED_START: "1876-03-04"})
    assert span.overlaps("1876-03-04")
    assert not span.overlaps("1876-03-05", "1876-03-06")


def test_open_ended_windows() -> None:
    span = read_span({OCCURRED_START: "2026-08-17"})
    assert span.overlaps(start="2026-01-01")  # everything from 2026 on
    assert span.overlaps(end="2027-01-01")  # everything up to 2027
    assert span.overlaps()  # unbounded window matches anything dated
    assert not span.overlaps(start="2027-01-01")


def test_window_end_widens_to_its_own_precision() -> None:
    """end='2026' must include all of 2026, not just its first instant."""
    span = read_span({OCCURRED_START: "2026-08-17"})
    assert span.overlaps("2026", "2026")


def test_span_with_no_dates_never_overlaps() -> None:
    assert not TemporalSpan().overlaps()


def test_bool_reflects_content() -> None:
    assert not TemporalSpan()
    assert TemporalSpan(recorded=datetime(2026, 8, 17, tzinfo=UTC))


# -- spine ids ---------------------------------------------------------------


def test_spine_id_granularities() -> None:
    assert spine_id("2026-08-17") == "t:2026-08-17"
    assert spine_id("2026-08-17", "month") == "t:2026-08"
    assert spine_id("2026-08-17", "year") == "t:2026"


def test_spine_id_time_collapses_to_day() -> None:
    """The spine's finest node is a day."""
    assert spine_id("2026-08-17T14:30:00", "time") == "t:2026-08-17"


def test_spine_id_is_deterministic() -> None:
    assert spine_id("2026-08-17") == spine_id(date(2026, 8, 17))


def test_spine_id_none_for_empty() -> None:
    assert spine_id(None) is None


def test_spine_chain_full_depth() -> None:
    assert spine_chain("2026-08-17") == ["t:2026", "t:2026-08", "t:2026-08-17"]


def test_spine_chain_truncates_at_precision() -> None:
    """A node dated only by year is never asserted to belong to a day."""
    assert spine_chain("2026") == ["t:2026"]
    assert spine_chain("2026-08") == ["t:2026", "t:2026-08"]


def test_spine_chain_empty_for_none() -> None:
    assert spine_chain(None) == []


def test_spine_ids_sort_chronologically() -> None:
    """Lexicographic order must match chronological order within a level."""
    ids = [spine_id(d) for d in ("2026-08-09", "2026-08-17", "2026-01-02", "2025-12-31")]
    assert sorted(ids) == [
        "t:2025-12-31",
        "t:2026-01-02",
        "t:2026-08-09",
        "t:2026-08-17",
    ]
