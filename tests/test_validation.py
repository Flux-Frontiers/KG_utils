"""Tests for kg_utils.validation: the shared query-argument bounds."""

from __future__ import annotations

import pytest

from kg_utils.validation import (
    MAX_HOP,
    MAX_K,
    MAX_MAX_NODES,
    MAX_QUERY_LEN,
    bounded_int,
    require_query,
)


class TestBoundedInt:
    def test_in_range_returns_value_unchanged(self) -> None:
        assert bounded_int("k", 8, 1, MAX_K) == 8

    def test_bounds_are_inclusive(self) -> None:
        assert bounded_int("k", 1, 1, MAX_K) == 1
        assert bounded_int("k", MAX_K, 1, MAX_K) == MAX_K

    def test_below_minimum_raises_and_names_the_parameter(self) -> None:
        with pytest.raises(ValueError, match=r"k must be between 1 and 100, got 0"):
            bounded_int("k", 0, 1, MAX_K)

    def test_above_maximum_raises(self) -> None:
        with pytest.raises(ValueError, match=r"hop must be between 0 and 5, got 6"):
            bounded_int("hop", 6, 0, MAX_HOP)

    # Arguments arrive as JSON from a model or a form, so loose types are the
    # normal case at this boundary, not the edge case (sweep item 52).
    @pytest.mark.parametrize(
        "value", ["8", True, False, 3.7, None, [8], float("nan"), float("inf")]
    )
    def test_non_integers_are_rejected_by_name(self, value: object) -> None:
        with pytest.raises(ValueError, match=r"k must be an integer"):
            bounded_int("k", value, 1, MAX_K)  # type: ignore[arg-type]

    def test_integral_values_of_other_types_pass_as_int(self) -> None:
        np = pytest.importorskip("numpy")
        for value in (3.0, np.int64(3), np.int32(3)):
            result = bounded_int("k", value, 1, MAX_K)  # type: ignore[arg-type]
            assert result == 3 and type(result) is int

    def test_zero_hop_is_valid(self) -> None:
        """hop=0 is pure semantic and must not be rejected."""
        assert bounded_int("hop", 0, 0, MAX_HOP) == 0


class TestRequireQuery:
    def test_strips_whitespace(self) -> None:
        assert require_query("  graph store  ") == "graph store"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match=r"q must not be empty"):
            require_query("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match=r"q must not be empty"):
            require_query("   \n\t ")

    def test_at_limit_passes(self) -> None:
        assert len(require_query("x" * MAX_QUERY_LEN)) == MAX_QUERY_LEN

    def test_over_limit_raises_with_both_numbers(self) -> None:
        with pytest.raises(
            ValueError, match=rf"at most {MAX_QUERY_LEN} characters, got {MAX_QUERY_LEN + 1}"
        ):
            require_query("x" * (MAX_QUERY_LEN + 1))

    @pytest.mark.parametrize("value", [None, 42, b"socrates", ["socrates"]])
    def test_non_strings_are_rejected(self, value: object) -> None:
        with pytest.raises(ValueError, match=r"q must be a string"):
            require_query(value)  # type: ignore[arg-type]

    def test_custom_limit_is_honoured(self) -> None:
        with pytest.raises(ValueError, match=r"at most 5 characters"):
            require_query("toolong", max_len=5)


def test_module_constants_are_the_fleet_bounds() -> None:
    """The bounds every fleet MCP server already documents, so lifting them
    into the SDK changes no caller's behaviour."""
    assert (MAX_K, MAX_HOP, MAX_MAX_NODES) == (100, 5, 500)
