"""kg_utils.temporal — the fleet's shared temporal contract.

Every KG module that knows *when* something happened writes the same three
metadata keys on its :class:`~kg_utils.specs.NodeSpec` nodes, so that a
federated query can filter and order across all of them without knowing which
module produced a node:

    ``occurred_start``  when the thing itself began
    ``occurred_end``    when it finished (omit for a point event)
    ``recorded_at``     when it was written down

The distinction between *occurred* and *recorded* is the one hard-won
semantic here: a diary entry written tonight about last Tuesday occurred on
Tuesday and was recorded tonight, and a timeline that conflates the two puts
it in the wrong place.

Values are ISO-8601 strings and keep the precision they were given. ``"1876"``
stays a year, not a silent ``1876-01-01`` — which matters, because a span's
implied extent follows its precision: a node dated ``"1876"`` overlaps any
query touching that year, while one dated ``"1876-03-04"`` overlaps only that
day. Omitting ``occurred_end`` therefore means "as long as the precision
implies", not "zero duration".

Typical producer::

    from kg_utils.temporal import temporal_metadata

    NodeSpec(..., metadata=temporal_metadata(occurred_start=entry.date))

Typical consumer::

    from kg_utils.temporal import read_span

    span = read_span(node["metadata"])
    if span and span.overlaps(window_start, window_end):
        ...

The spine helpers (:func:`spine_id`, :func:`spine_chain`) mint the
deterministic calendar-node IDs a timeline graph hangs its events from:
``t:2026`` → ``t:2026-08`` → ``t:2026-08-17``.

Stdlib only — this module is part of the core install.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

__all__ = [
    "OCCURRED_END",
    "OCCURRED_START",
    "RECORDED_AT",
    "SPINE_PREFIX",
    "TEMPORAL_KEYS",
    "Precision",
    "TemporalSpan",
    "TemporalValue",
    "parse_temporal",
    "read_span",
    "spine_chain",
    "spine_id",
    "temporal_metadata",
]

#: Metadata key: when the thing itself began.
OCCURRED_START = "occurred_start"
#: Metadata key: when the thing itself finished. Omit for a point event.
OCCURRED_END = "occurred_end"
#: Metadata key: when the thing was written down.
RECORDED_AT = "recorded_at"

#: Every key this contract defines. Modules may carry others; these are shared.
TEMPORAL_KEYS = frozenset({OCCURRED_START, OCCURRED_END, RECORDED_AT})

#: Node-ID prefix for calendar spine nodes.
SPINE_PREFIX = "t"

#: Granularity a temporal value was expressed at, coarsest first.
Precision = Literal["year", "month", "day", "time"]

_PRECISION_ORDER: tuple[Precision, ...] = ("year", "month", "day", "time")

#: Anything this module knows how to read a date out of.
TemporalValue = str | date | datetime | None

_YEAR_RE = re.compile(r"^(\d{4})$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _as_utc(value: datetime) -> datetime:
    """Return *value* as an aware UTC datetime.

    Naive input is *assumed* to be UTC rather than rejected: most corpora
    (diaries, changelogs, EXIF) carry local wall-clock time with no offset,
    and refusing them would exclude the bulk of the fleet's dated material.

    :param value: Datetime to normalise.
    :return: Timezone-aware datetime in UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_temporal(value: TemporalValue) -> tuple[datetime, Precision] | None:
    """Parse a temporal value into its instant and the precision it was given at.

    Accepts ``datetime``, ``date``, and ISO-8601 strings at year (``"2026"``),
    month (``"2026-08"``), day (``"2026-08-17"``), or time
    (``"2026-08-17T14:30:00Z"``) precision. The returned datetime is the
    *start* of the period named; use :meth:`TemporalSpan.end` or
    :func:`_period_end` for its far edge.

    :param value: The value to parse; ``None`` and blank strings return ``None``.
    :return: ``(instant, precision)``, or ``None`` if there is nothing to parse.
    :raises ValueError: If a non-empty string is not a recognisable ISO-8601 date.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return _as_utc(value), "time"

    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC), "day"

    text = str(value).strip()
    if not text:
        return None

    if m := _YEAR_RE.match(text):
        return datetime(int(m.group(1)), 1, 1, tzinfo=UTC), "year"

    if m := _MONTH_RE.match(text):
        return datetime(int(m.group(1)), int(m.group(2)), 1, tzinfo=UTC), "month"

    if m := _DAY_RE.match(text):
        year, month, day = (int(g) for g in m.groups())
        return datetime(year, month, day, tzinfo=UTC), "day"

    # Anything else with a time component: let the stdlib decide. Python 3.11+
    # accepts a trailing "Z", so most real-world stamps land here intact.
    return _as_utc(datetime.fromisoformat(text)), "time"


def _period_end(instant: datetime, precision: Precision) -> datetime:
    """Return the last representable moment of the period *instant* names.

    :param instant: Start of the period, as returned by :func:`parse_temporal`.
    :param precision: Granularity the period was expressed at.
    :return: The inclusive far edge of that period.
    """
    if precision == "year":
        return instant.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
    if precision == "month":
        last = calendar.monthrange(instant.year, instant.month)[1]
        return instant.replace(day=last, hour=23, minute=59, second=59, microsecond=999999)
    if precision == "day":
        return instant.replace(hour=23, minute=59, second=59, microsecond=999999)
    return instant


def _format(instant: datetime, precision: Precision) -> str:
    """Render *instant* back to a canonical ISO-8601 string at *precision*.

    :param instant: The instant to render.
    :param precision: Granularity to render at.
    :return: Canonical ISO-8601 text.
    """
    if precision == "year":
        return f"{instant.year:04d}"
    if precision == "month":
        return f"{instant.year:04d}-{instant.month:02d}"
    if precision == "day":
        return f"{instant.year:04d}-{instant.month:02d}-{instant.day:02d}"
    return instant.isoformat()


def temporal_metadata(
    *,
    occurred_start: TemporalValue = None,
    occurred_end: TemporalValue = None,
    recorded_at: TemporalValue = None,
) -> dict[str, str]:
    """Build the temporal slice of a node's metadata, normalised and precision-preserving.

    Keys whose value is ``None`` are omitted rather than written as null, so
    merging this into existing metadata never overwrites a known value with a
    blank one::

        metadata = {**domain_fields, **temporal_metadata(occurred_start=when)}

    :param occurred_start: When the thing itself began.
    :param occurred_end: When it finished; omit for a point event.
    :param recorded_at: When it was written down.
    :return: Dict of the contract's keys that have values, as ISO-8601 strings.
    """
    out: dict[str, str] = {}
    for key, value in (
        (OCCURRED_START, occurred_start),
        (OCCURRED_END, occurred_end),
        (RECORDED_AT, recorded_at),
    ):
        parsed = parse_temporal(value)
        if parsed is not None:
            out[key] = _format(*parsed)
    return out


@dataclass(frozen=True)
class TemporalSpan:
    """The resolved time a node covers, ready for comparison and ordering.

    :param start: First moment covered; ``None`` when only ``recorded_at`` is known.
    :param end: Last moment covered, inclusive, widened to the precision given.
    :param recorded: When the node was written down, if known.
    :param precision: Granularity ``occurred_start`` was expressed at.
    """

    start: datetime | None = None
    end: datetime | None = None
    recorded: datetime | None = None
    precision: Precision | None = None

    def __bool__(self) -> bool:
        """True when this span carries any temporal information at all."""
        return any(v is not None for v in (self.start, self.end, self.recorded))

    @property
    def sort_key(self) -> datetime:
        """A datetime suitable for ordering events on a timeline.

        Falls back to ``recorded`` when the node says only when it was written
        down, and to the epoch when it says nothing — so a mixed list always
        sorts rather than raising.

        :return: The instant to sort this span by.
        """
        return self.start or self.recorded or datetime.min.replace(tzinfo=UTC)

    def overlaps(self, start: TemporalValue = None, end: TemporalValue = None) -> bool:
        """Whether this span intersects the window ``[start, end]``.

        Either bound may be ``None`` for an open-ended window, so
        ``overlaps(start="2026-01-01")`` means "anything from 2026 onwards".
        A span with no occurrence dates falls back to ``recorded_at``; a span
        with no dates at all never overlaps.

        :param start: Window start; ``None`` for unbounded.
        :param end: Window end; ``None`` for unbounded. Widened to its precision,
            so ``end="2026"`` includes all of 2026.
        :return: ``True`` if the span and the window intersect.
        """
        span_start = self.start or self.recorded
        span_end = self.end or self.recorded or span_start
        if span_start is None or span_end is None:
            return False

        if (parsed := parse_temporal(start)) is not None:
            if span_end < parsed[0]:
                return False
        if (parsed := parse_temporal(end)) is not None:
            if span_start > _period_end(*parsed):
                return False
        return True


def read_span(metadata: dict[str, Any] | None) -> TemporalSpan | None:
    """Read a :class:`TemporalSpan` out of a node's metadata.

    Unparseable values are ignored rather than raised on: a single malformed
    date in one corpus must not abort a federated query across twenty.

    :param metadata: A node's metadata dict, or ``None``.
    :return: The span, or ``None`` if the metadata carries no temporal keys.
    """
    if not metadata:
        return None

    def _try(key: str) -> tuple[datetime, Precision] | None:
        try:
            return parse_temporal(metadata.get(key))
        except (ValueError, TypeError):
            return None

    start = _try(OCCURRED_START)
    end = _try(OCCURRED_END)
    recorded = _try(RECORDED_AT)

    if start is None and end is None and recorded is None:
        return None

    # An absent end means "as wide as the start's precision", not "instantaneous".
    resolved_end = _period_end(*end) if end is not None else None
    if resolved_end is None and start is not None:
        resolved_end = _period_end(*start)

    return TemporalSpan(
        start=start[0] if start else None,
        end=resolved_end,
        recorded=recorded[0] if recorded else None,
        precision=start[1] if start else None,
    )


def spine_id(value: TemporalValue, granularity: Precision = "day") -> str | None:
    """Mint the deterministic calendar-node ID for *value* at *granularity*.

    Spine IDs are stable across rebuilds and sort lexicographically in
    chronological order within a granularity::

        spine_id("2026-08-17")             -> "t:2026-08-17"
        spine_id("2026-08-17", "month")    -> "t:2026-08"

    :param value: Any parseable temporal value.
    :param granularity: Level of the spine to address. ``"time"`` is treated as
        ``"day"``; the spine's finest node is a day.
    :return: The node ID, or ``None`` if *value* holds no date.
    :raises ValueError: If *value* is a string that is not a recognisable date.
    """
    parsed = parse_temporal(value)
    if parsed is None:
        return None
    instant, _ = parsed
    level: Precision = "day" if granularity == "time" else granularity
    return f"{SPINE_PREFIX}:{_format(instant, level)}"


def spine_chain(value: TemporalValue) -> list[str]:
    """Return the year → month → day spine IDs an event hangs from.

    Truncated at the precision of *value*, so a node dated ``"1876"`` yields
    only ``["t:1876"]`` and is never asserted to belong to a day nobody
    recorded::

        spine_chain("2026-08-17")  -> ["t:2026", "t:2026-08", "t:2026-08-17"]
        spine_chain("2026")        -> ["t:2026"]

    :param value: Any parseable temporal value.
    :return: Spine IDs from coarsest to finest; empty if *value* holds no date.
    :raises ValueError: If *value* is a string that is not a recognisable date.
    """
    parsed = parse_temporal(value)
    if parsed is None:
        return []
    instant, precision = parsed
    depth = _PRECISION_ORDER.index("day" if precision == "time" else precision)
    return [f"{SPINE_PREFIX}:{_format(instant, level)}" for level in _PRECISION_ORDER[: depth + 1]]
