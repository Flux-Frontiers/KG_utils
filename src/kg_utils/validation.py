"""kg_utils/validation.py

Boundary validation for the arguments every KG module's ``query()`` and
``pack()`` accept from outside: a CLI argument, an MCP tool call, a web form.
Both surfaces take external input, so the base class checks it once, here,
before touching the index or the graph.

Lifted from ``genealogy_kg.validation`` on 2026-09-20, where it was the
reference implementation and had been copied into two more repos. Three
copies of one check is an SDK feature request.

The limits are the module constants below. A module that needs different
ones sets the matching class attribute on its :class:`~kg_utils.pipeline.KGModule`
subclass (``max_k``, ``max_hop``, ``max_max_nodes``, ``max_query_len``) and
never overrides ``query()`` to do it.

Author: Eric G. Suchanek, PhD
License: Elastic 2.0
"""

from __future__ import annotations

#: Search result count, ``k``.
MAX_K = 100
#: Graph expansion hops from the seeds. ``0`` is pure semantic and is valid.
MAX_HOP = 5
#: Returned node cap, ``max_nodes``.
MAX_MAX_NODES = 500
#: Natural-language query length, in characters. Generous on purpose:
#: ``kg-rag`` passes a user's prompt to ``query()`` unshortened, and the
#: embedder truncates a long one harmlessly. The check is against an empty
#: or absurd query, not a long one. A surface that wants a tighter bound,
#: as ``genealogy_kg``'s MCP server does at 500, keeps its own.
MAX_QUERY_LEN = 2000


def bounded_int(name: str, value: int, minimum: int, maximum: int) -> int:
    """Validate that an integer falls within an inclusive range.

    :param name: Parameter name, used in the error message.
    :param value: The value to validate.
    :param minimum: Inclusive lower bound.
    :param maximum: Inclusive upper bound.
    :return: ``value``, unchanged.
    :raises ValueError: If ``value`` is outside ``[minimum, maximum]``.
    """
    if not (minimum <= value <= maximum):
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def require_query(q: str, max_len: int = MAX_QUERY_LEN) -> str:
    """Validate a natural-language query string.

    :param q: The raw query.
    :param max_len: Inclusive upper bound on the stripped length.
    :return: ``q`` stripped of leading and trailing whitespace.
    :raises ValueError: If empty, whitespace-only, or longer than ``max_len``.
    """
    stripped = q.strip()
    if not stripped:
        raise ValueError("q must not be empty")
    if len(stripped) > max_len:
        raise ValueError(f"q must be at most {max_len} characters, got {len(stripped)}")
    return stripped
