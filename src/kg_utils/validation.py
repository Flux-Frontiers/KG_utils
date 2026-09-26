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
    """Validate that a value is an integer within an inclusive range.

    Strict about type, because these arguments arrive as JSON from a model or
    a form, where loose input is the normal case: a string (``"8"``), a bool
    (``True`` would read as ``1``) or a truncating float (``3.7``) is rejected
    with a clear message instead of failing deeper in or passing silently.  An
    integral value of another numeric type (``3.0``, a NumPy integer) is
    accepted and returned as a plain ``int``.  After ``connectome_kg``'s copy,
    the strictest of the three the fleet carried.

    :param name: Parameter name, used in the error message.
    :param value: The value to validate.
    :param minimum: Inclusive lower bound.
    :param maximum: Inclusive upper bound.
    :return: ``value`` as an ``int``.
    :raises ValueError: If ``value`` is not an integer, or is outside
        ``[minimum, maximum]``.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        ivalue = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if ivalue != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if not (minimum <= ivalue <= maximum):
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {ivalue}")
    return ivalue


def require_query(q: str, max_len: int = MAX_QUERY_LEN) -> str:
    """Validate a natural-language query string.

    :param q: The raw query.
    :param max_len: Inclusive upper bound on the stripped length.
    :return: ``q`` stripped of leading and trailing whitespace.
    :raises ValueError: If not a string, empty, whitespace-only, or longer
        than ``max_len``.
    """
    if not isinstance(q, str):
        raise ValueError(f"q must be a string, got {type(q).__name__}")
    stripped = q.strip()
    if not stripped:
        raise ValueError("q must not be empty")
    if len(stripped) > max_len:
        raise ValueError(f"q must be at most {max_len} characters, got {len(stripped)}")
    return stripped
