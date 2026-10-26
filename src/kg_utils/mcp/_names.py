"""kg_utils/mcp/_names.py

The two constants a CLI needs before anything imports ``mcp``.

Kept apart from :mod:`kg_utils.mcp.server` so that registering the standard
``<tool> mcp`` command costs nothing: importing ``mcp`` takes about half a
second, and a module's whole CLI would otherwise pay it on every command, not
just on ``mcp``.

Author: Eric G. Suchanek, PhD
License: Elastic 2.0
"""

from __future__ import annotations

__all__ = ["CANONICAL_TOOLS", "TRANSPORTS"]

#: The tools every fleet KG server answers, with the same names and meanings.
CANONICAL_TOOLS: tuple[str, ...] = ("stats", "query", "pack", "get_node", "neighbors")

#: Transports :func:`kg_utils.mcp.run` serves. HTTP+SSE is deprecated in the
#: 2026-07-28 MCP specification and is deliberately absent; streamable HTTP
#: replaces it.
TRANSPORTS: tuple[str, ...] = ("stdio", "streamable-http")
