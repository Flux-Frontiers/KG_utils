"""kg_utils.mcp -- one MCP server base for every knowledge-graph module.

Needs the ``mcp`` extra: ``pip install 'kgmodule-utils[mcp]'``.

A module's whole server::

    from kg_utils.mcp import Context, add_mcp_command, kg_from, make_server

    def build_server(repo, db, vectors, model):
        kwargs = {"model": model} if model else {}
        server = make_server(
            lambda: PyCodeKG(repo, db, vectors, **kwargs),
            name="pycodekg",
            kind="code",
            version=__version__,
        )

        @server.tool()
        def callers(node_id: str, ctx: Context) -> list[dict]:
            \"\"\"Every node that calls this one.\"\"\"
            return kg_from(ctx).callers(node_id)

        @server.canonical("query")   # same name, every canonical parameter kept
        def query(q: str, ctx: Context, k: int = 8, hop: int = 1, max_nodes: int = 25,
                  rels: list[str] | None = None, min_score: float = 0.0,
                  rerank_mode: str = "hybrid") -> dict:
            \"\"\"Hybrid search, ranked by blended semantic and lexical score.\"\"\"
            ...

        return server

    add_mcp_command(cli, build_server)   # `pycodekg mcp --repo . [--manifest]`

The server answers the canonical tools -- ``stats``, ``query``, ``pack``,
``get_node``, ``neighbors`` -- under the same names on every fleet server, plus
whatever the module adds. See :mod:`kg_utils.mcp.server` for what belongs here
and what stays in the KG, and kgrag_priv/docs/MCP_CENTRALIZATION_PLAN.md for
the decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kg_utils.mcp._names import CANONICAL_TOOLS, TRANSPORTS
from kg_utils.mcp.cli import BuildServer, add_mcp_command

if TYPE_CHECKING:
    from kg_utils.mcp.server import (
        Context,
        KGServer,
        SearchableKG,
        ServableKG,
        describe_tools,
        kg_from,
        make_server,
        manifest,
        run,
    )

__all__ = [
    "CANONICAL_TOOLS",
    "TRANSPORTS",
    "BuildServer",
    "Context",
    "KGServer",
    "SearchableKG",
    "ServableKG",
    "add_mcp_command",
    "describe_tools",
    "kg_from",
    "make_server",
    "manifest",
    "run",
]

#: Loaded from :mod:`kg_utils.mcp.server` on first use, so that importing this
#: package -- which a module's CLI does to register ``add_mcp_command`` -- does
#: not import ``mcp`` (about half a second) on every command.
_SERVER_NAMES = frozenset(
    {
        "Context",
        "KGServer",
        "SearchableKG",
        "ServableKG",
        "describe_tools",
        "kg_from",
        "make_server",
        "manifest",
        "run",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _SERVER_NAMES:
        raise AttributeError(f"module 'kg_utils.mcp' has no attribute {name!r}")
    try:
        from kg_utils.mcp import server  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover -- depends on install
        raise ImportError(
            "kg_utils.mcp needs the mcp extra: pip install 'kgmodule-utils[mcp]'"
        ) from exc
    return getattr(server, name)
