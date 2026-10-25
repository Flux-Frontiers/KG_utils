"""kg_utils/mcp/server.py

One MCP server for every knowledge-graph module in the fleet.

A module hands :func:`make_server` a zero-argument factory that opens its KG,
and gets back a :class:`KGServer` that already carries the canonical tools:

======================  =====================================================
``stats``               node and edge counts; never loads the embedding model
``query``               hybrid search, ranked nodes and the edges among them
``pack``                hybrid search rendered as a Markdown context pack
``get_node``            one node by id
``neighbors``           a node's one-hop neighbourhood (optional, on by default)
======================  =====================================================

The names are the same on every server. An agent sees them prefixed by the
server name (``mcp__pycodekg__query``, ``mcp__dockg__query``), so a per-kind
spelling such as ``query_codebase`` or ``query_docs`` told it nothing the
prefix did not, and made every skill describe a different vocabulary. Domain
tools are added to the returned server with the ordinary ``@server.tool()``
decorator and keep domain names (``callers``, ``find_path``). A module whose
own version of a canonical tool does more -- pycode_kg's ``query`` ranks with
a hybrid score and caps results per module -- replaces it with
``@server.canonical("query")``, under the same name and only if it keeps
every canonical parameter (:meth:`KGServer.canonical`).

What lives here and what does not. Everything about the *graph* stays in the
module's own methods: argument bounds (:mod:`kg_utils.validation`), lazy model
loading, ``close()``. This file holds only what is about the *wire*: tool
registration, the lifespan that opens and closes the KG, the generated
instructions, the manifest, and the transport. If a line here would also make
sense in a CLI command, it belongs in the KG instead.

This is the one file in the fleet that names ``FastMCP``. When ``mcp`` 2.0's
rename to ``MCPServer`` is taken (kgrag_priv/docs/MCP_MIGRATION.md), it is
taken here, and no module changes. The private ``_mcp_server`` and
``_tool_manager`` attributes are touched only through :func:`_lowlevel` and
:meth:`KGServer.tool_specs`, for the same reason.

Author: Eric G. Suchanek, PhD
License: Elastic 2.0
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.tools import Tool
from mcp.server.lowlevel.server import Server as LowLevelServer
from mcp.server.stdio import stdio_server

from kg_utils.specs import QueryResult, SnippetPack

__all__ = [
    "CANONICAL_TOOLS",
    "TRANSPORTS",
    "KGServer",
    "ServableKG",
    "describe_tools",
    "kg_from",
    "make_server",
    "manifest",
    "run",
]

#: The tools every fleet KG server answers, with the same names and meanings.
CANONICAL_TOOLS: tuple[str, ...] = ("stats", "query", "pack", "get_node", "neighbors")

#: Transports :func:`run` serves. HTTP+SSE is deprecated in the 2026-07-28 MCP
#: specification and is deliberately absent; streamable HTTP replaces it.
TRANSPORTS: tuple[str, ...] = ("stdio", "streamable-http")

#: Hosts for which FastMCP turns on DNS-rebinding protection by itself.
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


@runtime_checkable
class ServableKG(Protocol):
    """What a KG must offer to be served.

    :class:`kg_utils.pipeline.KGModule` satisfies it as it stands. A module
    with its own pipeline satisfies it by providing these five methods with
    these keyword arguments; ``neighbors()`` is optional (see
    :func:`make_server`). Each method validates its own arguments and raises
    ``ValueError`` on bad input; the server never re-checks them.
    """

    def stats(self) -> dict[str, Any]:
        """Return graph statistics. Must not load the embedding model."""
        ...

    def query(self, q: str, **kwargs: Any) -> QueryResult:
        """Hybrid query: semantic seeds expanded through the graph."""
        ...

    def pack(self, q: str, **kwargs: Any) -> SnippetPack:
        """Hybrid query with source-grounded snippets."""
        ...

    def node(self, node_id: str) -> dict[str, Any] | None:
        """Return one node, or ``None`` if the id is not in the graph."""
        ...

    def close(self) -> None:
        """Release the graph's resources."""
        ...


class KGServer(FastMCP):
    """A FastMCP server that knows which KG it serves and describes its own tools.

    It is a real :class:`~mcp.server.fastmcp.FastMCP`: ``@server.tool()``,
    ``server.settings`` and the in-memory test harness work unchanged. Build
    one with :func:`make_server`, not directly.

    The server's ``instructions`` -- the text a client shows the model on
    connect -- are regenerated every time a tool is added or removed: the
    module's own preamble followed by one line per tool. The fleet rule that
    a tool change must update the instructions in the same commit therefore
    holds by construction.

    :param name: Server name, the tool binary's name (``pycodekg``).
    :param kind: KG kind (``code``, ``doc``), reported in the manifest.
    :param version: The serving package's version, reported to clients.
    :param preamble: Hand-written text placed before the generated tool list.
    :param lifespan: Opens the KG on start and closes it on shutdown.
    """

    def __init__(
        self,
        name: str,
        *,
        kind: str,
        version: str = "",
        preamble: str | None = None,
        lifespan: Callable[[FastMCP], contextlib.AbstractAsyncContextManager[Any]] | None = None,
    ) -> None:
        # Set before FastMCP.__init__, which may register tools through add_tool().
        self.kind = kind
        self.version = version
        self._preamble = (preamble or "").strip()
        super().__init__(name, lifespan=lifespan)
        low = _lowlevel(self)
        if version:
            low.version = version
        self._refresh_instructions()

    def add_tool(self, fn: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Register a tool, then regenerate the instructions.

        Arguments pass straight through to :meth:`FastMCP.add_tool`.
        """
        super().add_tool(fn, *args, **kwargs)
        self._refresh_instructions()

    def remove_tool(self, name: str) -> None:
        """Remove a tool, then regenerate the instructions."""
        super().remove_tool(name)
        self._refresh_instructions()

    def canonical(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Replace a canonical tool with this module's richer version of it.

        The name stays, so the vocabulary stays the same on every server; what
        a module may change is everything an agent does not rely on. The
        replacement must keep each canonical parameter with the same JSON type
        and the same required-ness, add no new required parameter, and return
        the same kind of result (structured data or text). Defaults may differ
        and optional parameters may be added: pycode_kg's ``query`` defaults to
        hybrid reranking and adds ``paths`` and ``max_per_module``.

        Checked when the module builds its server, so a broken contract fails
        the module's own tests rather than an agent's call. The tool keeps its
        place in the tool list.

        :param name: One of :data:`CANONICAL_TOOLS` registered on this server.
        :return: A decorator that registers the function under *name*.
        :raises ValueError: If *name* is not a canonical tool registered here.
        :raises TypeError: If the function breaks the canonical contract; the
            message lists every way it does.
        """
        if name not in CANONICAL_TOOLS:
            raise ValueError(f"{name!r} is not a canonical tool; add it with @server.tool()")
        current = self._tool_manager.get_tool(name)
        if current is None:
            raise ValueError(f"canonical tool {name!r} is not registered on {self.name!r}")

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            candidate = Tool.from_function(fn, name=name)
            problems = _contract_problems(current, candidate)
            if problems:
                raise TypeError(
                    f"{self.name}: this `{name}` breaks the canonical contract: "
                    + "; ".join(problems)
                )
            # Replace in place: the manager keeps tools in an ordered dict, so
            # the canonical tool keeps its position in tools/list.
            self._tool_manager._tools[name] = candidate  # noqa: SLF001
            self._refresh_instructions()
            return fn

        return decorator

    def tool_specs(self) -> list[dict[str, Any]]:
        """Return each registered tool's name, description and input schema.

        Synchronous, unlike :meth:`FastMCP.list_tools`, so the manifest can be
        built without an event loop and without opening the KG.

        :return: One dict per tool, in registration order.
        """
        return [
            {
                "name": t.name,
                "description": (t.description or "").strip(),
                "input_schema": t.parameters,
            }
            for t in self._tool_manager.list_tools()
        ]

    def _refresh_instructions(self) -> None:
        text = describe_tools(self)
        _lowlevel(self).instructions = f"{self._preamble}\n\n{text}" if self._preamble else text


def _json_types(schema: dict[str, Any]) -> frozenset[str]:
    """The JSON types a parameter schema admits, arrays qualified by item type."""
    options = schema.get("anyOf") or [schema]
    out: set[str] = set()
    for opt in options:
        kind = opt.get("type", "any")
        if kind == "array":
            kind = f"array<{opt.get('items', {}).get('type', 'any')}>"
        elif "enum" in opt and kind == "any":
            kind = "string"
        out.add(kind)
    return frozenset(out)


def _contract_problems(canonical: Tool, candidate: Tool) -> list[str]:
    """Every way *candidate* fails to stand in for *canonical*; empty if it can."""
    base_props: dict[str, Any] = canonical.parameters.get("properties", {})
    new_props: dict[str, Any] = candidate.parameters.get("properties", {})
    base_req = set(canonical.parameters.get("required", []))
    new_req = set(candidate.parameters.get("required", []))
    problems: list[str] = []
    for param, schema in base_props.items():
        if param not in new_props:
            problems.append(f"drops parameter `{param}`")
            continue
        before, after = _json_types(schema), _json_types(new_props[param])
        if before != after:
            problems.append(
                f"`{param}` takes {'|'.join(sorted(after))}, not {'|'.join(sorted(before))}"
            )
        if (param in base_req) != (param in new_req):
            problems.append(f"`{param}` must {'' if param in base_req else 'not '}be required")
    for param in sorted(new_req - set(base_props)):
        problems.append(f"adds required parameter `{param}`")
    before, after = _result_kind(canonical), _result_kind(candidate)
    if before != after:
        problems.append(f"returns {after}, not {before}")
    return problems


def _result_kind(tool: Tool) -> str:
    """What a tool's result is, as an agent sees it.

    FastMCP wraps a plain return type (``str``, ``list[...]``) in a one-field
    ``{"result": ...}`` schema, so the presence of an output schema cannot tell
    text from structured data; the wrapped field's type can.
    """
    schema = tool.output_schema
    if schema is None:
        return "unstructured content"
    if getattr(tool.fn_metadata, "wrap_output", False):
        kind = schema.get("properties", {}).get("result", {}).get("type", "any")
        return {"string": "text", "array": "a list"}.get(kind, kind)
    return "structured data"


def _lowlevel(server: FastMCP) -> LowLevelServer:
    """The low-level server under a FastMCP. The one sanctioned private access."""
    return server._mcp_server  # noqa: SLF001


def kg_from(ctx: Context) -> Any:
    """Return the KG the current session opened.

    For domain tools added with ``@server.tool()``: declare a ``ctx: Context``
    parameter (FastMCP injects it and keeps it out of the tool's schema) and
    call this to reach the KG. The return type is ``Any`` so a domain tool can
    call its module's own methods without a cast.

    :param ctx: The injected request context.
    :return: The object the server's ``open_kg`` factory returned.
    """
    return ctx.request_context.lifespan_context


def describe_tools(server: KGServer) -> str:
    """Render the server's tools as the plain-text index used in its instructions.

    :param server: The server to describe.
    :return: A heading line, then one ``- `name`: summary`` line per tool.
    """
    specs = server.tool_specs()
    lines = [f"Tools on the `{server.name}` server ({server.kind} knowledge graph):"]
    for spec in specs:
        summary = spec["description"].splitlines()[0] if spec["description"] else ""
        lines.append(f"- `{spec['name']}`: {summary}")
    names = {s["name"] for s in specs}
    present = [n for n in CANONICAL_TOOLS if n in names]
    if present:
        lines.append("")
        lines.append(
            f"{', '.join(f'`{n}`' for n in present)} mean the same on every fleet KG "
            "server; the other tools are specific to this one."
        )
    return "\n".join(lines)


def manifest(server: KGServer) -> dict[str, Any]:
    """Return the server's identity and full tool list as JSON-ready data.

    Never opens the KG, so ``<tool> mcp --manifest`` answers without a graph
    on disk and without loading a model. Plugins and skill tool references are
    generated from this.

    :param server: The server to describe.
    :return: ``{"name", "kind", "version", "canonical", "tools"}``.
    """
    specs = server.tool_specs()
    names = {s["name"] for s in specs}
    return {
        "name": server.name,
        "kind": server.kind,
        "version": server.version,
        "canonical": [n for n in CANONICAL_TOOLS if n in names],
        "tools": specs,
    }


def make_server(
    open_kg: Callable[[], ServableKG],
    *,
    name: str,
    kind: str,
    version: str = "",
    instructions: str | None = None,
    neighbors: bool = True,
) -> KGServer:
    """Build a server for one KG, with the canonical tools registered.

    *open_kg* is called once per session, when the session starts, and the
    KG it returns is closed when the session ends. Building the server never
    calls it, so a module can build a server just to print its manifest.

    A KG whose ``db_path`` attribute names a file that does not exist is
    refused at session start rather than served: opening it would create an
    empty graph and answer every query with nothing, which is what an unbuilt
    vault did on 2026-09-22.

    :param open_kg: Zero-argument factory returning the KG to serve.
    :param name: Server name, the tool binary's name (``pycodekg``).
    :param kind: KG kind (``code``, ``doc``).
    :param version: The serving package's version, reported to clients.
    :param instructions: Hand-written preamble for the server's instructions;
        the tool list is generated and appended.
    :param neighbors: Register ``neighbors``. Pass ``False`` for a KG whose
        class has no ``neighbors()`` method.
    :return: A :class:`KGServer`; add domain tools with ``@server.tool()``.
    """

    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[ServableKG]:
        kg = open_kg()
        db_path = getattr(kg, "db_path", None)
        if isinstance(db_path, (str, Path)) and not Path(db_path).exists():
            kg.close()
            raise RuntimeError(
                f"{name}: no graph at {db_path}. Build it before serving it, "
                f"or point --repo/--db at the built one."
            )
        try:
            yield kg
        finally:
            kg.close()

    server = KGServer(name, kind=kind, version=version, preamble=instructions, lifespan=lifespan)
    _register_canonical(server, with_neighbors=neighbors)
    return server


def _register_canonical(server: KGServer, *, with_neighbors: bool) -> None:
    """Add the canonical tools. Validation is the KG's; errors become tool errors."""

    @server.tool()
    def stats(ctx: Context) -> dict[str, Any]:
        """Node and edge counts by kind, and the vector backend in use.

        Reads the graph only: never loads the embedding model, so it is the
        cheap first call of a session.
        """
        return kg_from(ctx).stats()

    @server.tool()
    def query(
        q: str,
        ctx: Context,
        k: int = 8,
        hop: int = 1,
        max_nodes: int = 25,
        rels: list[str] | None = None,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Hybrid search: semantic seeds expanded through the graph, ranked.

        Returns the matching nodes (with relevance scores and how they were
        reached) and the edges among them. Use `pack` instead when you want
        the source text itself.

        :param q: Natural-language query.
        :param k: Semantic seeds to start from (1-100).
        :param hop: Graph hops to expand from each seed (0 = semantic only, max 5).
        :param max_nodes: Most nodes to return.
        :param rels: Edge types to follow; omit for the graph's defaults.
        :param min_score: Drop seeds scoring below this (0.0-1.0).
        """
        kwargs: dict[str, Any] = {
            "k": k,
            "hop": hop,
            "max_nodes": max_nodes,
            "min_score": min_score,
        }
        if rels:
            kwargs["rels"] = tuple(rels)
        return kg_from(ctx).query(q, **kwargs).to_dict()

    @server.tool()
    def pack(
        q: str,
        ctx: Context,
        k: int = 8,
        hop: int = 1,
        context: int = 5,
        max_lines: int = 60,
        max_nodes: int = 15,
        rels: list[str] | None = None,
    ) -> str:
        """Hybrid search rendered as a Markdown context pack of source-grounded snippets.

        The answer to "show me the code/text about X": each hit with its
        location and the lines around it, deduplicated across overlapping spans.

        :param q: Natural-language query.
        :param k: Semantic seeds to start from (1-100).
        :param hop: Graph hops to expand from each seed (max 5).
        :param context: Extra lines around each snippet.
        :param max_lines: Longest snippet, in lines.
        :param max_nodes: Most snippets to return.
        :param rels: Edge types to follow; omit for the graph's defaults.
        """
        kwargs: dict[str, Any] = {
            "k": k,
            "hop": hop,
            "context": context,
            "max_lines": max_lines,
            "max_nodes": max_nodes,
        }
        if rels:
            kwargs["rels"] = tuple(rels)
        return kg_from(ctx).pack(q, **kwargs).to_markdown()

    @server.tool()
    def get_node(node_id: str, ctx: Context) -> dict[str, Any]:
        """One node by id, with its kind, location, docstring and metadata.

        :param node_id: The node's id, as `query`, `pack` or `neighbors` returned it.
        """
        node = kg_from(ctx).node(node_id)
        if node is None:
            raise ValueError(f"no node with id {node_id!r}")
        return node

    if not with_neighbors:
        return

    @server.tool()
    def neighbors(
        node_id: str,
        ctx: Context,
        rels: list[str] | None = None,
        direction: Literal["out", "in", "both"] = "both",
        limit: int = 50,
    ) -> dict[str, Any]:
        """A node's one-hop neighbourhood: its edges and the nodes at their far ends.

        Reads the graph only, so it never loads the embedding model.

        :param node_id: The node whose neighbours to return.
        :param rels: Edge types to follow; omit for all of them.
        :param direction: "out" (edges from the node), "in" (edges to it) or "both".
        :param limit: Most edges to return (1-500).
        """
        result = kg_from(ctx).neighbors(node_id, rels=rels, direction=direction, limit=limit)
        if result is None:
            raise ValueError(f"no node with id {node_id!r}")
        return result


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[io.TextIOWrapper | None]:
    """Reserve the real stdout for the protocol and send everything else to stderr.

    Under stdio, file descriptor 1 *is* the JSON-RPC stream, so one stray
    ``print()`` -- or a C library's diagnostics, as pymupdf writes when it
    parses a PDF (sweep item 87) -- corrupts the session. This duplicates fd 1
    for the protocol, then points fd 1 at stderr for the server's lifetime.
    Working at the descriptor level catches C extensions and subprocesses, not
    just Python's ``sys.stdout``.

    Yields ``None`` when the process has no real descriptors to work with
    (an embedding host or test runner replaced the streams); the transport
    then uses whatever ``sys.stdout`` is, as FastMCP itself would.
    """
    try:
        out_fd = sys.stdout.fileno()
        err_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        yield None
        return
    sys.stdout.flush()
    protocol_fd = os.dup(out_fd)
    os.dup2(err_fd, out_fd)
    protocol = io.TextIOWrapper(
        io.BufferedWriter(io.FileIO(protocol_fd, "wb", closefd=True)), encoding="utf-8"
    )
    try:
        yield protocol
    finally:
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        with contextlib.suppress(Exception):
            protocol.flush()
        os.dup2(protocol_fd, out_fd)
        with contextlib.suppress(Exception):
            protocol.close()


async def _run_stdio(server: KGServer) -> None:
    with _stdout_to_stderr() as protocol:
        stdout = anyio.wrap_file(protocol) if protocol is not None else None
        async with stdio_server(stdout=stdout) as (read_stream, write_stream):
            low = _lowlevel(server)
            await low.run(read_stream, write_stream, low.create_initialization_options())


def _configure_http(server: KGServer, host: str, port: int) -> None:
    """Point the server at *host*:*port* as if it had been constructed with them.

    FastMCP decides DNS-rebinding protection in its constructor from the host
    it was given: on for a loopback host, off otherwise. The server is built
    before the CLI knows the host, so the same decision is made here.
    """
    server.settings.host = host
    server.settings.port = port
    if host not in _LOCAL_HOSTS:
        server.settings.transport_security = None


def run(
    server: KGServer,
    transport: str = "stdio",
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Serve *server* until the client disconnects (stdio) or the process is stopped.

    :param server: A server from :func:`make_server`.
    :param transport: ``"stdio"`` (what an agent's MCP config launches) or
        ``"streamable-http"``.
    :param host: Bind address for streamable HTTP; ignored for stdio.
    :param port: Port for streamable HTTP; ignored for stdio.
    :raises ValueError: On any other transport, including the deprecated ``"sse"``.
    """
    if transport not in TRANSPORTS:
        raise ValueError(
            f"transport must be one of {', '.join(TRANSPORTS)}, got {transport!r}"
            + (" (HTTP+SSE is deprecated in the MCP specification)" if transport == "sse" else "")
        )
    if transport == "stdio":
        anyio.run(_run_stdio, server)
        return
    _configure_http(server, host, port)
    server.run("streamable-http")
