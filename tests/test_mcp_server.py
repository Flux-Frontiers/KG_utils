"""Tests for kg_utils.mcp, the fleet's MCP server base.

Driven through a real MCP client, never by calling tool functions directly:
``mcp.shared.memory.create_connected_server_and_client_session`` runs the
actual ``Server.run()`` and lifespan against a ``ClientSession`` in-process
(the harness FLEET_STANDARDS mandates, 2026-08-24), and one test spawns the
server as a real stdio subprocess.

Everything except the last test reads the graph only: the KG is built with
``build_graph()`` and the embedding model is never available, which is what
lets this run without the ``semantic`` extra's model download and is also the
guarantee under test -- stats, get_node, neighbors and the manifest must not
load a model.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
import click  # noqa: E402
from click.testing import CliRunner  # noqa: E402
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from mcp.shared.memory import (  # noqa: E402
    create_client_server_memory_streams,
    create_connected_server_and_client_session,
)

import kg_utils  # noqa: E402
import kg_utils.pipeline as pipeline_mod  # noqa: E402
from kg_utils.extractor import KGExtractor  # noqa: E402
from kg_utils.mcp import (  # noqa: E402
    CANONICAL_TOOLS,
    BuildServer,
    Context,
    KGServer,
    ServableKG,
    add_mcp_command,
    kg_from,
    make_server,
    manifest,
    run,
)
from kg_utils.mcp.server import _configure_http  # noqa: E402
from kg_utils.pipeline import KGModule  # noqa: E402
from kg_utils.specs import EdgeSpec, NodeSpec  # noqa: E402

# ---------------------------------------------------------------------------
# A minimal KG, graph-only
# ---------------------------------------------------------------------------


class _Extractor(KGExtractor):
    def node_kinds(self) -> list[str]:
        return ["module", "function"]

    def edge_kinds(self) -> list[str]:
        return ["CONTAINS", "CALLS"]

    def meaningful_node_kinds(self) -> list[str]:
        return ["function"]

    def extract(self) -> Iterator[NodeSpec | EdgeSpec]:
        yield NodeSpec(node_id="mod", kind="module", name="mod", qualname="mod", source_path="m.py")
        for n in ("alpha", "beta"):
            yield NodeSpec(
                node_id=n,
                kind="function",
                name=n,
                qualname=n,
                source_path="m.py",
                lineno=1,
                end_lineno=1,
                docstring=f"The {n} function.",
            )
            yield EdgeSpec(source_id="mod", target_id=n, relation="CONTAINS")
        yield EdgeSpec(source_id="alpha", target_id="beta", relation="CALLS")


class _KG(KGModule):
    _default_dir = ".mkg"

    def make_extractor(self) -> KGExtractor:
        return _Extractor(self.repo_root)

    def kind(self) -> str:
        return "test"

    def analyze(self) -> str:
        return "# test"


@pytest.fixture
def built(tmp_path: Path) -> Path:
    kg = _KG(tmp_path)
    kg.build_graph(wipe=True)
    kg.close()
    return tmp_path


@pytest.fixture
def no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any attempt to load the embedding model fail loudly."""

    def _refuse(*_a: object, **_k: object) -> None:
        raise RuntimeError("embedding model load refused by test")

    monkeypatch.setattr(pipeline_mod, "SentenceTransformerEmbedder", _refuse)


def _with_client(server: KGServer, fn: Callable[[ClientSession], Awaitable[Any]]) -> Any:
    async def main() -> Any:
        async with create_connected_server_and_client_session(server) as client:
            return await fn(client)

    return anyio.run(main)


def _call(server: KGServer, tool: str, args: dict[str, Any] | None = None) -> Any:
    return _with_client(server, lambda c: c.call_tool(tool, args or {}))


def _text(result: Any) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


# ---------------------------------------------------------------------------
# Tool set
# ---------------------------------------------------------------------------


def test_lists_exactly_the_canonical_tools(built: Path) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test")
    tools = _with_client(server, lambda c: c.list_tools())
    assert [t.name for t in tools.tools] == list(CANONICAL_TOOLS)


def test_neighbors_can_be_left_out(built: Path) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test", neighbors=False)
    tools = _with_client(server, lambda c: c.list_tools())
    assert [t.name for t in tools.tools] == ["stats", "query", "pack", "get_node"]


def test_context_parameter_never_reaches_a_schema(built: Path) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test")
    for spec in manifest(server)["tools"]:
        assert "ctx" not in spec["input_schema"].get("properties", {}), spec["name"]


def test_a_kgmodule_is_servable() -> None:
    assert isinstance(_KG(Path(".")), ServableKG)


# ---------------------------------------------------------------------------
# Graph-only tools work with no model; query is the one that needs it
# ---------------------------------------------------------------------------


def test_graph_tools_never_load_the_model(built: Path, no_model: None) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test")

    async def session(c: ClientSession) -> dict[str, Any]:
        return {
            "stats": await c.call_tool("stats", {}),
            "get_node": await c.call_tool("get_node", {"node_id": "alpha"}),
            "neighbors": await c.call_tool("neighbors", {"node_id": "alpha"}),
            "query": await c.call_tool("query", {"q": "alpha"}),
        }

    r = _with_client(server, session)
    assert not r["stats"].isError
    assert r["stats"].structuredContent["nodes"]["total"] == 3
    assert not r["get_node"].isError
    assert r["get_node"].structuredContent["id"] == "alpha"
    assert not r["neighbors"].isError
    assert [n["id"] for n in r["neighbors"].structuredContent["nodes"]] == ["beta", "mod"]
    # The control: query does need the model, and the refusal surfaces as a tool error.
    assert r["query"].isError
    assert "embedding model load refused by test" in _text(r["query"])


def test_manifest_and_instructions_never_open_the_kg() -> None:
    def explode() -> ServableKG:
        raise AssertionError("the KG was opened")

    server = make_server(explode, name="mkg", kind="test", version="1.2.3")
    m = manifest(server)
    assert m["name"] == "mkg" and m["kind"] == "test" and m["version"] == "1.2.3"
    assert m["canonical"] == list(CANONICAL_TOOLS)
    assert server.instructions and "`query`" in server.instructions


# ---------------------------------------------------------------------------
# Errors: the KG validates, the server reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args", "fragment"),
    [
        ("get_node", {"node_id": "nope"}, "no node with id 'nope'"),
        ("neighbors", {"node_id": "nope"}, "no node with id 'nope'"),
        ("neighbors", {"node_id": "alpha", "limit": 0}, "limit"),
        ("query", {"q": "   "}, "q must not be empty"),
        ("query", {"q": "x", "k": 1000}, "k must be between"),
        ("pack", {"q": "x", "hop": 9}, "hop must be between"),
    ],
)
def test_bad_input_is_a_tool_error_naming_the_problem(
    built: Path, no_model: None, tool: str, args: dict[str, Any], fragment: str
) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test")
    result = _call(server, tool, args)
    assert result.isError
    assert fragment in _text(result)


def test_unknown_direction_is_rejected_by_the_schema(built: Path) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test")
    result = _call(server, "neighbors", {"node_id": "alpha", "direction": "sideways"})
    assert result.isError


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


class _Counting(_KG):
    opened = 0
    closed = 0

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        type(self).opened += 1

    def close(self) -> None:
        type(self).closed += 1
        super().close()


def test_lifespan_opens_once_and_closes(built: Path) -> None:
    _Counting.opened = _Counting.closed = 0
    server = make_server(lambda: _Counting(built), name="mkg", kind="test")
    _with_client(server, lambda c: c.call_tool("stats", {}))
    assert (_Counting.opened, _Counting.closed) == (1, 1)


def test_an_unbuilt_graph_is_refused_not_served_empty(tmp_path: Path) -> None:
    _Counting.opened = _Counting.closed = 0
    server = make_server(lambda: _Counting(tmp_path), name="mkg", kind="test")
    with pytest.raises(BaseException) as info:
        _with_client(server, lambda c: c.call_tool("stats", {}))
    assert "no graph at" in repr(info.value) or any(
        "no graph at" in str(e) for e in getattr(info.value, "exceptions", ())
    )
    assert not (tmp_path / ".mkg" / "graph.sqlite").exists(), "refusing must not create the file"
    assert _Counting.closed == 1


# ---------------------------------------------------------------------------
# Domain tools and generated instructions
# ---------------------------------------------------------------------------


def test_domain_tools_ride_along_and_are_described(built: Path) -> None:
    server = make_server(
        lambda: _KG(built), name="mkg", kind="test", instructions="Hand-written preamble."
    )

    @server.tool()
    def kind_string(ctx: Context) -> str:
        """The KG's kind, from the module's own method."""
        return kg_from(ctx).kind()

    result = _call(server, "kind_string")
    assert not result.isError and _text(result) == "test"

    instructions = server.instructions or ""
    assert instructions.startswith("Hand-written preamble.")
    assert "- `kind_string`: The KG's kind, from the module's own method." in instructions
    assert manifest(server)["tools"][-1]["name"] == "kind_string"

    server.remove_tool("kind_string")
    assert "kind_string" not in (server.instructions or "")


def test_the_session_banner_reports_resolved_paths_on_stderr(
    built: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test")
    _call(server, "stats")
    err = capsys.readouterr().err
    assert "\\n" not in err, "real newlines, not escaped ones"
    lines = err.splitlines()
    assert lines[0] == "mkg: serving"
    assert f"  repo    : {built.resolve()}" in lines
    assert f"  db      : {built.resolve() / '.mkg' / 'graph.sqlite'}" in lines
    assert f"  vectors : {built.resolve() / '.mkg' / 'vectors.sqlite'}" in lines
    assert any(ln.startswith("  model   : ") for ln in lines)


def test_name_and_version_reach_the_client(built: Path) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test", version="4.5.6")

    async def main() -> tuple[str, str]:
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            low = server._mcp_server  # noqa: SLF001 -- run the real server on memory streams
            async with anyio.create_task_group() as tg:
                tg.start_soon(lambda: low.run(*server_streams, low.create_initialization_options()))
                async with ClientSession(*client_streams) as client:
                    init = await client.initialize()
                tg.cancel_scope.cancel()
        return init.serverInfo.name, init.serverInfo.version

    assert anyio.run(main) == ("mkg", "4.5.6")


# ---------------------------------------------------------------------------
# A module's richer version of a canonical tool
# ---------------------------------------------------------------------------


def test_a_superset_query_replaces_the_canonical_one_in_place(built: Path, no_model: None) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test")

    @server.canonical("query")
    def query(
        q: str,
        ctx: Context,
        k: int = 4,  # a different default is allowed
        hop: int = 1,
        max_nodes: int = 25,
        rels: list[str] | None = None,
        min_score: float = 0.0,
        rerank_mode: str = "hybrid",  # an added optional parameter is allowed
    ) -> dict[str, Any]:
        """Richer query: answers from the module's own code."""
        return {"q": q, "k": k, "rerank_mode": rerank_mode, "kind": kg_from(ctx).kind()}

    names = [t["name"] for t in manifest(server)["tools"]]
    assert names == list(CANONICAL_TOOLS), "the replacement kept its place"
    result = _call(server, "query", {"q": "alpha"})
    assert not result.isError
    assert result.structuredContent == {
        "q": "alpha",
        "k": 4,
        "rerank_mode": "hybrid",
        "kind": "test",
    }
    assert "- `query`: Richer query: answers from the module's own code." in (
        server.instructions or ""
    )


@pytest.mark.parametrize(
    ("signature", "fragment"),
    [
        ("q: str, ctx: Context, k: int = 8", "drops parameter `hop`"),
        (
            "q: str, ctx: Context, k: str = '8', hop: int = 1, max_nodes: int = 25, "
            "rels: list[str] | None = None, min_score: float = 0.0",
            "`k` takes string, not integer",
        ),
        (
            "q: str, ctx: Context, k: int = 8, hop: int = 1, max_nodes: int = 25, "
            "rels: str = '', min_score: float = 0.0",
            "`rels` takes string",
        ),
        (
            "q: str, ctx: Context, scope: str, k: int = 8, hop: int = 1, max_nodes: int = 25, "
            "rels: list[str] | None = None, min_score: float = 0.0",
            "adds required parameter `scope`",
        ),
        (
            "ctx: Context, q: str = '', k: int = 8, hop: int = 1, max_nodes: int = 25, "
            "rels: list[str] | None = None, min_score: float = 0.0",
            "`q` must be required",
        ),
    ],
)
def test_a_query_that_breaks_the_contract_is_refused(
    built: Path, signature: str, fragment: str
) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test")
    namespace: dict[str, Any] = {"Context": Context, "Any": Any}
    exec(f"def query({signature}) -> dict[str, Any]:\n    return {{}}", namespace)  # noqa: S102
    with pytest.raises(TypeError, match="breaks the canonical contract") as info:
        server.canonical("query")(namespace["query"])
    assert fragment in str(info.value)
    # Refused means untouched: the canonical tool is still the one registered.
    assert "rerank" not in json.dumps(manifest(server)["tools"][1])


def test_a_doc_shaped_query_and_pack_satisfy_the_contract(built: Path) -> None:
    """doc_kg's search has no ``min_score`` and sizes excerpts by characters.

    The contract is what every domain shares, not the default tools' extras,
    so a version without ``min_score``, ``context`` or ``max_lines`` is fine.
    """
    server = make_server(lambda: _KG(built), name="mkg", kind="test")

    @server.canonical("query")
    def query(
        q: str,
        ctx: Context,
        k: int = 8,
        hop: int = 1,
        max_nodes: int = 25,
        rels: list[str] | None = None,
        node_kinds: list[str] | None = None,
    ) -> dict[str, Any]:
        """Doc-style query."""
        return {}

    @server.canonical("pack")
    def pack(
        q: str,
        ctx: Context,
        k: int = 8,
        hop: int = 1,
        max_nodes: int = 15,
        rels: list[str] | None = None,
        max_chars: int = 2000,
        traced: bool = False,
    ) -> str:
        """Doc-style pack."""
        return ""

    props = {t["name"]: t["input_schema"]["properties"] for t in manifest(server)["tools"]}
    assert "max_chars" in props["pack"] and "context" not in props["pack"]
    assert "min_score" not in props["query"]


def test_a_kg_with_only_stats_node_and_close_is_servable() -> None:
    from kg_utils.mcp import SearchableKG

    class Minimal:
        def stats(self) -> dict[str, Any]:
            return {}

        def node(self, node_id: str) -> dict[str, Any] | None:
            return None

        def close(self) -> None:
            pass

    assert isinstance(Minimal(), ServableKG)
    assert not isinstance(Minimal(), SearchableKG)
    assert isinstance(_KG(Path(".")), SearchableKG)


def test_changing_the_result_kind_is_refused(built: Path) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test")

    def pack_as_dict(
        q: str,
        ctx: Context,
        k: int = 8,
        hop: int = 1,
        context: int = 5,
        max_lines: int = 60,
        max_nodes: int = 15,
        rels: list[str] | None = None,
    ) -> dict[str, Any]:
        return {}

    with pytest.raises(TypeError, match="returns structured data, not text"):
        server.canonical("pack")(pack_as_dict)


def test_only_registered_canonical_tools_can_be_replaced(built: Path) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test", neighbors=False)
    with pytest.raises(ValueError, match="not a canonical tool"):
        server.canonical("callers")
    with pytest.raises(ValueError, match="not registered"):
        server.canonical("neighbors")


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def test_sse_is_refused(built: Path) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="test")
    with pytest.raises(ValueError, match="deprecated"):
        run(server, "sse")


def test_http_host_decides_rebinding_protection_like_the_constructor(built: Path) -> None:
    local = make_server(lambda: _KG(built), name="mkg", kind="test")
    _configure_http(local, "127.0.0.1", 9001)
    assert local.settings.port == 9001
    assert local.settings.transport_security is not None

    public = make_server(lambda: _KG(built), name="mkg", kind="test")
    _configure_http(public, "0.0.0.0", 9002)  # noqa: S104 -- the case under test
    assert public.settings.transport_security is None


# ---------------------------------------------------------------------------
# The standard command
# ---------------------------------------------------------------------------


def _cli(build: BuildServer) -> click.Group:
    @click.group()
    def cli() -> None:
        pass

    add_mcp_command(cli, build)
    return cli


def test_manifest_flag_answers_without_a_graph(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def build(repo: Path, db: Path | None, vectors: Path | None, model: str | None) -> KGServer:
        seen.update(repo=repo, db=db, vectors=vectors, model=model)

        def explode() -> ServableKG:
            raise AssertionError("the KG was opened")

        return make_server(explode, name="mkg", kind="test")

    result = CliRunner().invoke(
        _cli(build), ["mcp", "--repo", str(tmp_path), "--db", "g.sqlite", "--manifest"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["canonical"] == list(CANONICAL_TOOLS)
    # A relative --db is taken from --repo, not from the working directory.
    root = tmp_path.resolve()
    assert seen == {"repo": root, "db": root / "g.sqlite", "vectors": None, "model": None}


def test_absolute_paths_and_model_pass_through(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def build(repo: Path, db: Path | None, vectors: Path | None, model: str | None) -> KGServer:
        seen.update(db=db, vectors=vectors, model=model)
        return make_server(lambda: _KG(repo), name="mkg", kind="test")

    vec = tmp_path / "elsewhere" / "v.sqlite"
    args = ["mcp", "--vectors", str(vec), "--model", "BAAI/bge-small-en-v1.5", "--manifest"]
    result = CliRunner().invoke(_cli(build), args)
    assert result.exit_code == 0, result.output
    assert seen == {"db": None, "vectors": vec, "model": "BAAI/bge-small-en-v1.5"}


def test_command_offers_no_sse(tmp_path: Path) -> None:
    def build(repo: Path, db: Path | None, vectors: Path | None, model: str | None) -> KGServer:
        return make_server(lambda: _KG(repo), name="mkg", kind="test")

    result = CliRunner().invoke(_cli(build), ["mcp", "--transport", "sse"])
    assert result.exit_code == 2
    assert "sse" in result.output


# ---------------------------------------------------------------------------
# A real stdio subprocess, with tools that write to stdout
# ---------------------------------------------------------------------------

_SERVER_SCRIPT = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path

    from kg_utils.extractor import KGExtractor
    from kg_utils.mcp import Context, kg_from, make_server, run
    from kg_utils.pipeline import KGModule
    from kg_utils.specs import NodeSpec

    class X(KGExtractor):
        def node_kinds(self): return ["function"]
        def edge_kinds(self): return []
        def meaningful_node_kinds(self): return ["function"]
        def extract(self):
            yield NodeSpec(node_id="f", kind="function", name="f", qualname="f", source_path="m.py")

    class K(KGModule):
        _default_dir = ".skg"
        def make_extractor(self): return X(self.repo_root)
        def kind(self): return "stdio"
        def analyze(self): return "#"

    root = Path(sys.argv[1])
    K(root).build_graph(wipe=True)
    print("noise before the server starts")
    server = make_server(lambda: K(root), name="skg", kind="stdio")

    @server.tool()
    def noisy(ctx: Context) -> str:
        \"\"\"Write to stdout three ways, then answer.\"\"\"
        print("python print to stdout")
        sys.stdout.write("sys.stdout.write\\n")
        sys.stdout.flush()
        os.write(1, b"raw fd 1 write, as a C library would\\n")
        return kg_from(ctx).kind()

    run(server)
    """
)


def test_stdio_subprocess_survives_stdout_writes(tmp_path: Path) -> None:
    script = tmp_path / "serve.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(script), str(tmp_path)],
        # Point the child at this checkout's kg_utils, installed or not.
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(Path(kg_utils.__file__).resolve().parents[1])]
                + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
            ),
        },
    )

    async def main() -> tuple[list[str], str, bool, dict[str, Any]]:
        with open(tmp_path / "stderr.log", "w", encoding="utf-8") as errlog:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    tools = await client.list_tools()
                    noisy = await client.call_tool("noisy", {})
                    stats = await client.call_tool("stats", {})
                    return (
                        [t.name for t in tools.tools],
                        _text(noisy),
                        noisy.isError,
                        stats.structuredContent or {},
                    )

    names, answer, is_error, stats = anyio.run(main)
    assert names == [*CANONICAL_TOOLS, "noisy"]
    assert (answer, is_error) == ("stdio", False)
    assert stats["nodes"]["total"] == 1
    stderr = (tmp_path / "stderr.log").read_text(encoding="utf-8")
    for line in (
        "python print to stdout",
        "sys.stdout.write",
        "raw fd 1 write",
        "skg MCP server starting (stdio)",
        "skg: serving",
    ):
        assert line in stderr, f"{line!r} did not reach stderr"


# ---------------------------------------------------------------------------
# With the real model: query and pack return content
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_query_and_pack_with_the_real_model(tmp_path: Path) -> None:
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("sqlite_vec")
    (tmp_path / "m.py").write_text("def alpha():\n    pass\n\ndef beta():\n    alpha()\n")
    kg = _KG(tmp_path)
    kg.build(wipe=True)
    kg.close()
    server = make_server(lambda: _KG(tmp_path), name="mkg", kind="test")

    async def session(c: ClientSession) -> tuple[Any, Any]:
        return (
            await c.call_tool("query", {"q": "the alpha function", "k": 2}),
            await c.call_tool("pack", {"q": "the alpha function", "k": 2}),
        )

    q, p = _with_client(server, session)
    assert not q.isError and q.structuredContent["returned_nodes"] >= 1
    assert not p.isError and "alpha" in _text(p)


# ---------------------------------------------------------------------------
# The stats shape (kg_utils.stats), enforced on every call
# ---------------------------------------------------------------------------


class _FlatStatsKG(_KG):
    def stats(self) -> dict[str, Any]:  # the pre-0.29 flat form
        return self.store.stats()


def test_stats_answers_in_the_fleet_shape(built: Path) -> None:
    out = _call(make_server(lambda: _KG(built), name="mkg", kind="code"), "stats")
    assert not out.isError
    assert set(out.structuredContent) >= {"kind", "nodes", "edges"}
    assert out.structuredContent["nodes"]["total"] == sum(
        out.structuredContent["nodes"]["by_kind"].values()
    )


def test_a_flat_stats_is_a_tool_error(built: Path) -> None:
    out = _call(make_server(lambda: _FlatStatsKG(built), name="mkg", kind="code"), "stats")
    assert out.isError and "not in the fleet shape" in _text(out)


def test_a_modules_own_stats_is_checked_too(built: Path) -> None:
    server = make_server(lambda: _KG(built), name="mkg", kind="code")

    @server.canonical("stats")
    def stats(ctx: Context) -> dict[str, Any]:
        """A module's richer stats that forgot the shape."""
        return {"node_count": 1}

    out = _call(server, "stats")
    assert out.isError and "`nodes` must be an object" in _text(out)
    assert [t["name"] for t in manifest(server)["tools"]][0] == "stats"
