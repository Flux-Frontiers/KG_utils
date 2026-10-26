"""kg_utils/mcp/cli.py

The standard ``<tool> mcp`` command, so every fleet server takes the same flags.

Before this, the fleet's servers were launched six different ways --
``pycodekg mcp --repo .``, ``dockg mcp --db ... --lancedb ...``, a separate
``diarykg-mcp`` script, ``metabokg mcp --db`` with no ``--repo`` at all -- and
the 2026-08-25 installer audit found every MCP config for two of them wrong.
One command shape, built here, ends that:

.. code-block:: text

    <tool> mcp [--repo DIR] [--db FILE] [--vectors FILE] [--model NAME]
               [--transport stdio|streamable-http] [--host H] [--port N]
               [--manifest]

``--repo``, ``--db``, ``--vectors`` and ``--model`` are the
:class:`~kg_utils.pipeline.KGModule` constructor's own arguments. ``--db`` and
``--vectors`` default to the module's paths under ``--repo``, and a relative
one is taken relative to ``--repo``, so a config that names
``--repo /abs/repo --db .mykg/graph.sqlite`` works from any directory.
``--model`` defaults to the module's own model. ``--manifest`` prints the
server's tool list as JSON and exits without opening the graph.

Author: Eric G. Suchanek, PhD
License: Elastic 2.0
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import click

from kg_utils.mcp._names import TRANSPORTS

if TYPE_CHECKING:
    from kg_utils.mcp.server import KGServer

__all__ = ["BuildServer", "add_mcp_command"]

#: ``(repo, db, vectors, model) -> server``. Each of ``db``, ``vectors`` and
#: ``model`` is ``None`` when its flag was not given, meaning "the module's
#: default"; ``db`` and ``vectors`` arrive absolute.
BuildServer = Callable[[Path, Path | None, Path | None, str | None], "KGServer"]


def add_mcp_command(
    cli: click.Group, build_server: BuildServer, *, name: str = "mcp"
) -> click.Command:
    """Add the standard ``mcp`` subcommand to a module's click group.

    Registering the command does not import ``mcp``; that happens only when
    the command runs, so a module's other commands start as fast as before.

    *build_server* must not open the KG: it is called for ``--manifest`` too,
    which has to answer without a graph on disk. :func:`~kg_utils.mcp.make_server`
    takes a factory for exactly this reason, so a module's builder is just::

        def build_server(repo, db, vectors, model):
            kwargs = {"model": model} if model else {}
            server = make_server(lambda: MyKG(repo, db, vectors, **kwargs), name="mykg", kind="my")
            # @server.tool() domain extras here
            return server

        add_mcp_command(cli, build_server)

    :param cli: The module's top-level click group.
    :param build_server: Builds the server from the command's path options.
    :param name: Subcommand name.
    :return: The registered command.
    """

    @cli.command(name)
    @click.option(
        "--repo",
        type=click.Path(file_okay=False, path_type=Path),
        default=".",
        show_default=True,
        help="Repository (or corpus) root the graph was built from.",
    )
    @click.option(
        "--db",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Graph database. Default: the module's path under --repo.",
    )
    @click.option(
        "--vectors",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Vector store. Default: the module's path under --repo.",
    )
    @click.option(
        "--model",
        default=None,
        help="Embedding model. Default: the module's own.",
    )
    @click.option(
        "--transport",
        type=click.Choice(TRANSPORTS),
        default="stdio",
        show_default=True,
        help="stdio for an agent's MCP config; streamable-http to serve over HTTP.",
    )
    @click.option(
        "--host", default="127.0.0.1", show_default=True, help="streamable-http bind address."
    )
    @click.option("--port", default=8765, show_default=True, type=int, help="streamable-http port.")
    @click.option(
        "--manifest",
        "show_manifest",
        is_flag=True,
        help="Print the server's tools as JSON and exit, without opening the graph.",
    )
    def mcp_command(
        repo: Path,
        db: Path | None,
        vectors: Path | None,
        model: str | None,
        transport: str,
        host: str,
        port: int,
        show_manifest: bool,
    ) -> None:
        """Serve this knowledge graph to an AI agent over MCP."""
        from kg_utils.mcp.server import manifest, run  # noqa: PLC0415 -- see add_mcp_command

        root = repo.resolve()

        def under_repo(path: Path | None) -> Path | None:
            return None if path is None else path if path.is_absolute() else root / path

        server = build_server(root, under_repo(db), under_repo(vectors), model)
        if show_manifest:
            click.echo(json.dumps(manifest(server), indent=2))
            return
        run(server, transport, host=host, port=port)

    return mcp_command
