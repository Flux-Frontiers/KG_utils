"""kg_utils/types/specs.py — Core dataclasses shared by all KG modules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NodeSpec:
    """Specification for a knowledge-graph node.

    :param node_id: Unique identifier, typically ``<kind>:<path>:<qualname>``.
    :param kind: Node kind (e.g. "file", "function", "class", "directory").
    :param name: Short display name.
    :param qualname: Fully-qualified name or relative path.
    :param source_path: Path to the source file (relative to repo root).
    :param lineno: 1-based start line (source-code KGs); ``None`` for doc/domain KGs.
    :param end_lineno: 1-based end line; ``None`` if not applicable.
    :param docstring: Semantic content for vector indexing.
    :param metadata: Domain-specific extension data.
    """

    node_id: str
    kind: str
    name: str
    qualname: str
    source_path: str
    lineno: int | None = None
    end_lineno: int | None = None
    docstring: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeSpec:
    """Specification for a knowledge-graph edge.

    :param source_id: Node ID of the edge source.
    :param target_id: Node ID of the edge target.
    :param relation: Relation type (e.g. "CONTAINS", "CALLS", "IMPORTS").
    :param weight: Edge weight for PageRank / centrality (default 1.0).
    :param metadata: Domain-specific edge data (serialised as evidence JSON in the store).
    """

    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildStats:
    """Statistics returned by KGModule.build() and related methods.

    :param repo_root: Repository root that was analysed.
    :param db_path: Path to the SQLite database.
    :param total_nodes: Total nodes written to SQLite.
    :param total_edges: Total edges written to SQLite.
    :param node_counts: Node counts broken down by kind.
    :param edge_counts: Edge counts broken down by relation.
    :param indexed_rows: Number of nodes embedded into LanceDB (``None`` if not built).
    :param index_dim: Embedding dimension (``None`` if not built).
    """

    repo_root: str
    db_path: str
    total_nodes: int
    total_edges: int
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    indexed_rows: int | None = None
    index_dim: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the stats to a plain dictionary."""
        return {
            "repo_root": self.repo_root,
            "db_path": self.db_path,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "node_counts": self.node_counts,
            "edge_counts": self.edge_counts,
            "indexed_rows": self.indexed_rows,
            "index_dim": self.index_dim,
        }

    def __str__(self) -> str:
        lines = [
            f"repo_root   : {self.repo_root}",
            f"db_path     : {self.db_path}",
            f"nodes       : {self.total_nodes}  {self.node_counts}",
            f"edges       : {self.total_edges}  {self.edge_counts}",
        ]
        if self.indexed_rows is not None:
            lines.append(f"indexed     : {self.indexed_rows} vectors  dim={self.index_dim}")
        return "\n".join(lines)


@dataclass
class QueryResult:
    """Result container returned by KGModule.query().

    :param query: The original query string.
    :param nodes: List of matched node dicts.
    :param edges: List of matched edge dicts.
    :param seeds: Number of seed nodes from vector search.
    :param expanded_nodes: Number of nodes after graph expansion.
    :param returned_nodes: Number of nodes actually returned.
    :param hop: Number of hops used in graph expansion.
    :param rels: Relation types used in expansion.
    """

    query: str = ""
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    seeds: int = 0
    expanded_nodes: int = 0
    returned_nodes: int = 0
    hop: int = 0
    rels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the query result to a plain dictionary."""
        return {
            "query": self.query,
            "seeds": self.seeds,
            "expanded_nodes": self.expanded_nodes,
            "returned_nodes": self.returned_nodes,
            "hop": self.hop,
            "rels": self.rels,
            "nodes": self.nodes,
            "edges": self.edges,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        sep = "=" * 80
        print(sep)
        print(f"QUERY: {self.query}")
        print(
            f"Seeds: {self.seeds} | Expanded: {self.expanded_nodes} "
            f"| Returned: {self.returned_nodes} | hop={self.hop}"
        )
        print(f"Rels: {', '.join(self.rels)}")
        print(sep)
        for n in self.nodes:
            print(
                f"{n['kind']:8s} {(n.get('module_path') or ''):40s} "
                f"{n.get('qualname') or n['name']}  [{n['id']}]"
            )
            if n.get("docstring"):
                ds0 = n["docstring"].strip().splitlines()[0]
                print(f"    {ds0[:120]}")
            print()
        print("-" * 80)
        print(f"EDGES (within returned set): {len(self.edges)}")
        print("-" * 80)
        for e in sorted(self.edges, key=lambda x: (x["rel"], x["src"], x["dst"])):
            print(f"  {e['src']} -[{e['rel']}]-> {e['dst']}")
        print(sep)


@dataclass
class SnippetPack:
    """Result container returned by KGModule.pack().

    :param query: The original query string.
    :param seeds: Number of seed nodes from vector search.
    :param expanded_nodes: Number of nodes after graph expansion.
    :param returned_nodes: Number of nodes actually returned.
    :param hop: Number of hops used in expansion.
    :param rels: Relation types used in expansion.
    :param model: Embedding model identifier.
    :param nodes: Node dicts included in the pack (each may have a ``snippet`` key).
    :param edges: Edge dicts included in the pack.
    :param snippets: Source-code snippets (for code KGs).
    :param warnings: Non-fatal issues encountered during packing.
    """

    query: str
    seeds: int = 0
    expanded_nodes: int = 0
    returned_nodes: int = 0
    hop: int = 0
    rels: list[str] = field(default_factory=list)
    model: str = ""
    nodes: list[Any] = field(default_factory=list)
    edges: list[Any] = field(default_factory=list)
    snippets: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the snippet pack to a plain dictionary."""
        return {
            "query": self.query,
            "seeds": self.seeds,
            "expanded_nodes": self.expanded_nodes,
            "returned_nodes": self.returned_nodes,
            "hop": self.hop,
            "rels": self.rels,
            "model": self.model,
            "nodes": self.nodes,
            "edges": self.edges,
            "warnings": self.warnings,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Render the snippet pack as a Markdown context document."""
        out: list[str] = []
        out.append("# KGModule Snippet Pack\n")
        out.append(f"**Query:** `{self.query}`  ")
        out.append(f"**Seeds:** {self.seeds}  ")
        out.append(f"**Expanded nodes:** {self.expanded_nodes} (returned: {self.returned_nodes})  ")
        out.append(f"**hop:** {self.hop}  ")
        out.append(f"**rels:** {', '.join(self.rels)}  ")
        out.append(f"**model:** {self.model}  ")
        out.append("\n---\n")

        if self.warnings:
            out.append("## Warnings\n")
            for w in self.warnings:
                out.append(f"- {w}")
            out.append("")

        out.append("## Nodes\n")
        for n in self.nodes:
            out.append(f"### {n['kind']} — `{n.get('qualname') or n['name']}`")
            out.append(f"- id: `{n['id']}`")
            if n.get("module_path"):
                out.append(f"- module: `{n['module_path']}`")
            if n.get("lineno") is not None:
                out.append(f"- line: {n['lineno']}")
            if n.get("docstring"):
                ds0 = n["docstring"].strip().splitlines()[0]
                out.append(f"- doc: {ds0[:140]}")
            if n.get("relevance"):
                rel = n["relevance"]
                _score = rel.get("score", 0.0)
                _tier = "HIGH" if _score >= 0.60 else ("MEDIUM" if _score >= 0.45 else "LOW")
                out.append(
                    "- relevance: "
                    f"{_score:.3f} [{_tier}] "
                    f"(semantic={rel.get('semantic', 0.0):.3f}, "
                    f"lexical={rel.get('lexical', 0.0):.3f}, "
                    f"docstring_signal={rel.get('docstring_signal', 0.0):.3f}, "
                    f"hop={rel.get('hop', 0)})"
                )
            sn = n.get("snippet")
            if sn:
                end_lineno = n.get("end_lineno")
                if end_lineno is not None and sn["end"] < end_lineno:
                    out.append(
                        f"*(truncated: showing lines {sn['start']}–{sn['end']} "
                        f"of {n.get('lineno', sn['start'])}–{end_lineno})*"
                    )
                out.append("")
                out.append(f"```\n{sn['text']}\n```")
            out.append("")

        out.append("\n---\n")
        out.append("## Edges\n")
        for e in self.edges:
            out.append(f"- `{e['src']}` -[{e['rel']}]-> `{e['dst']}`")
        out.append("")
        return "\n".join(out)

    def save(self, path: str | Path, *, fmt: str = "md") -> None:
        """Write the pack to a file.

        :param path: Output file path.
        :param fmt: ``"md"`` for Markdown or ``"json"`` for JSON.
        """
        text = self.to_markdown() if fmt == "md" else self.to_json()
        Path(path).write_text(text, encoding="utf-8")
