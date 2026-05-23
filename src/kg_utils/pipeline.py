"""kg_utils/pipeline.py — Concrete KGModule base class for production-grade KG modules.

Provides the full build/query/pack infrastructure so domain authors only
need to implement :meth:`KGModule.make_extractor`, :meth:`KGModule.kind`,
and :meth:`KGModule.analyze`.

Optional dependencies
---------------------
  lancedb, numpy, sentence-transformers

Install with: pip install 'kgmodule-utils[semantic]'

Typical domain usage::

    from kg_utils.pipeline import KGModule
    from my_domain.extractor import MyExtractor

    class MyKG(KGModule):
        _default_dir = ".mykg"

        def make_extractor(self):
            return MyExtractor(self.repo_root)

        def kind(self):
            return "my_domain"

        def analyze(self):
            return "# MyKG Analysis\\n..."

    kg = MyKG("/path/to/repo")
    kg.build(wipe=True)
    result = kg.query("authentication middleware")
    pack = kg.pack("error handling")
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from kg_utils.semantic import (
    DEFAULT_MODEL,
    Embedder,
    SemanticIndex,
    SentenceTransformerEmbedder,
    suppress_ingestion_logging,
)
from kg_utils.store import DEFAULT_RELS, GraphStore
from kg_utils.extractor import KGExtractor
from kg_utils.specs import BuildStats, EdgeSpec, NodeSpec, QueryResult, SnippetPack


# ---------------------------------------------------------------------------
# Pure utility functions (domain-agnostic)
# ---------------------------------------------------------------------------


def semantic_score_from_distance(distance: float) -> float:
    """Convert embedding distance to a bounded similarity-like score in [0, 1].

    :param distance: Distance value returned by vector search.
    :return: Score where higher is better.
    """
    d = max(0.0, float(distance))
    return 1.0 / (1.0 + d)


def query_tokens(query: str) -> set[str]:
    """Tokenize a query for lightweight lexical overlap scoring.

    :param query: Raw query text.
    :return: Lower-cased alphanumeric tokens with length >= 2.
    """
    tokens: set[str] = set()
    for tok in re.findall(r"[A-Za-z0-9_]+", query.lower()):
        if len(tok) >= 2:
            tokens.add(tok)
        if "_" in tok:
            for part in tok.split("_"):
                if len(part) >= 2:
                    tokens.add(part)
    return tokens


def normalize_query_text(query: str) -> str:
    """Normalize query text for semantic retrieval.

    :param query: Raw user query.
    :return: Normalized query string.
    """
    return re.sub(r"[_-]+", " ", query).strip()


def docstring_signal(docstring: str | None) -> float:
    """Estimate docstring signal quality in [0, 1].

    :param docstring: Node docstring text.
    :return: Signal score.
    """
    if not docstring:
        return 0.0
    tokens = re.findall(r"[A-Za-z0-9_]+", docstring.lower())
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    length_score = min(1.0, len(tokens) / 40.0)
    return max(0.0, min(1.0, 0.6 * length_score + 0.4 * unique_ratio))


def lexical_overlap_score(query_tokens_set: set[str], node: dict[str, Any]) -> float:
    """Compute lexical overlap between query tokens and node text features.

    :param query_tokens_set: Tokenized query terms.
    :param node: Node dictionary from the store.
    :return: Overlap score in [0, 1].
    """
    if not query_tokens_set:
        return 0.0
    haystack = " ".join(
        [
            str(node.get("name") or ""),
            str(node.get("qualname") or ""),
            str(node.get("module_path") or ""),
            str(node.get("docstring") or ""),
        ]
    ).lower()
    node_toks = set(re.findall(r"[A-Za-z0-9_]+", haystack))
    if not node_toks:
        return 0.0
    return len(query_tokens_set & node_toks) / len(query_tokens_set)


def safe_join(repo_root: Path, rel_path: str) -> Path:
    """Safely join a repo-relative path to the repository root.

    :param repo_root: Absolute path to the repository root.
    :param rel_path: Repository-relative path to join.
    :return: Resolved absolute Path within repo_root.
    :raises ValueError: If the resolved path escapes repo_root.
    """
    p = (repo_root / rel_path).resolve()
    rr = repo_root.resolve()
    if rr not in p.parents and p != rr:
        raise ValueError(f"Unsafe path outside repo_root: {rel_path!r}")
    return p


def read_lines(path: Path) -> list[str]:
    """Read all lines from a file, returning [] if the file is missing.

    :param path: Absolute path to the file.
    :return: List of lines (without newlines) or [] on error.
    """
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except FileNotFoundError:
        return []


def compute_span(
    kind: str,
    lineno: int | None,
    end_lineno: int | None,
    *,
    context: int,
    max_lines: int,
    file_nlines: int,
) -> tuple[int, int]:
    """Compute the 1-based (start, end) line span for a node's source snippet.

    :param kind: Node kind (e.g. ``"module"``, ``"function"``, ``"class"``).
    :param lineno: 1-based start line, or ``None``.
    :param end_lineno: 1-based end line, or ``None``.
    :param context: Extra lines to include before and after the span.
    :param max_lines: Maximum number of lines the span may contain.
    :param file_nlines: Total number of lines in the source file.
    :return: ``(start, end)`` tuple of 1-based line numbers.
    """
    if file_nlines <= 0:
        return (1, 0)
    if kind == "module":
        return (1, min(file_nlines, max_lines))
    if lineno is None:
        return (1, min(file_nlines, max_lines))
    if end_lineno is not None and end_lineno >= lineno:
        start = max(1, lineno - context)
        end = min(file_nlines, end_lineno + context)
        if (end - start + 1) > max_lines:
            end = min(file_nlines, start + max_lines - 1)
        return (start, end)
    start = max(1, lineno - context)
    end = min(file_nlines, lineno + context)
    if (end - start + 1) > max_lines:
        end = min(file_nlines, start + max_lines - 1)
    return (start, end)


def make_snippet(rel_path: str, lines: list[str], start: int, end: int) -> dict[str, Any]:
    """Build a snippet dictionary from a slice of source lines.

    :param rel_path: Repository-relative file path.
    :param lines: Full list of source lines for the file (0-indexed).
    :param start: 1-based first line of the snippet (inclusive).
    :param end: 1-based last line of the snippet (inclusive).
    :return: Dictionary with ``path``, ``start``, ``end``, and ``text`` keys.
    """
    s0 = max(0, start - 1)
    e0 = max(0, end)
    chunk = lines[s0:e0]
    numbered = "\n".join(f"{i:>5d}: {line}" for i, line in enumerate(chunk, start=start))
    return {"path": rel_path, "start": start, "end": end, "text": numbered}


def make_module_summary(
    rel_path: str,
    lines: list[str],
    docstring: str | None,
    contained_nodes: list[dict[str, Any]],
    max_lines: int,
) -> dict[str, Any]:
    """Build a summary snippet for a module-level node.

    :param rel_path: Repository-relative file path.
    :param lines: Full list of source lines for the file.
    :param docstring: Module docstring text, or ``None``.
    :param contained_nodes: List of node dicts directly contained by this module.
    :param max_lines: Maximum lines for the summary.
    :return: Snippet dict with ``is_summary=True``.
    """
    out: list[str] = []
    if docstring:
        for dl in docstring.strip().splitlines():
            out.append(dl)
        out.append("")

    by_kind: dict[str, list[dict[str, Any]]] = {}
    for cn in contained_nodes:
        k = cn.get("kind", "unknown")
        by_kind.setdefault(k, []).append(cn)

    for kind in ("class", "function", "method"):
        group = by_kind.get(kind, [])
        if not group:
            continue
        out.append(f"# {kind}s ({len(group)}):")
        for cn in sorted(group, key=lambda x: x.get("lineno") or 0):
            name = cn.get("qualname") or cn.get("name", "?")
            ln = cn.get("lineno")
            ds = cn.get("docstring") or ""
            ds_first = ds.strip().splitlines()[0][:80] if ds.strip() else ""
            loc = f"L{ln}" if ln else "?"
            if ds_first:
                out.append(f"#   {name} ({loc}) — {ds_first}")
            else:
                out.append(f"#   {name} ({loc})")
        out.append("")

    summary_lines = out[:max_lines]
    text = "\n".join(f"{'':>5s}  {line}" for line in summary_lines)
    return {
        "path": rel_path,
        "start": 1,
        "end": min(len(lines), max_lines),
        "text": text,
        "is_summary": True,
    }


def spans_overlap(a: tuple[int, int], b: tuple[int, int], gap: int = 2) -> bool:
    """Return True when two line spans are considered overlapping.

    :param a: First span ``(start, end)`` of 1-based line numbers.
    :param b: Second span ``(start, end)`` of 1-based line numbers.
    :param gap: Minimum separation for non-overlapping spans.
    :return: True if the spans overlap or are within gap lines of each other.
    """
    a0, a1 = a
    b0, b1 = b
    return not (a1 + gap < b0 or b1 + gap < a0)


# ---------------------------------------------------------------------------
# KGModule — concrete base class
# ---------------------------------------------------------------------------


class KGModule(ABC):
    """Concrete base class for production-grade knowledge graph modules.

    Domain authors subclass this and implement exactly three abstract methods:
    :meth:`make_extractor`, :meth:`kind`, and :meth:`analyze`.

    Everything else — build pipeline, hybrid query, snippet packing, lazy
    layer initialisation, snapshot support — is provided by this class.

    :param repo_root: Repository root directory.
    :param db_path: SQLite database path (defaults to ``.<kind>kg/graph.sqlite``
                    under ``repo_root``; set ``_default_dir`` in subclass to override).
    :param lancedb_dir: LanceDB directory (defaults to ``.<kind>kg/lancedb``).
    :param model: Sentence-transformer model name for embedding.
    :param table: LanceDB table name.
    """

    #: Override in subclass to change the default artefact directory name.
    _default_dir: str = ".kgcache"

    def __init__(
        self,
        repo_root: str | Path,
        db_path: str | Path | None = None,
        lancedb_dir: str | Path | None = None,
        *,
        model: str = DEFAULT_MODEL,
        table: str = "kg_nodes",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        _dir = self.repo_root / self._default_dir
        self.db_path = Path(db_path) if db_path is not None else _dir / "graph.sqlite"
        self.lancedb_dir = Path(lancedb_dir) if lancedb_dir is not None else _dir / "lancedb"
        self.model_name = model
        self.table_name = table

        self._store: GraphStore | None = None
        self._index: SemanticIndex | None = None
        self._embedder: Embedder | None = None

    # ------------------------------------------------------------------
    # Abstract interface — domain authors implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def make_extractor(self) -> KGExtractor:
        """Return the domain-specific extractor for this module.

        :return: A :class:`~kg_utils.types.KGExtractor` instance.
        """

    @abstractmethod
    def kind(self) -> str:
        """Return the KG kind string for this module (e.g. ``"code"``, ``"doc"``).

        :return: Kind string used in registry entries and MCP tool names.
        """

    @abstractmethod
    def analyze(self) -> str:
        """Run a full analysis of this KG and return a Markdown report.

        Must not raise — return a Markdown error message on failure.

        :return: Markdown-formatted analysis report string.
        """

    # ------------------------------------------------------------------
    # Layer accessors (lazy init)
    # ------------------------------------------------------------------

    @property
    def store(self) -> GraphStore:
        """SQLite persistence layer (lazy-initialised)."""
        if self._store is None:
            self._store = GraphStore(self.db_path)
        return self._store

    @property
    def embedder(self) -> Embedder:
        """Embedding backend (lazy-initialised, shared between index and query)."""
        if self._embedder is None:
            suppress_ingestion_logging()
            self._embedder = SentenceTransformerEmbedder(self.model_name)
        return self._embedder

    @property
    def index(self) -> SemanticIndex:
        """LanceDB semantic index (lazy-initialised)."""
        if self._index is None:
            extractor = self.make_extractor()
            self._index = SemanticIndex(
                self.lancedb_dir,
                embedder=self.embedder,
                table=self.table_name,
                index_kinds=extractor.meaningful_node_kinds(),
            )
        return self._index

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, *, wipe: bool = False) -> BuildStats:
        """Full pipeline: extraction → SQLite → LanceDB.

        :param wipe: Clear existing data before writing.
        :return: :class:`~kg_utils.types.BuildStats`.
        """
        graph_stats = self.build_graph(wipe=wipe)
        index_stats = self.build_index(wipe=wipe)
        graph_stats.indexed_rows = index_stats.indexed_rows
        graph_stats.index_dim = index_stats.index_dim
        return graph_stats

    def build_graph(self, *, wipe: bool = False) -> BuildStats:
        """Extraction → SQLite only (no vector indexing).

        Calls :meth:`make_extractor`, drains the iterator, writes to SQLite,
        then calls :meth:`_post_build_hook` for any domain-specific post-processing.

        :param wipe: Clear existing graph before writing.
        :return: :class:`~kg_utils.types.BuildStats` (``indexed_rows`` will be ``None``).
        """
        extractor = self.make_extractor()
        node_specs: list[NodeSpec] = []
        edge_specs: list[EdgeSpec] = []
        for item in extractor.extract():
            if isinstance(item, NodeSpec):
                node_specs.append(item)
            else:
                edge_specs.append(item)

        self.store.write(node_specs, edge_specs, wipe=wipe)
        self._post_build_hook(self.store)

        s = self.store.stats()
        return BuildStats(
            repo_root=str(self.repo_root),
            db_path=str(self.db_path),
            total_nodes=s["total_nodes"],
            total_edges=s["total_edges"],
            node_counts=s["node_counts"],
            edge_counts=s["edge_counts"],
        )

    def build_index(self, *, wipe: bool = False) -> BuildStats:
        """SQLite → LanceDB only (graph must already exist).

        :param wipe: Delete existing vectors before indexing.
        :return: :class:`~kg_utils.types.BuildStats` with ``indexed_rows`` and ``index_dim`` set.
        """
        idx_stats = self.index.build(self.store, wipe=wipe)
        s = self.store.stats()
        return BuildStats(
            repo_root=str(self.repo_root),
            db_path=str(self.db_path),
            total_nodes=s["total_nodes"],
            total_edges=s["total_edges"],
            node_counts=s["node_counts"],
            edge_counts=s["edge_counts"],
            indexed_rows=idx_stats["indexed_rows"],
            index_dim=idx_stats["dim"],
        )

    def _post_build_hook(self, store: GraphStore) -> None:
        """Hook called after ``store.write()`` in :meth:`build_graph`.

        Override for domain-specific post-processing, e.g. symbol resolution
        (``store.resolve_symbols()``).

        :param store: The :class:`~kg_utils.store.GraphStore` just written to.
        """

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        q: str,
        *,
        k: int = 8,
        hop: int = 1,
        rels: tuple[str, ...] = DEFAULT_RELS,
        include_symbols: bool = False,
        max_nodes: int = 25,
        min_score: float = 0.0,
        max_per_module: int | None = None,
        rerank_mode: str = "legacy",
        rerank_semantic_weight: float = 0.7,
        rerank_lexical_weight: float = 0.3,
    ) -> QueryResult:
        """Hybrid query: semantic seeding + structural graph expansion.

        :param q: Natural-language query.
        :param k: Top-K semantic hits.
        :param hop: Graph expansion hops (0 = pure semantic).
        :param rels: Edge types to follow during expansion.
        :param include_symbols: Include unresolved stub nodes in results.
        :param max_nodes: Maximum nodes to return.
        :param min_score: Minimum semantic score for seed inclusion.
        :param max_per_module: Optional cap on returned nodes per module path.
        :param rerank_mode: ``'legacy'`` | ``'semantic'`` | ``'hybrid'``.
        :param rerank_semantic_weight: Semantic weight for ``'hybrid'`` mode.
        :param rerank_lexical_weight: Lexical weight for ``'hybrid'`` mode.
        :return: :class:`~kg_utils.types.QueryResult`.
        """
        q_norm = normalize_query_text(q)
        hits = self.index.search(q_norm, k=k)
        if min_score > 0.0:
            hits = [h for h in hits if semantic_score_from_distance(h.distance) >= min_score]
        seed_ids: set[str] = {h.id for h in hits}
        seed_rank: dict[str, float] = {h.id: h.distance for h in hits}
        q_toks = query_tokens(q_norm)

        meta = self.store.expand(seed_ids, hop=hop, rels=rels)
        all_ids = set(meta.keys())

        nodes: list[dict[str, Any]] = []
        kept_ids: set[str] = set()
        module_counts: dict[str, int] = {}

        def _rank_key(nid: str) -> tuple[float, float, int, int, str]:
            prov = meta[nid]
            dist = seed_rank.get(prov.via_seed, 1e9)
            n = self.store.node(nid)
            kind = n["kind"] if n else "symbol"
            sem = semantic_score_from_distance(dist)
            lex = lexical_overlap_score(q_toks, n or {})
            hybrid = (
                rerank_semantic_weight * sem + rerank_lexical_weight * lex
                if (rerank_semantic_weight + rerank_lexical_weight) > 0
                else sem
            )
            if rerank_mode == "hybrid":
                return (-hybrid, float(prov.best_hop), self._kind_priority(kind), 0, nid)
            if rerank_mode == "semantic":
                return (-sem, float(prov.best_hop), self._kind_priority(kind), 0, nid)
            return (float(prov.best_hop), dist, self._kind_priority(kind), 0, nid)

        for nid in sorted(all_ids, key=_rank_key):
            if len(nodes) >= max_nodes:
                break
            n = self.store.node(nid)
            if not n:
                continue
            if not include_symbols and n["kind"] == "symbol":
                continue
            module_path = n.get("module_path") or ""
            if max_per_module is not None and module_path:
                if module_counts.get(module_path, 0) >= max_per_module:
                    continue
                module_counts[module_path] = module_counts.get(module_path, 0) + 1

            prov = meta[nid]
            dist = seed_rank.get(prov.via_seed, 1e9)
            sem = semantic_score_from_distance(dist)
            lex = lexical_overlap_score(q_toks, n)
            n["relevance"] = {
                "score": (
                    rerank_semantic_weight * sem + rerank_lexical_weight * lex
                    if rerank_mode == "hybrid"
                    else sem
                ),
                "semantic": sem,
                "lexical": lex,
                "docstring_signal": docstring_signal(n.get("docstring")),
                "hop": prov.best_hop,
                "via_seed": prov.via_seed,
                "mode": rerank_mode,
            }
            kept_ids.add(nid)
            nodes.append(n)

        if nodes:
            max_score = max(n["relevance"]["score"] for n in nodes) or 1.0
            if max_score > 0.0:
                for n in nodes:
                    n["relevance"]["score"] = round(n["relevance"]["score"] / max_score, 6)

        edges = self.store.edges_within(kept_ids)
        return QueryResult(
            query=q,
            seeds=len(seed_ids),
            expanded_nodes=len(all_ids),
            returned_nodes=len(nodes),
            hop=hop,
            rels=list(rels),
            nodes=nodes,
            edges=edges,
        )

    # ------------------------------------------------------------------
    # Snippet pack
    # ------------------------------------------------------------------

    def pack(
        self,
        q: str,
        *,
        k: int = 8,
        hop: int = 1,
        rels: tuple[str, ...] = DEFAULT_RELS,
        include_symbols: bool = False,
        context: int = 5,
        max_lines: int = 60,
        max_nodes: int | None = 15,
        min_score: float = 0.0,
        max_per_module: int | None = None,
        rerank_mode: str = "legacy",
        rerank_semantic_weight: float = 0.7,
        rerank_lexical_weight: float = 0.3,
        missing_lineno_policy: str = "cap_or_skip",
    ) -> SnippetPack:
        """Hybrid query + source-grounded snippet extraction.

        :param q: Natural-language query.
        :param k: Top-K semantic hits.
        :param hop: Graph expansion hops.
        :param rels: Edge types to follow during expansion.
        :param include_symbols: Include unresolved stub nodes.
        :param context: Extra context lines around each definition span.
        :param max_lines: Maximum lines per snippet block.
        :param max_nodes: Maximum nodes to return (``None`` = no limit).
        :param min_score: Minimum semantic score for seed inclusion.
        :param max_per_module: Optional cap on returned nodes per module.
        :param rerank_mode: ``'legacy'`` | ``'semantic'`` | ``'hybrid'``.
        :param rerank_semantic_weight: Semantic weight for ``'hybrid'`` mode.
        :param rerank_lexical_weight: Lexical weight for ``'hybrid'`` mode.
        :param missing_lineno_policy: ``'cap_or_skip'`` (default) or ``'legacy'``.
        :return: :class:`~kg_utils.types.SnippetPack`.
        """
        q_norm = normalize_query_text(q)
        hits = self.index.search(q_norm, k=k)
        if min_score > 0.0:
            hits = [h for h in hits if semantic_score_from_distance(h.distance) >= min_score]
        seed_rank: dict[str, dict[str, Any]] = {
            h.id: {"rank": h.rank, "dist": h.distance} for h in hits
        }
        seed_ids: set[str] = set(seed_rank.keys())
        q_toks = query_tokens(q_norm)
        warnings: list[str] = []

        meta = self.store.expand(seed_ids, hop=hop, rels=rels)
        all_ids = set(meta.keys())

        raw_nodes: list[dict[str, Any]] = []
        for nid in sorted(all_ids):
            n = self.store.node(nid)
            if not n:
                continue
            if not include_symbols and n["kind"] == "symbol":
                continue

            prov = meta[nid]
            base_dist = seed_rank.get(prov.via_seed, {"dist": 1e9})["dist"]
            kind_pri = self._kind_priority(n["kind"])
            sem = semantic_score_from_distance(base_dist)
            lex = lexical_overlap_score(q_toks, n)
            hybrid = (
                rerank_semantic_weight * sem + rerank_lexical_weight * lex
                if (rerank_semantic_weight + rerank_lexical_weight) > 0
                else sem
            )
            if rerank_mode == "hybrid":
                n["_rank_key"] = (-hybrid, float(prov.best_hop), kind_pri, 0, n["id"])
            elif rerank_mode == "semantic":
                n["_rank_key"] = (-sem, float(prov.best_hop), kind_pri, 0, n["id"])
            else:
                n["_rank_key"] = (float(prov.best_hop), base_dist, kind_pri, 0, n["id"])
            n["_best_hop"] = prov.best_hop
            n["_via_seed"] = prov.via_seed
            n["relevance"] = {
                "score": hybrid if rerank_mode == "hybrid" else sem,
                "semantic": sem,
                "lexical": lex,
                "docstring_signal": docstring_signal(n.get("docstring")),
                "hop": prov.best_hop,
                "via_seed": prov.via_seed,
                "mode": rerank_mode,
            }
            raw_nodes.append(n)

        # Attach spans (needed for dedup)
        file_cache: dict[str, list[str]] = {}
        spans_by_qualname: dict[tuple[str, str], tuple[int, int]] = {}
        for n in raw_nodes:
            mp = n.get("module_path")
            if not mp:
                n["_span"] = None
                continue
            if mp not in file_cache:
                file_cache[mp] = read_lines(safe_join(self.repo_root, mp))
            lines = file_cache[mp]
            lineno = n.get("lineno")
            if n.get("kind") != "module" and lineno is None and missing_lineno_policy != "legacy":
                qualname = n.get("qualname") or ""
                parent_span = None
                if qualname and "." in qualname:
                    parent_key = (mp, qualname.rsplit(".", 1)[0])
                    parent_span = spans_by_qualname.get(parent_key)
                if parent_span:
                    fallback_cap = min(max_lines, max(20, context * 4))
                    p_start, p_end = parent_span
                    capped_end = min(p_end, p_start + fallback_cap - 1)
                    n["_span"] = (p_start, capped_end)
                    warnings.append(
                        f"Missing line metadata for `{n['id']}`; "
                        f"using capped parent span {p_start}-{capped_end}."
                    )
                else:
                    n["_span"] = None
                    warnings.append(
                        f"Missing line metadata for `{n['id']}`; "
                        "snippet omitted (no parent span available)."
                    )
            else:
                n["_span"] = compute_span(
                    n["kind"],
                    lineno,
                    n.get("end_lineno"),
                    context=context,
                    max_lines=max_lines,
                    file_nlines=len(lines),
                )
            if n.get("qualname") and n.get("_span"):
                spans_by_qualname[(mp, n["qualname"])] = n["_span"]

        raw_nodes.sort(key=lambda x: x["_rank_key"])

        # Deduplicate by file + overlapping span
        kept: list[dict[str, Any]] = []
        kept_by_file: dict[str, list[tuple[tuple[int, int], str]]] = {}
        module_counts: dict[str, int] = {}

        for n in raw_nodes:
            if max_nodes is not None and len(kept) >= max_nodes:
                break
            mp = n.get("module_path") or ""
            span = n.get("_span")

            if not mp or not span or span[1] < span[0]:
                kept.append(n)
                continue

            if any(spans_overlap(span, s2) for s2, _ in kept_by_file.get(mp, [])):
                continue

            if max_per_module is not None and mp:
                if module_counts.get(mp, 0) >= max_per_module:
                    continue
                module_counts[mp] = module_counts.get(mp, 0) + 1

            kept.append(n)
            kept_by_file.setdefault(mp, []).append((span, n["id"]))

        kept_ids: set[str] = {n["id"] for n in kept}
        edges = self.store.edges_within(kept_ids)

        # Attach snippets
        for n in kept:
            mp = n.get("module_path")
            span = n.get("_span")
            if not mp or not span:
                continue
            if mp not in file_cache:
                file_cache[mp] = read_lines(safe_join(self.repo_root, mp))
            lines = file_cache[mp]
            start, end = span

            if n.get("kind") == "module" and len(lines) > max_lines:
                contained = [
                    self.store.node(cn_id)
                    for cn_id in (
                        row[0]
                        for row in self.store.con.execute(
                            "SELECT dst FROM edges WHERE src = ? AND rel = 'CONTAINS'",
                            (n["id"],),
                        ).fetchall()
                    )
                    if self.store.node(cn_id) is not None
                ]
                contained_nodes: list[dict[str, Any]] = [c for c in contained if c is not None]
                n["snippet"] = make_module_summary(
                    mp, lines, n.get("docstring"), contained_nodes, max_lines
                )
            elif end >= start and lines:
                n["snippet"] = make_snippet(mp, lines, start, end)

        # Strip internal keys
        for n in kept:
            for key in [k for k in n if k.startswith("_")]:
                del n[key]

        return SnippetPack(
            query=q,
            seeds=len(seed_ids),
            expanded_nodes=len(all_ids),
            returned_nodes=len(kept),
            hop=hop,
            rels=list(rels),
            model=self.model_name,
            nodes=kept,
            edges=edges,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def callers(self, node_id: str, *, rel: str = "CALLS") -> list[dict[str, Any]]:
        """Return all nodes that call *node_id*, resolving through stubs.

        :param node_id: Target node identifier.
        :param rel: Relation type to invert (default ``"CALLS"``).
        :return: Deduplicated list of caller node dicts.
        """
        return self.store.callers_of(node_id, rel=rel)

    def stats(self) -> dict[str, Any]:
        """Return store statistics (node/edge counts by kind/relation).

        :return: Dictionary from :meth:`~kg_utils.store.GraphStore.stats`.
        """
        return self.store.stats()

    def node(self, node_id: str) -> dict[str, Any] | None:
        """Fetch a single node by ID from the store.

        :param node_id: Stable node identifier.
        :return: Node dict or ``None`` if not found.
        """
        return self.store.node(node_id)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._store is not None:
            self._store.close()

    def __enter__(self) -> KGModule:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Domain hook
    # ------------------------------------------------------------------

    def _kind_priority(self, kind: str) -> int:
        """Return the sort priority for a node kind (lower = higher priority).

        Override in subclasses for domain-specific ordering::

            def _kind_priority(self, kind):
                return {"function": 0, "class": 1, "module": 2}.get(kind, 99)

        :param kind: Node kind string.
        :return: Integer priority (lower = ranked first).
        """
        return 99
