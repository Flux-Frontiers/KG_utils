"""
Render a knowledge graph as a self-contained interactive HTML page.

One implementation shared by every KG module.  Domain differences arrive as
data: a :class:`~kg_utils.viz.theme.GraphTheme` naming the domain's kinds and
relations, and a :class:`~kg_utils.viz.tooltip.TooltipSpec` naming the fields
worth showing.  A plain callable can replace the spec when a domain needs markup
the spec cannot express.

The output inlines vis-network, so the page opens from ``file://`` and survives
being embedded in a ``srcdoc`` iframe — both of which pyvis's default
``cdn_resources="local"`` breaks, silently, by emitting relative asset paths.

Requires the ``viz`` extra::

    pip install 'kgmodule-utils[viz]'
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from kg_utils.viz.theme import GraphTheme, with_alpha
from kg_utils.viz.tooltip import TooltipSpec

__all__ = [
    "CENTRALITY_MIN_OPACITY",
    "CENTRALITY_SIZE_RANGE",
    "SEED_FRACTION",
    "build_graph_html",
    "select_nodes",
]

#: Node diameter range in pixels when a metric drives sizing.
CENTRALITY_SIZE_RANGE: Final[tuple[int, int]] = (8, 42)

#: Opacity floor for the least central node.  Never 0 — an invisible node is
#: indistinguishable from a missing one.
CENTRALITY_MIN_OPACITY: Final[float] = 0.35

#: Share of the node budget spent on seeds; the rest goes to their neighbours.
SEED_FRACTION: Final[int] = 4

#: Border colour for nodes the caller asked to highlight.
_HIGHLIGHT_COLOR: Final[str] = "#FFD700"

_MAX_LABEL = 28

_PHYSICS_OPTIONS: Final[dict[str, Any]] = {
    "barnesHut": {
        "gravitationalConstant": -8000,
        "centralGravity": 0.3,
        "springLength": 120,
        "springConstant": 0.04,
        "damping": 0.09,
    },
    "stabilization": {"iterations": 150},
}

#: pyvis 0.3.2 hardcodes Bootstrap CDN tags in its own template, and
#: ``cdn_resources="in_line"`` governs only the vis-network assets -- it does not
#: touch these.  Our output uses exactly two Bootstrap classes (``.card`` wrapping
#: the graph and ``.card-body`` on the canvas container), so we strip the CDN tags
#: and supply their rules directly rather than making an offline page depend on a
#: third-party host.  ``bootstrap.bundle.min.js`` is entirely unused -- nothing we
#: emit has a dropdown, modal, tooltip, or collapse -- so it is simply dropped.
_BOOTSTRAP_CDN: Final[re.Pattern[str]] = re.compile(
    r"<link\b[^>]*cdn\.jsdelivr\.net[^>]*>"
    r"|<script\b[^>]*cdn\.jsdelivr\.net[^>]*>\s*</script>",
    re.S,
)

#: Copied verbatim from ``bootstrap@5.0.0-beta3``: the ``.card`` / ``.card-body``
#: rules, plus two Reboot rules that are load-bearing for the box model.  Without
#: the ``box-sizing`` reset ``.card-body`` overflows its parent by 32px; without
#: ``body { margin: 0 }`` the browser's default 8px margin narrows the card by
#: 16px.  The reset is scoped to ``.card`` rather than ``*`` so it cannot leak
#: into a host page that embeds this markup in a larger document.
_BOOTSTRAP_SHIM: Final[str] = """<style type="text/css">
body { margin: 0; }
.card, .card *, .card *::before, .card *::after { box-sizing: border-box; }
.card { position: relative; display: flex; flex-direction: column;
        min-width: 0; word-wrap: break-word; background-color: #fff;
        background-clip: border-box; border: 1px solid rgba(0,0,0,.125);
        border-radius: .25rem; }
.card-body { flex: 1 1 auto; padding: 1rem 1rem; }
</style>"""


def _inline_bootstrap(document: str) -> str:
    """Replace pyvis's Bootstrap CDN tags with the rules the page actually uses.

    :param document: HTML as pyvis wrote it.
    :return: The same page with no external references.
    """
    return _BOOTSTRAP_CDN.sub("", document).replace("</head>", _BOOTSTRAP_SHIM + "</head>", 1)


def _describe(scores: Any, kept: Sequence[str], suffix: str) -> str:
    """Summarise a selection for display above the graph.

    Reporting how many kept nodes carry no score matters once neighbours are
    pulled in by expansion: those are frequently unscored, and a reader should
    not assume every node on screen was ranked.

    :param scores: The active ``ScoreSet``.
    :param kept: Node IDs that were kept.
    :param suffix: Extra text describing the strategy.
    :return: Human-readable description.
    """
    unscored = sum(1 for i in kept if scores.rank(i) is None)
    tail = f" ({unscored} unscored)" if unscored else ""
    return f"most central by {scores.metric}{suffix}{tail}"


def select_nodes(
    nodes: list[dict],
    limit: int,
    scores: Any | None,
    mode: str = "central",
    expand: Callable[[set[str], int], set[str]] | None = None,
) -> tuple[list[dict], str]:
    """Reduce *nodes* to at most *limit*, choosing which ones to keep.

    A graph can only draw a few hundred nodes, so which ones survive the cap
    decides what the reader actually sees.  Two strategies:

    * ``"path"`` keeps the store's natural order.  Because that order is
      contiguous by source, the result is well connected — but it is an
      arbitrary slice, typically alphabetical.
    * ``"central"`` seeds on the most central nodes and pulls in their graph
      neighbours until the budget is spent.

    Seeding *and expanding* rather than taking the top N is the important
    detail.  The most central nodes are scattered, so keeping only them strands
    most of them: measured on a 10k-node code graph at a cap of 150,
    top-N-by-centrality left 47 nodes with no edge at all and halved the edge
    count, producing a field of dots rather than a graph.

    :param nodes: Candidate nodes, already filtered.
    :param limit: Maximum number to keep.
    :param scores: Active ``ScoreSet``, or ``None``.
    :param mode: ``"central"`` or ``"path"``.
    :param expand: Callable taking ``(seed_ids, hop)`` and returning the node IDs
        reachable within that many hops.  Without it, ``"central"`` degrades to
        top-N by centrality.
    :return: The kept nodes and a short description of how they were chosen.
    """
    if len(nodes) <= limit:
        return nodes, "all matching nodes"
    if mode != "central" or scores is None:
        return nodes[:limit], "first in store order"

    by_id = {n["id"]: n for n in nodes}
    ranked = sorted(by_id, key=lambda i: (scores.rank(i) is None, scores.rank(i) or 0, i))

    if expand is None:
        top = ranked[:limit]
        return [by_id[i] for i in top], _describe(scores, top, "")

    seed_count = max(1, limit // SEED_FRACTION)
    seeds = set(ranked[:seed_count])
    reachable = expand(seeds, 1) & by_id.keys()

    kept: list[str] = list(ranked[:seed_count])
    seen = set(kept)
    for node_id in sorted(
        reachable, key=lambda i: (scores.rank(i) is None, scores.rank(i) or 0, i)
    ):
        if len(kept) >= limit:
            break
        if node_id not in seen:
            kept.append(node_id)
            seen.add(node_id)
    for node_id in ranked:
        if len(kept) >= limit:
            break
        if node_id not in seen:
            kept.append(node_id)
            seen.add(node_id)

    return [by_id[i] for i in kept], _describe(scores, kept, ", plus neighbours")


def build_graph_html(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    theme: GraphTheme,
    tooltip: TooltipSpec | Callable[[Mapping[str, Any], str], str] | None = None,
    scores: Any | None = None,
    height: str = "620px",
    physics: bool = True,
    highlight_ids: set[str] | None = None,
    label_field: str = "name",
) -> str:
    """Render *nodes* and *edges* as a self-contained interactive HTML page.

    When *scores* is supplied, node diameter and opacity both encode the
    metric's rank percentile.  Without it, diameter comes from the theme's
    per-kind size and every node is fully opaque.

    :param nodes: Node attribute mappings; each needs at least an ``id``.
    :param edges: Edge mappings with ``src``, ``dst`` and ``rel`` keys.
    :param theme: Visual vocabulary for this domain.
    :param tooltip: A :class:`TooltipSpec`, or a callable taking
        ``(node, color)`` and returning HTML.  ``None`` shows kind and name only.
    :param scores: ``ScoreSet`` driving size and opacity, or ``None``.
    :param height: CSS height for the canvas, e.g. ``"620px"``.
    :param physics: Whether to run the Barnes-Hut simulation.
    :param highlight_ids: Node IDs to mark with a gold border — query seeds,
        search hits, whatever the caller wants to point at.
    :param label_field: Node key used for the on-canvas label.
    :return: A self-contained HTML document.
    """
    from pyvis.network import Network  # noqa: PLC0415 — keeps the viz extra optional

    net = Network(
        height=height,
        width="100%",
        bgcolor="#0e1117",
        font_color="#e0e0e0",
        directed=True,
        notebook=False,
        # pyvis defaults to cdn_resources="local", which emits *relative* asset
        # paths plus a cdnjs fallback, and writes a lib/ directory into the
        # working directory.  Relative paths cannot resolve inside a srcdoc
        # iframe, so the graph renders only when cdnjs is reachable and fails
        # silently with "vis is not defined" otherwise.  "in_line" inlines
        # vis-network, making the page genuinely self-contained.
        cdn_resources="in_line",
    )
    net.set_options(
        json.dumps(
            {
                "physics": {"enabled": physics, **_PHYSICS_OPTIONS},
                "edges": {
                    "smooth": {"type": "dynamic"},
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
                    "font": {"size": 10, "color": "#aaaaaa"},
                },
                "interaction": {
                    "hover": True,
                    "tooltipDelay": 80,
                    "navigationButtons": True,
                    "keyboard": True,
                },
            }
        )
    )

    highlight_ids = highlight_ids or set()
    lo_size, hi_size = CENTRALITY_SIZE_RANGE
    panel_data: dict[str, dict[str, Any]] = {}

    for node in nodes:
        node_id = node["id"]
        style = theme.style_of(node)
        color = style.color

        label = str(node.get(label_field) or node_id)
        if len(label) > _MAX_LABEL:
            label = label[: _MAX_LABEL - 3] + "…"

        if tooltip is None:
            title = f"{node.get('kind', '')}: {label}"
        elif isinstance(tooltip, TooltipSpec):
            title = tooltip.render(node, color)
        else:
            title = tooltip(node, color)

        if scores is None:
            size: float = style.size
            background = color
        else:
            size = scores.scaled(node_id, lo_size, hi_size, default=lo_size)
            alpha = CENTRALITY_MIN_OPACITY + (1.0 - CENTRALITY_MIN_OPACITY) * scores.percentile(
                node_id
            )
            background = with_alpha(color, alpha)

        highlighted = node_id in highlight_ids
        net.add_node(
            node_id,
            label=label,
            title=title,
            color={
                "background": background,
                "border": _HIGHLIGHT_COLOR if highlighted else color,
                "highlight": {"background": color, "border": "#FFFFFF"},
            },
            shape=style.shape,
            size=size,
            borderWidth=3 if highlighted else 1,
            font={"size": 11},
        )
        panel_data[node_id] = _panel_entry(node, color, tooltip, scores)

    for edge in edges:
        rel = str(edge.get("rel", ""))
        net.add_edge(
            edge["src"],
            edge["dst"],
            label=rel,
            color=theme.relation_color(rel),
            width=1.5,
            title=rel,
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as handle:
        tmp_path = handle.name
    try:
        net.save_graph(tmp_path)
        document = Path(tmp_path).read_text(encoding="utf-8")
    finally:
        os.unlink(tmp_path)

    document = _inline_bootstrap(document)
    return document.replace("</body>", _panel_markup(panel_data) + "</body>")


def _panel_entry(
    node: Mapping[str, Any],
    color: str,
    tooltip: TooltipSpec | Callable[..., str] | None,
    scores: Any | None,
) -> dict[str, Any]:
    """Build the click-panel payload for one node.

    Driven by the same spec as the hover tooltip so the two never disagree.

    :param node: Node attribute mapping.
    :param color: Accent colour.
    :param tooltip: The active tooltip spec, if it is a spec.
    :param scores: Active ``ScoreSet``, used to include rank.
    :return: JSON-serialisable panel entry.
    """
    spec = tooltip if isinstance(tooltip, TooltipSpec) else None
    if spec is None:
        title = str(node.get("name") or node["id"])
        meta: list[str] = []
        body = ""
    else:
        title = (
            str(node.get(spec.title) or node.get("name") or node["id"])
            if isinstance(spec.title, str)
            else spec.title(node)
        )
        meta = [text for text in (row.render(node) for row in spec.rows) if text]
        body = str(node.get(spec.body) or "").strip() if spec.body else ""

    rank = scores.rank(node["id"]) if scores is not None else None
    return {
        "id": node["id"],
        "kind": str(node.get("kind", "")),
        "color": color,
        "title": title,
        "meta": meta,
        "body": body,
        "rank": rank,
        "metric": getattr(scores, "metric", None) if scores is not None else None,
    }


def _panel_markup(panel_data: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the CSS, markup and script for the floating click-detail panel.

    :param panel_data: Per-node payload keyed by node ID.
    :return: HTML to inject before ``</body>``.
    """
    # The payload is embedded in a <script> block, so any "</script>" inside a
    # node's text would terminate the block during HTML parsing and inject
    # whatever followed.  Escaping the three characters that can start such a
    # sequence keeps the JSON valid while making breakout impossible.
    payload = (
        json.dumps(panel_data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return """
<style>
#kgviz-panel {
  display: none; position: fixed; top: 12px; right: 12px; width: 340px;
  max-height: 88vh; overflow-y: auto; background: #1e1e2e; border-radius: 10px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.6); z-index: 9999;
  font-family: sans-serif; font-size: 13px; color: #e0e0e0;
}
#kgviz-panel-inner { padding: 14px 16px 16px 16px; }
#kgviz-panel-close {
  position: absolute; top: 8px; right: 10px; cursor: pointer; font-size: 18px;
  color: #888; line-height: 1; background: none; border: none;
}
#kgviz-panel-close:hover { color: #fff; }
#kgviz-panel-body {
  background: #12121f; border: 1px solid #2a2a3e; border-radius: 6px;
  padding: 8px 10px; font-family: monospace; font-size: 12px; color: #c9d1d9;
  white-space: pre-wrap; word-break: break-word; margin-top: 8px;
  max-height: 300px; overflow-y: auto;
}
</style>
<div id="kgviz-panel">
  <button id="kgviz-panel-close"
          onclick="document.getElementById('kgviz-panel').style.display='none'">&#10005;</button>
  <div id="kgviz-panel-inner">
    <div id="kgviz-panel-badge"></div>
    <div id="kgviz-panel-title" style="font-size:15px;font-weight:bold;margin:6px 0 2px 0;"></div>
    <div id="kgviz-panel-meta" style="color:#888;font-size:11px;font-family:monospace;"></div>
    <div id="kgviz-panel-id"
         style="color:#444;font-size:10px;font-family:monospace;margin-top:2px;"></div>
    <div id="kgviz-panel-body"></div>
  </div>
</div>
<script>
(function () {
  var NODE_DATA = __KGVIZ_PAYLOAD__;
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function show(nodeId) {
    var n = NODE_DATA[nodeId];
    if (!n) { return; }
    document.getElementById("kgviz-panel-badge").innerHTML =
      "<span style='background:" + n.color + ";color:#fff;border-radius:4px;" +
      "padding:1px 7px;font-size:11px;font-weight:bold;'>" + esc(n.kind) + "</span>";
    document.getElementById("kgviz-panel-title").textContent = n.title;
    var meta = (n.meta || []).slice();
    if (n.rank) { meta.push(n.metric + " rank " + n.rank); }
    document.getElementById("kgviz-panel-meta").textContent = meta.join("  ·  ");
    document.getElementById("kgviz-panel-id").textContent = n.id;
    var body = document.getElementById("kgviz-panel-body");
    body.textContent = n.body || "";
    body.style.display = n.body ? "block" : "none";
    document.getElementById("kgviz-panel").style.display = "block";
  }
  function attach() {
    if (typeof network === "undefined") { window.setTimeout(attach, 120); return; }
    network.on("click", function (params) {
      if (params.nodes && params.nodes.length) { show(params.nodes[0]); }
    });
  }
  attach();
})();
</script>
""".replace("__KGVIZ_PAYLOAD__", payload)
