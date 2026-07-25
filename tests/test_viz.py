"""
test_viz.py

Tests for :mod:`kg_utils.viz` — the shared graph renderer.

The design claim under test is that one renderer serves every KG domain, with
the differences supplied as data. So the tests build *two* unrelated domains — a
code graph and a document graph, whose node schemas share only ``id``, ``kind``
and ``name`` — and assert both render correctly through the same code path.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pyvis")

from kg_utils.analysis.scores import ScoreSet  # noqa: E402
from kg_utils.viz import (  # noqa: E402
    GraphTheme,
    KindStyle,
    TooltipRow,
    TooltipSpec,
    build_graph_html,
    select_nodes,
    with_alpha,
)

# --- a code domain ---------------------------------------------------------

CODE_THEME = GraphTheme(
    kinds={
        "module": KindStyle("#4A90D9", shape="box", size=18),
        "function": KindStyle("#27AE60", shape="ellipse"),
        "private_function": KindStyle("#F1C40F", shape="ellipse"),
    },
    fallback=KindStyle("#95A5A6", shape="triangle"),
    relations={"CONTAINS": "#BDC3C7", "CALLS": "#E74C3C"},
    resolve_kind=lambda n: (
        "private_function"
        if n.get("kind") == "function" and str(n.get("name", "")).startswith("_")
        else str(n.get("kind", ""))
    ),
)
CODE_TOOLTIP = TooltipSpec(
    title="qualname",
    rows=(TooltipRow("module_path", prefix="📄 "),),
    body="docstring",
)
CODE_NODES: list[dict[str, Any]] = [
    {"id": "mod:a", "kind": "module", "name": "a", "qualname": "a", "module_path": "a.py"},
    {
        "id": "fn:hot",
        "kind": "function",
        "name": "hot",
        "qualname": "a.hot",
        "module_path": "a.py",
        "docstring": "Does the thing.",
    },
    {"id": "fn:_priv", "kind": "function", "name": "_priv", "qualname": "a._priv"},
]
CODE_EDGES = [{"src": "mod:a", "rel": "CONTAINS", "dst": "fn:hot"}]

# --- a document domain, sharing only id/kind/name --------------------------

DOC_THEME = GraphTheme(
    kinds={
        "document": KindStyle("#8E44AD", shape="box", size=20),
        "chunk": KindStyle("#1ABC9C", shape="dot"),
    },
    fallback=KindStyle("#95A5A6"),
    relations={"CONTAINS": "#BDC3C7", "SIMILAR_TO": "#F39C12"},
)
DOC_TOOLTIP = TooltipSpec(
    title="title",
    rows=(TooltipRow("file_path", prefix="📁 "),),
    body="text",
)
DOC_NODES: list[dict[str, Any]] = [
    {"id": "doc:g", "kind": "document", "name": "guide", "title": "Guide", "file_path": "g.md"},
    {"id": "chk:1", "kind": "chunk", "name": "c1", "title": "Intro", "text": "Hello."},
]
DOC_EDGES = [{"src": "doc:g", "rel": "CONTAINS", "dst": "chk:1"}]


# ---------------------------------------------------------------------------
# The design claim: one renderer, many domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nodes", "edges", "theme", "tooltip"),
    [
        (CODE_NODES, CODE_EDGES, CODE_THEME, CODE_TOOLTIP),
        (DOC_NODES, DOC_EDGES, DOC_THEME, DOC_TOOLTIP),
    ],
    ids=["code", "doc"],
)
def test_renders_any_domain(nodes, edges, theme, tooltip) -> None:
    """Domains sharing only id/kind/name both render through one code path."""
    html = build_graph_html(nodes, edges, theme=theme, tooltip=tooltip)
    for node in nodes:
        assert node["id"] in html


def test_domain_fields_reach_the_tooltip() -> None:
    """Each domain's own fields appear, not a hardcoded code schema."""
    doc = build_graph_html(DOC_NODES, DOC_EDGES, theme=DOC_THEME, tooltip=DOC_TOOLTIP)
    assert "g.md" in doc
    assert "module_path" not in doc

    code = build_graph_html(CODE_NODES, CODE_EDGES, theme=CODE_THEME, tooltip=CODE_TOOLTIP)
    assert "a.py" in code


def test_resolve_kind_hook_applies() -> None:
    """A render-only kind the store never stores still gets its own colour."""
    html = build_graph_html(CODE_NODES, CODE_EDGES, theme=CODE_THEME, tooltip=CODE_TOOLTIP)
    assert CODE_THEME.kinds["private_function"].color in html


def test_unknown_kind_falls_back_rather_than_vanishing() -> None:
    """Every graph eventually meets an unplanned kind; drawing it grey beats dropping it."""
    nodes = [{"id": "x:1", "kind": "surprise", "name": "x"}]
    html = build_graph_html(nodes, [], theme=DOC_THEME, tooltip=DOC_TOOLTIP)
    assert "x:1" in html
    assert DOC_THEME.fallback.color in html


# ---------------------------------------------------------------------------
# Self-containment
# ---------------------------------------------------------------------------


def test_output_is_self_contained() -> None:
    """The page must render offline and inside a srcdoc iframe."""
    html = build_graph_html(CODE_NODES, CODE_EDGES, theme=CODE_THEME)
    assert "cdnjs.cloudflare.com/ajax/libs/vis-network" not in html
    assert 'src="lib/bindings/utils.js"' not in html
    assert len(html) > 500_000


def test_render_writes_nothing_to_the_working_directory(tmp_path, monkeypatch) -> None:
    """pyvis's default writes a lib/ directory as a side effect; ours must not."""
    monkeypatch.chdir(tmp_path)
    build_graph_html(CODE_NODES, CODE_EDGES, theme=CODE_THEME)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Escape hatch and optional tooltip
# ---------------------------------------------------------------------------


def test_callable_tooltip_overrides_the_spec() -> None:
    """Option D's escape hatch: bespoke markup when the spec cannot express it."""
    html = build_graph_html(
        CODE_NODES,
        CODE_EDGES,
        theme=CODE_THEME,
        tooltip=lambda node, color: f"<b>CUSTOM {node['id']}</b>",
    )
    assert "CUSTOM mod:a" in html


def test_tooltip_is_optional() -> None:
    """Omitting the spec still produces a usable graph."""
    html = build_graph_html(CODE_NODES, CODE_EDGES, theme=CODE_THEME, tooltip=None)
    assert "mod:a" in html


def test_node_text_cannot_break_out_of_the_page() -> None:
    """Node text containing markup must never reach the page unescaped.

    Two vectors, and the payload one is the dangerous half: node data is
    embedded in a ``<script>`` block, so a node containing ``</script>`` would
    terminate that block during HTML parsing and inject whatever followed.
    Asserted on the property rather than a particular encoding, because pyvis
    applies its own escaping on top of ours.
    """
    hostile = "</script><img src=x onerror=alert(1)>"
    nodes = [{"id": "x", "kind": "chunk", "name": "x", "title": hostile, "text": hostile}]
    html = build_graph_html(nodes, [], theme=DOC_THEME, tooltip=DOC_TOOLTIP)

    assert hostile not in html
    assert "<img src=x" not in html
    # The text is still shown to the user, just inertly.
    assert "onerror=alert(1)" in html


# ---------------------------------------------------------------------------
# Centrality encoding
# ---------------------------------------------------------------------------


def _scores(ids: list[str]) -> ScoreSet:
    """Build a ScoreSet ranking *ids* best-first.

    :param ids: Node IDs, most central first.
    :return: A populated ScoreSet.
    """
    return ScoreSet(
        metric="m",
        table="centrality_scores",
        scores={i: float(len(ids) - n) for n, i in enumerate(ids)},
        ranks={i: n + 1 for n, i in enumerate(ids)},
    )


def test_without_scores_nodes_use_theme_sizes() -> None:
    """Uniform sizing falls back to the theme, fully opaque."""
    html = build_graph_html(CODE_NODES, CODE_EDGES, theme=CODE_THEME, tooltip=CODE_TOOLTIP)
    assert "rgba(" not in html.split("var NODE_DATA")[0].split("nodes = new vis.DataSet")[1][:2000]


def test_with_scores_more_central_renders_larger() -> None:
    """Size must track centrality, not declaration order."""
    import json
    import re

    scores = _scores(["fn:hot", "mod:a", "fn:_priv"])
    html = build_graph_html(
        CODE_NODES, CODE_EDGES, theme=CODE_THEME, tooltip=CODE_TOOLTIP, scores=scores
    )
    payload = re.search(r"nodes\s*=\s*new vis\.DataSet\((\[.*?\])\);", html, re.S)
    assert payload
    by_id = {n["id"]: n for n in json.loads(payload.group(1))}
    assert by_id["fn:hot"]["size"] > by_id["fn:_priv"]["size"]


def test_highlight_ids_are_marked() -> None:
    """Callers can point at nodes without knowing how highlighting is drawn."""
    html = build_graph_html(CODE_NODES, CODE_EDGES, theme=CODE_THEME, highlight_ids={"fn:hot"})
    assert "#FFD700" in html


# ---------------------------------------------------------------------------
# select_nodes
# ---------------------------------------------------------------------------


def test_select_returns_everything_under_the_limit() -> None:
    """No selection needed when the graph already fits."""
    kept, how = select_nodes(list(CODE_NODES), 10, None)
    assert len(kept) == len(CODE_NODES)
    assert how == "all matching nodes"


def test_select_seeds_then_expands() -> None:
    """Seeding and expanding keeps the picture connected."""
    nodes = [{"id": f"n{i}", "kind": "chunk", "name": str(i)} for i in range(10)]
    scores = _scores([f"n{i}" for i in range(9, -1, -1)])
    kept, how = select_nodes(
        nodes, 4, scores, "central", expand=lambda seeds, hop: seeds | {"n0", "n1"}
    )
    ids = [n["id"] for n in kept]
    assert ids[0] == "n9"
    assert {"n0", "n1"} <= set(ids)
    assert "plus neighbours" in how


def test_select_without_scores_uses_store_order() -> None:
    """With nothing to rank by, the natural order is the honest fallback."""
    nodes = [{"id": f"n{i}", "kind": "chunk", "name": str(i)} for i in range(10)]
    kept, how = select_nodes(nodes, 3, None, "central")
    assert [n["id"] for n in kept] == ["n0", "n1", "n2"]
    assert how == "first in store order"


# ---------------------------------------------------------------------------
# with_alpha
# ---------------------------------------------------------------------------


def test_with_alpha_converts_and_clamps() -> None:
    """Opacity conversion is total: valid hex converts, anything else passes through."""
    assert with_alpha("#4A90D9", 0.5) == "rgba(74, 144, 217, 0.500)"
    assert with_alpha("#4A90D9", 5.0).endswith("1.000)")
    assert with_alpha("#4A90D9", -1.0).endswith("0.000)")
    assert with_alpha("red", 0.5) == "red"
