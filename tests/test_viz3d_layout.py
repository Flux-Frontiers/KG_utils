"""
test_viz3d_layout.py

Tests for :mod:`kg_utils.viz3d` — the shared 3-D layout engine.

The design claim under test is that one layout engine serves every KG domain,
with the domain's hierarchy supplied as constructor arguments. So the tests lay
out *two* unrelated graphs — a code graph (modules/classes/functions) and a
document corpus (documents/sections/chunks) — through the same layouts, and
assert both are placed sensibly.

Author: Eric G. Suchanek, PhD
License: Elastic 2.0
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from kg_utils.viz3d import (  # noqa: E402
    AlliumLayout,
    FunnelLayout,
    Layout3D,
    LayoutEdge,
    LayoutNode,
    fibonacci_annulus,
    fibonacci_sphere,
    golden_spiral_2d,
)

# --- a code domain ---------------------------------------------------------

CODE_ZLEVELS = {"module": 0, "class": 1, "function": 2, "method": 2, "symbol": 3}
CODE_LEVEL_SIZES = {0: 1.2, 1: 0.9, 2: 0.7, 3: 0.4}

CODE_NODES = [
    LayoutNode("mod:a.py", "module", "a.py", module_path="a.py"),
    LayoutNode("cls:a.A", "class", "A", module_path="a.py", lineno=10, end_lineno=40),
    LayoutNode("fn:a.A.run", "method", "run", module_path="a.py", lineno=12, end_lineno=20),
    LayoutNode("fn:a.top", "function", "top", module_path="a.py"),
    LayoutNode("mod:b.py", "module", "b.py", module_path="b.py"),
    LayoutNode("cls:b.B", "class", "B", module_path="b.py"),
]
CODE_EDGES = [
    LayoutEdge("mod:a.py", "CONTAINS", "cls:a.A"),
    LayoutEdge("mod:a.py", "CONTAINS", "fn:a.top"),
    LayoutEdge("cls:a.A", "CONTAINS", "fn:a.A.run"),
    LayoutEdge("mod:b.py", "CONTAINS", "cls:b.B"),
    LayoutEdge("fn:a.top", "CALLS", "fn:a.A.run"),
]

# --- a document domain -----------------------------------------------------

DOC_ZLEVELS = {"document": 0, "section": 1, "chunk": 2}
DOC_LEVEL_SIZES = {0: 1.8, 1: 0.9, 2: 0.28}

DOC_NODES = [
    LayoutNode("doc:iliad", "document", "The Iliad", module_path="iliad.md"),
    LayoutNode("sec:iliad#1", "section", "Book I", docstring="Sing, O goddess..."),
    LayoutNode("chunk:iliad#1.1", "chunk", "1.1", docstring="the anger of Achilles"),
    LayoutNode("doc:odyssey", "document", "The Odyssey", module_path="odyssey.md"),
]
DOC_EDGES = [
    LayoutEdge("doc:iliad", "HAS_SECTION", "sec:iliad#1"),
    LayoutEdge("sec:iliad#1", "HAS_SECTION", "chunk:iliad#1.1"),
]


# ---------------------------------------------------------------------------
# Fibonacci utilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("samples", [0, 1, 2, 7, 100])
def test_fibonacci_sphere_count(samples):
    """Every requested sample comes back, including the degenerate counts."""
    assert len(fibonacci_sphere(samples)) == samples


def test_fibonacci_sphere_lies_on_sphere():
    """Points sit on the sphere of the requested radius, around the requested centre."""
    center = np.array([1.0, -2.0, 3.0])
    pts = fibonacci_sphere(50, radius=4.0, center=center)
    radii = [np.linalg.norm(p - center) for p in pts]
    assert np.allclose(radii, 4.0)


def test_fibonacci_sphere_single_sample_is_the_pole():
    """One sample degenerates to the north pole rather than dividing by zero."""
    (pt,) = fibonacci_sphere(1, radius=2.0)
    assert np.allclose(pt, [0.0, 0.0, 2.0])


@pytest.mark.parametrize("samples", [0, 1, 3, 40])
def test_fibonacci_annulus_count(samples):
    """Every requested sample comes back, including the degenerate counts."""
    assert len(fibonacci_annulus(samples)) == samples


def test_fibonacci_annulus_radii_within_band():
    """Points stay inside the requested annular band, flat when z_thickness is 0."""
    pts = fibonacci_annulus(30, inner_radius=5.0, outer_radius=9.0, z_thickness=0.0)
    for p in pts:
        r = np.hypot(p[0], p[1])
        assert 5.0 - 1e-9 <= r <= 9.0 + 1e-9
        assert p[2] == pytest.approx(0.0)


def test_fibonacci_annulus_jitter_is_deterministic():
    """Repeated calls must agree, or node positions would jump between renders."""
    first = fibonacci_annulus(20, z_thickness=0.5)
    second = fibonacci_annulus(20, z_thickness=0.5)
    assert np.allclose(np.array(first), np.array(second))
    assert np.any(np.array(first)[:, 2] != 0.0)  # jitter actually applied


def test_golden_spiral_2d_is_planar_and_bounded():
    """All points share the given Z and stay within the disc radius."""
    pts = golden_spiral_2d(60, radius=10.0, z=3.5)
    assert len(pts) == 60
    for p in pts:
        assert p[2] == pytest.approx(3.5)
        assert np.hypot(p[0], p[1]) <= 10.0 + 1e-9


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


def test_layout_node_from_dict_reads_optional_fields():
    """Optional provenance is picked up when present."""
    n = LayoutNode.from_dict(
        {
            "id": "fn:x",
            "kind": "function",
            "name": "x",
            "module_path": "x.py",
            "docstring": "does x",
            "lineno": 5,
            "end_lineno": 25,
        }
    )
    assert (n.id, n.kind, n.name) == ("fn:x", "function", "x")
    assert n.module_path == "x.py"
    assert n.line_count == 20


def test_layout_node_from_dict_tolerates_a_bare_node():
    """Only id/kind/name are common across domains; the rest may be absent."""
    n = LayoutNode.from_dict({"id": "cmpd:atp", "kind": "compound", "name": "ATP"})
    assert n.module_path is None
    assert n.docstring is None
    assert n.line_count == 0


def test_layout_edge_from_dict():
    """Edges carry the domain's own relation vocabulary verbatim."""
    e = LayoutEdge.from_dict({"src": "a", "rel": "SIMILAR_TO", "dst": "b"})
    assert (e.src, e.rel, e.dst) == ("a", "SIMILAR_TO", "b")


def test_layout3d_cannot_be_instantiated():
    """The ABC is a contract, not a usable layout."""
    with pytest.raises(TypeError):
        Layout3D()


# ---------------------------------------------------------------------------
# AlliumLayout
# ---------------------------------------------------------------------------


def test_allium_places_every_node_exactly_once():
    """No node is dropped, whatever its position in the hierarchy."""
    pos = AlliumLayout().compute(CODE_NODES, CODE_EDGES)
    assert set(pos) == {n.id for n in CODE_NODES}
    assert all(p.shape == (3,) for p in pos.values())


def test_allium_stems_sit_on_the_ground_and_heads_float():
    """Roots are at Z=0; their children cluster around the stem apex."""
    layout = AlliumLayout(stem_height=8.0)
    pos = layout.compute(CODE_NODES, CODE_EDGES)

    assert pos["mod:a.py"][2] == pytest.approx(0.0)
    assert pos["mod:b.py"][2] == pytest.approx(0.0)

    apex = np.array([pos["mod:a.py"][0], pos["mod:a.py"][1], 8.0])
    for child in ("cls:a.A", "fn:a.top"):
        assert np.linalg.norm(pos[child] - apex) < 5.0

    # The method orbits its own class, far closer than it is to the stem base
    assert np.linalg.norm(pos["fn:a.A.run"] - pos["cls:a.A"]) < 2.0


def test_allium_head_radius_formula_is_a_contract():
    """Head radius is ``base_head_radius + sqrt(n_children) * 0.4``.

    Downstream renderers size their node glyphs against the room this formula
    leaves them — pycode_kg's `test_viz3d_sizing` asserts a max-centrality
    function still fits inside the head of a four-child module. Changing the
    coefficient silently re-tunes every consumer's occlusion budget, so it is
    pinned here rather than left as an implementation detail.
    """
    n_children = 6
    root = LayoutNode("mod:big", "module", "big")
    kids = [LayoutNode(f"fn:{i}", "function", str(i)) for i in range(n_children)]
    edges = [LayoutEdge("mod:big", "CONTAINS", k.id) for k in kids]

    layout = AlliumLayout(stem_height=8.0, base_head_radius=2.0)
    pos = layout.compute([root, *kids], edges)

    apex = np.array([pos["mod:big"][0], pos["mod:big"][1], 8.0])
    expected = 2.0 + np.sqrt(n_children) * 0.4
    for k in kids:
        assert np.linalg.norm(pos[k.id] - apex) == pytest.approx(expected)


def test_allium_orbit_radius_formula_is_a_contract():
    """Grandchild orbit radius is ``method_orbit_radius + sqrt(n) * 0.15``."""
    n_grand = 9
    root = LayoutNode("mod:m", "module", "m")
    parent = LayoutNode("cls:C", "class", "C")
    grand = [LayoutNode(f"m:{i}", "method", str(i)) for i in range(n_grand)]
    edges = [
        LayoutEdge("mod:m", "CONTAINS", "cls:C"),
        *[LayoutEdge("cls:C", "CONTAINS", g.id) for g in grand],
    ]

    layout = AlliumLayout(method_orbit_radius=0.8)
    pos = layout.compute([root, parent, *grand], edges)

    expected = 0.8 + np.sqrt(n_grand) * 0.15
    for g in grand:
        assert np.linalg.norm(pos[g.id] - pos["cls:C"]) == pytest.approx(expected)


@pytest.mark.parametrize("layout_factory", [AlliumLayout, FunnelLayout])
def test_layouts_are_deterministic(layout_factory):
    """Identical input must give identical positions.

    A viewer recomputes the layout on every filter change and re-render; a
    layout that drifted would make nodes jump for no user-visible reason.
    """
    first = layout_factory().compute(CODE_NODES, CODE_EDGES)
    second = layout_factory().compute(CODE_NODES, CODE_EDGES)
    assert set(first) == set(second)
    for nid in first:
        assert np.allclose(first[nid], second[nid])


def test_allium_slot_assignment_follows_node_order():
    """Roots take annulus slots in list order — so callers must pass a stable one.

    This is a documented constraint, not an accident: reordering the node list
    moves every root to a different slot. Callers feed the layout whatever order
    their store returned, so a store whose ordering is not stable across
    rebuilds will make the whole scene shuffle between renders even though
    nothing in the graph changed.

    Sorting internally would make this moot but would also relocate every node
    in every existing scene, so the constraint is pinned here instead.
    """
    forward = AlliumLayout().compute(CODE_NODES, CODE_EDGES)
    reverse = AlliumLayout().compute(list(reversed(CODE_NODES)), CODE_EDGES)

    # Same node set, same slots occupied — but not the same node in each slot.
    assert set(forward) == set(reverse)
    assert not np.allclose(forward["mod:a.py"], reverse["mod:a.py"])


def test_allium_root_kind_and_contains_rel_are_domain_supplied():
    """A document corpus gets the same treatment by naming its own vocabulary."""
    layout = AlliumLayout(root_kind="document", contains_rel="HAS_SECTION")
    pos = layout.compute(DOC_NODES, DOC_EDGES)

    assert set(pos) == {n.id for n in DOC_NODES}
    assert pos["doc:iliad"][2] == pytest.approx(0.0)
    assert pos["doc:odyssey"][2] == pytest.approx(0.0)
    # The section hangs off its document's stem, not off the origin cluster
    assert pos["sec:iliad#1"][2] > 1.0


def test_allium_falls_back_to_unparented_nodes_when_no_kind_matches():
    """A graph with no node of root_kind still lays out, rooted on the orphans."""
    pos = AlliumLayout(root_kind="package").compute(CODE_NODES, CODE_EDGES)
    assert set(pos) == {n.id for n in CODE_NODES}
    # Both modules are unparented, so both become pseudo-roots at ground level
    assert pos["mod:a.py"][2] == pytest.approx(0.0)
    assert pos["mod:b.py"][2] == pytest.approx(0.0)


def test_allium_places_orphans():
    """Nodes reachable from nothing are still given a home."""
    nodes = [*CODE_NODES, LayoutNode("sym:os.path", "symbol", "os.path")]
    pos = AlliumLayout().compute(nodes, CODE_EDGES)
    assert "sym:os.path" in pos


def test_allium_handles_an_empty_graph():
    """Zero nodes is a valid graph, not an error."""
    assert AlliumLayout().compute([], []) == {}


# ---------------------------------------------------------------------------
# FunnelLayout
# ---------------------------------------------------------------------------


def test_funnel_stratifies_by_injected_zlevels():
    """Z comes from the caller's kind→level map, times the layer gap."""
    layout = FunnelLayout(layer_gap=10.0, zlevels=CODE_ZLEVELS, level_sizes=CODE_LEVEL_SIZES)
    pos = layout.compute(CODE_NODES, CODE_EDGES)

    assert set(pos) == {n.id for n in CODE_NODES}
    assert pos["mod:a.py"][2] == pytest.approx(0.0)
    assert pos["cls:a.A"][2] == pytest.approx(10.0)
    assert pos["fn:a.top"][2] == pytest.approx(20.0)
    assert pos["fn:a.A.run"][2] == pytest.approx(20.0)  # methods share the function layer


def test_funnel_serves_a_document_domain_with_its_own_levels():
    """The same layout, a different hierarchy — supplied as data."""
    layout = FunnelLayout(layer_gap=6.0, zlevels=DOC_ZLEVELS, level_sizes=DOC_LEVEL_SIZES)
    pos = layout.compute(DOC_NODES, DOC_EDGES)

    assert pos["doc:iliad"][2] == pytest.approx(0.0)
    assert pos["sec:iliad#1"][2] == pytest.approx(6.0)
    assert pos["chunk:iliad#1.1"][2] == pytest.approx(12.0)


def test_funnel_without_zlevels_is_a_flat_disc():
    """Declaring no hierarchy renders as no hierarchy rather than guessing one."""
    pos = FunnelLayout().compute(CODE_NODES, CODE_EDGES)
    assert set(pos) == {n.id for n in CODE_NODES}
    assert all(p[2] == pytest.approx(0.0) for p in pos.values())


def test_funnel_unknown_kinds_land_on_default_level():
    """Domains reserving a top layer for stubs pass its index as default_level."""
    nodes = [*CODE_NODES, LayoutNode("weird:1", "gizmo", "gizmo")]
    layout = FunnelLayout(
        layer_gap=10.0,
        zlevels=CODE_ZLEVELS,
        level_sizes=CODE_LEVEL_SIZES,
        default_level=3,
    )
    pos = layout.compute(nodes, CODE_EDGES)
    assert pos["weird:1"][2] == pytest.approx(30.0)


def test_funnel_disc_radius_grows_with_layer_population():
    """Radius derives from sqrt(n), so big layers spread instead of overlapping."""
    small = [LayoutNode(f"m{i}", "module", f"m{i}") for i in range(10)]
    large = [LayoutNode(f"m{i}", "module", f"m{i}") for i in range(400)]
    layout = FunnelLayout(zlevels=CODE_ZLEVELS, level_sizes=CODE_LEVEL_SIZES)

    def extent(nodes):
        pts = np.array(list(layout.compute(nodes, []).values()))
        return np.max(np.hypot(pts[:, 0], pts[:, 1]))

    assert extent(large) > extent(small)


def test_funnel_handles_an_empty_graph():
    """Zero nodes is a valid graph, not an error."""
    assert FunnelLayout().compute([], []) == {}


class TestExtraBoundaries:
    """The layering claim: importing a layout must not drag a renderer in.

    ``kg_utils.viz3d`` is the ``viz3d`` extra — numpy only. PyVista belongs to
    ``viz3d-render`` and PyQt to ``viz3d-qt``, and both are reached by importing
    a *different* module. Re-exporting :mod:`kg_utils.viz3d.qt` from the package
    ``__init__`` for convenience would silently make PyQt a hard requirement of
    every consumer that just wants coordinates, which is exactly the mistake
    these subprocess checks exist to catch.
    """

    @staticmethod
    def _imports_after(statement: str) -> set[str]:
        """:return: Third-party modules present in ``sys.modules`` after *statement*."""
        import subprocess
        import sys

        probe = f"import sys; {statement}; print(' '.join(sorted(sys.modules)))"
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        return set(out.stdout.split())

    def test_importing_viz3d_does_not_import_pyqt(self):
        assert "PyQt5" not in self._imports_after("import kg_utils.viz3d")

    def test_importing_viz3d_does_not_import_pyvista(self):
        assert "pyvista" not in self._imports_after("import kg_utils.viz3d")

    def test_qt_is_not_reachable_from_the_package_namespace(self):
        """It must be `from kg_utils.viz3d.qt import ...`, never a re-export."""
        import kg_utils.viz3d as viz3d

        assert not hasattr(viz3d, "PovRenderSession")
        assert "PovRenderSession" not in getattr(viz3d, "__all__", [])
