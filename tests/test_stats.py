"""kg_utils.stats -- the fleet's one shape for a KG's statistics.

Settled 2026-10-25 (kgrag_priv FLEET_STANDARDS, "KG stats shape"): every
module's public ``stats()`` and every server's ``stats`` tool answer
``{"kind", "nodes": {"total", "by_kind"}, "edges": {"total", "by_rel"}, ...}``.
Snapshots keep persisting a store's flat keys, whichever form they are handed.
"""

from __future__ import annotations

import pytest

from kg_utils.snapshots.manager import SnapshotManager
from kg_utils.stats import flat_stats, kg_stats, stats_from_store, stats_problems

_STORE = {
    "db_path": "/x/graph.sqlite",
    "total_nodes": 5,
    "meaningful_nodes": 4,
    "total_edges": 3,
    "node_counts": {"function": 4, "module": 1},
    "edge_counts": {"CONTAINS": 3},
    "docstring_coverage": 0.5,
}


def test_totals_are_summed_from_the_breakdowns():
    s = kg_stats("code", {"function": 4, "module": 1}, {"CALLS": 2, "CONTAINS": 3}, extra=1)
    assert s == {
        "kind": "code",
        "nodes": {"total": 5, "by_kind": {"function": 4, "module": 1}},
        "edges": {"total": 5, "by_rel": {"CALLS": 2, "CONTAINS": 3}},
        "extra": 1,
    }
    assert stats_problems(s) == []


def test_an_empty_graph_is_valid():
    assert stats_problems(kg_stats("doc", {}, {})) == []


@pytest.mark.parametrize("key", ["kind", "nodes", "edges"])
def test_extras_may_not_shadow_the_shape(key):
    with pytest.raises(ValueError, match="reserved"):
        kg_stats("doc", {}, {}, **{key: 1})


def test_from_store_drops_the_flat_keys_and_keeps_the_rest():
    s = stats_from_store("code", _STORE, vector_backend="sqlite-vec")
    assert s["nodes"]["total"] == 5 and s["edges"]["by_rel"] == {"CONTAINS": 3}
    assert {"total_nodes", "total_edges", "node_counts", "edge_counts"}.isdisjoint(s)
    assert s["meaningful_nodes"] == 4  # a domain measure stays an extra, never the total
    assert s["db_path"] == "/x/graph.sqlite" and s["vector_backend"] == "sqlite-vec"


def test_flat_stats_is_the_inverse():
    flat = flat_stats(stats_from_store("code", _STORE))
    assert flat == {k: v for k, v in _STORE.items()}
    assert flat_stats(_STORE) == _STORE and flat_stats(_STORE) is not _STORE


@pytest.mark.parametrize(
    ("stats", "fragment"),
    [
        ([], "mapping"),
        ({"nodes": {"total": 0, "by_kind": {}}, "edges": {"total": 0, "by_rel": {}}}, "kind"),
        ({"kind": "doc", "edges": {"total": 0, "by_rel": {}}}, "`nodes` must be an object"),
        ({"kind": "doc", "node_count": 3, "edge_count": 2}, "`nodes` must be an object"),
        (
            {
                "kind": "doc",
                "nodes": {"total": 3, "by_kind": {"a": 2}},
                "edges": {"total": 0, "by_rel": {}},
            },
            "sums to 2, not `total` 3",
        ),
        (
            {
                "kind": "doc",
                "nodes": {"total": "3", "by_kind": {}},
                "edges": {"total": 0, "by_rel": {}},
            },
            "`nodes.total` must be a non-negative integer",
        ),
        (
            {
                "kind": "doc",
                "nodes": {"total": 0, "by_kind": {}},
                "edges": {"total": 1, "by_rel": {"x": True}},
            },
            "`edges.by_rel` must map names to integers",
        ),
    ],
)
def test_problems_name_each_departure(stats, fragment):
    assert any(fragment in p for p in stats_problems(stats)), stats_problems(stats)


def test_snapshots_persist_flat_keys_from_either_shape(tmp_path):
    mgr = SnapshotManager(tmp_path / "snaps", package_name="x-kg")
    nested = mgr.capture(graph_stats_dict=stats_from_store("code", _STORE), key="k1", tree_hash="t")
    flat = mgr.capture(graph_stats_dict=dict(_STORE), key="k2", tree_hash="t")
    for snap in (nested, flat):
        assert snap.metrics["total_nodes"] == 5
        assert snap.metrics["node_counts"] == {"function": 4, "module": 1}
        assert "nodes" not in snap.metrics
