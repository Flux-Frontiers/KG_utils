"""Tests for kg_utils.snapshots."""

from __future__ import annotations

from pathlib import Path

import pytest

from kg_utils.snapshots import PruneResult, Snapshot, SnapshotManifest, SnapshotManager


# -- Snapshot model ----------------------------------------------------------


def test_snapshot_to_from_dict() -> None:
    snap = Snapshot(
        branch="main",
        timestamp="2026-01-01T00:00:00+00:00",
        version="1.0.0",
        metrics={"total_nodes": 10, "total_edges": 5},
        tree_hash="abc123",
    )
    d = snap.to_dict()
    assert d["key"] == "abc123"
    assert d["metrics"]["total_nodes"] == 10

    restored = Snapshot.from_dict(d)
    assert restored.key == "abc123"
    assert restored.metrics["total_nodes"] == 10
    assert restored.branch == "main"


def test_snapshot_key_property() -> None:
    snap = Snapshot(branch="main", timestamp="", metrics={}, tree_hash="deadbeef")
    assert snap.key == "deadbeef"


# -- SnapshotManifest --------------------------------------------------------


def test_manifest_round_trip() -> None:
    m = SnapshotManifest(
        format_version="1.0",
        last_update="2026-01-01",
        snapshots=[{"key": "a", "timestamp": "t1"}],
    )
    d = m.to_dict()
    restored = SnapshotManifest.from_dict(d)
    assert len(restored.snapshots) == 1
    assert restored.snapshots[0]["key"] == "a"


# -- SnapshotManager ---------------------------------------------------------


@pytest.fixture
def mgr(tmp_path: Path) -> SnapshotManager:
    return SnapshotManager(tmp_path / "snapshots", package_name="test-pkg")


def test_capture_and_save(mgr: SnapshotManager) -> None:
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 5, "total_edges": 3},
        tree_hash="hash1",
    )
    assert snap.key == "hash1"
    assert snap.metrics["total_nodes"] == 5

    path = mgr.save_snapshot(snap)
    assert path is not None and path.exists()


def test_save_rejects_zero_nodes(mgr: SnapshotManager) -> None:
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 0, "total_edges": 0},
        tree_hash="empty",
    )
    with pytest.raises(ValueError, match="0 nodes"):
        mgr.save_snapshot(snap)


def test_load_snapshot(mgr: SnapshotManager) -> None:
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="loadtest",
    )
    mgr.save_snapshot(snap)

    loaded = mgr.load_snapshot("loadtest")
    assert loaded is not None
    assert loaded.metrics["total_nodes"] == 10


def test_load_latest(mgr: SnapshotManager) -> None:
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 7, "total_edges": 2},
        tree_hash="latest1",
    )
    mgr.save_snapshot(snap)

    latest = mgr.load_snapshot("latest")
    assert latest is not None
    assert latest.key == "latest1"


def test_list_snapshots(mgr: SnapshotManager) -> None:
    for i, h in enumerate(["aaa", "bbb"]):
        snap = mgr.capture(
            version="0.1.0",
            branch="test",
            graph_stats_dict={"total_nodes": 10 + i, "total_edges": 5},
            tree_hash=h,
        )
        mgr.save_snapshot(snap, force=True)

    snaps = mgr.list_snapshots()
    assert len(snaps) == 2


def test_diff_snapshots(mgr: SnapshotManager) -> None:
    s1 = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5, "node_counts": {"file": 10}},
        tree_hash="diff_a",
    )
    s2 = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 15, "total_edges": 8, "node_counts": {"file": 15}},
        tree_hash="diff_b",
    )
    mgr.save_snapshot(s1, force=True)
    mgr.save_snapshot(s2, force=True)

    result = mgr.diff_snapshots("diff_a", "diff_b")
    assert "error" not in result
    assert result["delta"]["nodes"] == 5
    assert result["delta"]["edges"] == 3


def test_prune_dry_run(mgr: SnapshotManager) -> None:
    for h in ["p1", "p2", "p3"]:
        snap = mgr.capture(
            version="0.1.0",
            branch="test",
            graph_stats_dict={"total_nodes": 10, "total_edges": 5},
            tree_hash=h,
        )
        mgr.save_snapshot(snap, force=True)

    result = mgr.prune_snapshots(dry_run=True)
    assert isinstance(result, PruneResult)
    assert result.dry_run is True
    # p2 is a metric-duplicate interior entry
    assert len(result.removed) == 1


def test_prune_result_total_cleaned() -> None:
    pr = PruneResult(removed=["a"], orphaned_files=["b.json"], broken_entries=["c"], dry_run=False)
    assert pr.total_cleaned == 3


# -- Dedup behaviour (save_snapshot without force) ----------------------------


def test_save_dedup_refresh_inplace(mgr: SnapshotManager) -> None:
    snap1 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="hash1",
    )
    mgr.save_snapshot(snap1)

    snap2 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="hash2",
    )
    mgr.save_snapshot(snap2)  # same version + metrics → refresh in-place

    snaps = mgr.list_snapshots()
    assert len(snaps) == 1
    assert snaps[0]["key"] == "hash2"


def test_save_changed_metrics_appends(mgr: SnapshotManager) -> None:
    snap1 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="hash1",
    )
    mgr.save_snapshot(snap1)

    snap2 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 20, "total_edges": 10},
        tree_hash="hash2",
    )
    mgr.save_snapshot(snap2)  # changed metrics → new history entry

    snaps = mgr.list_snapshots()
    assert len(snaps) == 2


def test_save_force_always_appends(mgr: SnapshotManager) -> None:
    snap1 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="hash1",
    )
    mgr.save_snapshot(snap1)

    snap2 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="hash2",
    )
    mgr.save_snapshot(snap2, force=True)  # same metrics but force → new entry

    snaps = mgr.list_snapshots()
    assert len(snaps) == 2


def test_metrics_changed_subclass_hook(tmp_path: Path) -> None:
    class NodeOnlyManager(SnapshotManager):
        def _metrics_changed(
            self, new_metrics: dict, old_metrics: dict  # type: ignore[override]
        ) -> bool:
            return new_metrics.get("total_nodes") != old_metrics.get("total_nodes")

    node_mgr = NodeOnlyManager(tmp_path / "snapshots", package_name="test-pkg")

    snap1 = node_mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="hash1",
    )
    node_mgr.save_snapshot(snap1)

    # Same node count, different edge count → not "changed" per override → refresh in-place
    snap2 = node_mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 99},
        tree_hash="hash2",
    )
    node_mgr.save_snapshot(snap2)

    snaps = node_mgr.list_snapshots()
    assert len(snaps) == 1


# -- Extended prune scenarios --------------------------------------------------


def test_prune_baseline_and_latest_preserved(mgr: SnapshotManager) -> None:
    for h in ["p1", "p2", "p3"]:
        snap = mgr.capture(
            version="1.0.0",
            branch="main",
            graph_stats_dict={"total_nodes": 10, "total_edges": 5},
            tree_hash=h,
        )
        mgr.save_snapshot(snap, force=True)

    result = mgr.prune_snapshots(dry_run=True)
    assert "p1" not in result.removed  # baseline always kept
    assert "p3" not in result.removed  # latest always kept
    assert "p2" in result.removed


def test_prune_removes_actual_files(mgr: SnapshotManager) -> None:
    for h in ["p1", "p2", "p3"]:
        snap = mgr.capture(
            version="1.0.0",
            branch="main",
            graph_stats_dict={"total_nodes": 10, "total_edges": 5},
            tree_hash=h,
        )
        mgr.save_snapshot(snap, force=True)

    mgr.prune_snapshots(dry_run=False)

    assert not (mgr.snapshots_dir / "p2.json").exists()
    manifest = mgr.load_manifest()
    assert len(manifest.snapshots) == 2
    keys = {e["key"] for e in manifest.snapshots}
    assert keys == {"p1", "p3"}


def test_prune_orphaned_files(mgr: SnapshotManager) -> None:
    snap = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 5, "total_edges": 2},
        tree_hash="real",
    )
    mgr.save_snapshot(snap)

    orphan = mgr.snapshots_dir / "orphan.json"
    orphan.write_text('{"orphan": true}\n', encoding="utf-8")

    result = mgr.prune_snapshots(dry_run=True)
    assert "orphan.json" in result.orphaned_files


def test_prune_single_snapshot_noop(mgr: SnapshotManager) -> None:
    snap = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 5, "total_edges": 2},
        tree_hash="only",
    )
    mgr.save_snapshot(snap)

    result = mgr.prune_snapshots(dry_run=True)
    assert result.total_cleaned == 0


def test_prune_metrics_changed_override_respected(tmp_path: Path) -> None:
    class EdgeIgnoreManager(SnapshotManager):
        def _metrics_changed(
            self, new_metrics: dict, old_metrics: dict  # type: ignore[override]
        ) -> bool:
            return new_metrics.get("total_nodes") != old_metrics.get("total_nodes")

    edge_mgr = EdgeIgnoreManager(tmp_path / "snapshots", package_name="test-pkg")

    # All three entries have same node count → only differ in edges
    for i, h in enumerate(["q1", "q2", "q3"]):
        snap = edge_mgr.capture(
            version="1.0.0",
            branch="main",
            graph_stats_dict={"total_nodes": 10, "total_edges": i + 1},
            tree_hash=h,
        )
        edge_mgr.save_snapshot(snap, force=True)

    result = edge_mgr.prune_snapshots(dry_run=True)
    # Interior entry q2 is "unchanged" per EdgeIgnoreManager → should be pruned
    assert "q2" in result.removed
