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


# -- Snapshot.from_dict legacy fields ----------------------------------------


def test_snapshot_from_dict_legacy_tree_hash_key() -> None:
    """from_dict must handle dicts that use 'tree_hash' instead of 'key'."""
    data = {
        "tree_hash": "abc",
        "branch": "main",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "version": "1.0.0",
        "metrics": {"total_nodes": 5, "total_edges": 2},
    }
    snap = Snapshot.from_dict(data)
    assert snap.key == "abc"


def test_snapshot_from_dict_drops_legacy_commit_field() -> None:
    """Legacy 'commit' field must be silently ignored."""
    data = {
        "key": "def456",
        "commit": "some-old-commit-sha",
        "branch": "main",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "version": "1.0.0",
        "metrics": {"total_nodes": 3, "total_edges": 1},
    }
    snap = Snapshot.from_dict(data)
    assert snap.key == "def456"
    assert not hasattr(snap, "commit")


def test_snapshot_from_dict_version_defaults_to_empty_string() -> None:
    data = {
        "key": "v0",
        "branch": "main",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "metrics": {},
    }
    snap = Snapshot.from_dict(data)
    assert snap.version == ""


def test_snapshot_from_dict_vs_deltas_preserved() -> None:
    data = {
        "key": "hash1",
        "branch": "main",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "version": "1.0",
        "metrics": {"total_nodes": 10, "total_edges": 5},
        "vs_previous": {"nodes": 2, "edges": 1},
        "vs_baseline": {"nodes": 10, "edges": 5},
    }
    snap = Snapshot.from_dict(data)
    assert snap.vs_previous == {"nodes": 2, "edges": 1}
    assert snap.vs_baseline == {"nodes": 10, "edges": 5}


# -- SnapshotManifest.from_dict defaults -------------------------------------


def test_manifest_from_dict_missing_fields_default() -> None:
    manifest = SnapshotManifest.from_dict({})
    assert manifest.format_version == "1.0"
    assert manifest.last_update == ""
    assert manifest.snapshots == []


# -- SnapshotManager additional cases ----------------------------------------


def test_load_snapshot_missing_key_returns_none(mgr: SnapshotManager) -> None:
    assert mgr.load_snapshot("does-not-exist") is None


def test_load_latest_empty_manifest_returns_none(mgr: SnapshotManager) -> None:
    assert mgr.load_snapshot("latest") is None


def test_get_previous_empty_manifest_returns_none(mgr: SnapshotManager) -> None:
    assert mgr.get_previous("anything") is None


def test_get_baseline_empty_manifest_returns_none(mgr: SnapshotManager) -> None:
    assert mgr.get_baseline() is None


def test_get_previous_returns_older_snapshot(mgr: SnapshotManager) -> None:
    s1 = mgr.capture(
        version="0.1.0",
        branch="main",
        graph_stats_dict={"total_nodes": 5, "total_edges": 2},
        tree_hash="older",
    )
    mgr.save_snapshot(s1, force=True)

    s2 = mgr.capture(
        version="0.1.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 4},
        tree_hash="newer",
    )
    mgr.save_snapshot(s2, force=True)

    prev = mgr.get_previous("newer")
    assert prev is not None
    assert prev.key == "older"


def test_get_baseline_returns_oldest(mgr: SnapshotManager) -> None:
    for h, n in [("first", 5), ("second", 10), ("third", 15)]:
        snap = mgr.capture(
            version="0.1.0",
            branch="main",
            graph_stats_dict={"total_nodes": n, "total_edges": 2},
            tree_hash=h,
        )
        mgr.save_snapshot(snap, force=True)

    baseline = mgr.get_baseline()
    assert baseline is not None
    assert baseline.key == "first"


def test_list_snapshots_limit(mgr: SnapshotManager) -> None:
    for i, h in enumerate(["x1", "x2", "x3"]):
        snap = mgr.capture(
            version="0.1.0",
            branch="main",
            graph_stats_dict={"total_nodes": 10 + i, "total_edges": 2},
            tree_hash=h,
        )
        mgr.save_snapshot(snap, force=True)

    assert len(mgr.list_snapshots(limit=2)) == 2


def test_list_snapshots_branch_filter(mgr: SnapshotManager) -> None:
    for h, branch in [("br1", "feature"), ("br2", "main"), ("br3", "feature")]:
        snap = mgr.capture(
            version="0.1.0",
            branch=branch,
            graph_stats_dict={"total_nodes": 10, "total_edges": 2},
            tree_hash=h,
        )
        mgr.save_snapshot(snap, force=True)

    feature_snaps = mgr.list_snapshots(branch="feature")
    assert len(feature_snaps) == 2
    assert all(s["branch"] == "feature" for s in feature_snaps)


def test_diff_snapshots_missing_returns_error(mgr: SnapshotManager) -> None:
    result = mgr.diff_snapshots("no-such-a", "no-such-b")
    assert "error" in result


def test_save_snapshot_dedup_refreshes_in_place(mgr: SnapshotManager) -> None:
    """Saving with same version + metrics updates timestamp, not a new entry."""
    snap1 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="hash-a",
    )
    mgr.save_snapshot(snap1)

    snap2 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="hash-b",
    )
    mgr.save_snapshot(snap2)  # same metrics → dedup

    snaps = mgr.list_snapshots()
    assert len(snaps) == 1
    assert snaps[0]["key"] == "hash-b"


def test_save_snapshot_force_adds_new_entry(mgr: SnapshotManager) -> None:
    snap1 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="force-a",
    )
    mgr.save_snapshot(snap1, force=True)

    snap2 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="force-b",
    )
    mgr.save_snapshot(snap2, force=True)

    assert len(mgr.list_snapshots()) == 2


def test_prune_removes_duplicates(mgr: SnapshotManager) -> None:
    for h in ["dup1", "dup2", "dup3"]:
        snap = mgr.capture(
            version="0.1.0",
            branch="main",
            graph_stats_dict={"total_nodes": 10, "total_edges": 5},
            tree_hash=h,
        )
        mgr.save_snapshot(snap, force=True)

    result = mgr.prune_snapshots(dry_run=False)
    assert len(result.removed) == 1
    remaining = mgr.list_snapshots()
    assert len(remaining) == 2


def test_prune_removes_orphaned_files(mgr: SnapshotManager) -> None:
    snap = mgr.capture(
        version="0.1.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="real-snap",
    )
    mgr.save_snapshot(snap, force=True)

    orphan = mgr.snapshots_dir / "orphan-file.json"
    orphan.write_text('{"key": "orphan"}', encoding="utf-8")

    result = mgr.prune_snapshots(dry_run=False)
    assert "orphan-file.json" in result.orphaned_files
    assert not orphan.exists()


def test_prune_reports_broken_entries(mgr: SnapshotManager) -> None:
    """Broken = manifest entry whose JSON file is missing."""
    snap = mgr.capture(
        version="0.1.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="broken-snap",
    )
    mgr.save_snapshot(snap, force=True)

    # Delete the file behind the manifest entry
    (mgr.snapshots_dir / "broken-snap.json").unlink()

    result = mgr.prune_snapshots(dry_run=True)
    assert "broken-snap" in result.broken_entries


def test_compute_delta_from_metrics(mgr: SnapshotManager) -> None:
    delta = mgr._compute_delta_from_metrics(
        {"total_nodes": 20, "total_edges": 10},
        {"total_nodes": 15, "total_edges": 7},
    )
    assert delta == {"nodes": 5, "edges": 3}


def test_metrics_changed_same(mgr: SnapshotManager) -> None:
    m = {"total_nodes": 10, "total_edges": 5}
    assert not mgr._metrics_changed(m, m.copy())


def test_metrics_changed_different(mgr: SnapshotManager) -> None:
    assert mgr._metrics_changed(
        {"total_nodes": 10, "total_edges": 5},
        {"total_nodes": 11, "total_edges": 5},
    )


def test_load_manifest_normalises_legacy_tree_hash(mgr: SnapshotManager) -> None:
    import json

    raw = {
        "format": "1.0",
        "last_update": "2026-01-01T00:00:00+00:00",
        "snapshots": [{"tree_hash": "legacy-key", "timestamp": "2026-01-01T00:00:00+00:00"}],
    }
    mgr.manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    manifest = mgr.load_manifest()
    assert manifest.snapshots[0]["key"] == "legacy-key"
    assert "tree_hash" not in manifest.snapshots[0]
