"""Tests for kg_utils.snapshots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from kg_utils.snapshots import PruneResult, Snapshot, SnapshotManager, SnapshotManifest

# -- Snapshot model ----------------------------------------------------------


def test_snapshot_to_from_dict() -> None:
    snap = Snapshot(
        branch="main",
        timestamp="2026-01-01T00:00:00+00:00",
        version="1.0.0",
        metrics={"total_nodes": 10, "total_edges": 5},
        tree_hash="abc123",
        snapshot_key="abc123",
    )
    d = snap.to_dict()
    assert d["key"] == "abc123"
    assert d["metrics"]["total_nodes"] == 10

    restored = Snapshot.from_dict(d)
    assert restored.key == "abc123"
    assert restored.metrics["total_nodes"] == 10
    assert restored.branch == "main"


def test_snapshot_key_property() -> None:
    """A snapshot with no supplied key falls back to its tree hash."""
    snap = Snapshot(branch="main", timestamp="", metrics={}, tree_hash="deadbeef")
    assert snap.key == "deadbeef"


def test_supplied_key_wins_over_tree_hash() -> None:
    snap = Snapshot(
        branch="main", timestamp="", metrics={}, tree_hash="deadbeef", snapshot_key="v1.2.3"
    )
    assert snap.key == "v1.2.3"
    assert snap.to_dict()["tree_hash"] == "deadbeef"


def test_from_dict_reads_legacy_tree_hash_key() -> None:
    """Entries written before the key change stay addressable by their key."""
    legacy = {"key": "a" * 40, "branch": "main", "timestamp": "", "metrics": {}}
    snap = Snapshot.from_dict(legacy)
    assert snap.key == "a" * 40
    assert snap.tree_hash == "a" * 40  # recognised as a hash, kept as provenance


def test_from_dict_does_not_mistake_a_tag_for_a_tree_hash() -> None:
    snap = Snapshot.from_dict({"key": "v0.19.0", "branch": "main", "timestamp": "", "metrics": {}})
    assert snap.key == "v0.19.0"
    assert snap.tree_hash == ""


def test_from_dict_reads_legacy_commit_key() -> None:
    snap = Snapshot.from_dict(
        {"commit": "c0ffee", "branch": "main", "timestamp": "", "metrics": {}}
    )
    assert snap.key == "c0ffee"


def test_to_dict_serializes_dataclass_fields() -> None:
    """A subclass storing typed metrics needs no to_dict override."""
    from dataclasses import dataclass

    @dataclass
    class Metrics:
        total_nodes: int
        total_edges: int

    snap = Snapshot(branch="main", timestamp="", metrics=Metrics(3, 4))  # type: ignore[arg-type]
    assert snap.to_dict()["metrics"] == {"total_nodes": 3, "total_edges": 4}


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
        key="hash1",
    )
    assert snap.key == "hash1"
    assert snap.metrics["total_nodes"] == 5

    path = mgr.save_snapshot(snap)
    assert path is not None and path.exists()


def test_capture_relativizes_in_repo_paths(tmp_path: Path) -> None:
    """Snapshots are committed, so absolute in-repo paths must not be stored.

    An absolute ``db_path`` publishes the author's home directory and username
    and makes the snapshot machine-specific: two developers rebuilding the same
    tree would produce a diff recording only where each keeps their checkout.
    """
    repo = tmp_path / "repo"
    mgr = SnapshotManager(repo / ".dockg" / "snapshots", package_name="test-pkg")
    assert mgr.repo_root == repo

    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        tree_hash="hash-rel",
        key="hash-rel",
        graph_stats_dict={
            "total_nodes": 5,
            "total_edges": 3,
            "db_path": str(repo / ".dockg" / "graph.sqlite"),
        },
        repo_root=str(repo),
        nested={"vectors": str(repo / ".dockg" / "vectors.sqlite")},
        outside="/Volumes/corpus/books",
    )

    assert snap.metrics["db_path"] == ".dockg/graph.sqlite"
    assert snap.metrics["repo_root"] == "."
    assert snap.metrics["nested"]["vectors"] == ".dockg/vectors.sqlite"
    # Paths outside the repo are left alone — relativizing them would emit a
    # ../.. chain that leaks more about the machine than the original.
    assert snap.metrics["outside"] == "/Volumes/corpus/books"
    # Non-path values are untouched.
    assert snap.metrics["total_nodes"] == 5


def test_capture_relativizes_under_relative_snapshots_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative ``snapshots_dir`` must still resolve to the repo root.

    ``SnapshotManager(".dockg/snapshots")`` is the form every KG package's own
    docstring demonstrates. Without resolving, the grandparent of a relative
    path is ``.``, nothing is ever inside it, and the rewrite silently does
    nothing — a leak that fails *open*.
    """
    repo = (tmp_path / "repo").resolve()
    (repo / ".dockg").mkdir(parents=True)
    monkeypatch.chdir(repo)

    mgr = SnapshotManager(".dockg/snapshots", package_name="test-pkg")
    assert mgr.repo_root == repo

    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        tree_hash="hash-relative-dir",
        key="hash-relative-dir",
        graph_stats_dict={
            "total_nodes": 5,
            "db_path": str(repo / ".dockg" / "graph.sqlite"),
        },
    )

    assert snap.metrics["db_path"] == ".dockg/graph.sqlite"
    assert str(repo) not in snap.metrics["db_path"]


def test_capture_relativizes_across_a_symlinked_root(tmp_path: Path) -> None:
    """A recorded path and the root may describe one directory via a symlink.

    On macOS a repo under ``/tmp`` really lives at ``/private/tmp``, so a
    literal ``relative_to`` fails even though both name the same place.
    """
    real = (tmp_path / "real").resolve()
    (real / ".dockg" / "snapshots").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)

    mgr = SnapshotManager(link / ".dockg" / "snapshots", package_name="test-pkg")
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        tree_hash="hash-symlink",
        key="hash-symlink",
        graph_stats_dict={
            "total_nodes": 5,
            # Recorded through the *real* path while the manager was built
            # from the symlinked one.
            "db_path": str(real / ".dockg" / "graph.sqlite"),
        },
    )

    assert snap.metrics["db_path"] == ".dockg/graph.sqlite"


def test_subclass_may_reassign_repo_root(tmp_path: Path) -> None:
    """A subclass must be able to set ``self.repo_root`` after ``super().__init__``.

    Regression: 0.13.1 made ``repo_root`` a read-only property, which broke
    every subclass that stores its own — gutenberg_kg's manager does, because
    its corpus root and repo root differ, and construction raised
    ``AttributeError: property 'repo_root' ... has no setter``. That took out
    every ``gutenkg snapshot`` command in a released version.
    """
    repo = (tmp_path / "repo").resolve()
    elsewhere = (tmp_path / "corpus").resolve()
    (repo / ".gutenkg" / "snapshots").mkdir(parents=True)
    elsewhere.mkdir()

    class SubManager(SnapshotManager):
        def __init__(self, snapshots_dir: Path, repo_root: Path) -> None:
            super().__init__(snapshots_dir, package_name="test-pkg")
            self.repo_root = repo_root

    mgr = SubManager(repo / ".gutenkg" / "snapshots", repo_root=elsewhere)
    assert mgr.repo_root == elsewhere

    # Relativization must honour the root the subclass set, not the derived one.
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        tree_hash="hash-subclass",
        key="hash-subclass",
        graph_stats_dict={"total_nodes": 2, "db_path": str(elsewhere / "g.sqlite")},
    )
    assert snap.metrics["db_path"] == "g.sqlite"


def test_relativize_survives_an_unusable_repo_root(tmp_path: Path) -> None:
    """A subclass may set repo_root to None; capture must not raise."""

    class NoRoot(SnapshotManager):
        def __init__(self, snapshots_dir: Path) -> None:
            super().__init__(snapshots_dir, package_name="test-pkg")
            self.repo_root = None  # type: ignore[assignment]

    mgr = NoRoot(tmp_path / ".kg" / "snapshots")
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        tree_hash="hash-noroot",
        key="hash-noroot",
        graph_stats_dict={"total_nodes": 1, "db_path": "/abs/path/g.sqlite"},
    )
    assert snap.metrics["db_path"] == "/abs/path/g.sqlite"


def test_save_rejects_zero_nodes(mgr: SnapshotManager) -> None:
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 0, "total_edges": 0},
        tree_hash="empty",
        key="empty",
    )
    with pytest.raises(ValueError, match="0 nodes"):
        mgr.save_snapshot(snap)


def test_load_snapshot(mgr: SnapshotManager) -> None:
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="loadtest",
        key="loadtest",
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
        key="latest1",
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
            key=h,
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
        key="diff_a",
    )
    s2 = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 15, "total_edges": 8, "node_counts": {"file": 15}},
        tree_hash="diff_b",
        key="diff_b",
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
            key=h,
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
        key="older",
    )
    mgr.save_snapshot(s1, force=True)

    s2 = mgr.capture(
        version="0.1.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 4},
        tree_hash="newer",
        key="newer",
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
            key=h,
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
            key=h,
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
            key=h,
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
        key="hash-a",
    )
    mgr.save_snapshot(snap1)

    snap2 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="hash-b",
        key="hash-b",
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
        key="force-a",
    )
    mgr.save_snapshot(snap1, force=True)

    snap2 = mgr.capture(
        version="1.0.0",
        branch="main",
        graph_stats_dict={"total_nodes": 10, "total_edges": 5},
        tree_hash="force-b",
        key="force-b",
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
            key=h,
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
        key="real-snap",
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
        key="broken-snap",
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


# -- Key scheme --------------------------------------------------------------


def test_capture_defaults_to_a_timestamp_key_not_a_tree_hash(mgr: SnapshotManager) -> None:
    """The tree hash is provenance, never the key.

    It is read before ``git add`` stages the snapshot, so it names a tree that
    is never committed and cannot be resolved afterwards.
    """
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 1, "total_edges": 1},
        tree_hash="b" * 40,
    )
    assert snap.key != "b" * 40
    assert snap.tree_hash == "b" * 40
    datetime.fromisoformat(snap.key)  # a timestamp, and a parseable one


def test_capture_uses_a_supplied_release_key(mgr: SnapshotManager) -> None:
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 1, "total_edges": 1},
        key="v0.19.0",
        subject="repo:kg-utils",
    )
    mgr.save_snapshot(snap)

    assert snap.key == "v0.19.0"
    assert (mgr.snapshots_dir / "v0.19.0.json").exists()
    assert mgr.load_snapshot("v0.19.0") is not None


def test_capture_records_subject_and_tool_separately(mgr: SnapshotManager) -> None:
    """The version field names the measuring tool; subject names what was measured."""
    snap = mgr.capture(
        version="0.1.0",
        branch="test",
        graph_stats_dict={"total_nodes": 1, "total_edges": 1},
        key="v1",
        subject="corpus:pepys",
    )
    mgr.save_snapshot(snap)

    entry = mgr.list_snapshots()[0]
    assert entry["subject"] == "corpus:pepys"
    assert entry["tool"] == mgr.package_name
    assert snap.tool_version == mgr._package_version()


def test_manifest_dual_read_keeps_legacy_entries_addressable(mgr: SnapshotManager) -> None:
    """A tree-hash-keyed manifest written by an older release still loads."""
    key = "c" * 40
    (mgr.snapshots_dir / f"{key}.json").write_text(
        json.dumps(
            {
                "key": key,
                "branch": "main",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "metrics": {"total_nodes": 2, "total_edges": 1},
            }
        ),
        encoding="utf-8",
    )
    mgr.manifest_path.write_text(
        json.dumps(
            {
                "format": "1.0",
                "last_update": "",
                "snapshots": [
                    {
                        "tree_hash": key,
                        "branch": "main",
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "file": f"{key}.json",
                        "metrics": {"total_nodes": 2, "total_edges": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert mgr.load_manifest().snapshots[0]["key"] == key
    assert mgr.load_snapshot(key) is not None
