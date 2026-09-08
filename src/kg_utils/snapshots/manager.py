"""kg_utils/snapshots/manager.py — Snapshot capture, persistence, and comparison.

Usage
-----
>>> from kg_utils.snapshots import SnapshotManager
>>> mgr = SnapshotManager(".codekg/snapshots", package_name="code-kg")
>>> snapshot = mgr.capture(graph_stats_dict=kg.store.stats())
>>> mgr.save_snapshot(snapshot)
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kg_utils.snapshots.models import PruneResult, Snapshot, SnapshotManifest


def _issues_delta(issues_a: list[str], issues_b: list[str]) -> dict[str, list[str]]:
    """Return the issue strings introduced and resolved between two snapshots.

    Order follows the source list rather than set iteration, so the result is
    stable across runs and diffable.

    :param issues_a: Issues of the earlier snapshot.
    :param issues_b: Issues of the later snapshot.
    :return: ``{"introduced": [...], "resolved": [...]}``.
    """
    set_a, set_b = set(issues_a), set(issues_b)
    return {
        "introduced": list(dict.fromkeys(i for i in issues_b if i not in set_a)),
        "resolved": list(dict.fromkeys(i for i in issues_a if i not in set_b)),
    }


class SnapshotManager:
    """Manages snapshot capture, persistence, retrieval, and comparison.

    This is the single shared implementation. Domain-specific KG libraries
    subclass this to override :meth:`_compute_delta` or
    :meth:`_collect_extra_metrics` when they need domain-specific delta fields
    or automatic metric collection from SQLite.

    Four class attributes let a subclass configure the base instead of
    overriding a method to adapt to it. Set them on the subclass; do not
    override :meth:`__init__`, :meth:`diff_snapshots` or
    :meth:`_metrics_changed` to achieve the same effect.

    :cvar package_name: Package name for auto-detecting version
        (e.g. ``"pycode-kg"``, ``"doc-kg"``). Set this on the subclass rather
        than overriding ``__init__`` purely to change the default.
    :cvar dict_metric_deltas: Metric keys whose values are dicts of counts.
        :meth:`diff_snapshots` emits ``"<key>_delta"`` for each, holding only
        the entries whose count changed.
    :cvar metrics_ignore: Metric keys :meth:`_metrics_changed` ignores when
        deciding whether two snapshots differ meaningfully.

    :param snapshots_dir: Directory for snapshot JSON files and manifest.
    :param package_name: Package name for auto-detecting version. Overrides
        the :attr:`package_name` class attribute for this instance; omit it to
        use the class attribute.
    :param db_path: Optional SQLite database path for collecting per-module or
        per-directory node counts via :meth:`_collect_breakdown_counts`.
    """

    #: Default package name for version detection. Subclasses override this
    #: attribute instead of overriding ``__init__`` to change one string.
    package_name: str = "kg-utils"

    #: Metric keys holding a dict of counts, delta'd by :meth:`diff_snapshots`.
    dict_metric_deltas: tuple[str, ...] = ()

    #: Metric keys :meth:`_metrics_changed` ignores. Typically paths and other
    #: environment-dependent values that are not part of the measurement.
    metrics_ignore: frozenset[str] = frozenset()

    #: Deprecated ``capture()`` keywords, mapped to what replaced them. The
    #: target is either a metric name or ``"graph_stats_dict"``.
    #:
    #: Needed because ``capture()`` ends in ``**extra_metrics``, which accepts
    #: any keyword. A module that renames one of its own capture keywords gets
    #: no error from the old name: it lands in the metrics dict under the dead
    #: name and the new one is simply absent. Declaring the rename here keeps
    #: the old keyword working and warns, instead of failing silently.
    capture_aliases: dict[str, str] = {}

    def __init__(
        self,
        snapshots_dir: Path | str,
        *,
        package_name: str | None = None,
        db_path: Path | str | None = None,
    ) -> None:
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.snapshots_dir / "manifest.json"
        # Falls back to the class attribute, so a subclass sets one string
        # rather than carrying an __init__ that only forwards to super().
        self.package_name = package_name if package_name is not None else type(self).package_name
        self.db_path = Path(db_path) if db_path else None
        #: Repository root, inferred as the grandparent of ``snapshots_dir``
        #: — snapshot directories are laid out as ``<repo>/.<kind>kg/snapshots``
        #: across every KG package, so two levels up is the repo.
        #:
        #: Resolved first, because ``snapshots_dir`` is very often *relative*:
        #: ``SnapshotManager(".dockg/snapshots")`` is the form every KG
        #: package's own docstring demonstrates, and the grandparent of a
        #: relative path is ``.``, inside which nothing ever lies.
        #:
        #: A plain attribute rather than a property **on purpose**. Subclasses
        #: assign ``self.repo_root`` after calling ``super().__init__()`` —
        #: gutenberg_kg does, because its corpus root and repo root differ — and
        #: a read-only property makes that assignment raise AttributeError at
        #: construction. Assigning here lets a subclass override it, and
        #: :meth:`_relativize_paths` then relativizes against the root the
        #: subclass actually means.
        self.repo_root = self.snapshots_dir.resolve().parent.parent

    def _relativize_paths(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Rewrite absolute paths under the repo root as repo-relative.

        Snapshots are committed to git, so an absolute value like
        ``/Users/alice/repos/foo/.dockg/graph.sqlite`` in ``db_path`` publishes
        the author's home directory and username, and makes every snapshot
        machine-specific — two developers rebuilding the same tree produce
        diffs that record only where each of them keeps their checkout.

        Only paths *inside* the repo are rewritten. An absolute path elsewhere
        (a corpus on another volume, say) is left alone, because relativizing
        it would produce a ``../..`` chain that says even more about the
        machine than the original did.

        :param metrics: Snapshot metrics, possibly containing path strings.
        :return: The metrics with in-repo absolute paths made relative.
        """
        # Subclasses may reassign repo_root to anything — gutenberg_kg passes
        # one in, and it can be a str or None. Without a usable root there is
        # nothing to relativize against, so leave the metrics untouched rather
        # than raising inside a snapshot capture.
        if not self.repo_root:
            return dict(metrics)
        root = Path(self.repo_root)

        def fix(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("/"):
                candidate = Path(value)
                # Try the literal path first, then its resolved form. The
                # second attempt matters when only one side crosses a symlink
                # — on macOS a repo under /tmp is really /private/tmp, so a
                # recorded path and a resolved root can describe the same
                # directory and still fail a literal comparison.
                for probe in (candidate, candidate.resolve()):
                    try:
                        return probe.relative_to(root).as_posix()
                    except ValueError:
                        continue
                return value
            if isinstance(value, dict):
                return {k: fix(v) for k, v in value.items()}
            if isinstance(value, list):
                return [fix(v) for v in value]
            return value

        return {k: fix(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Package version detection
    # ------------------------------------------------------------------

    def _package_version(self) -> str:
        """Return the installed package version, or ``'unknown'``."""
        try:
            return importlib.metadata.version(self.package_name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    # ------------------------------------------------------------------
    # Capture & save
    # ------------------------------------------------------------------

    def capture(
        self,
        version: str | None = None,
        branch: str | None = None,
        graph_stats_dict: dict[str, Any] | None = None,
        tree_hash: str = "",
        hotspots: list[dict[str, Any]] | None = None,
        issues: list[str] | None = None,
        key: str = "",
        subject: str = "",
        **extra_metrics: Any,
    ) -> Snapshot:
        """Capture a snapshot from current state.

        ``graph_stats_dict`` is merged with ``extra_metrics`` to form the
        snapshot's ``metrics`` dict. Pass domain-specific fields as keyword
        arguments (e.g. ``coverage=0.85``, ``critical_issues=2``).

        :param version: Version string; auto-detected from package if None.
        :param branch: Git branch; auto-detected if None.
        :param graph_stats_dict: Output from the KG's ``stats()`` method.
        :param tree_hash: Git tree hash, recorded as provenance; auto-detected
            if not provided. It is no longer the snapshot's key.
        :param hotspots: Top hotspot entries.
        :param issues: Issue description strings.
        :param key: Snapshot identifier. Pass the release tag for a snapshot of
            a repo at a release; omit it for a corpus, which has no tag, and
            get a UTC timestamp instead. Never defaults to the tree hash: that
            hash is read before ``git add`` stages the snapshot, so it names a
            tree that is never committed and cannot be resolved afterwards.
        :param subject: What was measured, e.g. ``repo:doc-kg`` or
            ``corpus:pepys``. Distinguishes the measured thing from the
            measuring tool, which is recorded separately.
        :param extra_metrics: Additional domain-specific metric fields.
        :return: New :class:`Snapshot` instance (not yet persisted).
        """
        for deprecated, current in self.capture_aliases.items():
            if deprecated not in extra_metrics:
                continue
            value = extra_metrics.pop(deprecated)
            warnings.warn(
                f"{type(self).__name__}.capture({deprecated}=...) is deprecated; "
                f"pass {current}={value!r} instead. The old keyword is accepted "
                f"for now, but it is not a metric name and will be dropped.",
                DeprecationWarning,
                stacklevel=2,
            )
            if current == "graph_stats_dict":
                if graph_stats_dict is None:
                    graph_stats_dict = value
            else:
                extra_metrics.setdefault(current, value)

        if not version:
            version = self._package_version()
        if branch is None:
            branch = self._get_current_branch()
        if not tree_hash:
            tree_hash = self._get_current_tree_hash()
        if not key:
            key = datetime.now(UTC).isoformat()

        metrics: dict[str, Any] = dict(graph_stats_dict or {})
        metrics.update(self._domain_metrics(metrics))
        metrics.update(extra_metrics)
        metrics = self._relativize_paths(metrics)

        snapshot = Snapshot(
            branch=branch,
            timestamp=datetime.now(UTC).isoformat(),
            version=version,
            metrics=metrics,
            hotspots=hotspots or [],
            issues=issues or [],
            tree_hash=tree_hash,
            snapshot_key=key,
            subject=subject,
            tool=self.package_name,
            tool_version=self._package_version(),
        )

        prev = self.get_previous(snapshot.key)
        if prev:
            snapshot.vs_previous = self._compute_delta(snapshot, prev)

        baseline = self.get_baseline()
        if baseline:
            snapshot.vs_baseline = self._compute_delta(snapshot, baseline)

        return snapshot

    def _domain_metrics(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Return metrics the module collects or derives for itself.

        Override this rather than :meth:`capture`. A ``capture()`` override has
        to restate the base signature, and restating it is how an unnamed
        ``key=`` ends up swallowed into ``**extra_metrics`` and never reaches
        the base -- the defect that shipped as a tree-hash snapshot key in four
        packages. Collecting metrics here leaves the signature alone, so that
        cannot happen.

        Two things belong here. Metrics the module queries for itself, such as
        per-module node counts from SQLite; and values derived from *stats*,
        such as a node total that discounts one node kind. Returning a domain
        default is also useful: values returned here are overridden by any
        same-named ``extra_metrics`` keyword, so a default declared here
        survives only when the caller supplies nothing.

        :param stats: The graph stats passed to :meth:`capture`, already
            copied. Mutating it has no effect; return values instead.
        :return: Domain metric fields to merge into the snapshot's metrics.
        """
        return {}

    def save_snapshot(self, snapshot: Snapshot, *, force: bool = False) -> Path | None:
        """Persist a snapshot to disk and update the manifest.

        Rejects snapshots with zero ``total_nodes`` to protect against
        saving degenerate (unbuilt) state.

        If ``version`` and ``metrics`` are unchanged from the latest snapshot,
        the existing entry is refreshed in-place rather than creating a new
        history entry. Pass ``force=True`` to always create a new entry.

        :param snapshot: Snapshot to save.
        :param force: If ``True``, always write a new history entry.
        :return: Path to the saved JSON file, or ``None`` if no-op.
        :raises ValueError: If ``total_nodes`` is 0.
        """
        m = snapshot.metrics
        total_nodes = (
            m.get("total_nodes", 0) if isinstance(m, dict) else getattr(m, "total_nodes", 0)
        )
        if total_nodes == 0:
            raise ValueError(
                "Refusing to save degenerate snapshot with 0 nodes. "
                "Build the KG before capturing a snapshot."
            )

        manifest = self.load_manifest()

        # Dedup: refresh latest entry if nothing meaningful changed.
        if not force and manifest.snapshots:
            latest_entry = max(manifest.snapshots, key=lambda x: x.get("timestamp", ""))
            if snapshot.version == latest_entry.get("version", "") and not self._metrics_changed(
                snapshot.metrics, latest_entry.get("metrics", {})
            ):
                old_key = latest_entry["key"]
                old_file = self.snapshots_dir / latest_entry.get("file", f"{old_key}.json")

                snapshot_file = self.snapshots_dir / f"{snapshot.key}.json"
                snapshot_file.write_text(
                    json.dumps(snapshot.to_dict(), indent=2) + "\n", encoding="utf-8"
                )

                if old_key != snapshot.key and old_file.exists():
                    old_file.unlink()

                latest_entry["key"] = snapshot.key
                latest_entry["branch"] = snapshot.branch
                latest_entry["timestamp"] = snapshot.timestamp
                latest_entry["file"] = snapshot_file.name

                manifest.last_update = datetime.now(UTC).isoformat()
                self._save_manifest(manifest)
                return snapshot_file

        # Normal path: new or changed snapshot.
        snapshot_file = self.snapshots_dir / f"{snapshot.key}.json"
        snapshot_file.write_text(json.dumps(snapshot.to_dict(), indent=2) + "\n", encoding="utf-8")

        existing_idx = next(
            (i for i, s in enumerate(manifest.snapshots) if s.get("key") == snapshot.key),
            None,
        )

        manifest_entry: dict[str, Any] = {
            "key": snapshot.key,
            "branch": snapshot.branch,
            "timestamp": snapshot.timestamp,
            "version": snapshot.version,
            "subject": snapshot.subject,
            "tool": snapshot.tool,
            "tool_version": snapshot.tool_version,
            "file": snapshot_file.name,
            "metrics": snapshot.metrics,
            "deltas": {
                "vs_previous": snapshot.vs_previous,
                "vs_baseline": snapshot.vs_baseline,
            },
        }

        if existing_idx is not None:
            manifest.snapshots[existing_idx] = manifest_entry
        else:
            manifest.snapshots.append(manifest_entry)

        manifest.last_update = datetime.now(UTC).isoformat()
        self._save_manifest(manifest)
        return snapshot_file

    # ------------------------------------------------------------------
    # Loading & listing
    # ------------------------------------------------------------------

    def load_manifest(self) -> SnapshotManifest:
        """Load ``manifest.json``; return empty manifest if absent."""
        if not self.manifest_path.exists():
            return SnapshotManifest()
        manifest = SnapshotManifest.from_dict(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )
        # Dual-read, permanent rather than transitional: tree-hash-keyed
        # entries predate the key change and cannot be re-keyed onto versions,
        # so both shapes stay addressable indefinitely.
        for entry in manifest.snapshots:
            if not entry.get("key"):
                if "tree_hash" in entry:
                    entry["key"] = entry.pop("tree_hash")
                elif "commit" in entry:
                    entry["key"] = entry["commit"]
        return manifest

    def _save_manifest(self, manifest: SnapshotManifest) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    def load_snapshot(self, key: str) -> Snapshot | None:
        """Load a snapshot by key (release tag, timestamp, or legacy tree hash)
        or ``'latest'``.

        Missing ``vs_previous`` / ``vs_baseline`` deltas are backfilled
        on-the-fly from manifest metadata, through
        :meth:`_compute_delta_from_metrics` so that a subclass's domain delta
        fields are present here as they are in :meth:`list_snapshots` and
        :meth:`diff_snapshots`.

        The backfill is not a legacy path. ``capture()`` resolves
        ``vs_previous`` through :meth:`get_previous`, which looks the key up in
        the manifest -- and at capture time the snapshot is not saved yet, so
        the lookup fails and ``vs_previous`` is written as ``null`` for every
        first-time key. ``vs_baseline`` escapes because :meth:`get_baseline`
        does not depend on the unsaved key.
        """
        if key == "latest":
            manifest = self.load_manifest()
            if not manifest.snapshots:
                return None
            entry = max(manifest.snapshots, key=lambda x: x.get("timestamp", ""))
            key = entry["key"]

        snapshot_file = self.snapshots_dir / f"{key}.json"
        if not snapshot_file.exists():
            return None
        snap = Snapshot.from_dict(json.loads(snapshot_file.read_text(encoding="utf-8")))

        # Backfill missing deltas from manifest
        if snap.vs_previous is None or snap.vs_baseline is None:
            manifest = self.load_manifest()
            entries = sorted(manifest.snapshots, key=lambda x: x.get("timestamp", ""), reverse=True)
            idx = next((i for i, s in enumerate(entries) if s.get("key") == key), None)

            if idx is not None:
                # Both branches go through _compute_delta_from_metrics, the
                # documented extension point. Computing the delta inline here
                # instead meant a subclass's domain fields (coverage_delta,
                # kinetic_params_delta, ...) were absent from a backfilled
                # delta and present everywhere else.
                if snap.vs_previous is None and idx + 1 < len(entries):
                    prev_m = entries[idx + 1].get("metrics", {})
                    snap.vs_previous = self._compute_delta_from_metrics(snap.metrics, prev_m)
                if snap.vs_baseline is None and entries and entries[-1].get("key") != key:
                    base_m = entries[-1].get("metrics", {})
                    snap.vs_baseline = self._compute_delta_from_metrics(snap.metrics, base_m)
        return snap

    def get_previous(self, key: str) -> Snapshot | None:
        """Get the snapshot immediately before *key* (by timestamp)."""
        manifest = self.load_manifest()
        current_ts = next(
            (s["timestamp"] for s in manifest.snapshots if s.get("key") == key),
            None,
        )
        if not current_ts:
            return None
        prev_entry = None
        for s in sorted(manifest.snapshots, key=lambda x: x.get("timestamp", ""), reverse=True):
            if s.get("timestamp", "") < current_ts:
                prev_entry = s
                break
        if not prev_entry:
            return None
        prev_key = prev_entry.get("key", "")
        return self.load_snapshot(prev_key) if prev_key else None

    def get_baseline(self) -> Snapshot | None:
        """Get the oldest snapshot (baseline for comparison)."""
        manifest = self.load_manifest()
        if not manifest.snapshots:
            return None
        baseline_entry = min(manifest.snapshots, key=lambda x: x.get("timestamp", ""))
        baseline_key = baseline_entry.get("key", "")
        return self.load_snapshot(baseline_key) if baseline_key else None

    def list_snapshots(
        self,
        limit: int | None = None,
        branch: str | None = None,
    ) -> list[dict[str, Any]]:
        """List snapshots in reverse chronological order.

        :param limit: Max number to return; ``None`` = all.
        :param branch: If provided, filter by branch name.
        :return: List of snapshot metadata dicts.
        """
        manifest = self.load_manifest()
        all_snaps = sorted(manifest.snapshots, key=lambda x: x["timestamp"], reverse=True)

        if branch is not None:
            all_snaps = [s for s in all_snaps if s.get("branch") == branch]

        for i, snap in enumerate(all_snaps):
            if snap.get("deltas", {}).get("vs_previous") is None and i + 1 < len(all_snaps):
                prev = all_snaps[i + 1]
                snap.setdefault("deltas", {})["vs_previous"] = self._compute_delta_from_metrics(
                    snap["metrics"], prev["metrics"]
                )

        return all_snaps[:limit] if limit else all_snaps

    def diff_snapshots(self, key_a: str, key_b: str) -> dict[str, Any]:
        """Compare two snapshots side-by-side.

        :param key_a: First snapshot key.
        :param key_b: Second snapshot key.
        :return: Dict with metrics from both and computed deltas.
        """
        snap_a = self.load_snapshot(key_a)
        snap_b = self.load_snapshot(key_b)

        if not snap_a or not snap_b:
            return {"error": "One or both snapshots not found"}

        all_node_kinds = set(snap_a.metrics.get("node_counts", {})) | set(
            snap_b.metrics.get("node_counts", {})
        )
        all_edge_rels = set(snap_a.metrics.get("edge_counts", {})) | set(
            snap_b.metrics.get("edge_counts", {})
        )

        node_counts_delta = {
            k: snap_b.metrics.get("node_counts", {}).get(k, 0)
            - snap_a.metrics.get("node_counts", {}).get(k, 0)
            for k in all_node_kinds
        }
        edge_counts_delta = {
            k: snap_b.metrics.get("edge_counts", {}).get(k, 0)
            - snap_a.metrics.get("edge_counts", {}).get(k, 0)
            for k in all_edge_rels
        }

        result: dict[str, Any] = {
            "a": {
                "key": snap_a.key,
                "timestamp": snap_a.timestamp,
                "metrics": snap_a.metrics,
                "issues": snap_a.issues,
            },
            "b": {
                "key": snap_b.key,
                "timestamp": snap_b.timestamp,
                "metrics": snap_b.metrics,
                "issues": snap_b.issues,
            },
            "delta": self._compute_delta(snap_b, snap_a),
            "node_counts_delta": node_counts_delta,
            "edge_counts_delta": edge_counts_delta,
            "issues_delta": _issues_delta(snap_a.issues, snap_b.issues),
        }

        for metric in self.dict_metric_deltas:
            counts_a: dict[str, int] = snap_a.metrics.get(metric, {}) or {}
            counts_b: dict[str, int] = snap_b.metrics.get(metric, {}) or {}
            result[f"{metric}_delta"] = {
                k: counts_b.get(k, 0) - counts_a.get(k, 0)
                for k in list(counts_a) + [k for k in counts_b if k not in counts_a]
                if counts_b.get(k, 0) != counts_a.get(k, 0)
            }

        return result

    # ------------------------------------------------------------------
    # Delta computation — override for domain-specific delta fields
    # ------------------------------------------------------------------

    def _metrics_changed(self, new_metrics: dict[str, Any], old_metrics: dict[str, Any]) -> bool:
        """Return ``True`` if metrics represent a meaningful change.

        Keys named in the :attr:`metrics_ignore` class attribute are excluded
        from the comparison. Set that attribute rather than overriding this.

        :param new_metrics: Metrics of the snapshot being saved.
        :param old_metrics: Metrics of the snapshot it would replace.
        :return: ``True`` if the two differ outside :attr:`metrics_ignore`.
        """
        if not self.metrics_ignore:
            return new_metrics != old_metrics
        ignore = self.metrics_ignore
        return {k: v for k, v in new_metrics.items() if k not in ignore} != {
            k: v for k, v in old_metrics.items() if k not in ignore
        }

    def _compute_delta(self, snap_new: Snapshot, snap_old: Snapshot) -> dict[str, Any]:
        """Compute metrics delta (new - old).

        Override in subclasses to add domain-specific delta fields.
        """

        def _to_dict(m: Any) -> dict[str, Any]:
            if isinstance(m, dict):
                return m
            if dataclasses.is_dataclass(m) and not isinstance(m, type):
                return dataclasses.asdict(m)
            return {}

        return self._compute_delta_from_metrics(
            _to_dict(snap_new.metrics), _to_dict(snap_old.metrics)
        )

    def _compute_delta_from_metrics(
        self, new_m: dict[str, Any], old_m: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute delta from two raw metrics dicts.

        Override in subclasses to add domain-specific delta fields.
        """
        return {
            "nodes": new_m.get("total_nodes", 0) - old_m.get("total_nodes", 0),
            "edges": new_m.get("total_edges", 0) - old_m.get("total_edges", 0),
        }

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------

    def prune_snapshots(self, *, dry_run: bool = False) -> PruneResult:
        """Remove vestigial snapshots that carry no new metric information.

        :param dry_run: If ``True``, compute what would be removed without deleting.
        :return: :class:`PruneResult` summarising the cleanup.
        """
        manifest = self.load_manifest()
        by_time = sorted(manifest.snapshots, key=lambda x: x.get("timestamp", ""))

        removed_keys: list[str] = []
        broken_keys: list[str] = []
        orphaned_files: list[str] = []

        # Pass 1: separate valid entries from broken ones.
        valid: list[dict[str, Any]] = []
        for entry in by_time:
            key = entry.get("key", "")
            fname = entry.get("file", f"{key}.json")
            if not (self.snapshots_dir / fname).exists():
                broken_keys.append(key)
            else:
                valid.append(entry)

        # Pass 2: flag metric-duplicate interior entries.
        if len(valid) > 2:
            kept_metrics = valid[0].get("metrics", {})
            for entry in valid[1:-1]:
                m = entry.get("metrics", {})
                if not self._metrics_changed(m, kept_metrics):
                    removed_keys.append(entry.get("key", ""))
                else:
                    kept_metrics = m

        # Pass 3: find orphaned JSON files.
        referenced_files = {e.get("file", f"{e.get('key', '')}.json") for e in manifest.snapshots}
        for path in self.snapshots_dir.glob("*.json"):
            if path.name == "manifest.json":
                continue
            if path.name not in referenced_files:
                orphaned_files.append(path.name)

        if not dry_run:
            entry_by_key = {e.get("key"): e for e in manifest.snapshots}

            for key in removed_keys:
                entry = entry_by_key.get(key, {})
                fname = entry.get("file", f"{key}.json")
                p = self.snapshots_dir / fname
                if p.exists():
                    p.unlink()

            for fname in orphaned_files:
                p = self.snapshots_dir / fname
                if p.exists():
                    p.unlink()

            drop_keys = set(removed_keys) | set(broken_keys)
            manifest.snapshots = [e for e in manifest.snapshots if e.get("key") not in drop_keys]
            manifest.last_update = datetime.now(UTC).isoformat()
            self._save_manifest(manifest)

        return PruneResult(
            removed=removed_keys,
            orphaned_files=orphaned_files,
            broken_entries=broken_keys,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_current_tree_hash() -> str:
        """Get current git tree hash (HEAD^{tree})."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    @staticmethod
    def _get_current_branch() -> str:
        """Get current git branch name."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
