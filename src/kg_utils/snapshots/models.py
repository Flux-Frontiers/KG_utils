"""kg_utils/snapshots/models.py — Snapshot data models.

A snapshot is identified by a caller-supplied key and contains:
  - Timestamp and branch metadata
  - Metrics dict (domain-flexible: total_nodes, total_edges, node_counts, ...)
  - Hotspots list and issues list
  - Deltas vs. previous and baseline snapshots
  - Provenance: what was measured (``subject``) and what measured it
    (``tool`` / ``tool_version``)

The key used to be the git tree hash. That could not work: the hash is taken
before ``git add`` stages the snapshot, so it names a tree that never gets
committed. Tree hashes are still recorded as a field -- real provenance, just
not an identifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

_TREE_HASH_CHARS = set("0123456789abcdef")


def _is_tree_hash(key: str) -> bool:
    """Return ``True`` if *key* has the shape of a git object hash.

    Used only to decide whether a legacy key can also be recorded as
    ``tree_hash`` provenance. A version tag or timestamp key cannot.

    :param key: The stored snapshot key.
    :return: ``True`` for a 40-character lowercase hex string.
    """
    return len(key) == 40 and set(key) <= _TREE_HASH_CHARS


def _jsonable(value: Any) -> Any:
    """Return *value* in a JSON-serializable form.

    Subclasses in the KG modules store ``metrics`` and the delta fields as
    typed dataclasses rather than plain dicts. Converting here is what lets
    those subclasses use this base :meth:`Snapshot.to_dict` instead of
    overriding it.

    :param value: A dataclass instance, or any already-serializable value.
    :return: ``asdict(value)`` for a dataclass instance, otherwise *value*.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


@dataclass
class Snapshot:
    """A temporal snapshot of KG metrics.

    ``metrics`` is a free-form dict so that each domain can store whatever
    fields it needs (docstring_coverage, total_files, etc.) without requiring
    changes to this shared data model.  The only required keys are
    ``total_nodes`` and ``total_edges`` -- the manager uses these for delta
    computation.

    ``vs_previous`` and ``vs_baseline`` are also free-form dicts so that
    domain-specific delta fields (coverage_delta, files_delta, ...) can be
    stored alongside the universal ``nodes`` and ``edges`` deltas.
    """

    branch: str
    timestamp: str  # ISO 8601 UTC
    metrics: dict[str, Any]
    version: str = ""
    hotspots: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    vs_previous: dict[str, Any] | None = None
    vs_baseline: dict[str, Any] | None = None
    tree_hash: str = ""
    snapshot_key: str = ""
    subject: str = ""
    tool: str = ""
    tool_version: str = ""

    @property
    def key(self) -> str:
        """Identifier: the supplied key, falling back to the tree hash.

        The fallback exists so snapshots written before keys were supplied
        stay addressable by the key they were stored under.
        """
        return self.snapshot_key or self.tree_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        Reads the three structured fields out of ``__dict__`` rather than
        through attribute access, so a subclass that exposes them as typed
        properties serializes correctly without overriding this method.
        """
        d = self.__dict__
        return {
            "key": self.key,
            "branch": self.branch,
            "timestamp": self.timestamp,
            "version": self.version,
            "subject": self.subject,
            "tool": self.tool,
            "tool_version": self.tool_version,
            "tree_hash": self.tree_hash,
            "metrics": _jsonable(d.get("metrics")),
            "hotspots": self.hotspots,
            "issues": self.issues,
            "vs_previous": _jsonable(d.get("vs_previous")),
            "vs_baseline": _jsonable(d.get("vs_baseline")),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Snapshot:
        """Reconstruct a Snapshot from a dictionary loaded from JSON."""
        raw = dict(data)  # shallow copy to avoid mutating caller's data

        metrics = raw.pop("metrics", {})
        vs_prev = raw.pop("vs_previous", None)
        vs_base = raw.pop("vs_baseline", None)

        # Dual-read. Three key shapes have been written over this schema's
        # life: 'commit', 'tree_hash', and 'key'. All stay addressable.
        tree_hash = raw.pop("tree_hash", "")
        key = raw.pop("key", "") or tree_hash or raw.pop("commit", "")
        raw.pop("commit", None)  # drop legacy field if still present
        if not tree_hash and _is_tree_hash(key):
            tree_hash = key
        raw.setdefault("version", "")

        return Snapshot(
            snapshot_key=key,
            tree_hash=tree_hash,
            metrics=metrics,
            vs_previous=vs_prev,
            vs_baseline=vs_base,
            branch=raw.pop("branch", ""),
            timestamp=raw.pop("timestamp", ""),
            version=raw.pop("version", ""),
            subject=raw.pop("subject", ""),
            tool=raw.pop("tool", ""),
            tool_version=raw.pop("tool_version", ""),
            hotspots=raw.pop("hotspots", []),
            issues=raw.pop("issues", []),
        )


@dataclass
class PruneResult:
    """Summary of a :meth:`SnapshotManager.prune_snapshots` operation.

    :param removed: Keys of snapshots pruned as metric-duplicates.
    :param orphaned_files: Filenames of JSON files deleted from disk because
        they were not referenced by the manifest.
    :param broken_entries: Keys of manifest entries whose JSON file was missing.
    :param dry_run: ``True`` when the call was a dry run (nothing deleted).
    """

    removed: list[str]
    orphaned_files: list[str]
    broken_entries: list[str]
    dry_run: bool

    @property
    def total_cleaned(self) -> int:
        """Total number of items removed (or that *would* be removed in dry-run)."""
        return len(self.removed) + len(self.orphaned_files) + len(self.broken_entries)


@dataclass
class SnapshotManifest:
    """Index of all snapshots, with fast lookup by tree hash."""

    format_version: str = "1.0"
    last_update: str = ""
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "format": self.format_version,
            "last_update": self.last_update,
            "snapshots": self.snapshots,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SnapshotManifest:
        """Reconstruct from dict."""
        return SnapshotManifest(
            format_version=data.get("format", "1.0"),
            last_update=data.get("last_update", ""),
            snapshots=data.get("snapshots", []),
        )
