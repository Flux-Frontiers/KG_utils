# Release Notes — v0.19.1

> Released: 2026-09-06

A snapshot loaded from disk reported a different metrics delta than the same
snapshot listed or diffed. This release makes all four paths agree. Nothing on
disk changes, no migration is needed, and consumers pick the fix up by reading
a snapshot back.

## What changed

**A backfilled delta kept only nodes and edges.** `SnapshotManager` computes a
metrics delta in four places. `capture()`, `list_snapshots()` and
`diff_snapshots()` all call `_compute_delta_from_metrics`, the extension point
every KG module overrides to add its own fields. The backfill inside
`load_snapshot()` built a two-key dict inline instead, so a module's domain
fields were absent there and present everywhere else. Both branches of the
backfill now go through the extension point.

The practical effect is that `snapshot show` was the command reporting the
wrong numbers. DocKG regains `coverage_delta` and `issues_delta`, PyCodeKG
`coverage_delta` and `critical_issues_delta`, MetaboKG `kinetic_params_delta`
and `pathway_delta`, FTreeKG `files_delta` and `dirs_delta`, DiaryKG its chunk
and entry deltas.

**Why the backfill runs at all.** `capture()` resolves the previous snapshot
through `get_previous()`, which looks the key up in the manifest. At capture
time the snapshot has not been saved, so the lookup fails and `vs_previous` is
written as null for every first-time key. `vs_baseline` escapes because
`get_baseline()` does not depend on the unsaved key, which is why a release
snapshot could print a correct baseline delta beside a zeroed previous delta
for the same pair. That asymmetry is now covered by a test that states it is a
known gap rather than a contract.

## Upgrading

Nothing to do. Every module already floors at `>=0.19.0` and picks this up on
its next dependency resolve. Snapshot files and manifests are untouched, and no
public signature changed.

If you maintain a `SnapshotManager` subclass, the one thing worth knowing is
that a delta read back from `load_snapshot()` now carries your domain fields.
Code that worked around their absence can drop the workaround.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
