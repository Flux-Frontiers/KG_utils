# Release Notes — v0.21.0

> Released: 2026-09-10

Two defects that both let disk state go quietly wrong, plus the typed exception
that lets a caller tell one of them apart from a driver failure. Nothing here
changes an API you are already calling; the snapshot fix corrects data that
`save_snapshot` was writing incorrectly, so the value is in upgrading rather
than in adopting anything new.

## What changed

**Snapshot provenance no longer drifts between the manifest and the file.**
`SnapshotManager.save_snapshot()` has a dedup path: when a save matches the
newest entry's version and metrics, it refreshes that entry in place instead of
appending. It rewrote the snapshot file completely but copied only four fields
onto the manifest entry, so `subject`, `tool`, `tool_version` and `version` kept
whatever the previous save had written. Re-saving with a corrected `--subject`
updated the file and left the manifest stale, with a zero exit and nothing
printed. That matters because different commands read different halves:
`snapshot list` and fleet-wide audits read the manifest, `snapshot show` reads
the file. They could disagree indefinitely with nothing surfacing it, and in
`gutenberg_kg` they did. Both paths now build the entry through a single
`_manifest_entry()` method, which also closes the same gap for any metric named
in `metrics_ignore`.

**A missing vector store now says so.** Reading a store that nothing had built
surfaced as `sqlite3.OperationalError("unable to open database file")`, naming
neither the path nor the command that creates it. Where the parent directory
happened to exist it was worse: `sqlite3.connect` created an empty database, so
a mistyped path failed later as `no such table: vec_meta` and left a stray file
behind. The lazy read path now checks first and raises the new
`VectorStoreNotFoundError` with the path in the message. `open()`, which is what
legitimately creates a store, is unchanged.

**`VectorStoreNotFoundError` is a `FileNotFoundError`.** Callers already
catching `OSError` keep working untouched. The reason to subclass rather than
raise a bare `FileNotFoundError` is that a consumer which would rather degrade
than abort — capture the graph metrics, skip the vector-derived ones — can now
catch this one condition instead of pattern-matching driver text. `swift_kg` was
doing exactly that string match, and it is the reason this type exists.

## Upgrading

Nothing to migrate. Existing snapshots are readable as they were, and no
signature changed.

Two things are worth doing after upgrading. If you have ever re-saved a snapshot
with a corrected `--subject`, check that `snapshot list` and `snapshot show`
agree for that key — the fix stops new drift but does not repair an entry
already written. And if your module classifies a missing vector store by
matching on error text, switch to `except VectorStoreNotFoundError`; keep the
substring check only if you must also support a store written before 0.21.0.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
