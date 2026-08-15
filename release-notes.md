# Release Notes — v0.13.1

> Released: 2026-08-15

A single fix, in the one place every KG package's snapshots pass through: snapshots
no longer record absolute filesystem paths. Until now a captured snapshot stored
its database location verbatim — `/Users/<name>/repos/<project>/.dockg/graph.sqlite`
— and snapshots are committed to git. That put a home directory and a username into
version control across the ecosystem, and made every snapshot specific to the
machine that produced it.

## What changed

**Paths inside the repository are now stored relative to it.** `db_path` reads
`.dockg/graph.sqlite` instead of the full absolute path. The rewrite happens in
`SnapshotManager.capture()`, which every KG package reaches — either by inheriting
it or by delegating through `super().capture()` — so one change covers the family.
It walks nested dictionaries and lists, so a path buried inside a structured metric
is caught alongside a top-level one.

**Paths outside the repository are deliberately untouched.** A corpus on another
volume stays absolute, because rewriting it would produce a `../..` chain that
describes the machine more thoroughly than the original string did. The rewrite
only applies where the result is genuinely portable.

**The repository root is resolved before anything is compared.** This is the part
that makes the rest work. `SnapshotManager(".dockg/snapshots")` — the relative form
shown in every KG package's own documentation — has `.` as its grandparent, and no
path is ever inside `.`. Without resolving first, the rewrite would have quietly
done nothing while appearing to succeed. Comparison also retries against the
resolved candidate, so a repository reached through a symbolic link still matches;
on macOS a checkout under `/tmp` actually lives at `/private/tmp`, and both
spellings name one directory.

The public surface gains one property, `SnapshotManager.repo_root`. Nothing was
removed or renamed, and no existing signature changed.

## Upgrading

Upgrade and rebuild; there is nothing to migrate and no new configuration. From the
next snapshot each project captures, its paths will be recorded relative to the
repository.

This release does not rewrite snapshots that are already committed. Those keep the
absolute paths they were written with until each project rewrites them once — a
mechanical edit of the `db_path` field in `.<kind>kg/snapshots/*.json` and the
accompanying `manifest.json`. Projects that upgrade without doing that stop
accumulating new occurrences but keep the existing ones in their history.

One cosmetic consequence: because deduplication compares the metrics dictionary,
the first capture after upgrading will see `db_path` change format and record a
snapshot even when nothing else moved. It happens once per project.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
