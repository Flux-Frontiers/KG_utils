# Release Notes — v0.13.2

> Released: 2026-08-15

A regression fix. Version 0.13.1 introduced `SnapshotManager.repo_root` as a
read-only property, and a read-only property cannot be assigned. Any subclass
that stores its own repository root — the natural thing to do when a package's
data root and its repository root are not the same directory — raised
`AttributeError: property 'repo_root' has no setter` the moment it was
constructed. In `gutenberg_kg`, whose corpus lives outside the repository and
which therefore does exactly that, this took out every `gutenkg snapshot`
command in a released version.

## What changed

**`repo_root` is a plain attribute again, assigned in `__init__`.** It is
computed exactly as before — the resolved grandparent of `snapshots_dir` — so
the default behaviour and everything 0.13.1 fixed are unchanged. The difference
is only that a subclass can now overwrite it after calling `super().__init__()`,
and `_relativize_paths()` then relativizes against the root the subclass
actually means rather than the one inferred from the directory layout.

**`_relativize_paths()` tolerates a root it cannot use.** Because subclasses may
now assign anything, a root that is `None` or empty makes the metrics pass
through untouched instead of raising part-way into a snapshot capture. A failed
relativization should never be able to fail a capture.

Two regression tests cover both paths: a subclass reassigning `repo_root` and
having its choice honoured during relativization, and a subclass setting it to
`None` without breaking capture.

## Why it escaped

The downstream test suites already caught this — `gutenberg_kg` fails 29 tests
against 0.13.1. They were not run when the SDK changed. This release was
verified the other way round: every snapshot suite across the eight KG packages
that subclass `SnapshotManager` was run against the patched source, along with
a live capture through `gutenberg_kg`'s own manager.

## Upgrading

Upgrade in place; nothing to migrate and no configuration to change. Anyone on
0.13.1 with a `SnapshotManager` subclass should upgrade — on 0.13.1 that
subclass cannot be constructed at all.

```bash
pip install --upgrade kgmodule-utils
```
