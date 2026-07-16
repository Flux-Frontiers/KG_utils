# Release Notes — v0.6.1

> Released: 2026-07-15

A point fix for the sqlite-vec backend introduced in 0.6.0: the vector store's sidecar path
was computed from the wrong base, so any caller that passed an explicit `lancedb_dir` — which
is every pycodekg CLI command — wrote the store to one place and looked for it in another.
Building then querying such a knowledge graph failed with `unable to open database file`.
If you shipped a sqlite-vec integration on 0.6.0, upgrade to 0.6.1.

## What changed

**Sidecar path follows the LanceDB directory.** `KGModule.vectors_path` now derives from
`lancedb_dir.parent` (`<kg-dir>/vectors.sqlite`) instead of `repo_root/<_default_dir>`. An
explicit `lancedb_dir` — common when a CLI passes a placeholder `repo_root` — now relocates
the sqlite-vec store with it, matching doc_kg's `sqlite_vectors_path()` convention. The
default-path case is unchanged. Regression tests cover explicit-path construction and the
build-then-fresh-instance query flow that exposed the bug.

## Upgrading

Drop-in for 0.6.0. No API change and no rebuild needed for the default LanceDB backend. If
you built a sqlite-vec store under 0.6.0 with an explicit `lancedb_dir`, its sidecar landed
in the wrong directory — rebuild once on 0.6.1 so the store and its lookup path agree.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
