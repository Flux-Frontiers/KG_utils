# Release Notes — v0.6.0

> Released: 2026-07-15

This release finishes wiring the pluggable vector store introduced in 0.5.0 into the part of
the SDK every domain module actually uses. In 0.5.0 the `VectorBackend` seam existed on
`SemanticIndex`, but only doc_kg's heavier subclass could reach it — the fleet-wide
`KGModule` base still hardcoded LanceDB. Now any KG module can choose its vector store with a
single constructor argument, and the exact `sqlite-vec` backend becomes a first-class option
across the fleet without a line of plumbing per package.

## What changed

**Backend selection on `KGModule`.** The base class gains a `vector_backend` setting that
threads straight through to the `SemanticIndex` it builds. Pass `"lancedb"` (the default —
existing consumers are untouched), `"sqlite-vec"` for the exact, disk-light store, or
`"auto"` to let the module decide: sqlite-vec for a fresh knowledge graph, and lancedb only
when an un-migrated LanceDB store already exists on disk, so established corpora keep working
without a migration. sqlite-vec vectors live in a `vectors.sqlite` sidecar beside the
LanceDB directory.

**Introspection without loading a model.** `KGModule.stats()` now reports the resolved
`vector_backend` name. Resolution is purely path-based — it inspects what is on disk and
never loads the embedding model — so it is cheap to call anywhere.

**Reusable selection helpers.** The new `resolve_backend_name()` and `make_backend()`
factories in `kg_utils.vector_backend` back the selection logic and are usable on their own,
outside the pipeline. `kg_utils.semantic.META_COLUMNS` is now a public alias for domain
packages that construct backends directly.

## Upgrading

Nothing required. The default backend stays `lancedb` and existing behavior is unchanged. To
adopt the exact store, add the optional dependency (`pip install 'kgmodule-utils[sqlite-vec]'`)
and construct your module with `vector_backend="sqlite-vec"` (or `"auto"` once your package
declares the extra). A knowledge graph built under one backend is not read by the other —
rebuild when you switch.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
