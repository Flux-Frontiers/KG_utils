# Release Notes — v0.8.0

> Released: 2026-07-26

The default vector backend flips from LanceDB to sqlite-vec. `KGModule` now defaults to
`vector_backend="auto"`, which builds a fresh or already-migrated knowledge graph on the exact
(recall 1.0) sqlite-vec store and only falls back to LanceDB when an un-migrated LanceDB store
is already present on disk. Existing corpora keep working untouched; new ones get the smaller,
exact backend by default.

## What changed

**sqlite-vec is the default.** `"auto"` resolves to sqlite-vec for a fresh KG or one that has
already been migrated (a `vectors.sqlite` sidecar exists), and to LanceDB only when an
un-migrated LanceDB store is found on disk — so no existing store is stranded and nothing
needs a manual migration step. Pass `vector_backend="lancedb"` to pin the old behaviour, or
`"sqlite-vec"` to force the new store regardless of what is on disk. The `sqlite-vec`
dependency is now bundled in the `semantic` extra, so the default works out of the box; its
prebuilt wheels cover macOS (x86_64 + arm64), Linux (x86_64 + aarch64), and Windows.

**README back in sync with the package.** The 0.7.0 modules `kg_utils.viz` and
`kg_utils.analysis` are now documented (features, install snippet, API tables), the source
tree lists the previously-undocumented modules (`vector_backend`, `corpus_embedder`,
`retrieval/`, `worker/`, `synthesis/factory`), and the prose reflects sqlite-vec as the
default backend with LanceDB as the legacy option.

## Upgrading

No action required. Existing LanceDB-backed KGs are detected by `"auto"` and keep working as
before. To move a corpus onto sqlite-vec, rebuild it (the sidecar is written next to the
LanceDB directory); once `vectors.sqlite` exists, `"auto"` uses it. If you construct
`KGModule` and depend on LanceDB being the default, set `vector_backend="lancedb"` explicitly.

Installing the `semantic` extra now also installs `sqlite-vec` — no separate
`pip install 'kgmodule-utils[sqlite-vec]'` is needed for the default backend.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
