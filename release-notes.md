# Release Notes -- v0.28.0

> Released: 2026-10-23

### Added

- **`kg_utils.devicepack`: on-device search packs.** The format the
  Knowledge Press apps read, lifted out of gutenberg_kg's `export-swift` so
  any KG can write it: a SQLite passage pack with FTS5, an int8 or float32
  vector sidecar with a 32-byte header, a manifest naming the embedder, and
  golden reference rankings. `write_passage_pack()` writes a pack from
  `Passage` rows and a vector map; `search_pack()` is the hybrid reference
  search (dense plus BM25, fused by reciprocal rank) the app must reproduce;
  `build_manifest()` and `build_golden()` write the other two files. Search
  filters on `genre` only, as the Swift engine does. Cross-checked against
  gutenberg_kg's real packs: all 24 golden rankings identical, scores
  included. gutenberg_kg is unchanged. Needs NumPy for vectors.
- **`SqliteVecBackend.iter_vectors()`** yields every stored `(id, vector)`
  as float32 (an int8 store is divided back by 127), so an exporter can copy
  a store's vectors into a device pack without reading `vec0` itself.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
