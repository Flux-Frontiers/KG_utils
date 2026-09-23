# Release Notes -- v0.24.0

> Released: 2026-09-22

### Fixed

- **A wiped graph no longer leaves a stale vector index behind** (`kgrag_priv`
  sweep item 54). `KGModule.build_graph(wipe=True)` rewrote the graph and
  left the previous build's vectors in place, so a graph-only rebuild
  followed by `query()` seeded from nodes the new graph might no longer hold.
  It now drops the index first; `build()` rebuilds it straight after, and a
  graph-only build leaves no index, so `query()` raises
  `VectorStoreNotFoundError` rather than answering from the old one.
  `vault_kg` deleted the stale index in its own CLI and `connectome_kg`
  warned about it; both workarounds can go once they raise their floor.

### Added

- **`KGModule.drop_index()`** deletes the vector index and returns the paths
  removed: the sqlite-vec store with its `-wal`/`-shm`/`-journal` sidecars,
  and any legacy LanceDB directory beside it, which the `"auto"` backend
  would otherwise fall back to. The index object is closed but kept, so a
  caller-supplied index survives. `SemanticIndex.close()` closes the
  backend's connection when it has one.
- **`build_graph_html(..., edge_labels=False)`** leaves each edge's
  relation on hover only instead of printing it on the canvas, which keeps a
  dense neighbourhood readable; the edge colour still carries the relation.
  The default is unchanged (`kgrag_priv` sweep item 55).
- **Per-leaf sizes in `leaf_glyphs()`**: `size` takes an `(M,)` array as
  well as a scalar, so leaf size can carry data (backlinks, citations) the
  way `tint` carries colour, still in one glyph call. `leaf_frames()` accepts
  the same array for the clearance a clung leaf keeps from the wood. Sizes
  must be positive and one per leaf. A scalar draws exactly what it did
  before (sweep item 55).

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
