# Release Notes — v0.25.0

> Released: 2026-09-26

### Added

- **Tree species for organic trees: `kg_utils.viz3d.species`.** Space
  colonization follows its attractors, so trees with the same crown envelope
  and growth parameters all came out the same shape. A `Habit` is what a
  species adds on top of the data: a crown envelope (`dome`, `ellipsoid`,
  `ovoid`, `cone`, `vase`, `umbrella`, `spindle`, and the old `column`),
  width, clear bole, whorls and cluster spread/lift, placed by
  `crown_sections()` and `section_cluster()`; and how the wood grows toward
  it — tropism, influence radius, internode step, jitter, pipe exponent, a
  plumb trunk with an optional central leader, and a gravity droop.
  `SPECIES` holds nine tuned presets (oak, chestnut, fir, plane, blackthorn,
  pine, birch, willow, poplar), `vary_habit()` nudges one per tree by a
  seeded few percent, and `species_table()` gives the plain numbers for a
  mirror such as the Knowledge Press web forest. The data still sets the
  height, the sections and one crown point per chunk.
- **`grow_tree(..., habit=)`** grows a tree as a species, bends it with the
  habit's droop and records the crown the leaves should hang on as
  `Skeleton.crown` (droop carries each chunk with its twig). `habit=None`,
  the default, grows exactly as before.
- **`droop_skeleton()`** bends thin wood toward the ground, each segment
  inheriting its parent's turn and keeping its length, and moves any given
  points with the node they hang on.
- **`colonize()` options** `step_scale`, `influence_steps`, `plumb_trunk` and
  `leader`. Defaults are unchanged; `plumb_trunk` raises the trunk straight
  to the crown's base instead of leaning it toward the nearest chunk.
- **Textured bark: `bark_sweep()` and `bark_mesh()`** (sweep item 64,
  layer 1). `tree_mesh()` sweeps with `tube()`, which writes no texture
  coordinates, so no bark image could be applied. `bark_sweep()` sweeps the
  skeleton into one continuous tube per chain (following the thickest child,
  so no wood is drawn twice), with parallel-transported rings, `u` wrapping a
  whole number of tiles around each tube and `v` keeping the bark image's
  aspect -- the Knowledge Press web forest's `emitBark`, after ez-tree (MIT).
  NumPy only, returning a `BarkSweep` of points, normals, UVs and triangles;
  `bark_mesh()` wraps it as a `pv.PolyData` with active texture coordinates.
  `tree_mesh()` is unchanged.

### Changed

- **`bounded_int()` and `require_query()` are strict about type** (sweep item
  52). Their arguments arrive as JSON from a model or a form, where loose
  types are normal: `bounded_int` now rejects a string (`"8"`), a bool
  (`True` read as `1`), a truncating float (`3.7`), `None`, NaN and infinity
  with "`k` must be an integer", and returns an integral float or NumPy
  integer as a plain `int`; `require_query` rejects a non-string with
  "q must be a string". Rejecting input that used to pass is a behaviour
  change. `connectome_kg`'s copy was the reference; its copy and
  `swift_kg`'s (a superset of the old SDK check) and `genealogy_kg`'s can
  now be deleted in favour of these, with each surface's tighter query length
  set through `max_query_len`.
- **`quiltwright` relocked at 0.15.1** (sweep item 46).
- **`frame_tree(fov=...)` fits the points' own bounding sphere** — the
  farthest point from the frame's centre, root included — instead of the
  bounding box's half-diagonal. A rounded crown (a dome, an ellipsoid) never
  reaches its box's corners, so the old fit stood the camera back and left
  the tree small in a POV-Ray render; every point still fits. The
  ``fov=None`` standoff rule is unchanged.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
