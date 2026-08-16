# Release Notes — v0.14.0

> Released: 2026-08-16

Two things at once, both of them groundwork. The SDK finally takes LanceDB's
name off a store LanceDB never touches — a breaking rename of `lancedb_dir` to
`vectors_path` — and `viz3d` grows the geometry accessors that let a renderer
other than PyVista draw an organic tree. The second is what the POV-Ray quilt
pipeline is built on.

## What changed

**`lancedb_dir` is now `vectors_path`, and passing the old name raises
`TypeError`.** The parameter named for the retired backend was load-bearing for
the current one: `KGModule` derived its sqlite-vec store as
`self.lancedb_dir.parent / "vectors.sqlite"`, so every downstream signature
carried LanceDB's name for a store LanceDB has nothing to do with. `pycode_kg`
has zero `lancedb` occurrences in its own source and still could not shed the
name, because `SemanticIndex` handed it back. Three classes change —
`pipeline.KGModule`, `semantic.SemanticIndex`, `module.KGModule` — and all three
now take a *file* path where they took a *directory*. There is no deprecation
period, matching how memory-kg 0.7.0, Metabo_kg 0.10.0 and diary-kg 0.95.0 each
dropped this same parameter; a hard break is what surfaced stale call sites in
those repos instead of letting them fail later somewhere less obvious.
`vector_backend="auto"` still finds an un-migrated LanceDB store, but the probe
path is now derived privately rather than being a constructor argument —
exposing it is how the name propagated in the first place.

**`SemanticIndex` no longer quietly builds a LanceDB store.** Found while doing
the rename: with no explicit `backend=`, `_get_backend()` defaulted to
`LanceDBBackend`, so bare use of the SDK's index created a LanceDB store years
after the fleet migrated off it. `KGModule` always passes a backend, so no fleet
path hit it. It now builds a `SqliteVecBackend`. Reading a legacy store means
constructing `LanceDBBackend` explicitly, which is what `dockg convert-index`
already does.

**The NumPy halves of the mesh builders are now reachable.** `smooth_paths` and
`leaf_glyphs` each did two things — place geometry, then hand it to PyVista —
and the placement is pure NumPy while the PyVista part is a detail of one
renderer. A consumer describing a limb analytically, as a POV-Ray
`sphere_sweep` does, had no way to reach that geometry without dragging in VTK
to build a tube it would throw away. `leaf_frames` returns clung leaf positions
and aim vectors, `limb_paths` is the PyVista-free counterpart of `smooth_paths`,
and `LEAF_ASPECT` is the scale that flattens the leaf prototype into a blade.
`leaf_glyphs` now calls `leaf_frames` rather than keeping a second copy of the
clinging rule. All of it stays inside the NumPy-only `viz3d` extra, with no new
dependency. This is what lets `quiltwright.povgen` emit a tree as analytic
primitives: 839 KB of SDL for a 3000-leaf tree against roughly 12.5 MB for the
equivalent triangle dump.

**One camera rule for a grown tree, replacing three copies.** `frame_tree` and
`CameraFrame` consolidate a rule that existed line-for-line in two `cmd_quilt`
modules plus a NumPy re-derivation in `gutenberg_kg.povscene`, with a fourth
copy about to appear the moment `pycode_kg` grew POV-Ray output. The subtlety
worth knowing about is `fov`: the old standoff rule is correct where it lived,
because PyVista's `reset_camera()` re-fits afterwards, but POV-Ray has no such
pass — so hoisting the rule as-is silently dropped the fitting and the first
real render came out cropped top and bottom while every unit test still passed,
a badly-fitted frame being a structurally valid one. Passing `fov` fits the
bounding sphere to the lens; omitting it keeps the old behaviour, so PyVista
callers are unaffected. `leaf_facing` and `oriented_cluster` are promoted from
the same two consumers, verified equivalent over 500 randomized cases — with one
real divergence, an empty-cluster `ValueError` in the `gutenberg_kg` copy that
the promoted version's guard removes.

**Two import gates were costing most of this package's test coverage.**
`tests/test_vector_backend.py` opened with `importorskip("lancedb")`, which CI
never installs, so the whole file skipped and the storage seam under every
consumer ran at 45% with the *default* backend untested. The viz3d export tests
had the same shape and a sharper irony: a module-scope `importorskip("pyvista")`
guarding a file whose stated purpose is pinning that the accessors work
*without* PyVista. Both are fixed, CI now installs `viz3d-render`, and the suite
went from 81% to 85% with 56 tests that had never run in CI.

## Upgrading

Rename the keyword and hand it a file path rather than a directory:

```python
kg = PyCodeKG(repo, db_path, vectors_path=root / "vectors.sqlite")
```

If you never passed `lancedb_dir`, there is nothing to do. The derived default
is unchanged — it produced `<kg-dir>/vectors.sqlite` before and does now — so
`PyCodeKG`, `FileTreeKG` and `TypeScriptKG` resolve identical paths and no
rebuild is needed. Despite being a breaking signature change, this is
source-compatible for every current subclass in the fleet.

What does need updating is code that reads the name back rather than passing it:
`SemanticIndex.build()` returns `vectors_path` in its stats dict where it
returned `lancedb_dir`, and `repr()` changes to match. If you construct
`SemanticIndex` directly without an explicit `backend=`, note that you now get
sqlite-vec instead of LanceDB; pass `LanceDBBackend` explicitly to read a
pre-migration store.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
