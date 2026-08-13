# Release Notes — v0.12.1

> Released: 2026-08-13

This release ships the `viz3d` organic tree engine that 0.12.0 described but did not
contain. The promotion sat on a branch the `v0.12.0` tag never included, so that wheel
carried a `viz3d` package with two modules where the changelog claimed three, and every
symbol the notes advertised raised `ImportError` on import. 0.12.1 is the corrected
artifact — 0.12.0 cannot be fixed in place, because PyPI reserves an uploaded filename
permanently and will not accept a replacement even after the original is deleted.

## What changed

**A botanically credible tree engine, no longer trapped in a book corpus.** `kg_utils.viz3d.organic`
arrives verbatim from `gutenberg_kg.layout_organic`: space colonization after Runions,
Lane and Prusinkiewicz, the pipe model for branch radii (da Vinci's rule), root-to-tip
path extraction, mesh and leaf-glyph builders, crown spacing, and `grow_tree` as the
one-call entry point. This finishes what 0.11.0 began. That release moved the layouts out
of `pycode_kg` so that drawing a graph no longer obliged you to install a Python
source-code analyser; the tree engine was stranded in a Gutenberg corpus for exactly the
same reason, and is now free of it.

The engine takes crown attractors and a root, and knows nothing about what they mean. A
document corpus grows document → section → chunk; a diary grows trunk → period limb →
entry cluster → leaves. The hierarchy is the caller's business. That is also why
`seed_from_slug` became **`seed_from_key`** and `grow_tree(slug=...)` became
`grow_tree(key=...)` on the way over — the concept was never book-specific, only the
name was, and renaming on promotion beat carrying a book noun into a shared package.

**Rendering stays opt-in, and now has an extra that says so.** The `viz3d` extra remains
NumPy-only. Just three functions — `smooth_paths`, `tree_mesh` and `leaf_glyphs` — need
PyVista, they import it lazily, and calling one without it raises a `ModuleNotFoundError`
naming the install command rather than a bare `AttributeError`. That keeps VTK away from
the thirteen repos that depend on this package for coordinates alone. But it also left
PyVista a genuine runtime dependency belonging to no extra at all, so every caller
hand-declared it and CI installed it by hand to let `ty` resolve the import. The new
**`viz3d-render`** extra fixes that without collapsing the distinction: depend on it to
build geometry, on `viz3d` when you only want coordinates. It could not simply be folded
into `viz3d`, because `pycode_kg` draws this same line internally — it takes
`kgmodule-utils[semantic,viz3d]` in its main dependencies and keeps PyVista in its own
extra, and widening the shared one would have handed VTK to every `pycode-kg` install.

**A release gate that checks the artifact instead of the version string.** What went wrong
in 0.12.0 was invisible to every check a source checkout can perform: the tree was
self-consistent, the tests passed, and the version string was correct — the tag simply
pointed at a commit missing the feature. `scripts/verify_release.py` now builds a wheel,
installs it into a throwaway virtualenv with no source directory on the path, and imports
the public API from there. A release that documents a symbol it does not ship fails before
it reaches PyPI.

## Upgrading

If you are on 0.12.0 and import anything from `kg_utils.viz3d.organic`, upgrade — on
0.12.0 those imports fail outright. Otherwise this is additive and nothing is required.

Rendering callers should switch their dependency from `kgmodule-utils[viz3d]` plus a
hand-declared `pyvista` to `kgmodule-utils[viz3d-render]`; layout-only callers change
nothing and acquire no VTK. Anyone porting code off `gutenberg_kg.layout_organic` should
rename `seed_from_slug` to `seed_from_key` and the `grow_tree(slug=...)` argument to
`key=...`; the behaviour is otherwise identical.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
