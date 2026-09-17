# Release Notes — v0.22.0

> Released: 2026-09-17

### Added

- **`cast_scene_to_looking_glass(..., view_cone=...)`** -- the degrees the
  camera sweeps, honored as given even past the 35-degree cap. `None`, the
  default, takes the spec's own cone capped, which is the behaviour described
  under Fixed below.

### Changed

- **The `viz3d-qt` extra now needs `quiltwright>=0.14.1`** (was `>=0.7.0`), for
  `quiltwright.quilt.resolve_view_cone`. That function landed in 0.14.0; the
  floor sits at 0.14.1, the version the cast path is tested against. Only the
  cast path is affected; a consumer that does not install `viz3d-qt` sees
  nothing.

### Fixed

- **`cast_scene_to_looking_glass()` caps the sweep at 35 degrees instead of
  sweeping the preset's full cone.** It passed the spec to `render_quilt()`
  with no `view_cone`, so a cast swept whatever the preset carries -- 50
  degrees for `16-landscape`, the cone the panel can display rather than the
  cone that reliably fuses. quiltwright's own CLI and render scripts cap the
  sweep at 35 for that reason: the wider the sweep, the further a feature
  shifts between neighbouring views, and past roughly 5 px of shift hard edges
  ghost. Every viewer's Cast button therefore produced a wider sweep than a
  quilt rendered by the same repo's CLI, of the same scene, with no way to see
  the difference except on the glass.

  The cap is not redeclared here. quiltwright 0.14.0 pulled it into
  `resolve_view_cone()` and `STANDARD_VIEW_CONE` precisely so it would stop
  being a copy per caller, and the cast path now calls that function -- which
  is a cap, not an override: a spec whose cone is already under 35 keeps it,
  where a plain default would have widened it. The new `view_cone` parameter
  is honored as given, including past the cap; `None`, the default, takes the
  spec's own cone capped. When the cone is narrowed the progress message says
  so, so a silent widening is not replaced by a silent narrowing.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
