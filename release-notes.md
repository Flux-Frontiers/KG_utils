# Release Notes — v0.31.0

> Released: 2026-11-22

### Added

- **An Art Techno logo.** The backplane: one board carries a bus, and five KG
  modules stand in their own slots on it, with the tagline THE SDK UNDER EVERY
  GRAPH. `scripts/build_logo.py` draws the mark, a 64 px icon and a banner as
  SVG under `assets/` and writes the PNGs. The README shows the banner at the
  top, by absolute raw URL so PyPI shows it too.
- **A second image rewrite prompt, for non-prose text.**
  `TextSynthesizer.rewrite_for_image` takes `system=`; `None` keeps the
  historical-prose prompt, so existing callers are unchanged. The new
  `IMAGE_REWRITE_TECHNICAL_SYSTEM` asks for a clear technical or scientific
  illustration of source code, structured records, agent memory or graph data
  such as neurons and pathways, with no invented people or period scenes. Both
  prompts are exported from `kg_utils.synthesis`; the historical one is
  `IMAGE_REWRITE_SYSTEM`, and `_IMAGE_REWRITE_SYSTEM` still works.
  `WorkerClient.rewrite(system=...)` sends it as an optional `system` field of
  the `rewrite` op, and `handle_aux_ops` passes it through. A worker on an
  older kgmodule-utils ignores the field and uses the historical prompt.

### Changed

- **The dev group installs `doc-kg`, so no test is skipped.** The test that
  checks `doc_kg.index` still re-exports the embedder classes used to skip
  unless `doc-kg` had been installed by hand. `doc-kg` is now a dev
  dependency, and the test imports it directly, so a missing `doc_kg` fails.
  The lock gains `doc-kg` alone. Nothing in the wheel changes.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
