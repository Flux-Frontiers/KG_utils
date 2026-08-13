# Release Notes — v0.12.0

> Released: 2026-08-13

A consolidation release. No API changes and no new symbols: this one hardens the
`viz3d` layout engine that landed in 0.11.0 with the tests its downstream consumers were
implicitly relying on, documents a contract the move surfaced, and clears a security
advisory in the dev toolchain. Upgrading is a no-op.

## What changed

**The allium sizing formulas are pinned where they now live.** When `Layout3D` and friends
moved out of `pycode_kg.layout3d` in 0.11.0, the head-radius formula came with them — but
PyCodeKG's `test_viz3d_sizing` still asserts, against a hand-restated copy of that formula,
that a maximum-centrality function fits inside a four-child allium head. The formula was
here and the only test of it was over there. Changing a coefficient would have quietly
re-tuned every consumer's occlusion budget with nothing in this repo to catch it. The head
radius, orbit radius, and layout determinism are now covered by tests that live alongside
the code they constrain.

**`AlliumLayout` documents that it trusts your node ordering.** Roots take their annulus
slots in the order they appear in the node list, which means a store whose iteration order
varies between rebuilds will shuffle the entire scene even when the graph has not changed.
Sorting internally would make the problem disappear, but it would also relocate every node
in every scene anyone has already rendered. The behaviour is therefore pinned and
documented rather than changed: pass a stable order.

**pytest's dev pin moved to `>=9.0.3`,** resolving GHSA-6w46-j5rx-g56g / PYSEC-2026-1845
and lifting the `^8.0.0` cap deferred during the 0.10.x security pass. This is the dev
group only — pytest appears in no published extra and in neither the wheel nor the sdist
metadata, so nothing about the released artifact changes.

## Upgrading

Nothing to do. No public API moved, no extra changed its contents, and no rebuild is
required. If you render 3-D scenes with `AlliumLayout` and have noticed them reshuffling
between otherwise identical runs, the newly documented ordering requirement is the reason —
sort your node list before handing it over.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
