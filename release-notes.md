# Release Notes - v0.19.0

> Released: 2026-09-05

### Changed

- **Snapshots are no longer keyed by git tree hash.** The hash was read before
  `git add` staged the snapshot, so it named a tree that never got committed:
  of 605 snapshot keys across the fleet, 63 resolve. `Snapshot.key` is now a
  supplied value, and `SnapshotManager.capture()` takes `key=`. Pass a release
  tag for a repo snapshot; omit it for a corpus, which has no tag, and get a
  UTC timestamp. It never falls back to the tree hash.

  `tree_hash` is still recorded as a field. It is real provenance, it was just
  never an identifier.

  **This changes what `capture()` writes for every caller that does not pass a
  key.** Existing snapshots are unaffected and stay addressable -- see the dual
  read below.

- **`Snapshot.to_dict()` reads `metrics`, `vs_previous` and `vs_baseline` out
  of `__dict__` and converts dataclasses.** Four modules (`pycode_kg`,
  `doc_kg`, `memory_kg`, `Metabo_kg`) override `to_dict` only because they
  expose those fields as typed properties the base could not serialize. Those
  overrides can now be deleted, which is what lets the key change reach them
  rather than stopping at the SDK.

### Added

- **`subject`, `tool` and `tool_version` on `Snapshot`.** `version` records the
  version of the *measuring tool*, not of the thing measured -- so `doc_kg`'s
  `.pycodekg` snapshots carry pycode-kg's version numbers. `subject`
  (`repo:doc-kg`, `corpus:pepys`) names what was measured; the tool and its
  version are recorded separately. Both are surfaced in manifest entries.

### Fixed

- **The manifest loader accepts all three key shapes** -- `commit`,
  `tree_hash` and `key`. This schema has been migrated twice before; the
  tree-hash entries cannot be re-keyed onto versions, so the dual read is
  permanent rather than transitional.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
