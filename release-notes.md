# Release Notes — v0.26.0

> Released: 2026-09-26

### Added

- **`hang_leaves`: the web forest's leaf placement, in NumPy.** Each leaf's
  stalk sits on the line from its nearest skeleton node toward its chunk, at
  most `LEAF_REACH` out (the web's 0.1 m at its 1.7 / 4 scale). The blade
  points out along that line with 0.55 of up added, and the face turns as
  close to up as the blade allows. It returns `(bases, blades, faces)`, so a
  renderer can stamp a flat species leaf in the same frame the Knowledge
  Press web forest uses (`growTree.ts`, `emitLeaves`). `leaf_frames` is
  unchanged. It turns leaves along the branch with a random roll, which
  sets many of them edge-on, and POV-Ray trees drawn with it looked
  nothing like the web's.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
