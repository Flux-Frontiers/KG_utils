# Release Notes — v0.25.1

> Released: 2026-09-26

### Fixed

- **Blackthorn and fir limbs no longer wander.** Two of the nine species
  presets 0.25.0 shipped grew limbs that random-walked instead of reaching
  their chunks. The blackthorn (influence 6, jitter 0.3) saw too few chunks
  from each tip to steer by, and its limbs coiled into helices -- the median
  root-to-tip path ran 2.4x the straight distance on some books. The fir
  (influence 7) could not reach its wide bottom whorl from the trunk, so a
  branch from higher up grew back down to it (2.7x). The blackthorn now
  grows at influence 10 and jitter 0.18 (1.2-1.3x, and still the twiggiest
  broadleaf), the fir at influence 12 (1.3-1.4x). A new test bounds every
  species' wander at 2.0x on the books that exposed both, and proves the
  0.25.0 values fail it. The Knowledge Press web forest's `species.ts`
  mirrors both changes.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
