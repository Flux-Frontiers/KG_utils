# Release Notes — v0.30.1

> Released: 2026-11-17

### Changed

- **The image rewrite prompt asks for an accurate scene.** `_IMAGE_REWRITE_SYSTEM`
  keeps the art-director prompt and adds rules: show the scene at the end of
  the moment, show a person turned into something only as what they became,
  place a narrator or eyewitness in the scene, describe people by appearance
  and period clothing, keep everyone clothed, ask for no lettering, and stay
  under 120 words. Tested 2026-11-17 on four answers rendered through FLUX.2
  klein: with Qwen3.6-35B-A3B or Qwen3.8-27B as the rewrite model, Lot's wife
  appears only as the pillar of salt and Pepys watches from his boat. The 4B
  model ignores the transformation rule either way.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
