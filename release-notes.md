# Release Notes -- v0.29.2

> Released: 2026-11-02

### Security

- **Pillow 13.0.0 is the floor** in the `synthesis` and `synthesis-mflux`
  extras. `pillow>=10.0.0` let a fresh install resolve Pillow 12.3.0, which
  carries 41 advisories fixed in 13.0.0, some of them critical.
- **`viz3d-qt` floors quiltwright at 0.17.0**, the first quiltwright release
  that floors Pillow at 13.0.0, so the extra cannot pull the 12.x line in
  through it.
- **Lock refresh** (#47): Pillow 13.0.0, certifi 2026.10.23, filelock 4.2.7,
  setuptools 84.2.1 and virtualenv 21.15.2, clearing the 56 open Dependabot
  alerts.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
