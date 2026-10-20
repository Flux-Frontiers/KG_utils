# Release Notes -- v0.27.1

> Released: 2026-10-19

### Fixed

- **`[pdf]` and `[ingest]` installed no PDF code on a fresh resolve.**
  pymupdf4llm 2.0 (2026-10-16) is an empty placeholder: the pymupdf 2.0
  wheel now ships the `pymupdf4llm` module, and the placeholder installs no
  code and does not depend on pymupdf. Both extras declared
  `pymupdf4llm>=1.28.2`, so `pip install kgmodule-utils[pdf]==0.27.0` today
  gets the placeholder and no pymupdf, and `PdfConverter` cannot import.
  Both extras and the dev group now declare `pymupdf>=2.0`. `PdfConverter`
  is unchanged; it still calls `pymupdf4llm.to_markdown`. The converter
  tests import pymupdf directly instead of `importorskip`, so a broken PDF
  install fails the suite instead of skipping it. Sweep item 86.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
