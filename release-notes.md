# Release Notes — v0.17.0

> Released: 2026-08-18

This release adds the acquisition stage the SDK was missing. Until now a
corpus had to already exist as Markdown or plain text on disk before any
builder could touch it. `kg_utils.ingest`, behind a new `ingest` extra, walks
arbitrary sources — PDF, Word, PowerPoint, Excel, OpenDocument, RTF, EPUB,
CSV — converts what it can to Markdown, and materializes a staging corpus the
existing builders consume unchanged. No builder code changed; every `KGModule`
gained multi-format ingestion for free. The command-line surface lives one
level up, as `kgrag ingest` in KGRAG; this package supplies the library.

## What changed

**Conversion is anydoc.** The converter is `anydoc` (PyPI `firecrawl-anydoc`),
a Rust library with Python bindings that emits consistent GitHub-Flavored
Markdown; a real 40-page text PDF converts in about 20 ms. Markdown, plain
text and reStructuredText take a passthrough path that needs no dependency at
all, which is why the extra is optional rather than folded into the core.
Passthrough deliberately preserves the source suffix — a `.txt` stays `.txt`
in the staging corpus, because DocKG parses the two differently and silently
promoting flat text to Markdown would invent a heading hierarchy the document
never had.

**A corpus now explains its own gaps.** Every file a run *examines* gets an
`IngestRecord` — not just the ones that succeed — carrying the source path,
the SHA-256 of the source bytes, the converter and its version, a timestamp,
and a status of `ingested`, `skipped` or `failed` with a reason.
`IngestManifest.problems()` returns exactly the documents the KG does not
contain and why. This answers a real failure mode: DocKG's PDF path caught its
parse error and continued, so an unparseable PDF vanished from the corpus with
no record anywhere. `anydoc` performs no OCR, so scanned PDFs are a routine
occurrence rather than an edge case, and they now surface as a skip with a
reason instead of an absence. The manifest is plain JSON at
`<staging_root>/.ingest/manifest.json`, written atomically so an interrupted
run cannot leave a half-written ledger.

**A run rebuilds from nothing by default; `update=True` is the incremental
path.** The same contract the fleet's builders already settled on —
`dockg build` / `dockg build --update`, `pycodekg build` / `pycodekg update`.
Defaulting to a rebuild eliminates the phantom footgun where deleted or
renamed sources silently persist in the staging corpus, and means a converter
upgrade needs no special flag. Dedup is keyed on the SHA-256 of source bytes
rather than the filename, so the same document arriving twice under different
names is ingested once; two *different* documents that would land on the same
staged name get a short digest suffix instead of one overwriting the other.

## Compatibility

No breaking changes. The new sub-package is additive and its heavy dependency
is opt-in:

```bash
pip install 'kgmodule-utils[ingest]'
```

Sources that are already `.md`, `.txt` or `.rst` ingest with no extra
dependency at all.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
