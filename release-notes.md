# Release Notes — v0.27.0

> Released: 2026-10-14

This release gives the fleet one converter per document format and lets a
knowledge graph update its vectors without re-embedding everything. PDFs now
convert the same way whether they arrive through `kgrag ingest` or `dockg
build`, LaTeX and HTML become ingestible, and `KGModule.sync()` re-embeds only
the nodes whose text changed.

## What changed

**One PDF converter.** `PdfConverter` makes the same `pymupdf4llm` call that
`doc_kg` makes, so a PDF produces identical Markdown on every path. It
replaces anydoc for `.pdf`; anydoc keeps Word, PowerPoint, Excel,
OpenDocument, RTF, EPUB and CSV. The converter also records the PDF's title,
author, creation date and page count. A PDF with no text layer is reported
as needing OCR, never silently dropped.

**LaTeX and HTML.** `PandocConverter` reads `.tex`, `.latex`, `.html` and
`.htm` through the `pandoc` command. Pandoc is a system program, not a
Python dependency; without it these files are recorded as failed with an
install hint.

**Documents with their provenance.** `write_derived()` writes one converted
document with YAML frontmatter that says where it came from: the source path
relative to an import root, its SHA-256, the converter and its version, and
when it was converted. It never overwrites, so that timestamp is written
once. It is meant for curated imports such as `vaultkg add`; the staging
pipeline keeps writing bare Markdown until DocKG reads frontmatter.

**Incremental vectors.** `KGModule.sync()` rewrites the graph in full, then
embeds only the nodes whose index text is new or changed and deletes the
vectors of nodes that are gone. Its bookkeeping lives in two tables of its
own, so `build()` and the shared vector table are unchanged for every
consumer. The first sync of a store embeds everything once; later syncs are
incremental.

**New `[pdf]` extra.** `kgmodule-utils[pdf]` installs only `pymupdf4llm`, for
packages that read PDFs but need none of anydoc's formats. `[ingest]` now
carries both converters. `pymupdf4llm` brings in `pymupdf`, which is
AGPL-3.0 or commercially licensed.

## Upgrading

Nothing changes for code that calls `build()`. To ingest PDFs, install
`kgmodule-utils[ingest]` (every format) or `kgmodule-utils[pdf]` (PDF only);
a package that listed `firecrawl-anydoc` directly for PDF support should
switch to one of these extras. Install `pandoc` to ingest LaTeX and HTML.
Call `sync()` instead of `build(wipe=True)` where incremental rebuilds help.

Before tagging, the release wheel ran against all 16 fleet consumers: 5,325
tests, no regressions.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
