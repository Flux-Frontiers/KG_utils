"""kg_utils.ingest — Heterogeneous documents → staged Markdown corpus.

The acquisition stage that sits *in front of* every KGModule builder.  Point it
at ``.md``, ``.pdf``, ``.docx``, ``.epub``, ``.pptx``, ``.xlsx``, ``.csv`` (and
the rest of the ``anydoc`` format set); it converts each to Markdown,
materializes a staging corpus, and records provenance for every file it saw —
including the ones it could not convert.

Modules:
    kg_utils.ingest.converters — Converter protocol, PassthroughConverter,
                                 AnydocConverter, ConversionError.
    kg_utils.ingest.manifest   — IngestRecord, IngestManifest, IngestStats.
    kg_utils.ingest.pipeline   — IngestPipeline, ingest().

Requires the ``ingest`` extra for non-textual formats::

    pip install 'kgmodule-utils[ingest]'

Markdown and plain text pass through with no extra dependency at all.

Example::

    from kg_utils.ingest import IngestPipeline

    pipeline = IngestPipeline(staging_root="corpus/")
    stats = pipeline.run(["~/Documents/specs", "notes.docx"])

    print(f"{stats.ingested} staged, {stats.skipped} skipped, {stats.failed} failed")
    for record in pipeline.manifest().problems():
        print(f"  {record.source_path}: {record.reason}")

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

from kg_utils.ingest.converters import (
    ANYDOC_EXTENSIONS,
    PASSTHROUGH_EXTENSIONS,
    AnydocConverter,
    ConversionError,
    ConversionResult,
    Converter,
    PassthroughConverter,
    default_converters,
    resolve_converter,
    supported_extensions,
)
from kg_utils.ingest.manifest import (
    INGEST_DIR,
    MANIFEST_NAME,
    MANIFEST_VERSION,
    IngestManifest,
    IngestRecord,
    IngestStats,
    IngestStatus,
    sha256_file,
)
from kg_utils.ingest.pipeline import SKIP_DIRS, IngestPipeline, ingest

__all__ = [
    "ANYDOC_EXTENSIONS",
    "INGEST_DIR",
    "MANIFEST_NAME",
    "MANIFEST_VERSION",
    "PASSTHROUGH_EXTENSIONS",
    "SKIP_DIRS",
    "AnydocConverter",
    "ConversionError",
    "ConversionResult",
    "Converter",
    "IngestManifest",
    "IngestPipeline",
    "IngestRecord",
    "IngestStats",
    "IngestStatus",
    "PassthroughConverter",
    "default_converters",
    "ingest",
    "resolve_converter",
    "sha256_file",
    "supported_extensions",
]
