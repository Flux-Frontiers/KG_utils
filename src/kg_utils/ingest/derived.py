"""kg_utils/ingest/derived.py — Write one converted document with provenance.

A *derived note* is the Markdown a converter produced from one source file,
written to disk with YAML frontmatter that records where it came from:

.. code-block:: yaml

    ---
    source_path: "papers/turtlend.pdf"
    source_sha256: "9f2c..."
    source_format: "pdf"
    converter: "pymupdf4llm"
    converter_version: "1.28.2"
    ingested_at: "2026-10-14T21:03:11Z"
    title: "TurtleND"
    pages: "12"
    ---

The schema is rule 3 of ``kgrag_priv/docs/INGESTION_ROADMAP.md``.  Rule 5
(same bytes and converter version give the same file) holds for everything
except ``ingested_at``, which is written once: :func:`write_derived` refuses
to overwrite, so a derived note is never regenerated in place.

This is the writer for curated imports such as ``vaultkg add``.  The staging
pipeline (:class:`~kg_utils.ingest.pipeline.IngestPipeline`) does *not* use
it yet: DocKG cannot parse frontmatter until roadmap Phase 1, so staged files
stay bare Markdown with their provenance in the ingest manifest.

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from kg_utils.ingest.converters import ConversionResult
from kg_utils.ingest.manifest import sha256_file, utc_now

#: Frontmatter keys written first, in this order; converter metadata and
#: caller extras follow, sorted.
PROVENANCE_KEYS: tuple[str, ...] = (
    "source_path",
    "source_sha256",
    "source_format",
    "converter",
    "converter_version",
    "ingested_at",
)


def derived_frontmatter(
    result: ConversionResult,
    source: Path,
    *,
    source_root: Path | None = None,
    ingested_at: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the provenance mapping for one derived note.

    :param result: The conversion to describe.
    :param source: The source file that was converted.
    :param source_root: Directory ``source_path`` is recorded relative to.
        When ``None``, or when *source* is not under it, only the file name is
        recorded, so no absolute home-directory path reaches a tracked file.
    :param ingested_at: ISO timestamp to record; defaults to now (UTC).
    :param extra: Caller fields (for example ``kind``). They may not replace a
        provenance key.
    :return: Ordered mapping of frontmatter keys to string values.
    :raises ValueError: If *extra* names a provenance key.
    """
    source = Path(source)
    rel = source.name
    if source_root is not None:
        try:
            rel = source.resolve().relative_to(Path(source_root).resolve()).as_posix()
        except ValueError:
            rel = source.name
    fields: dict[str, str] = {
        "source_path": rel,
        "source_sha256": sha256_file(source),
        "source_format": result.metadata.get("source_format") or source.suffix.lower().lstrip("."),
        "converter": result.converter,
        "converter_version": result.converter_version,
        "ingested_at": ingested_at or utc_now(),
    }
    clash = sorted(set(extra or {}) & set(PROVENANCE_KEYS))
    if clash:
        raise ValueError(f"extra may not replace provenance keys: {clash}")
    tail = {k: v for k, v in result.metadata.items() if k not in fields}
    tail.update(extra or {})
    for key in sorted(tail):
        fields[key] = str(tail[key])
    return fields


def render_derived(fields: Mapping[str, str], markdown: str) -> str:
    """Render frontmatter plus body as one document.

    Values are written as JSON strings, which are valid double-quoted YAML
    scalars, so a title containing ``: `` or quotes cannot break the block.

    :param fields: Frontmatter keys and values.
    :param markdown: Document body.
    :return: The complete file text, ending in one newline.
    """
    lines = ["---"]
    lines += [
        f"{key}: {json.dumps(str(value), ensure_ascii=False)}" for key, value in fields.items()
    ]
    lines += ["---", ""]
    return "\n".join(lines) + markdown.rstrip("\n") + "\n"


def write_derived(
    result: ConversionResult,
    source: Path,
    dest: Path,
    *,
    source_root: Path | None = None,
    ingested_at: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> Path:
    """Write *result* to *dest* as a derived note with provenance frontmatter.

    :param result: The conversion to write.
    :param source: The source file that was converted.
    :param dest: Where to write the note. Its parent is created if needed.
    :param source_root: See :func:`derived_frontmatter`.
    :param ingested_at: See :func:`derived_frontmatter`.
    :param extra: See :func:`derived_frontmatter`.
    :return: *dest*.
    :raises FileExistsError: If *dest* already exists. A derived note is
        written once; deciding what an existing file means (an identical
        re-import, or a name collision) is the caller's job.
    """
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"derived note already exists: {dest}")
    fields = derived_frontmatter(
        result, source, source_root=source_root, ingested_at=ingested_at, extra=extra
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as fh:
        fh.write(render_derived(fields, result.markdown))
    return dest
