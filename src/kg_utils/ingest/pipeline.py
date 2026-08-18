"""kg_utils/ingest/pipeline.py — Source documents → staged Markdown corpus.

The stage every KG module was missing.  Until now a corpus had to *already*
exist as Markdown or plain text on disk; this walks arbitrary sources, converts
whatever it can to Markdown, and materializes a staging corpus that the
existing builders (``dockg build``, ``memorykg build``, …) consume unchanged.

Materializing rather than streaming is deliberate.  The staged corpus is
inspectable, diffable and re-buildable without re-running conversion, and no
builder internals had to change to gain multi-format ingestion.

A run rebuilds the staging corpus from nothing by default, and ``update=True``
selects the incremental path — the same contract as ``dockg build`` /
``dockg build --update`` and ``pycodekg build`` / ``pycodekg update``.

Defaulting to a full rebuild is what keeps the corpus honest. Under an
incremental default, a source document deleted or renamed upstream leaves its
staged copy behind forever and it keeps being built into the KG — the
phantom-node footgun the fleet's builders removed by making the wipe implicit::

    from kg_utils.ingest import IngestPipeline

    stats = IngestPipeline(staging_root="corpus/").run(["~/Documents/specs"])
    print(stats.ingested, "staged;", stats.failed, "could not be converted")

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from kg_utils.ingest.converters import (
    ConversionError,
    Converter,
    default_converters,
    resolve_converter,
)
from kg_utils.ingest.manifest import (
    INGEST_DIR,
    IngestManifest,
    IngestRecord,
    IngestStats,
    sha256_file,
    utc_now,
)

#: Directory names never descended into while walking sources.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        INGEST_DIR,
    }
)

#: Characters not allowed in a staged filename component.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

ProgressHook = Callable[[Path, IngestRecord], None]


class IngestPipeline:
    """Convert heterogeneous source documents into a staged Markdown corpus.

    :param staging_root: Directory to write the normalized corpus into.
                         Created if absent.
    :param converters: Converter chain, most specific first
                       (default: :func:`~kg_utils.ingest.converters.default_converters`).
    :param skip_dirs: Extra directory names to prune while walking
                      (combined with :data:`SKIP_DIRS`).
    :param follow_symlinks: Whether to descend into symlinked directories.
                            Off by default — a symlink loop would otherwise
                            walk forever.
    """

    def __init__(
        self,
        staging_root: str | Path,
        converters: Sequence[Converter] | None = None,
        skip_dirs: Iterable[str] | None = None,
        follow_symlinks: bool = False,
    ) -> None:
        self.staging_root = Path(staging_root).expanduser().resolve()
        self.converters: list[Converter] = list(converters) if converters else default_converters()
        self.skip_dirs = SKIP_DIRS | set(skip_dirs or ())
        self.follow_symlinks = follow_symlinks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        sources: Iterable[str | Path],
        update: bool = False,
        on_progress: ProgressHook | None = None,
    ) -> IngestStats:
        """Ingest *sources* into the staging corpus.

        By default the staging corpus and its manifest are deleted first and
        every source is re-converted, so the result reflects exactly the
        sources given — documents removed upstream do not linger. This also
        means a converter upgrade is picked up with no special flag.

        ``update=True`` selects the incremental path: existing staged documents
        are kept and sources already recorded as ingested are skipped by
        content digest, so a growing folder only converts what is new. The
        trade is that a source deleted upstream keeps its staged copy.

        Every file examined produces a manifest record, including the ones that
        could not be converted: a corpus always explains its own gaps.

        :param sources: Files and/or directories to ingest. Directories are
                        walked recursively.
        :param update: Incremental update — keep existing staged documents
                       instead of rebuilding from nothing.
        :param on_progress: Called with ``(source_path, record)`` after each
                            file is processed — for progress bars and logging.
        :return: Totals and per-file records for this run.
        """
        if not update and self.staging_root.exists():
            shutil.rmtree(self.staging_root)
        self.staging_root.mkdir(parents=True, exist_ok=True)

        manifest = IngestManifest.load(self.staging_root)
        stats = IngestStats(staging_root=str(self.staging_root))
        # Reserve names already claimed by previous runs so a re-ingest never
        # overwrites an unrelated document that happens to share a basename.
        taken = manifest.staged_paths()

        for source in self._iter_sources(sources):
            record = self._ingest_one(source, manifest, taken)
            manifest.add(record)
            stats.records.append(record)
            if record.status == "ingested":
                stats.ingested += 1
                taken.add(record.staged_path)
            elif record.status == "skipped":
                stats.skipped += 1
            else:
                stats.failed += 1
            if on_progress is not None:
                on_progress(source, record)

        manifest.save()
        return stats

    def manifest(self) -> IngestManifest:
        """Return the manifest currently on disk for this staging root.

        :return: The loaded manifest (empty if nothing has been ingested yet).
        """
        return IngestManifest.load(self.staging_root)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ingest_one(
        self,
        source: Path,
        manifest: IngestManifest,
        taken: set[str],
    ) -> IngestRecord:
        """Convert and stage a single source file, returning its record.

        :param source: Source file to ingest.
        :param manifest: Manifest consulted for dedup. After a rebuild it holds
                         only what this run has staged, so the same check
                         deduplicates within a run and across runs in update
                         mode, with no separate flag.
        :param taken: Staged paths already claimed in this staging root.
        :return: The record describing this file's outcome.
        """
        try:
            digest = sha256_file(source)
            size = source.stat().st_size
        except OSError as exc:
            return IngestRecord(
                source_path=str(source),
                sha256="",
                size_bytes=0,
                status="failed",
                reason=f"cannot read source: {exc}",
                ingested_at=utc_now(),
            )

        if manifest.has_ingested(digest):
            prior = manifest.get(digest)
            staged = prior.staged_path if prior else ""
            # Only a genuine no-op if the staged file is still on disk; if it
            # was deleted, fall through and re-stage it.
            if staged and (self.staging_root / staged).exists():
                return IngestRecord(
                    source_path=str(source),
                    sha256=digest,
                    size_bytes=size,
                    status="skipped",
                    staged_path=staged,
                    converter=prior.converter if prior else "",
                    converter_version=prior.converter_version if prior else "",
                    reason="already ingested (identical content)",
                    ingested_at=prior.ingested_at if prior else utc_now(),
                    metadata=dict(prior.metadata) if prior else {},
                )

        converter = resolve_converter(source, self.converters)
        if converter is None:
            return IngestRecord(
                source_path=str(source),
                sha256=digest,
                size_bytes=size,
                status="skipped",
                reason=f"unsupported format: {source.suffix or '(no suffix)'}",
                ingested_at=utc_now(),
            )

        try:
            result = converter.convert(source)
        except ConversionError as exc:
            return IngestRecord(
                source_path=str(source),
                sha256=digest,
                size_bytes=size,
                status="failed",
                converter=converter.name,
                reason=str(exc),
                ingested_at=utc_now(),
            )

        # A digest that already has a record owns its staged path: reuse it so
        # re-staging a deleted file restores the original name rather than
        # colliding with the entry it is replacing.
        previous = manifest.get(digest)
        staged_rel = (
            previous.staged_path
            if previous is not None and previous.staged_path
            else self._staged_path(source, result.suffix, digest, taken)
        )
        target = self.staging_root / staged_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(result.markdown, encoding="utf-8")
        except OSError as exc:
            return IngestRecord(
                source_path=str(source),
                sha256=digest,
                size_bytes=size,
                status="failed",
                converter=converter.name,
                converter_version=result.converter_version,
                reason=f"cannot write staged file: {exc}",
                ingested_at=utc_now(),
            )

        return IngestRecord(
            source_path=str(source),
            sha256=digest,
            size_bytes=size,
            status="ingested",
            staged_path=staged_rel,
            converter=result.converter,
            converter_version=result.converter_version,
            ingested_at=utc_now(),
            metadata=dict(result.metadata),
        )

    def _iter_sources(self, sources: Iterable[str | Path]) -> list[Path]:
        """Expand *sources* into a sorted list of candidate files.

        Directories are walked recursively with :data:`SKIP_DIRS` pruned;
        explicitly named files are always included, even if their suffix is
        unsupported, so that naming a file and getting no explanation is
        impossible.

        :param sources: Files and/or directories.
        :return: Deduplicated, sorted candidate paths.
        :raises FileNotFoundError: If a named source does not exist.
        """
        found: list[Path] = []
        seen: set[Path] = set()

        for raw in sources:
            root = Path(raw).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(f"source does not exist: {root}")

            if root.is_file():
                if root not in seen:
                    seen.add(root)
                    found.append(root)
                continue

            for dirpath, dirnames, filenames in os.walk(root, followlinks=self.follow_symlinks):
                dirnames[:] = sorted(
                    d for d in dirnames if d not in self.skip_dirs and not d.startswith(".")
                )
                for name in sorted(filenames):
                    if name.startswith("."):
                        continue
                    path = Path(dirpath) / name
                    # Never re-ingest our own staged output.
                    if self._within_staging(path) or path in seen:
                        continue
                    seen.add(path)
                    found.append(path)

        return sorted(found)

    def _within_staging(self, path: Path) -> bool:
        """Return ``True`` if *path* lies inside this pipeline's staging root.

        Guards the case where a source directory contains, or is, the staging
        root — otherwise a second run would ingest its own output.

        :param path: Path to test.
        :return: Whether the path is under the staging root.
        """
        return path == self.staging_root or self.staging_root in path.parents

    def _staged_path(self, source: Path, suffix: str, digest: str, taken: set[str]) -> str:
        """Return a collision-free staging-root-relative path for *source*.

        The source's own filename is kept so the staged corpus stays readable.
        When two different documents would land on the same name, the second
        gets a short digest suffix rather than overwriting the first.

        :param source: Source file being staged.
        :param suffix: Suffix the staged file must carry.
        :param digest: Source SHA-256, used to disambiguate collisions.
        :param taken: Staged paths already claimed.
        :return: Relative path such as ``report.md`` or ``report-a1b2c3d4.md``.
        """
        stem = _UNSAFE.sub("-", source.stem).strip("-.") or "document"
        candidate = f"{stem}{suffix}"
        if candidate not in taken and not (self.staging_root / candidate).exists():
            return candidate

        candidate = f"{stem}-{digest[:8]}{suffix}"
        if candidate not in taken and not (self.staging_root / candidate).exists():
            return candidate

        # Same stem *and* same digest prefix: fall back to the full digest,
        # which is unique by construction.
        return f"{stem}-{digest}{suffix}"


def ingest(
    sources: Iterable[str | Path],
    staging_root: str | Path,
    update: bool = False,
    converters: Sequence[Converter] | None = None,
    on_progress: ProgressHook | None = None,
) -> IngestStats:
    """Ingest *sources* into *staging_root* — the one-call form of the pipeline.

    :param sources: Files and/or directories to ingest.
    :param staging_root: Directory to write the normalized corpus into.
    :param update: Incremental update — keep existing staged documents instead
                   of rebuilding from nothing.
    :param converters: Converter chain override.
    :param on_progress: Called with ``(source_path, record)`` per file.
    :return: Totals and per-file records for this run.
    """
    pipeline = IngestPipeline(staging_root=staging_root, converters=converters)
    return pipeline.run(sources, update=update, on_progress=on_progress)
