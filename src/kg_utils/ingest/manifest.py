"""kg_utils/ingest/manifest.py — Provenance ledger for an ingested corpus.

Every file an ingest run *considers* gets a record, not just the ones that
succeed.  A corpus that quietly omits the three PDFs it could not parse is
indistinguishable from one that never saw them; the manifest is what makes the
difference visible and re-runnable.

Each record carries the source path, the SHA-256 of the source bytes, the
converter and its version, and a status of ``ingested``, ``skipped`` or
``failed`` with a reason.  The digest doubles as the dedup key: the same bytes
arriving twice under different names are ingested once.

The manifest lives at ``<staging_root>/.ingest/manifest.json`` and is plain
JSON — diffable in review and readable without this library.

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

#: Directory created inside the staging root to hold ingest bookkeeping.
INGEST_DIR = ".ingest"

#: Manifest filename inside :data:`INGEST_DIR`.
MANIFEST_NAME = "manifest.json"

#: Bumped when the on-disk record shape changes incompatibly.
MANIFEST_VERSION = 1

IngestStatus = Literal["ingested", "skipped", "failed"]


@dataclass
class IngestRecord:
    """One source file's outcome in an ingest run.

    :param source_path: Absolute path of the source document.
    :param sha256: Hex digest of the source bytes — the dedup key.
    :param size_bytes: Size of the source file in bytes.
    :param status: ``"ingested"``, ``"skipped"`` or ``"failed"``.
    :param staged_path: Staging-root-relative path of the written Markdown;
                        empty for skipped and failed records.
    :param converter: Converter that produced the staged file.
    :param converter_version: Version of the converting library.
    :param reason: Why a file was skipped or failed; empty when ingested.
    :param ingested_at: ISO 8601 UTC timestamp of the run that wrote this record.
    :param metadata: Converter-specific extras (e.g. detected source format).
    """

    source_path: str
    sha256: str
    size_bytes: int
    status: IngestStatus
    staged_path: str = ""
    converter: str = ""
    converter_version: str = ""
    reason: str = ""
    ingested_at: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestRecord:
        """Rebuild a record from its dictionary form, ignoring unknown keys.

        Unknown keys are dropped rather than raising so that a manifest written
        by a newer version stays loadable by an older one.

        :param data: Dictionary as produced by :meth:`to_dict`.
        :return: The reconstructed record.
        """
        known = {f for f in cls.__dataclass_fields__}  # pylint: disable=no-member
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class IngestStats:
    """Totals for one ingest run.

    :param ingested: Files converted and staged.
    :param skipped: Files deliberately not staged (duplicate, unchanged, unsupported).
    :param failed: Files that a converter rejected.
    :param staging_root: Directory the corpus was staged into.
    :param records: The records produced by this run.
    """

    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    staging_root: str = ""
    records: list[IngestRecord] = field(default_factory=list)

    @property
    def considered(self) -> int:
        """Total files examined by the run."""
        return self.ingested + self.skipped + self.failed

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary, records included."""
        return {
            "ingested": self.ingested,
            "skipped": self.skipped,
            "failed": self.failed,
            "considered": self.considered,
            "staging_root": self.staging_root,
            "records": [r.to_dict() for r in self.records],
        }


class IngestManifest:
    """The provenance ledger for one staging root.

    Records are keyed by source SHA-256, which makes re-ingesting the same
    bytes a no-op regardless of the filename they arrive under.

    :param path: Path to ``manifest.json``.
    :param records: Existing records, keyed by digest.
    """

    def __init__(self, path: Path, records: dict[str, IngestRecord] | None = None) -> None:
        self.path = path
        self.records: dict[str, IngestRecord] = records or {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def manifest_path(cls, staging_root: Path) -> Path:
        """Return the manifest path for *staging_root*.

        :param staging_root: Directory documents are staged into.
        :return: Path to ``<staging_root>/.ingest/manifest.json``.
        """
        return staging_root / INGEST_DIR / MANIFEST_NAME

    @classmethod
    def load(cls, staging_root: Path) -> IngestManifest:
        """Load the manifest for *staging_root*, or return an empty one.

        A corrupt or unreadable manifest yields an empty manifest rather than
        raising: the staged files are still on disk, and a fresh ledger lets
        the next run rebuild rather than dead-end.

        :param staging_root: Directory documents are staged into.
        :return: The loaded manifest.
        """
        path = cls.manifest_path(staging_root)
        if not path.exists():
            return cls(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get("records", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            return cls(path)

        records: dict[str, IngestRecord] = {}
        for item in raw:
            try:
                record = IngestRecord.from_dict(item)
            except TypeError:
                continue
            if record.sha256:
                records[record.sha256] = record
        return cls(path, records)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, record: IngestRecord) -> None:
        """Insert or replace *record*, keyed by its digest.

        :param record: The record to store.
        """
        self.records[record.sha256] = record

    def get(self, sha256: str) -> IngestRecord | None:
        """Return the record for *sha256*, or ``None``.

        :param sha256: Source digest to look up.
        :return: The matching record, if any.
        """
        return self.records.get(sha256)

    def has_ingested(self, sha256: str) -> bool:
        """Return ``True`` if *sha256* was previously ingested successfully.

        Only ``ingested`` counts: a previously failed file should be retried on
        the next run, since the converter or its version may have changed.

        :param sha256: Source digest to look up.
        :return: Whether a successful record exists for this digest.
        """
        record = self.records.get(sha256)
        return record is not None and record.status == "ingested"

    def staged_paths(self) -> set[str]:
        """Return every staged path currently recorded as ingested."""
        return {
            r.staged_path for r in self.records.values() if r.status == "ingested" and r.staged_path
        }

    def save(self) -> None:
        """Write the manifest to disk atomically.

        The file is written to a sibling temporary path and moved into place so
        an interrupted run cannot leave a half-written ledger behind.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "updated_at": utc_now(),
            "records": [
                r.to_dict() for r in sorted(self.records.values(), key=lambda r: r.source_path)
            ],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """Return record counts by status.

        :return: Mapping of status name to count, including a ``total``.
        """
        counts = {"ingested": 0, "skipped": 0, "failed": 0}
        for record in self.records.values():
            if record.status in counts:
                counts[record.status] += 1
        counts["total"] = len(self.records)
        return counts

    def problems(self) -> list[IngestRecord]:
        """Return every record that did not result in a staged document.

        This is the list a corpus owner needs to see — the documents the KG
        does *not* contain, and why.

        :return: Skipped and failed records, ordered by source path.
        """
        return sorted(
            (r for r in self.records.values() if r.status != "ingested"),
            key=lambda r: r.source_path,
        )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of *path*.

    Read in chunks so that a multi-gigabyte source never has to be held in
    memory at once.

    :param path: File to digest.
    :param chunk_size: Bytes to read per iteration (default 1 MiB).
    :return: Lower-case hex digest.
    :raises OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()
