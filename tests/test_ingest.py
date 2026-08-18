"""Tests for kg_utils.ingest — converters, manifest, and the staging pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg_utils.ingest import (
    AnydocConverter,
    ConversionError,
    IngestManifest,
    IngestPipeline,
    IngestRecord,
    PassthroughConverter,
    default_converters,
    ingest,
    resolve_converter,
    sha256_file,
    supported_extensions,
)

anydoc = pytest.importorskip("anydoc", reason="ingest extra not installed")


# -- Converters --------------------------------------------------------------


def test_passthrough_preserves_suffix(tmp_path: Path) -> None:
    """A .txt source must stay .txt so flat-text chunking still applies."""
    src = tmp_path / "notes.txt"
    src.write_text("plain body", encoding="utf-8")

    result = PassthroughConverter().convert(src)

    assert result.markdown == "plain body"
    assert result.suffix == ".txt"
    assert result.converter == "passthrough"


def test_passthrough_normalises_markdown_suffix(tmp_path: Path) -> None:
    src = tmp_path / "readme.markdown"
    src.write_text("# Title", encoding="utf-8")

    assert PassthroughConverter().convert(src).suffix == ".md"


def test_passthrough_replaces_undecodable_bytes(tmp_path: Path) -> None:
    """A partly-binary file is still worth staging; it must not raise."""
    src = tmp_path / "mixed.txt"
    src.write_bytes(b"good \xff\xfe bad")

    result = PassthroughConverter().convert(src)

    assert "good" in result.markdown


def test_anydoc_converts_csv_to_markdown_table(tmp_path: Path) -> None:
    src = tmp_path / "rows.csv"
    src.write_text("name,qty\nbolt,4\n", encoding="utf-8")

    result = AnydocConverter().convert(src)

    assert "bolt" in result.markdown
    assert result.suffix == ".md"
    assert result.converter == "anydoc"
    assert result.converter_version  # provenance must be recorded


def test_anydoc_rejects_unknown_format(tmp_path: Path) -> None:
    src = tmp_path / "mystery.bin"
    src.write_bytes(b"\x00\x01\x02not a document")

    with pytest.raises(ConversionError):
        AnydocConverter(extensions=frozenset({".bin"})).convert(src)


def test_converters_report_a_version_without_converting() -> None:
    """Readable off the instance, so a failure can record it too."""
    assert PassthroughConverter().version
    assert AnydocConverter().version not in ("", None)


def test_converter_resolution_prefers_passthrough() -> None:
    """Markdown must never be round-tripped through anydoc."""
    converter = resolve_converter(Path("a.md"), default_converters())

    assert converter is not None
    assert converter.name == "passthrough"


def test_resolve_converter_returns_none_for_unsupported() -> None:
    assert resolve_converter(Path("photo.jpeg"), default_converters()) is None


def test_supported_extensions_covers_named_formats() -> None:
    exts = supported_extensions()

    assert {".md", ".txt", ".pdf", ".docx", ".epub", ".pptx", ".xlsx", ".csv"} <= exts


# -- Manifest ----------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = IngestManifest(IngestManifest.manifest_path(tmp_path))
    manifest.add(
        IngestRecord(
            source_path="/src/a.docx",
            sha256="abc123",
            size_bytes=10,
            status="ingested",
            staged_path="a.md",
            converter="anydoc",
        )
    )
    manifest.save()

    reloaded = IngestManifest.load(tmp_path)

    assert reloaded.has_ingested("abc123")
    assert reloaded.get("abc123").staged_path == "a.md"


def test_manifest_survives_corruption(tmp_path: Path) -> None:
    """A corrupt ledger must not dead-end the next run."""
    path = IngestManifest.manifest_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    assert IngestManifest.load(tmp_path).records == {}


def test_manifest_ignores_unknown_keys(tmp_path: Path) -> None:
    """A manifest written by a newer version stays loadable."""
    path = IngestManifest.manifest_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "manifest_version": 99,
                "records": [
                    {
                        "source_path": "/x.md",
                        "sha256": "deadbeef",
                        "size_bytes": 1,
                        "status": "ingested",
                        "staged_path": "x.md",
                        "future_field": "ignored",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert IngestManifest.load(tmp_path).has_ingested("deadbeef")


def test_failed_records_are_retried_not_skipped() -> None:
    manifest = IngestManifest(Path("unused.json"))
    manifest.add(IngestRecord(source_path="/a.pdf", sha256="ff", size_bytes=1, status="failed"))

    assert not manifest.has_ingested("ff")


def test_sha256_file_matches_content(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same", encoding="utf-8")
    b.write_text("same", encoding="utf-8")

    assert sha256_file(a) == sha256_file(b)


# -- Pipeline ----------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A small mixed-format source tree."""
    src = tmp_path / "sources"
    (src / "nested").mkdir(parents=True)
    (src / "guide.md").write_text("# Guide\n\nBody.", encoding="utf-8")
    (src / "notes.txt").write_text("flat notes", encoding="utf-8")
    (src / "data.csv").write_text("col\n1\n", encoding="utf-8")
    (src / "nested" / "deep.md").write_text("# Deep", encoding="utf-8")
    (src / "photo.jpeg").write_bytes(b"\xff\xd8\xff not really a jpeg")
    return src


def test_pipeline_stages_mixed_formats(corpus: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staged"

    stats = IngestPipeline(staging_root=staging).run([corpus])

    assert stats.ingested == 4  # md, txt, csv, nested md
    assert stats.skipped == 1  # the jpeg is unsupported
    assert stats.failed == 0
    assert (staging / "guide.md").exists()
    assert (staging / "notes.txt").exists()  # suffix preserved
    assert (staging / "data.md").exists()  # csv converted to markdown
    assert (staging / "deep.md").exists()  # nested source flattened into staging


def test_failed_records_carry_the_converter_version(tmp_path: Path) -> None:
    """Which converter rejected a file is only actionable with its version."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "scan.pdf").write_text("not actually a pdf", encoding="utf-8")

    stats = IngestPipeline(staging_root=tmp_path / "staged").run([src])

    (record,) = [r for r in stats.records if r.status == "failed"]
    assert record.converter == "anydoc"
    assert record.converter_version == AnydocConverter().version
    assert record.converter_version  # not blank


def test_unsupported_files_are_recorded_with_a_reason(corpus: Path, tmp_path: Path) -> None:
    """The silent-drop bug this pipeline exists to fix."""
    pipeline = IngestPipeline(staging_root=tmp_path / "staged")
    pipeline.run([corpus])

    problems = pipeline.manifest().problems()

    assert len(problems) == 1
    assert problems[0].source_path.endswith("photo.jpeg")
    assert "unsupported format" in problems[0].reason


def test_update_makes_reruns_incremental(corpus: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staged"
    pipeline = IngestPipeline(staging_root=staging)

    first = pipeline.run([corpus])
    second = pipeline.run([corpus], update=True)

    assert first.ingested == 4
    assert second.ingested == 0
    assert second.skipped == 5  # 4 already-ingested + the unsupported jpeg


def test_default_reconverts_everything(corpus: Path, tmp_path: Path) -> None:
    """A converter upgrade needs no special flag — the default rebuilds."""
    pipeline = IngestPipeline(staging_root=tmp_path / "staged")
    pipeline.run([corpus])

    again = pipeline.run([corpus])

    assert again.ingested == 4
    assert again.skipped == 1  # only the unsupported jpeg, no dedup skips


def test_dedup_is_by_content_not_filename(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "one.md").write_text("identical", encoding="utf-8")
    (src / "two.md").write_text("identical", encoding="utf-8")

    stats = IngestPipeline(staging_root=tmp_path / "staged").run([src])

    assert stats.ingested == 1
    assert stats.skipped == 1


def test_name_collision_does_not_overwrite(tmp_path: Path) -> None:
    """Same basename, different content — both documents must survive."""
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    (src / "a" / "report.md").write_text("first report", encoding="utf-8")
    (src / "b" / "report.md").write_text("second report", encoding="utf-8")
    staging = tmp_path / "staged"

    stats = IngestPipeline(staging_root=staging).run([src])

    assert stats.ingested == 2
    staged = sorted(p.name for p in staging.glob("*.md"))
    assert len(staged) == 2
    bodies = {(staging / name).read_text(encoding="utf-8") for name in staged}
    assert bodies == {"first report", "second report"}


def test_default_rebuild_clears_staging(corpus: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staged"
    pipeline = IngestPipeline(staging_root=staging)
    pipeline.run([corpus])
    (staging / "stale.md").write_text("left over", encoding="utf-8")

    pipeline.run([corpus])

    assert not (staging / "stale.md").exists()
    assert (staging / "guide.md").exists()


def test_deleted_source_does_not_linger(corpus: Path, tmp_path: Path) -> None:
    """The phantom-document footgun a rebuild-by-default exists to prevent."""
    staging = tmp_path / "staged"
    pipeline = IngestPipeline(staging_root=staging)
    pipeline.run([corpus])
    (corpus / "guide.md").unlink()

    pipeline.run([corpus])

    assert not (staging / "guide.md").exists()


def test_update_keeps_a_deleted_source_staged(corpus: Path, tmp_path: Path) -> None:
    """The trade --update accepts: speed, at the cost of orphans."""
    staging = tmp_path / "staged"
    pipeline = IngestPipeline(staging_root=staging)
    pipeline.run([corpus])
    (corpus / "guide.md").unlink()

    pipeline.run([corpus], update=True)

    assert (staging / "guide.md").exists()


def test_restaged_when_staged_file_deleted(corpus: Path, tmp_path: Path) -> None:
    """A recorded digest is only a no-op while its staged file still exists."""
    staging = tmp_path / "staged"
    pipeline = IngestPipeline(staging_root=staging)
    pipeline.run([corpus])
    (staging / "guide.md").unlink()

    stats = pipeline.run([corpus], update=True)

    assert stats.ingested == 1
    assert (staging / "guide.md").exists()


def test_explicit_file_source_is_ingested(tmp_path: Path) -> None:
    src = tmp_path / "solo.md"
    src.write_text("# Solo", encoding="utf-8")

    stats = ingest([src], staging_root=tmp_path / "staged")

    assert stats.ingested == 1


def test_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        IngestPipeline(staging_root=tmp_path / "staged").run([tmp_path / "nope"])


def test_staging_output_is_never_reingested(tmp_path: Path) -> None:
    """Staging inside the source tree must not feed the run its own output."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "doc.md").write_text("# Doc", encoding="utf-8")
    staging = src / "staged"

    pipeline = IngestPipeline(staging_root=staging)
    pipeline.run([src])
    second = pipeline.run([src])

    assert second.ingested == 1


def test_progress_hook_sees_every_file(corpus: Path, tmp_path: Path) -> None:
    seen: list[str] = []

    IngestPipeline(staging_root=tmp_path / "staged").run(
        [corpus], on_progress=lambda path, record: seen.append(record.status)
    )

    assert len(seen) == 5


def test_stats_serialise(corpus: Path, tmp_path: Path) -> None:
    stats = IngestPipeline(staging_root=tmp_path / "staged").run([corpus])

    payload = stats.to_dict()

    assert payload["considered"] == 5
    assert json.dumps(payload)  # must be JSON-serialisable for CLI output


def test_manifest_written_to_staging_root(corpus: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staged"

    IngestPipeline(staging_root=staging).run([corpus])

    assert (staging / ".ingest" / "manifest.json").exists()
