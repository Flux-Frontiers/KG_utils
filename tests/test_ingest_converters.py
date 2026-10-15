"""PdfConverter, PandocConverter and write_derived (sweep item 48).

PDFs are generated with pymupdf at test time, so no binary fixture is
committed. Pandoc is a system binary: the conversion tests skip when it is
absent, and the missing-pandoc path is tested on every machine by hiding it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kg_utils.ingest import (
    ANYDOC_EXTENSIONS,
    PROVENANCE_KEYS,
    ConversionError,
    ConversionResult,
    IngestPipeline,
    PandocConverter,
    PdfConverter,
    default_converters,
    resolve_converter,
    supported_extensions,
    write_derived,
)
from kg_utils.ingest import converters as converters_mod

pymupdf = pytest.importorskip("pymupdf")
pymupdf4llm = pytest.importorskip("pymupdf4llm")

HAS_PANDOC = shutil.which("pandoc") is not None
needs_pandoc = pytest.mark.skipif(not HAS_PANDOC, reason="pandoc binary not on PATH")


def _pdf(path: Path, text: str | None, title: str = "") -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    if text is not None:
        page.insert_text((72, 72), text, fontsize=12)
    if title:
        doc.set_metadata({"title": title, "author": "Test Author"})
    doc.save(str(path))
    doc.close()
    return path


# -- routing -----------------------------------------------------------------


def test_pdf_routes_to_pymupdf4llm_not_anydoc() -> None:
    assert ".pdf" not in ANYDOC_EXTENSIONS
    converter = resolve_converter(Path("paper.PDF"), default_converters())
    assert converter is not None and converter.name == "pymupdf4llm"


@pytest.mark.parametrize("name", ["paper.tex", "page.html", "page.htm", "notes.latex"])
def test_tex_and_html_route_to_pandoc(name: str) -> None:
    converter = resolve_converter(Path(name), default_converters())
    assert converter is not None and converter.name == "pandoc"


def test_supported_extensions_include_the_new_formats() -> None:
    assert {".pdf", ".tex", ".html", ".htm", ".docx"} <= supported_extensions()


def test_converters_report_versions_without_converting() -> None:
    assert PdfConverter().version not in ("", "unknown")
    assert PandocConverter().version  # "unknown" when the binary is absent


# -- PdfConverter -------------------------------------------------------------


def test_pdf_output_equals_doc_kg_call(tmp_path: Path) -> None:
    """Byte-equal to the bare call doc_kg.pdf_reader makes."""
    pdf = _pdf(tmp_path / "a.pdf", "Disulfide bonds stabilize proteins.")
    result = PdfConverter().convert(pdf)
    assert result.markdown == pymupdf4llm.to_markdown(str(pdf))
    assert "Disulfide" in result.markdown
    assert result.suffix == ".md"


def test_pdf_metadata(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "a.pdf", "Text layer present.", title="A Title: With Colon")
    meta = PdfConverter().convert(pdf).metadata
    assert meta["source_format"] == "pdf"
    assert meta["pages"] == "1"
    assert meta["title"] == "A Title: With Colon"
    assert meta["author"] == "Test Author"


def test_pdf_is_deterministic(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "a.pdf", "Same bytes, same Markdown.")
    assert PdfConverter().convert(pdf) == PdfConverter().convert(pdf)


def test_scanned_pdf_is_a_conversion_error(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "scan.pdf", None)
    with pytest.raises(ConversionError, match="OCR"):
        PdfConverter().convert(pdf)


def test_unparseable_pdf_is_a_conversion_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(ConversionError, match="could not parse"):
        PdfConverter().convert(bad)


def test_pipeline_records_a_scanned_pdf_with_a_reason(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _pdf(src / "scan.pdf", None)
    _pdf(src / "text.pdf", "Readable text.")
    stats = IngestPipeline(tmp_path / "staged").run([src])
    by_name = {Path(r.source_path).name: r for r in stats.records}
    assert by_name["text.pdf"].status == "ingested"
    assert by_name["text.pdf"].converter == "pymupdf4llm"
    assert by_name["scan.pdf"].status == "failed" and "OCR" in by_name["scan.pdf"].reason
    staged = (tmp_path / "staged" / by_name["text.pdf"].staged_path).read_text(encoding="utf-8")
    assert not staged.startswith("---")  # staging stays bare Markdown (no frontmatter)


# -- PandocConverter ----------------------------------------------------------


def test_missing_pandoc_is_a_conversion_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(converters_mod.shutil, "which", lambda _name: None)
    tex = tmp_path / "a.tex"
    tex.write_text("\\section{Hello}\nWorld.\n", encoding="utf-8")
    converter = PandocConverter()
    assert converter.version == "unknown"
    with pytest.raises(ConversionError, match="pandoc is not installed"):
        converter.convert(tex)


@needs_pandoc
def test_pandoc_converts_latex(tmp_path: Path) -> None:
    tex = tmp_path / "a.tex"
    tex.write_text(
        "\\documentclass{article}\\begin{document}\\section{Hello}World.\\end{document}\n",
        encoding="utf-8",
    )
    result = PandocConverter().convert(tex)
    assert "# Hello" in result.markdown and "World." in result.markdown
    assert result.metadata == {"source_format": "latex"}


@needs_pandoc
def test_pandoc_converts_html(tmp_path: Path) -> None:
    html = tmp_path / "a.html"
    html.write_text("<h1>Title</h1><p>Body <b>text</b>.</p>", encoding="utf-8")
    result = PandocConverter().convert(html)
    assert "# Title" in result.markdown and "**text**" in result.markdown
    assert result.metadata == {"source_format": "html"}


@needs_pandoc
def test_pandoc_empty_output_is_a_conversion_error(tmp_path: Path) -> None:
    html = tmp_path / "empty.html"
    html.write_text("<html><body></body></html>", encoding="utf-8")
    with pytest.raises(ConversionError, match="empty"):
        PandocConverter().convert(html)


# -- write_derived -------------------------------------------------------------


def _result() -> ConversionResult:
    return ConversionResult(
        markdown="# Body\n\nText.\n",
        converter="pymupdf4llm",
        converter_version="9.9.9",
        suffix=".md",
        metadata={"source_format": "pdf", "title": 'Has: colon and "quotes"', "pages": "3"},
    )


def _frontmatter(text: str) -> dict[str, str]:
    import json

    head = text.split("\n---\n", 1)[0].splitlines()[1:]
    return {k: json.loads(v) for k, v in (line.split(": ", 1) for line in head)}


def test_write_derived_frontmatter_schema(tmp_path: Path) -> None:
    root = tmp_path / "import"
    src = root / "papers" / "a.pdf"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"%PDF fake bytes")
    dest = write_derived(
        _result(),
        src,
        tmp_path / "RAW" / "a.md",
        source_root=root,
        ingested_at="2026-10-14T00:00:00Z",
        extra={"kind": "articles"},
    )
    text = dest.read_text(encoding="utf-8")
    fm = _frontmatter(text)
    assert list(fm)[: len(PROVENANCE_KEYS)] == list(PROVENANCE_KEYS)
    assert fm["source_path"] == "papers/a.pdf"  # relative, never absolute
    assert len(fm["source_sha256"]) == 64
    assert fm["converter_version"] == "9.9.9"
    assert fm["ingested_at"] == "2026-10-14T00:00:00Z"
    assert fm["title"] == 'Has: colon and "quotes"'
    assert fm["kind"] == "articles" and fm["pages"] == "3"
    assert text.endswith("# Body\n\nText.\n")


def test_write_derived_records_only_the_name_outside_the_root(tmp_path: Path) -> None:
    src = tmp_path / "elsewhere" / "a.pdf"
    src.parent.mkdir()
    src.write_bytes(b"x")
    dest = write_derived(_result(), src, tmp_path / "a.md", source_root=tmp_path / "root")
    assert _frontmatter(dest.read_text(encoding="utf-8"))["source_path"] == "a.pdf"


def test_write_derived_is_deterministic_apart_from_the_timestamp(tmp_path: Path) -> None:
    src = tmp_path / "a.pdf"
    src.write_bytes(b"same bytes")
    one = write_derived(_result(), src, tmp_path / "one.md", ingested_at="t1")
    two = write_derived(_result(), src, tmp_path / "two.md", ingested_at="t1")
    assert one.read_bytes() == two.read_bytes()


def test_write_derived_refuses_to_overwrite(tmp_path: Path) -> None:
    src = tmp_path / "a.pdf"
    src.write_bytes(b"x")
    dest = write_derived(_result(), src, tmp_path / "a.md")
    before = dest.read_bytes()
    with pytest.raises(FileExistsError):
        write_derived(_result(), src, dest)
    assert dest.read_bytes() == before


def test_write_derived_rejects_extra_that_shadows_provenance(tmp_path: Path) -> None:
    src = tmp_path / "a.pdf"
    src.write_bytes(b"x")
    with pytest.raises(ValueError, match="provenance"):
        write_derived(_result(), src, tmp_path / "a.md", extra={"converter": "forged"})
    assert not (tmp_path / "a.md").exists()
