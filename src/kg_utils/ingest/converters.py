"""kg_utils/ingest/converters.py — Source document → Markdown conversion.

A :class:`Converter` turns one source file into Markdown text plus the
provenance needed to reproduce that conversion later.  Two are shipped:

``PassthroughConverter``
    Already-textual formats (``.md``, ``.txt``, ``.rst``).  The bytes are
    decoded, not transformed, and the original suffix is preserved so that
    downstream chunkers keep applying the right structural parse — a ``.txt``
    file must not become ``.md`` or it would suddenly be parsed for headings.

``PdfConverter``
    PDF, via ``pymupdf4llm``: the fleet's one PDF authority (sweep item 48).
    The call is the same bare ``pymupdf4llm.to_markdown(path)`` that
    ``doc_kg`` makes, so a PDF converts to the same Markdown whether it
    arrives through ``kgrag ingest`` or ``dockg build``.

``AnydocConverter``
    Office and publishing formats, via `anydoc <https://github.com/firecrawl/anydoc>`_
    (PyPI: ``firecrawl-anydoc``) — a Rust converter with Python bindings that
    emits consistent GitHub-Flavored Markdown for Word, PowerPoint, Excel,
    OpenDocument, RTF, EPUB and CSV.

``PandocConverter``
    LaTeX and HTML, via the ``pandoc`` command, which neither of the above
    reads.  Pandoc is a system binary, not a Python dependency: when it is
    absent the file is recorded as failed with that reason.

No converter performs OCR.  A scanned PDF converts to empty output, which
raises :class:`ConversionError`; the pipeline records that with a reason
rather than dropping the file silently.

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

#: Formats consumed as-is.  Mapped to themselves so the staged file keeps the
#: suffix the downstream chunker expects.
PASSTHROUGH_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".txt", ".rst"})

#: Extensions ``anydoc`` converts to Markdown.  Kept explicit rather than
#: probed at import time so that ``supported_extensions()`` answers without
#: requiring the optional dependency to be installed.
ANYDOC_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Word
        ".doc",
        ".docx",
        ".docm",
        # PowerPoint
        ".ppt",
        ".pps",
        ".pot",
        ".pptx",
        ".pptm",
        ".ppsx",
        ".ppsm",
        # Excel
        ".xls",
        ".xlsx",
        ".xlsm",
        ".xlsb",
        # OpenDocument
        ".odt",
        ".ods",
        ".odp",
        # Other
        ".rtf",
        ".epub",
        ".csv",
    }
)

#: Extensions :class:`PdfConverter` handles.
PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

#: Extensions :class:`PandocConverter` handles: formats no Python converter
#: here reads.
PANDOC_EXTENSIONS: frozenset[str] = frozenset({".tex", ".latex", ".html", ".htm"})


class ConversionError(RuntimeError):
    """Raised when a source file cannot be converted to Markdown.

    Carries a human-readable reason that is written verbatim into the ingest
    manifest, so a corpus always explains what it left out and why.
    """


@dataclass(frozen=True)
class ConversionResult:
    """Markdown extracted from one source file, with its provenance.

    :param markdown: The converted document text.
    :param converter: Name of the converter that produced it (e.g. ``"anydoc"``).
    :param converter_version: Version of the converting library, for reproducibility.
    :param suffix: Suffix the staged file must carry (``".md"`` for converted
                   documents, the original suffix for passthrough).
    :param metadata: Converter-specific extras (e.g. detected source format).
    """

    markdown: str
    converter: str
    converter_version: str
    suffix: str
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Converter(Protocol):
    """Protocol implemented by every source-format converter."""

    name: str

    @property
    def version(self) -> str:
        """Version of the underlying converting library.

        Readable without performing a conversion, so a *failed* conversion can
        still record which version rejected the file. Returns ``"unknown"``
        when the version cannot be determined.
        """
        ...

    def handles(self, path: Path) -> bool:
        """Return ``True`` if this converter can convert *path*.

        :param path: Candidate source file.
        :return: Whether :meth:`convert` should be called for this file.
        """
        ...

    def convert(self, path: Path) -> ConversionResult:
        """Convert *path* to Markdown.

        :param path: Source file to convert.
        :return: The extracted Markdown plus provenance.
        :raises ConversionError: If the file cannot be converted.
        """
        ...


#: Passthrough performs no transformation, so its "version" only needs to
#: change if the decoding policy in :meth:`PassthroughConverter.convert` changes.
_PASSTHROUGH_VERSION = "1"


class PassthroughConverter:
    """Decode already-textual sources without transforming them.

    Preserves the original suffix: ``.txt`` stays ``.txt`` so that DocKG's
    flat-text parse still applies, rather than being reinterpreted as Markdown.
    """

    name = "passthrough"

    def __init__(self, extensions: frozenset[str] | None = None) -> None:
        """:param extensions: Suffixes to accept (default: :data:`PASSTHROUGH_EXTENSIONS`)."""
        self.extensions = extensions if extensions is not None else PASSTHROUGH_EXTENSIONS

    @property
    def version(self) -> str:
        """Version of the passthrough decoding policy."""
        return _PASSTHROUGH_VERSION

    def handles(self, path: Path) -> bool:
        """Return ``True`` for suffixes in this converter's extension set."""
        return path.suffix.lower() in self.extensions

    def convert(self, path: Path) -> ConversionResult:
        """Read *path* as UTF-8, replacing undecodable bytes.

        Undecodable bytes are replaced rather than raising: a mostly-readable
        document is worth more to a corpus than a dropped one, and the
        substitution is visible in the staged output.

        :param path: Source file to read.
        :return: The file's text, with its suffix preserved.
        :raises ConversionError: If the file cannot be read.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ConversionError(f"cannot read {path.name}: {exc}") from exc
        except OSError as exc:
            raise ConversionError(f"cannot read {path.name}: {exc}") from exc

        suffix = path.suffix.lower()
        return ConversionResult(
            markdown=text,
            converter=self.name,
            converter_version=self.version,
            suffix=".md" if suffix == ".markdown" else suffix,
        )


class AnydocConverter:
    """Convert office and publishing formats to Markdown via ``anydoc``.

    The ``anydoc`` import is deferred to first use so that installing
    ``kgmodule-utils`` without the ``ingest`` extra stays viable, and so that
    merely listing supported extensions never requires the dependency.
    """

    name = "anydoc"

    def __init__(self, extensions: frozenset[str] | None = None) -> None:
        """:param extensions: Suffixes to accept (default: :data:`ANYDOC_EXTENSIONS`)."""
        self.extensions = extensions if extensions is not None else ANYDOC_EXTENSIONS
        self._module = None
        self._version = ""

    @property
    def version(self) -> str:
        """Installed ``firecrawl-anydoc`` version.

        Read from distribution metadata rather than the imported module, so it
        resolves without importing ``anydoc`` — and therefore still answers on
        the failure path, where the conversion never succeeded.
        """
        if not self._version:
            self._version = _installed_version("firecrawl-anydoc")
        return self._version

    def handles(self, path: Path) -> bool:
        """Return ``True`` for suffixes in this converter's extension set."""
        return path.suffix.lower() in self.extensions

    def _load(self):
        """Import and cache the ``anydoc`` module.

        :return: The imported ``anydoc`` module.
        :raises ConversionError: If the optional dependency is not installed.
        """
        if self._module is None:
            try:
                import anydoc  # pylint: disable=import-outside-toplevel
            except ImportError as exc:  # pragma: no cover — depends on install
                raise ConversionError(
                    "anydoc is not installed; install the ingest extra: "
                    "pip install 'kgmodule-utils[ingest]'"
                ) from exc
            self._module = anydoc
        return self._module

    def convert(self, path: Path) -> ConversionResult:
        """Convert *path* to Markdown with ``anydoc``.

        :param path: Source file to convert.
        :return: GitHub-Flavored Markdown plus provenance, staged as ``.md``.
        :raises ConversionError: If ``anydoc`` rejects the file — unsupported
            (including image-only PDFs, which need OCR), malformed, encrypted,
            or over a resource limit.
        """
        anydoc = self._load()
        try:
            markdown = anydoc.to_markdown(str(path))
        except anydoc.ConvertError as exc:
            raise ConversionError(f"{type(exc).__name__}: {exc}") from exc
        except OSError as exc:
            raise ConversionError(f"cannot read {path.name}: {exc}") from exc

        if not markdown.strip():
            # anydoc reports success for a text-layer-free PDF but returns
            # nothing usable. Surface it as a skip so the manifest explains the
            # gap and the file can be re-run through OCR later.
            raise ConversionError(
                "converted to empty output (likely a scanned/image-only document needing OCR)"
            )

        detected = anydoc.format_from_path(str(path)) or ""
        return ConversionResult(
            markdown=markdown,
            converter=self.name,
            converter_version=self.version,
            suffix=".md",
            metadata={"source_format": str(detected)} if detected else {},
        )


class PdfConverter:
    """Convert PDF to Markdown with ``pymupdf4llm``.

    The conversion is a bare ``pymupdf4llm.to_markdown(str(path))`` -- no page
    chunks, no anchors -- exactly what ``doc_kg.pdf_reader`` does, so the two
    paths produce byte-identical Markdown.  Page anchors are roadmap Phase 2.

    Title, author, creation date and page count are read from the PDF's own
    metadata into :attr:`ConversionResult.metadata` when present.
    """

    name = "pymupdf4llm"

    def __init__(self, extensions: frozenset[str] | None = None) -> None:
        """:param extensions: Suffixes to accept (default: :data:`PDF_EXTENSIONS`)."""
        self.extensions = extensions if extensions is not None else PDF_EXTENSIONS
        self._version = ""

    @property
    def version(self) -> str:
        """Installed ``pymupdf4llm`` version, read without importing it."""
        if not self._version:
            self._version = _installed_version("pymupdf4llm")
        return self._version

    def handles(self, path: Path) -> bool:
        """Return ``True`` for suffixes in this converter's extension set."""
        return path.suffix.lower() in self.extensions

    def convert(self, path: Path) -> ConversionResult:
        """Convert *path* to Markdown with ``pymupdf4llm``.

        :param path: PDF to convert.
        :return: Markdown plus provenance, staged as ``.md``.
        :raises ConversionError: If ``pymupdf4llm`` is not installed, cannot
            open or parse the file, or the file has no text layer.
        """
        try:
            import pymupdf  # pylint: disable=import-outside-toplevel
            import pymupdf4llm  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover — depends on install
            raise ConversionError(
                "pymupdf4llm is not installed; install the ingest extra: "
                "pip install 'kgmodule-utils[ingest]'"
            ) from exc
        try:
            markdown = pymupdf4llm.to_markdown(str(path))
        except Exception as exc:  # noqa: BLE001 — any parser failure is a recorded skip
            raise ConversionError(f"pymupdf4llm could not parse {path.name}: {exc}") from exc
        if not markdown.strip():
            raise ConversionError(
                "converted to empty output (likely a scanned/image-only document needing OCR)"
            )

        metadata = {"source_format": "pdf"}
        try:
            with pymupdf.open(str(path)) as doc:
                metadata["pages"] = str(doc.page_count)
                info = doc.metadata or {}
                for key, field_name in (
                    ("title", "title"),
                    ("author", "author"),
                    ("creationDate", "created"),
                ):
                    value = (info.get(key) or "").strip()
                    if value:
                        metadata[field_name] = value
        except Exception:  # noqa: BLE001 — metadata is optional; the text already converted
            pass
        return ConversionResult(
            markdown=markdown,
            converter=self.name,
            converter_version=self.version,
            suffix=".md",
            metadata=metadata,
        )


class PandocConverter:
    """Convert LaTeX and HTML to GitHub-Flavored Markdown with ``pandoc``.

    ``pandoc`` is found on ``PATH`` at conversion time.  It is a system
    binary rather than a Python dependency, so a missing ``pandoc`` is a
    recorded failure with an install hint, never a silent skip.
    """

    name = "pandoc"

    def __init__(self, extensions: frozenset[str] | None = None) -> None:
        """:param extensions: Suffixes to accept (default: :data:`PANDOC_EXTENSIONS`)."""
        self.extensions = extensions if extensions is not None else PANDOC_EXTENSIONS
        self._version = ""

    @property
    def version(self) -> str:
        """Version reported by ``pandoc --version``, or ``"unknown"``."""
        if not self._version:
            exe = shutil.which("pandoc")
            self._version = "unknown"
            if exe:
                try:
                    out = subprocess.run(
                        [exe, "--version"], capture_output=True, text=True, check=True, timeout=30
                    ).stdout
                    first = out.splitlines()[0] if out else ""
                    self._version = first.split()[-1] if first else "unknown"
                except (OSError, subprocess.SubprocessError):
                    pass
        return self._version

    def handles(self, path: Path) -> bool:
        """Return ``True`` for suffixes in this converter's extension set."""
        return path.suffix.lower() in self.extensions

    def convert(self, path: Path) -> ConversionResult:
        """Convert *path* to Markdown with ``pandoc -t gfm --wrap=none``.

        :param path: LaTeX or HTML file to convert.
        :return: Markdown plus provenance, staged as ``.md``.
        :raises ConversionError: If ``pandoc`` is not on ``PATH``, fails, or
            produces empty output.
        """
        exe = shutil.which("pandoc")
        if exe is None:
            raise ConversionError(
                "pandoc is not installed; install it (https://pandoc.org/installing.html, "
                "or `brew install pandoc`) to convert .tex and .html"
            )
        try:
            proc = subprocess.run(
                [exe, str(path), "-t", "gfm", "--wrap=none"],
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip().splitlines()
            raise ConversionError(
                f"pandoc failed on {path.name}: {detail[-1] if detail else exc}"
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise ConversionError(f"pandoc failed on {path.name}: {exc}") from exc
        if not proc.stdout.strip():
            raise ConversionError("converted to empty output")
        fmt = "latex" if path.suffix.lower() in (".tex", ".latex") else "html"
        return ConversionResult(
            markdown=proc.stdout,
            converter=self.name,
            converter_version=self.version,
            suffix=".md",
            metadata={"source_format": fmt},
        )


def _installed_version(distribution: str) -> str:
    """Return the installed version of *distribution*, or ``"unknown"``.

    :param distribution: PyPI distribution name.
    :return: Version string, or ``"unknown"`` if it cannot be determined.
    """
    try:
        from importlib.metadata import (  # pylint: disable=import-outside-toplevel
            PackageNotFoundError,
            version,
        )

        return version(distribution)
    except PackageNotFoundError:  # pragma: no cover — depends on install
        return "unknown"


def default_converters() -> list[Converter]:
    """Return the standard converter chain, most specific first.

    Passthrough is tried first so that Markdown and plain text are never
    round-tripped through a converter that would rewrite them.  The other
    three handle disjoint extension sets, so their order does not matter.

    :return: Converters in resolution order.
    """
    return [PassthroughConverter(), PdfConverter(), AnydocConverter(), PandocConverter()]


def resolve_converter(path: Path, converters: list[Converter]) -> Converter | None:
    """Return the first converter in *converters* that handles *path*.

    :param path: Candidate source file.
    :param converters: Converter chain, in resolution order.
    :return: The matching converter, or ``None`` if the format is unsupported.
    """
    for converter in converters:
        if converter.handles(path):
            return converter
    return None


def supported_extensions() -> frozenset[str]:
    """Return every suffix the default converter chain accepts.

    Answers without importing any converter library, so it is safe to call on
    a core-only install.  ``pandoc``'s formats are included whether or not the
    binary is installed; a missing binary is reported per file.

    :return: Lower-case suffixes including the leading dot.
    """
    return PASSTHROUGH_EXTENSIONS | PDF_EXTENSIONS | ANYDOC_EXTENSIONS | PANDOC_EXTENSIONS
