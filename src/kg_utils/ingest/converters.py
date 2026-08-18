"""kg_utils/ingest/converters.py — Source document → Markdown conversion.

A :class:`Converter` turns one source file into Markdown text plus the
provenance needed to reproduce that conversion later.  Two are shipped:

``PassthroughConverter``
    Already-textual formats (``.md``, ``.txt``, ``.rst``).  The bytes are
    decoded, not transformed, and the original suffix is preserved so that
    downstream chunkers keep applying the right structural parse — a ``.txt``
    file must not become ``.md`` or it would suddenly be parsed for headings.

``AnydocConverter``
    Everything else, via `anydoc <https://github.com/firecrawl/anydoc>`_
    (PyPI: ``firecrawl-anydoc``) — a Rust converter with Python bindings that
    emits consistent GitHub-Flavored Markdown for Word, PowerPoint, Excel,
    OpenDocument, RTF, EPUB, CSV and text-based PDF.

``anydoc`` performs no OCR.  Image-only PDFs raise
:class:`ConversionError`; the pipeline records that as a skip with a reason
rather than dropping the file silently.

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

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
        ".pdf",
    }
)


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

    Passthrough is tried before ``anydoc`` so that Markdown and plain text are
    never round-tripped through a converter that would rewrite them.

    :return: Converters in resolution order.
    """
    return [PassthroughConverter(), AnydocConverter()]


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

    Answers without importing ``anydoc``, so it is safe to call on a core-only
    install.

    :return: Lower-case suffixes including the leading dot.
    """
    return PASSTHROUGH_EXTENSIONS | ANYDOC_EXTENSIONS
