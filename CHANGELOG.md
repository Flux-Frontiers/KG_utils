# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Removed

### Fixed

## [0.2.3] - 2026-04-29

### Fixed

- `load_sentence_transformer`: removed save/restore logic around HF logging
  and TQDM state; now simply sets `TQDM_DISABLE=1` and `set_verbosity_error()`
  once and leaves them set, eliminating the mypy `[assignment]` error caused
  by the `Module | None` type mismatch on `_hf_logging`.
- **CI: mypy** — added `[[tool.mypy.overrides]]` for `sentence_transformers`,
  `transformers`, and `numpy` with `ignore_missing_imports = true`; added a
  separate override for `kg_utils.embedder` disabling `disallow_untyped_calls`
  so the `hf_logging.set_verbosity_error()` call requires no `type: ignore`
  regardless of whether `transformers` is installed.
- **CI: test** — marked all `sentence_transformers`-dependent tests in
  `tests/test_embedder.py` with `@pytest.mark.integration`; added
  `pytestmark = pytest.mark.integration` to `tests/test_integration.py`;
  updated CI test step to `pytest -m "not integration"` so these are skipped
  when the optional heavy deps are absent.
- **pytest.ini** — corrected stale `testpaths` (`./src/tests` → `./tests`);
  registered the `integration` marker to suppress unknown-mark warnings.
- **pylint** — added `[tool.pylint.main]` with `source-roots` and `init-hook`
  so pylint resolves the `src/` layout; added file-level disable in
  `tests/test_embedder.py` for pytest-pattern false positives
  (`redefined-outer-name`, `missing-function-docstring`, `too-few-public-methods`,
  `import-outside-toplevel`); fixed redundant `kg_utils` reimport in
  `test_doc_kg_re_exports_embedder_classes`.

## [0.2.2] - 2026-04-28

### Added

- `kg_utils.embedder` sub-package: concrete `SentenceTransformer` embedding
  implementation shared across all KGModule packages.
  - `Embedder` — abstract base with `embed_texts` + `embed_query` + `dim`.
  - `SentenceTransformerEmbedder` — concrete implementation with
    `local_files_only=True` guard on MPS to prevent SIGBUS on first `encode()`.
  - `load_sentence_transformer(model_name)` — canonical safe-load factory with
    four-step resolution: local path → HF cache → live network fetch.
  - `get_embedder(model_name)` — high-level factory returning a ready-to-use
    `SentenceTransformerEmbedder`.
  - `wrap_embedder(st_model, model_name)` — wraps a live `SentenceTransformer`
    as an `Embedder` to share a model across pipeline stages without reloading.
- Comprehensive test suite: `tests/test_embed.py`, `tests/test_embedder.py`,
  `tests/test_snapshots.py` (extended), `tests/test_types.py` (extended), and
  `tests/test_integration.py` covering cross-module protocol compliance, full
  snapshot lifecycle, subclass delta extensibility, and git subprocess
  integration.

### Fixed

- `SentenceTransformerEmbedder` and `wrap_embedder`: replaced direct
  `get_sentence_embedding_dimension()` call with a `getattr` fallback that
  tries `get_embedding_dimension` first (canonical in ST ≥ 5.4) then
  `get_sentence_embedding_dimension` (ST ≤ 5.3), eliminating the
  `FutureWarning` emitted by sentence-transformers ≥ 5.4.
- Aligned `sentence-transformers` minimum version to `>=5.4.1` in `code_kg`,
  `doc_kg`, and `diary_kg` so all KGModule packages resolve the same ST
  release and the `FutureWarning` cannot occur in any module.

## [0.2.1] - 2026-04-27

### Fixed

- `SnapshotManager._load_manifest`: extended legacy key normalization to handle
  manifest entries that carry a `commit` field instead of `tree_hash` or `key`,
  preventing `KeyError` when loading manifests written by older versions.
- `SnapshotManager.get_previous_snapshot` / `get_baseline`: replaced hard
  dict-key access (`entry["timestamp"]`, `entry["key"]`) with `.get()` calls
  and added explicit empty-key guards so malformed or legacy manifest entries
  no longer raise `KeyError` at runtime.
- Bumped `__version__` in `src/kg_utils/__init__.py` to track the package
  version (was stuck at `0.1.0`).

## [0.2.0] - 2026-04-26

### Added

- `kg_utils.embed` sub-package: shared embedding protocol and model-cache
  convention for the KGModule stack (stdlib-only, no external dependencies).
  - `Embedder` — `runtime_checkable` Protocol with `embed_query(text) -> list[float]`;
    KG modules and kgrag adapters type-hint against this without coupling to
    any concrete implementation.
  - `DEFAULT_MODEL` / `KNOWN_MODELS` — canonical default (`BAAI/bge-small-en-v1.5`)
    and short-alias mapping shared by all modules for consistent alias resolution.
  - `kg_model_cache_dir()` — returns `~/.kgrag/models/` by default; a single
    `KGRAG_MODEL_DIR` env-var redirects every module's cache simultaneously.
  - `resolve_model_path()` — converts a model name or alias to an absolute local
    cache path, with an optional per-module `local_fallback` for standalone use.
