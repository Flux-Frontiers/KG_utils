# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Removed

### Fixed

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
