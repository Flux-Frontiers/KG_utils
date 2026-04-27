# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Removed

### Fixed

## [0.3.0] - 2026-04-27

### Added

- Incorporated production snapshot infrastructure from `kg-snapshot` v0.3.0
  into `kg_utils.snapshots`, making this the canonical home.
- `SnapshotManager.prune_snapshots(dry_run=False)` — three-pass cleanup:
  (1) metric-duplicate interior entries (unchanged vs. prior kept entry per
  `_metrics_changed`); (2) broken manifest entries with missing JSON files;
  (3) orphaned JSON files on disk not referenced by the manifest. Baseline
  and latest entries are always preserved. Returns `PruneResult`.
- `PruneResult` dataclass (`removed`, `orphaned_files`, `broken_entries`,
  `dry_run`, `total_cleaned`) fully exported from public API.
- `SnapshotManager._metrics_changed(new, old)` hook — subclasses override to
  define what constitutes a meaningful metric change (default: full equality).
- `save_snapshot(force=False)` dedup logic: if version and metrics are
  unchanged vs. the latest snapshot, the existing entry is refreshed in-place
  (tree hash, timestamp, branch updated; old JSON removed) instead of
  appending a new history entry.
- 12 new tests covering dedup behaviour (refresh-in-place, changed-metrics
  appends, `force=True` override, `_metrics_changed` subclass hook) and
  extended prune scenarios (baseline/latest protection, actual file removal,
  orphaned file detection, single-snapshot no-op, override respect).

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
