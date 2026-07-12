# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`kg_utils.corpus_embedder.CorpusEmbedder` / `EmbeddingCache`.** Canonical home for the
  spawn-safe, multi-worker corpus embedding engine that had been independently forked at
  least three times (doc_kg, memory_kg, diary_kg) — most recently causing a real production
  incident (a 683k-node consolidated build OOM'd on Apple Silicon; see
  `gutenberg_kg/SUMMARY.md`, 2026-06-16/17) that had to be root-caused and fixed in doc_kg
  before the same bug resurfaced, unfixed, in memory_kg's independent copy. Carries forward
  doc_kg's proven fixes: a GPU→single-process guard (`embed()` never fans out to parallel
  workers when the resolved device is `mps`/`cuda` — a GPU allocator can't be shared across
  `spawn` workers, so N workers would stack N allocations into an OOM), shard recycling
  (`_RECYCLE_SHARD=25_000` + `Pool(maxtasksperchild=1)`, so long-lived workers don't
  accumulate allocator/heap/GC state across a large run), gzip cache support, and per-batch
  progress reporting. Downstream modules should import `CorpusEmbedder` from here instead of
  keeping their own copy.
- **`kg_utils.embedder.resolve_device(device=None)`.** Public device-resolution helper
  (explicit arg > `KG_EMBED_DEVICE` env > auto-detect), extracted from the logic
  `load_sentence_transformer` already had inline since 0.4.4. Exists so callers that need to
  gate a *decision* (e.g. `CorpusEmbedder`'s parallel-vs-single-process fan-out) on the
  resolved device can do so without loading a model first. `load_sentence_transformer` now
  calls this instead of duplicating the precedence logic.
- `semantic` extra now includes `rich>=13.0.0` (used by `CorpusEmbedder`'s parallel progress
  bar).

## [0.4.6] - 2026-07-09

### Changed

- **`embedder`: default per-call encode batch lowered 512 → 128.** New module constant `DEFAULT_ENCODE_BATCH = 128` now backs `Embedder.embed_texts`, `SentenceTransformerEmbedder.embed_texts`, and the `wrap_embedder` `_WrappedEmbedder.embed_texts` (which previously **hardcoded** `batch_size=512` with no way to override). Transformer attention memory scales with `batch × seq²`, so a large batch on long (near-max-sequence) chunks allocates many GB per `model.encode` call and OOMs / stalls MPS — observed as a 25–32 GB peak on a 528k-node build in a downstream module. Throughput is flat above ~128 on CPU and MPS for the models in use, so this is free; raise `encode_batch_size` only for a large-VRAM CUDA GPU with short sequences. `embed_texts` now takes a uniform optional `encode_batch_size` parameter across the base class, concrete, and wrapped implementations.

## [0.4.5] - 2026-07-07

### Added

- **`TextSynthesizer.complete(messages, *, model=None, max_tokens=None, temperature=0.7)`** —
  public general-purpose chat-completion entry point for callers that build their own message
  list (summarization, classification, arbitrary prompting) instead of using `synthesize_rag`
  or `rewrite_for_image`. Applies the same oMLX thinking suppression and `<think>` stripping as
  the other public methods. Promotes the previously private `_complete` to stable public API so
  downstream KG modules can drive oMLX/Ollama/OpenAI backends without reaching into internals.

## [0.4.4] - 2026-06-17

### Added

- **`load_sentence_transformer(model_name, device=...)`** — explicit device override with
  precedence: explicit arg > `KG_EMBED_DEVICE` env > CUDA→MPS→CPU auto-detect. The env channel
  lets spawn-based embedding workers (which inherit `os.environ` but can't easily take a Python
  arg) be pinned to a device — without it, N parallel workers each auto-select MPS and stack N
  GPU allocations into an OOM. This is what makes CPU multiprocessing embedding safe on Apple
  Silicon.

### Changed

- **`embedder.py`** — replaced `from X import Y` lazy imports with `importlib.import_module()`
  for `sentence_transformers`, `transformers.logging`, `torch`, and `numpy`.  `importlib` returns
  `Any`, so `ty` no longer flags these optional heavy dependencies as unresolved imports.

- **`synthesis/_image.py`** — same `importlib.import_module()` pattern for the `mflux` loader;
  removes the old `# type: ignore` override which is no longer needed.

### Fixed

- **CI `type-check` and `test` jobs** — both jobs now install `--extras "semantic" --extras
  "synthesis"` so that `sentence-transformers`, `transformers`, `torch`, `lancedb`, `httpx`,
  `openai`, and `pillow` are present in the CI virtualenv, matching local pre-commit behaviour.

- **`tests/test_synthesis_image.py`** — corrected four test assertions that still referenced
  the old `dall-e-3` default:
  - expected model updated from `dall-e-3` → `gpt-image-1`
  - landscape size updated from `1792x1024` → `1536x1024`
  - portrait size updated from `1024x1792` → `1024x1536`
  - `test_generate_openai_requests_b64_json` renamed to `test_generate_openai_does_not_set_response_format`
    and now asserts that `response_format` is absent from the OpenAI call kwargs (gpt-image-1
    does not accept this parameter)

## [0.4.3] - 2026-06-08

### Changed

- **`embedder.py`** — replaced `from X import Y` lazy imports with `importlib.import_module()`
  for `sentence_transformers`, `transformers.logging`, `torch`, and `numpy`.  `importlib` returns
  `Any`, so `ty` no longer flags these optional heavy dependencies as unresolved imports.

- **`synthesis/_image.py`** — same `importlib.import_module()` pattern for the `mflux` loader;
  removes the old `# type: ignore` override which is no longer needed.

### Fixed

- **CI `type-check` and `test` jobs** — both jobs now install `--extras "semantic" --extras
  "synthesis"` so that `sentence-transformers`, `transformers`, `torch`, `lancedb`, `httpx`,
  `openai`, and `pillow` are present in the CI virtualenv, matching local pre-commit behaviour.

- **`tests/test_synthesis_image.py`** — corrected four test assertions that still referenced
  the old `dall-e-3` default:
  - expected model updated from `dall-e-3` → `gpt-image-1`
  - landscape size updated from `1792x1024` → `1536x1024`
  - portrait size updated from `1024x1792` → `1024x1536`
  - `test_generate_openai_requests_b64_json` renamed to `test_generate_openai_does_not_set_response_format`
    and now asserts that `response_format` is absent from the OpenAI call kwargs (gpt-image-1
    does not accept this parameter)

## [0.4.3] - 2026-06-08

### Added

- **`_parse_size(size)`** — new helper in `kg_utils.synthesis._image` that parses an explicit
  `"WIDTHxHEIGHT"` string into a `(width, height)` tuple; returns `None` for invalid input.

- **`size` parameter on `ImageSynthesizer.generate()` and `generate_b64()`** — mflux backends
  (`mflux-local`, `mflux-serve`) now accept an explicit `"WIDTHxHEIGHT"` size override that
  takes priority over the aspect-ratio lookup table.  OpenAI backends ignore the parameter
  (they accept only a fixed set of sizes).

- **`size` parameter on `WorkerClient.imagine()`** — the RunPod `/runsync` payload now includes
  `size` when provided, enabling callers to pass pixel dimensions to mflux workers.

- **`size` handling in `handle_aux_ops`** (`kg_utils.worker.ops`) — `size` is extracted from
  the worker input dict and forwarded to `generate_b64()`; when present it is also included in
  the success response payload.

## [0.4.2] - 2026-06-08

### Added

- **`kg_utils.retrieval`** — new sub-package for shared retrieval helpers:
  - `hit_to_dict(hit, include_diary_timestamp)` — serializes a KGRAG hit object into a
    plain dictionary; optionally includes a `timestamp` field for diary-kind hits.
  - `attach_content_by_sqlite(hits, kg_sqlite_map)` — batched SQLite lookups that hydrate
    `content` on hit dicts in-place; missing or unreadable databases are silently skipped.

- **`kg_utils.worker`** — new sub-package centralizing RunPod `/runsync` protocol helpers:
  - `WorkerClient` — small HTTP client wrapping `list_models`, `rewrite`, `imagine`, and
    `query` operations with per-call `httpx.Timeout` tuning.
  - `WorkerError` — application-level error raised on structured worker failure payloads.
  - `decode_worker_response` / `extract_worker_error` — decode and surface RunPod error
    payloads in both `status: FAILED` and soft `output.error` forms.
  - `handle_aux_ops` — shared handler dispatch for `models`, `rewrite`, and `imagine`
    operations; eliminates duplicated logic across Streamlit worker handlers.

- **`kg_utils.synthesis.factory`** — synthesis backend factory helpers for per-request
  backend overrides, exported via `kg_utils.synthesis`:
  - `normalize_openai_base_url(endpoint)` — normalizes an endpoint string to end with `/v1`.
  - `text_synth_for_backend(backend, fallback)` — constructs a `TextSynthesizer` for the
    requested backend using env vars (`SYNTH_ENDPOINT`, `VLLM_*`, `OLLAMA_ENDPOINT`,
    `OPENAI_API_KEY`); returns `fallback` for unknown or empty values.
  - `image_synth_for_backend(backend, fallback)` — constructs an `ImageSynthesizer` for
    `openai`, `mflux-serve`, or `mflux-local` backends from env vars; returns `fallback`
    for unknown or empty values.

### Changed

- **`.gitignore`** — exclude `.claude/` project memory and settings directories.

## [0.4.1] - 2026-06-08

### Changed

- **`ImageBackend.OPENAI` default model** — upgraded from `dall-e-3` to `gpt-image-1`.
  `gpt-image-1` produces higher-quality images and supports portrait/landscape at
  1024×1536 (vs. DALL-E 3's 1792-wide variants).  Override with `IMAGE_MODEL=dall-e-3`
  to restore the previous behaviour.

### Fixed

- **`_generate_openai` size routing** — added `_GPT_IMAGE_SIZES` table for `gpt-image-1`
  (1024×1536 portrait/landscape); `_generate_openai` now selects the correct size table
  based on the model prefix (`gpt-image` vs. `dall-e`).
- **`response_format` removed from `gpt-image-1` calls** — `gpt-image-1` returns
  `b64_json` by default and does not accept the `response_format` parameter.  The
  `dall-e-3` fallback path now downloads via URL when `b64_json` is absent.
- **`docs/synthesis.md`** — annotated example API-key placeholder as a known
  false positive; regenerated `.secrets.baseline`.

## [0.4.0] - 2026-06-07

### Added

- **`kg_utils.synthesis`** — new sub-package providing unified text and image synthesis
  across six backends with a single, env-var-configurable API.

  **Text backends** (all use the OpenAI wire protocol):
  - `TextBackend.OMLX` — local oMLX / vLLM; MLX chain-of-thought suppressed via
    `extra_body` and `<think>` block stripping.  Default model:
    `Qwen3-4B-Instruct-2507-MLX-8bit`.
  - `TextBackend.OLLAMA` — local Ollama; no API key required.  Default model:
    `hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0`.
  - `TextBackend.OPENAI` — OpenAI cloud.  Default model: `gpt-4o-mini`.

  **Image backends:**
  - `ImageBackend.MFLUX_LOCAL` — in-process Flux2Klein via `mflux` (Apple Silicon);
    per-instance model cache avoids reloading across calls.
  - `ImageBackend.MFLUX_SERVE` — HTTP proxy to a running `mflux-serve` instance.
  - `ImageBackend.OPENAI` — DALL-E 3 with aspect-ratio → size mapping.

  **Public API surface:**
  - `TextConfig` / `ImageConfig` dataclasses with `resolved_endpoint()` and
    `resolved_model()` helpers.
  - `TextSynthesizer.list_models()` — available models at the endpoint.
  - `TextSynthesizer.synthesize_rag(query, snippets)` — grounded RAG answer; skips
    whitespace-only snippets; `max_k` cap; optional system prompt override.
  - `TextSynthesizer.rewrite_for_image(corpus_text)` — rewrites historical prose into
    an image-generation prompt; returns `(prompt, error)` — never raises.
  - `ImageSynthesizer.generate()` → PIL Image; `generate_b64()` → base64 PNG.
  - `text_config_from_env()` / `image_config_from_env()` — build configs from
    `SYNTH_*` / `IMAGE_*` env vars; honour legacy `VLLM_*` and `GUTENKG_IMAGE_MODEL`
    aliases with no migration required.
  - `text_synthesizer_from_env()` / `image_synthesizer_from_env()` — one-call
    convenience factories.

- **`[synthesis]` optional extra** — `httpx>=0.27.0`, `openai>=1.30.0`,
  `pillow>=10.0.0`.
- **`[synthesis-mflux]` optional extra** — all of `[synthesis]` plus `mflux>=0.9.0`.
- **Test suite — three new files** (116 tests total, stdlib + mocks only):
  - `tests/test_synthesis_config.py` (44 tests) — all config defaults and env-var
    priority chains; `clean_synth` / `clean_image` fixtures scrub env state.
  - `tests/test_synthesis_text.py` (38 tests) — `TextSynthesizer` with mocked
    `openai.OpenAI`; `<think>` stripping; empty-content filter; `rewrite_for_image`
    fallback behaviour.
  - `tests/test_synthesis_image.py` (34 tests) — `ImageSynthesizer` with mocked
    `httpx.post` (mflux-serve) and `_load_mflux` (local); DALL-E size mapping;
    base64 round-trip with a real 4×4 PNG.
- **`docs/synthesis.md`** — full reference document: env vars, backend defaults,
  API tables, DALL-E / mflux size maps, usage patterns, and integration notes.

### Changed

- **`mypy` → `ty`** throughout:
  - `pyproject.toml`: removed `[tool.mypy]` and both `[[tool.mypy.overrides]]` sections;
    added `ty = ">=0.0.41"` to dev group; added `[tool.ty.environment]` and
    `[tool.ty.rules]` (`unresolved-import = "ignore"`).
  - `.github/workflows/ci.yml`: `poetry run mypy src/` → `poetry run ty check src/`.
  - `.pre-commit-config.yaml`: `mypy` local hook → `ty` local hook,
    `entry: poetry run ty check src/`.
- **README** — version badge bumped to 0.4.0; synthesis added to Features, Installation,
  API Reference, and project structure tree.

## [0.3.1] - 2026-05-23

### Changed

- **Version bump** `0.3.0` → `0.3.1`.
- **README** — complete rewrite to reflect the 0.3.x expanded scope: updated
  version badge, description, feature list, Quick Start examples (now using
  correct `kg_utils.specs` / `kg_utils.extractor` / `kg_utils.pipeline`
  import paths), API reference tables for all new modules, revised project
  structure tree, and split test-suite instructions into fast vs. integration
  runs.
- **`tests/test_types.py`** — updated module docstring and import to use
  `kg_utils.specs` and `kg_utils.extractor`; removed stale KGModule tests
  (superseded by `test_pipeline_module.py`) and KGExtractor "raises
  NotImplementedError" tests (now enforced by ABC); rewrote config tests to
  use `DummyExtractor` instead of `KGExtractor.__new__`.
- **`tests/test_integration.py`** — replaced `from kg_utils.types import …`
  with imports from `kg_utils.extractor`, `kg_utils.pipeline`, and
  `kg_utils.specs`; added `analyze()` implementation and `_default_dir` to
  `_FileTreeModule` (required by the ABC and concrete `KGModule` base).

### Removed

- **`kg_utils.types` subpackage** (`types/__init__.py`, `types/specs.py`,
  `types/extractor.py`, `types/module.py`) — the parallel thin/abstract
  hierarchy has been eliminated.  All types now live at the canonical
  top-level locations: `kg_utils.specs`, `kg_utils.extractor`,
  `kg_utils.pipeline`.  This removes the dual-class friction where
  `isinstance` checks and type annotations could silently diverge depending
  on which import path was used.

## [0.3.0] - 2026-05-23

### Added

- **`kg_utils.specs`** — rich `NodeSpec` and `EdgeSpec` dataclasses with
  `lineno`, `end_lineno`, `metadata` fields; `BuildStats`, `QueryResult`, and
  `SnippetPack` return types for the full pipeline surface.
- **`kg_utils.extractor`** — `KGExtractor` abstract base class (yields
  `NodeSpec` / `EdgeSpec` iterators); domain authors subclass this to feed any
  source into the pipeline.
- **`kg_utils.store`** — `GraphStore`: SQLite-backed authoritative node/edge
  store with upsert, BFS expand, symbol resolution (`resolve_symbols`),
  caller lookup (`callers_of`), provenance recording, and a `ProvMeta`
  typed-dict.  SQLite is the single source of truth; the vector index is
  always derived from it.
- **`kg_utils.semantic`** — `SemanticIndex`: LanceDB vector index built from
  `GraphStore` nodes.  Includes `Embedder` abstract base,
  `SentenceTransformerEmbedder` (with ST ≥ 5.4 / ≤ 5.3 API fallback),
  `SeedHit` result dataclass, model registry (`_KNOWN_MODELS`), and
  `resolve_model_path` / `suppress_ingestion_logging` utilities.
- **`kg_utils.pipeline`** — `KGModule`: concrete abstract base class with the
  complete build → query → pack pipeline.  Domain authors implement only
  `make_extractor()`, `kind()`, and `analyze()`.  Provides hybrid
  semantic + lexical reranking, BFS graph expansion, configurable hop depth,
  `min_score` filtering, `max_nodes` capping, and snippet extraction with
  context lines.
- **`kg_utils.module`** — thin re-export shim providing `KGModule` and
  `KGExtractor` from a single import path for downstream compatibility.
- **`[semantic]` optional extra** in `pyproject.toml`: `lancedb>=0.19.0`,
  `numpy>=1.24.0`, `sentence-transformers>=5.4.1`, `torch>=2.5.1`,
  `transformers>=4.40.0,<4.57`.  Install with
  `pip install 'kgmodule-utils[semantic]'`.
- **`[kgdeps]` Poetry group** (optional): `pycode-kg>=0.18.1`,
  `doc-kg>=0.15.2` for integration testing against real KG modules.
- **`poetry.toml`** — local venv configuration (`in-project = true`).
- **`.pycodekg/snapshots/`** — initial CodeKG snapshot and manifest tracked
  for reproducible metrics across releases.
- **Test suite — three new files:**
  - `tests/test_store.py` (343 lines) — unit tests for `GraphStore`: write/read,
    edges, wipe, upsert, `query_nodes`, BFS expand, provenance, `resolve_symbols`,
    `callers_of`, `edges_from`, stats, and context-manager lifecycle.
  - `tests/test_pipeline_utils.py` (295 lines) — pure-function unit tests for
    all pipeline utilities: `semantic_score_from_distance`, `query_tokens`,
    `normalize_query_text`, `docstring_signal`, `lexical_overlap_score`,
    `safe_join`, `read_lines`, `compute_span`, `make_snippet`,
    `make_module_summary`, `spans_overlap`.
  - `tests/test_pipeline_module.py` (318 lines, `@pytest.mark.integration`) —
    end-to-end integration tests for a concrete `KGModule` implementation
    (`_TextKG` / `_TextExtractor`): `build_graph`, `build_index`, `stats`,
    `query` (semantic match, hop=0, hybrid rerank, `min_score`, `max_nodes`),
    `pack` (markdown, JSON, snippet text, key stripping), and lazy property
    initialisation.

### Changed

- **Version bump** `0.2.4` → `0.3.0` (significant new surface area).
- **Development status** classifier `3 - Alpha` → `4 - Beta`.
- **Package description** updated to reflect the expanded scope: "Shared
  types, graph store, semantic index, and pipeline base for the KGModule SDK".
- **`src/kg_utils/__init__.py`** — updated module docstring to document all
  new sub-modules and the `[semantic]` extra install path.
- **`lancedb` mypy override** added to `[[tool.mypy.overrides]]`
  `ignore_missing_imports` list so mypy strict mode passes without stubs.
- **`.gitignore`** — added exclusion rules for transient `.pycodekg/` and
  `.dockg/` artifacts (SQLite databases, LanceDB dirs, model caches) while
  keeping `snapshots/` tracked.
- **`.secrets.baseline`** regenerated to whitelist SHA git-tree hashes in
  `.pycodekg/snapshots/` that `detect-secrets` flags as `HexHighEntropyString`
  false positives.

### Fixed

- **`kg_utils.embedder`** — `load_sentence_transformer` and
  `SentenceTransformerEmbedder.__init__` now catch `(ImportError, ValueError)`
  instead of `ImportError` alone when suppressing HF logging, preventing an
  unhandled `ValueError` raised by some `transformers` versions when the
  logging backend is already initialised.

## [0.2.4] - 2026-04-29

### Fixed

- `load_sentence_transformer` / `SentenceTransformerEmbedder`: call
  `hf_logging.disable_progress_bar()` in addition to `set_verbosity_error()`
  and `TQDM_DISABLE=1`. `TQDM_DISABLE` alone misses the `_tqdm_active` gate
  inside `transformers`, leaving progress bars visible in worker processes.

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
