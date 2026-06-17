# Release Notes — v0.4.4

> Released: 2026-06-17

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

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
