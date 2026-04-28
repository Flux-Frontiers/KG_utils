# Release Notes — v0.2.2

> Released: 2026-04-28

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

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
