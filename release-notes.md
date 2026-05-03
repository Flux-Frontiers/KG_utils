# Release Notes — v0.2.4

> Released: 2026-04-29

### Fixed

- `load_sentence_transformer` / `SentenceTransformerEmbedder`: call
  `hf_logging.disable_progress_bar()` in addition to `set_verbosity_error()`
  and `TQDM_DISABLE=1`. `TQDM_DISABLE` alone misses the `_tqdm_active` gate
  inside `transformers`, leaving progress bars visible in worker processes.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
