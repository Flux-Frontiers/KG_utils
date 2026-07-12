# Release Notes — v0.4.7

> Released: 2026-07-11

This release ends the era of forked embedding engines: the spawn-safe, multi-worker corpus
embedder that doc_kg, memory_kg, and diary_kg each carried as an independent copy now lives
in kg_utils as `CorpusEmbedder`, with every hard-won production fix included. Downstream
modules should delete their local copies and import from here.

## What changed

**Centralized corpus embedding engine.** The new `kg_utils.corpus_embedder` module provides
`CorpusEmbedder` and `EmbeddingCache` as the canonical implementation of parallel corpus
embedding. It carries forward the fixes proven in doc_kg after the June 683k-node OOM
incident: a GPU→single-process guard (never fan out to spawn workers when the resolved
device is MPS or CUDA), shard recycling so long-lived workers don't accumulate allocator
state, gzip cache support, and per-batch progress reporting. The same bug had already
resurfaced unfixed in a sibling module's copy — with one shared implementation, a fix lands
once.

**Public device resolution.** `kg_utils.embedder.resolve_device()` exposes the device
precedence logic (explicit argument > `KG_EMBED_DEVICE` > auto-detect) that
`load_sentence_transformer` has used internally since 0.4.4, so callers can gate decisions
on the resolved device — such as `CorpusEmbedder`'s parallel-vs-single-process choice —
without loading a model first.

**`kg_utils.semantic` unified onto the shared embedder stack.** `semantic.py` had drifted
into a fourth independent embedding implementation, with no device awareness at all. Its
model registry and embedder classes are now re-exports from `kg_utils.embed` and
`kg_utils.embedder`, giving its consumers (pycode_kg) `KG_EMBED_DEVICE` support for free.
This also tightens security: `trust_remote_code=True` is no longer passed unconditionally
to every model load — it is gated to the known-safe `nomic-ai/` model family. The private
`_local_model_path` helper that pycode_kg imports directly is preserved as a
backward-compatible wrapper, so its on-disk model cache location doesn't move.

## Upgrading

No breaking changes. `pip install --upgrade kgmodule-utils` (the `semantic` extra now pulls
in `rich`, used by the parallel progress bar). Modules maintaining their own `CorpusEmbedder`
fork should migrate to `from kg_utils.corpus_embedder import CorpusEmbedder` and delete the
local copy. If you relied on `semantic.SentenceTransformerEmbedder` loading arbitrary models
with `trust_remote_code=True`, that now only applies to `nomic-ai/` models.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
