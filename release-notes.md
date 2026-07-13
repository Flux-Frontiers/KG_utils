# Release Notes — v0.4.9

> Released: 2026-07-13

This release closes out the corpus-embedding memory saga: `CorpusEmbedder` can now stream
vectors to disk as shards complete, so peak memory is bounded by shard size instead of
corpus size. The 689k-node Gutenberg consolidated build — which previously drove the parent
process past 10 GB RSS and the machine into 45 GB of swap before being killed — can run
single-pass on CPU again, retiring the per-genre build workaround.

## What changed

**Streaming embed-to-cache.** The new `embed_to_cache(texts, metadata, *, out_path)` method
is the streaming counterpart to `embed()`. In parallel mode, each spawn worker writes its
shard's rows batch-by-batch to a temp part file next to the output and returns the file
path — never the vectors — and the parent merges parts in shard order behind a `__meta__`
header once the pool drains. Output order exactly matches input order, and part files are
cleaned up on failure. The single-process path (small inputs, or any GPU run) streams
through the same worker with the usual progress bar.

**Drop-in cache format.** The output is the JSONL cache format doc_kg's `build_from_cache`
already reads — one row per text with `id, kind, name, title, file_path, text, vector`,
extra metadata keys preserved, gzip via a `.gz` suffix — so downstream index-from-cache
paths consume it without changes.

**Everything load-bearing is preserved.** The GPU→single-process guard, worker recycling
(`maxtasksperchild=1` + shard cadence), sequential fallback on pool failure, and the
embedding results themselves are unchanged — a 2-worker spawn smoke test confirmed vectors
bit-identical to the in-memory `embed()` path. `embed()` itself is untouched for callers
that want an in-memory `EmbeddingCache`.

## Upgrading

Nothing required; `embed()` behaves exactly as before. To take advantage of bounded memory
on large corpora, switch full-corpus builds to
`embedder.embed_to_cache(texts, metadata, out_path=Path("embeddings.jsonl"))` and feed the
resulting file to your existing index-from-cache path. Downstream wiring for gutenberg_kg's
`build-corpus` and doc_kg's `precompute_embeddings` lands separately.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
