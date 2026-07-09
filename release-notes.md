# Release Notes — v0.4.6

> Released: 2026-07-09

This patch lowers the default embedding batch size so large builds no longer exhaust
memory on Apple Silicon and CPU. If you build knowledge graphs over corpora with long
text chunks, this is the difference between a build that completes and one that stalls
or is killed by the OS.

## What changed

**Bounded encode memory by default.** The per-call encode batch used inside
`Embedder.embed_texts`, `SentenceTransformerEmbedder.embed_texts`, and the `wrap_embedder`
wrapper now defaults to 128 instead of 512, backed by a new module constant
`DEFAULT_ENCODE_BATCH`. Transformer attention memory scales with `batch × sequence²`, so a
batch of 512 over near-max-length chunks can allocate many gigabytes per `model.encode`
call — enough to peak at 25–32 GB on a 528k-node build and stall or OOM the MPS backend.
Throughput on CPU and MPS is flat above ~128 for the models in use, so the smaller default
costs nothing in practice. The `wrap_embedder` path previously hardcoded `batch_size=512`
with no override; it now honors the same knob as everything else.

## Upgrading

Nothing is required — the safer default applies automatically. `embed_texts` now accepts a
uniform optional `encode_batch_size` parameter across the base class, the concrete
`SentenceTransformerEmbedder`, and the wrapped embedder. Raise it (e.g. back to 512) only if
you are running on a large-VRAM CUDA GPU with short sequences and want maximum throughput.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
