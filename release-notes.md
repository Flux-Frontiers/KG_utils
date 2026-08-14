# Release Notes — v0.13.0

> Released: 2026-08-14

This release adds `TEIEmbedder`, a client for remote HuggingFace Text Embeddings
Inference servers that runs on the standard library alone — no torch, no
sentence-transformers, not even numpy. It is the first embedder usable from a core,
zero-dependency install, because the model lives in the server process rather than
the client.

## What changed

**Remote embeddings without the ML stack.** `kg_utils.embedder.TEIEmbedder` speaks
TEI's native `/embed` API over stdlib HTTP, honours the fleet contract
(`normalize=True`, `truncate=True`), clamps request batches to the server's
advertised ceiling, and retries transient failures with backoff while failing loudly
on anything that could corrupt a vector store. Verified against TEI 1.9.3 serving
`BAAI/bge-small-en-v1.5`: cosine parity ≥ 0.999997 with the in-process
sentence-transformers backend, so the two can share one store. Its wins are memory
(176 MiB vs 1.5 GiB RSS for the same model) and keeping torch out of the client —
on CPU it is not a speed upgrade.

**Developer tooling actually installs now.** The checked-in pre-commit configuration
was not installable from the dev group, and its ruff had drifted six minor versions
from CI's — the classic recipe for hooks passing locally while CI fails. Both tools
are now dev dependencies, the hook rev matches the dev floor, and the ruff rule set
is pinned explicitly rather than inherited from ruff's shifting defaults. The
README's test instructions were also fixed: the documented install could not even
collect the suite; it now describes what CI actually installs.

## Upgrading

Nothing to do — every change is additive or tooling-side, with no API, signature, or
default changes. To try the new embedder, point it at a running TEI server via
`TEIEmbedder(endpoint=...)` or the `KG_EMBED_ENDPOINT` / `KG_EMBED_API_KEY`
environment variables.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
