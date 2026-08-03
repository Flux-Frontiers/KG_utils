# Release Notes — v0.9.0

> Released: 2026-07-28

A security-driven release: kgmodule-utils moves to **transformers 5**. The old
`transformers<4.57` cap pinned every install to 4.56.2, which sits below two open
high-severity advisories — a remote-code-execution flaw (fixed in 5.3.0) and an
arbitrary-code-execution flaw in the LightGlue model-loading path (fixed in 5.5.0). The new
`>=5.5.0,<6` floor clears both. The move was validated to be behaviourally inert: embeddings
are bitwise identical, a rebuilt index is byte-for-byte the same, and existing indexes query
identically — **no re-index is required.**

## What changed

**transformers unpinned to `>=5.5.0,<6` (security).** The `<4.57` cap had no recorded
rationale and no longer matched the fleet — doc-kg had already shipped on transformers 5.6.2.
Before moving the floor, the upgrade was checked against transformers 5.14.1 with the rest of
the stack unchanged: embeddings came out bitwise identical on bge-small, bge-large, and
nomic-embed (including empty, whitespace, 3k-character, unicode, and CRLF inputs); a full
index rebuild produced a byte-identical `vectors.sqlite`; and queries against an index built
on 4.56.2 returned identical rankings. So the security fix costs you nothing at build or
query time.

**Progress bars stop leaking into builds and queries (bug fix).** transformers 5 removed the
`transformers.logging` submodule alias, so the suppression code silently hit
`ModuleNotFoundError` — swallowed by a bare `except` — and stopped muting Hugging Face's
output, leaking a "Loading weights" bar into every build and query under transformers 5. The
import now targets `transformers.utils.logging`, which resolves on both 4.x and 5.x, so the
suppression works again.

**Sibling KG packages no longer block dependency resolution (packaging).** The optional
`kgdeps` Poetry group declared `doc-kg` and `pycode-kg`, which each depend on the other. Since
Poetry locks optional groups too, relaxing the transformers pin deadlocked resolution against
the published siblings — neither could lock until the other released. Since nothing here
imports those packages, the dev-only group is gone (with by-hand install instructions left in
`pyproject.toml`), permanently breaking the cycle. The lock now resolves to transformers
5.14.1, huggingface-hub 1.25.1, and safetensors 0.8.0.

## Upgrading

**No re-index, no code change.** Rebuilding is optional — existing indexes keep working and
return identical results.

The one thing to know: **the `semantic` extra now requires transformers 5.** If your
environment is pinned to transformers 4.x, that pin must move to `>=5.5.0,<6` (bringing
huggingface-hub 1.x and safetensors 0.8.x with it). There is no API or embedding-output
change on this side.

If you previously relied on the `kgdeps` extra to pull the sibling KG packages, install them
directly instead:

```bash
pip install doc-kg pycode-kg
pip install 'kg-rag @ git+https://github.com/Flux-Frontiers/KGRAG.git'
pip install 'agent-kg @ git+https://github.com/Flux-Frontiers/agent_kg.git'
```

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
