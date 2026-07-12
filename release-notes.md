# Release Notes — v0.4.8

> Released: 2026-07-12

A focused follow-up to 0.4.7's `CorpusEmbedder` centralization: this release tames the
default memory footprint of parallel CPU embedding and gives the single-process path the
progress feedback it was missing. Both fixes were ported from doc_kg's `feat/embedderworker`
branch — landing them here means every consumer of the shared implementation gets them at
once, which is exactly why the engine was centralized.

## What changed

**Default worker count capped at 4.** `CorpusEmbedder` previously defaulted to
`cpu_count // 2` workers, unbounded. Each CPU worker loads its own full model copy plus a
torch runtime (~1.2 GB for bge-small, more for larger models), so on a 20-core machine the
old default spawned 10 workers and peaked around 21.5 GB RSS on a real 241-book corpus
build — well past the point where throughput stops improving, since large runs are I/O and
accumulator bound long before that. The default is now `min(4, cpu_count // 2)`; passing an
explicit `n_workers` behaves exactly as before.

**Progress bar on the single-process path.** Small corpora — and every `mps`/`cuda` run,
which always forces single-process — previously embedded silently. A new internal adapter
lets the sequential path drive the same per-batch progress reporting the parallel workers
use, so GPU runs now show live feedback instead of appearing hung.

## Upgrading

Nothing required. `pip install --upgrade kgmodule-utils`. If you were relying on the old
unbounded worker default for maximum parallelism on a many-core machine, pass
`n_workers` explicitly — the cap only applies to the default.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
