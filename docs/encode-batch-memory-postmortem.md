# Postmortem: Embedding OOM / MPS Stall Across the KG Module Stack

**Date:** 2026-07-09
**Scope:** `kg_utils`, `memory_kg`, `doc_kg`, `diary_kg`, `gutenberg_kg`
**Severity:** High — out-of-memory on large ingests, MPS stalls mid-build, latent risk masked across every KG module.
**Status:** Resolved. Fixed at the shared root (`kg_utils` 0.4.6, on PyPI) and at the one live exposure (`doc_kg`); every module pinned to `kgmodule-utils>=0.4.6`.

---

## TL;DR

A single oversized default — the per-call `model.encode(batch_size=…)` (**1024** in `doc_kg`, **512** in `kg_utils`) — caused transformer **attention buffers to balloon to 7–9 GB per call** on long chunks. Peak RSS hit **25 GB (CPU) / 32 GB (MPS)** on a 528k-node build, and MPS **stalled around 230k rows**. The batch bought nothing: throughput is **flat above ~128** on both CPU and MPS for the models in use.

Fix: cap the encode batch at **128** (`kg_utils` default + a `doc_kg` hard cap). Result: **peak RSS 25–32 GB → ~4 GB, flat**, same wall-clock, retrieval recall unchanged.

---

## The invariant

Peak memory for one `model.encode` call scales as:

```
peak ≈ batch_size × sequence_length² × (model hidden/heads/layers)
```

The `seq²` term is attention. The key consequence: **the batch size and the chunk (sequence) length multiply**. A large batch is safe on short chunks and catastrophic on long ones — which is exactly why this bug hid for so long (see "Why it was latent" below).

---

## Symptoms

Observed on a `memory_kg` LongMemEval ingest (528,083 nodes, `--chunk-strategy heading`, `bge-small-en-v1.5`, Apple M5 Max / 64 GB):

- **CPU run:** peak RSS **25 GB**, oscillating 11–25 GB; ETA growing.
- **MPS run:** `vmmap` peak **32.7 GB**; main thread wedged in `tensor.cpu()` → `MPSStream::synchronize` → `[_MTLCommandBuffer waitUntilCompleted]` → `__psynch_cvwait` — i.e. a hard GPU sync stall at **~230k rows**.
- Throughput degraded over the run; the process appeared to "stall."

---

## The diagnostic journey (including the false leads)

We chased **several wrong causes** before finding it. Documenting them, because the wrong turns are instructive:

1. **Suspected: whole-corpus node list + `all_vecs` list held in RAM.**
   Real inefficiency — fixed with streaming reads (`GraphStore.iter_nodes` / `count_nodes`) and a pre-allocated `(n_chunks × dim)` float32 matrix for SIMILAR_TO discovery (replacing a Python list-of-lists). **Genuine improvement, but not the dominant driver** — it addressed the smaller memory term.

2. **Suspected: MPS "allocator drift."**
   We had ported `doc_kg`'s GPU-drift machinery (adaptive/fixed embedder refresh, batch-shrink, per-window `empty_cache`/`gc`). On MPS it **misfired**: its `embed_ms ≥ 0.6` refresh trigger fired every window in steady state (~1.9 ms/row), reloading the model repeatedly and **making throughput worse**. Stripped.

3. **Suspected: LanceDB write amplification.**
   Plausible (fragment growth), but **disproven** by a microbenchmark: `tbl.add()` was flat (0.02 s/commit, no growth with fragment count).

4. **Measurement errors that misled us:**
   - A **stale LanceDB table handle** — calling `count_rows()` repeatedly on one handle read the snapshot at open time, so progress looked frozen at a fixed row count.
   - **Thread-name grep** counting idle `lancedb-tokio-worker` thread *headers* as if they were hot CPU frames.

5. **The actual cause: `encode_batch_size` too large.** Proven by direct measurement (below).

### The measurement that nailed it

Encoding 1024 long texts (~512-token sequences), `bge-small-en-v1.5`:

| encode batch | CPU: peak RAM / call | CPU time | MPS: driver mem | MPS ms/row |
|---:|---:|---:|---:|---:|
| 32 | +0.40 GB | 36.8 s | 1.06 GB | 2.02 |
| **128** | **+0.97 GB** | **17.5 s** | **2.09 GB** | **1.87** |
| 256 | — | — | 2.64 GB | 1.89 |
| 512 | — | — | 5.31 GB | 1.93 |
| **1024** | **+7.21 GB** | **17.1 s** | **8.75 GB** | **1.90** |

Two conclusions:
- Memory scales hard with batch (**1024 → 7.2 GB/CPU, 8.7 GB/MPS per call**).
- **Throughput is flat from 128 upward** — bge-small saturates at a small batch, so a large batch is pure memory cost. (128 and 1024 are the same wall-clock; MPS is ~9× faster than CPU regardless.)

---

## Why it was latent everywhere except one place

`peak ∝ batch × seq²`, and the siblings kept `seq` small **by accident**:

- Default chunking is **512 *characters* (~120–128 tokens)**, and `doc_kg` further truncates node text to `[:1024]` chars (~256 tokens) before embedding — well below the model's 512-token max.
- The model is 384-dim `bge-small`, not a 768-dim model.

So the oversized batch was masked by short sequences. `memory_kg`'s `--chunk-strategy heading` broke that assumption — it emits **long, near-max-sequence chunks** — and the latent bug detonated. **Any module was one big-model or long-chunk config away from the same OOM.**

---

## Cross-repo audit

| Repo | Embed path | Encode batch (before) | Exposed? |
|---|---|---:|---|
| **kg_utils** | shared `SentenceTransformerEmbedder` + `_WrappedEmbedder` | **512** (wrapped: hardcoded, no override) | root cause for all |
| **doc_kg** | `SemanticIndex.build` (direct) | **1024** for the first ~240k rows (128 cap only fired *after* 240k) | **Yes — live** |
| **memory_kg** | own `SemanticIndex.build` (copied from doc_kg) | **1024** | **Yes — where it detonated** |
| **diary_kg** | delegates → `kg_utils.wrap_embedder` | **512 hardcoded**, bypassing doc_kg's cap | Latent (short chunks/small model saved it) |
| **gutenberg_kg** | delegates → `doc_kg` `CorpusEmbedder` | **64** (single-process on GPU, per-book) | No |

---

## Remediation

**Layer 1 — fix the shared root (`kg_utils` 0.4.6, published to PyPI).**
Added `DEFAULT_ENCODE_BATCH = 128` and threaded a uniform optional `encode_batch_size` through **all three** `embed_texts`: the abstract `Embedder`, `SentenceTransformerEmbedder` (default 512 → 128), and `wrap_embedder`'s `_WrappedEmbedder` (was hardcoded 512 with no override — the `diary_kg` path). Throughput-neutral; protects every module and every path.

**Layer 2 — fix the one live exposure (`doc_kg`).**
`SemanticIndex.build` default `encode_batch_size` 1024 → 128, and the encode sub-batch is now `min(current_encode_batch, 128)` **unconditionally** (from row 0), replacing the fragile "cap only after 240k rows." CLI `dockg build-index --encode-batch` default 1024 → 128. The IVF ANN index was intentionally left in place.

**Layer 3 — `memory_kg`.** Its own build fixed (streaming reads + pre-allocated SIMILAR_TO matrix + default 128, GPU-drift machinery removed), shipped as v0.6.0.

**Layer 4 — pin the fix everywhere.** All modules now require `kgmodule-utils>=0.4.6` so the dependency cannot silently regress the default.

---

## Secondary finding: IVF ANN index regressed recall

While fixing memory in `memory_kg`, a trialled **IVF ANN index** (auto-built ≥50k rows for search speed) was found to **regress retrieval recall**: at the default `nprobes=50` it probed ~7% of partitions, giving **0.970 mean / 0.800 min** top-20 overlap with exact search — a −1 to −1.6 pp recall hit that recovered only by @50 (the classic approximate-search signature).

| LongMemEval recall | before | ANN index | after fix |
|---|---:|---:|---:|
| @10 | 0.994 | 0.978 | **0.992** |
| @30 | 1.000 | 0.994 | **1.000** |
| misses@10 | 3/500 | 11/500 | **4/500** |

`memory_kg` removed the ANN index and reverted to exact flat-scan search (recall depends on it). **`doc_kg` retains the IVF index by decision** — flagged here as a known trade-off (approximate search on large `doc_kg` corpora) for separate consideration.

---

## Verification

- **Memory:** 528k-node MPS build peaks **~4 GB, flat** (was 25–32 GB, stalling). Confirmed by a live RSS sampler across the full run.
- **Speed:** unchanged (128 == 1024 wall-clock); MPS ~9× faster than CPU.
- **Recall:** LongMemEval back to **1.000@30 / 0.992@10** (4/500 misses) — matches the pre-regression baseline.
- **Tests:** green across all modules (memory_kg 279, doc_kg 412, diary_kg 200, gutenberg_kg 243, kg_utils 309).

---

## Lessons learned

1. **Profile memory before porting mitigations.** We ported GPU-drift machinery to fight a symptom; the real driver was one default. Measure the actual allocation first.
2. **`batch × seq²` is the number that matters** — not batch alone. A safe batch on short chunks is a bomb on long ones. Bound the *product*.
3. **Bigger batch ≠ faster for small models.** Throughput was flat above ~128; the large default was pure downside. Don't assume larger batches help.
4. **Shared defaults belong in the shared library.** The bug lived in `kg_utils` and every module inherited it; fixing it once there fixed the stack. Conversely, a per-module hardcode (`_WrappedEmbedder`'s 512) silently defeated a downstream cap — avoid hardcoding what should be a shared, overridable default.
5. **Latent ≠ safe.** Four of five modules "worked" only because of an incidental short-sequence assumption. Audit for the config that breaks the assumption, not just current behavior.
6. **Measurement hygiene.** Stale DB handles and thread-name greps sent us down blind alleys. Verify the measurement before trusting the conclusion.

---

## References

- **kg_utils** #4 — `DEFAULT_ENCODE_BATCH = 128`; published to PyPI as 0.4.6.
- **memory_kg** #3 — streaming embedding + consolidation (v0.6.0); #4 — pin `>=0.4.6`.
- **doc_kg** #7 — hard-cap encode batch at 128.
- **diary_kg** #4, **gutenberg_kg** #13 — pin `>=0.4.6`.
