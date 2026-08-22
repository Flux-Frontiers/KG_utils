# Release Notes — v0.18.0

> Released: 2026-08-22

### Fixed

- **`NodeSpec.metadata` is now persisted. It never was.** The spec has always
  carried a `metadata` field documented as "domain-specific extension data",
  and `GraphStore` has always dropped it on write: the `nodes` table had no
  column for it, so every value any module attached to a node was discarded
  silently and read back as absent. Edge metadata was persisted (as
  `evidence`); node metadata was not, and nothing said so.

  Found while wiring the temporal contract, which is exactly the kind of
  consumer that made it matter — a module could write `occurred_start` onto a
  node and a federated query would never see it. Anything else that has tried
  to attach node metadata since the store was written has been losing it too.

  `nodes` gains a `metadata TEXT` column holding the mapping as JSON, and both
  node reads return it decoded. A blob that fails to parse, or that decodes to
  something other than an object, reads as `{}` rather than raising: extension
  data is not worth making a node unreadable over.

  **Existing databases are migrated on open.** `_SCHEMA_SQL` uses
  `CREATE TABLE IF NOT EXISTS`, which is a no-op against a database an earlier
  version created — so a column added to that statement alone would never
  reach a single existing store, and every KG in the fleet is an existing
  store. `_migrate()` adds the column with `ALTER TABLE` when absent,
  idempotently, on each connect. Without it a pre-0.18.0 database would raise
  "no such column: metadata" on its next query, before any rebuild.

### Added

- **`kg_utils.temporal` — the shared temporal contract, so time can become a
  federation axis instead of a per-module convention.** Modules that know
  *when* something happened now write the same three metadata keys —
  `occurred_start`, `occurred_end`, `recorded_at` — and a federated query can
  filter and order dated nodes across all of them without knowing which module
  produced any given node.

  Three decisions carry the design, each of which a plausible implementation
  gets wrong:

  - **Occurred is not recorded.** A diary entry written tonight about last
    Tuesday occurred on Tuesday and was recorded tonight. Conflating them puts
    it in the wrong place on the timeline, so the contract keeps both. The
    distinction is inherited from `personal_agent`, which learned it the hard
    way.
  - **Precision is preserved, and it determines extent.** `"1876"` stays a
    year rather than becoming a silent `1876-01-01`; a node dated that way
    overlaps *any* query touching 1876, while `"1876-03-04"` overlaps only
    that day. An absent `occurred_end` therefore means "as wide as the
    precision implies", not "zero duration" — which is what makes an
    undated-to-the-day publication behave correctly in a window query.
  - **Malformed dates are ignored, not raised on.** One bad date in one corpus
    must not abort a federated query across twenty.

  API: `temporal_metadata()` builds the metadata slice for producers (omitting
  absent keys, so merging never clobbers); `read_span()` returns a
  `TemporalSpan` for consumers, with `overlaps()` — supporting open-ended
  windows on either side — and a total-ordering `sort_key`. `spine_id()` and
  `spine_chain()` mint the deterministic calendar-node IDs a timeline graph
  hangs events from (`t:2026` → `t:2026-08` → `t:2026-08-17`), truncated at
  the precision available so a year-dated node is never asserted to belong to
  a day nobody recorded. Spine IDs sort lexicographically in chronological
  order.

  Stdlib only — no new dependency, no new extra, no new base class. 38 tests.
  This is layer 1 of the TimelineKG fusion; layer 2 is `QueryScope.time_range`
  in kg-rag, which calls `read_span().overlaps()`.

### Fixed

- **Converters report their version off the instance, so failures record it
  too.** `Converter` gained a `version` property, read from distribution
  metadata rather than the imported module — `AnydocConverter().version`
  resolves without importing `anydoc` at all, and answers `"unknown"` rather
  than raising when the extra is absent.

  The manifest previously named the converter on a failed record but not its
  version, so "which version rejected this file" was unanswerable from the
  ledger — a real hole in a structure whose whole job is provenance. Found
  while documenting the manifest schema. (0.17.0 on PyPI does not contain
  this fix; it shipped after that release.)

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
