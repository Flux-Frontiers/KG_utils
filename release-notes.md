# Release Notes -- v0.29.1

> Released: 2026-10-27

### Fixed

- **Symbol-resolution fan-out.** `GraphStore.resolve_symbols()` no longer
  links a stub to every definition that shares its bare name. A stub whose
  name matches more than one definition, with nothing to narrow it, stays
  unresolved, and a stub whose metadata sets `scope_local` (a per-scope
  variable or attribute node) is never resolved. On ansible (7,920 `.py`
  files) these two cases made 7.9M of the graph's 8.5M edges, an 18 GB
  store, and a build that had not finished after 35 minutes; it now
  builds in 198 s to 944 MB with 6,897 `RESOLVES_TO` edges, each to a
  single definition. `name_fallback_ambiguous` edges and their
  `candidate_count` are no longer written.
- **Exact-qualname matching on `lib/` layouts and packages.** Dotted module
  variants now also drop a leading `lib/` (as they already dropped `src/`)
  and name a package's `__init__` module by its package, so
  `from pkg.text import fmt` on a `lib/pkg/` layout resolves exactly
  instead of by name.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
