# Release Notes — v0.6.2

> Released: 2026-07-15

A packaging fix that unbreaks installation for downstream consumers. Through 0.6.1 the type
checker `ty` had leaked into kgmodule-utils' **runtime** dependencies as its only hard entry,
so every `pip install kgmodule-utils` dragged in `ty` under a tight `>=0.0.44,<0.0.45` pin —
enough to break dependency resolution in any project that pins `ty` differently. If a
downstream install started failing to resolve after adopting 0.6.x, this is the fix.

## What changed

**`ty` removed from runtime dependencies.** The package's `[project].dependencies` list held
exactly one entry — `ty` — which never belonged there: every genuine runtime dependency lives
in an optional extra (`semantic`, `synthesis`, `sqlite-vec`, …). Installing the package now
pulls in nothing mandatory beyond the standard library, and `ty` stays in the dev group where
it is actually used. No source, API, or behavior change.

## Upgrading

Drop-in for 0.6.x. If a downstream project failed to resolve because of the `ty` pin, upgrade
to 0.6.2 and the conflict disappears. Nothing else to do.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
