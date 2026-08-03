# Release Notes — v0.10.0

> Released: 2026-08-03

Two changes worth reading before you upgrade: **LanceDB is no longer installed by the
`semantic` extra**, and **generated graph pages no longer contact a CDN**. The first is a
breaking change for a narrow set of consumers; the second means any graph HTML you built
on 0.7.0–0.9.0 is worth regenerating.

## Breaking: `lancedb` moved to its own extra

`pip install 'kgmodule-utils[semantic]'` no longer pulls `lancedb`.

sqlite-vec has been the default backend since 0.8.0. `auto_resolve_backend()` selects it
for every fresh or already-migrated store, and falls back to LanceDB only when it finds an
un-migrated LanceDB store on disk. Nothing in the fleet uses `LanceDBBackend` directly —
pycode_kg imports `SqliteVecBackend` explicitly, and doc_kg declares `lancedb` as its own
direct dependency rather than inheriting it from here. So every `[semantic]` install was
carrying LanceDB and its transitive tree for nothing.

If you have a pre-existing, un-migrated LanceDB store, opt in explicitly:

```bash
pip install 'kgmodule-utils[semantic,lancedb]'
```

Everyone else needs no change. This is why the release is 0.10.0 rather than 0.9.1 —
removing a dependency from a published extra is breaking for whoever relied on it.

## Fixed: graph pages were not self-contained

`kg_utils.viz.build_graph_html` advertised "a self-contained HTML document". It wasn't.
pyvis 0.3.2 hardcodes two Bootstrap CDN tags in its own Jinja template, and the
`cdn_resources="in_line"` setting we already pass governs only the vis-network assets — it
does not touch them. Every generated page fetched `bootstrap.min.css` and
`bootstrap.bundle.min.js` from jsdelivr.

Two consequences. Offline — an air-gapped machine, a plane, a reviewer who was mailed the
file — both requests fail and the layout collapses: the network canvas stops filling its
container and an unstyled band appears below it. And a page generated from a private
codebase silently contacted a third-party host every time someone opened it.

The output uses exactly two Bootstrap classes, so the fix strips the CDN tags and supplies
those rules directly, copied verbatim from `bootstrap@5.0.0-beta3`, along with the two
Reboot rules that turn out to be load-bearing for the box model. The result measures
pixel-identical to real Bootstrap with zero external requests, and page size is essentially
unchanged. `bootstrap.bundle.min.js` was entirely unused — no dropdowns, modals, tooltips,
or collapses in our output — and is simply dropped.

**Regenerate any graph HTML built on 0.7.0 through 0.9.0.** Those files phone home and
break offline. No consumer code changes are needed; the fix lands once here and every
consumer inherits it on upgrade.

The bug survived because the test meant to catch it named a CDN host instead of the
property it claimed to test — it asserted the absence of `cdnjs.cloudflare.com` while the
page reached out to `cdn.jsdelivr.net`. It is now host-agnostic and fails on any external
reference, including one a future pyvis release might introduce.

## Also in this release

- **`rich` pinned to `>=13.0.0,<15.0.0`** rather than an open upper bound.
- **Packaging metadata modernised.** The deprecated `[project.license]` table form is now
  the PEP 639 SPDX expression, so the wheel ships as Metadata-Version 2.4 with
  `License-Expression: Elastic-2.0` and the LICENSE bundled.

## Upgrading

```bash
pip install --upgrade kgmodule-utils
```

Add the `lancedb` extra only if you have an un-migrated LanceDB store. Regenerate any
distributed graph HTML.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
