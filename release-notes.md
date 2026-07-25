# Release Notes — v0.7.0

> Released: 2026-07-25

This release lifts two capabilities that had been living inside individual KG modules into the
shared SDK: interactive graph rendering and reading persisted centrality scores back out of
SQLite. The graph renderer in particular consolidates code that had been copy-pasted per
domain — code, document and metabolic graphs now render through one implementation, with each
domain's differences supplied as data rather than a fork.

## What changed

**Shared graph rendering (`kg_utils.viz`, new `viz` extra).** A single interactive-HTML graph
renderer replaces the per-module copies. A `GraphTheme` names a domain's node kinds and edge
relations and a `TooltipSpec` names the fields worth showing, so a code graph
(`qualname`/`module_path`), a document graph (`title`/`file_path`) and a metabolic graph
(`formula`/`ec_number`) all flow through the same path. The output inlines vis-network, so a
page opens straight from `file://` and survives embedding in a `srcdoc` iframe — the previous
pyvis default emitted relative asset paths that failed silently offline. `select_nodes` also
keeps a display-capped graph connected by seeding on central nodes and expanding to
neighbours rather than truncating to the top N. The core install stays zero-dependency;
nothing in `kg_utils/__init__.py` imports the extra.

**Read centrality back out of SQLite (`kg_utils.analysis.scores`).** `available_metrics`,
`load_scores`, and a `ScoreSet` exposing raw score, dense rank, percentile and range scaling —
stdlib only. Ranks are derived on load rather than trusting a possibly-truncated stored `rank`
column, so `centrality_scores` and `node_metrics` behave identically.

**Rendering hardening.** Node data embedded in the page's `<script>` block now escapes `<`,
`>` and `&` as unicode sequences, so a node whose text contained `</script>` can no longer
terminate the block and inject markup. This affected every per-module renderer this code was
consolidated from.

## Upgrading

Nothing to do for existing code — the core install is unchanged and still pulls in no
mandatory dependencies. To use the new graph renderer, install the extra:

```bash
pip install 'kgmodule-utils[viz]'
```

The score reader (`kg_utils.analysis.scores`) needs no extra.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
