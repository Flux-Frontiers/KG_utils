# Release Notes -- v0.23.0

> Released: 2026-09-20

`KGModule.query()` and `pack()` now validate their arguments before touching
the index, and `__enter__` returns `Self`. Both were things three KG modules
had each worked around by hand; the base class does them now so the copies
can go.

## What changed

**Query bounds live in the base class.** Every KG module's `query()` and
`pack()` take input from a CLI, an MCP tool call or a UI, and until now each
module checked that input itself or not at all. `kg_utils.validation` carries
the check once: an empty query, `k` outside 1 to 100, `hop` outside 0 to 5 or
`max_nodes` outside 1 to 500 raises `ValueError` naming the parameter, before
the index or the graph is touched. `hop=0` (pure semantic) and
`pack(max_nodes=None)` (no cap) remain valid. The bounds were lifted from
`genealogy_kg`, checked against every fleet caller before being adopted, and
match what every fleet MCP server already documents. A module that needs
different ones sets the class attributes `max_k`, `max_hop`, `max_max_nodes`
or `max_query_len` on its subclass, and never overrides `query()` to do it.

**`with MyKG(...) as kg:` is typed as `MyKG`.** `KGModule.__enter__` and
`GraphStore.__enter__` returned the base class, so under `ty` every subclass
attribute failed type-checking inside a `with` block. They return `Self` now.
`genealogy_kg`, `swift_kg` and `connectome_kg` each carried an identical
`__enter__` override to work around this and can delete it once they floor at
0.23.0.

**Fleet floors.** `quiltwright` moves to `>=0.15.0` and `ruff` to `>=0.15`, a
currency bump with no behaviour behind it.

## Upgrading

No rebuild and no migration. A caller that was passing an out-of-range `k`,
`hop` or `max_nodes`, or an empty query, will now get a `ValueError` instead
of a silent result; no fleet caller does. Modules that copied the validation
or the `__enter__` override can delete their copies after raising their
`kgmodule-utils` floor. Verified against built-wheel runs of `pycode_kg`,
`doc_kg`, `kgrag` and `genealogy_kg` before release, all green.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
