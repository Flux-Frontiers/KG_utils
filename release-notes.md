# Release Notes -- v0.29.0

> Released: 2026-10-25

### Removed

- **LanceDB support.** `LanceDBBackend`, `_pq_subvectors()`,
  `resolve_backend_name()`, `make_backend()`, the `lancedb` extra and
  `vector_backend="lancedb"` are gone, along with the ANN-index branch in
  `SemanticIndex.build()` and `KGModule._legacy_store_dir`. sqlite-vec is the
  only backend; `VectorBackend` stays as the seam. Nothing in the fleet had
  built a LanceDB index since the sqlite-vec migration; the backend survived
  as a reader for un-migrated stores, and `dockg convert-index` was its last
  caller. Folded into the 0.29.0 wheel with the MCP base
  (`kgrag_priv/docs/MCP_CENTRALIZATION_PLAN.md`, step 0b). **Breaking:** a
  store that was never migrated can no longer be read; rebuild its index with
  `build_index()`, which derives it from the graph.

### Changed

- **`KGModule(vector_backend=)` accepts `"auto"` and `"sqlite-vec"`, both
  meaning sqlite-vec,** and raises `ValueError` naming the removal for
  `"lancedb"` and anything else. Kept because pycode_kg, tscode_kg, swift_kg,
  ftree_kg and others pass `"sqlite-vec"`. `stats()["vector_backend"]` is
  always `"sqlite-vec"`, kept because consumers display it; resolving it no
  longer inspects the disk.
- **`KGModule.drop_index()` no longer deletes a `lancedb/` directory** beside
  the store. Nothing reads one any more, so the SDK leaves it alone.
- **`KGModule.stats()` answers in the fleet's stats shape** (new
  `kg_utils.stats`, settled 2026-10-25 in kgrag_priv FLEET_STANDARDS, "KG
  stats shape"): `{"kind", "nodes": {"total", "by_kind"}, "edges": {"total",
  "by_rel"}}` plus every other key the store reports (`meaningful_nodes`,
  `docstring_coverage`, `db_path`, ...) and `vector_backend`. Before it the
  fleet answered `stats` in three spellings (`node_count`, `total_nodes`,
  `nodes`), and KGRAG translated each adapter by hand. **Breaking:** the flat
  `total_nodes` / `total_edges` / `node_counts` / `edge_counts` keys are gone
  from `KGModule.stats()`; they stay on `GraphStore.stats()`, which snapshots
  read. `SnapshotManager.capture()` flattens a nested `graph_stats_dict`, so
  snapshots persist the same keys whichever they are handed.
- **The canonical `stats` MCP tool checks the shape on every call**, the
  default tool and a module's `server.canonical("stats")` replacement alike;
  a departure is a tool error naming each problem. That includes a `kind`
  other than the server's own (`make_server(kind=...)`), which the shape
  alone cannot catch: a server wrapping another module's KG must say which
  KG it serves.

### Added

- **`kg_utils.stats`**: `kg_stats()` builds the fleet stats shape (totals summed
  from the breakdowns, so they cannot disagree), `stats_from_store()` reshapes a
  store's flat `stats()`, `flat_stats()` is the inverse, and `stats_problems()`
  lists every departure.
- **`kg_utils.mcp`: one MCP server base for every KG module** (new `mcp`
  extra: `mcp>=1.23.0,<2`, `click`). Step 0 of
  `kgrag_priv/docs/MCP_CENTRALIZATION_PLAN.md` (sweep item 93). The fleet ran
  ten servers in four implementation styles with no shared code; this is the
  shared part. `make_server(open_kg, name=, kind=)` returns a `KGServer`, a
  `FastMCP` subclass that already answers the canonical tools -- `stats`,
  `query`, `pack`, `get_node`, `neighbors` -- under the same names on every
  server; a module adds its domain tools with `@server.tool()` and reaches its
  KG with `kg_from(ctx)`. The KG is opened per session and closed after, and a
  KG whose graph file does not exist is refused at session start instead of
  served empty. The server's instructions are regenerated whenever a tool is
  added, so they cannot drift from the tool list. `manifest()` gives the tool
  list as JSON without opening the KG. `add_mcp_command()` adds the standard
  `<tool> mcp [--repo] [--db] [--vectors] [--transport] [--host] [--port]
  [--manifest]`. `run()` serves stdio or streamable HTTP and refuses the
  deprecated `sse`; under stdio it reserves the real stdout for the protocol
  and points file descriptor 1 at stderr, so a `print()` or a C library's
  output (sweep item 87) can no longer corrupt the session. Validation stays
  in the KG's own methods; the server reports their `ValueError` as a tool
  error. The MCP host's log gets a starting line naming the transport and,
  when a session opens, the resolved repo, graph, vector store and model
  it serves, on stderr; the per-module servers each printed one, and the
  base had dropped it. A module whose own version of a canonical tool does more replaces
  it under the same name with `@server.canonical("query")`; the replacement
  must keep every canonical parameter with its type and required-ness, add
  no required parameter and return the same kind of result, or the server
  refuses it at build time with every violation listed. Found while porting
  pycode_kg, whose `query` reranks with a hybrid score and caps results per
  module: swapping in the plain canonical tool would have regressed it. The
  standard command also takes `--model`, and a relative `--db` or
  `--vectors` is taken from `--repo`, as pycode_kg's command already did.
  What a replacement must keep is the canonical contract, only what every
  domain shares: `query` and `pack` take `q`, `k`, `hop`, `max_nodes` and
  `rels`. The default tools' own extras (`min_score`, `context`,
  `max_lines`) are not part of it, found when doc_kg, whose search sizes
  excerpts by characters, came to adopt the base. `ServableKG` is the
  minimum every served KG needs (`stats`, `node`, `close`); `SearchableKG`
  adds the `query` and `pack` shape the default tools call, which
  `KGModule` satisfies and a type check guards.
  Importing `kg_utils.mcp` to register the command does not import `mcp`,
  which loads only when a server is built: `mcp` costs about half a second
  against pycode_kg's whole CLI startup of about 0.17 s, and every command
  would have paid it. Nothing in the fleet uses it yet.
- **`KGModule.neighbors(node_id, rels=, direction=, limit=)`**: a node's
  one-hop neighbourhood, its edges and the existing nodes at their far ends.
  Graph only, so it never loads the embedding model; `limit` is bounded by
  `max_max_nodes`; an unknown node returns `None`, as `node()` does. Backed by
  the new **`GraphStore.edges_touching()`**, which returns edges in either or
  both directions, optionally filtered by relation, in a deterministic order.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
