# Features

The module-by-module feature list for **kgmodule-utils**. For installation and
usage, see the [README](../README.md); for the full class-level API, see the
README's [API Reference](../README.md#api-reference).

- **`kg_utils.specs`** — `NodeSpec`, `EdgeSpec`, `BuildStats`, `QueryResult`, `SnippetPack` dataclasses
- **`kg_utils.extractor`** — `KGExtractor` ABC: `extract()`, `node_kinds()`, `edge_kinds()`, `coverage_metric()`
- **`kg_utils.store`** — `GraphStore`: SQLite-backed node/edge store with BFS expansion, symbol resolution, caller lookup, and provenance recording
- **`kg_utils.semantic`** — `SemanticIndex`, `SentenceTransformerEmbedder`, `SeedHit`, model registry, `resolve_model_path()`
- **`kg_utils.vector_backend`** — `VectorBackend` protocol with `SqliteVecBackend` (default, exact recall) and `LanceDBBackend` (deprecated; un-migrated stores only); `make_backend()`, `resolve_backend_name()`
- **`kg_utils.pipeline`** — `KGModule`: full build → query → pack pipeline base with hybrid semantic + lexical reranking and snippet extraction
- **`kg_utils.ingest`** — `IngestPipeline`, `IngestManifest`, `AnydocConverter`: heterogeneous documents (PDF, Word, PowerPoint, Excel, OpenDocument, RTF, EPUB, CSV) → a staged Markdown corpus any builder can consume, with per-file provenance (`ingest` extra). Library layer only — the CLI surface is `kgrag ingest` in [kg-rag](https://github.com/Flux-Frontiers/KGRAG)
- **`kg_utils.embedder`** — `get_embedder()`, `wrap_embedder()`, `load_sentence_transformer()` factory functions
- **`kg_utils.embed`** — `Embedder` protocol, `DEFAULT_MODEL`, `KNOWN_MODELS`, `resolve_model_path()`
- **`kg_utils.snapshots`** — `Snapshot`, `SnapshotManager`, `SnapshotManifest` for temporal metric tracking
- **`kg_utils.synthesis`** — Unified text + image synthesis: oMLX, Ollama, and OpenAI text backends; mflux-local, mflux-serve, and DALL-E image backends; all env-var configurable
- **`kg_utils.viz`** — Shared interactive-HTML graph rendering (`viz` extra): `build_graph_html()`, `select_nodes()`, `GraphTheme`, `TooltipSpec` — one renderer for code, document, and metabolic graphs, with domain differences supplied as data
- **`kg_utils.viz3d`** — Shared 3-D graph layout (`viz3d` extra): `Layout3D`, `AlliumLayout`, `FunnelLayout`, `LayoutNode`, `LayoutEdge` — coordinates only, no renderer, so each module keeps its own viewer and shares the spatial reasoning. `kg_utils.viz3d.organic` adds space-colonization tree skeletons (`grow_tree`, `tree_mesh`) for corpora that should read as wood rather than as a scatter plot
- **`kg_utils.viz3d.qt`** — Qt render lifecycle for light-field output (`viz3d-qt` extra): `PovRenderSession` and `PovRenderWorker` keep POV-Ray off the GUI thread and clean up safely on window close, `ImagePopup` previews the result, and `cast_scene_to_looking_glass` runs the build → render → write → cast path to a Looking Glass display
- **`kg_utils.analysis`** — Read persisted centrality back out of SQLite: `load_scores()`, `available_metrics()`, `ScoreSet` (raw score, dense rank, percentile, range scaling); stdlib only
