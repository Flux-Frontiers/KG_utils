"""kg_utils — Shared types, store, semantic index, and pipeline base for the KGModule SDK.

Sub-packages / modules:
    kg_utils.specs      — NodeSpec, EdgeSpec, BuildStats, QueryResult, SnippetPack.
    kg_utils.extractor  — KGExtractor abstract base class.
    kg_utils.store      — GraphStore: SQLite-backed authoritative node/edge store.
    kg_utils.semantic   — Embedder, SentenceTransformerEmbedder, SemanticIndex, SeedHit.
    kg_utils.pipeline   — KGModule: concrete base class with full build/query/pack pipeline.
    kg_utils.snapshots  — Snapshot, SnapshotManager, SnapshotManifest, etc.
    kg_utils.embed      — Embedder protocol, DEFAULT_MODEL, KNOWN_MODELS,
                          kg_model_cache_dir(), resolve_model_path().
    kg_utils.embedder   — Concrete SentenceTransformerEmbedder, get_embedder(),
                          wrap_embedder(), load_sentence_transformer(), resolve_device().
    kg_utils.corpus_embedder — CorpusEmbedder, EmbeddingCache: multi-process,
                          device-safe corpus embedding engine.
    kg_utils.synthesis  — Unified text + image synthesis: TextSynthesizer, ImageSynthesizer.
                          Backends: omlx | ollama | openai (text);
                                    mflux-local | mflux-serve | openai (image).
    kg_utils.worker     — RunPod worker protocol helpers and WorkerClient for /runsync calls.
    kg_utils.retrieval  — Shared retrieval helpers: hit_to_dict, attach_content_by_sqlite.
    kg_utils.analysis   — ScoreSet, load_scores, available_metrics: persisted
                          centrality read back for ranking and visual encoding.
    kg_utils.viz        — GraphTheme, TooltipSpec, build_graph_html, select_nodes:
                          shared interactive graph rendering (needs the 'viz' extra).
    kg_utils.viz3d      — Layout3D, LayoutNode, LayoutEdge, AlliumLayout, FunnelLayout:
                          shared 3-D graph layout (needs the 'viz3d' extra).

Optional extras
---------------
The core install is stdlib-only; everything heavier is opt-in.

    pip install 'kgmodule-utils[semantic]'         # sentence-transformers + torch
                                                   #   + sqlite-vec (the default backend)
    pip install 'kgmodule-utils[sqlite-vec]'       # sqlite-vec alone
    pip install 'kgmodule-utils[lancedb]'          # LanceDBBackend, for un-migrated
                                                   #   legacy stores only
    pip install 'kgmodule-utils[viz]'              # pyvis — interactive graph HTML
    pip install 'kgmodule-utils[viz3d]'            # numpy — 3-D graph layout
    pip install 'kgmodule-utils[synthesis]'        # httpx + openai + pillow
    pip install 'kgmodule-utils[synthesis-mflux]'  # + mflux (Apple Silicon local gen)
"""

__version__ = "0.12.1"
