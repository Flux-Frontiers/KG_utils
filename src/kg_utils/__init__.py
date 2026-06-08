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
                          wrap_embedder(), load_sentence_transformer().
    kg_utils.synthesis  — Unified text + image synthesis: TextSynthesizer, ImageSynthesizer.
                          Backends: omlx | ollama | openai (text);
                                    mflux-local | mflux-serve | openai (image).
    kg_utils.worker     — RunPod worker protocol helpers and WorkerClient for /runsync calls.
    kg_utils.retrieval  — Shared retrieval helpers: hit_to_dict, attach_content_by_sqlite.

Optional extras
---------------
    pip install 'kgmodule-utils[semantic]'         # lancedb + sentence-transformers
    pip install 'kgmodule-utils[synthesis]'        # httpx + openai + pillow
    pip install 'kgmodule-utils[synthesis-mflux]'  # + mflux (Apple Silicon local gen)
"""

__version__ = "0.4.2"
