"""kg_utils — Shared types, store, semantic index, and pipeline base for the KGModule SDK.

Sub-packages / modules:
    kg_utils.types      — NodeSpec, EdgeSpec, BuildStats, QueryResult, SnippetPack,
                          KGExtractor (abstract), KGModule (abstract interface).
    kg_utils.store      — GraphStore: SQLite-backed authoritative node/edge store.
    kg_utils.semantic   — Embedder, SentenceTransformerEmbedder, SemanticIndex, SeedHit.
    kg_utils.pipeline   — KGModule: concrete base class with full build/query/pack pipeline.
    kg_utils.snapshots  — Snapshot, SnapshotManager, SnapshotManifest, etc.
    kg_utils.embed      — Embedder protocol, DEFAULT_MODEL, KNOWN_MODELS,
                          kg_model_cache_dir(), resolve_model_path().
    kg_utils.embedder   — Concrete SentenceTransformerEmbedder, get_embedder(),
                          wrap_embedder(), load_sentence_transformer().

Optional extras
---------------
    pip install 'kgmodule-utils[semantic]'   # lancedb + sentence-transformers
"""

__version__ = "0.3.0"
