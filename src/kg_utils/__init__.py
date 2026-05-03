"""kg_utils — Shared types, snapshots, and embedding protocol for the KGModule SDK.

Sub-packages:
    kg_utils.types      — NodeSpec, EdgeSpec, KGExtractor, KGModule, etc.
    kg_utils.snapshots  — Snapshot, SnapshotManager, SnapshotManifest, etc.
    kg_utils.embed      — Embedder protocol, DEFAULT_MODEL, KNOWN_MODELS,
                          kg_model_cache_dir(), resolve_model_path().
    kg_utils.embedder   — Concrete SentenceTransformerEmbedder, get_embedder(),
                          wrap_embedder(), load_sentence_transformer().
"""

__version__ = "0.2.4"
