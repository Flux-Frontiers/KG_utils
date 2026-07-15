
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: Elastic-2.0](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](https://www.elastic.co/licensing/elastic-license)
[![Version](https://img.shields.io/badge/version-0.5.0-blue.svg)](https://github.com/Flux-Frontiers/KG_utils/releases)
[![CI](https://github.com/Flux-Frontiers/KG_utils/actions/workflows/ci.yml/badge.svg)](https://github.com/Flux-Frontiers/KG_utils/actions/workflows/ci.yml)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21364005.svg)](https://doi.org/10.5281/zenodo.21364005)

# kgmodule-utils

**kgmodule-utils** — Shared graph store, semantic index, pipeline base, and snapshot infrastructure for the KGModule SDK.

*Author: Eric G. Suchanek, PhD*

*Flux-Frontiers, Liberty TWP, OH*

---

## Overview

kgmodule-utils is the **shared SDK layer** for the Flux-Frontiers knowledge-graph ecosystem. It provides everything a domain KG module needs — from type abstractions and SQLite graph storage through LanceDB vector indexing and a full build/query/pack pipeline — so domain authors implement only what is specific to their source domain.

Every KGModule implementation — [PyCodeKG](https://github.com/Flux-Frontiers/pycode_kg), [DocKG](https://github.com/Flux-Frontiers/doc_kg), and others — subclasses `KGModule` from here and implements exactly three methods: `make_extractor()`, `kind()`, and `analyze()`.

---

## Features

- **`kg_utils.specs`** — `NodeSpec`, `EdgeSpec`, `BuildStats`, `QueryResult`, `SnippetPack` dataclasses
- **`kg_utils.extractor`** — `KGExtractor` ABC: `extract()`, `node_kinds()`, `edge_kinds()`, `coverage_metric()`
- **`kg_utils.store`** — `GraphStore`: SQLite-backed node/edge store with BFS expansion, symbol resolution, caller lookup, and provenance recording
- **`kg_utils.semantic`** — `SemanticIndex` (LanceDB), `SentenceTransformerEmbedder`, `SeedHit`, model registry, `resolve_model_path()`
- **`kg_utils.pipeline`** — `KGModule`: full build → query → pack pipeline base with hybrid semantic + lexical reranking and snippet extraction
- **`kg_utils.embedder`** — `get_embedder()`, `wrap_embedder()`, `load_sentence_transformer()` factory functions
- **`kg_utils.embed`** — `Embedder` protocol, `DEFAULT_MODEL`, `KNOWN_MODELS`, `resolve_model_path()`
- **`kg_utils.snapshots`** — `Snapshot`, `SnapshotManager`, `SnapshotManifest` for temporal metric tracking
- **`kg_utils.synthesis`** — Unified text + image synthesis: oMLX, Ollama, and OpenAI text backends; mflux-local, mflux-serve, and DALL-E image backends; all env-var configurable

---

## Installation

**Requirements:** Python ≥ 3.12, < 3.14

### Core only (stdlib, no optional deps)

```bash
pip install kgmodule-utils
```

### With semantic search (LanceDB + sentence-transformers)

```bash
pip install 'kgmodule-utils[semantic]'
```

### With text + image synthesis (oMLX / Ollama / OpenAI / mflux-serve)

```bash
pip install 'kgmodule-utils[synthesis]'
```

### With local mflux image generation (Apple Silicon, includes synthesis)

```bash
pip install 'kgmodule-utils[synthesis-mflux]'
```

### In a Poetry project

```toml
[tool.poetry.dependencies]
kgmodule-utils = { version = ">=0.4.0", extras = ["semantic", "synthesis"] }
```

---

## Quick Start

### Build a domain KG module

```python
from collections.abc import Iterator
from pathlib import Path

from kg_utils.extractor import KGExtractor
from kg_utils.pipeline import KGModule
from kg_utils.specs import EdgeSpec, NodeSpec


class MyExtractor(KGExtractor):
    def node_kinds(self) -> list[str]:
        return ["document", "section"]

    def edge_kinds(self) -> list[str]:
        return ["CONTAINS"]

    def meaningful_node_kinds(self) -> list[str]:
        return ["section"]

    def extract(self) -> Iterator[NodeSpec | EdgeSpec]:
        for doc in self.repo_path.glob("**/*.md"):
            doc_id = f"document:{doc}"
            yield NodeSpec(node_id=doc_id, kind="document",
                           name=doc.stem, qualname=doc.stem,
                           source_path=str(doc))
            # … yield sections and CONTAINS edges


class MyKG(KGModule):
    _default_dir = ".mykg"

    def make_extractor(self) -> KGExtractor:
        return MyExtractor(self.repo_root)

    def kind(self) -> str:
        return "my"

    def analyze(self) -> str:
        s = self.stats()
        return f"# MyKG\nnodes={s['total_nodes']}"


# Build and query
kg = MyKG("/path/to/repo")
kg.build(wipe=True)

result = kg.query("authentication flow", k=8, hop=1)
pack   = kg.pack("error handling", max_nodes=10)
print(pack.to_markdown())
```

### Track metrics over time

```python
from kg_utils.snapshots import SnapshotManager

mgr = SnapshotManager(".mykg/snapshots", package_name="my-kg")

snapshot = mgr.capture(
    version="1.0.0",
    branch="main",
    graph_stats_dict=kg.stats(),
)
mgr.save_snapshot(snapshot)

snaps = mgr.list_snapshots(limit=5)
delta = mgr.diff_snapshots(snaps[-1]["key"], snaps[0]["key"])
```

---

## API Reference

### `kg_utils.specs`

| Class | Description |
|---|---|
| `NodeSpec` | Graph node: `node_id`, `kind`, `name`, `qualname`, `source_path`, `lineno`, `end_lineno`, `docstring`, `metadata` |
| `EdgeSpec` | Graph edge: `source_id`, `target_id`, `relation`, `weight`, `metadata` |
| `BuildStats` | Build result: node/edge counts, indexed rows, embedding dim |
| `QueryResult` | Query result: nodes, edges, seeds, hop, relevance metadata |
| `SnippetPack` | Pack result: nodes with snippets, `to_markdown()`, `to_json()`, `save()` |

### `kg_utils.extractor`

| Class | Description |
|---|---|
| `KGExtractor` | ABC — implement `node_kinds()`, `edge_kinds()`, `extract()` |

### `kg_utils.store`

| Class | Description |
|---|---|
| `GraphStore` | SQLite persistence: `write()`, `expand()`, `query_nodes()`, `resolve_symbols()`, `callers_of()`, `stats()` |

### `kg_utils.semantic`

| Class / function | Description |
|---|---|
| `SemanticIndex` | LanceDB vector index: `build()`, `search()` |
| `SentenceTransformerEmbedder` | Local embedding via sentence-transformers |
| `resolve_model_path()` | Resolve model name / alias to local cache path |
| `suppress_ingestion_logging()` | Silence verbose HF / tqdm output during ingestion |

### `kg_utils.pipeline`

| Class | Description |
|---|---|
| `KGModule` | Concrete base — implement `make_extractor()`, `kind()`, `analyze()`; get `build()`, `query()`, `pack()`, `stats()` for free |

### `kg_utils.snapshots`

| Class | Description |
|---|---|
| `Snapshot` | Temporal snapshot keyed by git tree hash with metrics and deltas |
| `SnapshotManager` | Capture, persist, load, list, diff, and prune snapshots |
| `SnapshotManifest` | Fast-lookup index with format versioning |

### `kg_utils.synthesis`

> Full reference: [docs/synthesis.md](docs/synthesis.md)

| Class / function | Description |
|---|---|
| `TextBackend` | Enum: `omlx` \| `ollama` \| `openai` |
| `ImageBackend` | Enum: `mflux-local` \| `mflux-serve` \| `openai` |
| `TextConfig` | Backend config dataclass with `resolved_endpoint()` / `resolved_model()` |
| `ImageConfig` | Backend config dataclass with `resolved_server_url()` / `resolved_model()` |
| `TextSynthesizer` | `list_models()`, `synthesize_rag()`, `rewrite_for_image()` |
| `ImageSynthesizer` | `generate()` → PIL Image, `generate_b64()` → base64 PNG |
| `text_config_from_env()` | Build `TextConfig` from `SYNTH_*` env vars |
| `image_config_from_env()` | Build `ImageConfig` from `IMAGE_*` env vars |
| `text_synthesizer_from_env()` | Convenience: config + synthesizer in one call |
| `image_synthesizer_from_env()` | Convenience: config + synthesizer in one call |

---

## Project Structure

```
KG_utils/
├── pyproject.toml
├── docs/
│   └── synthesis.md          # Synthesis sub-package reference
├── src/
│   └── kg_utils/
│       ├── __init__.py
│       ├── specs.py          # NodeSpec, EdgeSpec, BuildStats, QueryResult, SnippetPack
│       ├── extractor.py      # KGExtractor ABC
│       ├── store.py          # GraphStore (SQLite)
│       ├── semantic.py       # SemanticIndex, SentenceTransformerEmbedder, SeedHit
│       ├── pipeline.py       # KGModule concrete base class
│       ├── module.py         # Re-export shim
│       ├── embed.py          # Embedder protocol, model registry
│       ├── embedder.py       # SentenceTransformerEmbedder factory functions
│       ├── snapshots/
│       │   ├── __init__.py
│       │   ├── models.py     # Snapshot, SnapshotManifest, PruneResult
│       │   └── manager.py    # SnapshotManager
│       └── synthesis/
│           ├── __init__.py   # Public API + factory functions
│           ├── _config.py    # TextBackend, ImageBackend, TextConfig, ImageConfig, env factories
│           ├── _text.py      # TextSynthesizer
│           └── _image.py     # ImageSynthesizer
└── tests/
    ├── test_store.py               # GraphStore unit tests
    ├── test_pipeline_utils.py      # Pipeline utility function tests
    ├── test_pipeline_module.py     # End-to-end integration tests (--integration)
    ├── test_types.py               # Spec dataclass and KGExtractor tests
    ├── test_snapshots.py           # Snapshot lifecycle tests
    ├── test_integration.py         # Cross-module integration tests
    ├── test_synthesis_config.py    # Config defaults and env-var priority chains (44 tests)
    ├── test_synthesis_text.py      # TextSynthesizer with mocked openai client (38 tests)
    └── test_synthesis_image.py     # ImageSynthesizer with mocked backends (34 tests)
```

---

## Development

```bash
git clone https://github.com/Flux-Frontiers/KG_utils.git
cd KG_utils
poetry install --with dev
```

Run the fast test suite (no model downloads):

```bash
poetry run pytest -m "not integration"
```

Run all tests including semantic/integration (requires `[semantic]` extra):

```bash
poetry run pytest
```

---

## Citation

If you use kgmodule-utils in research or a project, please cite it:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21364005.svg)](https://doi.org/10.5281/zenodo.21364005)

**APA**

> Suchanek, E. G. (2026). *kgmodule-utils: Shared SDK for the KGModule Knowledge-Graph Ecosystem* (Version 0.5.0) [Software]. Flux-Frontiers. https://doi.org/10.5281/zenodo.21364005

**BibTeX**

```bibtex
@software{suchanek_kgmodule_utils,
  author    = {Suchanek, Eric G.},
  title     = {{kgmodule-utils}: Shared SDK for the KGModule Knowledge-Graph Ecosystem},
  version   = {0.5.0},
  year      = {2026},
  publisher = {Flux-Frontiers},
  url       = {https://github.com/Flux-Frontiers/KG_utils},
  doi       = {10.5281/zenodo.21364005},
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

---

## License

[Elastic License 2.0](https://www.elastic.co/licensing/elastic-license) — see [LICENSE](LICENSE).

Free to use, modify, and distribute. You may not offer the software as a hosted or managed service to third parties. Commercial use internally is permitted.
