
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: Elastic-2.0](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](https://www.elastic.co/licensing/elastic-license)
[![Version](https://img.shields.io/badge/version-0.12.1-blue.svg)](https://github.com/Flux-Frontiers/KG_utils/releases)
[![CI](https://github.com/Flux-Frontiers/KG_utils/actions/workflows/ci.yml/badge.svg)](https://github.com/Flux-Frontiers/KG_utils/actions/workflows/ci.yml)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21284866.svg)](https://doi.org/10.5281/zenodo.21284866)

# kgmodule-utils

**kgmodule-utils** — Shared graph store, semantic index, pipeline base, and snapshot infrastructure for the KGModule SDK.

*Author: Eric G. Suchanek, PhD*

*Flux-Frontiers, Liberty TWP, OH*

---

## Overview

kgmodule-utils is the **shared SDK layer** for the Flux-Frontiers knowledge-graph ecosystem. It provides everything a domain KG module needs — from type abstractions and SQLite graph storage through pluggable vector indexing (sqlite-vec by default, LanceDB optional) and a full build/query/pack pipeline — so domain authors implement only what is specific to their source domain.

Every KGModule implementation — [PyCodeKG](https://github.com/Flux-Frontiers/pycode_kg), [DocKG](https://github.com/Flux-Frontiers/doc_kg), and others — subclasses `KGModule` from here and implements exactly three methods: `make_extractor()`, `kind()`, and `analyze()`.

---

## Features

- **`kg_utils.specs`** — `NodeSpec`, `EdgeSpec`, `BuildStats`, `QueryResult`, `SnippetPack` dataclasses
- **`kg_utils.extractor`** — `KGExtractor` ABC: `extract()`, `node_kinds()`, `edge_kinds()`, `coverage_metric()`
- **`kg_utils.store`** — `GraphStore`: SQLite-backed node/edge store with BFS expansion, symbol resolution, caller lookup, and provenance recording
- **`kg_utils.semantic`** — `SemanticIndex`, `SentenceTransformerEmbedder`, `SeedHit`, model registry, `resolve_model_path()`
- **`kg_utils.vector_backend`** — `VectorBackend` protocol with `SqliteVecBackend` (default, exact recall) and `LanceDBBackend` (legacy); `make_backend()`, `resolve_backend_name()`
- **`kg_utils.pipeline`** — `KGModule`: full build → query → pack pipeline base with hybrid semantic + lexical reranking and snippet extraction
- **`kg_utils.embedder`** — `get_embedder()`, `wrap_embedder()`, `load_sentence_transformer()` factory functions
- **`kg_utils.embed`** — `Embedder` protocol, `DEFAULT_MODEL`, `KNOWN_MODELS`, `resolve_model_path()`
- **`kg_utils.snapshots`** — `Snapshot`, `SnapshotManager`, `SnapshotManifest` for temporal metric tracking
- **`kg_utils.synthesis`** — Unified text + image synthesis: oMLX, Ollama, and OpenAI text backends; mflux-local, mflux-serve, and DALL-E image backends; all env-var configurable
- **`kg_utils.viz`** — Shared interactive-HTML graph rendering (`viz` extra): `build_graph_html()`, `select_nodes()`, `GraphTheme`, `TooltipSpec` — one renderer for code, document, and metabolic graphs, with domain differences supplied as data
- **`kg_utils.viz3d`** — Shared 3-D graph layout (`viz3d` extra): `Layout3D`, `AlliumLayout`, `FunnelLayout`, `LayoutNode`, `LayoutEdge` — coordinates only, no renderer, so each module keeps its own viewer and shares the spatial reasoning. `kg_utils.viz3d.organic` adds space-colonization tree skeletons (`grow_tree`, `tree_mesh`) for corpora that should read as wood rather than as a scatter plot
- **`kg_utils.analysis`** — Read persisted centrality back out of SQLite: `load_scores()`, `available_metrics()`, `ScoreSet` (raw score, dense rank, percentile, range scaling); stdlib only

---

## Installation

**Requirements:** Python ≥ 3.12, < 3.14

### Core only (stdlib, no optional deps)

```bash
pip install kgmodule-utils
```

### With semantic search (sqlite-vec + sentence-transformers)

```bash
pip install 'kgmodule-utils[semantic]'
```

### With legacy LanceDB support (only for an un-migrated LanceDB store)

As of 0.10.0 `lancedb` is no longer part of `[semantic]`. Install it explicitly
only if you have a pre-existing, un-migrated LanceDB store on disk.

```bash
pip install 'kgmodule-utils[semantic,lancedb]'
```

### With text + image synthesis (oMLX / Ollama / OpenAI / mflux-serve)

```bash
pip install 'kgmodule-utils[synthesis]'
```

### With local mflux image generation (Apple Silicon, includes synthesis)

```bash
pip install 'kgmodule-utils[synthesis-mflux]'
```

### With interactive graph rendering (pyvis / vis-network)

```bash
pip install 'kgmodule-utils[viz]'
```

### With 3-D graph layout (numpy)

```bash
pip install 'kgmodule-utils[viz3d]'
```

### With 3-D rendering (numpy + pyvista)

The layouts above return coordinates and draw nothing. To build meshes from them
— `smooth_paths`, `tree_mesh`, `leaf_glyphs` — you need a renderer too:

```bash
pip install 'kgmodule-utils[viz3d-render]'
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
| `SemanticIndex` | Vector index over a pluggable backend (sqlite-vec default, LanceDB optional): `build()`, `search()` |
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

### `kg_utils.viz`

> Requires the `viz` extra.

| Class / function | Description |
|---|---|
| `build_graph_html()` | Render nodes + edges to a self-contained interactive HTML page (vis-network inlined) |
| `select_nodes()` | Cap a display graph while keeping it connected — seed on central nodes, expand to neighbours |
| `GraphTheme` | Names a domain's node kinds and edge relations (`KindStyle`, `with_alpha()`) |
| `TooltipSpec` | Names the node fields worth showing in a tooltip (`TooltipRow`) |

### `kg_utils.viz3d`

> Requires the `viz3d` extra for layouts, or `viz3d-render` to build meshes.

A layout maps nodes and edges onto `{node_id: [x, y, z]}` and draws nothing, so the
same layout feeds a PyVista desktop viewer, an off-screen light-field renderer, or a
plain scatter plot. Which kind is a root, which relation means containment, and which
kind sits on which Z level are all constructor arguments — one engine, every domain.

| Class / function | Description |
|---|---|
| `Layout3D` | ABC for layout strategies: `compute(nodes, edges) -> {id: [x, y, z]}` |
| `AlliumLayout` | Each root node becomes a Giant Allium — a stem with a Fibonacci-sphere head of its children (`root_kind`, `contains_rel`) |
| `FunnelLayout` | Node kind picks the Z layer, golden-angle disc spiral within each layer (`zlevels`, `level_sizes`, `default_level`) |
| `LayoutNode` / `LayoutEdge` | Domain-neutral node and edge DTOs (`from_dict()`) |
| `fibonacci_sphere()` / `fibonacci_annulus()` / `golden_spiral_2d()` | Even point distributions for building your own layout |

#### Organic trees — `kg_utils.viz3d.organic`

Lattice layouts place nodes; this half *grows* a skeleton toward them, so a
corpus reads as wood rather than as a scatter plot. Leaf-level positions become
attraction points and the branching comes from space colonization, so every limb
is a real structural path and the canopy's shape is the graph's shape.

The engine takes crown attractors and a root — **the hierarchy is yours to
choose**. A document corpus grows document → section → chunk; a diary grows
trunk → one limb per year → entry cluster → leaves. Because the pipe model sizes
a limb by what it carries, a prolific year grows visibly heavier wood.

| Function | Description |
|---|---|
| `grow_tree(attractors, root, key=...)` | One-call entry point: `colonize` then `pipe_radii`, seeded reproducibly from `key` |
| `colonize()` | Space colonization (Runions, Lane & Prusinkiewicz 2007) → `Skeleton` |
| `pipe_radii()` | Per-node branch radius by da Vinci's rule (`PIPE_EXPONENT`) |
| `root_to_tip_paths()` / `smooth_paths()` | Skeleton paths, and their Catmull-Rom smoothing |
| `tree_mesh()` / `leaf_glyphs()` | Swept-tube wood and foliage as `PolyData` |
| `crown_spacing()` / `seed_from_key()` | Natural length scale of a cloud; stable seed from any string |

The geometry above is NumPy-only and needs just `viz3d`. The three that return
PyVista objects — `smooth_paths`, `tree_mesh`, `leaf_glyphs` — import it lazily
and raise a `ModuleNotFoundError` naming the install if it is absent, so reach
for `viz3d-render` when you intend to build meshes.

```python
from kg_utils.viz3d import grow_tree, tree_mesh

skeleton = grow_tree(chunk_positions, root=[0, 0, 0], key="pepys")
wood = tree_mesh(skeleton)          # needs pyvista; see below
```

> **The `viz3d` extra installs NumPy only.** The geometry above is pure NumPy;
> only `smooth_paths`, `tree_mesh` and `leaf_glyphs` need PyVista, which they
> import lazily. Install `pyvista` alongside if you want meshes — everything
> else works without it.

```python
from kg_utils.viz3d import FunnelLayout, LayoutEdge, LayoutNode

layout = FunnelLayout(
    layer_gap=12.0,
    zlevels={"document": 0, "section": 1, "chunk": 2},
    level_sizes={0: 1.8, 1: 0.9, 2: 0.28},
)
positions = layout.compute(nodes, edges)   # {node_id: np.array([x, y, z])}
```

### `kg_utils.analysis`

| Class / function | Description |
|---|---|
| `load_scores()` | Read a centrality metric back out of SQLite into a `ScoreSet` |
| `available_metrics()` | List the centrality metrics persisted in a graph store (`MetricRef`) |
| `ScoreSet` | Per-node raw score, dense rank, percentile, and range scaling (`Scaler`) |

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
│       ├── vector_backend.py # VectorBackend protocol, SqliteVecBackend (default), LanceDBBackend
│       ├── pipeline.py       # KGModule concrete base class
│       ├── module.py         # Re-export shim
│       ├── embed.py          # Embedder protocol, model registry
│       ├── embedder.py       # SentenceTransformerEmbedder factory functions
│       ├── corpus_embedder.py # CorpusEmbedder / EmbeddingCache: multi-worker corpus embedding
│       ├── retrieval/        # Serialize + enrich KG hits (hit_to_dict, attach_content_by_sqlite)
│       │   ├── __init__.py
│       │   └── hits.py
│       ├── worker/           # RunPod /runsync client (WorkerClient, decode/handle helpers)
│       │   ├── __init__.py
│       │   ├── client.py
│       │   └── ops.py
│       ├── snapshots/
│       │   ├── __init__.py
│       │   ├── models.py     # Snapshot, SnapshotManifest, PruneResult
│       │   └── manager.py    # SnapshotManager
│       ├── synthesis/
│       │   ├── __init__.py   # Public API + factory functions
│       │   ├── _config.py    # TextBackend, ImageBackend, TextConfig, ImageConfig, env factories
│       │   ├── _text.py      # TextSynthesizer
│       │   ├── _image.py     # ImageSynthesizer
│       │   └── factory.py    # Per-request backend override helpers
│       ├── viz/              # Shared graph rendering (viz extra)
│       │   ├── __init__.py   # build_graph_html, select_nodes, GraphTheme, TooltipSpec
│       │   ├── graph_html.py # Interactive HTML renderer (vis-network inlined)
│       │   ├── theme.py      # GraphTheme, KindStyle, with_alpha
│       │   └── tooltip.py    # TooltipSpec, TooltipRow
│       ├── viz3d/            # Shared 3-D graph layout (viz3d extra)
│       │   ├── __init__.py   # Layout3D, AlliumLayout, FunnelLayout, LayoutNode, LayoutEdge
│       │   └── layout.py     # Layout engine + Fibonacci point distributions
│       └── analysis/
│           ├── __init__.py   # load_scores, available_metrics, ScoreSet
│           └── scores.py     # Read persisted centrality out of SQLite
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

Requires Python 3.12 or 3.13 (`requires-python = ">=3.12,<3.14"`); CI builds on
3.12.

```bash
git clone https://github.com/Flux-Frontiers/KG_utils.git
cd KG_utils
poetry env use python3.12
poetry install --with dev --extras "semantic" --extras "synthesis" --extras "viz"
```

**The extras are not optional for testing.** The core install is deliberately
zero-dependency (`dependencies = []`), so `poetry install --with dev` on its own
installs no runtime packages at all — pytest then aborts during *collection* on
missing `numpy` and `httpx` and runs nothing. The three extras above are exactly
what the CI test job installs.

Run the fast test suite (no model downloads) — **520 passed, 5 skipped**:

```bash
poetry run pytest -m "not integration"
```

The 5 skips are optional backends. Add two more extras to cover the LanceDB
backend and the PyVista renderers as well — **554 passed, 2 skipped** (the
remainder are the `doc_kg` sibling package, and one test that only runs when
PyVista is *absent*):

```bash
poetry install --with dev --extras "semantic" --extras "synthesis" \
  --extras "viz" --extras "lancedb" --extras "viz3d-render"
```

Run everything, including the `integration` marker — these download embedding
models on first use and are excluded from CI:

```bash
poetry run pytest
```

`TEIEmbedder`'s live tests are skipped unless a Text Embeddings Inference
server is reachable; point them at one to include them:

```bash
KG_EMBED_ENDPOINT=http://localhost:8080 poetry run pytest -m integration
```

---

## Citation

If you use kgmodule-utils in research or a project, please cite it:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21284866.svg)](https://doi.org/10.5281/zenodo.21284866)

**APA**

> Suchanek, E. G. (2026). *kgmodule-utils: Shared SDK for the KGModule Knowledge-Graph Ecosystem* (Version 0.12.1) [Software]. Flux-Frontiers. https://doi.org/10.5281/zenodo.21284866

**BibTeX**

```bibtex
@software{suchanek_kgmodule_utils,
  author    = {Suchanek, Eric G.},
  title     = {{kgmodule-utils}: Shared SDK for the KGModule Knowledge-Graph Ecosystem},
  version   = {0.12.1},
  year      = {2026},
  publisher = {Flux-Frontiers},
  url       = {https://github.com/Flux-Frontiers/KG_utils},
  doi       = {10.5281/zenodo.21284866},
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

---

## License

[Elastic License 2.0](https://www.elastic.co/licensing/elastic-license) — see [LICENSE](LICENSE).

Free to use, modify, and distribute. You may not offer the software as a hosted or managed service to third parties. Commercial use internally is permitted.
