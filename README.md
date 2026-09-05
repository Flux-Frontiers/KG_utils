
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: Elastic-2.0](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](https://www.elastic.co/licensing/elastic-license)
[![Version](https://img.shields.io/badge/version-0.18.1-blue.svg)](https://github.com/Flux-Frontiers/KG_utils/releases)
[![CI](https://github.com/Flux-Frontiers/KG_utils/actions/workflows/ci.yml/badge.svg)](https://github.com/Flux-Frontiers/KG_utils/actions/workflows/ci.yml)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21284866-blue.svg)](https://doi.org/10.5281/zenodo.21284866)

# kgmodule-utils

**kgmodule-utils** — Shared graph store, semantic index, pipeline base, and snapshot infrastructure for the KGModule SDK.

*Author: Eric G. Suchanek, PhD*

*Flux-Frontiers, Liberty TWP, OH*

---

## Overview

kgmodule-utils is the **shared SDK layer** for the Flux-Frontiers knowledge-graph ecosystem. It provides everything a domain KG module needs — from type abstractions and SQLite graph storage through pluggable vector indexing (sqlite-vec; a deprecated LanceDB backend remains only for un-migrated stores) and a full build/query/pack pipeline — so domain authors implement only what is specific to their source domain.

Fleet modules use it in two ways. [PyCodeKG](https://github.com/Flux-Frontiers/pycode_kg), [TypeScriptKG](https://github.com/Flux-Frontiers/tscode_kg), and [FTreeKG](https://github.com/Flux-Frontiers/ftree_kg) subclass `KGModule` from here and implement exactly three methods: `make_extractor()`, `kind()`, and `analyze()`. The rest — [DocKG](https://github.com/Flux-Frontiers/doc_kg), MemoryKG, MetaboKG, and others — keep their own pipelines and pull in the shared pieces they need: embedders, vector backends, snapshots, synthesis.

---

## Latest News

- **0.18.1 (2026-08-25)** — `resolve_symbols()` no longer guesses across
  classes when it doesn't have to. A dotted call stub's `RESOLVES_TO` edge
  used to match by trailing name alone, so any first-party definition sharing
  a method name could pick up a fabricated edge regardless of receiver. When
  a caller records a `receiver_class` in the stub's `metadata` (e.g. a
  visitor that traced a call's receiver back to a type annotation), matching
  now scopes to that class, and a typed stub with no match in it stays
  unresolved instead of falling back to the untyped guess.
- **0.18.0 (2026-08-22)** — New `kg_utils.temporal`: a shared temporal
  contract so time can become a federation axis instead of a per-module
  convention. Modules write the same `occurred_start` / `occurred_end` /
  `recorded_at` metadata keys, and a federated query can filter and order
  dated nodes across all of them. Precision is preserved — `"1876"` stays a
  year and overlaps any query touching 1876 — and *occurred* is kept distinct
  from *recorded*, so a diary entry written tonight about last Tuesday lands on
  Tuesday. Also fixes `NodeSpec.metadata`, which the store had always dropped
  on write; existing databases are migrated on open.
- **0.17.0 (2026-08-18)** — New `kg_utils.ingest` sub-package behind an
  `ingest` extra: `IngestPipeline` turns a folder of mixed-format documents
  (PDF, Word, PowerPoint, Excel, EPUB, and more) into a staged Markdown corpus
  any builder can consume. Conversion is anydoc; a 40-page text PDF converts in
  about 20 ms. Every file examined gets a manifest record — including the
  skips and failures — so a corpus explains its own gaps. The CLI surface is
  `kgrag ingest` in kg-rag; this package supplies the library. See
  [Document ingestion](#document-ingestion).
- **0.16.0 (2026-08-16)** — `cast_scene_to_looking_glass()` now returns a
  `CastResult` (`path`, `error`, `elapsed`, status-bar `message`) instead of a
  `(path, error)` tuple, and defaults its quilt spec sensibly. Breaking for
  callers of the tuple form.
- **0.15.0 (2026-08-16)** — New `viz3d-qt` extra: `PovRenderSession` and
  `PovRenderWorker` keep POV-Ray off the GUI thread and shut down safely on
  window close. Tag pushes now publish to PyPI through trusted publishing.

Older changes are in the [CHANGELOG](CHANGELOG.md).

---

## Features

The SDK covers the full life of a knowledge graph: core contracts and SQLite
storage (`specs`, `extractor`, `store`, `pipeline`), semantic indexing over
pluggable vector backends (`semantic`, `vector_backend`, `embed`, `embedder`),
document ingestion (`ingest`), a shared temporal contract for dated nodes
(`temporal`), temporal snapshots (`snapshots`), text + image
synthesis (`synthesis`), centrality analysis (`analysis`), and a visualization
stack that runs from interactive HTML (`viz`) through 3-D layouts and organic
trees (`viz3d`) to Looking Glass light-field casting (`viz3d.qt`).

The module-by-module feature list lives in [docs/features.md](docs/features.md).

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

### With document ingestion (PDF, Word, PowerPoint, Excel, EPUB, …)

Only needed for non-textual sources — `.md`, `.txt` and `.rst` ingest with no
extra dependency at all.

```bash
pip install 'kgmodule-utils[ingest]'
```

### With deprecated LanceDB support (only for an un-migrated LanceDB store)

**LanceDB is deprecated** — sqlite-vec is the backend for all new stores. As of
0.10.0 `lancedb` is no longer part of `[semantic]`. Install it explicitly only
if you have a pre-existing, un-migrated LanceDB store on disk; migrate when you
can.

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

### With the Qt render lifecycle (PyQt5 + quiltwright)

To ray-trace and cast to a Looking Glass display from a Qt viewer — the
`kg_utils.viz3d.qt` worker thread, session lifecycle, and cast path:

```bash
pip install 'kgmodule-utils[viz3d-qt]'
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
| `SemanticIndex` | Vector index over a pluggable backend (sqlite-vec default; LanceDB deprecated): `build()`, `search()` |
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
| `Snapshot` | Temporal snapshot keyed by a supplied release tag or timestamp, with metrics, deltas, and tree-hash provenance |
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

#### Qt render lifecycle — `kg_utils.viz3d.qt`

> Requires the `viz3d-qt` extra. Not re-exported from `kg_utils.viz3d`:
> these classes subclass `QThread`/`QDialog`/`QObject`, so importing them
> requires PyQt5 at class-definition time, and importing a layout must not.

The machinery a Qt viewer needs to ray-trace a scene and cast it to a
Looking Glass display, factored out of `pycode_kg` and `gutenberg_kg`.
Which node becomes a trunk stays per-repo; the session takes its progress
bar and status callback as constructor arguments and makes no assumptions
about the host window.

| Class / function | Description |
|---|---|
| `PovRenderSession` | Owns the render lifecycle: temp views directory, file-count progress, cleanup, and a `shutdown()` that detaches from a live worker so closing the window mid-render cannot abort the process |
| `PovRenderWorker` | `QThread` that runs POV-Ray off the GUI thread |
| `ImagePopup` | Dialog that previews the rendered image |
| `cast_scene_to_looking_glass()` | Build the PyVista scene, render, write the quilt, and cast it to the display; returns a `CastResult` |
| `CastResult` | Outcome of one cast: `path`, `error`, `elapsed`, and the `message` a status bar shows |
| `DEFAULT_QUILT_PRESET` / `DEFAULT_CAST_SCALE` | The preset and scale a cast uses when no `spec` is given (`"16-landscape"` at half size) |

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
│   ├── features.md           # Module-by-module feature list
│   ├── synthesis.md          # Synthesis sub-package reference
│   ├── encode-batch-memory-postmortem.md
│   └── viz-bootstrap-selfcontainment.md
├── src/
│   └── kg_utils/
│       ├── __init__.py
│       ├── specs.py          # NodeSpec, EdgeSpec, BuildStats, QueryResult, SnippetPack
│       ├── extractor.py      # KGExtractor ABC
│       ├── store.py          # GraphStore (SQLite)
│       ├── semantic.py       # SemanticIndex, SentenceTransformerEmbedder, SeedHit
│       ├── vector_backend.py # VectorBackend protocol, SqliteVecBackend (default), LanceDBBackend (deprecated)
│       ├── pipeline.py       # KGModule concrete base class
│       ├── module.py         # Re-export shim
│       ├── embed.py          # Embedder protocol, model registry
│       ├── embedder.py       # SentenceTransformerEmbedder factory functions
│       ├── corpus_embedder.py # CorpusEmbedder / EmbeddingCache: multi-worker corpus embedding
│       ├── ingest/          # Documents → staged Markdown corpus (ingest extra)
│       │   ├── __init__.py
│       │   ├── converters.py # Converter protocol, PassthroughConverter, AnydocConverter
│       │   ├── manifest.py   # IngestRecord, IngestManifest, IngestStats — per-file provenance
│       │   └── pipeline.py   # IngestPipeline: walk, convert, stage, dedup by content digest
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
│       │   ├── layout.py     # Layout engine + Fibonacci point distributions
│       │   ├── organic.py    # Space-colonization trees + POV-Ray export (viz3d-render extra for meshes)
│       │   └── qt.py         # Qt render lifecycle: worker thread, session, cast path (viz3d-qt extra)
│       └── analysis/
│           ├── __init__.py   # load_scores, available_metrics, ScoreSet
│           └── scores.py     # Read persisted centrality out of SQLite
└── tests/                    # One test module per source module; conftest.py
                              # forces Qt onto the offscreen platform
```

---

## Document ingestion

Turn a folder of mixed-format documents into a corpus any KGModule builder can
consume:

```python
from kg_utils.ingest import IngestPipeline

pipeline = IngestPipeline(staging_root="corpora/specs")
stats = pipeline.run(["~/Documents/specs", "handbook.docx"])

print(f"{stats.ingested} staged, {stats.skipped} skipped, {stats.failed} failed")
```

Then build over the staged corpus as usual — `dockg build --repo corpora/specs`.

This module is the library layer; the command-line surface lives one level up
in [kg-rag](https://github.com/Flux-Frontiers/KGRAG), whose `kgrag ingest`
stages a source tree, registers the corpus, and runs `dockg build` over it in
one step. kgmodule-utils itself ships no CLI.

A run rebuilds the staging corpus from nothing by default — the same contract
as `dockg build` and `pycodekg build`. The corpus therefore reflects exactly the
sources given: a document removed upstream does not linger as a phantom, and a
converter upgrade is picked up with no special flag.

Pass `update=True` for the incremental path (`dockg build --update`,
`pycodekg update`), which keeps what is already staged and converts only what is
new. The trade is that a source deleted upstream keeps its staged copy.

Either way, sources are deduplicated by the SHA-256 of their *bytes* rather than
their filename, so the same document arriving twice under different names is
ingested once.

Every file examined is accounted for, including the ones that did not make it —
`anydoc` performs no OCR, so scanned PDFs are skipped rather than converted, and
a corpus should say so rather than simply lack them:

```python
for record in pipeline.manifest().problems():
    print(f"{record.source_path}: {record.status} — {record.reason}")
```

The same records live in `<staging_root>/.ingest/manifest.json`, with the
converter and version that produced each staged file.

For the end-to-end pipeline — converters, staging layout, manifest schema,
re-run semantics, and the `kgrag ingest` command that drives build and
registration — see
[KGRAG's ingestion guide](https://github.com/Flux-Frontiers/KGRAG/blob/main/docs/INGESTION.md).

## Development

Requires Python 3.12 or 3.13 (`requires-python = ">=3.12,<3.14"`); CI builds on
3.12.

```bash
git clone https://github.com/Flux-Frontiers/KG_utils.git
cd KG_utils
poetry env use python3.12
poetry install --with dev --extras "semantic" --extras "synthesis" --extras "viz" \
  --extras "lancedb" --extras "viz3d-render" --extras "viz3d-qt"
```

**The extras are not optional for testing.** The core install is deliberately
zero-dependency (`dependencies = []`), so `poetry install --with dev` on its own
installs no runtime packages at all — pytest then aborts during *collection* on
missing `numpy` and `httpx` and runs nothing. The six extras above are what CI
installs: the test job omits `lancedb` (the vector-backend tests no longer need
it), but the type-check job requires it so ty can resolve the legacy backend's
imports — and pre-commit runs ty, so install it locally.

Run the fast test suite (no model downloads) — **650 passed, 1 skipped** (the
skip is a test that only runs when PyVista is *absent*):

```bash
poetry run pytest -m "not integration"
```

The Qt suite runs on the offscreen platform automatically (`tests/conftest.py`
sets `QT_QPA_PLATFORM=offscreen`), so no windows appear during a test run. To
see real widgets while debugging one, override it:
`QT_QPA_PLATFORM=cocoa poetry run pytest tests/test_viz3d_qt.py`.

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

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21284866-blue.svg)](https://doi.org/10.5281/zenodo.21284866)

**APA**

> Suchanek, E. G. (2026). *kgmodule-utils: Shared SDK for the KGModule Knowledge-Graph Ecosystem* (Version 0.18.1) [Software]. Flux-Frontiers. https://doi.org/10.5281/zenodo.21284866

**BibTeX**

```bibtex
@software{suchanek_kgmodule_utils,
  author    = {Suchanek, Eric G.},
  title     = {{kgmodule-utils}: Shared SDK for the KGModule Knowledge-Graph Ecosystem},
  version   = {0.18.1},
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
