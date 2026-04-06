
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: Elastic-2.0](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](https://www.elastic.co/licensing/elastic-license)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/Flux-Frontiers/KG_utils/releases)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

# kg-utils

**kg-utils** — Shared types and snapshot infrastructure for the KGModule SDK.

*Author: Eric G. Suchanek, PhD*

*Flux-Frontiers, Liberty TWP, OH*

---

## Overview

kg-utils is the **zero-dependency foundation package** for the Flux-Frontiers knowledge-graph ecosystem. It provides the canonical type abstractions and temporal snapshot infrastructure that all KGModule implementations — [PyCodeKG](https://github.com/Flux-Frontiers/pycode_kg), [FTreeKG](https://github.com/Flux-Frontiers/ftree_kg), [DocKG](https://github.com/Flux-Frontiers/doc_kg), [AgentKG](https://github.com/Flux-Frontiers/agent_kg) — depend on.

Every KGModule shares the same `NodeSpec`, `EdgeSpec`, `KGExtractor`, and `KGModule` base classes defined here, ensuring consistent interfaces across the ecosystem. The snapshot subsystem enables temporal metric tracking, delta comparison, and pruning across git commits.

---

## Features

- **Core type abstractions** — `NodeSpec`, `EdgeSpec`, `QueryResult`, `SnippetPack` dataclasses for knowledge-graph nodes, edges, and query results
- **KGExtractor base class** — Abstract interface for domain-specific extractors with `extract()`, `node_kinds()`, `edge_kinds()`, and `coverage_metric()`
- **KGModule base class** — Abstract interface for knowledge-graph modules with `build()`, `query()`, `pack()`, `stats()`, and `analyze()`
- **Snapshot models** — `Snapshot` dataclass keyed by git tree hash with free-form metrics, hotspots, issues, and delta tracking
- **SnapshotManager** — Capture, persist, load, list, diff, and prune snapshots with automatic deduplication and delta computation
- **SnapshotManifest** — Fast-lookup index of all snapshots with format versioning
- **Zero dependencies** — Stdlib-only; no external packages required at runtime

---

## Installation

**Requirements:** Python ≥ 3.12, < 3.14

### Standalone (pip)

```bash
pip install 'kg-utils @ git+https://github.com/Flux-Frontiers/KG_utils.git'
```

### Existing Poetry project

```bash
poetry add 'kg-utils @ git+https://github.com/Flux-Frontiers/KG_utils.git'
```

Or declare it directly in your `pyproject.toml`:

```toml
[tool.poetry.dependencies]
kg-utils = {git = "https://github.com/Flux-Frontiers/KG_utils.git"}
```

---

## Quick Start

### Types — Define a KGModule

```python
from kg_utils.types import NodeSpec, EdgeSpec, KGExtractor, KGModule

class MyExtractor(KGExtractor):
    def node_kinds(self) -> list[str]:
        return ["module", "function", "class"]

    def edge_kinds(self) -> list[str]:
        return ["CONTAINS", "CALLS", "IMPORTS"]

    def extract(self, source_root: str):
        # Yield NodeSpec and EdgeSpec objects from your domain
        yield NodeSpec(
            node_id="fn:main:hello",
            kind="function",
            name="hello",
            qualname="main.hello",
            source_path="main.py",
            docstring="Greet the user.",
        )
        yield EdgeSpec(
            source_id="mod:main",
            target_id="fn:main:hello",
            relation="CONTAINS",
        )
```

### Snapshots — Track metrics over time

```python
from kg_utils.snapshots import SnapshotManager

mgr = SnapshotManager(snapshots_dir=".my_kg/snapshots", package_name="my-kg")

# Capture a snapshot from current metrics
snapshot = mgr.capture(metrics={
    "total_nodes": 142,
    "total_edges": 387,
    "coverage": 0.78,
})

# Save with automatic deduplication
mgr.save_snapshot(snapshot)

# List and compare
snaps = mgr.list_snapshots(limit=5)
delta = mgr.diff_snapshots(key_a=snaps[0].key, key_b=snaps[-1].key)
```

---

## API Reference

### `kg_utils.types`

| Class | Description |
|---|---|
| `NodeSpec` | Dataclass for KG nodes: `node_id`, `kind`, `name`, `qualname`, `source_path`, `docstring` |
| `EdgeSpec` | Dataclass for KG edges: `source_id`, `target_id`, `relation` |
| `QueryResult` | Container for query responses with nodes, edges, and metadata |
| `SnippetPack` | Extended result container with source-code snippets |
| `KGExtractor` | Abstract base class for domain extractors |
| `KGModule` | Abstract base class for knowledge-graph modules |

### `kg_utils.snapshots`

| Class | Description |
|---|---|
| `Snapshot` | Temporal snapshot keyed by git tree hash with free-form metrics and deltas |
| `SnapshotManager` | Capture, persist, load, list, diff, and prune snapshots |
| `SnapshotManifest` | Index of all snapshots with format versioning and fast lookup |
| `PruneResult` | Summary of pruning operations: removed, orphaned, broken entries |

---

## Project Structure

```
KG_utils/
├── LICENSE
├── README.md
├── pyproject.toml
├── pytest.ini
├── src/
│   └── kg_utils/
│       ├── __init__.py
│       ├── py.typed                  # PEP 561 marker
│       ├── types/
│       │   ├── __init__.py           # Public re-exports
│       │   ├── specs.py              # NodeSpec, EdgeSpec, QueryResult, SnippetPack
│       │   ├── extractor.py          # KGExtractor ABC
│       │   └── module.py             # KGModule ABC
│       └── snapshots/
│           ├── __init__.py           # Public re-exports
│           ├── models.py             # Snapshot, SnapshotManifest, PruneResult
│           └── manager.py            # SnapshotManager
└── tests/
    ├── __init__.py
    ├── test_types.py
    └── test_snapshots.py
```

---

## Development

```bash
git clone https://github.com/Flux-Frontiers/KG_utils.git
cd KG_utils
poetry install --with dev
```

Run the test suite:

```bash
poetry run pytest
```

---

## License

[Elastic License 2.0](https://www.elastic.co/licensing/elastic-license) — see [LICENSE](LICENSE).

Free to use, modify, and distribute. You may not offer the software as a hosted or managed service to third parties. Commercial use internally is permitted.
