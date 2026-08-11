# Release Notes — v0.11.0

> Released: 2026-08-11

One new sub-package: **`kg_utils.viz3d`**, the shared 3-D graph layout engine. Nothing
breaks — this is additive, behind a new `viz3d` extra.

## `kg_utils.viz3d` — 3-D layout, shared

The layout engine has lived in `pycode_kg.layout3d` since PyCodeKG grew a 3-D viewer.
GutenbergKG has been importing it from there — `Layout3D`, `LayoutNode`, `LayoutEdge`,
`fibonacci_sphere`, `fibonacci_annulus` — and paying a full `pycode-kg` dependency in its
`viz3d` extra for five symbols that have nothing to do with parsing Python. It now lives
here, where every KG module can reach it on equal terms.

```bash
pip install 'kgmodule-utils[viz3d]'
```

The extra is numpy and nothing else. A layout maps nodes and edges onto
`{node_id: [x, y, z]}` and draws nothing, so pyvista, Qt, and the rest of the renderer
stack stay in whichever module actually opens a window.

```python
from kg_utils.viz3d import AlliumLayout, FunnelLayout, LayoutEdge, LayoutNode

layout = FunnelLayout(
    layer_gap=12.0,
    zlevels={"document": 0, "section": 1, "chunk": 2},
    level_sizes={0: 1.8, 1: 0.9, 2: 0.28},
)
positions = layout.compute(nodes, edges)
```

What ships:

| | |
|---|---|
| `Layout3D` | ABC — implement `compute(nodes, edges)` and you have a layout |
| `AlliumLayout` | Each root node becomes a Giant Allium: a stem with a Fibonacci-sphere head of its children |
| `FunnelLayout` | Node kind picks the Z layer; golden-angle disc spiral within each layer |
| `LayoutNode` / `LayoutEdge` | Domain-neutral DTOs with `from_dict()` |
| `fibonacci_sphere()` / `fibonacci_annulus()` / `golden_spiral_2d()` | Even point distributions, for rolling your own |

`golden_spiral_2d()` is newly public — it was `_golden_spiral_2d`, private for no better
reason than that only one layout happened to use it.

### Domain coupling became arguments

The old `FunnelLayout` imported `pycode_kg.theme` to learn that modules sit at Z level 0
and classes at Z level 1. That is a fact about Python code, not about knowledge graphs, so
it is now supplied by the caller:

```python
AlliumLayout(root_kind="document", contains_rel="HAS_SECTION")
FunnelLayout(zlevels=..., level_sizes=..., default_level=...)
```

Defaults preserve the previous behaviour, with one deliberate exception: `zlevels`
defaults to `None`, which lays every node out on a single flat disc. A domain that has
declared no hierarchy should render as having none, rather than silently inheriting
Python's.

## Also

`kg_utils/__init__.py` listed three of seven optional extras and still described
`[semantic]` as installing `lancedb`, which stopped being true in 0.10.0. It now lists
them all, correctly.

## Upgrading

Nothing to do. If you want the layouts, add the extra:

```bash
pip install 'kgmodule-utils[viz3d]'
```
