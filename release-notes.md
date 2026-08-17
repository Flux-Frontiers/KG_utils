# Release Notes — v0.16.0

> Released: 2026-08-16

One function changes shape. `cast_scene_to_looking_glass` was extracted from
`gutenberg_kg` in 0.15.0 with exactly one caller; `pycode_kg` became the second
within hours, and the second consumer showed the seam had been drawn through
the middle of the cast rather than around it. The function now owns the whole
button press and returns a `CastResult` instead of a tuple the caller has to
interpret. This is breaking for anyone calling the tuple form — which is both
viewers — and the fix in each is a deletion.

## What changed

**A signature every caller had to wrap.** `build_scene` was annotated
`Callable[[Any], None]`, but neither consumer's scene builder returns `None` —
both return a `(plotter, label, meta)` triple. Neither could pass its builder
directly, `ty` rejects a lambda in that position, and so both repos declared a
named wrapper function whose entire job was to discard a return value. The
parameter is now `Callable[[Any], object]`, the idiomatic way to say the result
is ignored, and both wrappers can go.

**The bookends had stayed duplicated.** After the 0.15.0 migration, each
viewer's `cast_to_looking_glass` still contained the same three-way branch on
`(path, error)`, phrased identically down to the parenthetical asking whether
Bridge is running, wrapped in the same `perf_counter` timing. That wording now
lives in the SDK: `CastResult` carries `path`, `error`, `elapsed`, and a
`message` a status bar can display unread. The consumer keeps only what is
genuinely its own — which nodes to draw, where the file lands, which button to
grey out.

**Two constants that were never domain claims.** Both repos declared
`QUILT_SPEC = "16-landscape"` and `CAST_SCALE = 0.5` with the same justifying
comment. Neither says anything about a corpus: the scale is a fact about how
Looking Glass Bridge's decode time grows with PNG area, and the preset is which
panel happens to be plugged in. They ship as `DEFAULT_QUILT_PRESET` and
`DEFAULT_CAST_SCALE`, and `spec` is now optional — omit it and the default
preset is resolved and scaled for you.

## Upgrading

Callers of `cast_scene_to_looking_glass` must stop unpacking a tuple:

```python
result = cast_scene_to_looking_glass(build, camera, out_stem, progress=step)
if result.path is None:
    logger.error("Cast failed: %s", result.error)
self.visualizer.status = result.message
```

Passing a `spec` explicitly still works and still wins over the default. Scene
builders that return a value no longer need a wrapper, and the local `elapsed`
timing around the call can go — `CastResult.elapsed` already has it.

Two things deliberately stayed in the consuming repo, because they are the only
parts that are actually repo-specific: the `try: from quiltwright import …
except ImportError` guard, whose message names your package's extra, and the
"nothing to cast" check, whose attribute differs per viewer.

Nothing else in the SDK changed, and no rebuild of any index is required.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
