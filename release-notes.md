# Release Notes -- v0.32.0

> Released: 2026-11-24

`pack()` no longer serves text from a source file that changed after the graph
was built. A node's snippet is cut from its live source file by the line
numbers stored at build, so an edited file used to return someone else's lines
under the node's name, with nothing to say they were wrong. This release also
removes the image-generation half of `kg_utils.synthesis`, which now lives in
imagewright. That removal is a breaking change for anyone who imported it from
here.

## What changed

**Source staleness check.** `build_graph()` and `sync()` record a sha256 for
every source file the graph points at, in a new `sources` table in
`graph.sqlite`. `pack()` hashes the files it already reads; when a file no
longer matches, it omits every snippet from that file, still returns the
nodes, and adds one warning per file asking for a rebuild. The new
`KGModule.stale_sources()` reports which files changed or went missing, for
status commands. This covers the KGs that quote their source files at pack
time: pycode_kg, tscode_kg, swift_kg, vault_kg and genealogy_kg. Through kgrag,
the stale text never reaches a federated answer, because kgrag already drops
nodes that carry no snippet.

**Image generation moved to imagewright.** `ImageSynthesizer`, `ImageBackend`,
`ImageConfig`, `image_config_from_env`, `image_synthesizer_from_env`,
`image_synth_for_backend` and the `[synthesis-mflux]` extra are gone, with no
re-export; imagewright carries the same code. The text side stays, including
`TextSynthesizer.rewrite_for_image()` and both image rewrite prompts. The
worker's `imagine` operation accepts any image backend with imagewright's
shape. Without mflux, the lock no longer limits installs to macOS and Linux.

## Upgrading

Stores built before this release have no recorded hashes and behave exactly as
before. Rebuild a store to turn the check on. A tool that writes the store
directly through `GraphStore.write()`, as pycode_kg's CLI does, should call
`GraphStore.record_sources(repo_root)` after its last write.

If you imported image classes from `kg_utils.synthesis`, import them from
`imagewright` instead, and replace the `kgmodule-utils[synthesis-mflux]` extra
with imagewright's own install. Nothing else needs to change.

Every fleet consumer's full suite passed against this release before the tag:
15 repositories, 6,912 tests, no regressions.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
