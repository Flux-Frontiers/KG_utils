# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Converters report their version off the instance, so failures record it
  too.** `Converter` gained a `version` property, read from distribution
  metadata rather than the imported module — `AnydocConverter().version`
  resolves without importing `anydoc` at all, and answers `"unknown"` rather
  than raising when the extra is absent.

  The manifest previously named the converter on a failed record but not its
  version, so "which version rejected this file" was unanswerable from the
  ledger — a real hole in a structure whose whole job is provenance. Found
  while documenting the manifest schema. (0.17.0 on PyPI does not contain
  this fix; it shipped after that release.)

## [0.17.0] - 2026-08-18

### Added

- **`kg_utils.ingest` — the acquisition stage every KG module was missing,
  behind a new `ingest` extra.** Until now a corpus had to *already* exist as
  Markdown or plain text on disk; `IngestPipeline` walks arbitrary sources,
  converts what it can to Markdown, and materializes a staging corpus that the
  existing builders consume unchanged. `dockg build`, `memorykg build` and
  every other `KGModule` gained multi-format ingestion without a line of
  builder code changing.

  Conversion is `anydoc` (PyPI `firecrawl-anydoc`), a Rust library with Python
  bindings that emits consistent GitHub-Flavored Markdown for Word,
  PowerPoint, Excel, OpenDocument, RTF, EPUB, CSV and text-based PDF — a real
  40-page PDF converts in ~20 ms. Markdown, plain text and reStructuredText
  take a passthrough path that needs no dependency at all, which is why the
  extra is optional rather than folded into the core.

  Passthrough deliberately **preserves the source suffix**: a `.txt` stays
  `.txt` in the staging corpus rather than becoming `.md`, because DocKG parses
  the two differently and silently promoting flat text to Markdown would invent
  a heading hierarchy the document never had.

- **An ingest manifest, so a corpus explains its own gaps.** Every file a run
  *examines* gets an `IngestRecord` — not just the ones that succeed — carrying
  the source path, the SHA-256 of the source bytes, the converter and its
  version, a timestamp, and a status of `ingested`, `skipped` or `failed` with
  a reason. `IngestManifest.problems()` returns exactly the documents the KG
  does not contain and why.

  This is the counterpart to a real failure mode: DocKG's PDF path caught its
  parse error and `continue`d, so an unparseable PDF vanished from the corpus
  with no record anywhere. A corpus that quietly omits three PDFs is
  indistinguishable from one that was never shown them. `anydoc` performs no
  OCR, so scanned PDFs are a routine occurrence rather than an edge case, and
  they now surface as a skip with a reason instead of an absence.

  The manifest is plain JSON at `<staging_root>/.ingest/manifest.json` —
  diffable in review, readable without this library, written atomically so an
  interrupted run cannot leave a half-written ledger.

- **A run rebuilds from nothing by default; `update=True` is the incremental
  path.** The same contract the fleet's builders already settled on —
  `dockg build` / `dockg build --update`, `pycodekg build` / `pycodekg update` —
  where the wipe is implicit and the incremental path is the named opt-in.

  Defaulting to a rebuild is what keeps the corpus honest, and for the reason
  pycode_kg gave when it made the same change: it eliminates the phantom
  footgun where deleted or renamed sources silently persist. One layer up the
  effect is identical — under an incremental default, a document removed
  upstream leaves its staged copy behind forever and keeps being built into the
  KG. It also means a converter upgrade needs no special flag.

  Dedup is keyed on the SHA-256 of source *bytes* rather than the filename, so
  the same document arriving twice under different names is ingested once. It
  needs no flag of its own: a rebuild starts from an empty manifest, so the one
  check deduplicates within a run and, in update mode, across runs.

  A recorded digest is only a no-op while its staged file still exists — delete
  a staged document and the next run restores it, under its original name
  rather than a collision-suffixed one. Two *different* documents that would
  land on the same staged name get a short digest suffix instead of one
  overwriting the other, and a staging root nested inside a source tree is
  never fed its own output.


## [0.16.0] - 2026-08-16

### Changed

- **`cast_scene_to_looking_glass` now owns the whole button press.** It returns
  a `CastResult` — `path`, `error`, `elapsed`, and the `message` to put on a
  status bar — instead of a `(path, error)` tuple the caller had to branch on,
  takes `build_scene` as `Callable[[Any], object]` so a builder that returns
  its plotter needs no wrapper, and defaults `spec` to `DEFAULT_QUILT_PRESET`
  scaled by `DEFAULT_CAST_SCALE`.

  **Breaking for callers of the tuple form**, which is both consumers. The
  function was extracted with one caller; the second showed the seam was drawn
  in the middle of the cast: `pycode_kg` and `gutenberg_kg` each still carried
  the same three-way `(path, error)` branch in the same words, the same
  `QUILT_SPEC = "16-landscape"` and `CAST_SCALE = 0.5`, and a named `-> None`
  wrapper whose only job was to discard a scene builder's return value —
  because `ty` rejects a lambda there. A signature every caller must wrap is
  drawn wrong.

  Neither constant is a claim about a corpus: `CAST_SCALE` is a fact about
  Bridge's PNG decode time, `QUILT_PRESET` is which panel is plugged in. They
  are exported as overridable defaults rather than something each viewer
  redeclares.

## [0.15.0] - 2026-08-16

### Added

- **`kg_utils.viz3d.qt` — the Qt machinery a viewer needs to ray-trace and
  cast, behind a new `viz3d-qt` extra.** `PovRenderWorker` keeps POV-Ray off
  the GUI thread, `PovRenderSession` owns the whole lifecycle around it (temp
  views directory, progress bar fed by counting files, cleanup, teardown),
  `ImagePopup` previews the result, and `cast_scene_to_looking_glass` runs the
  PyVista build → render → write → cast path.

  This is machinery, not wiring. Which node becomes a trunk stays per-repo;
  a thread wrapper and a progress loop carry no domain claim, and `pycode_kg`
  and `gutenberg_kg` had already copied the cast path between them closely
  enough that only the widget names differed (`cast_btn` vs `cast_button`).
  `PovRenderSession` therefore takes its widgets as constructor arguments
  rather than being a mixin that assumes one repo's attribute names.

  `PovRenderSession.shutdown()` fixes a crash the copies shared: closing the
  window during a render left a running `QThread` to be destroyed, which
  aborts the process, and let queued signals land on deleted widgets. It
  disconnects before waiting, and parks a worker it cannot stop in time.

  **Not re-exported from `kg_utils.viz3d`** — importing a layout must not
  require PyQt. Unlike `organic.py`, which spans `viz3d` and `viz3d-render` by
  importing PyVista inside its functions, Qt cannot be deferred that way:
  these classes subclass `QThread`/`QDialog`/`QObject`, so the base class must
  exist when the class body runs. `TestExtraBoundaries` pins the boundary.

- **Tag pushes publish to PyPI.** The Release workflow now hands the built
  wheel and sdist to a `publish` job that uploads them through PyPI trusted
  publishing (OIDC) — the same files the GitHub Release carries, no API token
  in the repository secrets. Releases no longer end with a manual upload.

### Changed

- CI covers the new extra: the type-check job installs `viz3d-qt` so ty can
  resolve the PyQt5 and quiltwright imports, and the test job runs the Qt
  suite headless. `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` for
  every run, local ones included, so widget windows no longer flash up during
  a test pass; override the variable to see real widgets while debugging.

- `quiltwright` floor raised to `>=0.6.0` for the Qt cast path.

## [0.14.0] - 2026-08-16

### Changed

- **BREAKING: `lancedb_dir` is now `vectors_path`.** The parameter named for the
  retired backend was load-bearing for the *current* one — `KGModule` derived
  `vectors.sqlite` from `self.lancedb_dir.parent`, so every downstream signature
  kept LanceDB's name for a sqlite-vec store. `pycode_kg` had zero `lancedb`
  occurrences in its own source and still could not shed the name, because
  `SemanticIndex` handed it back.

  Three classes change, all of them taking a *file* path where they used to take
  a *directory*:

  | Class | Was | Now |
  |---|---|---|
  | `kg_utils.pipeline.KGModule` | `lancedb_dir` (3rd positional) | `vectors_path` |
  | `kg_utils.semantic.SemanticIndex` | `lancedb_dir` (1st positional) | `vectors_path` |
  | `kg_utils.module.KGModule` | `lancedb_dir` | `vectors_path` |

  Passing `lancedb_dir=` raises `TypeError`. No deprecation period, matching how
  memory-kg 0.7.0, Metabo_kg 0.10.0 and diary-kg 0.95.0 each dropped the same
  parameter — that hard break is what surfaced stale call sites in those repos
  rather than letting them fail silently later.

  `vector_backend="auto"` still finds an un-migrated LanceDB store: the probe
  path is now derived as `vectors_path.parent / "lancedb"` behind the private
  `KGModule._legacy_store_dir`. It is deliberately not a constructor argument —
  exposing it was the whole reason the name propagated.

  `SemanticIndex.build()` returns `vectors_path` in its stats dict where it
  returned `lancedb_dir`; `repr()` changes to match.

- **`SemanticIndex` no longer defaults to `LanceDBBackend`.** With no explicit
  `backend=`, `_get_backend()` built a LanceDB store — years after the fleet
  migrated off it. It now builds a `SqliteVecBackend` over `vectors_path`.
  `KGModule` always passes an explicit backend, so no fleet code path hit this;
  bare use of `SemanticIndex` did. Reading a pre-migration store now requires
  constructing `LanceDBBackend` explicitly and passing it in, which is what
  `dockg convert-index` already does.

- **The vector-backend tests no longer require LanceDB, and CI now installs
  `viz3d-render`.** Two module-scope import gates were quietly costing most of
  this package's test coverage.

  `tests/test_vector_backend.py` opened with `importorskip("lancedb")`. CI's
  test job installs `semantic`, `synthesis` and `viz` — `semantic` bundles
  sqlite-vec, but nothing installs lancedb — so the whole file skipped and
  `vector_backend.py`, the storage seam under every consumer, ran at **45%**
  with the *default* sqlite-vec backend untested. LanceDB is legacy: the
  default moved to sqlite-vec and nothing in the fleet builds a LanceDB index
  any more, so the real-LanceDB tests are gone rather than the dependency
  being added back. `TestAnnGate` and the `_pq_subvectors` tests stay — they
  drive the ANN gate through a mock table and never call `open()`, so they
  need no install — which matters because `doc_kg` still imports
  `LanceDBBackend` and `_pq_subvectors` for un-migrated stores. Coverage of
  that module is now **63%**, and the docstring names the methods that are
  deliberately unexercised.

  `tests/test_viz3d_povexport.py` had the same shape and a sharper irony: a
  module-scope `importorskip("pyvista")` guarding a file whose stated purpose
  is pinning that `limb_paths`, `leaf_frames` and `LEAF_ASPECT` work *without*
  PyVista. Only three of its thirty tests compare against the PyVista path;
  those now carry `@requires_pyvista` individually and the other twenty-seven
  run everywhere. The "runs without PyVista" claim could never have been
  tested from inside that file — a module-scope skip aborts collection for
  everything after it — so it is now checked in a subprocess with the import
  blocked. CI also installs `viz3d-render`, taking `viz3d/organic.py` from
  **58%** to **94%**.

  Suite total: **81% → 85%**, with 56 tests that previously never ran in CI.

### Added

- **`viz3d.frame_tree` and `viz3d.CameraFrame` — one camera rule for a grown
  tree.** The rule (level view along `-y`, `+z` up, focal point at the bounds
  centre, standing off by 1.5x the vertical extent) existed three times across
  the fleet: `gutenberg_kg/cli/cmd_quilt.py` and `pycode_kg/cli/cmd_quilt.py`
  line for line, plus a NumPy re-derivation in `gutenberg_kg.povscene` written
  because the POV-Ray path has no plotter whose bounds it could read. A fourth
  copy was about to appear the moment `pycode_kg` grew POV-Ray output.

  `CameraFrame` is renderer-independent: a PyVista caller assigns its three
  fields onto `plotter.camera`, a POV-Ray caller converts them. A test asserts
  the returned frame matches the arithmetic those `cmd_quilt` copies perform,
  so this is a consolidation rather than a third behaviour.

  Two details the copies got differently and that are now decided once:
  `include_root` extends the bounds to the origin, because a grown skeleton's
  trunk carries no attractors and framing the crown alone cuts the tree off at
  the ankles; and a zero-height subject falls back to a unit depth instead of
  collapsing the standoff to nothing.

  `frame_tree(..., fov=)` fits the subject's bounding sphere to a lens instead
  of standing off a fixed multiple of its height. The standoff rule is what
  the `cmd_quilt` copies did, and it is right *there*, because PyVista's
  `plotter.reset_camera()` re-fits afterwards. POV-Ray has no such pass, so
  hoisting the rule as-is silently dropped the fitting and the first real
  render came out cropped top and bottom — while every unit test still passed,
  because a badly-fitted frame is a structurally valid one. Omitting `fov`
  keeps the standoff rule, so PyVista callers are unaffected.

  `margin=` then leaves headroom beyond the exact fit, nonzero by default. An
  exact fit puts the silhouette against the frame edge, which reads as cropped
  on a flat render — and on a light-field panel it *is* cropped, because the
  outermost views shear the subject sideways out of a frame with no room to
  give.

  Groundwork for `kgrag_priv/docs/POVRAY_QUILT_ROLLOUT_PLAN.md` — every KG
  module renderable as a quilt through both of quiltwright's backends.


- **`viz3d.limb_paths` and `viz3d.leaf_frames` — the NumPy halves of the two
  mesh builders.** `smooth_paths` and `leaf_glyphs` each do two things: place
  geometry, then hand it to PyVista. The placement is pure NumPy and the
  PyVista part is a detail of one renderer, so a consumer that describes a
  limb analytically — a POV-Ray `sphere_sweep`, say — had no way to reach the
  geometry without dragging in VTK to build a tube it was going to throw away.

  `leaf_frames(positions, skeleton)` returns the clung leaf positions and
  their aim vectors; `leaf_glyphs` now calls it and glyphs the result, so
  there is one copy of the clinging rule rather than two. `limb_paths` is the
  PyVista-free counterpart of `smooth_paths`, splining each root-to-tip path
  through a uniform Catmull-Rom instead of `pv.Spline`. Both stay available
  under the NumPy-only `viz3d` extra.

  `limb_paths` interpolates the same control points as `smooth_paths` but is
  **not** bit-identical to VTK's parametric spline, and says so — when two
  backends must agree to the pixel, call `smooth_paths` once and give both the
  same points rather than letting each smooth its own.

  The two agree in sample *count* for every path of three or more nodes, and
  not for a two-node one: Catmull-Rom needs three control points to curve, so
  `limb_paths` returns those two points unchanged where `smooth_paths` returns
  `subdivisions + 1` along the same straight segment. Both describe that line
  and share both endpoints, so nothing renders differently — but the parity
  test zips the two outputs, and so will callers. Now documented on
  `limb_paths` and pinned by
  `test_a_two_node_path_is_the_one_place_the_sample_counts_differ`, found
  while building the first downstream consumer.

  `limb_paths` also calls `pipe_radii` on a skeleton that has none, which sets
  `skeleton.radii` as a side effect. That matches `smooth_paths`, and the
  docstring now says so rather than leaving it to be discovered.

- **`viz3d.LEAF_ASPECT`** — the per-axis scale that flattens the leaf
  prototype from a ball into a blade, exported so a non-PyVista renderer can
  build the same shape from `leaf_frames` rather than re-deriving it.

  Together these let `quiltwright.povgen` emit an organic tree as analytic
  POV-Ray primitives: 839 KB of SDL for a 3000-leaf tree against roughly
  12.5 MB for the equivalent triangle dump.

- **`viz3d.leaf_facing` and `viz3d.oriented_cluster` — promoted from the two
  consumers.** Both existed as private copies in `gutenberg_kg/scene.py` and
  `pycode_kg/scene3d.py`: pure geometry with no domain knowledge, carried as
  an open item in `VISUALIZATION_STACK.md` and the 2026-08-14 journal since
  `_depth_report` went up into `quiltwright` 0.4.0. Their only dependency,
  `fibonacci_sphere`, was already here.

  `leaf_facing(outward, up_bias=0.6)` gives the direction a limb's foliage
  cluster should face — the limb's own outward run tilted upward, rather than
  world `+z`. A cluster that always points straight up is the clearest tell
  that a tree was assembled rather than grown, and parallax on a light-field
  panel makes that far more obvious than a flat projection does.
  `oriented_cluster(n, centre, facing, radius)` scatters a hemisphere opening
  along that facing, *reflecting* far-side points across the facing plane
  rather than discarding them, so a cluster of any size fills its hemisphere
  evenly and returns exactly the count asked for.

  The two copies were verified equivalent before merging — 500 randomized
  cases matching both originals exactly — with one real divergence:
  **`gutenberg_kg`'s `oriented_cluster` raised `ValueError` on an empty
  cluster** (`fibonacci_sphere` returns `[]`, and subtracting a `(3,)` centre
  from a `(0,)` array fails to broadcast), where `pycode_kg`'s guarded and
  returned `[]`. The promoted version takes the guard, so the latent crash
  does not survive the merge.

  Both assume a `+z`-up world, as the rest of the module does; the docstrings
  now say so.

### Fixed

- **The PyVista-absence test now runs.** `test_mesh_builders_explain_themselves_when_pyvista_is_absent`
  skipped whenever PyVista *was* installed — which is every machine that can run
  the rest of the suite, and CI too once the `viz3d-render` extra was added. So
  the message a caller without the render extra actually sees had no coverage
  anywhere. A test of an absence path cannot be gated on that absence; it has
  to manufacture it, which it now does in a subprocess with the import blocked.
  Confirmed by mutation: changing the install hint fails it.

## [0.13.2] - 2026-08-15

### Fixed

- **`SnapshotManager.repo_root` is assignable again.** 0.13.1 introduced it as
  a read-only property, and a read-only property cannot be assigned — so any
  subclass storing its own repository root raised
  `AttributeError: property 'repo_root' has no setter` at construction. That is
  the natural thing to do whenever a package's data root and its repository
  root differ; `gutenberg_kg`'s manager does exactly that, and the regression
  took out every `gutenkg snapshot` command in a released version.

  `repo_root` returns to a plain attribute set in `__init__`, computed exactly
  as before — the resolved grandparent of `snapshots_dir` — so the default
  behaviour and everything 0.13.1 fixed are unchanged. A subclass may now
  overwrite it after `super().__init__()`, and `_relativize_paths()` honours
  the root the subclass actually means rather than the one inferred from the
  directory layout.

- **A snapshot capture no longer fails on an unusable repository root.**
  Because subclasses may assign anything, `_relativize_paths()` passes metrics
  through untouched when the root is `None` or empty instead of raising
  part-way through a capture. A failed relativization should never fail the
  capture that contains it.

### Removed

- The `Last Revision:` header line from `vector_backend.py`. The field
  invalidates itself: correcting it changes the file, which moves git's
  last-change date to today and makes the corrected header wrong again.
  `git log -1 --format=%cd -- <file>` is exact and free. `Author:` and
  `License:` are kept.

## [0.13.1] - 2026-08-15

### Fixed

- **Snapshots no longer record absolute paths.** `SnapshotManager.capture()`
  now rewrites any metric value that is an absolute path *inside the repo* as
  a repo-relative one, so `db_path` reads `.dockg/graph.sqlite` rather than
  `/Users/<name>/repos/<repo>/.dockg/graph.sqlite`. Snapshots are committed to
  git, so the absolute form published the author's home directory and username
  and made every snapshot machine-specific — two developers rebuilding the same
  tree produced a diff recording only where each of them kept their checkout.

  Measured across the fleet before the fix: 166 committed snapshot files in 8
  repos carried an absolute path. This is the single shared point every KG
  package's snapshot flows through, so the fix reaches all of them, but only
  once each picks up the release. It does not retroactively clean files already
  committed — those need a one-time rewrite per repo.

  Paths *outside* the repo are deliberately left alone: relativizing them would
  emit a `../..` chain that says more about the machine than the original did.

### Added

- **`SnapshotManager.repo_root`** — the repository root, taken as the
  grandparent of a **resolved** `snapshots_dir`. Resolving matters more than it
  looks: `SnapshotManager(".dockg/snapshots")` is the relative form every KG
  package's own docstring demonstrates, and the grandparent of a relative path
  is `.`, inside which nothing ever lies. Without resolving, the rewrite above
  would silently do nothing while reporting success — the guard would fail
  open. Path comparison also retries against the resolved candidate, so a root
  reached through a symlink still matches; on macOS a repo under `/tmp` really
  lives at `/private/tmp`, and the two spellings name one directory.

## [0.13.0] - 2026-08-14

### Added

- **`kg_utils.embedder.TEIEmbedder` — embeddings from a remote HuggingFace
  Text Embeddings Inference server.** Purely additive: no existing class,
  signature or default changes, and nothing selects it unless a caller asks
  for it by name.
  - **Stdlib HTTP only** (`urllib`, `json`) — no torch, no sentence-transformers,
    not even numpy. It is therefore the one embedder that works from a **core,
    zero-dependency install**, because the model runs in the server process.
    Speaks TEI's native `POST /embed`.
  - Honours the fleet contract: `normalize=True` (matching every
    `normalize_embeddings=True` call site) and `truncate=True`, so over-long
    inputs are clipped the way sentence-transformers does silently rather than
    failing the batch.
  - **`dim` never costs a round trip on the hot path.** Pass `dim=` and
    construction does no network I/O at all; omit it and the server is probed
    exactly once, at construction. The probe measures the dimension by
    embedding one string because TEI's `/info` reports `max_client_batch_size`
    and `max_input_length` but *not* the embedding width. This matters because
    `VectorBackend` needs `dim` at table-creation time.
  - **Request batches are clamped to the server's ceiling.** A stock TEI
    defaults to `max_client_batch_size=32` and rejects anything larger with
    HTTP 422 — well below this package's 128-item `DEFAULT_ENCODE_BATCH`. The
    limit is read from `/info` and every call re-chunked to fit, so callers
    keep passing 128 and get transparent splitting instead of an error.
  - **Retries are bounded and failures are loud.** 429 (TEI sheds load rather
    than queueing), 502/503/504 and transport errors retry with exponential
    backoff; 4xx request-shape errors raise immediately rather than burning
    retries. A wrong-width vector or a short response raises instead of being
    written, since either would silently corrupt a vector store — vectors
    misaligned against nodes are far more expensive than an exception.
  - Configured by `endpoint` / `KG_EMBED_ENDPOINT` and `api_key` /
    `KG_EMBED_API_KEY`, following the `synthesis/_config.py` env-fallback
    pattern. A trailing `/v1` is trimmed: TEI's native routes live at the root
    and `/v1` is only its OpenAI-compatible alias.
  - 35 unit tests with a stubbed HTTP layer (no server, no heavy deps) plus two
    `integration` tests that run against a live server when `KG_EMBED_ENDPOINT`
    is set, including a cosine-parity gate against `SentenceTransformerEmbedder`.

  Verified against TEI 1.9.3 serving `BAAI/bge-small-en-v1.5`: vectors match
  sentence-transformers to cosine ≥ 0.999997 with 99.8% top-10 retrieval
  agreement, so the two backends can share one store. **This is not a
  performance upgrade on CPU** — in-process torch measured roughly 2× faster
  (41 vs 19 items/s on 4 shared cores) — its wins are memory (176 MiB vs
  1.5 GiB RSS to serve the same model) and keeping torch out of the client.
  Full evaluation in the kgrag repo at `docs/TEI_EVALUATION.md`.

### Fixed

- **The pre-commit hooks were configured but not installable, and their ruff
  had drifted six minor versions from CI's.** `.pre-commit-config.yaml` and
  `.secrets.baseline` were both checked in, but neither `pre-commit` nor
  `detect-secrets` appeared in the dev group — so `poetry install --with dev`
  gave you the configuration without the tool that runs it, and the hooks only
  ever guarded developers who had installed pre-commit by some other route. CI
  runs no detect-secrets job either, so nothing else covered that gap. Both are
  now dev dependencies at the same versions KGRAG uses.

  Separately, the config pinned `ruff-pre-commit` at **v0.9.10** while the dev
  group resolves **0.15.22** — two different formatters over one tree, which is
  precisely how a hook passes locally and CI fails. Pinned to `v0.15.22` to
  match, with the hook renamed to its current id (`ruff` → `ruff-check`), and a
  comment tying the rev to the dev floor so they are changed together. Still
  below 0.16, per the cap recorded beside that floor.

  Verified: `pre-commit run --all-files` from this project's own venv passes
  all twelve hooks, `ty` and `pytest` included, and `ruff format --check .`,
  `ruff check .` and the 554-test suite are unchanged.

### Changed

- **The ruff rule set is now pinned rather than inherited** (`pyproject.toml`).
  There was no `[tool.ruff.lint]` section, so this project took ruff's
  *defaults* — meaning the rules actually enforced changed whenever ruff did.
  That is not hypothetical: it is precisely how 0.16 arrived carrying 38 new
  findings, and why the dev floor is capped below it. Naming the set closes
  that: a version bump can still change how an existing rule behaves, but it
  can no longer add rules behind your back.

  `select` matches KGRAG's — `E`, `F`, `W`, `I`, `UP`, `B`, `BLE`, `PLC` — so
  the two repos now share one lint contract. Adopting it surfaced 71 findings,
  handled honestly rather than blanket-suppressed:

  - **18 fixed mechanically** — 17 `I001` (unsorted imports) and one
    `PLC0207`. Confined to import ordering and blank lines; the only source
    changes are three files whose imports are now alphabetised.
  - **38 ignored as intentional patterns**, matching KGRAG's own ignores.
    `PLC0415` (34) is this package's architecture, not an oversight — the core
    install is zero-dependency, so numpy, torch and sqlite_vec are imported
    inside the functions that need them, and hoisting them would break that
    guarantee. `BLE001` (4) is deliberate at optional-dependency boundaries.
  - **15 deferred with their reasons recorded in the config**, because each is
    a behaviour question rather than a formatting one: `B905` (12 `zip()` calls
    with no `strict=`, where `strict=True` would raise on ragged input that
    today truncates), `UP042` (2 — `str, Enum` → `StrEnum` changes what
    `str(member)` returns), and `B027` (1 — `KGModule._post_build_hook` is an
    intentional optional no-op that `@abstractmethod` would make mandatory for
    every subclass).

- **The README's documented test install could not run the test suite.**
  `Development` said `poetry install --with dev`, then pointed at
  `pytest -m "not integration"`. Because the core install is deliberately
  zero-dependency (`dependencies = []`), that command installs no runtime
  packages at all, and pytest aborts during *collection* on missing `numpy`
  and `httpx` — running zero tests rather than degrading to a partial run.
  Reproduced in a clean Python 3.12 venv. The section now documents what CI
  actually installs (`--extras "semantic" --extras "synthesis" --extras "viz"`
  → 520 passed, 5 skipped), the two further extras that also cover the LanceDB
  backend and the PyVista renderers (`lancedb`, `viz3d-render` → 554 passed,
  2 skipped), the Python 3.12/3.13 requirement, and how to point the new
  `TEIEmbedder` live tests at a running server.

## [0.12.1] - 2026-08-13

### Added

- **`kg_utils.viz3d.organic` — botanically credible tree skeletons**, promoted
  verbatim from `gutenberg_kg.layout_organic`. Finishes what 0.11.0 started: the
  layouts moved out of `pycode_kg` so drawing a graph no longer required a
  Python source-code analyser, and the tree engine was still trapped in a book
  corpus for the same reason.
  - Space colonization (`colonize`, Runions/Lane/Prusinkiewicz 2007), the
    pipe model (`pipe_radii`, da Vinci's rule), `root_to_tip_paths`,
    `smooth_paths`, `tree_mesh`, `leaf_glyphs`, `crown_spacing`, and
    `grow_tree` as the one-call entry point.
  - The engine takes crown attractors and a root; **the hierarchy is the
    caller's business**. A corpus grows document → section → chunk, a diary
    grows trunk → period limb → entry cluster → leaves. Nothing here knows
    which.
  - `seed_from_slug` → **`seed_from_key`**, and `grow_tree(slug=...)` →
    `grow_tree(key=...)`. The concept was never book-specific; only the name
    was. Renamed on promotion rather than carrying a book noun into a shared
    package.
  - **The `viz3d` extra stays NumPy-only.** Only `smooth_paths`, `tree_mesh`
    and `leaf_glyphs` need PyVista and they import it lazily, so the 13 repos
    that depend on `kgmodule-utils` do not acquire VTK for a layout import.
    Calling one without PyVista raises a `ModuleNotFoundError` naming the
    install command rather than an `AttributeError` on a missing module.
- **New `viz3d-render` extra** — the layouts plus the renderer they need.
  `smooth_paths`, `tree_mesh` and `leaf_glyphs` import PyVista lazily, so
  PyVista is a real runtime dependency of three shipped functions, and it
  belonged to no extra at all: every caller hand-declared it and CI installed it
  by hand so `ty` could resolve the import. It could not simply be added to
  `viz3d`, because `pycode_kg` has already drawn this line inside its own
  package: it takes `kgmodule-utils[semantic,viz3d]` in its *main* dependencies
  for the layouts, and puts `pyvista` and `pyvistaqt` in its own `viz3d` extra
  so that rendering stays opt-in. Adding PyVista to the shared `viz3d` extra
  would quietly undo that, handing VTK to everyone who installs `pycode-kg`
  whether they render or not. Depend on `viz3d-render` to build geometry,
  `viz3d` for coordinates only.

### Changed

### Removed

### Fixed

- **0.12.0 was published without the `viz3d` engine described above.** The
  promotion landed on a branch that the `v0.12.0` tag does not contain, so the
  tag's tree carries only `viz3d/__init__.py` and `viz3d/layout.py` and the
  published wheel has no `kg_utils.viz3d.organic` at all. Importing
  `grow_tree`, `colonize`, `crown_spacing`, `seed_from_key`, `Skeleton`,
  `pipe_radii`, `tree_mesh` or `leaf_glyphs` from 0.12.0 raises `ImportError`.
  This release ships what 0.12.0 was supposed to. 0.12.0 cannot be corrected in
  place — PyPI permanently reserves an uploaded filename, so a fixed 0.12.0
  could not be re-uploaded even after deleting the original.

## [0.12.0] - 2026-08-13

### Added

- **`tests/test_viz3d_layout.py` pins the `AlliumLayout` sizing contracts**
  downstream renderers depend on. pycode_kg's `test_viz3d_sizing` restates the
  head-radius formula as a literal to assert a max-centrality function still
  fits inside a four-child head; since that formula moved here in 0.11.0,
  changing its coefficient would have silently re-tuned every consumer's
  occlusion budget with nothing to catch it. The head and orbit radius formulas
  and layout determinism are now covered here.

### Changed

- **pytest dev pin raised to `>=9.0.3`**, resolving GHSA-6w46-j5rx-g56g /
  PYSEC-2026-1845. The `^8.0.0` cap deferred in the 0.10.x security pass is now
  lifted. Dev-group only — pytest is not part of any published extra and does
  not appear in the wheel or sdist metadata, so no released artifact changes.
- **`AlliumLayout` documents its node-ordering requirement.** Roots take their
  annulus slots in the order they appear in the node list, so callers must pass
  a stable order or the scene shuffles between renders even when the graph is
  unchanged. Sorting internally would make this moot but would relocate every
  node in every existing scene, so the behaviour is pinned and documented
  rather than changed.

## [0.11.0] - 2026-08-11

### Added

- **`kg_utils.viz3d` — shared 3-D graph layout**, behind a new `viz3d` extra
  (numpy only). Moved out of `pycode_kg.layout3d`, where gutenberg_kg was
  already importing it — paying a full `pycode-kg` dependency for five symbols
  it shares with no other part of that package.
  - `Layout3D` ABC, `LayoutNode` / `LayoutEdge` DTOs, `AlliumLayout`,
    `FunnelLayout`.
  - `fibonacci_sphere()`, `fibonacci_annulus()`, and `golden_spiral_2d()`
    (promoted from the private `_golden_spiral_2d`) for building your own
    layout.
  - The layouts return coordinates and draw nothing, so pyvista and Qt stay in
    whichever module actually opens a window.
  - Domain coupling is now constructor arguments rather than an import:
    `AlliumLayout(root_kind=..., contains_rel=...)` and
    `FunnelLayout(zlevels=..., level_sizes=..., default_level=...)`. Defaults
    preserve the previous code-graph behaviour except for `zlevels`, which now
    defaults to a flat disc — a domain that declares no hierarchy should render
    as having none rather than inheriting Python's.

### Changed

- `kg_utils/__init__.py` now lists every optional extra, not three of seven, and
  no longer claims `[semantic]` installs `lancedb` — that stopped being true in
  0.10.0.

## [0.10.0] - 2026-08-03

### Removed

- **BREAKING: `lancedb` is no longer part of the `semantic` extra.** It now has
  its own `lancedb` extra. `auto_resolve_backend()` selects sqlite-vec for every
  fresh or already-migrated store and falls back to LanceDB only when an
  un-migrated LanceDB store is found on disk, and no adapter in the fleet uses
  `LanceDBBackend` directly (pycode_kg imports `SqliteVecBackend` explicitly;
  doc_kg declares `lancedb` as its own direct dependency), so shipping it in
  every `[semantic]` install was dead weight. If you have an un-migrated LanceDB
  store, install `kgmodule-utils[semantic,lancedb]`.

### Changed

- **Pinned `rich` to `>=13.0.0,<15.0.0`** rather than leaving the upper bound
  open.
- **CI typecheck installs the `lancedb` extra.** `vector_backend.py` still
  imports lancedb lazily for `LanceDBBackend`, so `ty` needs it present to
  resolve the import even though it left the `semantic` extra.

### Added

- **`docs/viz-bootstrap-selfcontainment.md`** — the measured rationale behind the
  Bootstrap shim, retained so the copied CSS rules are not mistaken for
  guesswork.

### Fixed

- **`kg_utils.viz.build_graph_html` output is now genuinely self-contained.**
  pyvis 0.3.2 hardcodes two Bootstrap CDN tags in its own template, and
  `cdn_resources="in_line"` governs only the vis-network assets, so every
  generated page fetched `bootstrap.min.css` and `bootstrap.bundle.min.js` from
  jsdelivr — breaking offline with a collapsed layout, and silently contacting a
  third party for a page that may be generated from a private codebase.
  `_inline_bootstrap()` now strips both tags and supplies the two rules the page
  actually uses (`.card`, `.card-body`), copied verbatim from
  `bootstrap@5.0.0-beta3`, plus the `box-sizing` and `body { margin: 0 }` Reboot
  rules that are load-bearing for the box model. Measured pixel-identical to real
  Bootstrap with zero external requests; page size is unchanged. The
  `bootstrap.bundle.min.js` payload was entirely unused and is dropped.
  **Pages generated on 0.7.0–0.9.0 should be regenerated.**
- **`test_output_is_self_contained` now tests the property it claims.** It
  asserted the absence of one named CDN host, which stayed true while the page
  reached out to a different one; it is now host-agnostic and fails on any
  external reference.

## [0.9.0] - 2026-07-28

### Security

- **Unpinned `transformers` to `>=5.5.0,<6`, clearing two high-severity
  advisories.** The previous `<4.57` cap held the stack at 4.56.2, exposed to a
  remote-code-execution advisory (fixed in 5.3.0) and an arbitrary-code-execution
  flaw in the LightGlue model-loading path (fixed in 5.5.0). The cap had no
  recorded rationale and no longer matched the fleet (doc-kg had already shipped
  on transformers 5.6.2). Verified against 5.14.1: embeddings are bitwise
  identical on bge-small, bge-large, and nomic-embed (including empty, unicode,
  and CRLF inputs), a full index rebuild is byte-identical, and queries against a
  4.x-built index return identical rankings — **no re-index required**.

### Changed

- **`transformers` now requires 5.x** (`>=5.5.0,<6`; was `>=4.40.0,<4.57`).
  Installing the `semantic` extra pulls transformers 5; environments pinned to
  transformers 4.x must upgrade. No API or embedding-output change.
- **Dropped the optional `kgdeps` Poetry group (`pycode-kg`, `doc-kg`).** The two
  siblings each declared the other, and because Poetry locks optional groups too,
  any relaxed transformers pin deadlocked resolution against the published
  siblings — neither could lock until the other released. Neither package is
  imported here, so the dev-only convenience deps are removed (with manual install
  instructions left in `pyproject.toml`), permanently breaking the cycle. Lock
  refreshed to transformers 5.14.1, huggingface-hub 1.25.1, safetensors 0.8.0.

### Fixed

- **HF progress bars and logs no longer leak into builds and queries under
  transformers 5.** transformers ≥5 removed the `transformers.logging` submodule
  alias, so `import_module("transformers.logging")` raised `ModuleNotFoundError`,
  which the surrounding `except` swallowed — silently disabling the
  log/progress-bar suppression and leaking a "Loading weights" bar into every
  build and query. Now imports `transformers.utils.logging`, which resolves on
  both 4.x and 5.x.

## [0.8.0] - 2026-07-26

### Added

### Changed

- **Default vector backend is now `"auto"` (was `"lancedb"`).** `KGModule`
  builds a fresh or already-migrated KG on `sqlite-vec` and only falls back to
  LanceDB when an un-migrated LanceDB store already exists on disk, so existing
  corpora keep working untouched. The `sqlite-vec` dependency is now bundled in
  the `semantic` extra. Pass `vector_backend="lancedb"` to keep the old default.
- **README brought back in sync with the package.** Documents `kg_utils.viz`
  and `kg_utils.analysis` (features list, `[viz]` install snippet, API tables);
  fills in the previously-drifted source tree (`vector_backend`,
  `corpus_embedder`, `retrieval/`, `worker/`, `synthesis/factory`); and reflects
  sqlite-vec as the default vector backend with LanceDB as the legacy option.

### Removed

### Fixed

## [0.7.0] - 2026-07-25

### Added

- **`kg_utils.viz` — shared graph rendering (new `viz` extra).** One
  implementation of the interactive HTML graph, previously duplicated per
  module. Domain differences arrive as data rather than code: a `GraphTheme`
  names a domain's node kinds and edge relations, a `TooltipSpec` names the
  fields worth showing. Only `id`, `kind` and `name` are assumed common — code
  nodes carry `qualname`/`module_path`, document nodes carry `title`/`file_path`,
  metabolic nodes carry `formula`/`ec_number`, and all three render through the
  same path. `build_graph_html` also accepts a plain callable instead of a spec
  for markup the spec cannot express.

  The output inlines vis-network, so a page opens from `file://` and survives
  embedding in a `srcdoc` iframe. pyvis's default `cdn_resources="local"` breaks
  both: it emits relative asset paths that cannot resolve without a base URL, so
  the graph renders only when cdnjs happens to be reachable and otherwise fails
  silently with `vis is not defined`. It also writes a `lib/` directory into the
  working directory on every render.

  `select_nodes` decides which nodes survive a display cap. Seeding on the most
  central nodes and expanding to their neighbours, rather than taking the top N,
  keeps the result connected: measured on a 10k-node code graph at a cap of 150,
  top-N-by-centrality stranded 47 nodes with no edges and halved the edge count.

  Requires `pip install 'kgmodule-utils[viz]'`. The core install stays
  zero-dependency — nothing in `kg_utils/__init__.py` imports it.

- **`kg_utils.analysis.scores` — read persisted centrality back out of SQLite.**
  `available_metrics`, `load_scores`, and a `ScoreSet` exposing raw score, dense
  rank, percentile and range scaling. Stdlib only. Ranks are derived on load
  rather than read from the stored `rank` column, which may reflect a truncated
  ranking, so `centrality_scores` and `node_metrics` behave identically despite
  the latter having no rank column.

### Changed

### Removed

### Fixed

- **Node text could break out of the rendered page.** Graph node data is
  embedded in a `<script>` block, so a node whose text contained `</script>`
  would terminate that block during HTML parsing and inject whatever followed.
  The payload now escapes `<`, `>` and `&` as unicode sequences. This affected
  the per-module renderers this code was consolidated from.

## [0.6.2] - 2026-07-15

### Fixed

- **Removed `ty` from the package's runtime dependencies.** The type checker
  had leaked into the main `dependencies` list (it was the *only* entry —
  every real runtime dep lives in extras), forcing all consumers to install
  it and pinning them to `ty>=0.0.44,<0.0.45` — which broke dependency
  resolution for any downstream repo with its own `ty` pin (kgrag_priv's
  `^0.0.41`). `ty` remains in the dev group where it belongs.

## [0.6.1] - 2026-07-15

### Fixed

- **`KGModule.vectors_path` now derives from `lancedb_dir.parent`, not
  `repo_root/_default_dir`.** When a caller passed `lancedb_dir` explicitly
  (as every pycodekg CLI command does, sometimes with a placeholder
  `repo_root`), the sqlite-vec sidecar path pointed at a nonexistent
  `<repo_root>/<_default_dir>/vectors.sqlite`, so `query` failed with
  `sqlite3.OperationalError: unable to open database file` against a store
  the build had just written. The sidecar now always sits next to the
  lancedb dir (`<kg-dir>/vectors.sqlite`), matching doc_kg's
  `sqlite_vectors_path()` convention. Regression tests cover explicit-path
  construction and the build-then-fresh-instance-query CLI flow.

## [0.6.0] - 2026-07-15

### Added

- **`KGModule.vector_backend`** — the fleet-wide `kg_utils.pipeline.KGModule`
  base class now threads backend selection through to the `SemanticIndex` it
  builds, closing the gap left by 0.5.0's `VectorBackend` seam (which only
  reached doc_kg's heavier subclass). Accepts `"lancedb"` (default, unchanged
  behavior for existing consumers), `"sqlite-vec"`, or `"auto"` (picks
  sqlite-vec for a fresh KG, lancedb only when an un-migrated LanceDB store
  already exists on disk). `KGModule.stats()` now reports the resolved
  `vector_backend` name (path-based; never loads the embedding model).
  New `kg_utils.vector_backend.resolve_backend_name()` / `make_backend()`
  factory helpers back the selection and are reusable outside the pipeline.
- `kg_utils.semantic.META_COLUMNS` — public alias for the code-KG metadata
  column tuple, for domain packages that construct backends directly.

### Changed

### Removed

### Fixed

## [0.5.0] - 2026-07-14

### Added

- **`kg_utils.vector_backend` — a pluggable `VectorBackend` storage seam under
  `SemanticIndex`.** Two implementations ship: `LanceDBBackend` (the historical
  default; dummy-row table bootstrap, `delete`-then-`add` upsert with the
  fresh-table fast path, optional IVF ANN gated on row count) and
  `SqliteVecBackend` (exact brute-force `sqlite-vec`/`vec0` store with a
  `vec_meta` + `vec_nodes` twin-table layout, row-aligned by `rowid`, so a SQL
  `where` compiles to a true prefilter). Neither backend hardcodes domain
  columns — the owning index declares its `meta_columns`. The sqlite store is
  9–11× smaller than LanceDB and exact (recall 1.0) at comparable latency.
- **`SqliteVecBackend` supports fp32 and int8** (`dtype=`), the latter wrapping
  blobs with `vec_int8()` on both insert and match (a raw blob is silently
  parsed as float32).
- `sqlite-vec` optional extra: `pip install 'kgmodule-utils[sqlite-vec]'`
  (pinned `==0.1.9`, pre-1.0).

### Changed

- **`SemanticIndex.__init__` gains a `backend=` parameter** (defaults to a
  lazily-constructed `LanceDBBackend`, so existing callers are unaffected).
  The LanceDB table plumbing (`_open_table`/`_get_table`) moved into
  `LanceDBBackend`.
- **`SemanticIndex.search()` gains a `where: str | None` prefilter parameter**,
  unifying its signature with doc_kg's heavier `SemanticIndex`.

## [0.4.9] - 2026-07-13

### Added

- **`CorpusEmbedder.embed_to_cache(texts, metadata, *, out_path)` — stream shard vectors to
  disk, bounding peak memory by shard size instead of corpus size.** `embed()` holds every
  completed shard's vectors in the parent as nested Python float lists (~5–6× the raw float32
  bytes) until the whole run finishes, so peak RAM scales with total corpus size — on the
  688,852-node Gutenberg consolidated build the parent climbed past 10 GB RSS, drove the
  machine to 45 GB of swap, and per-row embed time rose ~14× mid-run. The new streaming mode
  has each worker write its shard directly to a JSONL part file next to *out_path* (batch by
  batch — worker RAM is bounded by one batch) and return the *path*, not the vectors; the
  parent then concatenates parts in shard order (preserving exact input order, which the
  id↔vector alignment of `build_from_cache` relies on) behind a `__meta__` header line. The
  output is drop-in for doc_kg's `build_from_cache`/`_build_from_jsonl_cache` JSONL format
  (`id, kind, name, title, file_path, text, vector` per row; `.gz` suffix writes gzip).
  Preserves all load-bearing behavior: the GPU→single-process guard, `maxtasksperchild=1` +
  `_RECYCLE_SHARD` worker recycling, sequential fallback on pool failure, and identical
  embedding results (same model, normalization, and nomic task-prefixing — verified
  bit-identical to `embed()` in a 2-worker spawn smoke test). Part files are cleaned up on
  failure. `embed()` is unchanged for callers that want an in-memory `EmbeddingCache`.
  Supersedes gutenberg_kg's per-genre build workaround once wired in downstream.

## [0.4.8] - 2026-07-12

### Changed

- **`kg_utils.corpus_embedder.CorpusEmbedder` default `n_workers` capped at `min(4, cpu_count // 2)`**,
  down from an unbounded `cpu_count // 2`. Each CPU worker loads its own full model copy plus a torch
  runtime (~1.2 GB for bge-small, more for mpnet-class models); on a 20-core machine the old default
  spawned 10 workers and peaked at ~21.5 GB RSS during gutenberg_kg's 241-book build-corpus run, well
  past where throughput stops improving (I/O + accumulator bound before 10 workers). Explicit
  `n_workers` is unaffected. Ported from doc_kg's `feat/embedderworker` branch so every consumer gets
  the fix through the shared implementation instead of it landing in one repo's fork.
- **`_embed_sequential` (the single-process/GPU path) now shows a live progress bar** via a new
  `_InlineProgress` adapter that speaks `_embed_shard`'s existing progress-queue `put()` protocol —
  previously this path (small corpora, or any `mps`/`cuda` run, which always forces single-process)
  embedded silently with no feedback. Same source as the worker-count fix above.

## [0.4.7] - 2026-07-11

### Added

- **`kg_utils.corpus_embedder.CorpusEmbedder` / `EmbeddingCache`.** Canonical home for the
  spawn-safe, multi-worker corpus embedding engine that had been independently forked at
  least three times (doc_kg, memory_kg, diary_kg) — most recently causing a real production
  incident (a 683k-node consolidated build OOM'd on Apple Silicon; see
  `gutenberg_kg/SUMMARY.md`, 2026-06-16/17) that had to be root-caused and fixed in doc_kg
  before the same bug resurfaced, unfixed, in memory_kg's independent copy. Carries forward
  doc_kg's proven fixes: a GPU→single-process guard (`embed()` never fans out to parallel
  workers when the resolved device is `mps`/`cuda` — a GPU allocator can't be shared across
  `spawn` workers, so N workers would stack N allocations into an OOM), shard recycling
  (`_RECYCLE_SHARD=25_000` + `Pool(maxtasksperchild=1)`, so long-lived workers don't
  accumulate allocator/heap/GC state across a large run), gzip cache support, and per-batch
  progress reporting. Downstream modules should import `CorpusEmbedder` from here instead of
  keeping their own copy.
- **`kg_utils.embedder.resolve_device(device=None)`.** Public device-resolution helper
  (explicit arg > `KG_EMBED_DEVICE` env > auto-detect), extracted from the logic
  `load_sentence_transformer` already had inline since 0.4.4. Exists so callers that need to
  gate a *decision* (e.g. `CorpusEmbedder`'s parallel-vs-single-process fan-out) on the
  resolved device can do so without loading a model first. `load_sentence_transformer` now
  calls this instead of duplicating the precedence logic.
- `semantic` extra now includes `rich>=13.0.0` (used by `CorpusEmbedder`'s parallel progress
  bar).

### Changed

- **`kg_utils.semantic` unified onto `kg_utils.embed`/`kg_utils.embedder` instead of carrying
  its own copy of the model registry and embedder classes.** `semantic.py` pre-dated
  `embed.py`/`embedder.py` and had drifted into a fourth independent embedding implementation
  inside kg_utils itself (after doc_kg, memory_kg, and diary_kg's separate forks of
  `CorpusEmbedder`) — its own `Embedder` base class, its own `_KNOWN_MODELS`/
  `resolve_model_path`/`_kg_model_cache_dir`, and a `SentenceTransformerEmbedder` with **no
  device awareness at all**: no `device` parameter, no `KG_EMBED_DEVICE` support, model
  construction had no `device=` argument. pycode_kg is the only consumer, and got none of the
  device-pinning work landed in `kg_utils.embedder`. Now:
  - `DEFAULT_MODEL` and `resolve_model_path` are re-exported from `kg_utils.embed` (removed
    the duplicate `_KNOWN_MODELS`/`_kg_model_cache_dir`/local `resolve_model_path`).
  - `Embedder` and `SentenceTransformerEmbedder` are re-exported from `kg_utils.embedder`
    (removed the local class definitions). `SemanticIndex` only ever called
    `embedder.embed_texts(texts)`/`embed_query(query)`/`.dim` — fully compatible with the
    `kg_utils.embedder` versions (which add an optional `encode_batch_size` kwarg with a
    default, so no call-site change). Consumers now get `KG_EMBED_DEVICE` support for free.
  - **Security tightening, not just consolidation:** the old `SentenceTransformerEmbedder`
    always passed `trust_remote_code=True` to every model load. The unified version gates
    that on `"nomic-ai/" in model_name` (matching `kg_utils.embedder`'s existing, narrower
    policy) — arbitrary-code execution from a model repo's custom code is now opt-in per
    known-safe model family, not unconditional.
  - `_local_model_path()` — a **private** symbol pycode_kg's `cmd_model.py`/`cmd_init.py`
    import directly for the `download-model` CLI command — is kept as a thin
    backward-compat wrapper around `resolve_model_path(model_name, local_fallback=Path.cwd()
    / ".kgcache" / "models")`, preserving its exact prior resolution (CWD-relative
    `.kgcache/models`, `KGRAG_MODEL_DIR` override) so pycode_kg's on-disk model cache
    location doesn't move.
  - New `tests/test_semantic.py` — this module had **zero** prior test coverage; added
    re-export identity tests and `_local_model_path` fallback/override coverage.

## [0.4.6] - 2026-07-09

### Changed

- **`embedder`: default per-call encode batch lowered 512 → 128.** New module constant `DEFAULT_ENCODE_BATCH = 128` now backs `Embedder.embed_texts`, `SentenceTransformerEmbedder.embed_texts`, and the `wrap_embedder` `_WrappedEmbedder.embed_texts` (which previously **hardcoded** `batch_size=512` with no way to override). Transformer attention memory scales with `batch × seq²`, so a large batch on long (near-max-sequence) chunks allocates many GB per `model.encode` call and OOMs / stalls MPS — observed as a 25–32 GB peak on a 528k-node build in a downstream module. Throughput is flat above ~128 on CPU and MPS for the models in use, so this is free; raise `encode_batch_size` only for a large-VRAM CUDA GPU with short sequences. `embed_texts` now takes a uniform optional `encode_batch_size` parameter across the base class, concrete, and wrapped implementations.

## [0.4.5] - 2026-07-07

### Added

- **`TextSynthesizer.complete(messages, *, model=None, max_tokens=None, temperature=0.7)`** —
  public general-purpose chat-completion entry point for callers that build their own message
  list (summarization, classification, arbitrary prompting) instead of using `synthesize_rag`
  or `rewrite_for_image`. Applies the same oMLX thinking suppression and `<think>` stripping as
  the other public methods. Promotes the previously private `_complete` to stable public API so
  downstream KG modules can drive oMLX/Ollama/OpenAI backends without reaching into internals.

## [0.4.4] - 2026-06-17

### Added

- **`load_sentence_transformer(model_name, device=...)`** — explicit device override with
  precedence: explicit arg > `KG_EMBED_DEVICE` env > CUDA→MPS→CPU auto-detect. The env channel
  lets spawn-based embedding workers (which inherit `os.environ` but can't easily take a Python
  arg) be pinned to a device — without it, N parallel workers each auto-select MPS and stack N
  GPU allocations into an OOM. This is what makes CPU multiprocessing embedding safe on Apple
  Silicon.

### Changed

- **`embedder.py`** — replaced `from X import Y` lazy imports with `importlib.import_module()`
  for `sentence_transformers`, `transformers.logging`, `torch`, and `numpy`.  `importlib` returns
  `Any`, so `ty` no longer flags these optional heavy dependencies as unresolved imports.

- **`synthesis/_image.py`** — same `importlib.import_module()` pattern for the `mflux` loader;
  removes the old `# type: ignore` override which is no longer needed.

### Fixed

- **CI `type-check` and `test` jobs** — both jobs now install `--extras "semantic" --extras
  "synthesis"` so that `sentence-transformers`, `transformers`, `torch`, `lancedb`, `httpx`,
  `openai`, and `pillow` are present in the CI virtualenv, matching local pre-commit behaviour.

- **`tests/test_synthesis_image.py`** — corrected four test assertions that still referenced
  the old `dall-e-3` default:
  - expected model updated from `dall-e-3` → `gpt-image-1`
  - landscape size updated from `1792x1024` → `1536x1024`
  - portrait size updated from `1024x1792` → `1024x1536`
  - `test_generate_openai_requests_b64_json` renamed to `test_generate_openai_does_not_set_response_format`
    and now asserts that `response_format` is absent from the OpenAI call kwargs (gpt-image-1
    does not accept this parameter)

## [0.4.3] - 2026-06-08

### Changed

- **`embedder.py`** — replaced `from X import Y` lazy imports with `importlib.import_module()`
  for `sentence_transformers`, `transformers.logging`, `torch`, and `numpy`.  `importlib` returns
  `Any`, so `ty` no longer flags these optional heavy dependencies as unresolved imports.

- **`synthesis/_image.py`** — same `importlib.import_module()` pattern for the `mflux` loader;
  removes the old `# type: ignore` override which is no longer needed.

### Fixed

- **CI `type-check` and `test` jobs** — both jobs now install `--extras "semantic" --extras
  "synthesis"` so that `sentence-transformers`, `transformers`, `torch`, `lancedb`, `httpx`,
  `openai`, and `pillow` are present in the CI virtualenv, matching local pre-commit behaviour.

- **`tests/test_synthesis_image.py`** — corrected four test assertions that still referenced
  the old `dall-e-3` default:
  - expected model updated from `dall-e-3` → `gpt-image-1`
  - landscape size updated from `1792x1024` → `1536x1024`
  - portrait size updated from `1024x1792` → `1024x1536`
  - `test_generate_openai_requests_b64_json` renamed to `test_generate_openai_does_not_set_response_format`
    and now asserts that `response_format` is absent from the OpenAI call kwargs (gpt-image-1
    does not accept this parameter)

## [0.4.3] - 2026-06-08

### Added

- **`_parse_size(size)`** — new helper in `kg_utils.synthesis._image` that parses an explicit
  `"WIDTHxHEIGHT"` string into a `(width, height)` tuple; returns `None` for invalid input.

- **`size` parameter on `ImageSynthesizer.generate()` and `generate_b64()`** — mflux backends
  (`mflux-local`, `mflux-serve`) now accept an explicit `"WIDTHxHEIGHT"` size override that
  takes priority over the aspect-ratio lookup table.  OpenAI backends ignore the parameter
  (they accept only a fixed set of sizes).

- **`size` parameter on `WorkerClient.imagine()`** — the RunPod `/runsync` payload now includes
  `size` when provided, enabling callers to pass pixel dimensions to mflux workers.

- **`size` handling in `handle_aux_ops`** (`kg_utils.worker.ops`) — `size` is extracted from
  the worker input dict and forwarded to `generate_b64()`; when present it is also included in
  the success response payload.

## [0.4.2] - 2026-06-08

### Added

- **`kg_utils.retrieval`** — new sub-package for shared retrieval helpers:
  - `hit_to_dict(hit, include_diary_timestamp)` — serializes a KGRAG hit object into a
    plain dictionary; optionally includes a `timestamp` field for diary-kind hits.
  - `attach_content_by_sqlite(hits, kg_sqlite_map)` — batched SQLite lookups that hydrate
    `content` on hit dicts in-place; missing or unreadable databases are silently skipped.

- **`kg_utils.worker`** — new sub-package centralizing RunPod `/runsync` protocol helpers:
  - `WorkerClient` — small HTTP client wrapping `list_models`, `rewrite`, `imagine`, and
    `query` operations with per-call `httpx.Timeout` tuning.
  - `WorkerError` — application-level error raised on structured worker failure payloads.
  - `decode_worker_response` / `extract_worker_error` — decode and surface RunPod error
    payloads in both `status: FAILED` and soft `output.error` forms.
  - `handle_aux_ops` — shared handler dispatch for `models`, `rewrite`, and `imagine`
    operations; eliminates duplicated logic across Streamlit worker handlers.

- **`kg_utils.synthesis.factory`** — synthesis backend factory helpers for per-request
  backend overrides, exported via `kg_utils.synthesis`:
  - `normalize_openai_base_url(endpoint)` — normalizes an endpoint string to end with `/v1`.
  - `text_synth_for_backend(backend, fallback)` — constructs a `TextSynthesizer` for the
    requested backend using env vars (`SYNTH_ENDPOINT`, `VLLM_*`, `OLLAMA_ENDPOINT`,
    `OPENAI_API_KEY`); returns `fallback` for unknown or empty values.
  - `image_synth_for_backend(backend, fallback)` — constructs an `ImageSynthesizer` for
    `openai`, `mflux-serve`, or `mflux-local` backends from env vars; returns `fallback`
    for unknown or empty values.

### Changed

- **`.gitignore`** — exclude `.claude/` project memory and settings directories.

## [0.4.1] - 2026-06-08

### Changed

- **`ImageBackend.OPENAI` default model** — upgraded from `dall-e-3` to `gpt-image-1`.
  `gpt-image-1` produces higher-quality images and supports portrait/landscape at
  1024×1536 (vs. DALL-E 3's 1792-wide variants).  Override with `IMAGE_MODEL=dall-e-3`
  to restore the previous behaviour.

### Fixed

- **`_generate_openai` size routing** — added `_GPT_IMAGE_SIZES` table for `gpt-image-1`
  (1024×1536 portrait/landscape); `_generate_openai` now selects the correct size table
  based on the model prefix (`gpt-image` vs. `dall-e`).
- **`response_format` removed from `gpt-image-1` calls** — `gpt-image-1` returns
  `b64_json` by default and does not accept the `response_format` parameter.  The
  `dall-e-3` fallback path now downloads via URL when `b64_json` is absent.
- **`docs/synthesis.md`** — annotated example API-key placeholder as a known
  false positive; regenerated `.secrets.baseline`.

## [0.4.0] - 2026-06-07

### Added

- **`kg_utils.synthesis`** — new sub-package providing unified text and image synthesis
  across six backends with a single, env-var-configurable API.

  **Text backends** (all use the OpenAI wire protocol):
  - `TextBackend.OMLX` — local oMLX / vLLM; MLX chain-of-thought suppressed via
    `extra_body` and `<think>` block stripping.  Default model:
    `Qwen3-4B-Instruct-2507-MLX-8bit`.
  - `TextBackend.OLLAMA` — local Ollama; no API key required.  Default model:
    `hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0`.
  - `TextBackend.OPENAI` — OpenAI cloud.  Default model: `gpt-4o-mini`.

  **Image backends:**
  - `ImageBackend.MFLUX_LOCAL` — in-process Flux2Klein via `mflux` (Apple Silicon);
    per-instance model cache avoids reloading across calls.
  - `ImageBackend.MFLUX_SERVE` — HTTP proxy to a running `mflux-serve` instance.
  - `ImageBackend.OPENAI` — DALL-E 3 with aspect-ratio → size mapping.

  **Public API surface:**
  - `TextConfig` / `ImageConfig` dataclasses with `resolved_endpoint()` and
    `resolved_model()` helpers.
  - `TextSynthesizer.list_models()` — available models at the endpoint.
  - `TextSynthesizer.synthesize_rag(query, snippets)` — grounded RAG answer; skips
    whitespace-only snippets; `max_k` cap; optional system prompt override.
  - `TextSynthesizer.rewrite_for_image(corpus_text)` — rewrites historical prose into
    an image-generation prompt; returns `(prompt, error)` — never raises.
  - `ImageSynthesizer.generate()` → PIL Image; `generate_b64()` → base64 PNG.
  - `text_config_from_env()` / `image_config_from_env()` — build configs from
    `SYNTH_*` / `IMAGE_*` env vars; honour legacy `VLLM_*` and `GUTENKG_IMAGE_MODEL`
    aliases with no migration required.
  - `text_synthesizer_from_env()` / `image_synthesizer_from_env()` — one-call
    convenience factories.

- **`[synthesis]` optional extra** — `httpx>=0.27.0`, `openai>=1.30.0`,
  `pillow>=10.0.0`.
- **`[synthesis-mflux]` optional extra** — all of `[synthesis]` plus `mflux>=0.9.0`.
- **Test suite — three new files** (116 tests total, stdlib + mocks only):
  - `tests/test_synthesis_config.py` (44 tests) — all config defaults and env-var
    priority chains; `clean_synth` / `clean_image` fixtures scrub env state.
  - `tests/test_synthesis_text.py` (38 tests) — `TextSynthesizer` with mocked
    `openai.OpenAI`; `<think>` stripping; empty-content filter; `rewrite_for_image`
    fallback behaviour.
  - `tests/test_synthesis_image.py` (34 tests) — `ImageSynthesizer` with mocked
    `httpx.post` (mflux-serve) and `_load_mflux` (local); DALL-E size mapping;
    base64 round-trip with a real 4×4 PNG.
- **`docs/synthesis.md`** — full reference document: env vars, backend defaults,
  API tables, DALL-E / mflux size maps, usage patterns, and integration notes.

### Changed

- **`mypy` → `ty`** throughout:
  - `pyproject.toml`: removed `[tool.mypy]` and both `[[tool.mypy.overrides]]` sections;
    added `ty = ">=0.0.41"` to dev group; added `[tool.ty.environment]` and
    `[tool.ty.rules]` (`unresolved-import = "ignore"`).
  - `.github/workflows/ci.yml`: `poetry run mypy src/` → `poetry run ty check src/`.
  - `.pre-commit-config.yaml`: `mypy` local hook → `ty` local hook,
    `entry: poetry run ty check src/`.
- **README** — version badge bumped to 0.4.0; synthesis added to Features, Installation,
  API Reference, and project structure tree.

## [0.3.1] - 2026-05-23

### Changed

- **Version bump** `0.3.0` → `0.3.1`.
- **README** — complete rewrite to reflect the 0.3.x expanded scope: updated
  version badge, description, feature list, Quick Start examples (now using
  correct `kg_utils.specs` / `kg_utils.extractor` / `kg_utils.pipeline`
  import paths), API reference tables for all new modules, revised project
  structure tree, and split test-suite instructions into fast vs. integration
  runs.
- **`tests/test_types.py`** — updated module docstring and import to use
  `kg_utils.specs` and `kg_utils.extractor`; removed stale KGModule tests
  (superseded by `test_pipeline_module.py`) and KGExtractor "raises
  NotImplementedError" tests (now enforced by ABC); rewrote config tests to
  use `DummyExtractor` instead of `KGExtractor.__new__`.
- **`tests/test_integration.py`** — replaced `from kg_utils.types import …`
  with imports from `kg_utils.extractor`, `kg_utils.pipeline`, and
  `kg_utils.specs`; added `analyze()` implementation and `_default_dir` to
  `_FileTreeModule` (required by the ABC and concrete `KGModule` base).

### Removed

- **`kg_utils.types` subpackage** (`types/__init__.py`, `types/specs.py`,
  `types/extractor.py`, `types/module.py`) — the parallel thin/abstract
  hierarchy has been eliminated.  All types now live at the canonical
  top-level locations: `kg_utils.specs`, `kg_utils.extractor`,
  `kg_utils.pipeline`.  This removes the dual-class friction where
  `isinstance` checks and type annotations could silently diverge depending
  on which import path was used.

## [0.3.0] - 2026-05-23

### Added

- **`kg_utils.specs`** — rich `NodeSpec` and `EdgeSpec` dataclasses with
  `lineno`, `end_lineno`, `metadata` fields; `BuildStats`, `QueryResult`, and
  `SnippetPack` return types for the full pipeline surface.
- **`kg_utils.extractor`** — `KGExtractor` abstract base class (yields
  `NodeSpec` / `EdgeSpec` iterators); domain authors subclass this to feed any
  source into the pipeline.
- **`kg_utils.store`** — `GraphStore`: SQLite-backed authoritative node/edge
  store with upsert, BFS expand, symbol resolution (`resolve_symbols`),
  caller lookup (`callers_of`), provenance recording, and a `ProvMeta`
  typed-dict.  SQLite is the single source of truth; the vector index is
  always derived from it.
- **`kg_utils.semantic`** — `SemanticIndex`: LanceDB vector index built from
  `GraphStore` nodes.  Includes `Embedder` abstract base,
  `SentenceTransformerEmbedder` (with ST ≥ 5.4 / ≤ 5.3 API fallback),
  `SeedHit` result dataclass, model registry (`_KNOWN_MODELS`), and
  `resolve_model_path` / `suppress_ingestion_logging` utilities.
- **`kg_utils.pipeline`** — `KGModule`: concrete abstract base class with the
  complete build → query → pack pipeline.  Domain authors implement only
  `make_extractor()`, `kind()`, and `analyze()`.  Provides hybrid
  semantic + lexical reranking, BFS graph expansion, configurable hop depth,
  `min_score` filtering, `max_nodes` capping, and snippet extraction with
  context lines.
- **`kg_utils.module`** — thin re-export shim providing `KGModule` and
  `KGExtractor` from a single import path for downstream compatibility.
- **`[semantic]` optional extra** in `pyproject.toml`: `lancedb>=0.19.0`,
  `numpy>=1.24.0`, `sentence-transformers>=5.4.1`, `torch>=2.5.1`,
  `transformers>=4.40.0,<4.57`.  Install with
  `pip install 'kgmodule-utils[semantic]'`.
- **`[kgdeps]` Poetry group** (optional): `pycode-kg>=0.18.1`,
  `doc-kg>=0.15.2` for integration testing against real KG modules.
- **`poetry.toml`** — local venv configuration (`in-project = true`).
- **`.pycodekg/snapshots/`** — initial CodeKG snapshot and manifest tracked
  for reproducible metrics across releases.
- **Test suite — three new files:**
  - `tests/test_store.py` (343 lines) — unit tests for `GraphStore`: write/read,
    edges, wipe, upsert, `query_nodes`, BFS expand, provenance, `resolve_symbols`,
    `callers_of`, `edges_from`, stats, and context-manager lifecycle.
  - `tests/test_pipeline_utils.py` (295 lines) — pure-function unit tests for
    all pipeline utilities: `semantic_score_from_distance`, `query_tokens`,
    `normalize_query_text`, `docstring_signal`, `lexical_overlap_score`,
    `safe_join`, `read_lines`, `compute_span`, `make_snippet`,
    `make_module_summary`, `spans_overlap`.
  - `tests/test_pipeline_module.py` (318 lines, `@pytest.mark.integration`) —
    end-to-end integration tests for a concrete `KGModule` implementation
    (`_TextKG` / `_TextExtractor`): `build_graph`, `build_index`, `stats`,
    `query` (semantic match, hop=0, hybrid rerank, `min_score`, `max_nodes`),
    `pack` (markdown, JSON, snippet text, key stripping), and lazy property
    initialisation.

### Changed

- **Version bump** `0.2.4` → `0.3.0` (significant new surface area).
- **Development status** classifier `3 - Alpha` → `4 - Beta`.
- **Package description** updated to reflect the expanded scope: "Shared
  types, graph store, semantic index, and pipeline base for the KGModule SDK".
- **`src/kg_utils/__init__.py`** — updated module docstring to document all
  new sub-modules and the `[semantic]` extra install path.
- **`lancedb` mypy override** added to `[[tool.mypy.overrides]]`
  `ignore_missing_imports` list so mypy strict mode passes without stubs.
- **`.gitignore`** — added exclusion rules for transient `.pycodekg/` and
  `.dockg/` artifacts (SQLite databases, LanceDB dirs, model caches) while
  keeping `snapshots/` tracked.
- **`.secrets.baseline`** regenerated to whitelist SHA git-tree hashes in
  `.pycodekg/snapshots/` that `detect-secrets` flags as `HexHighEntropyString`
  false positives.

### Fixed

- **`kg_utils.embedder`** — `load_sentence_transformer` and
  `SentenceTransformerEmbedder.__init__` now catch `(ImportError, ValueError)`
  instead of `ImportError` alone when suppressing HF logging, preventing an
  unhandled `ValueError` raised by some `transformers` versions when the
  logging backend is already initialised.

## [0.2.4] - 2026-04-29

### Fixed

- `load_sentence_transformer` / `SentenceTransformerEmbedder`: call
  `hf_logging.disable_progress_bar()` in addition to `set_verbosity_error()`
  and `TQDM_DISABLE=1`. `TQDM_DISABLE` alone misses the `_tqdm_active` gate
  inside `transformers`, leaving progress bars visible in worker processes.

## [0.2.3] - 2026-04-29

### Fixed

- `load_sentence_transformer`: removed save/restore logic around HF logging
  and TQDM state; now simply sets `TQDM_DISABLE=1` and `set_verbosity_error()`
  once and leaves them set, eliminating the mypy `[assignment]` error caused
  by the `Module | None` type mismatch on `_hf_logging`.
- **CI: mypy** — added `[[tool.mypy.overrides]]` for `sentence_transformers`,
  `transformers`, and `numpy` with `ignore_missing_imports = true`; added a
  separate override for `kg_utils.embedder` disabling `disallow_untyped_calls`
  so the `hf_logging.set_verbosity_error()` call requires no `type: ignore`
  regardless of whether `transformers` is installed.
- **CI: test** — marked all `sentence_transformers`-dependent tests in
  `tests/test_embedder.py` with `@pytest.mark.integration`; added
  `pytestmark = pytest.mark.integration` to `tests/test_integration.py`;
  updated CI test step to `pytest -m "not integration"` so these are skipped
  when the optional heavy deps are absent.
- **pytest.ini** — corrected stale `testpaths` (`./src/tests` → `./tests`);
  registered the `integration` marker to suppress unknown-mark warnings.
- **pylint** — added `[tool.pylint.main]` with `source-roots` and `init-hook`
  so pylint resolves the `src/` layout; added file-level disable in
  `tests/test_embedder.py` for pytest-pattern false positives
  (`redefined-outer-name`, `missing-function-docstring`, `too-few-public-methods`,
  `import-outside-toplevel`); fixed redundant `kg_utils` reimport in
  `test_doc_kg_re_exports_embedder_classes`.

## [0.2.2] - 2026-04-28

### Added

- `kg_utils.embedder` sub-package: concrete `SentenceTransformer` embedding
  implementation shared across all KGModule packages.
  - `Embedder` — abstract base with `embed_texts` + `embed_query` + `dim`.
  - `SentenceTransformerEmbedder` — concrete implementation with
    `local_files_only=True` guard on MPS to prevent SIGBUS on first `encode()`.
  - `load_sentence_transformer(model_name)` — canonical safe-load factory with
    four-step resolution: local path → HF cache → live network fetch.
  - `get_embedder(model_name)` — high-level factory returning a ready-to-use
    `SentenceTransformerEmbedder`.
  - `wrap_embedder(st_model, model_name)` — wraps a live `SentenceTransformer`
    as an `Embedder` to share a model across pipeline stages without reloading.
- Comprehensive test suite: `tests/test_embed.py`, `tests/test_embedder.py`,
  `tests/test_snapshots.py` (extended), `tests/test_types.py` (extended), and
  `tests/test_integration.py` covering cross-module protocol compliance, full
  snapshot lifecycle, subclass delta extensibility, and git subprocess
  integration.

### Fixed

- `SentenceTransformerEmbedder` and `wrap_embedder`: replaced direct
  `get_sentence_embedding_dimension()` call with a `getattr` fallback that
  tries `get_embedding_dimension` first (canonical in ST ≥ 5.4) then
  `get_sentence_embedding_dimension` (ST ≤ 5.3), eliminating the
  `FutureWarning` emitted by sentence-transformers ≥ 5.4.
- Aligned `sentence-transformers` minimum version to `>=5.4.1` in `code_kg`,
  `doc_kg`, and `diary_kg` so all KGModule packages resolve the same ST
  release and the `FutureWarning` cannot occur in any module.

## [0.2.1] - 2026-04-27

### Fixed

- `SnapshotManager._load_manifest`: extended legacy key normalization to handle
  manifest entries that carry a `commit` field instead of `tree_hash` or `key`,
  preventing `KeyError` when loading manifests written by older versions.
- `SnapshotManager.get_previous_snapshot` / `get_baseline`: replaced hard
  dict-key access (`entry["timestamp"]`, `entry["key"]`) with `.get()` calls
  and added explicit empty-key guards so malformed or legacy manifest entries
  no longer raise `KeyError` at runtime.
- Bumped `__version__` in `src/kg_utils/__init__.py` to track the package
  version (was stuck at `0.1.0`).

## [0.2.0] - 2026-04-26

### Added

- `kg_utils.embed` sub-package: shared embedding protocol and model-cache
  convention for the KGModule stack (stdlib-only, no external dependencies).
  - `Embedder` — `runtime_checkable` Protocol with `embed_query(text) -> list[float]`;
    KG modules and kgrag adapters type-hint against this without coupling to
    any concrete implementation.
  - `DEFAULT_MODEL` / `KNOWN_MODELS` — canonical default (`BAAI/bge-small-en-v1.5`)
    and short-alias mapping shared by all modules for consistent alias resolution.
  - `kg_model_cache_dir()` — returns `~/.kgrag/models/` by default; a single
    `KGRAG_MODEL_DIR` env-var redirects every module's cache simultaneously.
  - `resolve_model_path()` — converts a model name or alias to an absolute local
    cache path, with an optional per-module `local_fallback` for standalone use.
