---
name: release
description: Cut a kgmodule-utils release — verify every doc and version surface is in sync, promote the changelog, write release notes, tag, and push. Use when releasing kg_utils / kgmodule-utils, publishing to PyPI, or cutting a version.
---

# Release Workflow (kg_utils / kgmodule-utils)

Releases here are **tag-triggered**. Pushing a `v*` tag runs
`.github/workflows/release.yml`, which builds, creates the GitHub Release from
`release-notes.md`, and publishes to PyPI via OIDC trusted publishing. There
are no credentials to supply — **and no undo.** PyPI accepts a version number
once; a mistake can only be yanked, not replaced.

So the work is almost entirely *verification before the tag*. Do not treat any
step below as a formality.

---

## Step 0 — Preconditions

```bash
git checkout main && git pull origin main
git status --porcelain          # must be empty
```

Confirm `## [Unreleased]` in `CHANGELOG.md` has content. If it is empty there
is nothing to release — stop.

## Step 1 — Decide the version

Read `version` in `pyproject.toml`. If a bump already landed (a
`chore(release): X.Y.Z` commit), adopt it and skip Step 3. Otherwise ask the
user for patch / minor / major unless they said. The tag is `v<version>`.

## Step 2 — Promote the changelog

Replace `## [Unreleased]` with `## [<version>] - <YYYY-MM-DD>` and insert a
fresh empty `## [Unreleased]` above it.

**One ASCII hyphen** between version and date — `fleet_audit.py` parses this
heading strictly, and the strictness is the point. See `FLEET_STANDARDS.md`.

## Step 3 — Bump the version (skip if already done)

Both of:

- `pyproject.toml` → `version = "..."`
- `src/kg_utils/__init__.py` → `__version__ = "..."`

Then `poetry lock`.

## Step 4 — THE SYNC AUDIT (this is the step that catches things)

Every release so far has found at least one of these stale. Check all of them
explicitly; do not assume a previous release left them correct.

**Four version surfaces must agree.** `fleet_audit.py` cross-checks three of
them, so a mismatch is caught eventually — but after publication, which is too
late.

```bash
grep '^version' pyproject.toml
grep '__version__' src/kg_utils/__init__.py
grep -o 'version-[0-9.]*' README.md | head -1
grep '^version:' CITATION.cff
```

- [ ] `pyproject.toml`
- [ ] `src/kg_utils/__init__.py`
- [ ] `README.md` badge
- [ ] `CITATION.cff` — **also its `date-released:`**, which is a second stale
      field hiding behind the first

**Prose must describe what actually shipped.** A new public module is not
released until it is documented:

- [ ] `README.md` feature paragraph names any new module
- [ ] `docs/features.md` has a module bullet for it

Grep for the module name before believing it is documented — `grep -n temporal
README.md` returned two hits for the 0.18.0 release and **both were false
positives**, matching "temporal snapshots" and "temporal metric tracking" on the
unrelated `snapshots` module. Read every hit; do not count them.

## Step 5 — Write `release-notes.md`

The workflow feeds this file to `gh release create --notes-file`. If it is
stale, the GitHub Release describes **the previous release** — which is what
would have happened for 0.18.0, publishing the temporal contract under 0.17.0's
ingestion notes.

Generate it from the changelog rather than by hand, so the two cannot drift:

```python
import pathlib, re
VERSION, DATE = "0.18.0", "2026-08-22"
cl = pathlib.Path("CHANGELOG.md").read_text()
body = re.search(rf"^## \[{re.escape(VERSION)}\] - {DATE}\n(.*?)(?=^## \[)",
                 cl, re.S | re.M).group(1).strip("\n")
pathlib.Path("release-notes.md").write_text(
    f"# Release Notes — v{VERSION}\n\n> Released: {DATE}\n\n{body}\n\n"
    "---\n\n_Full changelog: [CHANGELOG.md](CHANGELOG.md)_\n"
)
```

Then confirm the first line says the version you are releasing.

## Step 6 — Verify the build is green

```bash
env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry run pytest -q
poetry run ruff format --check . && poetry run ruff check . && poetry run ty check src/
poetry check --lock
```

The `env -u` prefix is not optional: an inherited `VIRTUAL_ENV` silently
retargets `poetry run` at another repo's interpreter, and the tests then pass
against the wrong dependency set.

## Step 7 — Commit and tag

```bash
git add CHANGELOG.md release-notes.md README.md CITATION.cff docs/features.md \
        pyproject.toml poetry.lock src/kg_utils/__init__.py
git commit -m "chore(release): v<version> release notes"
git tag -a v<version> -m "v<version>"
```

## Step 8 — Push (ASK FIRST — always)

Never push autonomously. Show the tag and the sync-audit result, and get
explicit approval. The tag push is the irreversible step.

```bash
git push origin main
git push origin v<version>
```

Then watch the run: build → GitHub Release → PyPI publish.

---

## Downstream

Fleet repos floor `kgmodule-utils>=<version>`; their `poetry lock` fails until
this is on PyPI. After publishing, tell the user which repos are unblocked.

## Known environment quirk

`git push --delete` of a remote branch fails in the Claude Code remote
container (`send-pack: unexpected disconnect`, then a misleading "Everything
up-to-date"), and the GitHub MCP server has no delete-branch tool. Branch
deletion is a manual step for the user.
