# Release Notes — v0.18.1

> Released: 2026-08-25

A patch fixing a real correctness bug in `resolve_symbols()`, the pass that
links a `sym:` call stub to the first-party definition it actually calls.

## What changed

**Symbol resolution stopped guessing across classes.** The resolver has
always matched a dotted call stub like `plotter.render()` by its trailing
method name alone — `render` — against every first-party definition in the
graph sharing that name, with no idea what `plotter` actually was. Two real
collisions turned up while verifying this against the `gutenberg_kg` corpus:
a stdlib `re.Match.start()` call fabricated a `RESOLVES_TO` edge into an
unrelated `_LogCapture.start` method, and a PyVista `Plotter.reset_camera()`
call fabricated three phantom callers into a Qt window's own
`reset_camera()`, inflating its fan-in ranking in downstream analysis.

The fix is additive: when a stub's `metadata` carries a `receiver_class` key
— written by a caller that traced the call's receiver back to a parameter or
local variable's type annotation — matching now scopes to definitions on
that class instead of the whole graph. A typed stub that finds no match
stays unresolved rather than falling back to the old untyped guess, so a
wrong match never survives just because a right one wasn't found. Stubs with
no `receiver_class` in their metadata are unaffected.

## Upgrading

No action needed to pick up the fix — existing callers that already write
`receiver_class` metadata (the first is `pycode_kg`'s own visitor, landing
alongside this release) start resolving more precisely on their next graph
build. Nothing else changes: the schema, the `RESOLVES_TO` edge shape, and
every other resolution mode are untouched.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
