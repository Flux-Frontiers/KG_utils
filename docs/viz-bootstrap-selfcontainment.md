# `kg_utils.viz` output is not self-contained — it fetches Bootstrap from a CDN

**Status:** **fixed in 0.10.0** — `_inline_bootstrap()` in `kg_utils/viz/graph_html.py`,
covered by `tests/test_viz.py::test_output_is_self_contained` and
`::test_bootstrap_shim_survives`. Retained as the rationale for those rules.
**Affects:** `kg_utils.viz.build_graph_html` as shipped in 0.7.0 through 0.9.0, and
therefore every consumer of it (pycode_kg today; doc_kg and gutenberg_kg once they
adopt it). Anyone who generated and distributed a graph on those versions shipped a
file that phones home and breaks offline, and will want to regenerate it.

## Symptom

A page produced by `build_graph_html` makes two outbound requests when opened:

```
https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css
https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/js/bootstrap.bundle.min.js
```

Offline — an air-gapped machine, a plane, a reviewer who was sent the file — both fail
with `ERR_CONNECTION_RESET` and the page renders with broken layout: the network canvas
no longer fills its container and an unstyled white band appears below it.

This contradicts what we tell people. `build_graph_html`'s docstring says "a
self-contained HTML document", and `pycodekg viz-export --help` says the file "opens
straight from the filesystem and can be handed to someone who has neither the repo nor
Python". Both are currently false: they also need internet, and jsdelivr reachable.

There is a privacy dimension too. Opening a locally-generated graph silently contacts a
third-party CDN, which is a poor default for a file that may be generated from a private
codebase and mailed around.

## Root cause

Not our code. pyvis 0.3.2 hardcodes the two tags in its own Jinja template:

```
site-packages/pyvis/templates/template.html:41-45   <link ... bootstrap.min.css>
site-packages/pyvis/templates/template.html:47-50   <script ... bootstrap.bundle.min.js>
```

`cdn_resources="in_line"` — which we already pass, deliberately, and which is why
vis-network *is* inlined — governs only the vis-network assets. It does not touch these
two tags. There is no pyvis option that does.

Worth being explicit: **this is not a regression from the 0.7.0 promotion.** The
pre-promotion renderer in `pycode_kg/app.py` used the identical `cdn_resources="in_line"`
call and emitted the identical tags. The bug is as old as the renderer; consolidating it
into `kg_utils.viz` is what makes it fixable in one place.

## Blast radius is much smaller than it looks

Bootstrap is 155 KB of CSS plus a JS bundle. Our generated pages use **two classes of
it**. Measured on a real 150-node export:

```
$ grep -oE 'class="[^"]*"' graph.html | sort | uniq -c
      1 class="outerBorder"
      1 class="card-body"
      1 class="card"
```

- `.card` (template line 172) wraps the whole graph.
- `.card-body` (template line 239) is `#mynetwork`, the canvas container.
- `.outerBorder` is **not** Bootstrap — pyvis defines it in its own inline `<style>` at
  template line 119, so it already works offline.

The other Bootstrap classes in the template (`form-select`, `row no-gutters`, `col-*`,
`btn`) belong to pyvis's filter/select menu, which we do not enable. They never appear in
our output.

`bootstrap.bundle.min.js` is **entirely unused** — nothing in our output has a dropdown,
modal, tooltip, or collapse. It is pure dead weight and can simply be dropped.

So the whole dependency reduces to the handful of declarations behind `.card` and
`.card-body`.

## The fix

There is already a post-processing seam. `build_graph_html` ends with:

```python
# graph_html.py:285
return document.replace("</body>", _panel_markup(panel_data) + "</body>")
```

Add a second pass that strips the two tags and injects replacement CSS. Suggested shape —
a module-level constant plus a small helper, so the *reason* stays next to the code:

```python
#: pyvis 0.3.2 hardcodes Bootstrap CDN tags in its template, and
#: ``cdn_resources="in_line"`` covers only the vis-network assets.  Our output uses
#: exactly two Bootstrap classes, so we strip the CDN tags and supply their rules
#: directly rather than making an offline page depend on a third-party host.
_BOOTSTRAP_CDN = re.compile(
    r'<link\b[^>]*cdn\.jsdelivr\.net[^>]*>'
    r'|<script\b[^>]*cdn\.jsdelivr\.net[^>]*>\s*</script>',
    re.S,
)

_BOOTSTRAP_SHIM = """<style type="text/css">
body { margin: 0; }
.card, .card *, .card *::before, .card *::after { box-sizing: border-box; }
.card { position: relative; display: flex; flex-direction: column;
        min-width: 0; word-wrap: break-word; background-color: #fff;
        background-clip: border-box; border: 1px solid rgba(0,0,0,.125);
        border-radius: .25rem; }
.card-body { flex: 1 1 auto; padding: 1rem 1rem; }
</style>"""


def _inline_bootstrap(document: str) -> str:
    """Replace pyvis's Bootstrap CDN tags with the rules the page actually uses.

    :param document: HTML as pyvis wrote it.
    :return: The same page with no external references.
    """
    return _BOOTSTRAP_CDN.sub("", document).replace(
        "</head>", _BOOTSTRAP_SHIM + "</head>", 1
    )
```

The `.card` / `.card-body` declarations are copied verbatim from
`bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css`, so they are the genuine rules, not an
approximation.

### Two rules that are not in `.card`, and are both load-bearing

Both were found by measuring the rendered box model against real Bootstrap, not by
reading the CSS. A first draft of this shim carried only the two `.card` rules and looked
correct in a screenshot while being wrong:

1. **`box-sizing: border-box`.** Bootstrap's Reboot applies this globally. Without it,
   `.card-body` measured **1418px inside a 1386px parent** — a 32px horizontal overflow
   from padding and borders being added outside the declared width.
2. **`body { margin: 0 }`.** Also from Reboot. Without it the browser default 8px margin
   leaves the card 16px narrower than under real Bootstrap.

Scoping the `box-sizing` reset to `.card` and its descendants rather than `*` keeps it
from leaking into a host page if a consumer ever embeds this markup in a larger document.

## Verification

Compare the shim against genuine Bootstrap rather than eyeballing it. Build a control
page with the real stylesheet inlined, then measure both under Playwright — physics
placement is non-deterministic so screenshots differ run to run, but the box model does
not:

```python
g = await pg.evaluate("""() => {
  const m = s => { const r = document.querySelector(s).getBoundingClientRect();
                   return [Math.round(r.width), Math.round(r.height)]; };
  return {card: m('.card'), body: m('.card-body')};
}""")
```

Result at a 1400x900 viewport, on a 150-node export from this repo's own graph:

| Page | `.card` | `.card-body` | external requests | page errors |
|---|---|---|---|---|
| 0.7.0 as shipped | 1400x802 | 1398x800 | **2 (both fail offline)** | 2 |
| shim, first draft | 1384x802 | 1382x800 ⚠ overflow | 0 | 0 |
| **shim, final** | **1400x802** | **1398x800** | **0** | **0** |
| control (real Bootstrap inlined) | 1400x802 | 1398x800 | 0 | 0 |

The final shim is pixel-identical to real Bootstrap in both dimensions, with no network
access. Graph still draws: 341 vis nodes in the DOM (150 graph nodes plus vis-network's
smooth-edge support nodes).

Page size is essentially unchanged — 1,037,631 → 1,037,437 bytes — since we are removing
two tags and adding ~400 bytes of CSS, having never inlined Bootstrap itself.

## The test that should have caught this

`tests/test_viz.py::test_output_is_self_contained` asserts:

```python
assert "cdnjs.cloudflare.com/ajax/libs/vis-network" not in html
assert 'src="lib/bindings/utils.js"' not in html
```

Both are true while the page still reaches out — to a *different* CDN. The assertion
names one host it happens to know about instead of the property it claims to test.
Replace it with something host-agnostic:

```python
def test_output_makes_no_external_requests() -> None:
    """Self-contained means no host at all, not merely no cdnjs."""
    html = build_graph_html(CODE_NODES, CODE_EDGES, theme=CODE_THEME)
    for pattern in ('src="http', "src='http", 'href="http', "href='http", "@import"):
        assert pattern not in html
    assert 'src="lib/' not in html
```

That form fails on any new external reference, including one a future pyvis release
introduces from a host nobody has heard of yet. Worth adding a companion assertion that
the two Bootstrap classes still resolve, so the shim cannot silently drop out:

```python
    assert ".card-body" in html and "box-sizing" in html
```

## Downstream

The fix lands once and every consumer inherits it on upgrade — pycode_kg's `viz-export`
and Streamlit Graph tab, and doc_kg / gutenberg_kg when they adopt `kg_utils.viz`. No
consumer code changes.

Worth calling out in the 0.7.1 release notes under a "fixed" heading, since anyone who
generated and distributed a graph on 0.7.0 shipped a file that phones home and breaks
offline; they will want to regenerate.

## Residual, not fixed here

`.card` carries `background-color: #fff` from Bootstrap, so a sliver of white shows below
the dark canvas when the card is taller than the network. This is genuine Bootstrap
behaviour and identical in the control, so it is not caused by the shim — but since we
now own these rules, overriding the card background to match the graph background is a
one-line cosmetic follow-up if we want it.
