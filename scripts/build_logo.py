#!/usr/bin/env python3
"""Draw the kgmodule-utils logo in Art Techno and write every size the repo uses.

The mark is the backplane: one board carries a bus, and five KG modules stand
in their own slots on it, the middle one holding the die. Every fleet KG plugs
into kgmodule-utils. The palette, drawing helpers and outlined lettering come
from the Art Techno brand build, so the SVGs render the same everywhere.

Usage::

    python scripts/build_logo.py

Needs ``fonttools``, Google Chrome for the PNG renders, and a clone of
https://github.com/Flux-Frontiers/art_techno beside this repo. Set ``CHROME``
to the browser binary if it is not at the default macOS path.

Writes:

- ``assets/kgutils_logo.svg`` and ``assets/kgutils_logo.png``: the square mark
  (1024 px PNG)
- ``assets/kgutils_icon.svg``: the mark reduced for 64 px and below
- ``assets/kgutils_banner.svg`` and ``assets/kgutils_banner.png``
"""

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AT_BUILD = ROOT.parent / "art_techno" / "docs" / "brand" / "build.py"

_spec = importlib.util.spec_from_file_location("at_build", AT_BUILD)
at = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at)

f, line, pad, via, diamond, clipped_rect = (
    at.f,
    at.line,
    at.pad,
    at.via,
    at.diamond,
    at.clipped_rect,
)
OBSIDIAN, PANEL, COPPER, GOLD, GREEN, MINT, SILK = (
    at.OBSIDIAN,
    at.PANEL,
    at.COPPER,
    at.GOLD,
    at.GREEN,
    at.MINT,
    at.SILK,
)

NAME = "KG UTILS"
PART = "THE SDK UNDER EVERY GRAPH"


def rect(
    x: float, y: float, w: float, h: float, fill: str, stroke: str = "", width: float = 0
) -> str:
    s = f' stroke="{stroke}" stroke-width="{f(width)}"' if stroke else ""
    return f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" fill="{fill}"{s}/>'


def die(x: float, y: float, s: float, core: str) -> str:
    """A square green die in a gold line, centered on ``x, y``.

    :param x: Center x.
    :param y: Center y.
    :param s: Side length.
    :param core: Color of the square at its center.
    """
    return rect(x - s / 2, y - s / 2, s, s, GREEN, GOLD, s / 9) + rect(
        x - s / 6, y - s / 6, s / 3, s / 3, core
    )


def card(x: float, y: float, w: float, h: float, c: float, width: float) -> str:
    """A dark board or card with 45-degree corners in a gold line."""
    return (
        f'<path d="{clipped_rect(x, y, w, h, c)}" fill="{PANEL}" stroke="{GOLD}" '
        f'stroke-width="{f(width)}"/>'
    )


def backplane() -> str:
    """One board with a bus; five module cards stand in its slots, on a 512 field."""
    out = [card(92, 262, 328, 56, 12, 6)]
    out.append(line([(118, 290), (394, 290)], COPPER, 4))
    out.append(via(112, 290, 7, COPPER) + via(400, 290, 7, COPPER))
    for i, x in enumerate((134, 196, 258, 320, 382)):
        tall = 140 if i == 2 else 104 + 12 * (i % 2)
        y0 = 254 - tall
        out.append(rect(x - 6, 256, 12, 34, GOLD))
        out.append(card(x - 22, y0, 44, tall, 8, 4))
        for k in range(3):
            out.append(rect(x - 12, y0 + tall - 20 - 14 * k, 24, 6, COPPER))
        if i == 2:
            out.append(die(x, y0 + 22, 24, SILK))
        else:
            out.append(rect(x - 9, y0 + 13, 18, 18, GREEN, GOLD, 2))
    return "".join(out)


def backplane_small() -> str:
    """The backplane reduced for 64 px and below: the board, the bus, three cards."""
    out = [card(8, 40, 48, 14, 3, 2.6), line([(13, 47), (51, 47)], COPPER, 2)]
    for x in (18, 32, 46):
        out.append(rect(x - 2, 36, 4, 6, GOLD) + card(x - 5, 12, 10, 24, 2, 2))
        out.append(rect(x - 3, 15, 6, 6, GREEN))
    return "".join(out)


def frame(w: float, h: float, inset: float, c: float, heavy: float) -> str:
    """Double frame with 45-degree corners and a green diamond at each corner.

    :param w: Canvas width.
    :param h: Canvas height.
    :param inset: Distance of the heavy line from the canvas edge.
    :param c: Corner cut of the heavy line.
    :param heavy: Stroke width of the heavy line.
    """
    out = [
        f'<path d="{clipped_rect(inset, inset, w - 2 * inset, h - 2 * inset, c)}" fill="none" stroke="{GOLD}" stroke-width="{f(heavy)}"/>',
        f'<path d="{clipped_rect(inset * 1.9, inset * 1.9, w - 3.8 * inset, h - 3.8 * inset, c * 0.75)}" fill="none" stroke="{GOLD}" stroke-width="{f(heavy * 0.4)}" stroke-opacity=".6"/>',
    ]
    d = inset * 1.45 + c * 0.4375
    for cx, cy in ((d, d), (w - d, d), (d, h - d), (w - d, h - d)):
        out.append(diamond(cx, cy, inset * 0.5, GREEN))
    return "".join(out)


def svg(w: int, h: int, title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f"<title>{title}</title>"
        f'<rect width="{w}" height="{h}" fill="{OBSIDIAN}"/>{body}</svg>\n'
    )


def mark(display, mono) -> str:
    """512 square mark: the backplane, the name and the part line beneath."""
    body = frame(512, 512, 10, 26, 5) + backplane()
    body += display.path(NAME, 256, 430, 40, 0.2, "middle", GOLD)
    body += mono.path(PART, 256, 462, 13.5, 0.3, "middle", MINT)
    return svg(512, 512, "kgmodule-utils mark, Art Techno", body)


def icon() -> str:
    """64 icon: the reduced backplane in one heavy frame, no lettering."""
    body = f'<path d="{clipped_rect(2.5, 2.5, 59, 59, 7.5)}" fill="none" stroke="{GOLD}" stroke-width="3"/>'
    return svg(64, 64, "kgmodule-utils icon, Art Techno", body + backplane_small())


def banner(display, mono) -> str:
    """1280 x 320 banner: the backplane left, the name and part line right.

    The name is set as large as the room right of the backplane allows, and the
    pair is centered on the banner's height.
    """
    s = 0.78
    body = frame(1280, 320, 10, 18, 3)
    body += f'<g transform="translate({f(230 - 256 * s)} {f(160 - 212 * s)}) scale({s})">{backplane()}</g>'
    cx, room, gap = 820, 760, 34
    size = min(120, 120 * room / display.width(NAME, 120, 0.2))
    part = min(26, 26 * room / mono.width(PART, 26, 0.3))
    cap = display.cap * size / display.upm
    pcap = mono.cap * part / mono.upm
    base = (320 - (cap + gap + pcap)) / 2 + cap
    body += display.path(NAME, cx, base, size, 0.2, "middle", GOLD)
    body += mono.path(PART, cx, base + gap + pcap, part, 0.3, "middle", MINT)
    return svg(1280, 320, "kgmodule-utils banner, Art Techno", body)


def render(src: Path, out: Path, w: int, h: int) -> None:
    """Render the SVG ``src`` to ``out`` at ``w`` x ``h`` px with headless Chrome.

    :param src: SVG file; its ``width`` attribute sets the CSS width.
    :param out: PNG to write.
    :param w: Output width in px.
    :param h: Output height in px.
    """
    svg_w = int(src.read_text().split('width="', 1)[1].split('"', 1)[0])
    subprocess.run(
        [
            at.CHROME,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={svg_w},{svg_w * h // w}",
            f"--force-device-scale-factor={w / svg_w}",
            f"--screenshot={out}",
            src.as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    at.fetch_fonts()
    display = at.Face(at.CACHE / "JosefinSans.ttf", weight=600)
    mono = at.Face(at.CACHE / "IBMPlexMono-Medium.ttf")

    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    mark_svg = assets / "kgutils_logo.svg"
    icon_svg = assets / "kgutils_icon.svg"
    banner_svg = assets / "kgutils_banner.svg"
    mark_svg.write_text(mark(display, mono))
    icon_svg.write_text(icon())
    banner_svg.write_text(banner(display, mono))

    render(mark_svg, mark_svg.with_suffix(".png"), 1024, 1024)
    render(banner_svg, banner_svg.with_suffix(".png"), 2560, 640)


if __name__ == "__main__":
    main()
