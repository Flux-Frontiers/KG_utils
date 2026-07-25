"""
Declarative hover-tooltip specification for graph renderers.

Node schemas diverge sharply between KG domains — code nodes carry
``qualname``/``module_path``/``lineno``, document nodes carry
``title``/``file_path``/``char_start``, metabolic nodes carry
``formula``/``ec_number``.  Only ``id``, ``kind`` and ``name`` are universal.

A :class:`TooltipSpec` lets each domain name its own fields while the renderer
keeps one implementation of the markup, so the viewers stay visually consistent
without every module writing its own HTML.  When a domain needs markup the spec
cannot express, the renderer also accepts a plain callable instead — see
``build_graph_html``.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TooltipRow", "TooltipSpec"]


@dataclass(frozen=True)
class TooltipRow:
    """One metadata line beneath the tooltip's title.

    :param value: Either a node-dict key, or a callable taking the whole node
        and returning display text.  The callable form exists for values that
        span several fields, such as a line range built from ``lineno`` and
        ``end_lineno``.
    :param prefix: Short text or emoji placed before the value.
    :param skip_if_empty: Drop the row when the value is empty or ``None``.
        Almost always what you want — a tooltip full of blank labels is worse
        than a short one.
    """

    value: str | Callable[[Mapping[str, Any]], str]
    prefix: str = ""
    skip_if_empty: bool = True

    def render(self, node: Mapping[str, Any]) -> str:
        """Resolve this row against *node*.

        :param node: Node attribute mapping.
        :return: Display text, empty when the row should be dropped.
        """
        if isinstance(self.value, str):
            raw = node.get(self.value)
            text = "" if raw is None else str(raw)
        else:
            text = self.value(node) or ""
        text = text.strip()
        if not text and self.skip_if_empty:
            return ""
        return f"{self.prefix}{text}" if self.prefix else text


@dataclass(frozen=True)
class TooltipSpec:
    """How to build a node's hover tooltip from its fields.

    :param title: Node key holding the bold heading, or a callable.
    :param rows: Metadata lines shown under the heading, joined with separators.
    :param body: Node key holding free text — a docstring, chunk text or
        description — rendered in a monospaced block below a rule.
    :param body_lines: Maximum body lines before truncation.
    :param max_width: Tooltip width in pixels.
    """

    title: str | Callable[[Mapping[str, Any]], str] = "name"
    rows: Sequence[TooltipRow] = field(default_factory=tuple)
    body: str | None = None
    body_lines: int = 8
    max_width: int = 400

    def render(self, node: Mapping[str, Any], color: str) -> str:
        """Build the tooltip HTML for *node*.

        :param node: Node attribute mapping.
        :param color: Accent colour for the kind badge and left border.
        :return: An HTML string suitable for a pyvis node ``title``.
        """
        kind = html.escape(str(node.get("kind", "")))
        if isinstance(self.title, str):
            title_text = str(node.get(self.title) or node.get("name") or node.get("id", ""))
        else:
            title_text = self.title(node)
        title = html.escape(title_text)

        rendered = [r.render(node) for r in self.rows]
        meta = " &nbsp;·&nbsp; ".join(html.escape(r) for r in rendered if r)
        meta_html = f"<br><span style='color:#888;font-size:11px;'>{meta}</span>" if meta else ""

        body_html = ""
        if self.body:
            raw = str(node.get(self.body) or "").strip()
            if raw:
                lines = raw.splitlines()
                shown = [html.escape(line) for line in lines[: self.body_lines]]
                ellipsis = "…" if len(lines) > self.body_lines else ""
                body_html = (
                    "<hr style='border:0;border-top:1px solid #444;margin:6px 0;'>"
                    "<div style='font-family:monospace;font-size:11px;color:#ccc;"
                    "white-space:pre-wrap;'>" + "\n".join(shown) + ellipsis + "</div>"
                )

        return (
            f"<div style='font-family:sans-serif;font-size:12px;"
            f"background:#1e1e2e;color:#e0e0e0;padding:10px 14px;"
            f"border-radius:8px;border-left:4px solid {color};"
            f"max-width:{self.max_width}px;'>"
            f"<span style='background:{color};color:#fff;border-radius:4px;"
            f"padding:1px 7px;font-size:11px;font-weight:bold;'>{kind}</span>"
            f"&nbsp;&nbsp;<b style='font-size:13px;'>{title}</b>"
            f"{meta_html}{body_html}</div>"
        )
