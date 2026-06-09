# © 2026 Eric G. Suchanek, PhD — Flux-Frontiers · SPDX-License-Identifier: Elastic-2.0
"""Shared handler operation dispatch for models, rewrite, and imagine."""

from __future__ import annotations

from collections.abc import Callable

from kg_utils.synthesis._image import ImageSynthesizer
from kg_utils.synthesis._text import TextSynthesizer

__all__ = ["handle_aux_ops"]


def handle_aux_ops(
    inp: dict,
    text_synth_factory: Callable[[str], TextSynthesizer],
    image_synth_factory: Callable[[str], ImageSynthesizer],
) -> dict | None:
    """Handle shared non-query worker operations.

    Returns:
    - operation payload dict when op is recognized
    - ``None`` when input has no recognized operation
    """
    op = inp.get("op")

    if op == "models":
        synth = text_synth_factory(inp.get("backend", ""))
        # Existing handlers expose the active model via synthesizer config internals.
        return {
            "models": synth.list_models(),
            "default": synth._cfg.resolved_model(),  # pylint: disable=protected-access
        }

    if op == "rewrite":
        text = (inp.get("text") or "").strip()
        if not text:
            return {"error": "rewrite requires a non-empty 'text'"}

        synth = text_synth_factory(inp.get("backend", ""))
        model_override = (inp.get("model") or "").strip() or None
        prompt, error = synth.rewrite_for_image(text, model=model_override)
        return {"prompt": prompt, "error": error}

    if op == "imagine":
        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            return {"error": "imagine requires a non-empty 'prompt'"}

        aspect = inp.get("aspect_ratio", "3:2")
        size = inp.get("size")
        seed = inp.get("seed")
        steps = inp.get("steps")
        img_synth = image_synth_factory(inp.get("image_backend", ""))
        seed_int = int(seed) if seed is not None else None
        steps_int = int(steps) if steps is not None else None

        try:
            if size is not None:
                b64 = img_synth.generate_b64(
                    prompt,
                    aspect_ratio=aspect,
                    seed=seed_int,
                    steps=steps_int,
                    size=size,
                )
            else:
                b64 = img_synth.generate_b64(
                    prompt,
                    aspect_ratio=aspect,
                    seed=seed_int,
                    steps=steps_int,
                )
            result = {
                "image_b64": b64,
                "prompt": prompt,
                "aspect_ratio": aspect,
                "image_model": img_synth._cfg.resolved_model(),  # pylint: disable=protected-access
                "image_backend": img_synth._cfg.backend.value,  # pylint: disable=protected-access
            }
            if size is not None:
                result["size"] = size
            return result
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            return {"error": f"image generation failed: {exc}"}

    return None
