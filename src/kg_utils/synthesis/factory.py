# © 2026 Eric G. Suchanek, PhD — Flux-Frontiers · SPDX-License-Identifier: Elastic-2.0
"""Synthesis backend factory helpers for per-request backend overrides."""

from __future__ import annotations

import os

from kg_utils.synthesis._config import (
    ImageBackend,
    ImageConfig,
    TextBackend,
    TextConfig,
)
from kg_utils.synthesis._image import ImageSynthesizer
from kg_utils.synthesis._text import TextSynthesizer

__all__ = [
    "normalize_openai_base_url",
    "text_synth_for_backend",
    "image_synth_for_backend",
]


def normalize_openai_base_url(endpoint: str) -> str:
    """Normalize an OpenAI-wire endpoint so it ends with /v1.

    Returns an empty string when endpoint is empty.
    """
    ep = (endpoint or "").strip().rstrip("/")
    if not ep:
        return ""
    if ep.endswith("/v1"):
        return ep
    return f"{ep}/v1"


def text_synth_for_backend(backend: str, fallback: TextSynthesizer) -> TextSynthesizer:
    """Return a TextSynthesizer configured for a specific backend override.

    Unknown or empty backend strings return ``fallback``.
    """
    backend_str = (backend or "").strip().lower()
    if not backend_str:
        return fallback

    try:
        selected = TextBackend(backend_str)
    except ValueError:
        return fallback

    if selected == TextBackend.OMLX:
        endpoint = os.environ.get("SYNTH_ENDPOINT") or os.environ.get("VLLM_ENDPOINT_URL") or ""
        endpoint = normalize_openai_base_url(endpoint)
        api_key = os.environ.get("SYNTH_API_KEY") or os.environ.get("VLLM_API_KEY") or ""
        model = os.environ.get("SYNTH_MODEL") or os.environ.get("VLLM_MODEL") or ""
        return TextSynthesizer(
            TextConfig(backend=selected, endpoint=endpoint, api_key=api_key, model=model)
        )

    if selected == TextBackend.OLLAMA:
        endpoint = os.environ.get("OLLAMA_ENDPOINT") or ""
        return TextSynthesizer(TextConfig(backend=selected, endpoint=endpoint))

    if selected == TextBackend.OPENAI:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SYNTH_API_KEY") or ""
        return TextSynthesizer(TextConfig(backend=selected, api_key=api_key))

    return fallback


def image_synth_for_backend(backend: str, fallback: ImageSynthesizer) -> ImageSynthesizer:
    """Return an ImageSynthesizer configured for a specific backend override.

    Unknown or empty backend strings return ``fallback``.
    """
    backend_str = (backend or "").strip().lower()
    if not backend_str:
        return fallback

    try:
        selected = ImageBackend(backend_str)
    except ValueError:
        return fallback

    if selected == ImageBackend.OPENAI:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("IMAGE_API_KEY") or ""
        return ImageSynthesizer(ImageConfig(backend=selected, api_key=api_key))

    if selected == ImageBackend.MFLUX_SERVE:
        server_url = os.environ.get("IMAGE_ENDPOINT") or ""
        return ImageSynthesizer(ImageConfig(backend=selected, server_url=server_url))

    if selected == ImageBackend.MFLUX_LOCAL:
        model = os.environ.get("IMAGE_MODEL") or os.environ.get("GUTENKG_IMAGE_MODEL") or ""
        return ImageSynthesizer(ImageConfig(backend=selected, model=model))

    return fallback
