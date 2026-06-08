# © 2026 Eric G. Suchanek, PhD — Flux-Frontiers · SPDX-License-Identifier: Elastic-2.0
"""Synthesis backend configuration — enums, dataclasses, and env-var factories."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class TextBackend(str, Enum):
    """LLM backend for text synthesis."""

    OMLX = "omlx"  # local oMLX / vLLM — OpenAI wire protocol with MLX thinking suppression
    OLLAMA = "ollama"  # local Ollama       — OpenAI wire protocol, no api_key required
    OPENAI = "openai"  # OpenAI cloud API   — requires OPENAI_API_KEY / SYNTH_API_KEY


class ImageBackend(str, Enum):
    """Image generation backend."""

    MFLUX_LOCAL = "mflux-local"  # local Flux2Klein via mflux (Apple Silicon only)
    MFLUX_SERVE = "mflux-serve"  # HTTP proxy to a running mflux-serve instance
    OPENAI = "openai"  # OpenAI DALL-E — requires OPENAI_API_KEY / IMAGE_API_KEY


# Per-backend defaults — filled in when the user has not provided an override.
_TEXT_DEFAULTS: dict[TextBackend, dict[str, str]] = {
    TextBackend.OMLX: {
        "endpoint": "http://localhost:8080/v1",
        "model": "Qwen3-4B-Instruct-2507-MLX-8bit",
    },
    TextBackend.OLLAMA: {
        "endpoint": "http://localhost:11434/v1",
        "model": "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0",
    },
    TextBackend.OPENAI: {
        "endpoint": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}

_IMAGE_DEFAULTS: dict[ImageBackend, dict[str, str]] = {
    ImageBackend.MFLUX_LOCAL: {
        "model": "mlx-community/flux2-klein-4b-4bit",
    },
    ImageBackend.MFLUX_SERVE: {
        "server_url": "http://localhost:8090",
        "model": "flux2-klein-4b",
    },
    ImageBackend.OPENAI: {
        "model": "dall-e-3",
    },
}


@dataclass
class TextConfig:
    """Configuration for a text synthesis backend.

    :param backend: Which LLM provider to use.
    :param endpoint: Base URL override (empty = use backend default).
    :param api_key: Bearer token / OpenAI key (empty = not required for omlx/ollama).
    :param model: Model-id override (empty = use backend default).
    :param max_tokens: Maximum tokens in the generated response.
    :param suppress_thinking: Strip ``<think>`` blocks and pass ``extra_body`` to disable
                              chain-of-thought for oMLX backends.
    """

    backend: TextBackend = TextBackend.OMLX
    endpoint: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = 2048
    suppress_thinking: bool = True

    def resolved_endpoint(self) -> str:
        return self.endpoint or _TEXT_DEFAULTS[self.backend]["endpoint"]

    def resolved_model(self) -> str:
        return self.model or _TEXT_DEFAULTS[self.backend]["model"]


@dataclass
class ImageConfig:
    """Configuration for an image synthesis backend.

    :param backend: Which image provider to use.
    :param server_url: mflux-serve base URL override (empty = use backend default).
    :param api_key: OpenAI API key (empty = read from OPENAI_API_KEY at call time).
    :param model: Model-id override (empty = use backend default).
    :param steps: Inference steps for mflux backends (ignored for OpenAI).
    """

    backend: ImageBackend = ImageBackend.MFLUX_SERVE
    server_url: str = ""
    api_key: str = ""
    model: str = ""
    steps: int = 4

    def resolved_server_url(self) -> str:
        return self.server_url or _IMAGE_DEFAULTS[self.backend].get("server_url", "")

    def resolved_model(self) -> str:
        return self.model or _IMAGE_DEFAULTS[self.backend].get("model", "")


# ---------------------------------------------------------------------------
# Env-var factories
# ---------------------------------------------------------------------------


def text_config_from_env() -> TextConfig:
    """Build a TextConfig from environment variables.

    Reads ``SYNTH_BACKEND`` (default ``omlx``) plus endpoint/key/model overrides.
    Legacy vars ``VLLM_ENDPOINT_URL``, ``VLLM_API_KEY``, and ``VLLM_MODEL`` are
    honoured as fallbacks so existing deployments need no changes.
    """
    backend_str = os.environ.get("SYNTH_BACKEND", "omlx")
    try:
        backend = TextBackend(backend_str)
    except ValueError:
        backend = TextBackend.OMLX

    endpoint = os.environ.get("SYNTH_ENDPOINT") or os.environ.get("VLLM_ENDPOINT_URL") or ""
    api_key = (
        os.environ.get("SYNTH_API_KEY")
        or os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    model = os.environ.get("SYNTH_MODEL") or os.environ.get("VLLM_MODEL") or ""
    return TextConfig(backend=backend, endpoint=endpoint, api_key=api_key, model=model)


def image_config_from_env() -> ImageConfig:
    """Build an ImageConfig from environment variables.

    Reads ``IMAGE_BACKEND`` (default ``mflux-serve``) plus server-url/key/model/steps overrides.
    Legacy var ``GUTENKG_IMAGE_MODEL`` is honoured as a fallback for ``IMAGE_MODEL``.
    """
    backend_str = os.environ.get("IMAGE_BACKEND", "mflux-serve")
    try:
        backend = ImageBackend(backend_str)
    except ValueError:
        backend = ImageBackend.MFLUX_SERVE

    server_url = os.environ.get("IMAGE_ENDPOINT") or ""
    api_key = os.environ.get("IMAGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    model = os.environ.get("IMAGE_MODEL") or os.environ.get("GUTENKG_IMAGE_MODEL") or ""
    steps = int(os.environ.get("IMAGE_STEPS", "4"))
    return ImageConfig(
        backend=backend, server_url=server_url, api_key=api_key, model=model, steps=steps
    )
