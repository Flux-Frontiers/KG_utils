"""kg_utils.synthesis — Unified text and image synthesis backends.

Text backends  (TextBackend):
    omlx    — local oMLX / vLLM  (OpenAI wire protocol; MLX chain-of-thought suppressed)
    ollama  — local Ollama        (OpenAI wire protocol; no api_key required)
    openai  — OpenAI cloud API    (requires OPENAI_API_KEY or SYNTH_API_KEY)

Image backends  (ImageBackend):
    mflux-local  — local Flux2Klein via mflux  (Apple Silicon only)
    mflux-serve  — HTTP proxy to mflux-serve   (runs anywhere)
    openai       — OpenAI DALL-E               (requires OPENAI_API_KEY or IMAGE_API_KEY)

Env vars
--------
Text synthesis:
    SYNTH_BACKEND   omlx | ollama | openai          (default: omlx)
    SYNTH_ENDPOINT  override base URL
    SYNTH_API_KEY   bearer token / OpenAI key
    SYNTH_MODEL     model-id override
    Legacy aliases: VLLM_ENDPOINT_URL → SYNTH_ENDPOINT
                    VLLM_API_KEY      → SYNTH_API_KEY
                    VLLM_MODEL        → SYNTH_MODEL

Image synthesis:
    IMAGE_BACKEND   mflux-local | mflux-serve | openai   (default: mflux-serve)
    IMAGE_ENDPOINT  mflux-serve base URL
    IMAGE_API_KEY   OpenAI key (also reads OPENAI_API_KEY)
    IMAGE_MODEL     model-id override
    IMAGE_STEPS     inference steps  (default: 4, mflux backends only)
    Legacy alias:   GUTENKG_IMAGE_MODEL → IMAGE_MODEL

Quick start
-----------
    from kg_utils.synthesis import text_synthesizer_from_env, image_synthesizer_from_env

    text  = text_synthesizer_from_env()
    image = image_synthesizer_from_env()

    answer = text.synthesize_rag(query, hits)
    b64    = image.generate_b64(prompt)
    rewritten_prompt, err = text.rewrite_for_image(corpus_passage)
"""

from __future__ import annotations

from kg_utils.synthesis._config import (
    ImageBackend,
    ImageConfig,
    TextBackend,
    TextConfig,
    image_config_from_env,
    text_config_from_env,
)
from kg_utils.synthesis._image import ImageSynthesizer
from kg_utils.synthesis._text import TextSynthesizer
from kg_utils.synthesis.factory import (
    image_synth_for_backend,
    normalize_openai_base_url,
    text_synth_for_backend,
)


def text_synthesizer_from_env() -> TextSynthesizer:
    """Build a TextSynthesizer configured from the current process environment."""
    return TextSynthesizer(text_config_from_env())


def image_synthesizer_from_env() -> ImageSynthesizer:
    """Build an ImageSynthesizer configured from the current process environment."""
    return ImageSynthesizer(image_config_from_env())


__all__ = [
    "TextBackend",
    "ImageBackend",
    "TextConfig",
    "ImageConfig",
    "TextSynthesizer",
    "ImageSynthesizer",
    "text_synthesizer_from_env",
    "image_synthesizer_from_env",
    "text_config_from_env",
    "image_config_from_env",
    "normalize_openai_base_url",
    "text_synth_for_backend",
    "image_synth_for_backend",
]
