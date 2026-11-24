"""kg_utils.synthesis — Unified text synthesis backends.

Image generation moved to the ``imagewright`` package (0.32.0). This module
keeps the text side, including :meth:`TextSynthesizer.rewrite_for_image`,
which turns a passage into an image prompt.

Text backends  (TextBackend):
    omlx    — local oMLX / vLLM  (OpenAI wire protocol; MLX chain-of-thought suppressed)
    ollama  — local Ollama        (OpenAI wire protocol; no api_key required)
    openai  — OpenAI cloud API    (requires OPENAI_API_KEY or SYNTH_API_KEY)

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


Quick start
-----------
    from kg_utils.synthesis import text_synthesizer_from_env

    text  = text_synthesizer_from_env()

    answer = text.synthesize_rag(query, hits)
    rewritten_prompt, err = text.rewrite_for_image(corpus_passage)
"""

from __future__ import annotations

from kg_utils.synthesis._config import (
    TextBackend,
    TextConfig,
    text_config_from_env,
)
from kg_utils.synthesis._text import (
    IMAGE_REWRITE_SYSTEM,
    IMAGE_REWRITE_TECHNICAL_SYSTEM,
    TextSynthesizer,
)
from kg_utils.synthesis.factory import (
    normalize_openai_base_url,
    text_synth_for_backend,
)


def text_synthesizer_from_env() -> TextSynthesizer:
    """Build a TextSynthesizer configured from the current process environment."""
    return TextSynthesizer(text_config_from_env())


__all__ = [
    "TextBackend",
    "TextConfig",
    "TextSynthesizer",
    "text_synthesizer_from_env",
    "text_config_from_env",
    "normalize_openai_base_url",
    "text_synth_for_backend",
    "IMAGE_REWRITE_SYSTEM",
    "IMAGE_REWRITE_TECHNICAL_SYSTEM",
]
