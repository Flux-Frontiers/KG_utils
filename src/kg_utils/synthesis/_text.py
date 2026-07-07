# © 2026 Eric G. Suchanek, PhD — Flux-Frontiers · SPDX-License-Identifier: Elastic-2.0
"""TextSynthesizer — unified LLM completion across oMLX, Ollama, and OpenAI backends."""

from __future__ import annotations

import re

from kg_utils.synthesis._config import TextBackend, TextConfig

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_RAG_SYSTEM = (
    "You are a literary guide to the Project Gutenberg corpus. "
    "Answer the question using ONLY the provided source passages. "
    "Do NOT use any prior knowledge — if something is in the passages, "
    "report it; if it is not in the passages, say so. "
    "Never contradict or override what the passages say based on what "
    "you believe to be true. Be concise and specific. "
    "Cite the author and work when relevant."
)

_IMAGE_REWRITE_SYSTEM = (
    "You are an expert art director. Given a passage of historical text, write a single concise "
    "image generation prompt (one paragraph, no bullet points, no quotation marks) that vividly "
    "describes the scene for a text-to-image model. Focus on visual elements: setting, lighting, "
    "figures, mood, and artistic style. Do NOT include any text, labels, captions, or words in "
    "the scene description. Output ONLY the prompt, nothing else."
)


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class TextSynthesizer:
    """Unified text completion client across oMLX, Ollama, and OpenAI backends.

    Uses the ``openai`` Python package for all three — each backend is just a
    different ``base_url`` and optionally an ``extra_body`` for thinking suppression.

    :param config: Backend configuration produced by ``text_config_from_env()`` or
                   built directly as a ``TextConfig`` dataclass.
    """

    def __init__(self, config: TextConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self):  # type: ignore[no-untyped-def]
        from openai import OpenAI  # type: ignore[import-unresolved]

        cfg = self._cfg
        api_key = cfg.api_key or "not-needed"
        return OpenAI(base_url=cfg.resolved_endpoint(), api_key=api_key)

    def _extra_body(self) -> dict | None:
        if self._cfg.suppress_thinking and self._cfg.backend == TextBackend.OMLX:
            return {"think": False, "chat_template_kwargs": {"enable_thinking": False}}
        return None

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str | None:
        try:
            client = self._client()
            mdl = model or self._cfg.resolved_model()
            extra = self._extra_body()
            kwargs: dict = {
                "model": mdl,
                "messages": messages,
                "max_tokens": max_tokens or self._cfg.max_tokens,
                "temperature": temperature,
            }
            if extra:
                kwargs["extra_body"] = extra
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            return self._strip_thinking(raw) or None
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str | None:
        """Run a raw chat completion against the configured backend.

        General-purpose entry point for callers that build their own message
        list — summarization, classification, arbitrary prompting — rather than
        using :meth:`synthesize_rag` or :meth:`rewrite_for_image`. Applies the
        same oMLX thinking suppression and ``<think>`` stripping as the other
        public methods.

        :param messages: OpenAI-style chat messages (``[{"role": ..., "content": ...}]``).
        :param model: Override the configured model for this single call.
        :param max_tokens: Override the configured ``max_tokens`` for this call.
        :param temperature: Sampling temperature (lower = more deterministic).
        :returns: The completion text with thinking stripped, or ``None`` on
                  failure or empty output.
        """
        return self._complete(messages, model=model, max_tokens=max_tokens, temperature=temperature)

    def list_models(self) -> list[str]:
        """Return the model IDs available at the configured endpoint.

        :returns: List of model ID strings; empty list on failure or if the
                  endpoint does not support the ``/v1/models`` route.
        """
        try:
            return [m.id for m in self._client().models.list().data if m.id]
        except Exception:  # noqa: BLE001
            return []

    def synthesize_rag(
        self,
        query: str,
        snippets: list[dict],
        *,
        model: str | None = None,
        max_k: int = 12,
        system: str | None = None,
    ) -> str | None:
        """Synthesize a grounded answer from corpus retrieval snippets.

        Formats snippet metadata (genre · author · title) as a structured context
        block, then calls the configured LLM to answer *only* from those passages.

        :param query: Natural-language question.
        :param snippets: Hit dicts from KGRAG — must contain a ``content`` key.
        :param model: Override the configured model for this single call.
        :param max_k: Maximum number of snippets included in the context window.
        :param system: Override the default RAG system prompt.
        :returns: Synthesized answer, or ``None`` on failure or when no snippets
                  carry content.
        """
        items = [s for s in snippets[:max_k] if (s.get("content") or "").strip()]
        if not items:
            return None

        ctx_parts: list[str] = []
        for s in items:
            genre = s.get("genre", s.get("kg_kind", ""))
            author = s.get("author") or ""
            title = s.get("title") or s.get("name") or ""
            header = " · ".join(x for x in [genre, author, title] if x)
            ctx_parts.append(f"[{header}]\n{s['content'].strip()}")

        ctx = "\n\n".join(ctx_parts)
        messages = [
            {"role": "system", "content": system or _RAG_SYSTEM},
            {"role": "user", "content": f"Source passages:\n{ctx}\n\nQuestion: {query}"},
        ]
        return self._complete(messages, model=model, temperature=0.3)

    def rewrite_for_image(
        self,
        corpus_text: str,
        *,
        model: str | None = None,
    ) -> tuple[str, str | None]:
        """Rewrite a historical corpus passage into an image-generation prompt.

        Uses the same backend and model as RAG synthesis so no second model load
        is required.  Pass ``model=`` to use a lighter model for this step if
        synthesis and rewrite are called from the same hot path.

        :param corpus_text: Raw historical passage to convert into a visual scene.
        :param model: Override the configured model for this single call.
        :returns: ``(prompt, error)`` — prompt is the rewritten scene description
                  (falls back to the original text on failure); error is ``None``
                  on success or a short message otherwise.
        """
        messages = [
            {"role": "system", "content": _IMAGE_REWRITE_SYSTEM},
            {"role": "user", "content": corpus_text},
        ]
        result = self._complete(messages, model=model, max_tokens=300, temperature=0.7)
        if result is None:
            return corpus_text, "synthesis returned no content"
        return result, None
