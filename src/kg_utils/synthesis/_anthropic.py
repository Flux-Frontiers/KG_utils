# © 2026 Eric G. Suchanek, PhD — Flux-Frontiers · SPDX-License-Identifier: Elastic-2.0
"""Anthropic Messages API calls for :class:`~kg_utils.synthesis._text.TextSynthesizer`.

The other backends speak the OpenAI wire protocol; Claude is called through
the official ``anthropic`` SDK, so it lives in its own module and
``TextSynthesizer`` hands it OpenAI-style messages.
"""

from __future__ import annotations

import logging

from kg_utils.synthesis._config import TextConfig

_log = logging.getLogger(__name__)

# Claude thinks before it answers, and the thinking counts against max_tokens,
# so a budget sized for the answer alone can end before the answer starts.
_MIN_MAX_TOKENS = 4096

# A grounded answer from given passages is chat-shaped work: low effort keeps
# the thinking short. Older models reject the parameter; see complete().
_EFFORT = "low"

# Models that take the server-side fallback: on a policy decline the API
# re-runs the request on a fallback model inside the same call.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _client(cfg: TextConfig):  # type: ignore[no-untyped-def]
    import anthropic  # type: ignore[import-unresolved]

    # No key in the config means the SDK's own resolution: ANTHROPIC_API_KEY,
    # ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile.
    return anthropic.Anthropic(api_key=cfg.api_key or None, base_url=cfg.endpoint or None)


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Move OpenAI-style system messages into the Messages API's ``system``."""
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    rest = [
        {"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "system"
    ]
    return system, rest


def complete(cfg: TextConfig, messages: list[dict], *, model: str, max_tokens: int) -> str | None:
    """One Messages API call; the answer's text, or ``None`` on failure.

    No sampling parameters are sent: current Claude models reject a
    non-default temperature. Effort is sent when the model takes it, and a
    model that refuses it is asked again without.

    :param cfg: The synthesizer's configuration (key and endpoint).
    :param messages: OpenAI-style chat messages; system messages become ``system``.
    :param model: The Claude model id.
    :param max_tokens: The caller's budget; raised to leave room for thinking.
    :returns: The text blocks joined, or ``None`` on an error, a refusal or no text.
    """
    import anthropic  # type: ignore[import-unresolved]

    system, rest = _split_system(messages)
    kwargs: dict = {
        "model": model,
        "max_tokens": max(max_tokens, _MIN_MAX_TOKENS),
        "messages": rest,
        "output_config": {"effort": _EFFORT},
    }
    if system:
        kwargs["system"] = system
    if model in _FALLBACK_MODELS:
        kwargs["betas"] = [_FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    try:
        client = _client(cfg)
        try:
            response = client.beta.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            if "effort" not in str(exc):
                raise
            del kwargs["output_config"]
            response = client.beta.messages.create(**kwargs)
    except anthropic.APIError as exc:
        _log.warning("text synthesis failed (anthropic, model %s): %s", model, exc)
        return None
    if response.stop_reason == "refusal":
        _log.warning(
            "text synthesis declined (anthropic, model %s): %s", model, response.stop_details
        )
        return None
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or None


def list_models(cfg: TextConfig) -> list[str]:
    """The Claude model ids the key can use, newest first; empty on failure."""
    try:
        return [m.id for m in _client(cfg).models.list() if m.id]
    except Exception:  # noqa: BLE001
        return []
