"""Tests for kg_utils.synthesis._text — TextSynthesizer.

Requires the ``synthesis`` optional extra (openai package).
The entire module is skipped if openai is not installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai", reason="openai package not installed — skipping synthesis text tests")

from kg_utils.synthesis._config import TextBackend, TextConfig
from kg_utils.synthesis._text import TextSynthesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str) -> MagicMock:
    """Minimal mock ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _snippet(content: str, **meta) -> dict:
    return {
        "content": content,
        "genre": "fiction",
        "author": "A. Author",
        "title": "A Book",
        **meta,
    }


def _make_synth(backend: TextBackend = TextBackend.OMLX, **kw) -> TextSynthesizer:
    return TextSynthesizer(TextConfig(backend=backend, **kw))


# ---------------------------------------------------------------------------
# _extra_body — thinking suppression
# ---------------------------------------------------------------------------


def test_extra_body_omlx_suppress_thinking_on() -> None:
    synth = _make_synth(TextBackend.OMLX, suppress_thinking=True)
    extra = synth._extra_body()
    assert extra is not None
    assert extra["think"] is False


def test_extra_body_omlx_suppress_thinking_off() -> None:
    synth = _make_synth(TextBackend.OMLX, suppress_thinking=False)
    assert synth._extra_body() is None


def test_extra_body_ollama_always_none() -> None:
    synth = _make_synth(TextBackend.OLLAMA, suppress_thinking=True)
    assert synth._extra_body() is None


def test_extra_body_openai_always_none() -> None:
    synth = _make_synth(TextBackend.OPENAI, suppress_thinking=True)
    assert synth._extra_body() is None


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------


def test_strip_thinking_removes_think_block() -> None:
    raw = "<think>internal reasoning here</think>Final answer."
    assert TextSynthesizer._strip_thinking(raw) == "Final answer."


def test_strip_thinking_multiline_block() -> None:
    raw = "<think>\nline one\nline two\n</think>\nClean response."
    assert TextSynthesizer._strip_thinking(raw) == "Clean response."


def test_strip_thinking_no_block_unchanged() -> None:
    raw = "Plain response with no thinking."
    assert TextSynthesizer._strip_thinking(raw) == raw


def test_strip_thinking_multiple_blocks() -> None:
    raw = "<think>a</think>middle<think>b</think>end"
    assert TextSynthesizer._strip_thinking(raw) == "middleend"


def test_strip_thinking_only_block_returns_empty() -> None:
    raw = "<think>nothing but thinking</think>"
    assert TextSynthesizer._strip_thinking(raw) == ""


# ---------------------------------------------------------------------------
# complete — general-purpose chat completion
# ---------------------------------------------------------------------------


def test_complete_returns_content() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("Summary text.")
        synth = _make_synth()
        result = synth.complete([{"role": "user", "content": "Summarize this."}])
    assert result == "Summary text."


def test_complete_strips_think_blocks() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion(
            "<think>reasoning</think>Clean summary."
        )
        synth = _make_synth()
        result = synth.complete([{"role": "user", "content": "q"}])
    assert result == "Clean summary."


def test_complete_returns_none_when_llm_raises() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.side_effect = RuntimeError("connection refused")
        synth = _make_synth()
        result = synth.complete([{"role": "user", "content": "q"}])
    assert result is None


def test_complete_returns_none_for_empty_completion() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("")
        synth = _make_synth()
        result = synth.complete([{"role": "user", "content": "q"}])
    assert result is None


def test_complete_passes_messages_verbatim() -> None:
    messages = [
        {"role": "system", "content": "You summarize."},
        {"role": "user", "content": "Fold this conversation."},
    ]
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("ok")
        synth = _make_synth()
        synth.complete(messages)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["messages"] == messages


def test_complete_model_override_passed_to_llm() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("ok")
        synth = _make_synth()
        synth.complete([{"role": "user", "content": "q"}], model="config-override")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "config-override"


def test_complete_uses_config_model_when_no_override() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("ok")
        synth = _make_synth(model="cfg-model")
        synth.complete([{"role": "user", "content": "q"}])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "cfg-model"


def test_complete_max_tokens_and_temperature_override() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("ok")
        synth = _make_synth()
        synth.complete([{"role": "user", "content": "q"}], max_tokens=64, temperature=0.2)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["max_tokens"] == 64
    assert call_kwargs["temperature"] == 0.2


def test_complete_omlx_passes_extra_body() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("ok")
        synth = _make_synth(TextBackend.OMLX, suppress_thinking=True)
        synth.complete([{"role": "user", "content": "q"}])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["extra_body"]["think"] is False


def test_complete_ollama_no_extra_body() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("ok")
        synth = _make_synth(TextBackend.OLLAMA)
        synth.complete([{"role": "user", "content": "q"}])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert "extra_body" not in call_kwargs


# ---------------------------------------------------------------------------
# synthesize_rag — input validation
# ---------------------------------------------------------------------------


def test_synthesize_rag_empty_snippets_returns_none() -> None:
    synth = _make_synth()
    assert synth.synthesize_rag("question", []) is None


def test_synthesize_rag_snippets_without_content_returns_none() -> None:
    synth = _make_synth()
    snippets = [{"summary": "no content key here"}]
    assert synth.synthesize_rag("question", snippets) is None


def test_synthesize_rag_skips_empty_content_strings() -> None:
    synth = _make_synth()
    snippets = [{"content": ""}, {"content": "   "}]
    # Empty-string content is falsy, so both are filtered.
    assert synth.synthesize_rag("question", snippets) is None


# ---------------------------------------------------------------------------
# synthesize_rag — happy path (mocked LLM)
# ---------------------------------------------------------------------------


def test_synthesize_rag_returns_content() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("The answer is 42.")
        synth = _make_synth()
        result = synth.synthesize_rag("question", [_snippet("passage text")])
    assert result == "The answer is 42."


def test_synthesize_rag_strips_think_blocks_from_response() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion(
            "<think>internal</think>Clean answer."
        )
        synth = _make_synth()
        result = synth.synthesize_rag("question", [_snippet("passage")])
    assert result == "Clean answer."


def test_synthesize_rag_returns_none_when_llm_raises() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.side_effect = RuntimeError("connection refused")
        synth = _make_synth()
        result = synth.synthesize_rag("question", [_snippet("text")])
    assert result is None


def test_synthesize_rag_returns_none_for_empty_completion() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("")
        synth = _make_synth()
        result = synth.synthesize_rag("question", [_snippet("text")])
    assert result is None


def test_synthesize_rag_returns_none_for_only_think_completion() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion(
            "<think>just thinking</think>"
        )
        synth = _make_synth()
        result = synth.synthesize_rag("question", [_snippet("text")])
    assert result is None


# ---------------------------------------------------------------------------
# synthesize_rag — message structure
# ---------------------------------------------------------------------------


def test_synthesize_rag_messages_contain_query() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("answer")
        synth = _make_synth()
        synth.synthesize_rag("What is the meaning of life?", [_snippet("text")])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    user_msg = call_kwargs["messages"][1]["content"]
    assert "What is the meaning of life?" in user_msg


def test_synthesize_rag_messages_contain_snippet_content() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("answer")
        synth = _make_synth()
        synth.synthesize_rag("q", [_snippet("The quick brown fox")])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    user_msg = call_kwargs["messages"][1]["content"]
    assert "The quick brown fox" in user_msg


def test_synthesize_rag_messages_include_author_and_title_header() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("answer")
        synth = _make_synth()
        synth.synthesize_rag("q", [_snippet("text", author="Tolstoy", title="War and Peace")])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    user_msg = call_kwargs["messages"][1]["content"]
    assert "Tolstoy" in user_msg
    assert "War and Peace" in user_msg


def test_synthesize_rag_system_message_used() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("answer")
        synth = _make_synth()
        synth.synthesize_rag("q", [_snippet("text")])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["messages"][0]["role"] == "system"


def test_synthesize_rag_custom_system_prompt() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("answer")
        synth = _make_synth()
        synth.synthesize_rag("q", [_snippet("text")], system="Custom instructions.")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["messages"][0]["content"] == "Custom instructions."


def test_synthesize_rag_model_override_passed_to_llm() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("answer")
        synth = _make_synth()
        synth.synthesize_rag("q", [_snippet("text")], model="llama3.2:3b")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "llama3.2:3b"


def test_synthesize_rag_uses_config_model_when_no_override() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("answer")
        synth = _make_synth(model="config-model")
        synth.synthesize_rag("q", [_snippet("text")])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "config-model"


# ---------------------------------------------------------------------------
# synthesize_rag — max_k
# ---------------------------------------------------------------------------


def test_synthesize_rag_respects_max_k() -> None:
    snippets = [_snippet(f"passage {i}") for i in range(20)]
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("answer")
        synth = _make_synth()
        synth.synthesize_rag("q", snippets, max_k=3)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    user_msg = call_kwargs["messages"][1]["content"]
    # Only 3 passages should appear; check that passage 3+ is absent.
    assert "passage 0" in user_msg
    assert "passage 2" in user_msg
    assert "passage 3" not in user_msg


# ---------------------------------------------------------------------------
# synthesize_rag — extra_body for oMLX
# ---------------------------------------------------------------------------


def test_synthesize_rag_omlx_passes_extra_body() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("answer")
        synth = _make_synth(TextBackend.OMLX, suppress_thinking=True)
        synth.synthesize_rag("q", [_snippet("text")])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert "extra_body" in call_kwargs
    assert call_kwargs["extra_body"]["think"] is False


def test_synthesize_rag_ollama_no_extra_body() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("answer")
        synth = _make_synth(TextBackend.OLLAMA)
        synth.synthesize_rag("q", [_snippet("text")])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert "extra_body" not in call_kwargs


# ---------------------------------------------------------------------------
# rewrite_for_image
# ---------------------------------------------------------------------------


def test_rewrite_for_image_returns_rewritten_prompt() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion(
            "A candlelit tavern with rough-hewn wooden tables."
        )
        synth = _make_synth()
        prompt, err = synth.rewrite_for_image("Original passage about a tavern.")
    assert prompt == "A candlelit tavern with rough-hewn wooden tables."
    assert err is None


def test_rewrite_for_image_strips_think_blocks() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion(
            "<think>thinking...</think>Bright sunlit meadow."
        )
        synth = _make_synth()
        prompt, err = synth.rewrite_for_image("A meadow passage.")
    assert prompt == "Bright sunlit meadow."
    assert err is None


def test_rewrite_for_image_fallback_on_llm_failure() -> None:
    original = "The original corpus passage."
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.side_effect = RuntimeError("timeout")
        synth = _make_synth()
        prompt, err = synth.rewrite_for_image(original)
    assert prompt == original
    assert err is not None
    assert len(err) > 0


def test_rewrite_for_image_fallback_on_empty_completion() -> None:
    original = "Source text."
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("")
        synth = _make_synth()
        prompt, err = synth.rewrite_for_image(original)
    assert prompt == original
    assert err is not None


def test_rewrite_for_image_model_override() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("scene description")
        synth = _make_synth()
        synth.rewrite_for_image("passage", model="fast-model")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "fast-model"


def test_rewrite_for_image_uses_limited_max_tokens() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.chat.completions.create.return_value = _completion("scene")
        synth = _make_synth()
        synth.rewrite_for_image("passage")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
    # rewrite_for_image is intentionally short — max_tokens should be <= 300
    assert call_kwargs["max_tokens"] <= 300


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


def _mock_model(id_: str) -> MagicMock:
    m = MagicMock()
    m.id = id_
    return m


def test_list_models_returns_ids() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.models.list.return_value.data = [
            _mock_model("model-a"),
            _mock_model("model-b"),
        ]
        synth = _make_synth()
        models = synth.list_models()
    assert models == ["model-a", "model-b"]


def test_list_models_filters_empty_ids() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.models.list.return_value.data = [
            _mock_model("model-a"),
            _mock_model(""),  # empty id — should be filtered
        ]
        synth = _make_synth()
        models = synth.list_models()
    assert models == ["model-a"]


def test_list_models_returns_empty_on_exception() -> None:
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.models.list.side_effect = ConnectionError("unreachable")
        synth = _make_synth()
        models = synth.list_models()
    assert models == []
