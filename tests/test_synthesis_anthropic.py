"""Tests for kg_utils.synthesis._anthropic -- the Claude backend of TextSynthesizer.

The anthropic SDK is mocked; no request leaves the machine.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

anthropic = pytest.importorskip("anthropic", reason="anthropic package not installed")

from kg_utils.synthesis._config import TextBackend, TextConfig, text_config_from_env  # noqa: E402
from kg_utils.synthesis._text import TextSynthesizer  # noqa: E402


def _response(*texts: str, stop_reason: str = "end_turn") -> MagicMock:
    blocks = []
    for t in texts:
        b = MagicMock()
        b.type = "text"
        b.text = t
        blocks.append(b)
    thinking = MagicMock()
    thinking.type = "thinking"
    resp = MagicMock()
    resp.content = [thinking, *blocks]
    resp.stop_reason = stop_reason
    return resp


def _bad_request(message: str) -> Exception:
    return anthropic.BadRequestError(message, response=MagicMock(status_code=400), body=None)


FAKE_KEY = "test"


def _synth(**kw) -> TextSynthesizer:
    return TextSynthesizer(TextConfig(backend=TextBackend.ANTHROPIC, api_key=FAKE_KEY, **kw))


MESSAGES = [
    {"role": "system", "content": "Answer from the passages."},
    {"role": "user", "content": "Who is Ishmael?"},
]


def test_complete_returns_the_text_blocks() -> None:
    with patch("anthropic.Anthropic") as cls:
        cls.return_value.beta.messages.create.return_value = _response("Ishmael ", "narrates.")
        assert _synth().complete(MESSAGES) == "Ishmael narrates."


def test_system_messages_become_the_system_parameter() -> None:
    with patch("anthropic.Anthropic") as cls:
        create = cls.return_value.beta.messages.create
        create.return_value = _response("ok")
        _synth().complete(MESSAGES)
        kwargs = create.call_args[1]
    assert kwargs["system"] == "Answer from the passages."
    assert kwargs["messages"] == [{"role": "user", "content": "Who is Ishmael?"}]


def test_no_temperature_and_room_for_thinking() -> None:
    with patch("anthropic.Anthropic") as cls:
        create = cls.return_value.beta.messages.create
        create.return_value = _response("ok")
        _synth().complete(MESSAGES, max_tokens=300, temperature=0.7)
        kwargs = create.call_args[1]
    assert "temperature" not in kwargs
    assert kwargs["max_tokens"] >= 4096
    assert kwargs["output_config"] == {"effort": "low"}


def test_the_default_model_gets_the_server_side_fallback() -> None:
    with patch("anthropic.Anthropic") as cls:
        create = cls.return_value.beta.messages.create
        create.return_value = _response("ok")
        _synth().complete(MESSAGES)
        kwargs = create.call_args[1]
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]


def test_an_older_model_gets_no_fallback() -> None:
    with patch("anthropic.Anthropic") as cls:
        create = cls.return_value.beta.messages.create
        create.return_value = _response("ok")
        _synth().complete(MESSAGES, model="claude-haiku-4-5")
        kwargs = create.call_args[1]
    assert "fallbacks" not in kwargs and "betas" not in kwargs


def test_a_model_that_refuses_effort_is_asked_again_without() -> None:
    with patch("anthropic.Anthropic") as cls:
        create = cls.return_value.beta.messages.create
        create.side_effect = [
            _bad_request("effort is not supported on this model"),
            _response("ok"),
        ]
        assert _synth().complete(MESSAGES, model="claude-haiku-4-5") == "ok"
    assert "output_config" in create.call_args_list[0][1]
    assert "output_config" not in create.call_args_list[1][1]


def test_other_errors_return_none_and_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    with patch("anthropic.Anthropic") as cls:
        cls.return_value.beta.messages.create.side_effect = _bad_request("model not found")
        with caplog.at_level("WARNING"):
            assert _synth().complete(MESSAGES) is None
    assert "model not found" in caplog.text
    assert cls.return_value.beta.messages.create.call_count == 1


def test_a_refusal_returns_none() -> None:
    with patch("anthropic.Anthropic") as cls:
        cls.return_value.beta.messages.create.return_value = _response("", stop_reason="refusal")
        assert _synth().complete(MESSAGES) is None


def test_synthesize_rag_runs_through_claude() -> None:
    with patch("anthropic.Anthropic") as cls:
        cls.return_value.beta.messages.create.return_value = _response(
            "From the passage: a sailor."
        )
        answer = _synth().synthesize_rag("Who is Ishmael?", [{"content": "Call me Ishmael."}])
    assert answer == "From the passage: a sailor."


def test_list_models_returns_the_ids() -> None:
    models = [MagicMock(id="claude-opus-5-5"), MagicMock(id="claude-haiku-5-5")]
    with patch("anthropic.Anthropic") as cls:
        cls.return_value.models.list.return_value = models
        assert _synth().list_models() == ["claude-opus-5-5", "claude-haiku-5-5"]


def test_env_config_never_hands_claude_an_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_BACKEND", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("SYNTH_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = text_config_from_env()
    assert cfg.backend == TextBackend.ANTHROPIC
    assert cfg.api_key == ""


def test_missing_credentials_return_none(caplog: pytest.LogCaptureFixture) -> None:
    with patch("anthropic.Anthropic") as cls:
        cls.return_value.beta.messages.create.side_effect = TypeError(
            "Could not resolve authentication method"
        )
        with caplog.at_level("WARNING"):
            assert _synth().complete(MESSAGES) is None
    assert "authentication" in caplog.text
