"""Tests for kg_utils.synthesis._config — enums, dataclasses, and env-var factories.

All tests are pure Python (stdlib only); no external packages required.
"""

from __future__ import annotations


import pytest

from kg_utils.synthesis._config import (
    ImageBackend,
    ImageConfig,
    TextBackend,
    TextConfig,
    _IMAGE_DEFAULTS,
    _TEXT_DEFAULTS,
    image_config_from_env,
    text_config_from_env,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYNTH_VARS = [
    "SYNTH_BACKEND",
    "SYNTH_ENDPOINT",
    "SYNTH_API_KEY",
    "SYNTH_MODEL",
    "VLLM_ENDPOINT_URL",
    "VLLM_API_KEY",
    "VLLM_MODEL",
    "OPENAI_API_KEY",
]

_IMAGE_VARS = [
    "IMAGE_BACKEND",
    "IMAGE_ENDPOINT",
    "IMAGE_API_KEY",
    "IMAGE_MODEL",
    "IMAGE_STEPS",
    "GUTENKG_IMAGE_MODEL",
    "OPENAI_API_KEY",
]


@pytest.fixture()
def clean_synth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all synthesis text env vars."""
    for k in _SYNTH_VARS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture()
def clean_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all synthesis image env vars."""
    for k in _IMAGE_VARS:
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# TextBackend enum
# ---------------------------------------------------------------------------


def test_text_backend_values() -> None:
    assert TextBackend.OMLX.value == "omlx"
    assert TextBackend.OLLAMA.value == "ollama"
    assert TextBackend.OPENAI.value == "openai"


def test_text_backend_from_string() -> None:
    assert TextBackend("omlx") is TextBackend.OMLX
    assert TextBackend("ollama") is TextBackend.OLLAMA
    assert TextBackend("openai") is TextBackend.OPENAI


def test_text_backend_invalid_raises() -> None:
    with pytest.raises(ValueError):
        TextBackend("vllm")


# ---------------------------------------------------------------------------
# ImageBackend enum
# ---------------------------------------------------------------------------


def test_image_backend_values() -> None:
    assert ImageBackend.MFLUX_LOCAL.value == "mflux-local"
    assert ImageBackend.MFLUX_SERVE.value == "mflux-serve"
    assert ImageBackend.OPENAI.value == "openai"


def test_image_backend_from_string() -> None:
    assert ImageBackend("mflux-local") is ImageBackend.MFLUX_LOCAL
    assert ImageBackend("mflux-serve") is ImageBackend.MFLUX_SERVE
    assert ImageBackend("openai") is ImageBackend.OPENAI


# ---------------------------------------------------------------------------
# TextConfig defaults per backend
# ---------------------------------------------------------------------------


def test_text_config_omlx_defaults() -> None:
    cfg = TextConfig(backend=TextBackend.OMLX)
    assert cfg.resolved_endpoint() == _TEXT_DEFAULTS[TextBackend.OMLX]["endpoint"]
    assert cfg.resolved_model() == _TEXT_DEFAULTS[TextBackend.OMLX]["model"]
    assert "localhost:8080" in cfg.resolved_endpoint()


def test_text_config_ollama_defaults() -> None:
    cfg = TextConfig(backend=TextBackend.OLLAMA)
    assert "localhost:11434" in cfg.resolved_endpoint()
    assert cfg.resolved_model() == "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0"


def test_text_config_openai_defaults() -> None:
    cfg = TextConfig(backend=TextBackend.OPENAI)
    assert "api.openai.com" in cfg.resolved_endpoint()
    assert cfg.resolved_model() == "gpt-4o-mini"


def test_text_config_endpoint_override() -> None:
    cfg = TextConfig(backend=TextBackend.OMLX, endpoint="http://myserver:9999/v1")
    assert cfg.resolved_endpoint() == "http://myserver:9999/v1"


def test_text_config_model_override() -> None:
    cfg = TextConfig(backend=TextBackend.OLLAMA, model="llama3.2:3b")
    assert cfg.resolved_model() == "llama3.2:3b"


def test_text_config_max_tokens_default() -> None:
    assert TextConfig().max_tokens == 2048


def test_text_config_suppress_thinking_default() -> None:
    assert TextConfig().suppress_thinking is True


# ---------------------------------------------------------------------------
# ImageConfig defaults per backend
# ---------------------------------------------------------------------------


def test_image_config_mflux_serve_defaults() -> None:
    cfg = ImageConfig(backend=ImageBackend.MFLUX_SERVE)
    assert "localhost:8090" in cfg.resolved_server_url()
    assert cfg.resolved_model() == _IMAGE_DEFAULTS[ImageBackend.MFLUX_SERVE]["model"]


def test_image_config_mflux_local_defaults() -> None:
    cfg = ImageConfig(backend=ImageBackend.MFLUX_LOCAL)
    assert cfg.resolved_model() == "mlx-community/flux2-klein-4b-4bit"
    assert cfg.resolved_server_url() == ""


def test_image_config_openai_defaults() -> None:
    cfg = ImageConfig(backend=ImageBackend.OPENAI)
    assert cfg.resolved_model() == "dall-e-3"
    assert cfg.resolved_server_url() == ""


def test_image_config_server_url_override() -> None:
    cfg = ImageConfig(backend=ImageBackend.MFLUX_SERVE, server_url="http://gpu-box:8090")
    assert cfg.resolved_server_url() == "http://gpu-box:8090"


def test_image_config_model_override() -> None:
    cfg = ImageConfig(backend=ImageBackend.OPENAI, model="dall-e-2")
    assert cfg.resolved_model() == "dall-e-2"


def test_image_config_steps_default() -> None:
    assert ImageConfig().steps == 4


# ---------------------------------------------------------------------------
# text_config_from_env — clean environment
# ---------------------------------------------------------------------------


def test_text_config_from_env_defaults(clean_synth: None) -> None:
    cfg = text_config_from_env()
    assert cfg.backend is TextBackend.OMLX
    assert cfg.endpoint == ""
    assert cfg.api_key == ""
    assert cfg.model == ""


def test_text_config_from_env_synth_backend(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("SYNTH_BACKEND", "ollama")
    cfg = text_config_from_env()
    assert cfg.backend is TextBackend.OLLAMA


def test_text_config_from_env_synth_backend_openai(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("SYNTH_BACKEND", "openai")
    cfg = text_config_from_env()
    assert cfg.backend is TextBackend.OPENAI


def test_text_config_from_env_invalid_backend_fallback(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("SYNTH_BACKEND", "not-a-backend")
    cfg = text_config_from_env()
    assert cfg.backend is TextBackend.OMLX


def test_text_config_from_env_synth_endpoint(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("SYNTH_ENDPOINT", "http://synth-server/v1")
    cfg = text_config_from_env()
    assert cfg.endpoint == "http://synth-server/v1"


def test_text_config_from_env_synth_model(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("SYNTH_MODEL", "my-custom-llm")
    cfg = text_config_from_env()
    assert cfg.model == "my-custom-llm"


def test_text_config_from_env_synth_api_key(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("SYNTH_API_KEY", "sk-test-123")
    cfg = text_config_from_env()
    assert cfg.api_key == "sk-test-123"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# text_config_from_env — legacy VLLM_* aliases
# ---------------------------------------------------------------------------


def test_text_config_from_env_legacy_endpoint(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("VLLM_ENDPOINT_URL", "http://legacy:8080/v1")
    cfg = text_config_from_env()
    assert cfg.endpoint == "http://legacy:8080/v1"


def test_text_config_from_env_legacy_model(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("VLLM_MODEL", "Qwen3-legacy")
    cfg = text_config_from_env()
    assert cfg.model == "Qwen3-legacy"


def test_text_config_from_env_legacy_api_key(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "legacy-token")
    cfg = text_config_from_env()
    assert cfg.api_key == "legacy-token"  # pragma: allowlist secret


def test_text_config_from_env_synth_takes_priority_over_legacy(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("SYNTH_ENDPOINT", "http://new/v1")
    monkeypatch.setenv("VLLM_ENDPOINT_URL", "http://old/v1")
    cfg = text_config_from_env()
    assert cfg.endpoint == "http://new/v1"


def test_text_config_from_env_synth_model_takes_priority_over_legacy(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("SYNTH_MODEL", "new-model")
    monkeypatch.setenv("VLLM_MODEL", "old-model")
    cfg = text_config_from_env()
    assert cfg.model == "new-model"


def test_text_config_from_env_openai_api_key_fallback(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback-key")
    cfg = text_config_from_env()
    assert cfg.api_key == "openai-fallback-key"  # pragma: allowlist secret


def test_text_config_from_env_synth_api_key_beats_openai(
    monkeypatch: pytest.MonkeyPatch, clean_synth: None
) -> None:
    monkeypatch.setenv("SYNTH_API_KEY", "synth-wins")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-loses")
    cfg = text_config_from_env()
    assert cfg.api_key == "synth-wins"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# image_config_from_env — clean environment
# ---------------------------------------------------------------------------


def test_image_config_from_env_defaults(clean_image: None) -> None:
    cfg = image_config_from_env()
    assert cfg.backend is ImageBackend.MFLUX_SERVE
    assert cfg.server_url == ""
    assert cfg.api_key == ""
    assert cfg.model == ""
    assert cfg.steps == 4


def test_image_config_from_env_backend_local(
    monkeypatch: pytest.MonkeyPatch, clean_image: None
) -> None:
    monkeypatch.setenv("IMAGE_BACKEND", "mflux-local")
    cfg = image_config_from_env()
    assert cfg.backend is ImageBackend.MFLUX_LOCAL


def test_image_config_from_env_backend_openai(
    monkeypatch: pytest.MonkeyPatch, clean_image: None
) -> None:
    monkeypatch.setenv("IMAGE_BACKEND", "openai")
    cfg = image_config_from_env()
    assert cfg.backend is ImageBackend.OPENAI


def test_image_config_from_env_invalid_backend_fallback(
    monkeypatch: pytest.MonkeyPatch, clean_image: None
) -> None:
    monkeypatch.setenv("IMAGE_BACKEND", "diffusers")
    cfg = image_config_from_env()
    assert cfg.backend is ImageBackend.MFLUX_SERVE


def test_image_config_from_env_endpoint(monkeypatch: pytest.MonkeyPatch, clean_image: None) -> None:
    monkeypatch.setenv("IMAGE_ENDPOINT", "http://gpu-box:8090")
    cfg = image_config_from_env()
    assert cfg.server_url == "http://gpu-box:8090"


def test_image_config_from_env_model(monkeypatch: pytest.MonkeyPatch, clean_image: None) -> None:
    monkeypatch.setenv("IMAGE_MODEL", "dall-e-2")
    cfg = image_config_from_env()
    assert cfg.model == "dall-e-2"


def test_image_config_from_env_steps(monkeypatch: pytest.MonkeyPatch, clean_image: None) -> None:
    monkeypatch.setenv("IMAGE_STEPS", "8")
    cfg = image_config_from_env()
    assert cfg.steps == 8


def test_image_config_from_env_api_key(monkeypatch: pytest.MonkeyPatch, clean_image: None) -> None:
    monkeypatch.setenv("IMAGE_API_KEY", "img-key-abc")
    cfg = image_config_from_env()
    assert cfg.api_key == "img-key-abc"  # pragma: allowlist secret


def test_image_config_from_env_openai_key_fallback(
    monkeypatch: pytest.MonkeyPatch, clean_image: None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    cfg = image_config_from_env()
    assert cfg.api_key == "sk-openai-fallback"  # pragma: allowlist secret


def test_image_config_from_env_image_api_key_beats_openai(
    monkeypatch: pytest.MonkeyPatch, clean_image: None
) -> None:
    monkeypatch.setenv("IMAGE_API_KEY", "img-wins")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-loses")
    cfg = image_config_from_env()
    assert cfg.api_key == "img-wins"  # pragma: allowlist secret


def test_image_config_from_env_legacy_model(
    monkeypatch: pytest.MonkeyPatch, clean_image: None
) -> None:
    monkeypatch.setenv("GUTENKG_IMAGE_MODEL", "flux2-klein-4b")
    cfg = image_config_from_env()
    assert cfg.model == "flux2-klein-4b"


def test_image_config_from_env_image_model_beats_legacy(
    monkeypatch: pytest.MonkeyPatch, clean_image: None
) -> None:
    monkeypatch.setenv("IMAGE_MODEL", "dall-e-3")
    monkeypatch.setenv("GUTENKG_IMAGE_MODEL", "old-flux-model")
    cfg = image_config_from_env()
    assert cfg.model == "dall-e-3"
