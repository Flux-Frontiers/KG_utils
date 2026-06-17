"""Tests for kg_utils.synthesis.factory."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from kg_utils.synthesis import (
    ImageConfig,
    ImageSynthesizer,
    TextConfig,
    TextSynthesizer,
)
from kg_utils.synthesis._config import ImageBackend, TextBackend
from kg_utils.synthesis.factory import (
    image_synth_for_backend,
    normalize_openai_base_url,
    text_synth_for_backend,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "SYNTH_ENDPOINT",
        "VLLM_ENDPOINT_URL",
        "SYNTH_API_KEY",
        "VLLM_API_KEY",
        "SYNTH_MODEL",
        "VLLM_MODEL",
        "OLLAMA_ENDPOINT",
        "OPENAI_API_KEY",
        "IMAGE_API_KEY",
        "IMAGE_ENDPOINT",
        "IMAGE_MODEL",
        "GUTENKG_IMAGE_MODEL",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_normalize_openai_base_url_empty() -> None:
    assert normalize_openai_base_url("") == ""


def test_normalize_openai_base_url_adds_v1() -> None:
    assert normalize_openai_base_url("http://localhost:8080") == "http://localhost:8080/v1"


def test_normalize_openai_base_url_keeps_existing_v1() -> None:
    assert normalize_openai_base_url("http://localhost:8080/v1") == "http://localhost:8080/v1"


def test_text_synth_for_backend_empty_returns_fallback() -> None:
    fallback = TextSynthesizer(TextConfig())
    assert text_synth_for_backend("", fallback) is fallback


def test_text_synth_for_backend_unknown_returns_fallback() -> None:
    fallback = TextSynthesizer(TextConfig())
    assert text_synth_for_backend("bad-backend", fallback) is fallback


def test_text_synth_for_backend_omlx_uses_synth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_ENDPOINT", "http://omlx-host:8080")
    monkeypatch.setenv("SYNTH_API_KEY", "synth-key")
    monkeypatch.setenv("SYNTH_MODEL", "Qwen3-test")

    fallback = TextSynthesizer(TextConfig())
    with patch("kg_utils.synthesis.factory.TextSynthesizer") as mock_cls:
        text_synth_for_backend("omlx", fallback)
    cfg = mock_cls.call_args.args[0]
    assert cfg.backend is TextBackend.OMLX
    assert cfg.endpoint == "http://omlx-host:8080/v1"
    assert cfg.api_key == "synth-key"  # pragma: allowlist secret
    assert cfg.model == "Qwen3-test"


def test_text_synth_for_backend_omlx_falls_back_to_vllm_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ENDPOINT_URL", "http://legacy:9000")
    monkeypatch.setenv("VLLM_API_KEY", "legacy-key")
    monkeypatch.setenv("VLLM_MODEL", "legacy-model")

    fallback = TextSynthesizer(TextConfig())
    with patch("kg_utils.synthesis.factory.TextSynthesizer") as mock_cls:
        text_synth_for_backend("omlx", fallback)
    cfg = mock_cls.call_args.args[0]
    assert cfg.endpoint == "http://legacy:9000/v1"
    assert cfg.api_key == "legacy-key"  # pragma: allowlist secret
    assert cfg.model == "legacy-model"


def test_text_synth_for_backend_ollama_uses_ollama_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_ENDPOINT", "http://localhost:11434/v1")

    fallback = TextSynthesizer(TextConfig())
    with patch("kg_utils.synthesis.factory.TextSynthesizer") as mock_cls:
        text_synth_for_backend("ollama", fallback)
    cfg = mock_cls.call_args.args[0]
    assert cfg.backend is TextBackend.OLLAMA
    assert cfg.endpoint == "http://localhost:11434/v1"


def test_text_synth_for_backend_openai_prefers_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("SYNTH_API_KEY", "synth-key")

    fallback = TextSynthesizer(TextConfig())
    with patch("kg_utils.synthesis.factory.TextSynthesizer") as mock_cls:
        text_synth_for_backend("openai", fallback)
    cfg = mock_cls.call_args.args[0]
    assert cfg.backend is TextBackend.OPENAI
    assert cfg.api_key == "openai-key"  # pragma: allowlist secret


def test_image_synth_for_backend_empty_returns_fallback() -> None:
    fallback = ImageSynthesizer(ImageConfig())
    assert image_synth_for_backend("", fallback) is fallback


def test_image_synth_for_backend_unknown_returns_fallback() -> None:
    fallback = ImageSynthesizer(ImageConfig())
    assert image_synth_for_backend("bad-backend", fallback) is fallback


def test_image_synth_for_backend_openai_prefers_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("IMAGE_API_KEY", "image-key")

    fallback = ImageSynthesizer(ImageConfig())
    with patch("kg_utils.synthesis.factory.ImageSynthesizer") as mock_cls:
        image_synth_for_backend("openai", fallback)
    cfg = mock_cls.call_args.args[0]
    assert cfg.backend is ImageBackend.OPENAI
    assert cfg.api_key == "openai-key"  # pragma: allowlist secret


def test_image_synth_for_backend_mflux_serve_uses_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_ENDPOINT", "http://gpu:8090")

    fallback = ImageSynthesizer(ImageConfig())
    with patch("kg_utils.synthesis.factory.ImageSynthesizer") as mock_cls:
        image_synth_for_backend("mflux-serve", fallback)
    cfg = mock_cls.call_args.args[0]
    assert cfg.backend is ImageBackend.MFLUX_SERVE
    assert cfg.server_url == "http://gpu:8090"


def test_image_synth_for_backend_mflux_local_prefers_image_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_MODEL", "flux-new")
    monkeypatch.setenv("GUTENKG_IMAGE_MODEL", "flux-old")

    fallback = ImageSynthesizer(ImageConfig())
    with patch("kg_utils.synthesis.factory.ImageSynthesizer") as mock_cls:
        image_synth_for_backend("mflux-local", fallback)
    cfg = mock_cls.call_args.args[0]
    assert cfg.backend is ImageBackend.MFLUX_LOCAL
    assert cfg.model == "flux-new"


def test_image_synth_for_backend_mflux_local_falls_back_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUTENKG_IMAGE_MODEL", "flux-legacy")

    fallback = ImageSynthesizer(ImageConfig())
    with patch("kg_utils.synthesis.factory.ImageSynthesizer") as mock_cls:
        image_synth_for_backend("mflux-local", fallback)
    cfg = mock_cls.call_args.args[0]
    assert cfg.model == "flux-legacy"
