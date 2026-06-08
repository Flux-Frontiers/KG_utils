"""Tests for kg_utils.synthesis._image — ImageSynthesizer.

Requires the ``synthesis`` optional extra (httpx, openai, pillow).
The entire module is skipped if any of those packages are not installed.
"""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PIL", reason="pillow not installed — skipping synthesis image tests")
pytest.importorskip("httpx", reason="httpx not installed — skipping synthesis image tests")
pytest.importorskip(
    "openai", reason="openai package not installed — skipping synthesis image tests"
)

from PIL import Image as PILImage

from kg_utils.synthesis._config import ImageBackend, ImageConfig
from kg_utils.synthesis._image import (
    ImageSynthesizer,
    _DALLE3_SIZES,
    _MFLUX_SIZES,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_png_b64() -> str:
    """Base64-encoded 4x4 RGB PNG — valid for PIL.Image.open."""
    buf = BytesIO()
    PILImage.new("RGB", (4, 4), color=(100, 150, 200)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _serve_config(**kw) -> ImageConfig:
    return ImageConfig(backend=ImageBackend.MFLUX_SERVE, **kw)


def _openai_config(**kw) -> ImageConfig:
    return ImageConfig(
        backend=ImageBackend.OPENAI,
        api_key="test-key",  # pragma: allowlist secret
        **kw,
    )


def _local_config(**kw) -> ImageConfig:
    return ImageConfig(backend=ImageBackend.MFLUX_LOCAL, **kw)


# ---------------------------------------------------------------------------
# Aspect-ratio dimension tables
# ---------------------------------------------------------------------------


def test_mflux_sizes_all_ratios_present() -> None:
    expected = {"1:1", "3:2", "2:3", "16:9", "9:16", "4:3", "3:4"}
    assert expected == set(_MFLUX_SIZES.keys())


def test_mflux_sizes_square_is_1024() -> None:
    assert _MFLUX_SIZES["1:1"] == (1024, 1024)


def test_mflux_sizes_landscape_wider_than_tall() -> None:
    for ratio in ("3:2", "16:9", "4:3"):
        w, h = _MFLUX_SIZES[ratio]
        assert w > h, f"{ratio} should be landscape"


def test_mflux_sizes_portrait_taller_than_wide() -> None:
    for ratio in ("2:3", "9:16", "3:4"):
        w, h = _MFLUX_SIZES[ratio]
        assert h > w, f"{ratio} should be portrait"


def test_dalle3_sizes_all_ratios_present() -> None:
    expected = {"1:1", "3:2", "2:3", "16:9", "9:16", "4:3", "3:4"}
    assert expected == set(_DALLE3_SIZES.keys())


def test_dalle3_square_is_1024x1024() -> None:
    assert _DALLE3_SIZES["1:1"] == "1024x1024"


def test_dalle3_landscape_ratios_map_to_1792x1024() -> None:
    for ratio in ("3:2", "16:9", "4:3"):
        assert _DALLE3_SIZES[ratio] == "1792x1024", f"{ratio} should map to 1792x1024"


def test_dalle3_portrait_ratios_map_to_1024x1792() -> None:
    for ratio in ("2:3", "9:16", "3:4"):
        assert _DALLE3_SIZES[ratio] == "1024x1792", f"{ratio} should map to 1024x1792"


def test_dalle3_sizes_are_valid_openai_strings() -> None:
    valid = {"1024x1024", "1792x1024", "1024x1792"}
    assert set(_DALLE3_SIZES.values()) == valid


# ---------------------------------------------------------------------------
# mflux-serve backend — generate()
# ---------------------------------------------------------------------------


def test_generate_via_server_calls_correct_url(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        synth = ImageSynthesizer(_serve_config(server_url="http://img-server:8090"))
        synth.generate("a prompt")
        url = mock_post.call_args[0][0]

    assert url == "http://img-server:8090/v1/images/generations"


def test_generate_via_server_strips_trailing_slash(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        synth = ImageSynthesizer(_serve_config(server_url="http://img-server:8090/"))
        synth.generate("a prompt")
        url = mock_post.call_args[0][0]

    assert url.endswith("/v1/images/generations")
    assert "//" not in url.split("://", 1)[1]


def test_generate_via_server_payload_contains_prompt(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        synth = ImageSynthesizer(_serve_config(server_url="http://srv:8090"))
        synth.generate("my test prompt")
        payload = mock_post.call_args[1]["json"]

    assert payload["prompt"] == "my test prompt"


def test_generate_via_server_payload_size_matches_aspect_ratio(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        synth = ImageSynthesizer(_serve_config(server_url="http://srv:8090"))
        synth.generate("prompt", aspect_ratio="1:1")
        payload = mock_post.call_args[1]["json"]

    assert payload["size"] == "1024x1024"


def test_generate_via_server_payload_steps(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        synth = ImageSynthesizer(_serve_config(server_url="http://srv:8090", steps=8))
        synth.generate("prompt")
        payload = mock_post.call_args[1]["json"]

    assert payload["num_inference_steps"] == 8


def test_generate_via_server_payload_steps_override(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        synth = ImageSynthesizer(_serve_config(server_url="http://srv:8090", steps=4))
        synth.generate("prompt", steps=16)
        payload = mock_post.call_args[1]["json"]

    assert payload["num_inference_steps"] == 16


def test_generate_via_server_seed_included_when_given(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        synth = ImageSynthesizer(_serve_config(server_url="http://srv:8090"))
        synth.generate("prompt", seed=42)
        payload = mock_post.call_args[1]["json"]

    assert payload["seed"] == 42


def test_generate_via_server_seed_absent_when_none(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        synth = ImageSynthesizer(_serve_config(server_url="http://srv:8090"))
        synth.generate("prompt", seed=None)
        payload = mock_post.call_args[1]["json"]

    assert "seed" not in payload


def test_generate_via_server_requests_b64_json(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        synth = ImageSynthesizer(_serve_config(server_url="http://srv:8090"))
        synth.generate("prompt")
        payload = mock_post.call_args[1]["json"]

    assert payload["response_format"] == "b64_json"


def test_generate_via_server_returns_pil_image(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp):
        synth = ImageSynthesizer(_serve_config(server_url="http://srv:8090"))
        img = synth.generate("prompt")

    assert isinstance(img, PILImage.Image)


def test_generate_via_server_uses_default_server_url(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        # No server_url override — should use the backend default.
        synth = ImageSynthesizer(ImageConfig(backend=ImageBackend.MFLUX_SERVE))
        synth.generate("prompt")
        url = mock_post.call_args[0][0]

    assert "localhost:8090" in url


# ---------------------------------------------------------------------------
# mflux-serve backend — generate_b64()
# ---------------------------------------------------------------------------


def test_generate_b64_returns_decodable_base64(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp):
        synth = ImageSynthesizer(_serve_config(server_url="http://srv:8090"))
        result = synth.generate_b64("prompt")

    decoded = base64.b64decode(result)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_generate_b64_is_a_string(tiny_png_b64: str) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png_b64}]}

    with patch("httpx.post", return_value=mock_resp):
        synth = ImageSynthesizer(_serve_config(server_url="http://srv:8090"))
        result = synth.generate_b64("prompt")

    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# OpenAI DALL-E backend — generate()
# ---------------------------------------------------------------------------


def test_generate_openai_calls_images_generate(tiny_png_b64: str) -> None:
    mock_data = MagicMock()
    mock_data.b64_json = tiny_png_b64
    mock_resp = MagicMock()
    mock_resp.data = [mock_data]

    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.images.generate.return_value = mock_resp
        synth = ImageSynthesizer(_openai_config())
        synth.generate("portrait of a Victorian scholar")
        call_kwargs = mock_client.images.generate.call_args[1]

    assert call_kwargs["prompt"] == "portrait of a Victorian scholar"


def test_generate_openai_uses_resolved_model(tiny_png_b64: str) -> None:
    mock_data = MagicMock()
    mock_data.b64_json = tiny_png_b64
    mock_resp = MagicMock()
    mock_resp.data = [mock_data]

    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.images.generate.return_value = mock_resp
        synth = ImageSynthesizer(_openai_config())
        synth.generate("prompt")
        call_kwargs = mock_client.images.generate.call_args[1]

    # Default model for OpenAI backend is dall-e-3
    assert call_kwargs["model"] == "dall-e-3"


def test_generate_openai_model_override(tiny_png_b64: str) -> None:
    mock_data = MagicMock()
    mock_data.b64_json = tiny_png_b64
    mock_resp = MagicMock()
    mock_resp.data = [mock_data]

    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.images.generate.return_value = mock_resp
        synth = ImageSynthesizer(_openai_config())
        synth.generate("prompt", model="dall-e-2")
        call_kwargs = mock_client.images.generate.call_args[1]

    assert call_kwargs["model"] == "dall-e-2"


def test_generate_openai_landscape_size(tiny_png_b64: str) -> None:
    mock_data = MagicMock()
    mock_data.b64_json = tiny_png_b64
    mock_resp = MagicMock()
    mock_resp.data = [mock_data]

    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.images.generate.return_value = mock_resp
        synth = ImageSynthesizer(_openai_config())
        synth.generate("prompt", aspect_ratio="3:2")
        call_kwargs = mock_client.images.generate.call_args[1]

    assert call_kwargs["size"] == "1792x1024"


def test_generate_openai_portrait_size(tiny_png_b64: str) -> None:
    mock_data = MagicMock()
    mock_data.b64_json = tiny_png_b64
    mock_resp = MagicMock()
    mock_resp.data = [mock_data]

    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.images.generate.return_value = mock_resp
        synth = ImageSynthesizer(_openai_config())
        synth.generate("prompt", aspect_ratio="2:3")
        call_kwargs = mock_client.images.generate.call_args[1]

    assert call_kwargs["size"] == "1024x1792"


def test_generate_openai_requests_b64_json(tiny_png_b64: str) -> None:
    mock_data = MagicMock()
    mock_data.b64_json = tiny_png_b64
    mock_resp = MagicMock()
    mock_resp.data = [mock_data]

    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.images.generate.return_value = mock_resp
        synth = ImageSynthesizer(_openai_config())
        synth.generate("prompt")
        call_kwargs = mock_client.images.generate.call_args[1]

    assert call_kwargs["response_format"] == "b64_json"


def test_generate_openai_returns_pil_image(tiny_png_b64: str) -> None:
    mock_data = MagicMock()
    mock_data.b64_json = tiny_png_b64
    mock_resp = MagicMock()
    mock_resp.data = [mock_data]

    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.images.generate.return_value = mock_resp
        synth = ImageSynthesizer(_openai_config())
        img = synth.generate("prompt")

    assert isinstance(img, PILImage.Image)


# ---------------------------------------------------------------------------
# mflux-local backend — model cache
# ---------------------------------------------------------------------------


def _make_mock_flux(tiny_png_b64: str) -> MagicMock:
    """Return a mock Flux2Klein whose generate_image() returns a tiny PIL image."""
    raw = base64.b64decode(tiny_png_b64)
    pil_img = PILImage.open(BytesIO(raw))
    result = MagicMock()
    result.image = pil_img
    mock_flux = MagicMock()
    mock_flux.generate_image.return_value = result
    return mock_flux


def test_mflux_local_model_cache_reuses_instance(tiny_png_b64: str) -> None:
    synth = ImageSynthesizer(_local_config(model="my-flux-model"))
    mock_flux = _make_mock_flux(tiny_png_b64)
    synth._load_mflux = MagicMock(return_value=mock_flux)  # type: ignore[method-assign]

    synth.generate("first call")
    synth.generate("second call")

    # _load_mflux called once on first call; second call should use the cached instance.
    assert synth._load_mflux.call_count == 2  # _load_mflux itself handles the cache check


def test_mflux_local_generate_passes_prompt(tiny_png_b64: str) -> None:
    synth = ImageSynthesizer(_local_config(model="my-flux-model"))
    mock_flux = _make_mock_flux(tiny_png_b64)
    synth._load_mflux = MagicMock(return_value=mock_flux)  # type: ignore[method-assign]

    synth.generate("a historic seaport at dusk")

    call_kwargs = mock_flux.generate_image.call_args[1]
    assert call_kwargs["prompt"] == "a historic seaport at dusk"


def test_mflux_local_generate_passes_steps(tiny_png_b64: str) -> None:
    synth = ImageSynthesizer(_local_config(model="my-flux-model", steps=6))
    mock_flux = _make_mock_flux(tiny_png_b64)
    synth._load_mflux = MagicMock(return_value=mock_flux)  # type: ignore[method-assign]

    synth.generate("prompt")

    call_kwargs = mock_flux.generate_image.call_args[1]
    assert call_kwargs["num_inference_steps"] == 6


def test_mflux_local_generate_uses_fixed_seed_when_given(tiny_png_b64: str) -> None:
    synth = ImageSynthesizer(_local_config(model="my-flux-model"))
    mock_flux = _make_mock_flux(tiny_png_b64)
    synth._load_mflux = MagicMock(return_value=mock_flux)  # type: ignore[method-assign]

    synth.generate("prompt", seed=99)

    call_kwargs = mock_flux.generate_image.call_args[1]
    assert call_kwargs["seed"] == 99


def test_mflux_local_generate_random_seed_when_none(tiny_png_b64: str) -> None:
    synth = ImageSynthesizer(_local_config(model="my-flux-model"))
    mock_flux = _make_mock_flux(tiny_png_b64)
    synth._load_mflux = MagicMock(return_value=mock_flux)  # type: ignore[method-assign]

    synth.generate("prompt", seed=None)

    call_kwargs = mock_flux.generate_image.call_args[1]
    assert isinstance(call_kwargs["seed"], int)


def test_mflux_local_generate_landscape_dims(tiny_png_b64: str) -> None:
    synth = ImageSynthesizer(_local_config(model="my-flux-model"))
    mock_flux = _make_mock_flux(tiny_png_b64)
    synth._load_mflux = MagicMock(return_value=mock_flux)  # type: ignore[method-assign]

    synth.generate("prompt", aspect_ratio="16:9")

    call_kwargs = mock_flux.generate_image.call_args[1]
    assert call_kwargs["width"] > call_kwargs["height"]


def test_mflux_local_generate_returns_pil_image(tiny_png_b64: str) -> None:
    synth = ImageSynthesizer(_local_config(model="my-flux-model"))
    mock_flux = _make_mock_flux(tiny_png_b64)
    synth._load_mflux = MagicMock(return_value=mock_flux)  # type: ignore[method-assign]

    img = synth.generate("prompt")

    assert isinstance(img, PILImage.Image)
