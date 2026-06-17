"""Tests for kg_utils.worker.ops."""

from __future__ import annotations

from unittest.mock import MagicMock

from kg_utils.synthesis._config import ImageBackend, ImageConfig, TextConfig
from kg_utils.synthesis._image import ImageSynthesizer
from kg_utils.synthesis._text import TextSynthesizer
from kg_utils.worker.ops import handle_aux_ops


def _text_synth() -> TextSynthesizer:
    synth = TextSynthesizer(TextConfig(model="text-default"))
    synth.list_models = MagicMock(return_value=["m1", "m2"])  # type: ignore[method-assign]
    synth.rewrite_for_image = MagicMock(return_value=("rewritten prompt", None))  # type: ignore[method-assign]
    return synth


def _image_synth() -> ImageSynthesizer:
    synth = ImageSynthesizer(ImageConfig(backend=ImageBackend.MFLUX_SERVE, model="flux2-klein"))
    synth.generate_b64 = MagicMock(return_value="abc123")  # type: ignore[method-assign]
    return synth


def test_handle_aux_ops_returns_none_for_unknown_op() -> None:
    result = handle_aux_ops({"op": "other"}, lambda _: _text_synth(), lambda _: _image_synth())
    assert result is None


def test_handle_aux_ops_models() -> None:
    text = _text_synth()
    result = handle_aux_ops(
        {"op": "models", "backend": "omlx"},
        lambda _: text,
        lambda _: _image_synth(),
    )
    assert result == {"models": ["m1", "m2"], "default": "text-default"}


def test_handle_aux_ops_rewrite_success() -> None:
    text = _text_synth()
    result = handle_aux_ops(
        {"op": "rewrite", "text": " original text ", "backend": "omlx", "model": "small"},
        lambda _: text,
        lambda _: _image_synth(),
    )
    assert result == {"prompt": "rewritten prompt", "error": None}


def test_handle_aux_ops_rewrite_requires_text() -> None:
    result = handle_aux_ops(
        {"op": "rewrite", "text": "   "},
        lambda _: _text_synth(),
        lambda _: _image_synth(),
    )
    assert result == {"error": "rewrite requires a non-empty 'text'"}


def test_handle_aux_ops_imagine_success() -> None:
    image = _image_synth()
    result = handle_aux_ops(
        {
            "op": "imagine",
            "prompt": "scene",
            "image_backend": "openai",
            "aspect_ratio": "16:9",
            "seed": "7",
            "steps": "12",
        },
        lambda _: _text_synth(),
        lambda _: image,
    )
    assert result == {
        "image_b64": "abc123",
        "prompt": "scene",
        "aspect_ratio": "16:9",
        "image_model": "flux2-klein",
        "image_backend": "mflux-serve",
    }
    image.generate_b64.assert_called_once_with("scene", aspect_ratio="16:9", seed=7, steps=12)


def test_handle_aux_ops_imagine_requires_prompt() -> None:
    result = handle_aux_ops(
        {"op": "imagine", "prompt": "   "},
        lambda _: _text_synth(),
        lambda _: _image_synth(),
    )
    assert result == {"error": "imagine requires a non-empty 'prompt'"}


def test_handle_aux_ops_imagine_failure_returns_error() -> None:
    image = _image_synth()
    image.generate_b64 = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    result = handle_aux_ops(
        {"op": "imagine", "prompt": "scene"},
        lambda _: _text_synth(),
        lambda _: image,
    )
    assert result == {"error": "image generation failed: boom"}
