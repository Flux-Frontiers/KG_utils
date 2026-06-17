# © 2026 Eric G. Suchanek, PhD — Flux-Frontiers · SPDX-License-Identifier: Elastic-2.0
"""ImageSynthesizer — unified image generation across mflux-local, mflux-serve, and DALL-E."""

from __future__ import annotations

import base64
import importlib
import random
from io import BytesIO
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage  # type: ignore[import-unresolved]

from kg_utils.synthesis._config import ImageBackend, ImageConfig

# ---------------------------------------------------------------------------
# Aspect-ratio → pixel dimension tables
# ---------------------------------------------------------------------------

# Used by mflux backends (arbitrary resolution).
_MFLUX_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "3:2": (1536, 1024),
    "2:3": (1024, 1536),
    "16:9": (1536, 864),
    "9:16": (864, 1536),
    "4:3": (1365, 1024),
    "3:4": (1024, 1365),
}

# DALL-E 3 only supports three sizes; map aspect ratios to the nearest valid option.
_DALLE3_SIZES: dict[str, str] = {
    "1:1": "1024x1024",
    "3:2": "1792x1024",
    "2:3": "1024x1792",
    "16:9": "1792x1024",
    "9:16": "1024x1792",
    "4:3": "1792x1024",
    "3:4": "1024x1792",
}

# gpt-image-1 supports portrait/landscape 1024×1536 instead of 1792-wide.
_GPT_IMAGE_SIZES: dict[str, str] = {
    "1:1": "1024x1024",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
    "16:9": "1536x1024",
    "9:16": "1024x1536",
    "4:3": "1536x1024",
    "3:4": "1024x1536",
}


# ---------------------------------------------------------------------------
# Size parsing
# ---------------------------------------------------------------------------


def _parse_size(size: str | None) -> tuple[int, int] | None:
    """Parse an explicit ``"WIDTHxHEIGHT"`` string into an ``(width, height)`` pair.

    :param size: Size string such as ``"768x512"`` (case-insensitive ``x``), or None.
    :returns: ``(width, height)`` when *size* parses to two positive ints, else None.
    """
    if not size:
        return None
    try:
        w_str, h_str = size.lower().split("x", 1)
        width, height = int(w_str), int(h_str)
    except (ValueError, AttributeError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class ImageSynthesizer:
    """Unified image generation across mflux-local, mflux-serve, and OpenAI DALL-E.

    All ``generate*`` methods return a PIL ``Image`` — callers that need a base64
    string (e.g. HTTP wire responses) should use ``generate_b64()``.  A per-instance
    mflux model cache avoids reloading across calls to the same synthesizer.

    :param config: Backend configuration produced by ``image_config_from_env()`` or
                   built directly as an ``ImageConfig`` dataclass.
    """

    def __init__(self, config: ImageConfig) -> None:
        self._cfg = config
        self._mflux_model: Any = None
        self._mflux_model_name: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_mflux(self, model_name: str) -> Any:
        if self._mflux_model is not None and self._mflux_model_name == model_name:
            return self._mflux_model

        Flux2Klein = importlib.import_module(
            "mflux.models.flux2.variants.txt2img.flux2_klein"
        ).Flux2Klein

        self._mflux_model = Flux2Klein(model_path=model_name)
        self._mflux_model_name = model_name
        return self._mflux_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "3:2",
        seed: int | None = None,
        steps: int | None = None,
        model: str | None = None,
        size: str | None = None,
    ) -> PILImage:
        """Generate an image and return a PIL Image.

        :param prompt: Text description of the scene to generate.
        :param aspect_ratio: One of 1:1, 3:2, 2:3, 16:9, 9:16, 4:3, 3:4.
        :param seed: Random seed for reproducibility (random int if omitted).
        :param steps: Override inference steps (mflux backends only; ignored for OpenAI).
        :param model: Override the configured model for this single call.
        :param size: Explicit ``"WIDTHxHEIGHT"`` override (mflux backends only). When given
                     and valid, it takes priority over the *aspect_ratio* size table. OpenAI
                     backends ignore it because they accept only a fixed set of sizes.
        :returns: PIL Image.
        """
        cfg = self._cfg
        effective_steps = steps if steps is not None else cfg.steps
        effective_model = model or cfg.resolved_model()

        if cfg.backend == ImageBackend.MFLUX_LOCAL:
            return self._generate_local(
                prompt,
                model=effective_model,
                seed=seed,
                steps=effective_steps,
                aspect_ratio=aspect_ratio,
                size=size,
            )
        elif cfg.backend == ImageBackend.MFLUX_SERVE:
            return self._generate_via_server(
                prompt,
                server_url=cfg.resolved_server_url(),
                seed=seed,
                steps=effective_steps,
                aspect_ratio=aspect_ratio,
                size=size,
            )
        else:
            return self._generate_openai(
                prompt,
                model=effective_model,
                aspect_ratio=aspect_ratio,
            )

    def generate_b64(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "3:2",
        seed: int | None = None,
        steps: int | None = None,
        model: str | None = None,
        size: str | None = None,
    ) -> str:
        """Generate an image and return it as a base64-encoded PNG string.

        Convenience wrapper around ``generate()`` for HTTP wire responses.

        :param prompt: Text description of the scene to generate.
        :param aspect_ratio: One of 1:1, 3:2, 2:3, 16:9, 9:16, 4:3, 3:4.
        :param seed: Random seed for reproducibility.
        :param steps: Override inference steps (mflux backends only).
        :param model: Override the configured model for this single call.
        :param size: Explicit ``"WIDTHxHEIGHT"`` override (mflux backends only).
        :returns: Base64-encoded PNG string.
        """
        img = self.generate(
            prompt, aspect_ratio=aspect_ratio, seed=seed, steps=steps, model=model, size=size
        )
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _generate_local(
        self,
        prompt: str,
        *,
        model: str,
        seed: int | None,
        steps: int,
        aspect_ratio: str,
        size: str | None = None,
    ) -> PILImage:
        width, height = _parse_size(size) or _MFLUX_SIZES.get(aspect_ratio, _MFLUX_SIZES["3:2"])
        effective_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        flux = self._load_mflux(model)
        result = flux.generate_image(
            seed=effective_seed,
            prompt=prompt,
            width=width,
            height=height,
            guidance=1.0,
            num_inference_steps=steps,
            scheduler="flow_match_euler_discrete",
        )
        return result.image

    def _generate_via_server(
        self,
        prompt: str,
        *,
        server_url: str,
        seed: int | None,
        steps: int,
        aspect_ratio: str,
        size: str | None = None,
    ) -> PILImage:
        import httpx
        from PIL import Image  # type: ignore[import-unresolved]

        width, height = _parse_size(size) or _MFLUX_SIZES.get(aspect_ratio, _MFLUX_SIZES["3:2"])
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n": 1,
            "size": f"{width}x{height}",
            "num_inference_steps": steps,
            "response_format": "b64_json",
        }
        if seed is not None:
            payload["seed"] = seed

        resp = httpx.post(
            server_url.rstrip("/") + "/v1/images/generations",
            json=payload,
            timeout=httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0),
        )
        resp.raise_for_status()
        b64 = resp.json()["data"][0]["b64_json"]
        return Image.open(BytesIO(base64.b64decode(b64)))

    def _generate_openai(
        self,
        prompt: str,
        *,
        model: str,
        aspect_ratio: str,
    ) -> PILImage:
        import os

        from openai import OpenAI  # type: ignore[import-unresolved]
        from PIL import Image  # type: ignore[import-unresolved]

        cfg = self._cfg
        api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY", "")

        # gpt-image-1 uses different valid sizes and does not accept response_format.
        # dall-e-3 returns a URL by default; gpt-image-1 returns b64_json by default.
        if model.startswith("gpt-image"):
            size = _GPT_IMAGE_SIZES.get(aspect_ratio, _GPT_IMAGE_SIZES["3:2"])
        else:
            size = _DALLE3_SIZES.get(aspect_ratio, _DALLE3_SIZES["3:2"])

        client = OpenAI(api_key=api_key)
        resp = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size,  # type: ignore[arg-type]
        )
        item = resp.data[0] if resp.data else None
        if item and item.b64_json:
            return Image.open(BytesIO(base64.b64decode(item.b64_json)))
        if item and item.url:
            import httpx  # type: ignore[import-unresolved]

            raw = httpx.get(item.url, timeout=httpx.Timeout(30.0)).content
            return Image.open(BytesIO(raw))
        raise RuntimeError("OpenAI image response contained neither b64_json nor url")
