# © 2026 Eric G. Suchanek, PhD — Flux-Frontiers · SPDX-License-Identifier: Elastic-2.0
"""RunPod worker client utilities for chat and handler front-ends.

This module centralizes payload construction and response/error decoding for
``/runsync`` worker calls used by Streamlit clients.
"""

from __future__ import annotations

import json
from typing import Any, cast

import httpx


class WorkerError(Exception):
    """Raised when a worker response contains a structured application-level error."""


def _format_error_data(error_data: object) -> str:
    if isinstance(error_data, str):
        try:
            decoded = json.loads(error_data)
        except (ValueError, TypeError):
            return error_data
        if isinstance(decoded, dict):
            err_type = decoded.get("error_type", "Unknown")
            err_msg = decoded.get("error_message", str(decoded))
            return f"{err_type}: {err_msg}"
        return str(decoded)

    if isinstance(error_data, dict):
        d = cast("dict[str, Any]", error_data)
        err_type = d.get("error_type", "Unknown")
        err_msg = d.get("error_message", str(error_data))
        return f"{err_type}: {err_msg}"

    return str(error_data)


def extract_worker_error(data: object) -> str | None:
    """Extract a readable worker error from a raw RunPod response payload."""
    if not isinstance(data, dict):
        return str(data)

    d = cast("dict[str, Any]", data)
    if d.get("status") == "FAILED" or "error_type" in d:
        return _format_error_data(d.get("error", d))

    out = d.get("output")
    if isinstance(out, dict) and isinstance(out.get("error"), str):
        return out["error"]

    return None


def decode_worker_response(data: object) -> dict:
    """Decode a worker response payload and raise WorkerError on application errors."""
    error = extract_worker_error(data)
    if error:
        raise WorkerError(error)

    if not isinstance(data, dict):
        raise WorkerError(f"unexpected worker response type: {type(data).__name__}")

    d2 = cast("dict[str, Any]", data)
    out = d2.get("output", d2)
    if not isinstance(out, dict):
        raise WorkerError(f"unexpected worker output type: {type(out).__name__}")
    return out


class WorkerClient:
    """Small client for RunPod ``/runsync`` worker endpoints."""

    def __init__(self, base_url: str, secret: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._secret = secret

    def _post(self, payload: dict, timeout: httpx.Timeout) -> dict:
        resp = httpx.post(f"{self._base_url}/runsync", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def list_models(self, backend: str = "") -> tuple[list[str], str]:
        payload: dict = {"input": {"op": "models"}}
        if backend:
            payload["input"]["backend"] = backend
        if self._secret:
            payload["input"]["secret"] = self._secret

        try:
            data = self._post(
                payload,
                timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
            )
            out = data.get("output", {}) if isinstance(data, dict) else {}
            if not isinstance(out, dict):
                return [], ""
            return out.get("models", []), out.get("default", "")
        except Exception:  # noqa: BLE001
            return [], ""

    def rewrite(
        self,
        text: str,
        backend: str = "",
        model: str = "",
    ) -> tuple[str, str | None]:
        payload: dict = {"input": {"op": "rewrite", "text": text}}
        if backend:
            payload["input"]["backend"] = backend
        if model:
            payload["input"]["model"] = model
        if self._secret:
            payload["input"]["secret"] = self._secret

        try:
            data = self._post(
                payload,
                timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
            )
            err = extract_worker_error(data)
            if err:
                return text, err
            out = data.get("output", {}) if isinstance(data, dict) else {}
            if not isinstance(out, dict):
                return text, "unexpected worker output"
            return out.get("prompt", text), out.get("error")
        except Exception as exc:  # noqa: BLE001
            return text, str(exc)

    def imagine(
        self,
        prompt: str,
        *,
        image_backend: str = "",
        aspect_ratio: str = "3:2",
        steps: int | None = None,
        size: str | None = None,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        inp: dict[str, Any] = {"op": "imagine", "prompt": prompt, "aspect_ratio": aspect_ratio}
        if image_backend:
            inp["image_backend"] = image_backend
        if steps is not None:
            inp["steps"] = steps
        if size:
            inp["size"] = size
        if self._secret:
            inp["secret"] = self._secret
        payload: dict[str, Any] = {"input": inp}

        try:
            data = self._post(
                payload,
                timeout=httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0),
            )
            err = extract_worker_error(data)
            if err:
                return None, None, None, err

            out = data.get("output", {}) if isinstance(data, dict) else {}
            if not isinstance(out, dict):
                return None, None, None, "unexpected worker output"
            if "error" in out:
                return None, None, None, str(out["error"])
            return out.get("image_b64"), out.get("image_model"), out.get("image_backend"), None
        except Exception as exc:  # noqa: BLE001
            return None, None, None, str(exc)

    def query(
        self,
        query: str,
        *,
        corpus: str = "all",
        k: int = 8,
        min_score: float = 0.0,
        semantic_floor: float = 0.0,
        synthesize: bool = False,
        model: str = "",
        backend: str = "",
    ) -> dict:
        payload: dict = {
            "input": {
                "query": query,
                "corpus": corpus,
                "k": k,
                "min_score": min_score,
                "semantic_floor": semantic_floor,
                "synthesize": synthesize,
            }
        }
        if model:
            payload["input"]["model"] = model
        if backend:
            payload["input"]["backend"] = backend
        if self._secret:
            payload["input"]["secret"] = self._secret

        data = self._post(
            payload,
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0),
        )
        return decode_worker_response(data)
