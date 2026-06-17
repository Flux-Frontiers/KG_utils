"""Tests for kg_utils.worker.client.

Requires the ``synthesis`` optional extra (httpx package).
The module is skipped if httpx is not installed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("httpx", reason="httpx package not installed — skipping worker client tests")

from kg_utils.worker.client import (
    WorkerClient,
    WorkerError,
    decode_worker_response,
    extract_worker_error,
)


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_extract_worker_error_handles_failed_string_error() -> None:
    payload = {"status": "FAILED", "error": "boom"}
    assert extract_worker_error(payload) == "boom"


def test_extract_worker_error_handles_failed_json_string_error() -> None:
    payload = {
        "status": "FAILED",
        "error": '{"error_type":"ValidationError","error_message":"bad input"}',
    }
    assert extract_worker_error(payload) == "ValidationError: bad input"


def test_extract_worker_error_handles_top_level_error_type() -> None:
    payload = {"error_type": "HandlerError", "error_message": "explode"}
    assert extract_worker_error(payload) == "HandlerError: explode"


def test_extract_worker_error_handles_soft_output_error() -> None:
    payload = {"output": {"error": "worker soft failure"}}
    assert extract_worker_error(payload) == "worker soft failure"


def test_decode_worker_response_returns_output_dict() -> None:
    out = decode_worker_response({"output": {"hits": []}})
    assert out == {"hits": []}


def test_decode_worker_response_raises_worker_error() -> None:
    with pytest.raises(WorkerError, match="HandlerError: explode"):
        decode_worker_response({"error_type": "HandlerError", "error_message": "explode"})


def test_list_models_returns_data_on_success() -> None:
    client = WorkerClient("http://localhost:8000", secret="s")
    with patch("httpx.post", return_value=_Resp({"output": {"models": ["m1"], "default": "m1"}})):
        models, default = client.list_models(backend="omlx")
    assert models == ["m1"]
    assert default == "m1"


def test_list_models_returns_empty_on_exception() -> None:
    client = WorkerClient("http://localhost:8000")
    with patch("httpx.post", side_effect=RuntimeError("network down")):
        models, default = client.list_models()
    assert models == []
    assert default == ""


def test_query_builds_expected_payload_and_decodes() -> None:
    client = WorkerClient("http://localhost:8000", secret="sec")
    with patch("httpx.post", return_value=_Resp({"output": {"hits": [{"id": 1}]}})) as mock_post:
        out = client.query(
            "hello",
            corpus="diary",
            k=3,
            min_score=0.4,
            semantic_floor=0.2,
            synthesize=True,
            model="qwen",
            backend="omlx",
        )

    assert out == {"hits": [{"id": 1}]}
    kwargs = mock_post.call_args.kwargs
    payload = kwargs["json"]
    assert payload["input"]["query"] == "hello"
    assert payload["input"]["corpus"] == "diary"
    assert payload["input"]["k"] == 3
    assert payload["input"]["synthesize"] is True
    assert payload["input"]["model"] == "qwen"
    assert payload["input"]["backend"] == "omlx"
    assert payload["input"]["secret"] == "sec"  # pragma: allowlist secret


def test_query_raises_worker_error_for_failed_payload() -> None:
    client = WorkerClient("http://localhost:8000")
    failed = {"status": "FAILED", "error": "bad worker"}
    with patch("httpx.post", return_value=_Resp(failed)):
        with pytest.raises(WorkerError, match="bad worker"):
            client.query("x")


def test_rewrite_returns_prompt_and_none_error() -> None:
    client = WorkerClient("http://localhost:8000")
    with patch("httpx.post", return_value=_Resp({"output": {"prompt": "new prompt"}})):
        prompt, err = client.rewrite("old prompt")
    assert prompt == "new prompt"
    assert err is None


def test_rewrite_returns_original_text_and_error_on_failed_worker() -> None:
    client = WorkerClient("http://localhost:8000")
    with patch("httpx.post", return_value=_Resp({"status": "FAILED", "error": "rewrite failed"})):
        prompt, err = client.rewrite("old prompt")
    assert prompt == "old prompt"
    assert err == "rewrite failed"


def test_imagine_returns_image_tuple_on_success() -> None:
    client = WorkerClient("http://localhost:8000")
    payload = {
        "output": {
            "image_b64": "abc",
            "image_model": "gpt-image-1",
            "image_backend": "openai",
        }
    }
    with patch("httpx.post", return_value=_Resp(payload)):
        b64, model, backend, err = client.imagine("scene")
    assert b64 == "abc"
    assert model == "gpt-image-1"
    assert backend == "openai"
    assert err is None


def test_imagine_returns_error_tuple_on_soft_output_error() -> None:
    client = WorkerClient("http://localhost:8000")
    with patch("httpx.post", return_value=_Resp({"output": {"error": "gen failed"}})):
        b64, model, backend, err = client.imagine("scene")
    assert b64 is None
    assert model is None
    assert backend is None
    assert err == "gen failed"


def test_imagine_passes_optional_fields() -> None:
    client = WorkerClient("http://localhost:8000", secret="sec")
    with patch("httpx.post", return_value=_Resp({"output": {"image_b64": "x"}})) as mock_post:
        client.imagine("scene", image_backend="openai", aspect_ratio="16:9", steps=12)

    payload = mock_post.call_args.kwargs["json"]
    assert payload["input"]["image_backend"] == "openai"
    assert payload["input"]["aspect_ratio"] == "16:9"
    assert payload["input"]["steps"] == 12
    assert payload["input"]["secret"] == "sec"  # pragma: allowlist secret
