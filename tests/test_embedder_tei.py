"""Tests for kg_utils.embedder.TEIEmbedder — remote Text Embeddings Inference backend.

The unit tests stub the HTTP layer, so the whole file runs with no server and
no heavy dependencies.  The handful of tests needing a live TEI are marked
``integration`` and skip unless ``KG_EMBED_ENDPOINT`` is set.
"""

# pylint: disable=redefined-outer-name,missing-function-docstring,protected-access

from __future__ import annotations

import io
import json
import math
import os
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from kg_utils.embedder import (
    DEFAULT_TEI_ENDPOINT,
    TEI_DEFAULT_CLIENT_BATCH,
    Embedder,
    TEIEmbedder,
)

# ---------------------------------------------------------------------------
# HTTP stubbing helpers
# ---------------------------------------------------------------------------

# BytesIO is already a context manager returning self, which is all the code
# under test asks of urlopen's return value.


def _json_resp(payload: Any) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode())


def _vecs(n: int, dim: int = 384) -> list[list[float]]:
    """n unit-ish vectors of width dim."""
    return [[1.0] + [0.0] * (dim - 1) for _ in range(n)]


class _Recorder:
    """Callable stub for urlopen that records requests and replays responses.

    :param responses: One entry per expected call — either a payload to encode
        as JSON, or an exception instance to raise.
    """

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, req: Any, timeout: float | None = None) -> io.BytesIO:
        body = req.data.decode() if getattr(req, "data", None) else ""
        self.calls.append(
            {
                "url": req.full_url,
                "timeout": timeout,
                "headers": {k.lower(): v for k, v in req.headers.items()},
                "payload": json.loads(body) if body else None,
            }
        )
        nxt = self.responses.pop(0) if self.responses else []
        if isinstance(nxt, BaseException):
            raise nxt
        return _json_resp(nxt)


def _http_error(code: int) -> HTTPError:
    return HTTPError("http://x/embed", code, "boom", {}, io.BytesIO(b"detail"))


@pytest.fixture
def embedder() -> TEIEmbedder:
    """A TEIEmbedder constructed offline (dim given => no probe, no network)."""
    return TEIEmbedder("http://tei.local:8080", dim=384, max_retries=0)


# ---------------------------------------------------------------------------
# Construction & configuration
# ---------------------------------------------------------------------------


def test_is_an_embedder(embedder: TEIEmbedder) -> None:
    assert isinstance(embedder, Embedder)


def test_explicit_dim_performs_no_network_io() -> None:
    with patch("kg_utils.embedder.urlopen", side_effect=AssertionError("network!")):
        emb = TEIEmbedder("http://tei.local:8080", dim=768)
    assert emb.dim == 768
    assert emb.max_batch == TEI_DEFAULT_CLIENT_BATCH


def test_endpoint_precedence_arg_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KG_EMBED_ENDPOINT", "http://from-env:9999")
    assert TEIEmbedder("http://explicit:1234", dim=1).endpoint == "http://explicit:1234"


def test_endpoint_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KG_EMBED_ENDPOINT", "http://from-env:9999")
    assert TEIEmbedder(dim=1).endpoint == "http://from-env:9999"


def test_endpoint_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KG_EMBED_ENDPOINT", raising=False)
    assert TEIEmbedder(dim=1).endpoint == DEFAULT_TEI_ENDPOINT


@pytest.mark.parametrize(
    "given,expected",
    [
        ("http://h:8080/", "http://h:8080"),
        ("http://h:8080/v1", "http://h:8080"),
        ("http://h:8080/v1/", "http://h:8080"),
        ("  http://h:8080  ", "http://h:8080"),
    ],
)
def test_endpoint_normalisation(given: str, expected: str) -> None:
    """Native TEI routes live at the root; /v1 is only the OpenAI-compat alias."""
    assert TEIEmbedder(given, dim=1).endpoint == expected


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KG_EMBED_API_KEY", "dummy-token")  # pragma: allowlist secret
    assert TEIEmbedder(dim=1).api_key == "dummy-token"  # pragma: allowlist secret


def test_repr_mentions_endpoint_and_dim(embedder: TEIEmbedder) -> None:
    assert "TEIEmbedder" in repr(embedder)
    assert "tei.local" in repr(embedder)
    assert "384" in repr(embedder)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def test_probe_learns_dim_batch_and_model() -> None:
    """/info carries the limits; dim is measured, since /info does not report it."""
    rec = _Recorder(
        {
            "max_client_batch_size": 128,
            "model_id": "BAAI/bge-small-en-v1.5",
            "max_input_length": 512,
        },
        _vecs(1, 384),
    )
    with patch("kg_utils.embedder.urlopen", rec):
        emb = TEIEmbedder("http://tei.local:8080")

    assert emb.dim == 384
    assert emb.max_batch == 128
    assert emb.model_name == "BAAI/bge-small-en-v1.5"
    assert rec.calls[0]["url"].endswith("/info")
    assert rec.calls[1]["url"].endswith("/embed")


def test_probe_survives_missing_info_endpoint() -> None:
    """/info is advisory — a 404 there must not stop the dimension probe."""
    rec = _Recorder(_http_error(404), _vecs(1, 768))
    with patch("kg_utils.embedder.urlopen", rec):
        emb = TEIEmbedder("http://tei.local:8080")

    assert emb.dim == 768
    assert emb.max_batch == TEI_DEFAULT_CLIENT_BATCH


def test_probe_raises_when_server_unreachable() -> None:
    rec = _Recorder(URLError("refused"), URLError("refused"))
    with (
        patch("kg_utils.embedder.urlopen", rec),
        pytest.raises(RuntimeError, match="TEI request failed"),
    ):
        TEIEmbedder("http://tei.local:8080", max_retries=0)


def test_explicit_max_batch_survives_probe() -> None:
    rec = _Recorder({"max_client_batch_size": 128}, _vecs(1))
    with patch("kg_utils.embedder.urlopen", rec):
        emb = TEIEmbedder("http://tei.local:8080", max_batch=16)
    assert emb.max_batch == 16


# ---------------------------------------------------------------------------
# embed_texts — batching
# ---------------------------------------------------------------------------


def test_embed_texts_empty_short_circuits(embedder: TEIEmbedder) -> None:
    with patch("kg_utils.embedder.urlopen", side_effect=AssertionError("network!")):
        assert embedder.embed_texts([]) == []


def test_embed_texts_single_request(embedder: TEIEmbedder) -> None:
    rec = _Recorder(_vecs(3))
    with patch("kg_utils.embedder.urlopen", rec):
        out = embedder.embed_texts(["a", "b", "c"])

    assert len(out) == 3
    assert all(len(v) == 384 for v in out)
    assert len(rec.calls) == 1
    assert rec.calls[0]["payload"]["inputs"] == ["a", "b", "c"]


def test_embed_texts_clamps_to_server_batch_ceiling() -> None:
    """A caller asking for 128 must be split to the server's limit, not rejected.

    This is the HTTP 422 that a stock TEI (max_client_batch_size=32) returns
    when handed the fleet's 128-item convention.
    """
    emb = TEIEmbedder("http://tei.local:8080", dim=4, max_batch=32)
    rec = _Recorder(_vecs(32, 4), _vecs(32, 4), _vecs(6, 4))
    with patch("kg_utils.embedder.urlopen", rec):
        out = emb.embed_texts([f"t{i}" for i in range(70)], encode_batch_size=128)

    assert len(out) == 70
    assert [len(c["payload"]["inputs"]) for c in rec.calls] == [32, 32, 6]


def test_caller_batch_smaller_than_ceiling_is_respected() -> None:
    emb = TEIEmbedder("http://tei.local:8080", dim=4, max_batch=128)
    rec = _Recorder(_vecs(2, 4), _vecs(2, 4))
    with patch("kg_utils.embedder.urlopen", rec):
        emb.embed_texts(["a", "b", "c", "d"], encode_batch_size=2)

    assert [len(c["payload"]["inputs"]) for c in rec.calls] == [2, 2]


def test_batch_order_is_preserved() -> None:
    emb = TEIEmbedder("http://tei.local:8080", dim=1, max_batch=2)
    rec = _Recorder([[1.0], [2.0]], [[3.0]])
    with patch("kg_utils.embedder.urlopen", rec):
        out = emb.embed_texts(["a", "b", "c"])
    assert out == [[1.0], [2.0], [3.0]]


# ---------------------------------------------------------------------------
# embed_texts — request shape
# ---------------------------------------------------------------------------


def test_requests_normalized_and_truncated_vectors(embedder: TEIEmbedder) -> None:
    """Matches the fleet contract: normalize_embeddings=True, ST-style silent truncation."""
    rec = _Recorder(_vecs(1))
    with patch("kg_utils.embedder.urlopen", rec):
        embedder.embed_texts(["x"])

    assert rec.calls[0]["payload"]["normalize"] is True
    assert rec.calls[0]["payload"]["truncate"] is True


def test_no_auth_header_without_key(embedder: TEIEmbedder) -> None:
    rec = _Recorder(_vecs(1))
    with patch("kg_utils.embedder.urlopen", rec):
        embedder.embed_texts(["x"])
    assert "authorization" not in rec.calls[0]["headers"]


def test_bearer_header_when_key_set() -> None:
    emb = TEIEmbedder(
        "http://tei.local:8080",
        dim=384,
        api_key="dummy-token",  # pragma: allowlist secret
    )
    rec = _Recorder(_vecs(1))
    with patch("kg_utils.embedder.urlopen", rec):
        emb.embed_texts(["x"])
    assert rec.calls[0]["headers"]["authorization"] == "Bearer dummy-token"


def test_embed_query_delegates(embedder: TEIEmbedder) -> None:
    rec = _Recorder(_vecs(1))
    with patch("kg_utils.embedder.urlopen", rec):
        vec = embedder.embed_query("a query")

    assert len(vec) == 384
    assert rec.calls[0]["payload"]["inputs"] == ["a query"]


# ---------------------------------------------------------------------------
# Failure handling — loud, never silent
# ---------------------------------------------------------------------------


def test_retries_429_then_succeeds() -> None:
    """TEI sheds load with 429 rather than queueing; that must be survivable."""
    emb = TEIEmbedder("http://tei.local:8080", dim=384, max_retries=2)
    rec = _Recorder(_http_error(429), _vecs(1))
    with patch("kg_utils.embedder.urlopen", rec), patch("kg_utils.embedder.time.sleep"):
        out = emb.embed_texts(["x"])

    assert len(out) == 1
    assert len(rec.calls) == 2


@pytest.mark.parametrize("code", [502, 503, 504])
def test_retries_transient_5xx(code: int) -> None:
    emb = TEIEmbedder("http://tei.local:8080", dim=384, max_retries=1)
    rec = _Recorder(_http_error(code), _vecs(1))
    with patch("kg_utils.embedder.urlopen", rec), patch("kg_utils.embedder.time.sleep"):
        assert len(emb.embed_texts(["x"])) == 1


def test_retries_transport_errors() -> None:
    emb = TEIEmbedder("http://tei.local:8080", dim=384, max_retries=1)
    rec = _Recorder(URLError("connection reset"), _vecs(1))
    with patch("kg_utils.embedder.urlopen", rec), patch("kg_utils.embedder.time.sleep"):
        assert len(emb.embed_texts(["x"])) == 1


def test_gives_up_after_max_retries() -> None:
    emb = TEIEmbedder("http://tei.local:8080", dim=384, max_retries=2)
    rec = _Recorder(*[_http_error(429)] * 3)
    with (
        patch("kg_utils.embedder.urlopen", rec),
        patch("kg_utils.embedder.time.sleep"),
        pytest.raises(RuntimeError, match="after 3 attempts"),
    ):
        emb.embed_texts(["x"])
    assert len(rec.calls) == 3


@pytest.mark.parametrize("code", [400, 413, 422])
def test_client_errors_raise_immediately_without_retry(code: int) -> None:
    """A 422 is a request-shape bug — retrying it just wastes time."""
    emb = TEIEmbedder("http://tei.local:8080", dim=384, max_retries=3)
    rec = _Recorder(_http_error(code), _vecs(1))
    with (
        patch("kg_utils.embedder.urlopen", rec),
        patch("kg_utils.embedder.time.sleep"),
        pytest.raises(RuntimeError, match=f"HTTP {code}"),
    ):
        emb.embed_texts(["x"])
    assert len(rec.calls) == 1


def test_wrong_dimension_is_refused(embedder: TEIEmbedder) -> None:
    """Mixed-width vectors would corrupt the store; fail instead of writing them."""
    rec = _Recorder(_vecs(1, 768))
    with (
        patch("kg_utils.embedder.urlopen", rec),
        pytest.raises(RuntimeError, match="768-dim vector"),
    ):
        embedder.embed_texts(["x"])


def test_short_response_is_refused(embedder: TEIEmbedder) -> None:
    """Fewer vectors than inputs would silently misalign vectors with nodes."""
    rec = _Recorder(_vecs(2))
    with (
        patch("kg_utils.embedder.urlopen", rec),
        pytest.raises(RuntimeError, match="2 vectors for 3 inputs"),
    ):
        embedder.embed_texts(["a", "b", "c"])


# ---------------------------------------------------------------------------
# Live server (opt-in)
# ---------------------------------------------------------------------------

_LIVE = os.environ.get("KG_EMBED_ENDPOINT")


@pytest.mark.integration
@pytest.mark.skipif(not _LIVE, reason="set KG_EMBED_ENDPOINT to a live TEI server")
def test_live_probe_and_embed() -> None:
    emb = TEIEmbedder(_LIVE)
    assert emb.dim > 0

    vecs = emb.embed_texts(["Call me Ishmael.", "The whale surfaced."])
    assert len(vecs) == 2
    assert all(len(v) == emb.dim for v in vecs)
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.integration
@pytest.mark.skipif(not _LIVE, reason="set KG_EMBED_ENDPOINT to a live TEI server")
def test_live_matches_sentence_transformers() -> None:
    """Parity gate from the Phase 0 evaluation: cosine >= 0.999 against ST."""
    ste = pytest.importorskip("kg_utils.embedder").SentenceTransformerEmbedder
    texts = ["Call me Ishmael.", "The try-works were started at nine o'clock."]

    tei_vecs = TEIEmbedder(_LIVE).embed_texts(texts)
    st_vecs = ste().embed_texts(texts)

    for a, b in zip(tei_vecs, st_vecs):
        assert sum(x * y for x, y in zip(a, b)) > 0.999
