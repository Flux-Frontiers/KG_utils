"""The rewrite op end to end: WorkerClient -> HTTP -> handle_aux_ops -> TextSynthesizer.

The other worker tests check each side alone, with the other side faked. These
run the real client against a real HTTP server that serves the op the way the
fleet's workers do (``{"input": ...}`` in, ``{"output": ...}`` out), with a
real ``TextSynthesizer`` behind it. Only the model backend is faked.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("httpx", reason="httpx package not installed -- skipping worker round trip")
pytest.importorskip("openai", reason="openai package not installed -- skipping worker round trip")

from kg_utils.synthesis import IMAGE_REWRITE_SYSTEM, IMAGE_REWRITE_TECHNICAL_SYSTEM
from kg_utils.synthesis._config import TextBackend, TextConfig
from kg_utils.synthesis._text import TextSynthesizer
from kg_utils.worker import WorkerClient, handle_aux_ops


def _completion(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture
def worker() -> Iterator[tuple[WorkerClient, MagicMock]]:
    """A worker on an ephemeral port, and the fake model backend behind it.

    :returns: The client pointed at the worker, and the backend's ``create``
        mock, whose calls show what reached the model.
    """
    with patch("openai.OpenAI") as openai_cls:
        create = openai_cls.return_value.chat.completions.create
        create.return_value = _completion("a rewritten scene")
        synth = TextSynthesizer(TextConfig(backend=TextBackend.OMLX))

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                job = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                out = handle_aux_ops(job.get("input", {}), lambda _: synth, lambda _: MagicMock())
                body = json.dumps({"output": out}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield WorkerClient(f"http://127.0.0.1:{server.server_address[1]}"), create
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def _system_sent(create: MagicMock) -> str:
    messages = create.call_args[1]["messages"]
    assert messages[0]["role"] == "system"
    return messages[0]["content"]


def test_rewrite_system_travels_from_client_to_model(worker) -> None:
    client, create = worker
    prompt, err = client.rewrite("def walk(graph): ...", system=IMAGE_REWRITE_TECHNICAL_SYSTEM)
    assert (prompt, err) == ("a rewritten scene", None)
    assert _system_sent(create) == IMAGE_REWRITE_TECHNICAL_SYSTEM
    assert create.call_args[1]["messages"][1] == {"role": "user", "content": "def walk(graph): ..."}


def test_rewrite_without_system_gets_the_historical_prompt(worker) -> None:
    client, create = worker
    prompt, err = client.rewrite("He came home late from the office.")
    assert (prompt, err) == ("a rewritten scene", None)
    assert _system_sent(create) == IMAGE_REWRITE_SYSTEM


def test_rewrite_blank_system_falls_back_to_the_historical_prompt(worker) -> None:
    client, create = worker
    client.rewrite("He came home late from the office.", system="   ")
    assert _system_sent(create) == IMAGE_REWRITE_SYSTEM
