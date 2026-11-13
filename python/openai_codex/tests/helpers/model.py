"""A credential-free Responses endpoint for the actual bundled Codex runtime."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from openai_codex import CodexConfig


class LocalModel:
    """Serve deterministic model responses without any provider access."""

    def __init__(self, root: Path, *, error_status: int | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.error_status = error_status
        self.gate = threading.Event()
        self.gate.set()
        self.root = root
        (root / "codex-home").mkdir()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def config(self) -> CodexConfig:
        return CodexConfig(
            cwd=str(self.root),
            env={"CODEX_HOME": str(self.root / "codex-home")},
            config_overrides=(
                'model_provider="local"',
                'model="test-model"',
                'model_providers.local.name="Local test model"',
                f'model_providers.local.base_url="http://127.0.0.1:{self.server.server_port}/v1"',
                'model_providers.local.wire_api="responses"',
                "model_providers.local.requires_openai_auth=false",
                "model_context_window=100000",
                "model_supports_reasoning=false",
                'web_search="disabled"',
            ),
        )

    def close(self) -> None:
        self.gate.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        model = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model.requests.append(body)
                if model.error_status is not None:
                    data = json.dumps(
                        {
                            "error": {
                                "message": "Provider authentication failed",
                                "type": "authentication_error",
                            }
                        }
                    ).encode()
                    self.send_response(model.error_status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                model.gate.wait(15)
                item = {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "phase": "final_answer",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello from local Codex.",
                            "annotations": [],
                        }
                    ],
                }
                response = {
                    "id": "resp_test",
                    "object": "response",
                    "status": "completed",
                    "output": [item],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 6,
                        "total_tokens": 16,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                }
                events: list[dict[str, Any]] = [
                    {
                        "type": "response.created",
                        "response": {"id": "resp_test", "status": "in_progress"},
                    },
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {**item, "status": "in_progress", "content": []},
                    },
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "item_id": "msg_test",
                        "delta": "Hello from local Codex.",
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item,
                    },
                    {"type": "response.completed", "response": response},
                ]
                data = "".join(
                    f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pass

            def log_message(self, format: str, *args: Any) -> None:
                pass

        return Handler
