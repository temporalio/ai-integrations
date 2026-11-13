"""The real ``OpenAIChatClient`` against a local HTTP server: statuses, headers, SDK retries.

This module defines no workflow, so the OpenAI SDK's import tree stays out of
the workflow sandbox.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from agent_framework import Message
from agent_framework.openai import OpenAIChatClient, OpenAIChatCompletionClient
from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field

from temporalio.exceptions import ApplicationError
from temporalio.microsoft_agent_framework._activities import (
    ChatClientActivities,
    InvokeModelInput,
)
from temporalio.microsoft_agent_framework._serde import dump_options
from temporalio.testing import ActivityEnvironment


class _Handler(BaseHTTPRequestHandler):
    status = 500
    extra_headers: dict[str, str] = {}
    requests: list[str] = []
    bodies: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).requests.append(self.path)
        type(self).bodies.append(json.loads(raw))
        body = json.dumps(
            {"error": {"message": "nope", "type": "server_error"}}
        ).encode()
        self.send_response(type(self).status)
        for name, value in type(self).extra_headers.items():
            self.send_header(name, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture
def local_server() -> Iterator[tuple[str, type[_Handler]]]:
    handler = type(
        "Handler", (_Handler,), {"requests": [], "bodies": [], "extra_headers": {}}
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", handler
    finally:
        server.shutdown()
        server.server_close()


async def test_openai_client_retries(
    local_server: tuple[str, type[_Handler]], caplog: pytest.LogCaptureFixture
) -> None:
    url, handler = local_server
    activities = ChatClientActivities(
        {
            "strict": lambda: OpenAIChatClient(
                model="gpt-4o",
                async_client=AsyncOpenAI(api_key="test", base_url=url, max_retries=0),
            ),
            "default": lambda: OpenAIChatClient(
                model="gpt-4o", api_key="test", base_url=url
            ),
        }
    )
    env = ActivityEnvironment()
    messages = [Message(role="user", contents=["hi"]).to_dict()]

    handler.status = 401
    with pytest.raises(ApplicationError) as excinfo:
        await env.run(
            activities.invoke_model, InvokeModelInput(model="strict", messages=messages)
        )
    assert excinfo.value.type == "AuthenticationError" and excinfo.value.non_retryable
    assert len(handler.requests) == 1

    handler.status = 429
    handler.extra_headers = {"retry-after-ms": "250"}
    with pytest.raises(ApplicationError) as excinfo:
        await env.run(
            activities.invoke_model, InvokeModelInput(model="strict", messages=messages)
        )
    assert excinfo.value.type == "RateLimitError" and not excinfo.value.non_retryable
    assert excinfo.value.next_retry_delay == timedelta(milliseconds=250)
    assert len(handler.requests) == 2

    handler.status = 500
    handler.extra_headers = {}
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ApplicationError) as excinfo:
            await env.run(
                activities.invoke_model,
                InvokeModelInput(model="default", messages=messages),
            )
    assert (
        excinfo.value.type == "InternalServerError" and not excinfo.value.non_retryable
    )
    # The SDK's default client retried twice on its own, and the Activity said so.
    assert len(handler.requests) == 5
    assert any("retries 2 times" in record.getMessage() for record in caplog.records)


class _Address(BaseModel):
    city: str
    zip: str | None = None


class _Person(BaseModel):
    name: str
    age: int
    nickname: str | None = None
    tags: list[str] = Field(default_factory=list)
    address: _Address | None = None
    note: str = Field("", description="free text")


@pytest.mark.parametrize(
    "client_class",
    [OpenAIChatClient, OpenAIChatCompletionClient],
    ids=["responses", "completions"],
)
async def test_response_format_matches_the_native_request(
    local_server: tuple[str, type[_Handler]], client_class: type[Any]
) -> None:
    url, handler = local_server
    handler.status = 400

    def build() -> Any:
        return client_class(
            model="gpt-4o",
            async_client=AsyncOpenAI(api_key="test", base_url=url, max_retries=0),
        )

    messages = [Message(role="user", contents=["hi"])]
    # Natively: the framework's client with the Pydantic class itself.
    with pytest.raises(Exception):
        await build().get_response(messages, options={"response_format": _Person})
    # Through the plugin: the class crosses the Activity boundary as a schema,
    # and the worker's client sends that.
    activities = ChatClientActivities({"model": build})
    with pytest.raises(ApplicationError):
        await ActivityEnvironment().run(
            activities.invoke_model,
            InvokeModelInput(
                model="model",
                messages=[m.to_dict() for m in messages],
                options=dump_options({"response_format": _Person}),
            ),
        )
    native, through_plugin = handler.bodies
    if client_class is OpenAIChatClient:
        sent = native["text"]["format"]
        assert through_plugin["text"] == native["text"]
    else:
        sent = native["response_format"]["json_schema"]
        assert through_plugin["response_format"] == native["response_format"]
    # Both carry the SDK's strict form: strict on, every property required,
    # no additional properties, the None defaults dropped, the nested model
    # rewritten the same way.
    assert sent["strict"] is True
    assert sent["schema"] == to_strict_json_schema(_Person)
    assert sent["schema"]["additionalProperties"] is False
    assert sent["schema"]["required"] == list(_Person.model_fields)
    assert sent["schema"]["$defs"]["_Address"]["required"] == ["city", "zip"]
