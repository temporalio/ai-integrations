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
from agent_framework import ChatResponse, Message
from agent_framework.openai import OpenAIChatClient, OpenAIChatCompletionClient
from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field

from temporalio.exceptions import ApplicationError
from temporalio.microsoft_agent_framework._activities import (
    ChatClientActivities,
    InvokeModelInput,
    InvokeModelStreamingInput,
)
from temporalio.microsoft_agent_framework._serde import dump_options, load_update
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


class _SSEHandler(BaseHTTPRequestHandler):
    """A Responses API endpoint that streams ``deltas`` as server-sent events."""

    deltas: list[str] = []
    bodies: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).bodies.append(json.loads(raw))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in self._events():
            payload = json.dumps(event)
            self.wfile.write(f"event: {event['type']}\ndata: {payload}\n\n".encode())
        self.wfile.flush()

    def _events(self) -> Iterator[dict[str, Any]]:
        sequence = 0

        def event(data: dict[str, Any]) -> dict[str, Any]:
            nonlocal sequence
            data["sequence_number"] = sequence
            sequence += 1
            return data

        def response(status: str, output: list[Any], usage: Any = None) -> dict:
            return {
                "id": "resp_1",
                "object": "response",
                "created_at": 1700000000,
                "status": status,
                "model": "gpt-4o",
                "output": output,
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": {},
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
                "usage": usage,
            }

        full = "".join(type(self).deltas)
        part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        item = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        where = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
        yield event(
            {"type": "response.created", "response": response("in_progress", [])}
        )
        yield event(
            {"type": "response.in_progress", "response": response("in_progress", [])}
        )
        yield event(
            {"type": "response.output_item.added", "output_index": 0, "item": item}
        )
        yield event({"type": "response.content_part.added", **where, "part": part})
        for delta in type(self).deltas:
            yield event(
                {
                    "type": "response.output_text.delta",
                    **where,
                    "delta": delta,
                    "logprobs": [],
                }
            )
        yield event(
            {"type": "response.output_text.done", **where, "text": full, "logprobs": []}
        )
        done_part = {**part, "text": full}
        yield event({"type": "response.content_part.done", **where, "part": done_part})
        done_item = {**item, "status": "completed", "content": [done_part]}
        yield event(
            {"type": "response.output_item.done", "output_index": 0, "item": done_item}
        )
        usage = {
            "input_tokens": 3,
            "output_tokens": len(type(self).deltas),
            "total_tokens": 3 + len(type(self).deltas),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        yield event(
            {
                "type": "response.completed",
                "response": response("completed", [done_item], usage),
            }
        )

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture
def sse_server() -> Iterator[tuple[str, type[_SSEHandler]]]:
    handler = type("SSEHandler", (_SSEHandler,), {"deltas": [], "bodies": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", handler
    finally:
        server.shutdown()
        server.server_close()


async def test_streamed_record_stays_small_with_the_real_client(
    sse_server: tuple[str, type[_SSEHandler]],
) -> None:
    url, handler = sse_server
    handler.deltas = [f"t{i} " for i in range(200)]
    text = "".join(handler.deltas)
    activities = ChatClientActivities(
        {
            "model": lambda: OpenAIChatClient(
                model="gpt-4o",
                async_client=AsyncOpenAI(api_key="test", base_url=url, max_retries=0),
            )
        }
    )
    record = await ActivityEnvironment().run(
        activities.invoke_model_streaming,
        InvokeModelStreamingInput(
            model="model",
            messages=[Message(role="user", contents=["hi"]).to_dict()],
            options={"store": False},
        ),
    )
    # The real client emits each delta with merge_mode="append" and empty
    # logprobs; the 200 deltas record as one entry, next to the created,
    # in-progress, item-added, final-text, part-done, item-done and usage
    # updates. Unmerged, the same stream was 205 entries and 54 KB for about
    # 1 KB of text.
    assert len(record) == 8
    assert len(json.dumps(record)) < 2 * len(text) + 2048
    texts = [
        c["text"] for entry in record for c in entry["contents"] if c["type"] == "text"
    ]
    assert texts == [text, text]
    (merged,) = [
        c
        for entry in record
        for c in entry["contents"]
        if c["type"] == "text" and c.get("merge_mode") == "append"
    ]
    assert "logprobs" not in merged["additional_properties"]
    rebuilt = ChatResponse.from_updates([load_update(entry) for entry in record])
    assert rebuilt.text == text.strip()
    assert rebuilt.usage_details is not None
    assert rebuilt.usage_details.get("output_token_count") == 200
    # The option the workflow sends reached the request body.
    assert handler.bodies[0]["store"] is False
