"""``agent.run(stream=True)``, the ``streaming_topic`` publisher, and the streamed record."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from agent_framework import (
    Agent,
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
    tool,
)

from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamClient
from temporalio.exceptions import ApplicationError
from temporalio.microsoft_agent_framework import (
    MicrosoftAgentFrameworkPlugin,
    TemporalChatClient,
    _activities,
)
from temporalio.microsoft_agent_framework._activities import (
    ChatClientActivities,
    InvokeModelStreamingInput,
)
from temporalio.microsoft_agent_framework._serde import load_update
from temporalio.microsoft_agent_framework.testing import (
    FakeChatClient,
    fake_chat_client,
    function_call_response,
    text_response,
)
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import ActivityEnvironment
from tests.helpers.microsoft_agent_framework import (
    INVOKE_MODEL_STREAMING,
    SlowFakeChatClient,
    contents_summary,
    count_scheduled_activities,
    count_signals,
    new_worker,
    workflow_id,
)


@dataclass
class StreamInput:
    prompt: str = "weather in Oslo?"
    topic: str | None = None
    batch_interval_ms: float = 100


def _kinds(update: Any) -> list[str]:
    return [content.type for content in update.contents]


@workflow.defn
class StreamingWorkflow:
    def __init__(self) -> None:
        self._stream = WorkflowStream()

    @workflow.run
    async def run(self, input: StreamInput) -> dict[str, Any]:
        log: list[str] = []

        @tool
        async def get_weather(city: str) -> str:
            """Get the weather for a city."""
            log.append(city)
            return f"rainy in {city}"

        client = TemporalChatClient(
            "model",
            streaming_topic=input.topic,
            streaming_batch_interval=timedelta(milliseconds=input.batch_interval_ms),
        )
        agent = Agent(client=client, tools=[get_weather])
        stream = agent.run(input.prompt, stream=True)
        updates = [(_kinds(update), update.text) async for update in stream]
        final = await stream.get_final_response()
        return {
            "updates": updates,
            "text": final.text,
            "contents": contents_summary(final),
            "log": log,
        }


_TEXT = "Streamed answer: rainy in Oslo, bring a coat."
_SCRIPT = [
    function_call_response("get_weather", {"city": "Oslo"}, call_id="call-1"),
    text_response(_TEXT),
]


async def test_run_stream(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(_SCRIPT)}
    )
    async with new_worker(client, StreamingWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            StreamingWorkflow.run,
            StreamInput(),
            id=workflow_id("stream"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    assert result["text"] == _TEXT
    assert result["log"] == ["Oslo"]
    # The same sequence a native run yields: the call once, the result the
    # workflow produced, then the text, which the Activity records as one run.
    # Newer framework releases add one empty update after a tool round, so
    # only updates with contents are compared.
    assert [u for u in result["updates"] if u[0]] == [
        [["function_call"], ""],
        [["function_result"], ""],
        [["text"], _TEXT],
    ]
    assert result["contents"] == [
        "call:get_weather",
        "result:rainy in Oslo",
        f"text:{_TEXT}",
    ]
    assert await count_scheduled_activities(handle) == {INVOKE_MODEL_STREAMING: 2}


async def test_streaming_topic(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(_SCRIPT)}
    )
    async with new_worker(client, StreamingWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            StreamingWorkflow.run,
            StreamInput(topic="events"),
            id=workflow_id("stream-topic"),
            task_queue=worker.task_queue,
        )
        stream_client = WorkflowStreamClient.create(client, handle.id)
        items = [item async for item in stream_client.subscribe(["events"])]
        result = await handle.result()
    assert all(item.topic == "events" for item in items)
    updates = [ChatResponseUpdate.from_dict(item.data) for item in items]
    # Subscribers get every provider update: the call once, then the text in
    # the 12-character chunks the scripted client streams.
    chunks = [_TEXT[i : i + 12] for i in range(0, len(_TEXT), 12)]
    assert [(_kinds(u), u.text) for u in updates] == [
        (["function_call"], ""),
        *[(["text"], chunk) for chunk in chunks],
    ]
    assert "".join(u.text for u in updates) == result["text"]


@pytest.mark.parametrize(
    ("batch_interval_ms", "single_batch"), [(5000, True), (10, False)]
)
async def test_streaming_batch_interval(
    client: Client, batch_interval_ms: float, single_batch: bool
) -> None:
    text = "A long enough streamed answer to split into several update chunks."
    slow = SlowFakeChatClient([text], delay=0.1)
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": lambda: slow})
    async with new_worker(client, StreamingWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            StreamingWorkflow.run,
            StreamInput(topic="events", batch_interval_ms=batch_interval_ms),
            id=workflow_id("stream-batch"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    assert result["text"] == text
    # Each flushed batch is one Signal to the workflow.
    signals = await count_signals(handle)
    assert (signals == 1) is single_batch, signals


class _FakeStream:
    """Stands in for the workflow stream client; records publishes, fails on exit if asked."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.published: list[Any] = []

    def topic(self, _name: str) -> _FakeStream:
        return self

    def publish(self, value: Any) -> None:
        self.published.append(value)

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self.error is not None:
            raise self.error


def _install_fake_stream(monkeypatch: pytest.MonkeyPatch, fake: _FakeStream) -> None:
    class _StreamClient:
        @staticmethod
        def from_within_activity(**_kwargs: Any) -> _FakeStream:
            return fake

    monkeypatch.setattr(_activities, "WorkflowStreamClient", _StreamClient)


def _messages() -> list[dict[str, Any]]:
    return [Message(role="user", contents=["hi"]).to_dict()]


async def test_long_stream_record_stays_small(monkeypatch: pytest.MonkeyPatch) -> None:
    # 300k characters: the scripted client streams 25k updates of 12 characters.
    text = "word " * 60_000
    fake = _FakeStream()
    _install_fake_stream(monkeypatch, fake)
    activities = ChatClientActivities({"model": lambda: FakeChatClient([text])})
    record = await ActivityEnvironment().run(
        activities.invoke_model_streaming,
        InvokeModelStreamingInput(
            model="model", messages=_messages(), streaming_topic="events"
        ),
    )
    # One text entry for the whole answer, so the Activity result is the text
    # plus one envelope, far under the 2 MiB payload limit (and the 512 KiB
    # warning), where one entry per update would have crossed 2 MiB.
    assert len(record) == 1
    assert record[0]["contents"][0]["text"] == text
    assert len(json.dumps(record)) < len(text) + 256
    assert sum(len(json.dumps(item)) for item in fake.published) > 2 * 1024 * 1024
    # The topic still received every update.
    assert len(fake.published) == -(-len(text) // 12)
    assert "".join(item["contents"][0]["text"] for item in fake.published) == text


async def test_stream_record_keeps_calls_and_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeStream()
    _install_fake_stream(monkeypatch, fake)
    script = [function_call_response("get_weather", {"city": "Oslo"}, call_id="call-1")]
    activities = ChatClientActivities({"model": lambda: FakeChatClient(script)})
    declared = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "w",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    record = await ActivityEnvironment().run(
        activities.invoke_model_streaming,
        InvokeModelStreamingInput(
            model="model",
            messages=_messages(),
            options={"tools": [declared]},
            streaming_topic="events",
        ),
    )
    # The provider layer's marked copy of the call is dropped from the record
    # and from the topic; the call itself is recorded once, without the marker.
    assert [[c["type"] for c in item["contents"]] for item in record] == [
        ["function_call"]
    ]
    assert "user_input_request" not in record[0]["contents"][0]
    assert record[0]["contents"][0]["call_id"] == "call-1"
    assert len(fake.published) == 1


@pytest.mark.parametrize(
    ("publish_error", "non_retryable"),
    [
        (
            RPCError(
                "Blob data size exceeds limit.", RPCStatusCode.INVALID_ARGUMENT, b""
            ),
            True,
        ),
        (
            RPCError(
                "grpc: received message after decompression larger than max 4194304",
                RPCStatusCode.RESOURCE_EXHAUSTED,
                b"",
            ),
            True,
        ),
        (TimeoutError("Flush retry exceeded max_retry_duration (0:10:00)."), False),
        (RPCError("connection refused", RPCStatusCode.UNAVAILABLE, b""), False),
    ],
    ids=["refused", "too-large", "gave-up", "unavailable"],
)
async def test_stream_publish_failure_types(
    monkeypatch: pytest.MonkeyPatch, publish_error: Exception, non_retryable: bool
) -> None:
    fake = _FakeStream(publish_error)
    _install_fake_stream(monkeypatch, fake)
    activities = ChatClientActivities(
        {"model": lambda: FakeChatClient(["streamed text"])}
    )
    with pytest.raises(ApplicationError) as excinfo:
        await ActivityEnvironment().run(
            activities.invoke_model_streaming,
            InvokeModelStreamingInput(
                model="model", messages=_messages(), streaming_topic="events"
            ),
        )
    assert excinfo.value.type == "microsoft_agent_framework.StreamError"
    assert excinfo.value.non_retryable is non_retryable
    assert "'model'" in excinfo.value.message and "'events'" in excinfo.value.message
    assert excinfo.value.__cause__ is publish_error
    assert fake.published


class _DeltaClient(BaseChatClient):
    """Streams prepared updates, with no function-invocation layer in between."""

    OTEL_PROVIDER_NAME = "delta"

    def __init__(self, updates: list[ChatResponseUpdate]) -> None:
        super().__init__()
        self._updates = updates

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        async def updates() -> AsyncIterator[ChatResponseUpdate]:
            for update in self._updates:
                yield update

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)


def _responses_style_deltas(full: str, chunk: int) -> list[ChatResponseUpdate]:
    # The OpenAI Responses client: one append delta per argument token, then
    # one replace update carrying the full arguments.
    extra = {"output_index": 1, "fc_id": "fc_1"}
    parts = [full[i : i + chunk] for i in range(0, len(full), chunk)]
    updates = [
        ChatResponseUpdate(
            role="assistant",
            response_id="resp_1",
            message_id="msg_1",
            contents=[
                Content.from_function_call(
                    "call_1",
                    "write_file",
                    arguments=part,
                    merge_mode="append",
                    additional_properties=dict(extra),
                )
            ],
        )
        for part in parts
    ]
    updates.append(
        ChatResponseUpdate(
            role="assistant",
            response_id="resp_1",
            message_id="msg_1",
            contents=[
                Content.from_function_call(
                    "call_1",
                    "write_file",
                    arguments=full,
                    merge_mode="replace",
                    additional_properties=dict(extra),
                )
            ],
        )
    )
    return updates


def _completions_style_deltas(full: str, chunk: int) -> list[ChatResponseUpdate]:
    # The Chat Completions client: the first chunk names the call, the
    # continuation chunks carry an empty id and the call's index.
    parts = [full[i : i + chunk] for i in range(0, len(full), chunk)]
    return [
        ChatResponseUpdate(
            role="assistant",
            response_id="chatcmpl_1",
            message_id="chatcmpl_1",
            contents=[
                Content.from_function_call(
                    "call_1" if index == 0 else "",
                    "write_file" if index == 0 else "",
                    arguments=part,
                    merge_mode="append",
                    additional_properties={"tool_call_index": 0},
                )
            ],
        )
        for index, part in enumerate(parts)
    ]


def _calls(response: ChatResponse) -> list[tuple[str | None, str | None, Any]]:
    return [
        (c.call_id, c.name, c.arguments)
        for m in response.messages
        for c in m.contents
        if c.type == "function_call"
    ]


@pytest.mark.parametrize(("style", "entries"), [("responses", 2), ("completions", 1)])
async def test_stream_record_merges_argument_deltas(
    monkeypatch: pytest.MonkeyPatch, style: str, entries: int
) -> None:
    # 100k characters of arguments in 25k deltas of 4 characters.
    full = json.dumps({"path": "a.py", "content": "x" * 99_960})
    build = (
        _responses_style_deltas if style == "responses" else _completions_style_deltas
    )
    updates = build(full, 4)
    assert len(updates) >= 24_000
    fake = _FakeStream()
    _install_fake_stream(monkeypatch, fake)
    activities = ChatClientActivities({"model": lambda: _DeltaClient(updates)})
    record = await ActivityEnvironment().run(
        activities.invoke_model_streaming,
        InvokeModelStreamingInput(
            model="model", messages=_messages(), streaming_topic="events"
        ),
    )
    # One merged append entry per call (plus the Responses client's replace
    # entry): the record is the arguments plus envelopes, where one entry per
    # delta would be over 2 MiB.
    assert len(record) == entries
    assert len(json.dumps(record)) < entries * len(full) + 1024
    assert sum(len(json.dumps(item)) for item in fake.published) > 2 * 1024 * 1024
    assert len(fake.published) == len(updates)
    # The workflow rebuilds the same call from the record as from the raw deltas.
    rebuilt = ChatResponse.from_updates([load_update(entry) for entry in record])
    assert _calls(rebuilt) == _calls(ChatResponse.from_updates(updates))
    (call,) = _calls(rebuilt)
    assert call[:2] == ("call_1", "write_file")
    arguments = call[2]
    assert json.loads(arguments) if isinstance(arguments, str) else arguments
