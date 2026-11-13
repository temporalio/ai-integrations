"""``agent.run(stream=True)`` and the ``streaming_topic`` publisher."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from agent_framework import Agent, ChatResponseUpdate, Message, tool

from temporalio import workflow
from temporalio.agent_framework import (
    AgentFrameworkPlugin,
    TemporalChatClient,
    _activities,
)
from temporalio.agent_framework._activities import (
    ChatClientActivities,
    InvokeModelStreamingInput,
)
from temporalio.agent_framework.testing import (
    FakeChatClient,
    fake_chat_client,
    function_call_response,
    text_response,
)
from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamClient
from temporalio.exceptions import ApplicationError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import ActivityEnvironment
from tests.helpers.agent_framework import (
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
        updates = [update.text async for update in stream]
        final = await stream.get_final_response()
        return {
            "updates": updates,
            "text": final.text,
            "contents": contents_summary(final),
            "log": log,
        }


_SCRIPT = [
    function_call_response("get_weather", {"city": "Oslo"}, call_id="call-1"),
    text_response("Streamed answer: rainy in Oslo, bring a coat."),
]


async def test_run_stream(client: Client) -> None:
    plugin = AgentFrameworkPlugin(chat_clients={"model": fake_chat_client(_SCRIPT)})
    async with new_worker(client, StreamingWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            StreamingWorkflow.run,
            StreamInput(),
            id=workflow_id("stream"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    text = "Streamed answer: rainy in Oslo, bring a coat."
    assert result["text"] == text
    assert result["log"] == ["Oslo"]
    chunks = [u for u in result["updates"] if u]
    assert len(chunks) > 1 and "".join(chunks) == text
    assert result["contents"][-1] == f"text:{text}"
    assert await count_scheduled_activities(handle) == {INVOKE_MODEL_STREAMING: 2}


async def test_streaming_topic(client: Client) -> None:
    plugin = AgentFrameworkPlugin(chat_clients={"model": fake_chat_client(_SCRIPT)})
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
    updates = [ChatResponseUpdate.from_dict(item.data) for item in items]
    assert all(item.topic == "events" for item in items)
    assert "".join(u.text for u in updates) == result["text"]
    assert any(c.type == "function_call" for u in updates for c in u.contents)


@pytest.mark.parametrize(
    ("batch_interval_ms", "single_batch"), [(5000, True), (10, False)]
)
async def test_streaming_batch_interval(
    client: Client, batch_interval_ms: float, single_batch: bool
) -> None:
    slow = SlowFakeChatClient(
        ["A long enough streamed answer to split into several update chunks."],
        delay=0.1,
    )
    plugin = AgentFrameworkPlugin(chat_clients={"model": lambda: slow})
    async with new_worker(client, StreamingWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            StreamingWorkflow.run,
            StreamInput(topic="events", batch_interval_ms=batch_interval_ms),
            id=workflow_id("stream-batch"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    assert len([u for u in result["updates"] if u]) >= 5
    # Each flushed batch is one Signal to the workflow.
    signals = await count_signals(handle)
    assert (signals == 1) is single_batch, signals


class _RefusingStream:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.published: list[Any] = []

    def topic(self, _name: str) -> _RefusingStream:
        return self

    def publish(self, value: Any) -> None:
        self.published.append(value)

    async def __aenter__(self) -> _RefusingStream:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        raise self.error


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
    refusing = _RefusingStream(publish_error)

    class _StreamClient:
        @staticmethod
        def from_within_activity(**_kwargs: Any) -> _RefusingStream:
            return refusing

    monkeypatch.setattr(_activities, "WorkflowStreamClient", _StreamClient)
    activities = ChatClientActivities(
        {"model": lambda: FakeChatClient(["streamed text"])}
    )
    with pytest.raises(ApplicationError) as excinfo:
        await ActivityEnvironment().run(
            activities.invoke_model_streaming,
            InvokeModelStreamingInput(
                model="model",
                messages=[Message(role="user", contents=["hi"]).to_dict()],
                streaming_topic="events",
            ),
        )
    assert excinfo.value.type == "agent_framework.StreamError"
    assert excinfo.value.non_retryable is non_retryable
    assert "'model'" in excinfo.value.message and "'events'" in excinfo.value.message
    assert excinfo.value.__cause__ is publish_error
    assert refusing.published
