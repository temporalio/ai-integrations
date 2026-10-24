"""Model input and output that carry ``bytes`` cross the model activity intact.

Strands content blocks hold raw ``bytes`` in image and document sources and in
``reasoningContent.redactedContent``. The JSON payload converter cannot carry
them as-is (non-UTF-8 bytes fail to serialize, ASCII bytes come back as ``str``),
so the activity boundary tags them as base64.
"""

import asyncio
import base64
import json
from collections.abc import AsyncIterable
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from strands.models import Model
from strands.types.content import ContentBlock, Messages
from strands.types.session import decode_bytes_values
from strands.types.streaming import StreamEvent

from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamClient
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents._model_activity import (
    ModelActivity,
    _InvokeModelInput,
    _StreamingInvokeModelInput,
)
from temporalio.strands_agents._plugin import _data_converter
from temporalio.testing import ActivityEnvironment
from temporalio.worker import Replayer, Worker
from tests.common import get_activities

PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
CSV = b"a,b\n1,2\n"
REDACTED = b"\xff\x00\xfe-redacted"

IMAGE_BLOCK: ContentBlock = {"image": {"format": "png", "source": {"bytes": PNG}}}
DOCUMENT_BLOCK: ContentBlock = {
    "document": {"format": "csv", "name": "data", "source": {"bytes": CSV}}
}
PROMPT: list[ContentBlock] = [{"text": "describe"}, IMAGE_BLOCK, DOCUMENT_BLOCK]
REDACTED_BLOCK: ContentBlock = {"reasoningContent": {"redactedContent": REDACTED}}

REDACTED_EVENTS: list[Any] = [
    {"messageStart": {"role": "assistant"}},
    {
        "contentBlockDelta": {
            "delta": {"reasoningContent": {"redactedContent": REDACTED}}
        }
    },
    {"contentBlockStop": {}},
    {"contentBlockStart": {"start": {}}},
    {"contentBlockDelta": {"delta": {"text": "ok"}}},
    {"contentBlockStop": {}},
    {"messageStop": {"stopReason": "end_turn"}},
]


class _BytesModel(Model):
    """Records the messages it receives and answers with a ``redactedContent`` block."""

    seen: list[Any] = []

    def update_config(self, **_model_config: Any) -> None:
        return None

    def get_config(self) -> dict[str, Any]:
        return {}

    def structured_output(self, *_args: Any, **_kwargs: Any):
        raise NotImplementedError

    async def stream(
        self, messages: Any, *_args: Any, **_kwargs: Any
    ) -> AsyncIterable[StreamEvent]:
        _BytesModel.seen.append(messages)
        for event in REDACTED_EVENTS:
            yield event


def _bytes_types(messages: Any) -> list[list[str]]:
    """The Python type of each bytes value per message, so a workflow can return it as JSON."""
    summary: list[list[str]] = []
    for message in messages:
        types: list[str] = []
        for block in message["content"]:
            if "image" in block:
                types.append(type(block["image"]["source"]["bytes"]).__name__)
            elif "document" in block:
                types.append(type(block["document"]["source"]["bytes"]).__name__)
            elif "reasoningContent" in block:
                types.append(
                    type(block["reasoningContent"]["redactedContent"]).__name__
                )
        summary.append(types)
    return summary


@workflow.defn
class BytesWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="bytes",
            start_to_close_timeout=timedelta(seconds=15),
        )

    @workflow.run
    async def run(self) -> list[list[list[str]]]:
        # Turn one sends an image and a document; the reply holds redactedContent.
        await self.agent.invoke_async(PROMPT)
        after_turn_one = _bytes_types(self.agent.messages)
        # Turn two sends the whole history back, bytes included.
        await self.agent.invoke_async("and now?")
        return [after_turn_one, _bytes_types(self.agent.messages)]


@workflow.defn
class StreamingBytesWorkflow:
    def __init__(self) -> None:
        self.stream = WorkflowStream()
        self.agent = TemporalAgent(
            model="bytes",
            start_to_close_timeout=timedelta(seconds=15),
            streaming_topic="events",
        )

    @workflow.run
    async def run(self) -> list[list[str]]:
        await self.agent.invoke_async(PROMPT[:2])
        return _bytes_types(self.agent.messages)


@pytest.mark.parametrize(
    "block",
    [IMAGE_BLOCK, DOCUMENT_BLOCK, REDACTED_BLOCK],
    ids=["image", "document", "redactedContent"],
)
async def test_bytes_in_activity_input_reach_the_model(block: ContentBlock):
    """A message with bytes reaches the model unchanged, through the plugin's converter."""
    converter = _data_converter(None)
    messages = [{"role": "user", "content": [{"text": "hi"}, block]}]
    seen: list[Any] = []

    class Recorder(_BytesModel):
        async def stream(self, messages: Any, *_args: Any, **_kwargs: Any):  # type: ignore[override]
            seen.append(messages)
            yield {"messageStart": {"role": "assistant"}}

    # The workflow encodes the activity input; the worker decodes it and runs the model.
    payloads = await converter.encode([_InvokeModelInput("m", messages)])
    [received] = await converter.decode(payloads, [_InvokeModelInput])
    activity = ModelActivity({"m": Recorder})
    await ActivityEnvironment().run(activity.invoke_model, received)

    # ``==`` tells ``bytes`` and ``str`` apart, so this also fails when ASCII bytes
    # come back as text.
    assert seen == [messages]


async def test_bytes_in_activity_result_can_be_returned():
    """Stream events with ``redactedContent`` encode and decode through the converter."""
    converter = _data_converter(None)
    activity = ModelActivity({"m": _BytesModel})

    returned = await ActivityEnvironment().run(
        activity.invoke_model, _InvokeModelInput("m", [])
    )
    payloads = await converter.encode([returned])
    [decoded] = await converter.decode(payloads, [list[dict[str, Any]]])

    assert decode_bytes_values(decoded) == REDACTED_EVENTS


async def test_messages_without_bytes_are_unchanged():
    """Text-only messages encode to the same JSON as before bytes were supported."""
    converter = _data_converter(None)
    messages = [
        {"role": "user", "content": [{"text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "t", "name": "n", "input": {"a": [1, "b"]}}}
            ],
        },
    ]

    activity_input = _InvokeModelInput("m", messages)
    [payload] = await converter.encode([activity_input])

    assert activity_input.messages == messages
    assert json.loads(payload.data)["messages"] == messages


async def test_encoding_leaves_the_callers_messages_alone():
    """The agent's own history keeps its bytes; only the activity input is encoded."""
    messages: Messages = [{"role": "user", "content": [IMAGE_BLOCK]}]

    activity_input = _InvokeModelInput("m", messages)

    assert messages[0]["content"][0] == IMAGE_BLOCK
    assert activity_input.messages[0]["content"][0]["image"]["source"]["bytes"] == {
        "__bytes_encoded__": True,
        "data": base64.b64encode(PNG).decode(),
    }


async def test_streaming_input_carries_bytes():
    """The streaming activity input encodes messages the same way."""
    converter = _data_converter(None)
    messages = [{"role": "user", "content": [IMAGE_BLOCK, DOCUMENT_BLOCK]}]

    payloads = await converter.encode([_StreamingInvokeModelInput("m", messages)])
    [received] = await converter.decode(payloads, [_StreamingInvokeModelInput])

    assert decode_bytes_values(received.messages) == messages


async def test_bytes_across_a_workflow(client: Client):
    """An image and a document go in and ``redactedContent`` comes out, over two turns."""
    task_queue = f"test_model_bytes-{uuid4()}"
    plugin = StrandsPlugin(models={"bytes": _BytesModel})
    _BytesModel.seen.clear()

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[BytesWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            BytesWorkflow.run,
            id=f"test_model_bytes_{uuid4()}",
            task_queue=task_queue,
        )
        summary = await handle.result()

    # The workflow holds real bytes, not text or tagged dicts, in the history it
    # built from the prompt and from the model's reply.
    assert summary == [
        [["bytes", "bytes"], ["bytes"]],
        [["bytes", "bytes"], ["bytes"], [], ["bytes"]],
    ]
    # The model received the same bytes on both turns.
    first, second = _BytesModel.seen
    assert first[0]["content"][1:] == [IMAGE_BLOCK, DOCUMENT_BLOCK]
    assert second[0]["content"][1:] == [IMAGE_BLOCK, DOCUMENT_BLOCK]
    assert second[1]["content"][0] == REDACTED_BLOCK

    history = await handle.fetch_history()
    assert get_activities(history) == ["invoke_model", "invoke_model"]

    await Replayer(
        workflows=[BytesWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


async def test_bytes_across_a_streaming_workflow(client: Client):
    """Subscribers get the base64-tagged events; the workflow still gets bytes."""
    task_queue = f"test_model_bytes_streaming-{uuid4()}"
    workflow_id = f"test_model_bytes_streaming_{uuid4()}"
    plugin = StrandsPlugin(models={"bytes": _BytesModel})
    _BytesModel.seen.clear()

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StreamingBytesWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StreamingBytesWorkflow.run,
            id=workflow_id,
            task_queue=task_queue,
        )

        stream = WorkflowStreamClient.create(client, workflow_id)
        events: list[Any] = []

        async def collect() -> None:
            async for item in stream.subscribe(
                ["events"],
                from_offset=0,
                result_type=dict[str, Any],
                poll_cooldown=timedelta(milliseconds=50),
            ):
                events.append(item.data)
                if len(events) >= len(REDACTED_EVENTS):
                    break

        collect_task = asyncio.create_task(collect())
        summary = await handle.result()
        await asyncio.wait_for(collect_task, timeout=10.0)

    assert summary == [["bytes"], ["bytes"]]
    assert _BytesModel.seen[0][0]["content"][1] == IMAGE_BLOCK
    # Events on the topic are JSON, so ``bytes`` appear in their tagged form.
    assert decode_bytes_values(events) == REDACTED_EVENTS
    assert events[1]["contentBlockDelta"]["delta"]["reasoningContent"][
        "redactedContent"
    ] == {"__bytes_encoded__": True, "data": base64.b64encode(REDACTED).decode()}

    history = await handle.fetch_history()
    assert get_activities(history) == ["invoke_model_streaming"]

    await Replayer(
        workflows=[StreamingBytesWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)
