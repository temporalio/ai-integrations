"""Model input and output that carry ``bytes`` cross the model activity intact.

Strands content blocks hold raw ``bytes`` in image and document sources and in
``reasoningContent.redactedContent``. The JSON payload converter cannot carry
them as-is (non-UTF-8 bytes fail to serialize, ASCII bytes come back as ``str``),
so the activity boundary tags them as base64.
"""

import asyncio
import base64
import json
from collections.abc import AsyncIterable, Callable
from datetime import timedelta
from typing import Any, Literal
from uuid import uuid4

import pytest
from strands.models import Model
from strands.types.content import ContentBlock, Messages
from strands.types.session import decode_bytes_values
from strands.types.streaming import StreamEvent

from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.contrib.workflow_streams import WorkflowStream, WorkflowStreamClient
from temporalio.converter import DataConverter
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents._model_activity import (
    ModelActivity,
    _InvokeModelInput,
    _StreamingInvokeModelInput,
    _topic_payload,
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


def _redacted_events(redacted: bytes) -> list[Any]:
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"redactedContent": redacted}}
            }
        },
        {"contentBlockStop": {}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "ok"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]


REDACTED_EVENTS = _redacted_events(REDACTED)
UTF8_REDACTED = b"redacted-utf8"
UTF8_EVENTS = _redacted_events(UTF8_REDACTED)
TEXT_EVENTS: list[Any] = [
    {"messageStart": {"role": "assistant"}},
    {"contentBlockStart": {"start": {}}},
    {"contentBlockDelta": {"delta": {"text": "ok"}}},
    {"contentBlockStop": {}},
    {"messageStop": {"stopReason": "end_turn"}},
]


class _BytesModel(Model):
    """Records the messages it receives and answers with a ``redactedContent`` block."""

    seen: list[Any] = []
    events: list[Any] = REDACTED_EVENTS

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
        for event in self.events:
            yield event


class _Utf8Model(_BytesModel):
    events = UTF8_EVENTS


class _TextModel(_BytesModel):
    events = TEXT_EVENTS


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
    @workflow.init
    def __init__(self, model: str) -> None:
        self.stream = WorkflowStream()
        self.agent = TemporalAgent(
            model=model,
            start_to_close_timeout=timedelta(seconds=15),
            streaming_topic="events",
        )

    @workflow.run
    async def run(
        self,
        model: str,  # pyright: ignore[reportUnusedParameter]
    ) -> dict[str, Any]:
        await self.agent.invoke_async(PROMPT[:2])
        reply = self.agent.messages[-1]["content"][0]
        redacted = reply.get("reasoningContent", {}).get("redactedContent")
        return {
            "types": _bytes_types(self.agent.messages),
            "redacted_hex": redacted.hex() if redacted is not None else None,
        }


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


# The plugin on the client makes the whole run use the Pydantic data converter;
# the plugin on the worker only leaves the client, and so the activity's
# publisher and the subscriber, on the default converter.
Converter = Literal["pydantic", "default"]
CONVERTERS: list[Converter] = ["pydantic", "default"]


class _Run:
    """What one streaming workflow run produced, as a subscriber and the workflow saw it."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.error: BaseException | None = None
        self.result: dict[str, Any] = {}
        self.history: Any = None


async def _stream_run(
    client: Client,
    converter: Converter,
    model: type[_BytesModel],
    result_type: Any,
    expected_events: int,
) -> _Run:
    """Run a streaming workflow and subscribe to its topic with ``result_type``.

    A subscription that raises (as a validation failure does) is recorded in
    ``error`` rather than lost in the background task.
    """
    task_queue = f"test_model_bytes_streaming-{uuid4()}"
    workflow_id = f"test_model_bytes_streaming_{uuid4()}"
    plugin = StrandsPlugin(models={"bytes": model})
    worker_plugins = []
    if converter == "pydantic":
        config = client.config()
        config["plugins"] = [*config["plugins"], plugin]
        client = Client(**config)
        assert client.data_converter.payload_converter_class is (
            pydantic_data_converter.payload_converter_class
        )
    else:
        worker_plugins.append(plugin)
        assert (
            client.data_converter.payload_converter_class
            is DataConverter.default.payload_converter_class
        )
    run = _Run()

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StreamingBytesWorkflow],
        plugins=worker_plugins,
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StreamingBytesWorkflow.run,
            "bytes",
            id=workflow_id,
            task_queue=task_queue,
        )
        stream = WorkflowStreamClient.create(client, workflow_id)

        async def collect() -> None:
            try:
                async for item in stream.subscribe(
                    ["events"],
                    from_offset=0,
                    result_type=result_type,
                    poll_cooldown=timedelta(milliseconds=50),
                ):
                    run.events.append(item.data)
                    if len(run.events) >= expected_events:
                        break
            except Exception as err:
                run.error = err

        collect_task = asyncio.create_task(collect())
        run.result = await handle.result()
        await asyncio.wait_for(collect_task, timeout=10.0)

    run.history = await handle.fetch_history()
    # One activity ran, and it completed: the first attempt did not fail.
    assert get_activities(run.history) == ["invoke_model_streaming"]
    assert not [
        event
        for event in run.history.events
        if event.HasField("activity_task_failed_event_attributes")
    ]
    await Replayer(
        workflows=[StreamingBytesWorkflow], plugins=[plugin]
    ).replay_workflow(run.history)
    return run


@pytest.mark.parametrize("converter", CONVERTERS)
async def test_stream_event_subscriber_reads_utf8_redacted_content(
    client: Client, converter: Converter
):
    """The README's ``result_type=StreamEvent`` subscription gets the original bytes."""
    run = await _stream_run(
        client, converter, _Utf8Model, StreamEvent, len(UTF8_EVENTS)
    )

    assert run.error is None
    assert run.events == UTF8_EVENTS
    # The workflow got the bytes from the activity result, not from the topic.
    assert run.result["redacted_hex"] == UTF8_REDACTED.hex()


@pytest.mark.parametrize("converter", CONVERTERS)
async def test_stream_event_subscriber_survives_non_utf8_redacted_content(
    client: Client, converter: Converter
):
    """Non-UTF-8 ``redactedContent`` does not fail the activity or end a typed subscription.

    The default converter carries any bytes, so the subscriber gets them. The
    Pydantic converter cannot write them and a ``StreamEvent`` cannot read them,
    so the subscriber gets the event with the bytes emptied, and every event
    after it. The workflow gets the real bytes either way.
    """
    run = await _stream_run(
        client, converter, _BytesModel, StreamEvent, len(REDACTED_EVENTS)
    )

    assert run.error is None
    expected = REDACTED if converter == "default" else b""
    assert run.events == _redacted_events(expected)
    assert run.result["types"] == [["bytes"], ["bytes"]]
    assert run.result["redacted_hex"] == REDACTED.hex()


@pytest.mark.parametrize("converter", CONVERTERS)
async def test_untyped_subscriber_reads_what_it_read_before_the_tagging(
    client: Client, converter: Converter
):
    """Without a type the topic carries the converter's own form, never the tagged dict.

    The default converter writes bytes as a list of ints and the Pydantic
    converter as a UTF-8 string.
    """
    utf8 = await _stream_run(
        client, converter, _Utf8Model, dict[str, Any], len(UTF8_EVENTS)
    )
    other = await _stream_run(
        client, converter, _BytesModel, dict[str, Any], len(REDACTED_EVENTS)
    )

    def redacted(run: _Run) -> Any:
        return run.events[1]["contentBlockDelta"]["delta"]["reasoningContent"][
            "redactedContent"
        ]

    assert utf8.error is None and other.error is None
    assert len(utf8.events) == len(UTF8_EVENTS)
    assert len(other.events) == len(REDACTED_EVENTS)
    if converter == "default":
        assert redacted(utf8) == list(UTF8_REDACTED)
        assert redacted(other) == list(REDACTED)
    else:
        assert redacted(utf8) == UTF8_REDACTED.decode()
        assert redacted(other) == ""
    # Everything that is not the bytes field is the same on both.
    assert utf8.events[0] == UTF8_EVENTS[0]
    assert utf8.events[2:] == UTF8_EVENTS[2:]


@pytest.mark.parametrize("converter", CONVERTERS)
@pytest.mark.parametrize("result_type", [StreamEvent, dict[str, Any], None])
async def test_events_without_bytes_are_published_unchanged(
    client: Client, converter: Converter, result_type: Any
):
    run = await _stream_run(
        client, converter, _TextModel, result_type, len(TEXT_EVENTS)
    )

    assert run.error is None
    assert run.events == TEXT_EVENTS
    assert run.result["redacted_hex"] is None


@pytest.mark.parametrize(
    "converter_factory",
    [lambda: DataConverter.default, lambda: pydantic_data_converter],
    ids=["default", "pydantic"],
)
def test_topic_payload_matches_the_converter_for_events_it_can_carry(
    converter_factory: Callable[[], DataConverter],
):
    converter = converter_factory().payload_converter
    event = {"contentBlockDelta": {"delta": {"text": "hi"}}}

    assert _topic_payload(event, converter) == converter.to_payload(event)  # type: ignore[arg-type]


def test_topic_payload_does_not_hide_other_encoding_errors():
    """Only ``bytes`` are emptied; an event the converter cannot write still raises."""
    with pytest.raises(TypeError):
        _topic_payload(
            {"contentBlockDelta": {"delta": {"text": object()}}},  # type: ignore[typeddict-item]
            DataConverter.default.payload_converter,
        )
