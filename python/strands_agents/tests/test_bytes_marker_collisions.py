"""Ordinary JSON that looks like Strands' encoded-bytes marker is left alone.

Strands tags ``bytes`` as ``{"__bytes_encoded__": True, "data": "<base64>"}``. A
tool input, tool result or MCP payload is arbitrary JSON, so it can contain a
dictionary of the same shape. Only the fields Strands types as ``bytes`` may be
decoded; everything else must reach the model and the agent exactly as sent.
"""

import sys
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from mcp import StdioServerParameters, stdio_client
from strands import tool
from strands.models import Model
from strands.tools.mcp.mcp_client import MCPClient

from temporalio import workflow
from temporalio.client import Client
from temporalio.strands_agents import StrandsPlugin, TemporalAgent, TemporalMCPClient
from temporalio.strands_agents._bytes_values import decode_event
from temporalio.strands_agents._model_activity import (
    ModelActivity,
    _InvokeModelInput,
    _stream,
    _StreamingInvokeModelInput,
)
from temporalio.strands_agents._plugin import _data_converter
from temporalio.testing import ActivityEnvironment
from temporalio.worker import Replayer, Worker
from tests.common import get_activities
from tests.mock_model import MockModel

PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
REDACTED = b"\xff\x00\xfe-redacted"

# JSON that is not bytes but has the marker's keys.
LOOKALIKES: dict[str, Any] = {
    "extra_keys": {"__bytes_encoded__": True, "data": "SGVsbG8=", "other": "retained"},
    "exact_marker": {"__bytes_encoded__": True, "data": "SGVsbG8="},
    "non_string_data": {"__bytes_encoded__": True, "data": 7},
    "invalid_base64": {"__bytes_encoded__": True, "data": "not base64!"},
    "nested": {"outer": [{"__bytes_encoded__": True, "data": "SGVsbG8=", "k": 1}]},
}


def _messages_with(lookalike: Any) -> list[Any]:
    """A conversation whose tool input and tool result hold ``lookalike``, plus real bytes."""
    return [
        {"role": "user", "content": [{"text": "go"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "t1", "name": "n", "input": lookalike}},
                {"reasoningContent": {"redactedContent": REDACTED}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "status": "success",
                        "content": [
                            {"json": lookalike},
                            {"image": {"format": "png", "source": {"bytes": PNG}}},
                        ],
                    }
                }
            ],
        },
    ]


class _Recorder(Model):
    """Records the messages and answers with events that hold ordinary JSON."""

    seen: list[Any] = []
    events: list[Any] = []

    def update_config(self, **_model_config: Any) -> None:
        return None

    def get_config(self) -> dict[str, Any]:
        return {}

    def structured_output(self, *_args: Any, **_kwargs: Any):
        raise NotImplementedError

    async def stream(self, messages: Any, *_args: Any, **_kwargs: Any):  # type: ignore[no-untyped-def, override]
        _Recorder.seen.append(messages)
        for event in _Recorder.events:
            yield event


@pytest.mark.parametrize("lookalike", LOOKALIKES.values(), ids=LOOKALIKES.keys())
@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "streaming"])
async def test_lookalike_json_in_activity_input_reaches_the_model(
    lookalike: Any, streaming: bool
):
    """Tool input and tool result JSON that matches the marker is not turned into bytes."""
    converter = _data_converter(None)
    messages = _messages_with(lookalike)
    _Recorder.seen.clear()
    _Recorder.events = [{"messageStart": {"role": "assistant"}}]

    input_class = _StreamingInvokeModelInput if streaming else _InvokeModelInput
    payloads = await converter.encode([input_class("m", messages)])
    [received] = await converter.decode(payloads, [input_class])
    if streaming:
        # The streaming activity needs a server to publish to; both activities
        # hand the model whatever ``_stream`` yields, so drive that directly.
        async for _ in _stream(_Recorder(), received):
            pass
    else:
        await ActivityEnvironment().run(
            ModelActivity({"m": _Recorder}).invoke_model, received
        )

    # ``==`` tells ``bytes`` and ``dict`` apart; the real bytes still arrive as bytes.
    assert _Recorder.seen == [messages]


@pytest.mark.parametrize("lookalike", LOOKALIKES.values(), ids=LOOKALIKES.keys())
async def test_lookalike_json_in_events_is_unchanged(lookalike: Any):
    """Events are decoded only at ``redactedContent``; other JSON passes through."""
    converter = _data_converter(None)
    _Recorder.seen.clear()
    _Recorder.events = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"redactedContent": REDACTED}}
            }
        },
        {"metadata": {"trace": lookalike}},
    ]

    returned = await ActivityEnvironment().run(
        ModelActivity({"m": _Recorder}).invoke_model, _InvokeModelInput("m", [])
    )
    payloads = await converter.encode([returned])
    [decoded] = await converter.decode(payloads, [list[dict[str, Any]]])

    assert [decode_event(event) for event in decoded] == _Recorder.events


async def test_decode_event_accepts_a_missing_or_null_redacted_content():
    """Events without a ``redactedContent`` value are returned as they are."""
    events: list[Any] = [
        {"contentBlockDelta": {"delta": {"text": "hi"}}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "t"}}}},
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"redactedContent": None}}
            }
        },
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    assert [decode_event(event) for event in events] == events


# Async because Strands runs a sync tool on a thread, which a workflow cannot do.
@tool
async def lookup_settings(
    outer: list[Any],  # pyright: ignore[reportUnusedParameter]
) -> dict[str, Any]:
    """Return JSON that has the shape of an encoded-bytes marker."""
    return {
        "toolUseId": "unused",
        "status": "success",
        "content": [{"json": LOOKALIKES["extra_keys"]}],
    }


@workflow.defn
class LookalikeToolWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="recorder",
            start_to_close_timeout=timedelta(seconds=15),
            tools=[lookup_settings],
        )

    @workflow.run
    async def run(self) -> list[Any]:
        await self.agent.invoke_async("go")
        return self.agent.messages


async def test_lookalike_json_across_a_workflow(client: Client):
    """Tool input and tool result JSON survive a model round trip, and replay."""
    task_queue = f"test_bytes_marker_collisions-{uuid4()}"
    seen: list[Any] = []
    script: list[str | dict[str, Any]] = [
        {"name": "lookup_settings", "input": LOOKALIKES["nested"]},
        "Done!",
    ]

    class RecordingModel(MockModel):
        async def stream(self, messages, *args, **kwargs):  # type: ignore[no-untyped-def, override]
            seen.append(messages)
            async for event in super().stream(messages, *args, **kwargs):
                yield event

    plugin = StrandsPlugin(models={"recorder": lambda: RecordingModel(script)})
    config = client.config()
    config["plugins"] = [*config["plugins"], plugin]
    client = Client(**config)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[LookalikeToolWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            LookalikeToolWorkflow.run,
            id=f"test_bytes_marker_collisions_{uuid4()}",
            task_queue=task_queue,
        )
        messages = await handle.result()

    tool_use = messages[1]["content"][0]["toolUse"]
    tool_result = messages[2]["content"][0]["toolResult"]
    assert tool_use["input"] == LOOKALIKES["nested"]
    assert tool_result["content"] == [{"json": LOOKALIKES["extra_keys"]}]
    # The second model call received the same JSON the workflow holds.
    assert seen[1][1]["content"][0]["toolUse"]["input"] == LOOKALIKES["nested"]
    assert seen[1][2]["content"][0]["toolResult"]["content"] == [
        {"json": LOOKALIKES["extra_keys"]}
    ]

    history = await handle.fetch_history()
    assert get_activities(history) == ["invoke_model", "invoke_model"]
    await Replayer(workflows=[LookalikeToolWorkflow], plugins=[plugin]).replay_workflow(
        history
    )


@workflow.defn
class MCPLookalikeWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[
                TemporalMCPClient(
                    server="marker",
                    cache_tools=True,
                    start_to_close_timeout=timedelta(seconds=30),
                )
            ],
        )

    @workflow.run
    async def run(self) -> dict[str, Any]:
        await self.agent.invoke_async("look it up")
        [result] = [
            block["toolResult"]
            for message in self.agent.messages
            for block in message["content"]
            if "toolResult" in block
        ]
        # JSON cannot carry the image bytes; report the type instead.
        return {
            **result,
            "content": [
                {
                    "image_bytes_type": type(
                        item["image"]["source"].get("bytes")
                    ).__name__
                }
                if "image" in item
                else item
                for item in result["content"]
            ],
        }


async def test_mcp_structured_content_and_metadata_that_look_like_bytes(
    client: Client,
):
    """MCP structured content and metadata stay JSON; the image still arrives as bytes."""
    task_queue = f"test_bytes_marker_collisions_mcp-{uuid4()}"
    model_saw: list[Any] = []

    class RecordingModel(MockModel):
        async def stream(self, messages, *args, **kwargs):  # type: ignore[no-untyped-def, override]
            for message in messages:
                for block in message["content"]:
                    if "toolResult" in block:
                        model_saw.append(block["toolResult"])
            async for event in super().stream(messages, *args, **kwargs):
                yield event

    script: list[str | dict[str, Any]] = [{"name": "lookup", "input": {}}, "Done!"]
    plugin = StrandsPlugin(
        models={"mock": lambda: RecordingModel(script)},
        mcp_clients={
            "marker": lambda: MCPClient(
                lambda: stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=[str(Path(__file__).parent / "marker_mcp_server.py")],
                    )
                )
            ),
        },
    )
    config = client.config()
    config["plugins"] = [*config["plugins"], plugin]
    client = Client(**config)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPLookalikeWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MCPLookalikeWorkflow.run,
            id=f"test_bytes_marker_collisions_mcp_{uuid4()}",
            task_queue=task_queue,
        )
        result = await handle.result()

    expected_structured = {
        "__bytes_encoded__": True,
        "data": "SGVsbG8=",
        "other": "retained",
    }
    expected_meta = {"__bytes_encoded__": True, "data": 7, "nested": {"data": "x"}}
    assert result["structuredContent"] == expected_structured
    assert result["metadata"] == expected_meta
    assert result["content"] == [
        {"text": "found"},
        {"image_bytes_type": "bytes"},
    ]
    # The model received the same result, with the PNG as real bytes.
    [saw] = model_saw
    assert saw["structuredContent"] == expected_structured
    assert saw["content"][1] == {"image": {"format": "png", "source": {"bytes": PNG}}}

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "marker-list-tools",
        "invoke_model",
        "marker-call-tool",
        "invoke_model",
    ]
    await Replayer(workflows=[MCPLookalikeWorkflow], plugins=[plugin]).replay_workflow(
        history
    )
