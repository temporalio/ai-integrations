import json
import zlib
from collections.abc import Sequence
from typing import Any

from temporalio.api.common.v1 import Payload
from temporalio.api.enums.v1 import EventType
from temporalio.client import WorkflowHistory
from temporalio.converter import PayloadCodec


def get_activities(history: WorkflowHistory) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]


class CompressionCodec(PayloadCodec):
    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload(
                metadata={"encoding": b"binary/zlib"},
                data=zlib.compress(payload.SerializeToString()),
            )
            for payload in payloads
        ]

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload.FromString(zlib.decompress(payload.data)) for payload in payloads
        ]


def get_tool_results(history: WorkflowHistory) -> list[Any]:
    """The content of each tool result the last model call got."""
    model_input = json.loads(
        [
            event.activity_task_scheduled_event_attributes.input.payloads[0].data
            for event in history.events
            if event.activity_task_scheduled_event_attributes.activity_type.name
            == "invoke_model"
        ][-1]
    )
    return [
        content["toolResult"]["content"]
        for message in model_input["messages"]
        for content in message["content"]
        if "toolResult" in content
    ]
