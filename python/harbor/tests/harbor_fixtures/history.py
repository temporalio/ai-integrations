"""Read what a workflow's history recorded about its activities."""

from __future__ import annotations

from typing import Any

from temporalio.api.enums.v1 import EventType
from temporalio.client import WorkflowHandle


async def scheduled(handle: WorkflowHandle[Any, Any]) -> list[Any]:
    """Every ActivityTaskScheduled event, in order."""
    return [
        event
        async for event in handle.fetch_history_events()
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]


async def final_attempts(handle: WorkflowHandle[Any, Any]) -> list[int]:
    """The attempt each activity finished on (history keeps only the last)."""
    return [
        event.activity_task_started_event_attributes.attempt
        async for event in handle.fetch_history_events()
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED
    ]


async def summaries(handle: WorkflowHandle[Any, Any], converter: Any) -> list[str]:
    """The summary each scheduled activity carries for the Temporal UI."""
    out = []
    for event in await scheduled(handle):
        payload = event.user_metadata.summary
        out.append(converter.payload_converter.from_payload(payload, str))
    return out
