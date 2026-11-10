"""Read what a workflow's history recorded about its activities."""

from __future__ import annotations

from typing import Any

from temporalio.api.enums.v1 import EventType
from temporalio.api.failure.v1 import Failure
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


def _failure_text(failure: Failure) -> list[str]:
    text = [
        failure.message,
        failure.stack_trace,
        failure.encoded_attributes.data.decode(),
    ]
    text += [p.data.decode() for p in failure.application_failure_info.details.payloads]
    text += [
        p.data.decode()
        for p in failure.timeout_failure_info.last_heartbeat_details.payloads
    ]
    if failure.HasField("cause"):
        text += _failure_text(failure.cause)
    return text


async def recorded_output(handle: WorkflowHandle[Any, Any]) -> str:
    """Everything the workers sent back: activity and workflow results and failures."""
    text: list[str] = []
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_completed_event_attributes"):
            result = event.activity_task_completed_event_attributes.result
            text += [p.data.decode() for p in result.payloads]
        elif event.HasField("activity_task_failed_event_attributes"):
            text += _failure_text(event.activity_task_failed_event_attributes.failure)
        elif event.HasField("activity_task_timed_out_event_attributes"):
            text += _failure_text(
                event.activity_task_timed_out_event_attributes.failure
            )
        elif event.HasField("workflow_execution_completed_event_attributes"):
            result = event.workflow_execution_completed_event_attributes.result
            text += [p.data.decode() for p in result.payloads]
        elif event.HasField("workflow_execution_failed_event_attributes"):
            text += _failure_text(
                event.workflow_execution_failed_event_attributes.failure
            )
    return "\n".join(text)


async def activity_failures(handle: WorkflowHandle[Any, Any]) -> list[Failure]:
    """The failure of each activity that failed or timed out, as the server recorded it."""
    failures = []
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_failed_event_attributes"):
            failures.append(event.activity_task_failed_event_attributes.failure)
        elif event.HasField("activity_task_timed_out_event_attributes"):
            failures.append(event.activity_task_timed_out_event_attributes.failure)
    return failures
