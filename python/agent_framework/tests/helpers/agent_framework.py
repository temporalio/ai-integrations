"""Shared helpers for the Agent Framework plugin test suite."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any

from agent_framework import ChatResponse, ChatResponseUpdate, Message, ResponseStream

from temporalio.agent_framework import AgentFrameworkPlugin
from temporalio.agent_framework.testing import FakeChatClient, ScriptEntry
from temporalio.api.enums.v1 import EventType
from temporalio.api.history.v1 import (
    ActivityTaskScheduledEventAttributes,
    ActivityTaskStartedEventAttributes,
    HistoryEvent,
)
from temporalio.client import Client, WorkflowHandle
from temporalio.worker import Worker

INVOKE_MODEL = "agent_framework.invoke_model"
INVOKE_MODEL_STREAMING = "agent_framework.invoke_model_streaming"


def workflow_id(prefix: str) -> str:
    """Return a unique workflow id for one test run."""
    return f"{prefix}-{uuid.uuid4()}"


def new_worker(
    client: Client,
    *workflows: type,
    plugin: AgentFrameworkPlugin,
    activities: Sequence[Any] = (),
    task_queue: str | None = None,
    **kwargs: Any,
) -> Worker:
    """Return a worker on a fresh task queue that replays every workflow task from history."""
    return Worker(
        client,
        task_queue=task_queue or f"agent-framework-{uuid.uuid4()}",
        workflows=list(workflows),
        activities=list(activities),
        plugins=[plugin],
        max_cached_workflows=0,
        **kwargs,
    )


async def scheduled_activity_events(handle: WorkflowHandle) -> list[HistoryEvent]:
    """Return the ``ActivityTaskScheduled`` events of a run, in order."""
    events: list[HistoryEvent] = []
    async for event in handle.fetch_history_events():
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            events.append(event)
    return events


async def scheduled_activities(
    handle: WorkflowHandle,
) -> list[ActivityTaskScheduledEventAttributes]:
    """Return the ``ActivityTaskScheduled`` attributes of a run, in order."""
    return [
        event.activity_task_scheduled_event_attributes
        for event in await scheduled_activity_events(handle)
    ]


async def started_activities(
    handle: WorkflowHandle,
) -> list[ActivityTaskStartedEventAttributes]:
    """Return the ``ActivityTaskStarted`` attributes of a run, in order."""
    attrs: list[ActivityTaskStartedEventAttributes] = []
    async for event in handle.fetch_history_events():
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            attrs.append(event.activity_task_started_event_attributes)
    return attrs


async def count_scheduled_activities(handle: WorkflowHandle) -> Counter[str]:
    """Count ``ActivityTaskScheduled`` events by activity type name."""
    counts: Counter[str] = Counter()
    for attrs in await scheduled_activities(handle):
        counts[attrs.activity_type.name] += 1
    return counts


async def count_signals(handle: WorkflowHandle) -> int:
    """Count ``WorkflowExecutionSignaled`` events of a run."""
    count = 0
    async for event in handle.fetch_history_events():
        if event.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_SIGNALED:
            count += 1
    return count


async def activity_results(handle: WorkflowHandle) -> list[Any]:
    """Return the decoded result of every completed activity, in order."""
    results: list[Any] = []
    async for event in handle.fetch_history_events():
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED:
            payloads = event.activity_task_completed_event_attributes.result.payloads
            results.append(json.loads(payloads[0].data) if payloads else None)
    return results


def summary_of(event: HistoryEvent) -> str | None:
    """Return the ``summary`` an ``ActivityTaskScheduled`` event carries."""
    if not event.HasField("user_metadata") or not event.user_metadata.HasField(
        "summary"
    ):
        return None
    return json.loads(event.user_metadata.summary.data)


def role_of(message: Message) -> str:
    """Return a message's role as a plain string."""
    role = message.role
    return str(getattr(role, "value", role))


def contents_summary(response: Any) -> list[str]:
    """Flatten a response's contents into ``kind:detail`` strings."""
    out: list[str] = []
    for message in response.messages:
        for content in message.contents:
            if content.type == "text":
                out.append(f"text:{content.text}")
            elif content.type == "function_call":
                out.append(f"call:{content.name}")
            elif content.type == "function_result":
                out.append(f"result:{_result_text(content)}")
            elif content.type == "function_approval_request":
                out.append(f"approval:{content.function_call.name}")
            else:
                out.append(content.type)
    return out


def _result_text(content: Any) -> str:
    result = content.result
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(getattr(item, "text", "") or "" for item in result)
    return json.dumps(result, default=str)


class SlowFakeChatClient(FakeChatClient):
    """A ``FakeChatClient`` that takes ``delay`` seconds per call, or per streamed update."""

    def __init__(self, responses: Sequence[ScriptEntry], *, delay: float) -> None:
        super().__init__(responses)
        self.delay = delay

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        inner = super()._inner_get_response(
            messages=messages, stream=stream, options=options, **kwargs
        )
        if isinstance(inner, ResponseStream):

            async def paced() -> Any:
                async for update in inner:
                    await asyncio.sleep(self.delay)
                    yield update

            return ResponseStream(
                paced(),
                finalizer=lambda updates: ChatResponse.from_updates(
                    updates, output_format_type=options.get("response_format")
                ),
            )

        async def slow() -> ChatResponse:
            try:
                await asyncio.sleep(self.delay)
            except BaseException:
                # A cancelled call never awaits the scripted response; close it
                # so Python does not warn about an un-awaited coroutine.
                close = getattr(inner, "close", None)
                if close is not None:
                    close()
                raise
            return await inner

        return slow()


class StatusError(Exception):
    """A provider SDK error carrying an HTTP status and response headers."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = _Response(headers or {})


class _Response:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
