"""Live output: agent events through Temporal Workflow Streams.

The Workflow publishes what it decides (the prompt, tool calls and their status,
approvals, Continue-As-New, the final answer). The segment Activity publishes the
text Claude writes, one assistant message at a time. Both land on one topic, in order.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any

from temporalio.client import Client
from temporalio.contrib.workflow_streams import WorkflowStreamClient

TOPIC = "claude"
"""The Workflow Streams topic that carries agent events."""

FIELD_LIMIT = 32 * 1024
"""Longest text or tool input (as JSON) an event carries, in characters."""

_EMIT: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "temporalio_claude_agent_sdk_emit", default=None
)


def cap_event(event: dict[str, Any]) -> dict[str, Any]:
    """Cut long fields so one event stays small.

    ``text``, ``result`` and ``error`` keep their first :data:`FIELD_LIMIT`
    characters. A tool ``input`` longer than that as JSON becomes that much of its
    JSON text. Either way the event gets ``truncated: True``. An ``input`` that is
    already text is cut like the other texts, so a capped event stays as it is.

    Args:
        event: The event.

    Returns:
        The event, cut where needed.
    """
    out: dict[str, Any] = {}
    truncated = False
    for key, value in event.items():
        if key in ("text", "result", "error", "input") and isinstance(value, str):
            if len(value) > FIELD_LIMIT:
                value, truncated = value[:FIELD_LIMIT], True
        elif key == "input":
            text = json.dumps(value, ensure_ascii=False, default=str)
            if len(text) > FIELD_LIMIT:
                value, truncated = text[:FIELD_LIMIT], True
        out[key] = value
    if truncated:
        out["truncated"] = True
    return out


def emit(event: dict[str, Any]) -> None:
    """Publish an event from inside a segment Activity.

    Does nothing when the agent's live output is off. The event gets the time it
    happened (``at``) and the segment and attempt that produced it.

    Args:
        event: A JSON-serializable dict with a ``type`` key.
    """
    publish = _EMIT.get()
    if publish is not None:
        publish(cap_event({**event, "at": datetime.now(timezone.utc).isoformat()}))


@contextlib.asynccontextmanager
async def publishing(
    enabled: bool, *, segment: int = 0, attempt: int = 1
) -> AsyncIterator[None]:
    """Route :func:`emit` to the parent Workflow's stream for the duration.

    Leaving the context flushes what was published, so the Workflow has every
    event from a segment before it sees the segment's result.

    Args:
        enabled: Whether the agent's live output is on.
        segment: The segment's index, added to every event.
        attempt: The Activity attempt, added to every event.
    """
    if not enabled:
        yield
        return
    client = WorkflowStreamClient.from_within_activity(
        batch_interval=timedelta(milliseconds=200)
    )
    async with client:
        topic = client.topic(TOPIC)

        def publish(event: dict[str, Any]) -> None:
            topic.publish({**event, "segment": segment, "attempt": attempt})

        token = _EMIT.set(publish)
        try:
            yield
        finally:
            _EMIT.reset(token)


async def follow_agent(
    client: Client,
    workflow_id: str,
    *,
    from_offset: int = 0,
    poll_cooldown: timedelta = timedelta(milliseconds=100),
) -> AsyncIterator[dict[str, Any]]:
    """Yield an agent's live events, following Continue-As-New.

    Each event is a dict with a ``type``, the time it happened (``at``, ISO 8601,
    UTC) and its ``offset`` in the stream. The types: ``prompt``, ``text``,
    ``tool_call``, ``approval_needed``, ``tool_result``, ``retry``,
    ``continued_as_new``, ``done``, ``error`` and ``cancelled``. ``text`` and
    ``retry`` events also carry the ``segment`` and ``attempt`` that produced them:
    after a ``retry`` event, earlier attempts' text of that segment is superseded.

    To resume after a disconnect, pass the last offset you handled plus one. Offsets
    older than the events carried across Continue-As-New are gone; the stream then
    starts at the oldest event it still has. The iteration ends when the Workflow
    closes.

    Args:
        client: A Temporal client.
        workflow_id: The agent Workflow's ID.
        from_offset: The first offset to read.
        poll_cooldown: Minimum time between polls when caught up.
    """
    stream = WorkflowStreamClient.create(client, workflow_id)
    async for item in stream.subscribe(
        TOPIC, from_offset=from_offset, poll_cooldown=poll_cooldown
    ):
        data = item.data
        event = dict(data) if isinstance(data, dict) else {"type": "data", "data": data}
        event["offset"] = item.offset
        yield event
