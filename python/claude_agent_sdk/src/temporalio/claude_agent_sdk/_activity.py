"""Activity side: runs one model segment through a pluggable runner."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

from temporalio import activity
from temporalio.exceptions import ApplicationError

from ._events import emit, publishing
from ._models import (
    TOOL_CALL_NOT_RUN,
    TRY_AGAIN,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolStepInput,
)
from ._workflow import SEGMENT_ACTIVITY_NAME, TOOL_STEP_ACTIVITY_NAME


class SegmentRunner(Protocol):
    """Runs one model segment. Implemented by ``ClaudeAgentSdkRunner`` and ``ScriptedClaude``.

    The contract that makes retries safe: every segment that is not an error returns a
    ``checkpoint``, and the next segment of the session receives it. A segment that
    runs again (``attempt`` > 1, or ``inp.fork``) must continue from
    ``inp.checkpoint`` and ignore anything an unfinished attempt wrote after it.

    A runner that lets the Workflow hold the conversation reads it from
    ``inp.conversation`` (or ``inp.transcript``) and returns what changed in
    ``transcript_keep`` and ``transcript_add``; one that keeps it elsewhere returns
    ``transcript_keep=None``. Both report ``external_storage``, so the Workflow knows
    whether a large state can move to a new run.
    """

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """Run one segment.

        Args:
            inp: What to run.
            attempt: The Activity attempt number, starting at 1.

        Returns:
            Where the segment stopped: a pause at a tool call, or a final answer.
        """
        ...


def make_segment_activity(
    runner: SegmentRunner, *, heartbeat_every: float = 5.0
) -> Callable[..., Any]:
    """Build the segment Activity around a runner.

    Args:
        runner: Runs the segment.
        heartbeat_every: Seconds between heartbeats while the segment runs (more often
            when the segment's heartbeat timeout needs it: at least three per timeout).

    Returns:
        The Activity function, named ``run_claude_segment``.
    """

    @activity.defn(name=SEGMENT_ACTIVITY_NAME)
    async def run_claude_segment(inp: SegmentInput) -> SegmentOutput:
        info = activity.info()
        every = heartbeat_every
        if info.heartbeat_timeout:
            every = min(every, info.heartbeat_timeout.total_seconds() / 3)

        async def beat() -> None:
            while True:
                activity.heartbeat(inp.segment_index)
                await asyncio.sleep(every)

        attempt = info.attempt
        beater = asyncio.create_task(beat())
        try:
            async with publishing(
                inp.live_output, segment=inp.segment_index, attempt=attempt
            ):
                if attempt > 1:
                    emit({"type": "retry"})  # this segment's earlier text is superseded
                return await runner.run(inp, attempt)
        finally:
            beater.cancel()

    return run_claude_segment


def make_tool_step_activity(
    runner: Any, *, heartbeat_every: float = 5.0
) -> Callable[..., Any]:
    """Build the tool step Activity around a runner that has ``run_tool_step``.

    ``run_tool_step(step, attempt)`` returns the call's result. A step that fails says
    on which side of the call's start it failed: an ``ApplicationError`` of type
    ``TOOL_CALL_NOT_RUN`` means the call did not run, so the Workflow may try again
    (as a new Activity); any other failure means it may have run
    (``TOOL_CALL_INTERRUPTED`` says so), so it is not run again unless the tool is in
    ``repeatable_tools``. The Activity adds to a ``TOOL_CALL_NOT_RUN`` failure whether
    the call may be tried again (``step.retry``), so the Workflow's decision is in
    the history.

    Args:
        runner: Runs the tool step (``ClaudeAgentSdkRunner``, ``ScriptedClaude``).
        heartbeat_every: Seconds between heartbeats while the step runs (more often
            when the step's heartbeat timeout needs it: at least three per timeout).

    Returns:
        The Activity function, named ``run_claude_tool_step``.
    """

    @activity.defn(name=TOOL_STEP_ACTIVITY_NAME)
    async def run_claude_tool_step(step: ToolStepInput) -> ToolOutcome:
        info = activity.info()
        every = heartbeat_every
        if info.heartbeat_timeout:
            every = min(every, info.heartbeat_timeout.total_seconds() / 3)

        async def beat() -> None:
            while True:
                activity.heartbeat(step.call.id)
                await asyncio.sleep(every)

        beater = asyncio.create_task(beat())
        try:
            return await runner.run_tool_step(step, info.attempt)
        except ApplicationError as err:
            if step.retry is None or err.type != TOOL_CALL_NOT_RUN:
                raise
            raise ApplicationError(
                err.message,
                {TRY_AGAIN: _may_try_again(step, err)},  # first: the Workflow reads it
                *err.details,
                type=TOOL_CALL_NOT_RUN,
                non_retryable=err.non_retryable,
                next_retry_delay=err.next_retry_delay,
                category=err.category,
            ) from err.__cause__
        finally:
            beater.cancel()

    return run_claude_tool_step


def _may_try_again(step: ToolStepInput, err: ApplicationError) -> bool:
    """Whether a step whose call did not run may be tried again (``step.retry``).

    Not when the step's error is non-retryable, when its type or that of the error
    behind it (as Temporal names it: an ``ApplicationError``'s type, or the class
    name) is one of the non-retryable types, or when no attempt is left.
    """
    retry = step.retry
    if retry is None or err.non_retryable:
        return False
    behind = err.__cause__
    types = {TOOL_CALL_NOT_RUN}
    if isinstance(behind, ApplicationError):
        types.add(behind.type or "")
    elif behind is not None:
        types.add(type(behind).__name__)
    if types & set(retry.non_retryable_error_types):
        return False
    return not 0 < retry.maximum_attempts <= step.attempt
