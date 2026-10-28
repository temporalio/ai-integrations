"""Dispatch one Claude turn from workflow code.

Safe to import from workflow code: this module never imports ``claude_agent_sdk``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.claude_agent_sdk._models import (
    CLAUDE_TURN_CONFIG_ERROR_TYPE,
    RUN_CLAUDE_TURN_ACTIVITY,
    ClaudeTurnInput,
    ClaudeTurnResult,
)
from temporalio.common import RetryPolicy
from temporalio.contrib.workflow_streams._stream import _PUBLISH_SIGNAL
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.workflow import ActivityCancellationType, ActivityConfig

__all__ = ["run_turn"]

DEFAULT_START_TO_CLOSE_TIMEOUT = timedelta(minutes=10)
DEFAULT_HEARTBEAT_TIMEOUT = timedelta(seconds=30)

_FIXED = ("retry_policy", "cancellation_type")


def _require_stream(turn: ClaudeTurnInput) -> None:
    """Fail before the Activity when a topic is set and no ``WorkflowStream`` is hosted."""
    topic = turn.streaming_topic
    if topic is None:
        return
    if not topic.strip():
        raise config_error("streaming_topic must be a non-empty topic name")
    interval = turn.streaming_batch_interval_ms
    if interval is not None and interval < 0:
        raise config_error("streaming_batch_interval_ms cannot be negative")
    if workflow.get_signal_handler(_PUBLISH_SIGNAL) is None:
        raise config_error(
            f"streaming_topic {topic!r} is set but this workflow has no WorkflowStream. "
            "Construct WorkflowStream() in the workflow's @workflow.init "
            "(from temporalio.contrib.workflow_streams)."
        )


def config_error(problem: str) -> ApplicationError:
    """A setup mistake found in workflow code (``CLAUDE_TURN_CONFIG_ERROR_TYPE``). Non-retryable,
    so a message handler that raises it fails its message; a ``TypeError`` or ``ValueError``
    would fail the workflow task instead, and Temporal retries that task forever.
    """
    return ApplicationError(
        problem, type=CLAUDE_TURN_CONFIG_ERROR_TYPE, non_retryable=True
    )


async def run_turn(
    input: ClaudeTurnInput, *, activity_config: ActivityConfig | None = None
) -> ClaudeTurnResult:
    """Run one Claude turn Activity from workflow code and return its result.

    ``activity_config`` is merged over the defaults: a 10 minute start-to-close timeout, a 30
    second heartbeat timeout, and the summary ``"Claude turn (<model>)"``.

    Two options are fixed, and setting either raises a non-retryable ``ClaudeTurnConfigError``:

    * ``retry_policy``: one attempt, because a blind retry would replay model output and tool
      side effects. ``run_claude_turn`` resumes a lost turn by calling this again.
    * ``cancellation_type``: ``WAIT_CANCELLATION_COMPLETED``, so a following turn on the same
      session never overlaps a cancelled one.

    A Claude failure is re-raised as a non-retryable ``ApplicationError`` with the Activity's
    message, type and details. A ``streaming_topic`` requires a ``WorkflowStream`` in
    ``@workflow.init``.
    """
    config = activity_config or ActivityConfig()
    _require_stream(input)
    if fixed := [key for key in _FIXED if key in config]:
        raise config_error(
            f"run_turn fixes {', '.join(fixed)}: a Claude turn runs exactly once and waits for "
            "its cancellation to complete; remove it from activity_config"
        )
    options: dict[str, Any] = {
        "start_to_close_timeout": DEFAULT_START_TO_CLOSE_TIMEOUT,
        "heartbeat_timeout": DEFAULT_HEARTBEAT_TIMEOUT,
        "summary": f"Claude turn ({input.options.model or 'default model'})",
        **config,
    }
    try:
        return await workflow.execute_activity(
            RUN_CLAUDE_TURN_ACTIVITY,
            input,
            result_type=ClaudeTurnResult,
            retry_policy=RetryPolicy(maximum_attempts=1),
            cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            **options,
        )
    except ActivityError as exc:
        # An ActivityError's own text is only "Activity task failed"; surface Claude's.
        # Another ActivityError can sit between this one and that ApplicationError.
        cause: BaseException | None = exc.cause
        while isinstance(cause, ActivityError):
            cause = cause.cause
        if not isinstance(cause, ApplicationError):
            raise
        raise ApplicationError(
            cause.message,
            *cause.details,
            type=cause.type,
            non_retryable=cause.non_retryable,
        ) from exc
