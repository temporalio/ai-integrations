"""Workflow-side helpers: the plugin's failure type and ``activity_as_tool``."""

from __future__ import annotations

import asyncio
import inspect
import typing
from collections.abc import Callable
from datetime import timedelta
from typing import Any, ClassVar, Literal

from agent_framework import FunctionTool

from temporalio import activity, workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    is_cancelled_exception,
)
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityConfig,
    VersioningIntent,
)

ApprovalMode = Literal["always_require", "never_require"]
"""The approval modes Agent Framework accepts on a tool."""


class AgentFrameworkWorkflowError(ApplicationError):
    """A configuration mistake the plugin finds in workflow code.

    Raised for a ``TemporalChatClient`` without a model name, an option or
    tool that cannot be recorded in workflow history, and an MCP tool whose
    name the agent already uses. It is a non-retryable ``ApplicationError``
    of type ``AgentFrameworkWorkflowError``, registered as a workflow failure
    type, so it fails the workflow instead of retrying the workflow task.
    Match it by ``ApplicationError.type``.
    """

    TYPE: ClassVar[str] = "AgentFrameworkWorkflowError"

    def __init__(self, message: str, *details: Any, non_retryable: bool = True) -> None:
        """Create the error with ``message`` and optional serializable details."""
        super().__init__(message, *details, type=self.TYPE, non_retryable=non_retryable)


def activity_as_tool(
    activity_fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    approval_mode: ApprovalMode | None = None,
    task_queue: str | None = None,
    schedule_to_close_timeout: timedelta | None = None,
    schedule_to_start_timeout: timedelta | None = None,
    start_to_close_timeout: timedelta | None = None,
    heartbeat_timeout: timedelta | None = None,
    retry_policy: RetryPolicy | None = None,
    cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL,
    activity_id: str | None = None,
    versioning_intent: VersioningIntent | None = None,
    summary: str | None = None,
    priority: Priority = Priority.default,
) -> FunctionTool:
    """Wrap a Temporal activity as an Agent Framework ``FunctionTool``.

    ``activity_fn`` must carry ``@activity.defn`` and be a function or a method
    passed as ``Class.method`` (the worker registers the bound method). The
    tool's name defaults to the activity name, its description to the
    docstring, and its parameter schema is inferred by Agent Framework from the
    activity's signature, without a method's ``self``. When the model calls
    the tool, the arguments are bound by name and the activity runs through
    ``workflow.execute_activity`` with the given options; ``summary`` defaults
    to the tool name.

    A failed activity raises its failure cause from the tool, which Agent
    Framework turns into an error result for the model. Cancellation of the
    workflow or of the tool task propagates as ``asyncio.CancelledError``.

    Raises:
        ValueError: If ``activity_fn`` is not an ``@activity.defn`` function or
            method, or takes ``*args``, keyword-only arguments or ``**kwargs``,
            which the tool cannot pass by position.
    """
    if inspect.isclass(activity_fn) or not (
        inspect.isfunction(activity_fn) or inspect.ismethod(activity_fn)
    ):
        raise ValueError(
            "activity_as_tool takes a function or a method decorated with "
            f"@activity.defn, not {activity_fn!r}"
        )
    defn = activity._Definition.from_callable(activity_fn)
    if defn is None or not defn.name:
        raise ValueError("activity_fn must be decorated with @activity.defn")
    activity_name = defn.name

    signature = inspect.signature(activity_fn)
    params = list(signature.parameters.values())
    # A first parameter named `self` is the receiver of a method passed as
    # Class.method: the worker's bound method supplies it.
    if inspect.isfunction(activity_fn) and params and params[0].name == "self":
        params = params[1:]
    for param in params:
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                f"Activity {activity_name!r} cannot take *args, keyword-only "
                f"arguments or **kwargs ({param.name!r}): the tool passes one "
                "argument per parameter, by position"
            )
    # The model's arguments bind by name; the activity still gets them by position.
    call_signature = signature.replace(
        parameters=[
            p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            if p.kind is inspect.Parameter.POSITIONAL_ONLY
            else p
            for p in params
        ]
    )
    hints = typing.get_type_hints(activity_fn, include_extras=True)
    result_type = hints.get("return")
    tool_name = name or activity_name
    options = ActivityConfig(
        task_queue=task_queue,
        schedule_to_close_timeout=schedule_to_close_timeout,
        schedule_to_start_timeout=schedule_to_start_timeout,
        start_to_close_timeout=start_to_close_timeout,
        heartbeat_timeout=heartbeat_timeout,
        retry_policy=retry_policy,
        cancellation_type=cancellation_type,
        activity_id=activity_id,
        versioning_intent=versioning_intent,
        summary=summary if summary is not None else tool_name,
        priority=priority,
    )

    async def dispatch(**arguments: Any) -> Any:
        bound = call_signature.bind(**arguments)
        bound.apply_defaults()
        positional = list(bound.arguments.values())
        cancel_requests = _cancel_requests()
        try:
            if not positional:
                return await workflow.execute_activity(
                    activity_name, result_type=result_type, **options
                )
            if len(positional) == 1:
                return await workflow.execute_activity(
                    activity_name, positional[0], result_type=result_type, **options
                )
            return await workflow.execute_activity(
                activity_name, args=positional, result_type=result_type, **options
            )
        except ActivityError as error:
            if is_cancelled_exception(error) and _cancel_requests() > cancel_requests:
                # Agent Framework catches Exception as a tool failure and would
                # keep the run going; a cancelled workflow has to stop instead.
                raise asyncio.CancelledError() from error
            cause = error.__cause__
            # The ActivityError says "Activity task failed"; its cause carries
            # the activity's own message, which is what the model should read.
            if isinstance(cause, Exception):
                raise cause
            raise

    # Agent Framework infers the parameter schema from the signature it sees.
    dispatch.__signature__ = call_signature  # type: ignore[attr-defined]
    dispatch.__annotations__ = {
        p.name: hints[p.name] for p in params if p.name in hints
    }
    dispatch.__name__ = tool_name
    dispatch.__qualname__ = tool_name
    dispatch.__doc__ = inspect.getdoc(activity_fn)
    return FunctionTool(
        name=tool_name,
        description=description
        if description is not None
        else (inspect.getdoc(activity_fn) or ""),
        func=dispatch,
        approval_mode=approval_mode,
    )


def _cancel_requests() -> int:
    task = asyncio.current_task()
    return task.cancelling() if task is not None else 0
