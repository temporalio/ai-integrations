"""Expose a Temporal activity to a Locus agent as a tool."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from locus.tools.decorator import Tool, tool
from temporalio import activity, workflow


def activity_as_tool(
    activity_fn: Callable[..., Any],
    *,
    idempotent: bool = False,
    **activity_options: Any,
) -> Tool:
    """Return a Locus ``Tool`` that runs ``activity_fn`` as a Temporal activity.

    The tool takes its name from the activity, and its description and
    parameters from the function's docstring and signature, as ``@tool`` does.
    ``activity_options`` are the options of ``workflow.execute_activity``
    (``start_to_close_timeout``, ``retry_policy`` and so on). ``idempotent`` is
    Locus's own flag: the agent reuses the result of a repeated call.
    """
    defn = activity._Definition.from_callable(activity_fn)
    if defn is None or not defn.name:
        raise ValueError(
            "activity_as_tool needs a function decorated with @activity.defn"
        )
    if (
        activity_options.get("start_to_close_timeout") is None
        and activity_options.get("schedule_to_close_timeout") is None
    ):
        raise ValueError(
            f"activity_as_tool({defn.name}) needs start_to_close_timeout or "
            "schedule_to_close_timeout"
        )
    name = defn.name
    signature = inspect.signature(activity_fn)
    parameters = [p for p in signature.parameters.values() if p.name != "self"]
    signature = signature.replace(parameters=parameters)

    async def run(**kwargs: Any) -> Any:
        bound = signature.bind(**kwargs)
        bound.apply_defaults()
        args = list(bound.arguments.values())
        return await workflow.execute_activity(name, args=args, **activity_options)

    # Locus builds the tool's schema from the function it is given, so give the
    # dispatcher the activity's own name, docstring and parameters.
    run.__name__ = name
    run.__qualname__ = name
    run.__doc__ = activity_fn.__doc__
    run.__signature__ = signature  # type: ignore[attr-defined]
    run.__annotations__ = {
        k: v
        for k, v in getattr(activity_fn, "__annotations__", {}).items()
        if k != "self"
    }
    return tool(name=name, idempotent=idempotent)(run)
