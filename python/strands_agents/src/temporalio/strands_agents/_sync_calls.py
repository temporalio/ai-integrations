"""Refuse Strands' synchronous calls in Workflow code.

Strands runs every sync entry point (``agent(...)``, ``A2AAgent(...)(...)``,
``agent.structured_output(...)``, ``graph(...)``, ``swarm(...)``,
``agent.tool.name(...)`` and others) through ``strands._async.run_async``, which
starts a new thread. That thread runs outside the Workflow sandbox and off the
Workflow event loop, so a model or remote-agent call made there is not recorded
in History and runs again on replay.
"""

import contextvars
import functools
import sys
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import strands._async
from strands import Agent
from strands.multiagent.graph import Graph
from strands.multiagent.swarm import Swarm

from temporalio import workflow

_T = TypeVar("_T")

_run_async = strands._async.run_async

# Strands also uses run_async while an Agent, Graph or Swarm is built, to load
# tool providers and run init hooks. TemporalAgent relies on that, so it stays
# allowed while one of those constructors runs.
_constructing = contextvars.ContextVar("temporalio_strands_constructing", default=False)


def _guarded_run_async(async_func: Callable[[], Awaitable[_T]]) -> _T:
    # The sandbox imports Workflow modules before the Workflow runtime is
    # attached, so module-level code there is Workflow code too.
    if not _constructing.get() and (
        workflow.in_workflow() or workflow.unsafe.in_sandbox()
    ):
        raise RuntimeError(
            "Strands runs this sync call on a new thread, which cannot use the "
            "Workflow event loop and is not recorded in Workflow History. In "
            "Workflow code, use the async form instead, such as "
            "`await agent.invoke_async(...)` with a TemporalAgent (with "
            "`structured_output_model=` for structured output) or "
            "`await graph.invoke_async(...)`, and call other agents, such as an "
            "A2AAgent, from an Activity."
        )
    return _run_async(async_func)


def _allow_during(init: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(init)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        token = _constructing.set(True)
        try:
            init(self, *args, **kwargs)
        finally:
            _constructing.reset(token)

    return wrapper


@functools.cache
def refuse_sync_calls_in_workflows() -> None:
    """Make Strands' sync calls raise in Workflow code; runs once per process."""
    strands._async.run_async = _guarded_run_async
    # Strands modules bind run_async when they are imported; rebind the ones
    # already loaded. Modules imported later get the guarded one.
    for name, module in list(sys.modules.items()):
        if name.startswith("strands.") and vars(module).get("run_async") is _run_async:
            setattr(module, "run_async", _guarded_run_async)
    for cls in (Agent, Graph, Swarm):
        cls.__init__ = _allow_during(cls.__init__)  # type: ignore[method-assign]
