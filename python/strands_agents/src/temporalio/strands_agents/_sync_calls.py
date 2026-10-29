"""Refuse Strands' synchronous agent calls in Workflow code.

``Agent.__call__``, ``A2AAgent.__call__`` and the other sync entry points run
the async call with ``strands._async.run_async``, which starts a new thread.
That thread runs outside the Workflow sandbox and off the Workflow event loop,
so a model or remote-agent call made there is not recorded in History and runs
again on replay.
"""

import functools
from collections.abc import Callable
from typing import Any

from strands import Agent
from strands.multiagent.base import MultiAgentBase
from strands.multiagent.graph import Graph
from strands.multiagent.swarm import Swarm

from temporalio import workflow

_GUARDED = "__temporal_strands_sync_guard__"


def _targets() -> list[tuple[type, str]]:
    targets: list[tuple[type, str]] = [
        (Agent, "__call__"),
        (Agent, "structured_output"),
        (MultiAgentBase, "__call__"),
        (Graph, "__call__"),
        (Swarm, "__call__"),
    ]
    try:
        from strands.agent.a2a_agent import A2AAgent
    except ImportError:  # the a2a extra is not installed
        pass
    else:
        targets.append((A2AAgent, "__call__"))
    return targets


def _guard(name: str, method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        # The sandbox imports Workflow modules before the Workflow runtime is
        # attached, so module-level code there is Workflow code too.
        if workflow.in_workflow() or workflow.unsafe.in_sandbox():
            raise RuntimeError(
                f"{type(self).__name__}.{name}() runs the call on a new thread, "
                "outside the Workflow sandbox, so its model calls and other I/O "
                "would not be recorded in Workflow History. In Workflow code, use "
                "`await agent.invoke_async(...)` with a TemporalAgent, and call "
                "other agents, such as an A2AAgent, from an Activity."
            )
        return method(self, *args, **kwargs)

    setattr(guarded, _GUARDED, True)
    return guarded


def refuse_sync_calls_in_workflows() -> None:
    """Make Strands' sync agent calls raise in Workflow code; idempotent."""
    for cls, name in _targets():
        method = cls.__dict__.get(name)
        if method is None or getattr(method, _GUARDED, False):
            continue
        setattr(cls, name, _guard(name, method))
