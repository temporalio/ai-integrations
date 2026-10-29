"""Refuse Strands' synchronous calls in Workflow code.

Most Strands sync entry points (``agent(...)``, ``A2AAgent(...)(...)``,
``agent.structured_output(...)``, ``graph(...)``, ``swarm(...)``,
``agent.tool.name(...)`` and others) run their async form through
``strands._async.run_async``, which starts a new thread. That thread runs
outside the Workflow sandbox and off the Workflow event loop, so a model or
remote-agent call made there is not recorded in History and runs again on
replay. ``strands.tools.mcp.MCPClient`` uses its own background thread instead
and is not covered here.
"""

import functools
import sys
import threading
import types
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from strands import Agent
from strands.multiagent.graph import Graph
from strands.multiagent.swarm import Swarm

from temporalio import workflow
from temporalio.worker.workflow_sandbox import RestrictedWorkflowAccessError

_T = TypeVar("_T")

_MESSAGE = (
    "Strands runs this sync call on a new thread, which cannot use the Workflow "
    "event loop and is not recorded in Workflow History. In Workflow code, use "
    "the call's async form, such as `await agent.invoke_async(...)` with a "
    "TemporalAgent or TemporalAgent nodes, or move the call into an Activity. "
    "If Strands made the call itself, as a SummarizingConversationManager or "
    "SnapshotSessionManager does, that feature does not work in Workflow code."
)


class _Building(threading.local):
    depth = 0


# Strands also uses run_async while an Agent, Graph or Swarm is built, to load
# tool providers and run init hooks. TemporalAgent relies on that, so it stays
# allowed while one of those constructors runs on this thread. Unlike a
# ContextVar, this is not inherited by tasks created during construction.
_building = _Building()

_install_lock = threading.Lock()
_installed = False


def _in_workflow_code() -> bool:
    if _building.depth or workflow.unsafe.is_sandbox_unrestricted():
        return False
    # The sandboxed runner imports Workflow modules before the Workflow runtime
    # is attached, so module-level code there is Workflow code too.
    return workflow.in_workflow() or workflow.unsafe.in_sandbox()


def _allow_during(init: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(init)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        _building.depth += 1
        try:
            init(self, *args, **kwargs)
        finally:
            _building.depth -= 1

    return wrapper


def refuse_sync_calls_in_workflows() -> None:
    """Make Strands' sync calls raise in Workflow code; runs once per process."""
    global _installed
    with _install_lock:
        if _installed:
            return
        try:
            import strands._async

            run_async = strands._async.run_async
        except (ImportError, AttributeError) as err:
            raise RuntimeError(
                "The Strands plugin could not find strands._async.run_async, "
                "which it guards so that Strands' sync calls fail in Workflow "
                "code. This strands-agents version is not supported."
            ) from err

        def guarded(async_func: Callable[[], Awaitable[_T]]) -> _T:
            if _in_workflow_code():
                raise RestrictedWorkflowAccessError(
                    "strands._async.run_async", override_message=_MESSAGE
                )
            return run_async(async_func)

        strands._async.run_async = guarded
        # Strands modules bind run_async when they are imported; rebind the
        # ones already loaded. Modules imported later get the guarded one.
        for name, module in list(sys.modules.items()):
            if (
                name.startswith("strands.")
                and type(module) is types.ModuleType
                and vars(module).get("run_async") is run_async
            ):
                setattr(module, "run_async", guarded)
        for cls in (Agent, Graph, Swarm):
            cls.__init__ = _allow_during(cls.__init__)  # type: ignore[method-assign]
        _installed = True
