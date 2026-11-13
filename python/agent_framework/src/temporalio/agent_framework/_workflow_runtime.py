"""Workflow-aware stand-ins for the clocks and id generators Agent Framework reads in agent code.

Agent Framework has no time or id provider hook. Its modules import
``uuid4``, ``perf_counter`` and ``time`` as module-local names, and those
modules pass through the workflow sandbox, so the plugin rebinds the names on
the module objects. Every stand-in checks ``workflow.in_workflow()`` and falls
through to the stdlib elsewhere, so Activities and plain Agent Framework code
in the same worker keep their normal clocks and ids.

What each seam feeds:

- ``_agents.uuid4``: the default ``Agent.id``.
- ``_tools.uuid4``: function-call occurrence ids a provider omitted, and
  replacement approval request ids.
- ``_sessions.uuid``: the default ``AgentSession.session_id``.
- ``_tools.perf_counter`` and ``_tools.time``: the ``max_duration_seconds``
  budget of the function-invocation loop, which decides whether the loop
  keeps calling tools.
- ``observability.perf_counter``: span durations of the agent telemetry layer.

``time_ns`` in the same modules only timestamps exceptions on spans, which
the tracer records in wall time anyway, so it is left alone.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import threading
import time
import uuid
from typing import Any

from temporalio import workflow

_MISSING = object()


def _uuid4() -> uuid.UUID:
    return workflow.uuid4() if workflow.in_workflow() else uuid.uuid4()


def _perf_counter() -> float:
    return workflow.time() if workflow.in_workflow() else time.perf_counter()


def _time() -> float:
    return workflow.time() if workflow.in_workflow() else time.time()


class _UuidModule:
    """Stands in for the ``uuid`` module where a module imported it whole."""

    uuid4 = staticmethod(_uuid4)

    def __getattr__(self, name: str) -> Any:
        return getattr(uuid, name)


_uuid_module = _UuidModule()

# (module, attribute, the stdlib object the attribute holds, the stand-in)
_SEAMS: tuple[tuple[str, str, Any, Any], ...] = (
    ("agent_framework._agents", "uuid4", uuid.uuid4, _uuid4),
    ("agent_framework._tools", "uuid4", uuid.uuid4, _uuid4),
    ("agent_framework._sessions", "uuid", uuid, _uuid_module),
    ("agent_framework._tools", "perf_counter", time.perf_counter, _perf_counter),
    ("agent_framework._tools", "time", time.time, _time),
    ("agent_framework.observability", "perf_counter", time.perf_counter, _perf_counter),
)

_install_lock = threading.Lock()


def installed_version() -> str:
    """Return the installed ``agent-framework-core`` version for error messages."""
    try:
        return importlib.metadata.version("agent-framework-core")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _unsupported(seam: str, detail: str) -> RuntimeError:
    return RuntimeError(
        f"agent-framework-core {installed_version()} changed {seam}: {detail}. "
        "The Temporal plugin cannot make this seam replay-safe; pin "
        "agent-framework-core to a version the plugin supports."
    )


def check_sync_tool_seam() -> None:
    """Fail unless ``FunctionTool`` still has the switch that runs sync tools inline.

    Agent Framework runs a synchronous tool through ``asyncio.to_thread``,
    which the workflow event loop does not implement. Its own harness sets
    ``_invoke_sync_on_event_loop`` on a tool to run it inline instead, and the
    plugin's function middleware sets the same flag.
    """
    from agent_framework import FunctionTool

    probe = FunctionTool(name="temporal_probe", func=None)
    if getattr(probe, "_invoke_sync_on_event_loop", None) is not False:
        raise _unsupported(
            "FunctionTool._invoke_sync_on_event_loop",
            "the attribute is missing or no longer defaults to False",
        )
    try:
        source = inspect.getsource(FunctionTool._invoke_function)
    except (OSError, TypeError):
        return
    if "_invoke_sync_on_event_loop" not in source:
        raise _unsupported(
            "FunctionTool._invoke_function",
            "it no longer reads _invoke_sync_on_event_loop",
        )


def install_workflow_runtime() -> None:
    """Rebind Agent Framework's clock and id seams for Workers and Replayers.

    Checks every seam before changing any of them, so a worker fails to start
    with one error naming the seam and the installed version instead of
    running with a replay hazard. Safe to call more than once.

    Raises:
        RuntimeError: If a seam is missing or holds something other than the
            stdlib object or this module's stand-in.
    """
    with _install_lock:
        modules: dict[str, Any] = {}
        for module_name, attr, expected, shim in _SEAMS:
            seam = f"{module_name}.{attr}"
            if module_name not in modules:
                try:
                    modules[module_name] = importlib.import_module(module_name)
                except ImportError as err:
                    raise _unsupported(
                        seam, f"the module cannot be imported ({err})"
                    ) from err
            current = getattr(modules[module_name], attr, _MISSING)
            if current is _MISSING:
                raise _unsupported(seam, "the name is gone")
            if current is not expected and current is not shim:
                raise _unsupported(
                    seam, f"expected the stdlib {expected!r}, found {current!r}"
                )
        check_sync_tool_seam()
        for module_name, attr, _expected, shim in _SEAMS:
            setattr(modules[module_name], attr, shim)
