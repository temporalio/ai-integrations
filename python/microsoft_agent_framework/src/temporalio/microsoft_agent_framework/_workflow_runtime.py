"""Workflow-aware stand-ins for what Agent Framework reads from the stdlib in agent code.

Agent Framework has no time, id or thread provider hook. Its modules import
``uuid4``, ``perf_counter``, ``time`` and ``asyncio`` as module-local names,
and those modules pass through the workflow sandbox, so the plugin rebinds
the names on the module objects. Every stand-in checks
``workflow.in_workflow()`` and falls through to the stdlib elsewhere, so
Activities and plain Agent Framework code in the same worker keep their
normal clocks, ids and threads.

What each seam feeds:

- ``_agents.uuid4``: the default ``Agent.id``.
- ``_tools.uuid4``: function-call occurrence ids a provider omitted, and
  replacement approval request ids.
- ``_sessions.uuid``: the default ``AgentSession.session_id``.
- ``_tools.perf_counter`` and ``_tools.time``: the ``max_duration_seconds``
  budget of the function-invocation loop, which decides whether the loop
  keeps calling tools.
- ``_tools.asyncio``: ``to_thread``, which runs a synchronous tool. The
  workflow event loop has no executor, so in a workflow the tool runs inline
  on the workflow task; the tool object is not changed, so one shared with an
  Activity or a plain agent in the same process still runs on a thread there.
- ``_sessions.asyncio``: ``to_thread``, which the file-backed session store
  and history provider use for their file reads and writes. In a workflow it
  raises ``MicrosoftAgentFrameworkWorkflowError`` naming the call, so the run
  fails instead of the workflow task failing on the loop's
  ``NotImplementedError``.
- ``observability.perf_counter``: span durations of the agent telemetry layer.

``time_ns`` in the same modules only timestamps exceptions on spans, which
the tracer records in wall time anyway, so it is left alone.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import inspect
import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from temporalio import workflow
from temporalio.microsoft_agent_framework.workflow import (
    MicrosoftAgentFrameworkWorkflowError,
)

TESTED_VERSIONS = "1.23.0 through 1.26.0"
"""The ``agent-framework-core`` releases this plugin's test suite ran against."""

logger = logging.getLogger(__name__)

_MISSING = object()


def _uuid4() -> uuid.UUID:
    return workflow.uuid4() if workflow.in_workflow() else uuid.uuid4()


def _perf_counter() -> float:
    return workflow.time() if workflow.in_workflow() else time.perf_counter()


def _time() -> float:
    return workflow.time() if workflow.in_workflow() else time.time()


async def _to_thread(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    if workflow.in_workflow():
        return func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


async def _no_thread_in_workflow(
    func: Callable[..., Any], /, *args: Any, **kwargs: Any
) -> Any:
    if workflow.in_workflow():
        name = getattr(func, "__qualname__", repr(func))
        raise MicrosoftAgentFrameworkWorkflowError(
            f"Agent Framework tried to run {name} on a thread inside a workflow, "
            "which has none. File-backed session stores and history providers "
            "(FileSessionStore, FileHistoryProvider) do not work in a workflow; "
            "use the default in-memory history and carry session.to_dict() across "
            "continue-as-new."
        )
    return await asyncio.to_thread(func, *args, **kwargs)


class _UuidModule:
    """Stands in for the ``uuid`` module where a module imported it whole."""

    uuid4 = staticmethod(_uuid4)

    def __getattr__(self, name: str) -> Any:
        return getattr(uuid, name)


class _AsyncioModule:
    """Stands in for the ``asyncio`` module in the framework's tool module."""

    to_thread = staticmethod(_to_thread)

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)


class _SessionsAsyncioModule:
    """Stands in for the ``asyncio`` module in the framework's sessions module."""

    to_thread = staticmethod(_no_thread_in_workflow)

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)


_uuid_module = _UuidModule()
_asyncio_module = _AsyncioModule()
_sessions_asyncio_module = _SessionsAsyncioModule()

# (module, attribute, the stdlib object the attribute holds, the stand-in)
_SEAMS: tuple[tuple[str, str, Any, Any], ...] = (
    ("agent_framework._agents", "uuid4", uuid.uuid4, _uuid4),
    ("agent_framework._tools", "uuid4", uuid.uuid4, _uuid4),
    ("agent_framework._sessions", "uuid", uuid, _uuid_module),
    ("agent_framework._sessions", "asyncio", asyncio, _sessions_asyncio_module),
    ("agent_framework._tools", "perf_counter", time.perf_counter, _perf_counter),
    ("agent_framework._tools", "time", time.time, _time),
    ("agent_framework._tools", "asyncio", asyncio, _asyncio_module),
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
        f"agent-framework-core to a release the plugin was tested with "
        f"({TESTED_VERSIONS})."
    )


def check_sync_tool_seam() -> None:
    """Fail unless ``FunctionTool`` still hands a synchronous tool to ``asyncio.to_thread``.

    That call, through the tool module's ``asyncio`` name, is what the
    ``_tools.asyncio`` stand-in intercepts. The framework's own harness flips
    the private ``_invoke_sync_on_event_loop`` switch on the tool instead; the
    plugin leaves that switch alone because a tool object can be shared with
    code outside workflows. When the source is not available (an install that
    ships only ``.pyc`` files) the check logs a warning and the seam is
    installed anyway: the rebind itself does not depend on the source.
    """
    from agent_framework import FunctionTool

    try:
        source = inspect.getsource(FunctionTool._invoke_function)
    except (OSError, TypeError) as err:
        logger.warning(
            "Cannot read the source of FunctionTool._invoke_function (%s), so the "
            "agent_framework._tools.asyncio seam is installed unchecked; a "
            "synchronous tool runs inline in a workflow only while the framework "
            "hands it to asyncio.to_thread.",
            err,
        )
        return
    if "asyncio.to_thread(" not in source:
        raise _unsupported(
            "FunctionTool._invoke_function",
            "it no longer runs synchronous tools through asyncio.to_thread",
        )


def install_workflow_runtime() -> None:
    """Rebind Agent Framework's clock, id and thread seams for Workers and Replayers.

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
