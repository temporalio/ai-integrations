"""Run Strands' sync tools directly in the Workflow.

Strands runs a sync tool function with ``asyncio.to_thread``. The Workflow
event loop has no threads, so the call raises ``NotImplementedError`` before the
tool's body runs. The modules that make that call pass through the Workflow
sandbox, so adapt their module-local ``asyncio`` reference, as
``_workflow_time`` does for ``time``. Every call checks its execution context,
so Activities and native Strands calls in the same worker keep their threads.
"""

import asyncio
import contextvars
import importlib
import inspect
import logging
import threading
from collections.abc import Callable
from typing import Any

from temporalio import workflow


async def _to_thread(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    if workflow.in_workflow():
        # Like asyncio.to_thread, run it in a copy of the current context.
        return contextvars.copy_context().run(func, *args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


class _Asyncio:
    to_thread = staticmethod(_to_thread)

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)


_asyncio = _Asyncio()
_install_lock = threading.Lock()
_logger = logging.getLogger(__name__)


def _check_sync_tool_seam(stream: Callable[..., Any]) -> None:
    name = f"{stream.__module__}.{stream.__qualname__}"
    try:
        source = inspect.getsource(stream)
    except (OSError, TypeError) as err:
        _logger.warning(
            "Cannot read the source of %s (%s), so its asyncio seam is installed "
            "unchecked; a sync tool runs inline in a Workflow only while "
            "Strands hands it to asyncio.to_thread.",
            name,
            err,
        )
        return
    if "asyncio.to_thread(" not in source:
        raise RuntimeError(
            f"Unsupported Strands sync tool call in {name}; "
            "it no longer runs sync tools through asyncio.to_thread."
        )


def run_sync_tools_in_workflows() -> None:
    """Adapt Strands' sync tool calls for both Workers and Replayers."""
    with _install_lock:
        # The only asyncio.to_thread calls in these modules run a sync tool.
        names = (
            ("strands.tools.decorator", "DecoratedFunctionTool"),
            ("strands.tools.tools", "PythonAgentTool"),
        )
        modules = []
        for name, tool_type in names:
            module = importlib.import_module(name)
            current = getattr(module, "asyncio", None)
            if current is not asyncio and current is not _asyncio:
                raise RuntimeError(
                    f"Unsupported Strands asyncio dependency in {module.__name__}; "
                    "the Strands plugin requires a module-local asyncio dependency "
                    "to run sync tools in the Workflow."
                )
            _check_sync_tool_seam(getattr(module, tool_type).stream)
            modules.append(module)
        for module in modules:
            setattr(module, "asyncio", _asyncio)
