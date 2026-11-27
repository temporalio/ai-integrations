"""Run Locus's synchronous tools inside the workflow.

Locus runs a synchronous ``@tool`` function on the event loop's thread pool.
The workflow event loop has none, so the call would fail and the model would
be handed a bare ``NotImplementedError``. In workflow code such a tool runs
inline instead, as an ``async def`` tool does. Like any workflow code it must
be deterministic; a tool that does I/O belongs in an activity
(``activity_as_tool``).
"""

from __future__ import annotations

import functools
import inspect
import threading
from typing import Any

from locus.tools.decorator import Tool
from temporalio import workflow

_lock = threading.Lock()
_installed = False


def run_sync_tools_in_workflows() -> None:
    """Make ``Tool.execute`` call a sync tool inline in workflow code. Idempotent."""
    global _installed
    with _lock:
        if _installed:
            return
        original = Tool.execute

        @functools.wraps(original)
        async def execute(self: Tool, ctx: Any = None, **kwargs: Any) -> Any:
            fn = self.fn
            if (
                not workflow.in_workflow()
                or workflow.unsafe.is_sandbox_unrestricted()
                or self.sandbox is not None
                or inspect.iscoroutinefunction(fn)
            ):
                return await original(self, ctx, **kwargs)

            # Same name, docstring and signature, so Locus coerces and binds the
            # arguments as before; only the thread hop is gone.
            @functools.wraps(fn)
            async def inline(*args: Any, **kw: Any) -> Any:
                return fn(*args, **kw)

            return await original(self.model_copy(update={"fn": inline}), ctx, **kwargs)

        Tool.execute = execute  # type: ignore[method-assign]
        _installed = True
