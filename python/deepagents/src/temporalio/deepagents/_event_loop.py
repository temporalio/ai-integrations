"""Schedule LangChain's same-loop stream writes without allowing thread hops."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from typing import Any

from temporalio import workflow
from temporalio.worker import (
    ExecuteWorkflowInput,
    Interceptor,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
)

_PATCH = "langchain-same-loop-streams"


def enable_workflow_streams() -> None:
    """Enable same-loop stream callbacks on new runs, preserving old histories."""
    if not workflow.patched(_PATCH):
        return
    loop = asyncio.get_running_loop()
    if getattr(loop, "_temporal_langchain_streams", False):
        return
    call_soon_threadsafe = loop.call_soon_threadsafe

    def on_workflow_loop() -> bool:
        try:
            return asyncio.get_running_loop() is loop
        except RuntimeError:
            return False

    def schedule(
        callback: Callable[..., Any],
        *args: Any,
        context: contextvars.Context | None = None,
    ) -> asyncio.Handle:
        # LangChain uses the thread-safe API even when writer and reader are
        # on the same loop. Never admit a callback from another thread/loop.
        if not on_workflow_loop():
            return call_soon_threadsafe(callback, *args, context=context)
        return loop.call_soon(callback, *args, context=context)

    setattr(loop, "call_soon_threadsafe", schedule)
    setattr(loop, "_temporal_langchain_streams", True)


class StreamingInterceptor(Interceptor):
    """Enable local stream scheduling for Workers and Replayers."""

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> type[WorkflowInboundInterceptor]:
        """Wrap each Workflow before its event stream starts."""

        class Inbound(WorkflowInboundInterceptor):
            async def execute_workflow(self, input: ExecuteWorkflowInput) -> Any:
                enable_workflow_streams()
                return await self.next.execute_workflow(input)

        return Inbound
