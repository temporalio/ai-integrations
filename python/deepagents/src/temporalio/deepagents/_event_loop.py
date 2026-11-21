"""Adapt LangChain stream scheduling and trace IDs for Workflow replay."""

from __future__ import annotations

import asyncio
import contextvars
import importlib
import importlib.util
from collections.abc import Callable, Iterable
from random import Random
from typing import Any
from uuid import UUID

from temporalio import workflow
from temporalio.worker import (
    ExecuteWorkflowInput,
    HandleSignalInput,
    HandleUpdateInput,
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
        install_workflow_run_ids(input)

        class Inbound(WorkflowInboundInterceptor):
            async def handle_signal(self, input: HandleSignalInput) -> None:
                enable_workflow_streams()
                await self.next.handle_signal(input)

            async def handle_update_handler(self, input: HandleUpdateInput) -> Any:
                enable_workflow_streams()
                return await self.next.handle_update_handler(input)

            async def execute_workflow(self, input: ExecuteWorkflowInput) -> Any:
                enable_workflow_streams()
                return await self.next.execute_workflow(input)

        return Inbound


_RUN_IDS_PATCH = "langchain-replay-stable-run-ids"


def install_workflow_run_ids(input: WorkflowInterceptorClassInput) -> None:
    """Give LangChain's root and child trace runs replay-stable IDs in Workflows."""
    from langchain_core.callbacks import manager
    from langchain_core.tracers import event_stream

    input.unsafe_extern_functions["temporalio.langchain.run_id"] = _workflow_run_id

    # The event streamer creates the root ID; callback managers create child IDs.
    # Both modules import the public UUID helper by value.
    for module in (manager, event_stream):
        if not getattr(module.uuid7, "_temporal_langchain_run_ids", False):
            setattr(module, "uuid7", _workflow_uuid_generator(module.uuid7))


def _workflow_uuid_generator(original: Callable[..., UUID]) -> Callable[..., UUID]:
    def generate(*args: Any, **kwargs: Any) -> UUID:
        if workflow.in_workflow():
            provider = workflow.extern_functions().get("temporalio.langchain.run_id")
            if provider is not None and (run_id := provider()) is not None:
                return run_id
        return original(*args, **kwargs)

    setattr(generate, "_temporal_langchain_run_ids", True)
    return generate


def _workflow_run_id() -> UUID | None:
    """Supply IDs only for this plugin's writable, patch-enabled Workflows."""
    if workflow.unsafe.is_read_only() or not workflow.patched(_RUN_IDS_PATCH):
        return None
    loop = asyncio.get_running_loop()
    random: Random | None = getattr(loop, "_temporal_langchain_run_id_random", None)
    if random is None:
        # A separate named stream does not consume application randomness.
        random = workflow.new_random("langchain-stream-run-ids")
        setattr(loop, "_temporal_langchain_run_id_random", random)
    return UUID(int=random.getrandbits(128), version=4)


def validate_stream_id_plugins(plugins: Iterable[Any]) -> None:
    """Refuse older LangSmith hooks before a Worker/Replayer processes history."""
    message = (
        "LangSmithPlugin is incompatible with replay-stable streaming IDs; "
        "upgrade temporalio-langsmith together with the streaming plugin."
    )
    for plugin in plugins:
        if plugin.name() == "langchain.LangSmithPlugin" and not getattr(
            plugin, "_temporal_langchain_stream_ids", False
        ):
            raise ValueError(message)
    # Its hooks are process-wide: an older LangSmith on a different Worker can
    # overwrite these generators too. Reject an incompatible installed package
    # even when this particular Worker/Replayer does not configure it.
    if importlib.util.find_spec("temporalio.langsmith") is not None:
        tracing = importlib.import_module("temporalio.langsmith")
        if not getattr(
            tracing.LangSmithPlugin, "_temporal_langchain_stream_ids", False
        ):
            raise ValueError(message)
