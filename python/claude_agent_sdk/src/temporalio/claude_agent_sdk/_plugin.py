"""One-line worker setup."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from typing import Any

from temporalio.client import Client
from temporalio.plugin import SimplePlugin
from temporalio.worker import Worker

from ._activity import SegmentRunner, make_segment_activity, make_tool_step_activity


class ClaudeAgentPlugin(SimplePlugin):
    """Registers the Activities that run Claude on the Worker.

    ``run_claude_segment`` runs the model, and ``run_claude_tool_step`` (for runners
    that have ``run_tool_step``) runs one Claude Code tool call as its own Activity.

    Example:
        .. code-block:: python

            runner = ClaudeAgentSdkRunner(cwd="/srv/agent")
            worker = Worker(
                client,
                task_queue="agents",
                workflows=[MyAgentWorkflow],
                activities=[my_tool],
                plugins=[ClaudeAgentPlugin(runner)],
            )
    """

    def __init__(
        self,
        runner: SegmentRunner,
        *,
        heartbeat_every: float = 5.0,
        native_max_concurrent_activities: int | None = None,
    ) -> None:
        """Create the plugin.

        Args:
            runner: Runs each model segment, for example ``ClaudeAgentSdkRunner``.
            heartbeat_every: Seconds between heartbeats while a segment or a tool
                step runs.
            native_max_concurrent_activities: Capacity per companion worker. Defaults
                to the main worker's explicit Activity limit, or 100.
        """
        added = [make_segment_activity(runner, heartbeat_every=heartbeat_every)]
        if (
            native_max_concurrent_activities is not None
            and native_max_concurrent_activities < 1
        ):
            raise ValueError("native_max_concurrent_activities must be positive")
        self._native_capacity = native_max_concurrent_activities
        if callable(getattr(runner, "run_tool_step", None)):
            # Claude Code tools that run as their own Activities.
            added.append(
                make_tool_step_activity(runner, heartbeat_every=heartbeat_every)
            )

        def activities(existing: Sequence[Any] | None) -> list[Any]:
            return [*(existing or []), *added]

        super().__init__("ClaudeAgentPlugin", activities=activities)
        self._runner = runner
        self._native_activity = added[-1] if len(added) > 1 else None

    async def run_worker(
        self, worker: Worker, next: Callable[[Worker], Awaitable[None]]
    ) -> None:
        """Provide independent Activity capacity for each supported subagent depth."""
        original_shutdown = worker.shutdown
        closed = asyncio.Event()

        async def shutdown() -> None:
            await original_shutdown()
            await closed.wait()

        # Worker.__aexit__ cancels its run task as soon as core shutdown finishes.
        # Include companion cleanup in shutdown, so that cancellation cannot cut
        # off their Activity/process cleanup.
        setattr(worker, "shutdown", shutdown)
        try:
            async with AsyncExitStack() as stack:
                config = worker.config(active_config=True)
                client_config = worker.client.config(active_config=True)
                client_config["plugins"] = []
                native_client = Client(**client_config)
                created: set[int] = set()
                creating = asyncio.Lock()
                owner = asyncio.current_task()

                async def ensure_depth(depth: int) -> None:
                    if not -1 <= depth <= 8:
                        raise ValueError(
                            "Native subagent depth must be between zero and eight"
                        )
                    async with creating:
                        if depth in created:
                            return
                        auxiliary = Worker(
                            native_client,
                            task_queue=f"{worker.task_queue}.__claude_native_{'controller' if depth == -1 else depth}",
                            activities=list(
                                worker.config(active_config=True).get("activities")
                                or []
                            ),
                            max_concurrent_activities=self._native_capacity
                            or config.get("max_concurrent_activities")
                            or 100,
                            activity_executor=config.get("activity_executor"),
                            interceptors=config.get("interceptors") or [],
                        )

                        async def join() -> None:
                            run_task = getattr(
                                auxiliary, "_async_context_run_task", None
                            )
                            if run_task is not None:
                                await asyncio.gather(run_task, return_exceptions=True)

                        stack.push_async_callback(join)
                        await stack.enter_async_context(auxiliary)
                        setattr(auxiliary, "_async_context_inner_task", owner)
                        created.add(depth)

                if self._native_activity is not None and hasattr(
                    self._runner, "_native_batches"
                ):
                    workers = getattr(self._runner, "_native_workers", {})
                    workers[worker.task_queue] = ensure_depth
                    setattr(self._runner, "_native_workers", workers)
                    await ensure_depth(-1)
                    await ensure_depth(0)
                try:
                    await next(worker)
                finally:
                    batches = getattr(self._runner, "_native_batches", {})
                    owned = [
                        b for b in batches.values() if b.queue == worker.task_queue
                    ]
                    for batch in owned:
                        if batch.task is not None:
                            batch.task.cancel()
                    tasks = [b.task for b in owned if b.task is not None]
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    workers = getattr(self._runner, "_native_workers", {})
                    workers.pop(worker.task_queue, None)
        finally:
            setattr(worker, "shutdown", original_shutdown)
            closed.set()
