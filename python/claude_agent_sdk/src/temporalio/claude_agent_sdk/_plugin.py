"""Worker plugin that lets Workflows use the Claude Agent SDK and runs the Activities that reach Claude Code."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import uuid
from collections.abc import Awaitable, Callable

from claude_agent_sdk import ClaudeSDKError

from temporalio.plugin import SimplePlugin
from temporalio.worker import Worker, WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ._activities import ConnectionActivities
from ._subprocess import SubprocessConnector
from ._types import Connector


class ClaudeAgentSDKPlugin(SimplePlugin):
    """Use the Claude Agent SDK in Workflows.

    In the Workflow, pass the Claude Agent SDK a :class:`TemporalTransport` as
    its ``transport``. This plugin runs the transport's Activities, which reach
    Claude Code through ``connector``.
    """

    def __init__(self, *, connector: Connector | None = None) -> None:
        """Create the plugin.

        Args:
            connector: Starts Claude Code and reaches it. Defaults to
                :class:`SubprocessConnector`, which runs it in a subprocess of
                the Worker, as the Claude Agent SDK does.
        """
        # Only this process polls this task queue, so the Activities that reach
        # the Claude Code it started run here.
        self._own_queue = f"claude-agent-sdk-{uuid.uuid4()}"
        self._connection_activities = ConnectionActivities(
            connector or SubprocessConnector(), self._own_queue
        )

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if not runner:
                raise ValueError("No WorkflowRunner provided to ClaudeAgentSDKPlugin.")

            if isinstance(runner, SandboxedWorkflowRunner):
                # Import anyio's asyncio backend here, outside the sandbox.
                # Inside, anyio imports it lazily with import_module, which
                # skips the passthrough below and fails on its threading imports.
                importlib.import_module("anyio._backends._asyncio")

                return dataclasses.replace(
                    runner,
                    # Pass these through, so the Workflow shares one copy of
                    # them with the Worker. anyio, the Claude Agent SDK's async
                    # runtime, uses threads, and sniffio keeps the running
                    # async library in a thread-local.
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "claude_agent_sdk", "anyio", "sniffio"
                    ),
                )
            return runner

        super().__init__(
            name="ClaudeAgentSDKPlugin",
            workflow_runner=workflow_runner,
            # Fail the Workflow on a Claude Agent SDK error such as ProcessError,
            # as it would end a program. Otherwise Temporal retries the Workflow
            # Task forever.
            workflow_failure_exception_types=[ClaudeSDKError],
            activities=[self._connection_activities.connect],
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Configure the Worker, except the one :meth:`run_worker` starts.

        That Worker gets this plugin from the Client, if the plugin is
        registered there. It needs nothing beyond what :meth:`run_worker` gives it.
        """
        if config.get("task_queue") == self._own_queue:
            return config

        return super().configure_worker(config)

    async def run_worker(
        self, worker: Worker, next: Callable[[Worker], Awaitable[None]]
    ) -> None:
        """Run the Worker, and beside it a Worker for the Activities that only this process can run."""
        if worker.task_queue == self._own_queue:
            await next(worker)
            return

        own = Worker(
            worker.client,
            task_queue=self._own_queue,
            activities=self._connection_activities.connected_activities(),
            interceptors=worker.config().get("interceptors", []),
        )

        stopped = False
        try:
            async with own:
                await super().run_worker(worker, next)
                stopped = True
        except asyncio.CancelledError:
            # Remove this once the minimum temporalio includes
            # https://github.com/temporalio/sdk-python/pull/2000, which waits
            # for plugins instead of cancelling them.
            if not stopped:
                raise
