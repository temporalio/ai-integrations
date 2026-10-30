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
        self._connector = connector or SubprocessConnector()

        # Each Worker's Activities, by its task queue.
        self._workers: dict[str, ConnectionActivities] = {}

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
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Configure the Worker, except the one :meth:`run_worker` starts.

        That Worker gets this plugin from the Client, if the plugin is
        registered there. It needs nothing beyond what :meth:`run_worker` gives it.
        """
        task_queue = config.get("task_queue", "")
        if task_queue in self._connector_queues():
            return config

        # Only this process polls the connector queue, so the Activities that
        # reach the Claude Code this Worker started run here.
        activities = ConnectionActivities(
            self._connector, f"claude-agent-sdk-{uuid.uuid4()}"
        )
        self._workers[task_queue] = activities
        config["activities"] = [*config.get("activities", []), activities.connect]
        return super().configure_worker(config)

    async def run_worker(
        self, worker: Worker, next: Callable[[Worker], Awaitable[None]]
    ) -> None:
        """Run the Worker, and beside it a Worker for the Activities that only this process can run."""
        if worker.task_queue in self._connector_queues():
            await next(worker)
            return

        activities = self._workers[worker.task_queue]
        connector_worker = Worker(
            worker.client,
            task_queue=activities.connector_queue,
            activities=activities.connected_activities(),
            interceptors=worker.config().get("interceptors", []),
        )

        try:
            async with connector_worker:
                await super().run_worker(worker, next)
        finally:
            # Make sure this Worker's Claude Codes are stopped, even if this plugin is cancelled.
            await asyncio.shield(activities.shut_down())

    def _connector_queues(self) -> set[str]:
        return {activities.connector_queue for activities in self._workers.values()}
