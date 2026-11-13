"""Worker plugin for Activity-backed Codex sessions."""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from builtins import next as next_item
from collections.abc import Awaitable, Callable

from openai_codex import CodexError
from temporalio.plugin import SimplePlugin
from temporalio.worker import Worker, WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ._activities import ConnectionActivities
from ._subprocess import SubprocessConnector
from ._types import Connector


class OpenAICodexPlugin(SimplePlugin):
    """Use AsyncCodex in Workflows.

    Activities reach Codex through the configured connector. A private Worker
    queue routes subsequent operations to the process holding the connection.
    """

    def __init__(self, *, connector: Connector | None = None) -> None:
        """Create the plugin.

        Args:
            connector: Starts Codex and reaches it. Defaults to
                :class:`SubprocessConnector`, which runs it in a subprocess of
                the Worker, as the Codex SDK does.
        """
        from ._compat import install_adapters

        install_adapters()
        self._connector = connector or SubprocessConnector()

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if not runner:
                raise ValueError("No WorkflowRunner provided to OpenAICodexPlugin.")

            if isinstance(runner, SandboxedWorkflowRunner):
                return dataclasses.replace(
                    runner,
                    # Share the SDK's Workflow-aware adapters and permit direct
                    # imports in Workflow code, including during replay.
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "openai_codex", "temporalio.openai_codex"
                    ),
                )
            return runner

        super().__init__(
            name="OpenAICodexPlugin",
            workflow_runner=workflow_runner,
            # Fail the Workflow on a Codex SDK error,
            # as it would end a program. Otherwise Temporal retries the Workflow
            # Task forever.
            workflow_failure_exception_types=[
                CodexError,
                FileNotFoundError,
                PermissionError,
            ],
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Configure the Worker, except the one :meth:`run_worker` starts.

        That Worker gets this plugin from the Client, if the plugin is
        registered there. It needs nothing beyond what :meth:`run_worker` gives it.
        """
        task_queue = config.get("task_queue", "")
        if task_queue.startswith("openai-codex-connector-"):
            return config
        if any(
            isinstance(getattr(fn, "__self__", None), ConnectionActivities)
            for fn in config.get("activities", [])
        ):
            return config

        # Only this process polls the connector queue, so the Activities that
        # reach the Codex this Worker started run here.
        activities = ConnectionActivities(
            self._connector, f"openai-codex-connector-{uuid.uuid4()}"
        )
        config["activities"] = [*config.get("activities", []), activities.connect]
        return super().configure_worker(config)

    async def run_worker(
        self, worker: Worker, next: Callable[[Worker], Awaitable[None]]
    ) -> None:
        """Run the Worker, and beside it a Worker for the Activities that only this process can run."""
        if worker.task_queue.startswith("openai-codex-connector-"):
            await next(worker)
            return
        activities = next_item(
            (
                owner
                for fn in worker.config(active_config=True).get("activities", [])
                if isinstance(
                    owner := getattr(fn, "__self__", None), ConnectionActivities
                )
            ),
            None,
        )
        if activities is None or activities.connector_worker_running:
            await next(worker)
            return
        activities.connector_worker_running = True
        try:
            connector_worker = Worker(
                worker.client,
                task_queue=activities.connector_queue,
                activities=activities.connected_activities(),
                interceptors=worker.config().get("interceptors", []),
            )
            async with connector_worker:
                await super().run_worker(worker, next)
        finally:
            # Make sure this Worker's Codex processes are stopped, even if this plugin is cancelled.
            try:
                await asyncio.shield(activities.shut_down())
            finally:
                activities.connector_worker_running = False
