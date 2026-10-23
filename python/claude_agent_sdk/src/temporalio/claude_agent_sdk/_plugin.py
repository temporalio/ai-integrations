"""Worker plugin that runs the sandbox Activities and lets Workflows use the Claude Agent SDK."""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping

from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ._activities import SandboxActivities
from ._sandbox import SandboxFactory


class ClaudeAgentSDKPlugin(SimplePlugin):
    """Run ``ClaudeSDKClient`` with :class:`TemporalTransport` in Workflows.

    ``sandboxes`` maps each sandbox name to the :data:`SandboxFactory` that
    builds it for each Activity.
    """

    def __init__(self, sandboxes: Mapping[str, SandboxFactory]) -> None:
        """Create the plugin."""
        sandbox_activities = SandboxActivities(sandboxes)

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if not runner:
                raise ValueError("No WorkflowRunner provided to ClaudeAgentSDKPlugin.")
            if isinstance(runner, SandboxedWorkflowRunner):
                # anyio loads its asyncio backend lazily with import_module,
                # which bypasses the sandbox's passthrough and fails on the
                # backend's threading imports. Load it outside the sandbox so
                # the anyio passthrough below serves the host module.
                importlib.import_module("anyio._backends._asyncio")
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "claude_agent_sdk",
                        # The Claude Agent SDK's async runtime; uses threads
                        # and thread-locals.
                        "anyio",
                        # Detects the running async library through a
                        # thread-local shared by anyio and the Claude Agent SDK.
                        "sniffio",
                    ),
                )
            return runner

        super().__init__(
            name="ClaudeAgentSDKPlugin",
            workflow_runner=workflow_runner,
            activities=[
                sandbox_activities.connect,
                sandbox_activities.launch,
                sandbox_activities.write_stdin,
                sandbox_activities.watch_output,
                sandbox_activities.close,
            ],
        )
