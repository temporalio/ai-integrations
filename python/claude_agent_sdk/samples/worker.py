"""Runs the Worker for AgentWorkflow: ``python -m samples.worker``.

Start a dev server:

    temporal server start-dev

Give Claude Code credentials with ``sbx secret set -g anthropic``, or set
``ANTHROPIC_API_KEY`` or ``CLAUDE_CODE_OAUTH_TOKEN`` (from ``claude setup-token``)
for the Worker. Either way, they are never in history.
"""

import asyncio
import os

from samples.docker_sandbox import DockerSandbox
from samples.workflow import AgentWorkflow, roll_dice
from temporalio.claude_agent_sdk import (
    ClaudeAgentSDKPlugin,
    Sandbox,
    SandboxContext,
    SandboxFactory,
)
from temporalio.client import Client
from temporalio.worker import Worker

TASK_QUEUE = "claude-agent-sdk"


def create_worker(client: Client, factory: SandboxFactory) -> Worker:
    plugin = ClaudeAgentSDKPlugin({"docker": factory})
    return Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[roll_dice],
        plugins=[plugin],
    )


async def main() -> None:
    client = await Client.connect("localhost:7233")
    template = os.environ.get("CLAUDE_SANDBOX_TEMPLATE")
    names = ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"]
    env = {name: os.environ[name] for name in names if name in os.environ}

    def factory(context: SandboxContext) -> Sandbox:
        return DockerSandbox(
            context.first_execution_run_id,
            env=env,
            create_args=["--template", template] if template else [],
        )

    await create_worker(client, factory).run()


if __name__ == "__main__":
    asyncio.run(main())
