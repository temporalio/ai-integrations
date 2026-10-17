"""A Worker process for the chaos tests: the real runner, talking to the test's fake model.

Settings come from the environment, so every Worker the test starts after a kill is
the same: ``CHAOS_ENGINE_ENV`` (the runner's ``env``, as JSON), ``ENGINE_CWD``,
``RUNNER_MODE`` (``held`` or ``store``), ``SESSION_DIR`` and ``SHOP_DIR``. Optional:
``WARM_ENGINES`` (the runner's ``warm_engines``), ``STEP_DELAY`` (seconds each tool
step waits before it starts, so a test can find a call waiting in its warm engine) and
``WORKSPACE_STORE`` (a folder: the runner's ``workspace_store``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    FileSessionStore,
    FileWorkspaceStore,
)
from temporalio.client import Client
from temporalio.worker import Worker
from tests.endless.activities import ALL
from tests.engine_tools.workflows import ShellWorkflow
from tests.helpers.workers import FAIL_FAST


class SlowSteps(ClaudeAgentSdkRunner):
    """The runner; a tool step whose call waits in a warm engine here first waits
    ``STEP_DELAY`` seconds."""

    async def run_tool_step(self, step: Any, attempt: int) -> Any:
        warm = self._warm.get((step.session_id, step.checkpoint))  # type: ignore[reportPrivateUsage]
        if warm is not None and warm.state == "call_waits":
            await asyncio.sleep(float(os.environ.get("STEP_DELAY") or 0))
        return await super().run_tool_step(step, attempt)


async def main() -> None:
    """Run until killed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--task-queue", required=True)
    args = parser.parse_args()
    store = os.environ.get("RUNNER_MODE", "held") == "store"
    runner = SlowSteps(
        session_store=FileSessionStore(os.environ["SESSION_DIR"]) if store else None,
        cwd=os.environ["ENGINE_CWD"],
        env=json.loads(os.environ["CHAOS_ENGINE_ENV"]),
        warm_engines=int(os.environ.get("WARM_ENGINES") or 0),
        warm_seconds=60.0,
        workspace_store=FileWorkspaceStore(os.environ["WORKSPACE_STORE"])
        if os.environ.get("WORKSPACE_STORE")
        else None,
    )
    client = await Client.connect(args.address)
    worker = Worker(
        client,
        task_queue=args.task_queue,
        workflows=[ShellWorkflow],
        activities=ALL,
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
        **FAIL_FAST,
    )
    print("worker ready", flush=True)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
