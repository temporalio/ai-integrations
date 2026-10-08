"""A real worker process for native batch crash recovery."""

from __future__ import annotations

import argparse
import asyncio
import os

from temporalio.claude_agent_sdk import ClaudeAgentPlugin, ClaudeAgentSdkRunner
from temporalio.client import Client
from temporalio.worker import Worker
from tests.native_batches.activities import record
from tests.native_batches.workflows import NativeWorkflow


async def main() -> None:
    """Run the stock SDK with separate one-slot native pools until killed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--task-queue", required=True)
    args = parser.parse_args()
    client = await Client.connect(args.address)
    worker = Worker(
        client,
        task_queue=args.task_queue,
        workflows=[NativeWorkflow],
        activities=[record],
        max_concurrent_activities=1,
        plugins=[
            ClaudeAgentPlugin(
                ClaudeAgentSdkRunner(cwd=os.environ["ENGINE_CWD"]), heartbeat_every=0.5
            )
        ],
    )
    print("worker ready", flush=True)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
