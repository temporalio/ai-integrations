"""A worker process for the crash tests: scripted Claude by default, the real engine with --real."""

from __future__ import annotations

import argparse
import asyncio
import os

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    FileSessionStore,
    SegmentRunner,
)
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client
from temporalio.worker import Worker
from tests.helpers.workers import FAIL_FAST
from tests.refund.activities import ALL
from tests.refund.policy import refund_policy
from tests.refund.workflows import RefundAgentWorkflow


def build_runner(real: bool) -> SegmentRunner:
    """The real engine (with a fake model in tests), or the scripted runner.

    ``RUNNER_MODE=store`` keeps conversations in a store every Worker shares;
    otherwise (the default) each Workflow holds its own, and no storage is shared.
    ``WARM_ENGINES`` is the real runner's ``warm_engines`` (0 by default).
    """
    shared = os.environ.get("RUNNER_MODE", "held") == "store"
    if real:
        return ClaudeAgentSdkRunner(
            session_store=FileSessionStore(os.environ["SESSION_DIR"])
            if shared
            else None,
            cwd=os.environ.get("ENGINE_CWD"),
            warm_engines=int(os.environ.get("WARM_ENGINES", "0")),
        )
    return ScriptedClaude(
        refund_policy,
        os.environ.get("FAKE_STATE_DIR", ".fake_claude") if shared else None,
        think_seconds=float(os.environ.get("FAKE_THINK", "0")),
    )


async def main() -> None:
    """Run until killed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--task-queue", required=True)
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    client = await Client.connect(args.address)
    worker = Worker(
        client,
        task_queue=args.task_queue,
        workflows=[RefundAgentWorkflow],
        activities=ALL,
        plugins=[ClaudeAgentPlugin(build_runner(args.real), heartbeat_every=1.0)],
        **FAIL_FAST,
    )
    print("worker ready", flush=True)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
