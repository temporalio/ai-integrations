"""A disposable Worker process for the hybrid crash tests."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker
from tests.hybrid.activities import HybridActivities
from tests.hybrid.effects import configure_effects
from tests.hybrid.executor_store import ExecutionStore
from tests.hybrid.executor_workflow import NativeExecutionWorkflow
from tests.hybrid.native import NativeActivities
from tests.hybrid.native_executor import CheckpointActivities
from tests.hybrid.native_replay import ReplayActivities
from tests.hybrid.native_store import NativeStore
from tests.hybrid.replay_store import ReplayStore
from tests.hybrid.replay_workflow import NativeReplayWorkflow
from tests.hybrid.store import TranscriptStore
from tests.hybrid.workflows import HybridWorkflow


async def main() -> None:
    client = await Client.connect(os.environ["HYBRID_ADDRESS"])
    root = Path(os.environ["HYBRID_ROOT"])
    env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("ANTHROPIC", "CLAUDE", "DISABLE_", "NO_PROXY", "no_proxy"))
    }
    if os.environ.get("HYBRID_NATIVE_REPLAY"):
        replay = ReplayActivities(
            root, env, ReplayStore(root, os.environ.get("HYBRID_NATIVE_PHASE", "live"))
        )
        async with Worker(
            client,
            task_queue=os.environ["HYBRID_QUEUE"],
            workflows=[NativeReplayWorkflow],
            activities=[replay.decide, replay.execute],
        ):
            print("worker ready", flush=True)
            await asyncio.Event().wait()
        return
    acts = HybridActivities(client, root, env, TranscriptStore(root / "store.db"))
    if phase := os.environ.get("HYBRID_NATIVE_PHASE"):
        store = NativeStore(root, phase)
        if os.environ.get("HYBRID_NATIVE_EFFECTS"):
            configure_effects(store)
        acts = NativeActivities(client, root, env, store)
    acts.recovery = os.environ.get("HYBRID_MAIN_RECOVERY") == "1"
    if os.environ.get("HYBRID_HOLD_DELIVERY"):
        acts.before_delivery = asyncio.Event()
    if os.environ.get("HYBRID_HOLD_REQUEST"):
        acts.before_request = asyncio.Event()
    if os.environ.get("HYBRID_HOLD_CHECKPOINT"):
        acts.after_checkpoint = asyncio.Event()
    if os.environ.get("HYBRID_HOLD_SUSPENSION"):
        acts.after_suspension = asyncio.Event()
    if os.environ.get("HYBRID_NATIVE_EXECUTOR"):
        executor = CheckpointActivities(
            root,
            env,
            ExecutionStore(root, os.environ.get("HYBRID_NATIVE_PHASE", "live")),
        )
        async with Worker(
            client,
            task_queue=os.environ["HYBRID_QUEUE"],
            workflows=[NativeExecutionWorkflow],
            activities=[executor.decide, executor.execute],
        ):
            print("worker ready", flush=True)
            await asyncio.Event().wait()
        return
    async with Worker(
        client,
        task_queue=os.environ["HYBRID_QUEUE"],
        workflows=[HybridWorkflow],
        activities=[acts.burst, acts.tool],
    ):
        print("worker ready", flush=True)
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
