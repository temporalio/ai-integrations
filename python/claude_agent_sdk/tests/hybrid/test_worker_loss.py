"""Kill a Worker and inspect durable call outcomes on a replacement Worker."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.client import Client, WorkflowFailureError
from tests.conftest import PLUGIN_ROOT
from tests.helpers.fake_messages_api import engine_env
from tests.hybrid.models import State
from tests.hybrid.policy import rounds
from tests.hybrid.test_engine import alive
from tests.hybrid.test_workflow import until
from tests.hybrid.workflows import HybridWorkflow

pytestmark = pytest.mark.timeout(120)


async def launch(
    address: str,
    queue: str,
    root: Path,
    api: Any,
    machine: int,
    hold: bool,
    checkpoint_hold: bool = False,
    recovery: bool = False,
    request_hold: bool = False,
    suspension_hold: bool = False,
    native_phase: str | None = None,
    native_executor: bool = False,
    native_replay: bool = False,
    native_effects: bool = False,
) -> subprocess.Popen[bytes]:
    log = root / f"worker-{machine}.log"
    env = {
        **os.environ,
        **engine_env(api, str(root / f"machine-{machine}")),
        "HYBRID_ADDRESS": address,
        "HYBRID_QUEUE": queue,
        "HYBRID_ROOT": str(root),
        "PYTHONPATH": str(PLUGIN_ROOT),
    }
    if checkpoint_hold:
        env["HYBRID_HOLD_CHECKPOINT"] = "1"
    env["HYBRID_MAIN_RECOVERY"] = "1" if recovery else "0"
    if request_hold:
        env["HYBRID_HOLD_REQUEST"] = "1"
    if suspension_hold:
        env["HYBRID_HOLD_SUSPENSION"] = "1"
    if hold:
        env["HYBRID_HOLD_DELIVERY"] = "1"
    if native_phase is not None:
        env["HYBRID_NATIVE_PHASE"] = native_phase
    if native_executor:
        env["HYBRID_NATIVE_EXECUTOR"] = "1"
    if native_replay:
        env["HYBRID_NATIVE_REPLAY"] = "1"
    if native_effects:
        env["HYBRID_NATIVE_EFFECTS"] = "1"
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tests.hybrid.worker"],
            cwd=PLUGIN_ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )

    async def ready() -> bool:
        assert proc.poll() is None, log.read_text(errors="replace")
        return "worker ready" in log.read_text(errors="replace")

    await until(ready)
    return proc


@pytest.mark.parametrize(
    "phase", ["before-scheduling", "after-completion", "partial-batch"]
)
async def test_replacement_worker_preserves_ledger_and_fails_closed(
    client: Client, address: str, tmp_path: Path, phase: str
) -> None:
    api = rounds(
        1,
        3 if phase == "partial-batch" else 1,
        approval=phase == "before-scheduling",
        delay=1 if phase == "partial-batch" else 0,
    )
    queue = "worker-loss-" + uuid.uuid4().hex
    procs: list[subprocess.Popen[bytes]] = []
    pids: list[int] = []
    try:
        procs.append(await launch(address, queue, tmp_path, api, 1, True))
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["work"]),
            id=queue,
            task_queue=queue,
        )

        async def reached() -> Any:
            snap = await handle.query(HybridWorkflow.snapshot)
            if phase == "before-scheduling":
                return snap if snap.ledger else None
            if phase == "after-completion":
                return (
                    snap
                    if snap.ledger and all(e.outcome for e in snap.ledger.values())
                    else None
                )
            # Complete one tool in the batch, leave the others awaiting approval.
            return (
                snap
                if len(snap.ledger) == 3
                and any(e.outcome for e in snap.ledger.values())
                else None
            )

        if phase == "partial-batch":
            # Only n=0 completes quickly; hold the other tool Activities through a slow delay.
            original = api.decide

            def partial(body: dict[str, Any]) -> list[dict[str, Any]]:
                blocks = original(body)
                for b in blocks:
                    if b.get("type") == "tool_use":
                        b["input"]["delay"] = 0 if b["input"]["n"] == 0 else 6
                return blocks

            api.decide = partial
        before = await until(reached)
        pids = [
            r["pid"]
            for r in map(
                json.loads, (tmp_path / "cli-pids.jsonl").read_text().splitlines()
            )
        ]
        procs[0].kill()
        procs[0].wait()
        orphan_alive = any(alive(pid) for pid in pids)
        procs.append(await launch(address, queue, tmp_path, api, 2, False))
        with pytest.raises(WorkflowFailureError) as failed:
            await asyncio.wait_for(handle.result(), 45)
        assert failed.value.cause is not None
        cause = failed.value.cause
        while getattr(cause, "cause", None) is not None:
            cause = getattr(cause, "cause")
        assert "uncheckpointed native MCP calls" in str(cause)
        after = await handle.query(HybridWorkflow.snapshot)
        assert set(after.ledger) == set(before.ledger)
        for tid, entry in before.ledger.items():
            if entry.outcome:
                assert after.ledger[tid].outcome == entry.outcome
        scheduled = [
            e.activity_task_scheduled_event_attributes.activity_id
            for e in (await handle.fetch_history()).events
            if e.HasField("activity_task_scheduled_event_attributes")
        ]
        tool_ids = [s for s in scheduled if s.startswith("tool-")]
        assert len(tool_ids) == len(set(tool_ids))
        if phase == "before-scheduling":
            assert not tool_ids
        print(
            "HYBRID_WORKER_LOSS "
            + json.dumps(
                {
                    "phase": phase,
                    "orphan_alive_immediately": orphan_alive,
                    "ledger_ids": sorted(after.ledger),
                    "scheduled_tool_ids": tool_ids,
                }
            ),
            flush=True,
        )
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        # A SIGKILL cannot execute Python teardown. The harness explicitly reaps known children.
        # Production would require a supervisor/container to provide the same guarantee.
        for pid in pids:
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
        api.stop()


async def test_worker_loss_after_acknowledged_checkpoint_reuses_outcomes(
    client: Client, address: str, tmp_path: Path
) -> None:
    api = rounds(1)
    queue = "checkpoint-loss-" + uuid.uuid4().hex
    procs: list[subprocess.Popen[bytes]] = []
    pids: list[int] = []
    try:
        procs.append(
            await launch(address, queue, tmp_path, api, 1, False, checkpoint_hold=True)
        )
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["work"]),
            id=queue,
            task_queue=queue,
        )

        async def checkpoint() -> Any:
            snap = await handle.query(HybridWorkflow.snapshot)
            return snap if snap.checkpoints else None

        before = await until(checkpoint)
        pids = [before.checkpoints[0].pid]
        procs[0].kill()
        procs[0].wait()
        procs.append(await launch(address, queue, tmp_path, api, 2, False))
        state = await asyncio.wait_for(handle.result(), 30)
        assert state.answers == ["DONE 1"]
        assert state.ledger == before.ledger and state.checkpoints == before.checkpoints
        assert len((tmp_path / "cli-pids.jsonl").read_text().splitlines()) == 1
        schedules = [
            e.activity_task_scheduled_event_attributes.activity_id
            for e in (await handle.fetch_history()).events
            if e.HasField("activity_task_scheduled_event_attributes")
        ]
        assert len([s for s in schedules if s.startswith("tool-")]) == 1
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for pid in pids:
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
        api.stop()
