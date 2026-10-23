"""Kill the Worker process at the worst moments. Every run must finish; money moves once.

Each test runs with the conversation in the Workflow (the default: the Workers share
no storage at all) and in a store every Worker shares.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from temporalio.client import Client, WorkflowHandle
from tests.conftest import PLUGIN_ROOT, wait_for_approval
from tests.refund import shop
from tests.refund.workflows import MANAGER, RefundAgentWorkflow

PROMPT = "Order A-1001 arrived broken, I want my money back."
pytestmark = [
    pytest.mark.timeout(180),
    pytest.mark.parametrize("mode", ["held", "store"]),
]


async def start_worker(
    address: str, queue: str, env: dict[str, str], log: Path, *, real: bool = False
) -> subprocess.Popen[bytes]:
    """Start ``tests.refund.worker`` in its own process and wait until it is ready."""
    python_path = os.pathsep.join(
        p for p in (str(PLUGIN_ROOT), os.environ.get("PYTHONPATH")) if p
    )
    argv = [
        sys.executable,
        "-m",
        "tests.refund.worker",
        "--address",
        address,
        "--task-queue",
        queue,
    ]
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [*argv, *(["--real"] if real else [])],
            cwd=PLUGIN_ROOT,
            env={**os.environ, **env, "PYTHONPATH": python_path},
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    deadline = time.monotonic() + 60
    while "worker ready" not in log.read_text(encoding="utf-8", errors="replace"):
        if proc.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError(
                f"worker failed to start:\n{log.read_text(errors='replace')}"
            )
        await asyncio.sleep(0.1)
    return proc


def kill(proc: subprocess.Popen[bytes]) -> None:
    """Kill without any cleanup, like a power cut."""
    proc.kill()
    proc.wait()


async def wait_until(check: Callable[[], Any], timeout: float = 45) -> None:
    """Poll until ``check()`` is true."""
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise TimeoutError("condition never became true")
        await asyncio.sleep(0.05)


async def activity_completed(
    handle: WorkflowHandle[Any, Any], activity_type: str
) -> bool:
    """Whether an Activity of this type has completed, according to the history."""
    scheduled: set[int] = set()
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            attrs = event.activity_task_scheduled_event_attributes
            if attrs.activity_type.name == activity_type:
                scheduled.add(event.event_id)
        if event.HasField("activity_task_completed_event_attributes"):
            if (
                event.activity_task_completed_event_attributes.scheduled_event_id
                in scheduled
            ):
                return True
    return False


def worker_env(tmp: Path, shop_dir: Path, mode: str) -> dict[str, str]:
    """Settings every Worker process in a test shares (only the shop, unless ``store``)."""
    env = {"SHOP_DIR": str(shop_dir), "RUNNER_MODE": mode}
    if mode == "store":
        env["FAKE_STATE_DIR"] = str(tmp / "fake")
    return env


async def test_crash_right_after_refund_is_recorded(
    client: Client, address: str, shop_dir: Path, tmp_path: Path, mode: str
) -> None:
    queue = f"crash-{uuid.uuid4().hex[:8]}"
    env = worker_env(tmp_path, shop_dir, mode)
    w1 = await start_worker(address, queue, env, tmp_path / "w1.log")
    handle = await client.start_workflow(
        RefundAgentWorkflow.run, PROMPT, id=queue, task_queue=queue
    )
    pending = await wait_for_approval(handle)
    assert pending is not None
    await handle.execute_update(
        RefundAgentWorkflow.review, args=[pending["id"], True, MANAGER]
    )
    deadline = time.monotonic() + 45
    while not await activity_completed(handle, "issue_refund"):
        assert time.monotonic() < deadline
        await asyncio.sleep(0.05)
    kill(w1)
    w2 = await start_worker(address, queue, env, tmp_path / "w2.log")
    try:
        result = await asyncio.wait_for(handle.result(), 90)
    finally:
        kill(w2)
    assert result.startswith("Done.")
    assert (
        len(shop.executions("issue_refund")) == 1
    )  # the finished call never ran again
    assert len(shop.read("refunds.jsonl")) == 1


async def test_crash_after_money_moved_but_before_the_reply(
    client: Client, address: str, shop_dir: Path, tmp_path: Path, mode: str
) -> None:
    queue = f"crash-{uuid.uuid4().hex[:8]}"
    env = worker_env(tmp_path, shop_dir, mode)
    w1 = await start_worker(
        address, queue, {**env, "REFUND_DELAY": "5"}, tmp_path / "w1.log"
    )
    handle = await client.start_workflow(
        RefundAgentWorkflow.run, PROMPT, id=queue, task_queue=queue
    )
    pending = await wait_for_approval(handle)
    assert pending is not None
    await handle.execute_update(
        RefundAgentWorkflow.review, args=[pending["id"], True, MANAGER]
    )
    await wait_until(lambda: len(shop.read("refunds.jsonl")) == 1)
    kill(w1)
    w2 = await start_worker(address, queue, env, tmp_path / "w2.log")
    try:
        result = await asyncio.wait_for(handle.result(), 90)
    finally:
        kill(w2)
    assert result.startswith("Done.")
    assert (
        len(shop.executions("issue_refund")) == 2
    )  # Temporal retried the unfinished call...
    assert (
        len(shop.read("refunds.jsonl")) == 1
    )  # ...and the idempotency key moved money once


async def test_crash_in_the_middle_of_a_claude_segment(
    client: Client, address: str, shop_dir: Path, tmp_path: Path, mode: str
) -> None:
    queue = f"crash-{uuid.uuid4().hex[:8]}"
    env = worker_env(tmp_path, shop_dir, mode)
    w1 = await start_worker(
        address, queue, {**env, "FAKE_THINK": "4"}, tmp_path / "w1.log"
    )
    handle = await client.start_workflow(
        RefundAgentWorkflow.run, PROMPT, id=queue, task_queue=queue
    )
    await asyncio.sleep(1.5)  # Claude is still "thinking" about segment 1
    kill(w1)
    w2 = await start_worker(address, queue, env, tmp_path / "w2.log")
    try:
        pending = await wait_for_approval(handle)
        assert pending is not None
        await handle.execute_update(
            RefundAgentWorkflow.review, args=[pending["id"], True, MANAGER]
        )
        result = await asyncio.wait_for(handle.result(), 90)
    finally:
        kill(w2)
    assert result.startswith("Done.")
    for tool in ("look_up_order", "issue_refund", "email_customer"):
        assert len(shop.executions(tool)) == 1, tool
