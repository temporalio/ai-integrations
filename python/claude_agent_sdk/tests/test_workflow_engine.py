"""End to end: Temporal + the real Claude Code engine + a local fake Messages API.

Run 1 is a normal refund with a manager's approval. Run 2 kills the Worker right after
the refund, and a new Worker with an empty Claude config folder (a different machine)
finishes the job: with the conversation in the Workflow, the Workers share nothing
but the Temporal server; with a session store, they share the store. Then the Worker
dies in the middle of a segment, and a segment attempt outlives its timeout: each time
the segment runs again from its checkpoint, and every tool runs once.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    FileSessionStore,
    follow_agent,
)
from temporalio.client import Client
from temporalio.worker import Worker
from tests.conftest import wait_for_approval
from tests.endless.activities import ALL as COUNTING
from tests.endless.policy import count_policy
from tests.endless.workflows import LongTaskWorkflow, TaskOptions
from tests.helpers.fake_messages_api import (
    FakeMessagesAPI,
    engine_env,
    start_with_policy,
)
from tests.helpers.workers import FAIL_FAST
from tests.refund import shop
from tests.refund.policy import refund_policy
from tests.refund.workflows import MANAGER, RefundAgentWorkflow
from tests.test_crash import activity_completed, kill, start_worker

PROMPT = "Order A-1001 arrived broken, I want my money back."
pytestmark = pytest.mark.timeout(240)


def shared_settings(tmp_path: Path, shop_dir: Path, mode: str) -> dict[str, str]:
    """What every Worker process of a test shares: the shop, and the store if any."""
    settings = {
        "SHOP_DIR": str(shop_dir),
        "ENGINE_CWD": str(tmp_path / "work"),
        "RUNNER_MODE": mode,
    }
    if mode == "store":
        settings["SESSION_DIR"] = str(tmp_path / "sessions")
    return settings


@pytest.mark.parametrize(
    ("crash", "mode"),
    [(False, "held"), (True, "held"), (True, "store")],
    ids=["clean", "crash-after-refund", "crash-after-refund-store"],
)
async def test_real_engine_refund_with_approval(
    client: Client,
    address: str,
    shop_dir: Path,
    tmp_path: Path,
    crash: bool,
    mode: str,
) -> None:
    api = start_with_policy(refund_policy)
    queue = f"e2e-{uuid.uuid4().hex[:8]}"
    shared = shared_settings(tmp_path, shop_dir, mode)
    (tmp_path / "work").mkdir()

    def machine(n: int) -> dict[str, str]:
        return {**shared, **engine_env(api, str(tmp_path / f"claude-config-{n}"))}

    workers = [
        await start_worker(address, queue, machine(1), tmp_path / "w1.log", real=True)
    ]
    try:
        handle = await client.start_workflow(
            RefundAgentWorkflow.run, PROMPT, id=queue, task_queue=queue
        )
        pending = await wait_for_approval(handle, timeout=120)
        assert pending is not None, (tmp_path / "w1.log").read_text(errors="replace")[
            -3000:
        ]
        assert pending["name"] == "issue_refund"
        await handle.execute_update(
            RefundAgentWorkflow.review, args=[pending["id"], True, MANAGER]
        )
        if crash:
            while not await activity_completed(handle, "issue_refund"):
                await asyncio.sleep(0.05)
            kill(workers[0])
            workers.append(
                await start_worker(
                    address, queue, machine(2), tmp_path / "w2.log", real=True
                )
            )
        result = await asyncio.wait_for(handle.result(), 180)
    finally:
        for worker in workers:
            if worker.poll() is None:
                kill(worker)
        api.stop()

    kinds: Counter[str] = Counter()
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            kinds[
                event.activity_task_scheduled_event_attributes.activity_type.name
            ] += 1
    assert result.startswith("Done. Refunded 49.99 EUR for order A-1001")
    assert dict(kinds) == {
        "run_claude_segment": 4,
        "look_up_order": 1,
        "issue_refund": 1,
        "email_customer": 1,
    }
    assert len(shop.read("refunds.jsonl")) == 1
    assert len(shop.executions("issue_refund")) == 1
    assert api.errors == []  # every request followed the real API's tool rules
    if mode == "held":  # nothing of the conversation stayed on the Workers' disks
        for n in (1, 2):
            assert not list((tmp_path / f"claude-config-{n}").glob("projects/*/*"))


def hang_on_request(api: FakeMessagesAPI, number: int) -> tuple[Event, Event]:
    """Make the fake model hang on its ``number``-th turn until released."""
    decide = api.decide
    arrived, release = Event(), Event()
    seen = 0

    def hang(body: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal seen
        seen += 1
        if seen == number:
            arrived.set()
            release.wait(90)
        return decide(body)

    api.decide = hang
    return arrived, release


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_real_engine_crash_in_the_middle_of_a_segment(
    client: Client, address: str, shop_dir: Path, tmp_path: Path, mode: str
) -> None:
    """The Worker dies while Claude is answering; a new Worker redoes that segment cleanly."""
    api = start_with_policy(refund_policy)
    arrived, release = hang_on_request(api, 2)  # the turn after look_up_order's result
    queue = f"midseg-{uuid.uuid4().hex[:8]}"
    shared = shared_settings(tmp_path, shop_dir, mode)
    (tmp_path / "work").mkdir()

    def machine(n: int) -> dict[str, str]:
        return {**shared, **engine_env(api, str(tmp_path / f"claude-config-{n}"))}

    workers = [
        await start_worker(address, queue, machine(1), tmp_path / "w1.log", real=True)
    ]
    try:
        handle = await client.start_workflow(
            RefundAgentWorkflow.run, PROMPT, id=queue, task_queue=queue
        )
        assert await asyncio.to_thread(arrived.wait, 120)
        kill(workers[0])  # a power cut while the model call is in flight
        workers.append(
            await start_worker(
                address, queue, machine(2), tmp_path / "w2.log", real=True
            )
        )
        pending = await wait_for_approval(handle, timeout=120)
        assert pending is not None and pending["name"] == "issue_refund"
        await handle.execute_update(
            RefundAgentWorkflow.review, args=[pending["id"], True, MANAGER]
        )
        result = await asyncio.wait_for(handle.result(), 180)
    finally:
        release.set()
        for worker in workers:
            if worker.poll() is None:
                kill(worker)
        api.stop()
    attempts = [
        event.activity_task_started_event_attributes.attempt
        async for event in handle.fetch_history_events()
        if event.HasField("activity_task_started_event_attributes")
    ]
    assert result.startswith("Done. Refunded 49.99 EUR for order A-1001")
    assert 2 in attempts  # the interrupted segment ran again on the new Worker
    for tool in ("look_up_order", "issue_refund", "email_customer"):
        assert len(shop.executions(tool)) == 1, tool
    assert len(shop.read("refunds.jsonl")) == 1
    assert api.errors == []


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_real_engine_segment_that_times_out_is_retried_cleanly(
    client: Client, shop_dir: Path, tmp_path: Path, mode: str
) -> None:
    """Attempt 1 hangs past its timeout and keeps running; attempt 2 must not build on it."""
    del shop_dir
    api = start_with_policy(count_policy)
    arrived, release = hang_on_request(api, 3)
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        session_store=(
            FileSessionStore(tmp_path / "sessions") if mode == "store" else None
        ),
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
    )
    queue = f"zombie-{uuid.uuid4().hex[:8]}"
    options = TaskOptions(continue_as_new=False, live=True, segment_timeout=12)
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[LongTaskWorkflow],
            activities=COUNTING,
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=1.0)],
            **FAIL_FAST,
        ):
            handle = await client.start_workflow(
                LongTaskWorkflow.run,
                args=["count to 3", options, None],
                id=queue,
                task_queue=queue,
            )
            events = []
            async for event in follow_agent(client, queue):
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
            result = await asyncio.wait_for(handle.result(), 60)
            assert arrived.is_set()
    finally:
        release.set()
        api.stop()
    assert result == "counted to 3"
    assert [e["detail"] for e in shop.executions("count")] == ["1", "2", "3"]
    retries = [e for e in events if e["type"] == "retry"]
    assert [(r["segment"], r["attempt"]) for r in retries] == [(2, 2)]
    assert api.errors == [] and runner.stub_calls == 0
