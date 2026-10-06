"""Live native Write/Bash/MCP tools, retries and Worker failover with original IDs."""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import statistics
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio import activity
from temporalio.client import Client, WorkflowFailureError
from temporalio.worker import Replayer, Worker
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env, history_of
from tests.hybrid.effects import NAME, configure_effects
from tests.hybrid.models import Call, Reply, State
from tests.hybrid.native import NativeActivities
from tests.hybrid.native_store import NativeStore
from tests.hybrid.test_native_workspace import stop_worker
from tests.hybrid.test_worker_loss import launch
from tests.hybrid.test_workflow import until
from tests.hybrid.workflows import HybridWorkflow

pytestmark = pytest.mark.timeout(120)


def sequence_api(calls: list[tuple[str, dict[str, Any]]]) -> FakeMessagesAPI:
    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if any(h.is_error for h in history):
            return [{"type": "text", "text": "ERROR"}]
        if len(history) == len(calls):
            return [{"type": "text", "text": "DONE"}]
        name, arguments = calls[len(history)]
        return [
            {
                "type": "tool_use",
                "id": f"toolu_effect_{len(history)}",
                "name": name,
                "input": arguments,
            }
        ]

    return FakeMessagesAPI(policy, primary_tools={name for name, _ in calls}).start()


class LostReplyActivities(NativeActivities):
    @activity.defn(name="hybrid_tool")
    async def tool(self, call: Call) -> Reply:
        self.tool_calls.append(call.id)
        reply = await super().tool(call)
        if activity.info().attempt == 1:
            raise RuntimeError("lost native Activity completion after publication")
        return reply


async def test_read_edit_and_overwrite_write_keep_native_context(
    client: Client, tmp_path: Path
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    path = str(store.workspace / "note.txt")
    api = sequence_api(
        [
            ("Read", {"file_path": path}),
            (
                "Edit",
                {"file_path": path, "old_string": "BEFORE", "new_string": "EDITED"},
            ),
            ("Write", {"file_path": path, "content": "WRITTEN\n"}),
            (
                "Write",
                {"file_path": str(store.workspace / "new.txt"), "content": "NEW\n"},
            ),
        ]
    )
    acts = NativeActivities(
        client, tmp_path, engine_env(api, str(tmp_path / "cfg")), store
    )
    acts.recovery = True
    queue = "native-context-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[HybridWorkflow],
            activities=[acts.burst, acts.tool],
        ):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["work"]),
                id=queue,
                task_queue=queue,
            )
            state = await asyncio.wait_for(handle.result(), 45)
        assert state.answers == ["DONE"]
        assert store.durable_text() == "WRITTEN\n"
        assert (store.workspace / "new.txt").read_text() == "NEW\n"
        assert len(acts.bursts) == 1
        assert set(state.ledger) == {f"toolu_effect_{i}" for i in range(4)}
        assert all(e.outcome and not e.outcome.is_error for e in state.ledger.values())
        await Replayer(workflows=[HybridWorkflow]).replay_workflow(
            await handle.fetch_history()
        )
        assert api.errors == []
    finally:
        api.stop()


@pytest.mark.parametrize("tool", ["Bash", NAME])
async def test_activity_retry_reuses_actual_effect_result(
    client: Client, tmp_path: Path, tool: str
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    configure_effects(store)
    effect = tmp_path / "external-effect.log"
    arguments: dict[str, Any] = (
        {"command": f"echo EFFECT >> {shlex.quote(str(effect))}"}
        if tool == "Bash"
        else {"amount": 7}
    )
    api = sequence_api([(tool, arguments)])
    acts = LostReplyActivities(
        client, tmp_path, engine_env(api, str(tmp_path / "cfg")), store
    )
    acts.recovery = True
    queue = "effect-retry-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[HybridWorkflow],
            activities=[acts.burst, acts.tool],
        ):
            state = await client.execute_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["work"]),
                id=queue,
                task_queue=queue,
            )
        assert state.answers == ["DONE"]
        assert acts.tool_calls == ["toolu_effect_0", "toolu_effect_0"]
        if tool == "Bash":
            assert effect.read_text().splitlines() == ["EFFECT"]
        else:
            with store.connect() as db:
                assert db.execute("SELECT id,amount FROM charges").fetchall() == [
                    ("toolu_effect_0", 7)
                ]
        assert len(acts.bursts) == 1
        assert api.errors == []
    finally:
        api.stop()


@pytest.mark.parametrize("tool", ["Write", "Bash", NAME])
@pytest.mark.parametrize("committed", [True, False])
async def test_worker_loss_never_repeats_native_effect(
    client: Client, address: str, tmp_path: Path, tool: str, committed: bool
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    configure_effects(store)
    effect = tmp_path / "external-effect.log"
    arguments: dict[str, Any]
    if tool == "Write":
        arguments = {
            "file_path": str(store.workspace / "nested.txt"),
            "content": "NEW\n",
        }
    elif tool == "Bash":
        arguments = {"command": f"echo EFFECT >> {shlex.quote(str(effect))}"}
    else:
        arguments = {"amount": 7}
    api = sequence_api([(tool, arguments)])
    phase = tool + (
        "-after-commit-before-completion"
        if committed
        else "-after-effect"
        if tool == NAME
        else "-after-write"
    )
    queue = "effect-loss-" + uuid.uuid4().hex
    procs: list[subprocess.Popen[bytes]] = []
    try:
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                1,
                False,
                recovery=True,
                native_phase=phase,
                native_effects=True,
            )
        )
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["work"]),
            id=queue,
            task_queue=queue,
        )

        async def reached() -> Any:
            row = store.execution("toolu_effect_0")
            if row and (
                row["result"] is not None
                if committed
                else row["phase"] in {"executed", "effect-without-result"}
            ):
                return row
            return None

        original = await until(reached)
        # Kill only the Worker. Its supervisor must stop the old process tree
        # and release the workspace lock before a replacement can restore it.
        procs[0].kill()
        procs[0].wait()

        async def unlocked() -> bool:
            from temporalio.claude_agent_sdk._process import workspace_lock

            try:
                with workspace_lock(store.lock_path):
                    return True
            except OSError:
                return False

        await until(unlocked)
        shutil.rmtree(store.workspace)
        shutil.rmtree(tmp_path / "machine-1", ignore_errors=True)
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                2,
                False,
                recovery=True,
                native_phase="replacement",
                native_effects=True,
            )
        )
        if committed:
            state = await asyncio.wait_for(handle.result(), 45)
            assert state.answers == ["DONE"]
            outcome = state.ledger["toolu_effect_0"].outcome
            assert outcome and json.loads(outcome.text) == original["result"]
            await Replayer(workflows=[HybridWorkflow]).replay_workflow(
                await handle.fetch_history()
            )
        else:
            with pytest.raises(WorkflowFailureError):
                await asyncio.wait_for(handle.result(), 45)
        if tool == "Bash":
            assert effect.read_text().splitlines() == ["EFFECT"]
        elif tool == NAME:
            with store.connect() as db:
                assert db.execute("SELECT id,amount FROM charges").fetchall() == [
                    ("toolu_effect_0", 7)
                ]
        else:
            assert (store.workspace / "nested.txt").exists() == committed
        with store.connect() as db:
            assert db.execute(
                "SELECT COUNT(*) FROM native_events WHERE phase='executed'"
            ).fetchone()[0] == (0 if not committed and tool == NAME else 1)
        assert api.errors == []
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()


async def test_native_rounds_share_one_process(client: Client, tmp_path: Path) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    api = sequence_api([("Bash", {"command": f"echo ROUND-{i}"}) for i in range(10)])
    times: list[float] = []
    policy = api.decide

    def timed(body: dict[str, Any]) -> list[dict[str, Any]]:
        times.append(time.monotonic())
        return policy(body)

    api.decide = timed
    acts = NativeActivities(
        client, tmp_path, engine_env(api, str(tmp_path / "cfg")), store
    )
    queue = "native-rounds-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[HybridWorkflow],
            activities=[acts.burst, acts.tool],
        ):
            state = await client.execute_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["work"]),
                id=queue,
                task_queue=queue,
            )
        assert state.answers == ["DONE"] and len(state.ledger) == 10
        assert len(acts.bursts) == 1 and len(times) == 11
        print(
            f"NATIVE_MEDIAN_ROUND_MS {statistics.median(1000 * (b - a) for a, b in zip(times, times[1:])):.1f}"
        )
        assert api.errors == []
    finally:
        api.stop()


@pytest.mark.parametrize("tool", ["Edit", "Write"])
async def test_overwrite_failover_keeps_actual_native_result(
    client: Client, address: str, tmp_path: Path, tool: str
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    path = str(store.workspace / "note.txt")
    arguments = (
        {"file_path": path, "old_string": "BEFORE", "new_string": "AFTER"}
        if tool == "Edit"
        else {"file_path": path, "content": "AFTER\n"}
    )
    api = sequence_api([("Read", {"file_path": path}), (tool, arguments)])
    queue = "overwrite-loss-" + uuid.uuid4().hex
    procs: list[subprocess.Popen[bytes]] = []
    try:
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                1,
                False,
                recovery=True,
                native_phase=tool + "-after-commit-before-completion",
            )
        )
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["work"]),
            id=queue,
            task_queue=queue,
        )

        async def committed() -> Any:
            row = store.execution("toolu_effect_1")
            return row if row and row["result"] is not None else None

        original = await until(committed)
        assert store.durable_text() == "AFTER\n"
        stop_worker(procs[0], tmp_path)
        shutil.rmtree(store.workspace)
        shutil.rmtree(tmp_path / "machine-1", ignore_errors=True)
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                2,
                False,
                recovery=True,
                native_phase="replacement",
            )
        )
        state = await asyncio.wait_for(handle.result(), 45)
        assert state.answers == ["DONE"]
        outcome = state.ledger["toolu_effect_1"].outcome
        assert outcome and json.loads(outcome.text) == original["result"]
        assert store.durable_text() == "AFTER\n"
        assert not any(h.is_error for h in history_of(api.requests[-1])[2])
        with store.connect() as db:
            assert db.execute(
                "SELECT id FROM native_events WHERE phase='executed' ORDER BY seq"
            ).fetchall() == [("toolu_effect_0",), ("toolu_effect_1",)]
        await Replayer(workflows=[HybridWorkflow]).replay_workflow(
            await handle.fetch_history()
        )
        assert api.errors == []
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()


async def test_worker_loss_stops_running_native_bash(
    client: Client, address: str, tmp_path: Path
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    effect = tmp_path / "running-bash.log"
    target = shlex.quote(str(effect))
    api = sequence_api(
        [
            (
                "Bash",
                {"command": f"echo START >> {target}; sleep 3; echo LATE >> {target}"},
            )
        ]
    )
    queue = "running-bash-loss-" + uuid.uuid4().hex
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = await launch(
            address, queue, tmp_path, api, 1, False, recovery=True, native_phase="live"
        )
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["work"]),
            id=queue,
            task_queue=queue,
        )

        async def started() -> bool:
            return effect.exists() and effect.read_text().strip() == "START"

        await until(started)
        stop_worker(proc, tmp_path)
        await asyncio.sleep(3.2)
        assert effect.read_text().splitlines() == ["START"]
        await handle.cancel()
    finally:
        if proc is not None:
            stop_worker(proc, tmp_path)
        api.stop()
