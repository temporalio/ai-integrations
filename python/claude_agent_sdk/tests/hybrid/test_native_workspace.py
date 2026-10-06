"""Real native Read/Edit across Worker loss and disposable workspace restoration."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    SessionKey,
    SessionStoreEntry,
    project_key_for_directory,
)

from temporalio.client import Client, WorkflowFailureError
from temporalio.worker import Replayer, Worker
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env, history_of
from tests.hybrid.models import Attempt, Call, State
from tests.hybrid.native import NativeActivities
from tests.hybrid.native_store import NativeStore
from tests.hybrid.test_engine import alive
from tests.hybrid.test_worker_loss import launch
from tests.hybrid.test_workflow import until
from tests.hybrid.workflows import HybridWorkflow

pytestmark = pytest.mark.timeout(120)
recovery_sdk = pytest.mark.skipif(
    not hasattr(ClaudeAgentOptions, "recover_pending_tool"),
    reason="requires the sibling main-agent recovery SDK wheel",
)


def native_api(root: Path) -> tuple[FakeMessagesAPI, dict[str, str]]:
    emitted: dict[str, str] = {}
    holder: list[FakeMessagesAPI] = []

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, texts, history = history_of(body)
        wanted = next(
            (t for t in reversed(texts) if t in {"native", "read", "edit"}), "native"
        )
        if any(h.is_error for h in history):
            return [{"type": "text", "text": "NATIVE ERROR"}]
        done = {h.name for h in history}
        name = "Read" if "Read" not in done else "Edit"
        if wanted == "read" and "Read" in done:
            return [{"type": "text", "text": "READ DONE"}]
        if "Edit" in done:
            return [{"type": "text", "text": "EDIT DONE"}]
        tid = holder[0].next_id("toolu_native")
        emitted[tid] = name
        arguments: dict[str, Any] = {"file_path": str(root / "workspace" / "note.txt")}
        if name == "Edit":
            arguments.update(old_string="BEFORE", new_string="AFTER")
        return [{"type": "tool_use", "id": tid, "name": name, "input": arguments}]

    api = FakeMessagesAPI(policy, primary_tools={"Read", "Edit"})
    holder.append(api)
    return api.start(), emitted


def native_requests(api: FakeMessagesAPI) -> list[dict[str, Any]]:
    return [
        body
        for body in api.requests
        if any(t["name"] in {"Read", "Edit"} for t in body.get("tools", []))
    ]


def stop_worker(proc: subprocess.Popen[bytes], root: Path) -> None:
    log = root / "cli-pids.jsonl"
    processes = (
        [
            r
            for r in map(json.loads, log.read_text().splitlines())
            if r["worker"] == proc.pid
        ]
        if log.exists()
        else []
    )
    if proc.poll() is None:
        proc.kill()
    # A supervising host owns this cleanup; Python teardown cannot run after
    # SIGKILL. Never kill a process that was not recorded by this test Worker.
    for recorded in processes:
        pid = recorded["pid"]
        if alive(pid):
            with suppress(ProcessLookupError):
                # A supervised PID is the launcher, not its child engine. Let
                # it finish process-tree cleanup instead of SIGKILLing it.
                os.kill(
                    pid,
                    signal.SIGTERM if recorded.get("supervised") else signal.SIGKILL,
                )
    proc.wait()
    supervised = [r["pid"] for r in processes if r.get("supervised")]
    deadline = time.monotonic() + 5
    while any(alive(pid) for pid in supervised) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not any(alive(pid) for pid in supervised), "engine supervisor did not stop"


async def assert_native_history(
    handle: Any, state: State, emitted: dict[str, str], api: FakeMessagesAPI
) -> None:
    assert set(state.ledger) == set(emitted)
    assert all(e.outcome and not e.outcome.is_error for e in state.ledger.values())
    history = await handle.fetch_history()
    schedules = [
        e.activity_task_scheduled_event_attributes.activity_id
        for e in history.events
        if e.HasField("activity_task_scheduled_event_attributes")
        and e.activity_task_scheduled_event_attributes.activity_id.startswith("tool-")
    ]
    assert set(schedules) == {"tool-" + tid for tid in emitted}
    assert len(schedules) == len(set(schedules))
    try:
        await Replayer(workflows=[HybridWorkflow]).replay_workflow(history)
    except Exception:
        # Leave a standalone reproduction when server/SDK replay rejects a
        # history that a live replacement Worker was able to complete.
        (
            Path(tempfile.gettempdir()) / (state.session_id + "-native-history.json")
        ).write_text(history.to_json())
        raise
    uses, _, results = history_of(native_requests(api)[-1])
    assert set(uses) == set(emitted) == {h.id for h in results}
    assert not any(h.is_error for h in results)
    assert api.errors == []


async def test_native_read_edit_are_individually_scheduled(
    client: Client, tmp_path: Path
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    acts = NativeActivities(
        client, tmp_path, engine_env(api, str(tmp_path / "cfg")), store
    )
    queue = "native-live-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[HybridWorkflow],
            activities=[acts.burst, acts.tool],
        ):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["native"]),
                id=queue,
                task_queue=queue,
            )
            state = await asyncio.wait_for(handle.result(), 40)
        assert state.answers == ["EDIT DONE"]
        assert (
            store.durable_text()
            == (store.workspace / "note.txt").read_text()
            == "AFTER\n"
        )
        assert sorted(emitted.values()) == ["Edit", "Read"]
        assert len(acts.bursts) == 1
        assert len(native_requests(api)) == 3
        await assert_native_history(handle, state, emitted, api)
    finally:
        api.stop()


@recovery_sdk
@pytest.mark.parametrize(
    "phase",
    [
        "Read-after-commit-before-completion",
        "Read-after-commit",
        "Edit-after-commit-before-completion",
        "Edit-after-commit",
    ],
)
async def test_native_worker_loss_reuses_committed_results(
    client: Client, address: str, tmp_path: Path, phase: str
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    queue = "native-recovery-" + uuid.uuid4().hex
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
            )
        )
        session = str(uuid.uuid4())
        prompt = "read" if phase.startswith("Read-") else "native"
        handle = await client.start_workflow(
            HybridWorkflow.run, State(session, [prompt]), id=queue, task_queue=queue
        )

        async def committed() -> Any:
            snapshot = await handle.query(HybridWorkflow.snapshot)
            for tid, row in store.executions().items():
                if row["name"] == phase.split("-")[0] and row["result"] is not None:
                    entry = snapshot.ledger.get(tid)
                    if entry and (
                        phase.endswith("before-completion") or entry.outcome is not None
                    ):
                        return snapshot, tid, row
            return None

        before, tid, row = await until(committed)
        original_ids = set(before.ledger)
        expected_text = "BEFORE\n" if prompt == "read" else "AFTER\n"
        assert store.durable_text() == expected_text
        key: SessionKey = {
            "project_key": project_key_for_directory(str(store.workspace)),
            "session_id": session,
        }
        transcripts = await store.transcripts(key)
        assert not any(
            b.get("tool_use_id") == tid
            for e in transcripts[""]
            for b in e.get("message", {}).get("content", [])
            if isinstance(b, dict)
        )
        stop_worker(procs[0], tmp_path)
        shutil.rmtree(store.workspace)
        shutil.rmtree(tmp_path / "machine-1", ignore_errors=True)
        assert not store.workspace.exists()
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
        assert state.answers == ["READ DONE" if prompt == "read" else "EDIT DONE"]
        assert set(state.ledger) == original_ids == set(emitted)
        assert json.loads(state.ledger[tid].outcome.text) == row["result"]  # type: ignore[union-attr]
        assert (
            store.durable_text()
            == (store.workspace / "note.txt").read_text()
            == expected_text
        )
        assert (store.workspace / "note.txt").stat().st_mode & 0o777 == 0o640
        with store.connect() as db:
            executed = [
                r[0]
                for r in db.execute(
                    "SELECT id FROM native_events WHERE phase='executed'"
                )
            ]
        assert sorted(executed) == sorted(original_ids)
        await assert_native_history(handle, state, emitted, api)
        assert len((tmp_path / "cli-pids.jsonl").read_text().splitlines()) == 2
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()


@recovery_sdk
@pytest.mark.parametrize(
    "phase", ["Read-before-execution", "Edit-before-execution", "Edit-after-write"]
)
async def test_uncommitted_native_execution_blocks_recovery(
    client: Client, address: str, tmp_path: Path, phase: str
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    queue = "native-blocker-" + uuid.uuid4().hex
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
            )
        )
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["native"]),
            id=queue,
            task_queue=queue,
        )

        async def held() -> Any:
            snapshot = await handle.query(HybridWorkflow.snapshot)
            wanted = (
                "executed" if phase.endswith("after-write") else "held-before-execution"
            )
            return (
                snapshot
                if any(
                    r["name"] == phase.split("-")[0] and r["phase"] == wanted
                    for r in store.executions().values()
                )
                else None
            )

        before = await until(held)
        await handle.execute_update(HybridWorkflow.suspend)
        await asyncio.sleep(0.15)
        assert (await handle.query(HybridWorkflow.snapshot)).pending is None
        pids = [
            row["pid"]
            for row in map(
                json.loads, (tmp_path / "cli-pids.jsonl").read_text().splitlines()
            )
        ]
        assert all(alive(pid) for pid in pids)
        if phase.endswith("after-write"):
            assert (store.workspace / "note.txt").read_text() == "AFTER\n"
        assert store.durable_text() == "BEFORE\n"
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
        with pytest.raises(WorkflowFailureError) as failed:
            await asyncio.wait_for(handle.result(), 45)
        cause: Any = failed.value
        while getattr(cause, "cause", None) is not None:
            cause = cause.cause
        assert "no committed execution result" in str(cause)
        after = await handle.query(HybridWorkflow.snapshot)
        assert set(after.ledger) == set(before.ledger) == set(emitted)
        assert (
            store.durable_text()
            == (store.workspace / "note.txt").read_text()
            == "BEFORE\n"
        )
        assert len((tmp_path / "cli-pids.jsonl").read_text().splitlines()) == 1
        assert api.errors == []
        await Replayer(workflows=[HybridWorkflow]).replay_workflow(
            await handle.fetch_history()
        )
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()


@recovery_sdk
async def test_read_context_restores_before_native_edit(
    client: Client, address: str, tmp_path: Path
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    queue = "native-read-context-" + uuid.uuid4().hex
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
                native_phase="between-turns",
            )
        )
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["read", "edit"]),
            id=queue,
            task_queue=queue,
        )

        async def read_finished() -> Any:
            snapshot = await handle.query(HybridWorkflow.snapshot)
            with store.connect() as db:
                held = db.execute(
                    "SELECT 1 FROM native_events WHERE phase='between-turns'"
                ).fetchone()
            return snapshot if held and 0 in snapshot.turns else None

        before = await until(read_finished)
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
        assert state.answers == ["READ DONE", "EDIT DONE"]
        assert state.turns[0] == before.turns[0]
        assert sorted(emitted.values()) == ["Edit", "Read"]
        assert store.durable_text() == "AFTER\n"
        await assert_native_history(handle, state, emitted, api)
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()


@recovery_sdk
@pytest.mark.parametrize(
    "phase", ["Read-before-execution", "Edit-before-execution", "Edit-after-write"]
)
async def test_ordinary_resume_interrupts_pending_native_calls(
    client: Client, address: str, tmp_path: Path, phase: str
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    queue = "native-ordinary-" + uuid.uuid4().hex
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
            )
        )
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["native"]),
            id=queue,
            task_queue=queue,
        )

        async def held() -> Any:
            snapshot = await handle.query(HybridWorkflow.snapshot)
            wanted = (
                "executed" if phase.endswith("after-write") else "held-before-execution"
            )
            for tid, row in store.executions().items():
                if row["name"] == phase.split("-")[0] and row["phase"] == wanted:
                    return snapshot, tid
            return None

        before, pending = await until(held)
        stop_worker(procs[0], tmp_path)
        shutil.rmtree(store.workspace)
        shutil.rmtree(tmp_path / "machine-1", ignore_errors=True)
        # This mode disables the opt-in recovery callback. If the CLI really
        # re-dispatches a native call, the probe records the hook and denies it
        # before any effect; that would disprove the observed interruption limit.
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                2,
                False,
                recovery=True,
                native_phase="ordinary-probe",
            )
        )
        with pytest.raises(WorkflowFailureError):
            await asyncio.wait_for(handle.result(), 45)
        requests = native_requests(api)
        _, _, results = history_of(requests[-1])
        interrupted = next(h for h in results if h.id == pending)
        assert (
            interrupted.is_error
            and "interrupted" in json.dumps(interrupted.content).lower()
        )
        with store.connect() as db:
            assert not db.execute(
                "SELECT 1 FROM native_events WHERE phase='ordinary-hook'"
            ).fetchone()
        assert set(emitted) == set(before.ledger)
        assert (
            store.durable_text()
            == (store.workspace / "note.txt").read_text()
            == "BEFORE\n"
        )
        assert api.errors == []
        print(
            "NATIVE_RESUME_BLOCKER "
            + json.dumps(
                {
                    "phase": phase,
                    "native_id": pending,
                    "engine_result": interrupted.content,
                }
            ),
            flush=True,
        )
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()


def test_stale_native_workspace_writer_is_fenced(tmp_path: Path) -> None:
    old = NativeStore(tmp_path)
    old.initialize("BEFORE\n")
    first = Attempt(0, 1, "first")
    old.checkout(first)
    call = Call(
        "native-original",
        "Edit",
        {"file_path": str(old.workspace / "note.txt")},
        first,
        "stored-uuid",
    )
    old.prepare(call)
    replacement = NativeStore(tmp_path)
    replacement.checkout(Attempt(0, 2, "replacement"))
    with pytest.raises(RuntimeError, match="stale workspace snapshot"):
        old.stage(call.id)
    with pytest.raises(RuntimeError, match="stale native result writer"):
        old.capture(
            cast(
                list[SessionStoreEntry],
                [
                    {
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": call.id,
                                    "content": "old",
                                }
                            ]
                        }
                    }
                ],
            )
        )
    assert replacement.durable_text() == "BEFORE\n"
