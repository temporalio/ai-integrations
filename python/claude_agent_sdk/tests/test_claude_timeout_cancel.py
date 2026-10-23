"""Logical-turn timeout across a resume, and workflow cancellation of a live CLI.

Activity-only cancellation is in ``test_claude_turn_activity``.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import ClaudeAgentPlugin, FileLeaseStore
from temporalio.claude_agent_sdk._session import _API_RETRY_BASE_DELAY
from temporalio.client import Client, WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import CancelledError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from .conftest import scripted_cli as _cli
from .helpers import fixtures as fx
from .helpers import kill_worker as kw
from .helpers.workflows import ClaudeLive, ClaudeStopped, ClaudeTimed

# A raise, unlike a marker, also tells type checkers that the POSIX calls below are unreachable.
if sys.platform == "win32":
    raise pytest.skip.Exception(fx.POSIX_ONLY_REASON, allow_module_level=True)

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
# Left on the deadline after the API-error backoff. The CLI then blocks longer than this,
# so only a deadline that was not restarted cancels the resumed attempt.
_SLACK = timedelta(seconds=15)
_CLI_SLEEP_S = int(_SLACK.total_seconds()) + 5
# Heartbeats start after the CLI is resolved. Two seconds loses that race
# when the suite runs in parallel, and the attempt is cancelled before the
# hang. Five still bounds how long a killed worker's attempt stays open.
_HEARTBEAT_S = 5
_KILL_BOUND_S = 20


class _MemoryStore:
    """SessionStore marker: heartbeats say the attempt is resumable. Nothing is shared."""

    async def append(self, _key: Any, _entries: list[Any]) -> None:
        return None

    async def load(self, _key: Any) -> list[Any] | None:
        return None


def _plugin(cli: str, root: str | Path, **extra: Any) -> ClaudeAgentPlugin:
    """The plugin for these tests. ``extra`` is untyped so a partial session store passes."""
    options: dict[str, Any] = {
        "cli_path": cli,
        "workspace_root": root,
        "disconnect_timeout": timedelta(seconds=1),
        **extra,
    }
    return ClaudeAgentPlugin(**options)


async def _worker_main(spec: dict[str, Any]) -> None:
    client = await Client.connect(
        spec["target"],
        namespace=spec["namespace"],
        data_converter=pydantic_data_converter,
    )
    async with Worker(
        client,
        task_queue=spec["task_queue"],
        workflows=[ClaudeLive],
        max_cached_workflows=0,
        plugins=[
            _plugin(
                spec["cli"],
                spec["workspace"],
                session_store=_MemoryStore(),
                lease_store=FileLeaseStore(spec["leases"], shared=True),
                interrupt_timeout=timedelta(milliseconds=200),
                drain_timeout=timedelta(milliseconds=200),
            )
        ],
    ):
        Path(spec["ready"]).write_text("up")
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(_worker_main(json.loads(sys.argv[1])))


@pytest.fixture(autouse=True)
def _config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))


async def test_a_logical_timeout_is_not_reset_when_the_turn_resumes(
    tmp_path: Path,
) -> None:
    """A resume after an API error keeps the message's deadline instead of starting a new one."""
    timeout = _API_RETRY_BASE_DELAY + _SLACK
    store = _MemoryStore()
    state = tmp_path / "state.json"
    async with await WorkflowEnvironment.start_time_skipping() as skipping:
        client = kw.client(skipping)
        cli, _record = _cli(
            tmp_path / "cli",
            turn=[
                {"fake": "trigger", "point": "before_prompt_persisted"},
                {"fake": "sleep", "seconds": _CLI_SLEEP_S},
                *fx.COMPLETED_ONLY,
            ],
            state=str(state),
            crash_at="before_prompt_persisted",
            crash_mode="api_error",
            crash_prompt="What's the weather?",
            trigger_settle=0,
            trigger_dir=str(tmp_path / "triggers"),
            log=str(tmp_path / "cli.jsonl"),
        )
        task_queue = f"claude-{uuid.uuid4().hex[:8]}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ClaudeTimed],
            plugins=[
                _plugin(
                    cli,
                    tmp_path / "workspaces",
                    session_store=store,
                    interrupt_timeout=timedelta(milliseconds=200),
                    drain_timeout=timedelta(milliseconds=200),
                )
            ],
        ):
            elapsed = await client.execute_workflow(
                ClaudeTimed.run,
                int(timeout.total_seconds()),
                id=task_queue,
                task_queue=task_queue,
            )

    # Time-skipping can deliver the resumed attempt's result in the same task that
    # first sees the deadline already passed, so the observed time is that task's
    # time: after the backoff, and before a timeout restarted on the resume.
    backoff = _API_RETRY_BASE_DELAY.total_seconds()
    assert backoff < elapsed < backoff + timeout.total_seconds()
    # The state file is incremented once per CLI process, before the crash or the sleep.
    assert json.loads(state.read_text())["runs"] == 2


async def test_cancelling_the_workflow_kills_the_live_cli(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """The cancelled workflow ends ``CancelledError`` only after the CLI's group is gone."""
    client = kw.client(env)
    cli, record = _cli(
        tmp_path / "cli",
        turn=fx.STREAMING_UNTIL_INTERRUPT,
        hang_after_prompt=True,
        interrupt_result=fx.INTERRUPTED_RESULT,
        child=["sleep", "60"],
    )
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeLive],
        plugins=[
            _plugin(
                cli,
                tmp_path / "workspaces",
                interrupt_timeout=timedelta(milliseconds=200),
                drain_timeout=timedelta(milliseconds=200),
            )
        ],
    ):
        handle = await client.start_workflow(
            ClaudeLive.run, 30, id=task_queue, task_queue=task_queue
        )
        await kw.until(Path(str(record) + ".ready").exists)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()

    assert isinstance(err.value.cause, CancelledError)
    pids = json.loads(record.read_text())
    assert not kw.alive(pids["pid"])
    assert not kw.alive(pids["child_pid"])


async def test_cancelling_a_streaming_turn_is_delivered_within_seconds(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """With the default 30 s heartbeat timeout the SDK would hold a cancellation for its
    throttled heartbeat (24 s); the plugin caps that, so the interrupt starts within a second."""
    client = kw.client(env)
    cli, record = _cli(
        tmp_path / "cli",
        turn=[
            *fx.STREAMING_UNTIL_INTERRUPT,
            *[{"fake": "sleep", "seconds": 0.25}] * 120,
        ],
        listen_while_streaming=True,
        interrupt_result=fx.INTERRUPTED_RESULT,
    )
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeLive],
        plugins=[_plugin(cli, tmp_path / "workspaces")],
    ):
        handle = await client.start_workflow(
            ClaudeLive.run, 30, id=task_queue, task_queue=task_queue
        )
        await kw.until(record.exists)
        await asyncio.sleep(1)
        started = time.monotonic()
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()
        elapsed = time.monotonic() - started

    assert isinstance(err.value.cause, CancelledError)
    assert elapsed < 6, f"cancellation took {elapsed:.1f}s"


async def test_a_stopped_message_charges_the_cost_its_interrupted_run_reported(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """A cancelled turn returns no result, so the cost the interrupted CLI reports while
    draining reaches the session (and its ``max_cost_usd``) through a progress signal."""
    client = kw.client(env)
    cli, record = _cli(
        tmp_path / "cli",
        turns=[["first", fx.STREAMING_UNTIL_INTERRUPT], ["second", fx.COMPLETED_ONLY]],
        hang_after_prompt=True,
        interrupt_result={**fx.INTERRUPTED_RESULT, "total_cost_usd": 0.05},
    )
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeStopped],
        plugins=[
            _plugin(
                cli,
                tmp_path / "workspaces",
                interrupt_timeout=timedelta(milliseconds=500),
                drain_timeout=timedelta(seconds=2),
            )
        ],
    ):
        handle = await client.start_workflow(
            ClaudeStopped.run,
            args=["first", "second"],
            id=task_queue,
            task_queue=task_queue,
        )
        await kw.until(Path(str(record) + ".ready").exists)
        await handle.signal(ClaudeStopped.stop)
        result = await handle.result()

    completed = fx.COMPLETED_ONLY[-1]["total_cost_usd"]
    assert result["cost_usd"] == pytest.approx(0.05 + completed), result


@pytest.mark.timeout(120)
async def test_cancelling_as_the_worker_is_killed_does_not_resume(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """The worker is SIGKILLed, then the workflow is cancelled, with a second worker polling.

    The dead attempt never hears the cancel, so it ends by heartbeat timeout like a lost
    attempt. The workflow still ends ``CancelledError`` and the second worker runs no turn.
    """
    client = kw.client(env)
    hang, record = _cli(
        tmp_path / "hang",
        turn=[{"fake": "trigger", "point": "after_tool_call"}],
        crash_at="after_tool_call",
        trigger_settle=0,
        trigger_block=30,
        trigger_dir=str(tmp_path / "triggers"),
        log=str(tmp_path / "hang.jsonl"),
    )
    done, _done_record = _cli(
        tmp_path / "done",
        turn=fx.COMPLETED_ONLY,
        log=str(tmp_path / "done.jsonl"),
    )
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    leases = tmp_path / "leases"
    ready = tmp_path / "worker-a-ready"
    spec = {
        "target": env.client.service_client.config.target_host,
        "namespace": env.client.namespace,
        "task_queue": task_queue,
        "cli": hang,
        "workspace": str(tmp_path / "workspaces"),
        "leases": str(leases),
        "ready": str(ready),
    }
    worker_log = (tmp_path / "worker-a.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.test_claude_timeout_cancel", json.dumps(spec)],
        cwd=_PLUGIN_ROOT,
        env={
            **os.environ,
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude-config"),
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
        },
        stdout=worker_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    cli_pid = 0
    try:
        await kw.until(ready.exists)
        handle = await client.start_workflow(
            ClaudeLive.run, _HEARTBEAT_S, id=task_queue, task_queue=task_queue
        )
        marker = tmp_path / "triggers" / "after_tool_call"
        try:
            await kw.until(marker.exists, timeout=45)
        except TimeoutError:
            log = (tmp_path / "worker-a.log").read_text()[-4000:]
            raise AssertionError(f"CLI never reached the hang:\n{log}") from None
        cli_pid = int(json.loads(record.read_text())["pid"])
        # Cached: temporalio 1.34.0 hangs evicting a workflow whose Activity cancel is
        # pending (fixed by temporalio/sdk-python#1947), which max_cached_workflows=0 forces.
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ClaudeLive],
            plugins=[
                _plugin(
                    done,
                    tmp_path / "workspaces",
                    session_store=_MemoryStore(),
                    lease_store=FileLeaseStore(leases, shared=True),
                )
            ],
        ):
            kw.killpg(proc.pid)
            proc.wait(timeout=5)
            await handle.cancel()
            with pytest.raises(WorkflowFailureError) as err:
                await asyncio.wait_for(handle.result(), _KILL_BOUND_S)
    finally:
        kw.killpg(proc.pid)
        if cli_pid:
            kw.killpg(cli_pid)
        if proc.poll() is None:
            proc.wait(timeout=5)
        worker_log.close()

    assert isinstance(err.value.cause, CancelledError)
    assert not (tmp_path / "done.jsonl").exists()


@pytest.mark.timeout(120)
async def test_sigkill_of_the_worker_kills_the_hanging_cli_group(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """SIGKILL of the worker's group also ends the CLI hung in its own session.

    Nothing signals the CLI before the check; the ``finally`` reap runs after it.
    """
    client = kw.client(env)
    cli, record = _cli(
        tmp_path / "hang",
        turn=[{"fake": "trigger", "point": "after_tool_call"}],
        crash_at="after_tool_call",
        trigger_settle=0,
        trigger_block=120,
        trigger_dir=str(tmp_path / "triggers"),
        child=["sleep", "120"],
        detach_child=True,
    )
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    ready = tmp_path / "worker-ready"
    spec = {
        "target": env.client.service_client.config.target_host,
        "namespace": env.client.namespace,
        "task_queue": task_queue,
        "cli": cli,
        "workspace": str(tmp_path / "workspaces"),
        "leases": str(tmp_path / "leases"),
        "ready": str(ready),
    }
    worker_log = (tmp_path / "worker.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.test_claude_timeout_cancel", json.dumps(spec)],
        cwd=_PLUGIN_ROOT,
        env={
            **os.environ,
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude-config"),
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
        },
        stdout=worker_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    cli_pid = 0
    try:
        await kw.until(ready.exists)
        await client.start_workflow(
            ClaudeLive.run, _HEARTBEAT_S, id=task_queue, task_queue=task_queue
        )
        marker = tmp_path / "triggers" / "after_tool_call"
        try:
            await kw.until(marker.exists, timeout=45)
        except TimeoutError:
            log = (tmp_path / "worker.log").read_text()[-4000:]
            raise AssertionError(f"CLI never reached the hang:\n{log}") from None
        pids = json.loads(record.read_text())
        cli_pid = int(pids["pid"])
        child_pid = int(pids["child_pid"])
        assert os.getpgid(cli_pid) == cli_pid
        assert os.getpgid(cli_pid) != os.getpgid(proc.pid)
        assert os.getpgid(child_pid) != cli_pid
        kw.killpg(proc.pid)
        proc.wait(timeout=5)
        try:
            await kw.until(
                lambda: not kw.group_alive(cli_pid) and not kw.alive(child_pid),
                timeout=5,
            )
        except TimeoutError:
            raise AssertionError(
                f"CLI group alive={kw.group_alive(cli_pid)} "
                f"descendant alive={kw.alive(child_pid)}"
            ) from None
    finally:
        kw.killpg(proc.pid)
        if cli_pid:
            kw.killpg(cli_pid)
        if proc.poll() is None:
            proc.wait(timeout=5)
        worker_log.close()


_ = _config_dir
