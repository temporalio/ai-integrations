"""Durability of a turn that loses its worker: tool dedup, transcript failover, budget."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk import project_key_for_directory

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    FileLeaseStore,
    FileSessionStore,
)
from temporalio.client import Client, WorkflowHandle
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from .conftest import scripted_cli as _cli
from .helpers import fixtures as fx
from .helpers import kill_worker as kw
from .helpers.workflows import ClaudeDurable, ClaudeHeldTool

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]

# A raise, unlike a marker, also tells type checkers that the POSIX calls below are unreachable.
if sys.platform == "win32":
    raise pytest.skip.Exception(fx.POSIX_ONLY_REASON, allow_module_level=True)

PROMPT = "Weather in Paris?"


@pytest.fixture(autouse=True)
def _config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))


class _SessionStore:
    """In-process SessionStore: append and load, keyed by project and session."""

    def __init__(self) -> None:
        self.rows: dict[str, list[Any]] = {}

    async def append(self, key: Any, entries: list[Any]) -> None:
        self.rows.setdefault(_store_key(key), []).extend(entries)

    async def load(self, key: Any) -> list[Any] | None:
        return self.rows.get(_store_key(key))


def _store_key(key: Any) -> str:
    return f"{key['project_key']}/{key['session_id']}"


@asynccontextmanager
async def _worker(
    env: WorkflowEnvironment,
    tmp_path: Path,
    scenario: dict[str, Any],
    store: _SessionStore,
    workflows: list[type[Any]] | None = None,
) -> AsyncIterator[tuple[Client, str]]:
    client = kw.client(env)
    cli, _ = _cli(tmp_path / "cli", **scenario)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=workflows or [ClaudeDurable],
        plugins=[
            ClaudeAgentPlugin(
                cli_path=cli,
                workspace_root=tmp_path / "workspaces",
                session_store=cast(Any, store),
                lease_store=FileLeaseStore(tmp_path / "leases"),
                disconnect_timeout=timedelta(seconds=1),
            )
        ],
    ):
        yield client, task_queue


async def test_a_workflow_tool_runs_once_after_a_lost_attempt(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """The CLI dies after get_weather ran. The resumed attempt re-issues that call, and the
    workflow answers it from the record instead of running the tool again."""
    store = _SessionStore()
    async with _worker(
        env,
        tmp_path,
        {
            "plans": [
                [
                    "Weather",
                    [
                        {"tool": "get_weather", "input": {"city": "Paris"}},
                        {"text": "All done."},
                    ],
                ]
            ],
            "crash_at": "after_tool_run",
            "crash_mode": "die",
            "crash_prompt": "Weather",
            "trigger_settle": 0,
            "trigger_dir": str(tmp_path / "triggers"),
            "log": str(tmp_path / "cli.jsonl"),
            "mirror_lag_rounds": 1,
        },
        store,
    ) as (client, task_queue):
        handle: WorkflowHandle[ClaudeDurable, str] = await client.start_workflow(
            ClaudeDurable.run, id=task_queue, task_queue=task_queue
        )
        assert await handle.result() == "All done."
        assert await handle.query(ClaudeDurable.weather_calls) == 1


def _tool_blocks(config: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in config.rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            content = (json.loads(line).get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            found.extend(
                block
                for block in content
                if isinstance(block, dict)
                and block.get("type") in ("tool_use", "tool_result")
            )
    return found


async def test_a_reissued_tool_attaches_while_the_call_is_still_running(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """The CLI is killed while get_weather is waiting. Before the resumed CLI starts, the
    Activity asks the workflow about that call under its own tool_use_id; it waits for the
    execution already running and its result is stored as the call's answer. The side-effect
    file gains one line."""
    effect = tmp_path / "weather-effect.txt"
    store = _SessionStore()
    async with _worker(
        env,
        tmp_path,
        {
            "plans": [
                [
                    "Weather",
                    [
                        {"tool": "get_weather", "input": {"city": "Paris"}},
                        {"text": "All done."},
                    ],
                ]
            ],
            "log": str(tmp_path / "cli.jsonl"),
            "mirror_lag_rounds": 0,
        },
        store,
        workflows=[ClaudeHeldTool],
    ) as (client, task_queue):
        handle: WorkflowHandle[ClaudeHeldTool, str] = await client.start_workflow(
            ClaudeHeldTool.run,
            str(effect),
            id=task_queue,
            task_queue=task_queue,
        )
        result: asyncio.Task[str] = asyncio.create_task(handle.result())
        try:
            await kw.until(
                lambda: _tool_started(handle, effect), timeout=20, interval=0
            )
            pid = int(json.loads((tmp_path / "cli" / "record.json").read_text())["pid"])
            os.kill(pid, signal.SIGKILL)
            await kw.until(lambda: not kw.alive(pid), timeout=20, interval=0)

            async def _attached() -> bool:
                calls = await handle.query(ClaudeHeldTool.calls)
                if calls == 0:
                    return False
                assert calls == 1
                if not effect.is_file() or effect.read_text() != "Paris\n":
                    return False
                return await handle.query(ClaudeHeldTool.bridge_tasks) == 2

            await kw.until(_attached, timeout=20, interval=0)
            await handle.signal(ClaudeHeldTool.release)
            assert (await result) == "All done."
        finally:
            if not result.done():
                result.cancel()
                await asyncio.gather(result, return_exceptions=True)

        assert effect.read_text() == "Paris\n"
        assert await handle.query(ClaudeHeldTool.calls) == 1
        killed = [
            block["id"]
            for block in _tool_blocks(tmp_path / "claude-config")
            if block.get("type") == "tool_use"
        ]
        log = [
            json.loads(line)
            for line in (tmp_path / "cli.jsonl").read_text().splitlines()
            if line.strip()
        ]
        detail = json.dumps({"killed": killed, "log": log}, default=str)
        assert len(log) == 1, detail
        resume = log[0]
        # The Activity answered the dangling call in the store before the CLI started, so the
        # CLI patched nothing ("outcome unknown") and Claude had nothing to re-issue.
        assert killed and resume["repairs"] == ["no_response"], detail
        assert resume["mcp"] == [], detail
        answers = [
            block
            for rows in store.rows.values()
            for entry in rows
            for block in (entry.get("message") or {}).get("content") or []
            if isinstance(block, dict)
            and block.get("type") == "tool_result"
            and block["tool_use_id"] == killed[0]
        ]
        assert answers, detail
        assert answers[-1]["content"][0]["text"] == "sunny in Paris", detail
        assert not answers[-1].get("is_error"), detail


async def _tool_started(
    handle: WorkflowHandle[ClaudeHeldTool, str], effect: Path
) -> bool:
    """The held call has entered and its one side effect is on disk."""
    if await handle.query(ClaudeHeldTool.calls) != 1:
        return False
    return effect.is_file() and effect.read_text() == "Paris\n"


def _prompt_entries(store: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in store.rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            content = (entry.get("message") or {}).get("content")
            if content == PROMPT:
                found.append(entry)
    return found


@pytest.mark.timeout(120)
async def test_a_second_worker_resumes_the_transcript_after_the_first_is_killed(
    env: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker B, with an empty config dir, finishes the turn from the store after A is SIGKILLed.

    Both workers share one workspace_root: the store key is
    ``project_key_for_directory(workspace_root/session_id)``.
    """
    disk_a = tmp_path / "disk-a"
    disk_b = tmp_path / "disk-b"
    workspace = (tmp_path / "workspace").resolve()
    store = tmp_path / "store"
    leases = tmp_path / "leases"
    for path in (disk_a, disk_b, workspace):
        path.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(disk_b))
    cli, record = _cli(
        tmp_path / "cli",
        plans=[
            [
                "Weather",
                [
                    {"tool": "get_weather", "input": {"city": "Paris"}},
                    {"text": "All done."},
                ],
            ]
        ],
        crash_at="before_prompt_persisted",
        crash_prompt="Weather",
        trigger_settle=0,
        trigger_block=30,
        mirror_lag_rounds=0,
        trigger_dir=str(tmp_path / "triggers"),
        log=str(tmp_path / "cli.jsonl"),
    )
    client = kw.client(env)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    ready = tmp_path / "worker-a-ready"
    spec = {
        "target": env.client.service_client.config.target_host,
        "namespace": env.client.namespace,
        "task_queue": task_queue,
        "cli": cli,
        "workspace": str(workspace),
        "store": str(store),
        "leases": str(leases),
        "ready": str(ready),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.helpers.durability_worker", json.dumps(spec)],
        cwd=_PLUGIN_ROOT,
        env={
            **os.environ,
            "CLAUDE_CONFIG_DIR": str(disk_a),
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
            "PYTHONPATH": str(_PLUGIN_ROOT),
        },
        start_new_session=True,
    )
    cli_pid = 0
    try:
        await kw.until(ready.exists)
        handle: WorkflowHandle[ClaudeDurable, str] = await client.start_workflow(
            ClaudeDurable.run, id=task_queue, task_queue=task_queue
        )
        await kw.until(lambda: bool(_prompt_entries(store)), timeout=40)
        cli_pid = int(json.loads(record.read_text())["pid"])
        kw.killpg(proc.pid)
        # The watchdog kills this group. The signal is the backstop so resume
        # cannot race a live CLI.
        kw.killpg(cli_pid)
        proc.wait(timeout=5)
        await kw.until(lambda: not kw.alive(cli_pid))
        shutil.rmtree(disk_a)
        assert not disk_a.exists()
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ClaudeDurable],
            max_cached_workflows=0,
            plugins=[
                ClaudeAgentPlugin(
                    cli_path=cli,
                    workspace_root=workspace,
                    session_store=cast(Any, FileSessionStore(store)),
                    lease_store=FileLeaseStore(leases),
                    disconnect_timeout=timedelta(seconds=1),
                )
            ],
        ):
            assert await handle.result() == "All done."
            assert await handle.query(ClaudeDurable.weather_calls) == 1
    finally:
        kw.killpg(proc.pid)
        if cli_pid:
            kw.killpg(cli_pid)
        if proc.poll() is None:
            proc.wait(timeout=5)

    files = list(store.rglob("*.jsonl"))
    assert len(files) == 1
    session_id = files[0].stem
    assert files[0].parent.name == project_key_for_directory(workspace / session_id), (
        "store key is project_key_for_directory(workspace_root/session_id)"
    )
    originals = _prompt_entries(store)
    assert len(originals) == 1 and originals[0]["uuid"]
    logged = [
        json.loads(line)
        for line in (tmp_path / "cli.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(logged) == 1
    assert logged[0]["prompt"].startswith("Continue")
    assert any(arg.startswith("--resume=") for arg in logged[0]["argv"])
    assert not kw.alive(cli_pid)


_ = _config_dir
