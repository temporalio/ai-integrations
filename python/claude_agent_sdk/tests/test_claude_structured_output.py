"""A typed Claude result survives the turn, a lost attempt, and replay.

The happy-path history is committed and replayed by ``test_claude_replay``. Re-record it
with ``CLAUDE_RECORD_HISTORIES=1 make test PYTEST_ARGS=tests/test_claude_structured_output.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import pytest

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    FileLeaseStore,
    FileSessionStore,
)
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from .conftest import PLUGIN_ROOT
from .conftest import scripted_cli as _cli
from .helpers import fixtures as fx
from .helpers import histories
from .helpers import kill_worker as kw
from .helpers.workflows import ClaudeStructured, Forecast
from .test_claude_workflow import _service

# A raise, unlike a marker, also tells type checkers that the POSIX calls below are unreachable.
if sys.platform == "win32":
    raise pytest.skip.Exception(fx.POSIX_ONLY_REASON, allow_module_level=True)

# Heartbeats start after the CLI is resolved. Two seconds loses that race
# when the suite runs in parallel, and the attempt is cancelled before the
# prompt is mirrored. Five still bounds how long a killed worker's attempt stays open.
_HEARTBEAT_S = 5


_FORECAST = {"city": "Paris", "temp_f": 72}
_PROMPT = "forecast"


def _schema(argv: list[str]) -> Any:
    return json.loads(argv[argv.index("--json-schema") + 1])


def _reap(pid: int) -> None:
    """SIGKILL ``pid`` and its process group. Leak net: the watchdog should already have."""
    if pid <= 0:
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    kw.killpg(pgid)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def _plugin(cli: str, root: Path, store: Path, leases: Path) -> ClaudeAgentPlugin:
    return ClaudeAgentPlugin(
        cli_path=cli,
        workspace_root=root,
        session_store=cast(Any, FileSessionStore(store)),
        lease_store=FileLeaseStore(leases),
    )


def _stored_prompts(store: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in store.rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if (entry.get("message") or {}).get("content") == _PROMPT:
                found.append(entry)
    return found


async def _worker_main(spec: dict[str, Any]) -> None:
    client = await Client.connect(
        spec["target"],
        namespace=spec["namespace"],
        data_converter=pydantic_data_converter,
    )
    async with Worker(
        client,
        task_queue=spec["task_queue"],
        workflows=[ClaudeStructured],
        max_cached_workflows=0,
        plugins=[
            _plugin(
                spec["cli"],
                Path(spec["workspace"]),
                Path(spec["store"]),
                Path(spec["leases"]),
            )
        ],
    ):
        Path(spec["ready"]).write_text("up")
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(_worker_main(json.loads(sys.argv[1])))


async def test_a_turn_with_an_output_type_returns_the_validated_object(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    log = tmp_path / "cli.jsonl"
    async with _service(
        env,
        tmp_path,
        [ClaudeStructured],
        {"turn": fx.STRUCTURED, "log": str(log)},
    ) as (client, task_queue, _cli_path):
        handle = await client.start_workflow(
            ClaudeStructured.run, 30, id=task_queue, task_queue=task_queue
        )
        assert await handle.result() == Forecast(city="Paris", temp_f=72)
        await histories.record("structured_output", handle)

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [_schema(row["argv"]) for row in rows] == [Forecast.model_json_schema()]


@pytest.mark.timeout(120)
async def test_a_lost_attempt_resumes_with_the_output_schema(
    env: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker A mirrors the prompt, then its process group is SIGKILLed and its config dir is
    removed. Worker B resumes that transcript and still sends the output schema.
    """
    disk_a = tmp_path / "disk-a"
    disk_b = tmp_path / "disk-b"
    workspace = (tmp_path / "workspaces").resolve()
    store = tmp_path / "store"
    for path in (disk_a, disk_b, workspace):
        path.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(disk_b))
    log = tmp_path / "cli.jsonl"
    cli, record = _cli(
        tmp_path / "cli",
        plans=[
            [
                _PROMPT,
                [{"text": json.dumps(_FORECAST), "structured_output": _FORECAST}],
            ]
        ],
        crash_at="before_prompt_persisted",
        crash_prompt=_PROMPT,
        trigger_settle=0,
        trigger_block=30,
        mirror_lag_rounds=0,
        trigger_dir=str(tmp_path / "triggers"),
        log=str(log),
    )
    client = kw.client(env)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    leases = tmp_path / "leases"
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
    worker_log = (tmp_path / "worker-a.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.test_claude_structured_output", json.dumps(spec)],
        cwd=PLUGIN_ROOT,
        env={
            **os.environ,
            "CLAUDE_CONFIG_DIR": str(disk_a),
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
        },
        stdout=worker_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    cli_pid = 0
    try:
        await kw.until(ready.exists, timeout=20)
        handle = await client.start_workflow(
            ClaudeStructured.run,
            _HEARTBEAT_S,
            id=task_queue,
            task_queue=task_queue,
        )
        try:
            await kw.until(lambda: bool(_stored_prompts(store)), timeout=40)
        except TimeoutError:
            logged = (tmp_path / "worker-a.log").read_text()[-4000:]
            raise AssertionError(f"prompt never reached the store:\n{logged}") from None
        stored = _stored_prompts(store)
        assert len(stored) == 1 and stored[0]["uuid"]

        async def recovery_heartbeat_recorded() -> bool:
            description = await handle.describe()
            for pending in description.raw_description.pending_activities:
                details = await client.data_converter.decode(
                    pending.heartbeat_details.payloads
                )
                if (
                    details
                    and isinstance(details[0], dict)
                    and details[0].get("resumable") is True
                    and details[0].get("lease") is not None
                ):
                    return True
            return False

        # Prompt persistence can precede the heartbeat that authorizes recovery.
        await kw.until(recovery_heartbeat_recorded, timeout=40)
        first_argv = json.loads(record.read_text())["argv"]
        cli_pid = int(json.loads(record.read_text())["pid"])
        kw.killpg(proc.pid)
        kw.killpg(cli_pid)
        proc.wait(timeout=5)
        await kw.until(lambda: not kw.alive(cli_pid), timeout=20)
        shutil.rmtree(disk_a)
        assert not disk_a.exists()
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ClaudeStructured],
            max_cached_workflows=0,
            plugins=[_plugin(cli, workspace, store, leases)],
        ):
            result = await handle.result()
    finally:
        kw.killpg(proc.pid)
        if cli_pid:
            _reap(cli_pid)
        if proc.poll() is None:
            proc.wait(timeout=5)
        worker_log.close()

    assert result == Forecast(city="Paris", temp_f=72)
    resumed = [
        json.loads(line) for line in log.read_text().splitlines() if line.strip()
    ]
    assert len(resumed) == 1
    schema = Forecast.model_json_schema()
    assert _schema(first_argv) == schema
    assert _schema(resumed[0]["argv"]) == schema
    assert resumed[0]["prompt"].startswith("Continue"), resumed[0]["prompt"]
    assert any(arg.startswith("--resume=") for arg in resumed[0]["argv"])
    assert not kw.alive(cli_pid)
