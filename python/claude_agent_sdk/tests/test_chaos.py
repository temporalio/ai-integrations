"""Random agents, and Worker processes killed at random moments: the promises must hold.

Each run gives a seeded random agent (``tests/chaos/policy.py``) to the real Claude
Code engine, with Edit, Write and Bash as tool steps and a durable tool, and kills the
Worker process while an Activity runs on it (a segment or a tool step, chosen at
random), at a random moment, starting a new Worker each time. Each Worker has its own
Claude Code folders, as Workers on other machines would. Whatever happened, at the
end:

- the Workflow finished, with no failed Workflow task, and no tool step that failed
  for a reason a kill does not explain;
- no command and no edit ran twice (each has a token that shows in the files);
- every call Claude was told had worked really did;
- Claude was told a call may have run at most once per kill;
- Claude Code never checked an edit or a write again ("modified since read",
  anthropics/claude-code#99041), in any request;
- every request the model got was valid (each call with its result).

Each run takes minutes, so the test runs only when ``CHAOS_RUNS`` sets the number of
seeds per conversation mode (``CHAOS_SEED`` is the first one), for example:
``CHAOS_RUNS=50 make test PYTEST_ARGS=tests/test_chaos.py``. With ``CHAOS_WARM=1`` the
Workers keep warm engines, so calls wait in them for their tool steps and kills land
there too. With ``CHAOS_SLOW=1`` the model takes its time after each call in some runs
(0.5 or 1.5 s), so Claude Code pauses at a call before the rest of its message
arrives. With ``CHAOS_WORKSPACE=1`` the Workers have a workspace store: the agent
works in a folder of its own, and each new Worker starts on a machine whose disk does
not have it (the folder is removed before it starts); the promises are then checked
on the agent's last snapshot, restored from the store in a new folder (and the last
Worker's folder, if it ran a step, must be that snapshot). With ``CHAOS_LOG`` set to a
file, each passed run adds a line to it: its seed, mode, kills and outcome.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import json
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.api.enums.v1 import PendingActivityState
from temporalio.claude_agent_sdk import (
    TOOL_CALL_NOT_RUN,
    AgentState,
    FileWorkspaceStore,
    Workspace,
)
from temporalio.claude_agent_sdk._workspace import Workspaces
from temporalio.client import Client, WorkflowHandle
from tests.chaos.policy import STALE, ChaosAgent
from tests.conftest import PLUGIN_ROOT
from tests.engine_tools.workflows import ShellOptions, ShellWorkflow
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env
from tests.test_workspace import describe

RUNS = int(os.environ.get("CHAOS_RUNS", "0"))
WARM = os.environ.get("CHAOS_WARM") == "1"
SLOW = os.environ.get("CHAOS_SLOW") == "1"
WORKSPACE = os.environ.get("CHAOS_WORKSPACE") == "1"
FIRST = int(os.environ.get("CHAOS_SEED", "0"))
MAX_KILLS = 3
KINDS = ["run_claude_segment", "run_claude_tool_step"]
DELAY = {"run_claude_segment": 1.0, "run_claude_tool_step": 0.8}
"""The longest wait before a kill, once an Activity of that kind runs (seconds). The
durable call is too quick to aim at; kills land in it by chance."""
RUN_SECONDS = 360
pytestmark = [
    pytest.mark.timeout(RUN_SECONDS + 60),
    pytest.mark.skipif(not RUNS, reason="set CHAOS_RUNS to the seeds per mode"),
]
OPTIONS = ShellOptions(
    builtin_tools=["Read", "Edit", "Write", "Bash"],
    tool_activities=["Edit", "Write", "Bash"],
    segment_timeout=60,
    tool_timeout=60,
)
STARTED = PendingActivityState.PENDING_ACTIVITY_STATE_STARTED


async def start_worker(
    address: str, queue: str, env: dict[str, str], folder: Path
) -> subprocess.Popen[bytes]:
    """Start ``tests.chaos.worker`` in its own process and wait until it is ready."""
    folder.mkdir()
    (folder / "tmp").mkdir()
    log = folder / "worker.log"
    python_path = os.pathsep.join(
        p for p in (str(PLUGIN_ROOT), os.environ.get("PYTHONPATH")) if p
    )
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.chaos.worker",
                "--address",
                address,
                "--task-queue",
                queue,
            ],
            cwd=PLUGIN_ROOT,
            env={
                **os.environ,
                **env,
                "PYTHONPATH": python_path,
                "TMPDIR": str(folder / "tmp"),  # its hook folders go with the test's
            },
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    deadline = time.monotonic() + 60
    while "worker ready" not in log.read_text(encoding="utf-8", errors="replace"):
        if proc.poll() is not None or time.monotonic() > deadline:
            kill(proc)
            raise RuntimeError(
                f"worker failed to start:\n{log.read_text(errors='replace')}"
            )
        await asyncio.sleep(0.1)
    return proc


def kill(proc: subprocess.Popen[bytes]) -> None:
    """Kill without any cleanup, like an out-of-memory kill."""
    if proc.poll() is None:
        proc.kill()
    proc.wait()


def tokens(path: Path) -> collections.Counter[str]:
    if not path.exists():
        return collections.Counter()
    words = path.read_text(encoding="utf-8").split()
    return collections.Counter(w for w in words if w.startswith("tok"))


async def running_on(
    handle: WorkflowHandle[Any, Any], pid: int
) -> list[tuple[str, str, int]]:
    """The Activities running on the Worker with process id ``pid``: their type, id
    and attempt."""
    description = await handle.describe()
    return [
        (info.activity_type.name, info.activity_id, info.attempt)
        for info in description.raw_description.pending_activities
        if info.state == STARTED and info.last_worker_identity.startswith(f"{pid}@")
    ]


async def failures(handle: WorkflowHandle[Any, Any]) -> tuple[list[str], int]:
    """Failures no kill explains, and the tool steps that did not run their call.

    A kill only times Activities out. A tool step may fail before its call could
    start (it could not read the conversation while no Worker answered, say): the
    Workflow tries it again. Any other failure is not expected.
    """
    names: dict[int, str] = {}
    found: list[str] = []
    not_run = 0
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            scheduled = event.activity_task_scheduled_event_attributes
            names[event.event_id] = scheduled.activity_type.name
        elif event.HasField("activity_task_failed_event_attributes"):
            failed = event.activity_task_failed_event_attributes
            name = names.get(failed.scheduled_event_id, "?")
            info = failed.failure.application_failure_info
            if (
                name.endswith("tool_step")
                and info.type == TOOL_CALL_NOT_RUN
                and not info.non_retryable
            ):
                not_run += 1
            elif name.endswith("tool_step"):
                found.append(f"{name} ({info.type}): {failed.failure.message}")
        elif event.HasField("workflow_task_failed_event_attributes"):
            task = event.workflow_task_failed_event_attributes
            found.append(f"workflow task: {task.failure.message}")
    return found, not_run


@pytest.mark.parametrize("seed", range(FIRST, FIRST + max(RUNS, 1)))
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_chaos(
    client: Client, address: str, tmp_path: Path, mode: str, seed: int
) -> None:
    options = OPTIONS
    store = tmp_path / "workspaces"
    root = tmp_path / "agents"  # the runners' cwd, the same path on every Worker
    session = ""
    if WORKSPACE:
        # The agent's folder starts as a snapshot made here, given in its state.
        session = str(uuid.uuid4())
        first = Workspaces(
            str(tmp_path / "seed"),
            FileWorkspaceStore(store),
            max_bytes=None,
            exclude=(),
        )
        async with first.use(Workspace(session)) as use:
            Path(use.folder, "notes.txt").write_text("start\nEND\n", encoding="utf-8")
            Path(use.folder, "over.txt").write_text("start\n", encoding="utf-8")
            begin = await use.save()
        state = AgentState(session_id=session, workspace=Workspace(session, begin))
        options = dataclasses.replace(OPTIONS, state=state)
        work = root / session
    else:
        work = tmp_path / "work"
        work.mkdir()
        (work / "notes.txt").write_text("start\nEND\n", encoding="utf-8")
        (work / "over.txt").write_text("start\n", encoding="utf-8")
    agent = ChaosAgent(seed, work)
    queue = f"chaos-{mode}-{seed}-{uuid.uuid4().hex[:6]}"
    rng = random.Random(f"{mode}-{seed}")
    workers: list[subprocess.Popen[bytes]] = []
    kills: list[tuple[float, str]] = []
    live: dict[str, Any] = {}  # the Workflow's handle and result, once it started
    api = FakeMessagesAPI(agent.decide).start()
    agent.api = api
    if SLOW:  # its own draw: the kills of a seed stay as they were
        api.pause_after_call = random.Random(f"slow-{mode}-{seed}").choice(
            [0.0, 0.5, 1.5]
        )

    async def new_worker() -> None:
        n = len(workers)
        if WORKSPACE:  # a new machine: the agent's folder is not on its disk
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(exist_ok=True)
        env = {
            "CHAOS_ENGINE_ENV": json.dumps(engine_env(api, str(tmp_path / f"cfg{n}"))),
            "ENGINE_CWD": str(root if WORKSPACE else work),
            "RUNNER_MODE": mode,
            "SESSION_DIR": str(tmp_path / "sessions"),
            "SHOP_DIR": str(tmp_path / "shop"),
            "WARM_ENGINES": "4" if WARM else "0",
            "WORKSPACE_STORE": str(store) if WORKSPACE else "",
        }
        workers.append(await start_worker(address, queue, env, tmp_path / f"w{n}"))

    def report() -> str:
        seen = agent.read(agent.final or (api.requests[-1] if api.requests else {}))
        logs = "\n".join(
            f"--- w{n}: {(tmp_path / f'w{n}' / 'worker.log').read_text(errors='replace')[-1500:]}"
            for n in range(len(workers))
        )
        return (
            f"seed={seed} mode={mode} kills {kills}, plan "
            f"{[[(i.tool, i.token) for i in m] for m in agent.plan]}, seen {seen}\n"
            f"{logs}"
        )

    async def run() -> Any:
        await new_worker()
        started = time.monotonic()
        handle = await client.start_workflow(
            ShellWorkflow.run, args=["chaos", options], id=queue, task_queue=queue
        )
        live["handle"] = handle
        result_task = asyncio.ensure_future(handle.result())
        live["result"] = result_task
        # Each kill waits for an Activity of a random kind (a segment or a tool step)
        # to run on the current Worker, lets a random number of them go by, then
        # kills the Worker at a random moment of the next one.
        kinds = [rng.choice(KINDS) for _ in range(MAX_KILLS)]
        skips = [rng.randint(0, 2) for _ in range(MAX_KILLS)]
        passed: set[tuple[str, int]] = set()
        while not result_task.done():
            if workers[-1].poll() is not None:
                raise AssertionError(f"a Worker exited by itself\n{report()}")
            k = len(kills)
            running = (
                [] if k >= MAX_KILLS else await running_on(handle, workers[-1].pid)
            )
            target = (
                [(i, a) for kind, i, a in running if kind == kinds[k]]
                if running
                else []
            )
            new = [t for t in target if t not in passed]
            if not new:
                await asyncio.wait([result_task], timeout=0.1 if k < MAX_KILLS else 1.0)
                continue
            if skips[k] > 0:
                skips[k] -= 1
                passed.update(new)
                continue
            await asyncio.wait([result_task], timeout=rng.uniform(0.0, DELAY[kinds[k]]))
            if result_task.done():
                break
            at = [kind for kind, _, _ in await running_on(handle, workers[-1].pid)]
            kill(workers[-1])
            kills.append((round(time.monotonic() - started, 1), ",".join(at) or "idle"))
            await new_worker()
        return result_task.result()

    try:
        try:
            result = await asyncio.wait_for(run(), RUN_SECONDS)
        except asyncio.TimeoutError:
            raise AssertionError(f"no result in {RUN_SECONDS} s\n{report()}") from None
        failed, not_run = await failures(live["handle"])
        if WORKSPACE:
            # The promises hold on the agent's last snapshot, read back from the store
            # into a new folder. The last Worker's folder is that snapshot, if it ran
            # a step: a kill after the last step leaves a Worker that ran none, and a
            # disk without the folder.
            final = await live["handle"].query(ShellWorkflow.agent_state)
            assert final.workspace is not None and final.workspace.id == session
            check = Workspaces(
                str(tmp_path / "final"),
                FileWorkspaceStore(store),
                max_bytes=None,
                exclude=(),
            )
            async with check.use(final.workspace):
                pass
            if work.exists():
                assert describe(tmp_path / "final" / session) == describe(work), (
                    report()
                )
            work = tmp_path / "final" / session
    finally:
        task = live.get("result")
        if task is not None and not task.done():
            task.cancel()
        if "handle" in live:
            try:
                await live["handle"].terminate("the chaos test ended")
            except Exception:  # it ended already
                pass
        for proc in workers:
            kill(proc)
        api.stop()
    final = agent.final
    assert final is not None, report()
    seen = agent.read(final)
    assert result == "FINAL", report()
    assert failed == [], (failed, report())
    assert api.errors == [], report()
    # Claude Code never checked an edit or a write again, in any request.
    assert STALE not in json.dumps(api.requests), report()
    assert seen.errors == [], report()
    assert seen.interrupted <= len(kills), report()  # one per kill at most
    effects = tokens(work / "effects.log")
    notes = tokens(work / "notes.txt")
    assert all(n == 1 for n in effects.values()), report()  # no command ran twice
    assert all(n == 1 for n in notes.values()), report()  # no edit ran twice
    counted = (
        [
            json.loads(line)["detail"]
            for line in (tmp_path / "shop" / "executions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if json.loads(line)["tool"] == "count"
        ]
        if (tmp_path / "shop" / "executions.jsonl").exists()
        else []
    )
    over = (work / "over.txt").read_text(encoding="utf-8").strip()
    for token, tool in seen.ok.items():  # what Claude was told worked, did
        if tool == "Bash":
            assert effects[token] == 1, (token, report())
        elif tool == "Edit":
            assert notes[token] == 1, (token, report())
        elif tool == "Write":
            written = (work / f"w_{token}.txt").read_text(encoding="utf-8")
            assert written == token, (token, report())
        elif tool == "count":
            assert token.removeprefix("tok") in counted, (token, report())
    for path in work.glob("w_tok*.txt"):  # a write that may have run wrote its own
        assert path.read_text(encoding="utf-8") == path.stem[2:], (path, report())
    overwrites = [i.token for m in agent.plan for i in m if i.tool == "Overwrite"]
    assert over in ["start", *overwrites], report()
    if overwrites and seen.ok.get(overwrites[-1]) == "Overwrite":
        assert over == overwrites[-1], report()  # the last one Claude was told worked
    log = os.environ.get("CHAOS_LOG")
    if log:
        line = {
            "seed": seed,
            "mode": mode,
            "warm": WARM,
            "workspace": WORKSPACE,
            "pause": api.pause_after_call,
            "kills": kills,
            "calls": sum(len(m) for m in agent.plan),
            "interrupted": seen.interrupted,
            "not_run_steps": not_run,
            "requests": len(api.requests),
        }
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
