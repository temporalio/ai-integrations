"""A workflow that hosts Claude turns itself: sessions, tools, and history.

The worker runs the offline fake CLI, so no credentials are needed.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk import InMemorySessionStore

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeSessionState,
    FileLeaseStore,
    FileSessionStore,
    SharedDirWorkspaceStore,
    _native,
)
from temporalio.claude_agent_sdk._fake_cli import _flag
from temporalio.claude_agent_sdk._models import (
    CLAUDE_NATIVE_TOOL_UNSAFE_ERROR_TYPE,
    CLAUDE_TURN_BUDGET_ERROR_TYPE,
    CLAUDE_TURN_REJECTED_ERROR_TYPE,
    CLAUDE_TURN_TIMEOUT_ERROR_TYPE,
    CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE,
    CONTINUE_PROMPT,
)
from temporalio.claude_agent_sdk._session import NATIVE_ABANDONED_REASON
from temporalio.claude_agent_sdk.testing import SESSION, START, answer, text_turn
from temporalio.client import Client, WorkflowFailureError, WorkflowHandle
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.runtime import MetricBuffer, Runtime, TelemetryConfig
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from .conftest import scripted_cli as _cli
from .helpers import fixtures as fx
from .helpers import histories
from .helpers import kill_worker as kw
from .helpers.workflows import (
    ClaudeBounded,
    ClaudeCapped,
    ClaudeChat,
    ClaudeContinue,
    ClaudeFailover,
    ClaudeNative,
    ClaudeNativeDenied,
    ClaudeNativeSafety,
    ClaudeNativeTool,
    ClaudeNativeWithTool,
    ClaudeResume,
    ClaudeSpend,
    ClaudeStructuredContinue,
    Forecast,
)
from .test_claude_workspace import _FailingStore

pytestmark = fx.POSIX_ONLY


@pytest.fixture(autouse=True)
def _offline_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))


@asynccontextmanager
async def _chat(
    env: WorkflowEnvironment, tmp_path: Path, **scenario: Any
) -> AsyncIterator[WorkflowHandle[ClaudeChat, None]]:
    """A running ``ClaudeChat`` on a worker whose Claude CLI is the fake."""
    client = kw.client(env)
    cli, _ = _cli(tmp_path / "cli", **scenario)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeChat],
        plugins=[
            ClaudeAgentPlugin(cli_path=cli, workspace_root=tmp_path / "workspaces")
        ],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            ClaudeChat.run, id=task_queue, task_queue=task_queue
        )
        yield handle
        await handle.terminate()


def _effects(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [json.loads(line)["id"] for line in path.read_text().splitlines()]


@asynccontextmanager
async def _service(
    env: WorkflowEnvironment,
    tmp_path: Path,
    workflows: Sequence[type],
    scenario: dict[str, Any],
    **plugin: Any,
) -> AsyncIterator[tuple[Client, str, str]]:
    """A worker and ``(client, task queue, cli path)`` for one scripted CLI."""
    client = kw.client(env)
    cli, _ = _cli(tmp_path / "cli", **scenario)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=list(workflows),
        plugins=[
            ClaudeAgentPlugin(
                cli_path=cli,
                workspace_root=tmp_path / "workspaces",
                **plugin,
            )
        ],
    ):
        yield client, task_queue, cli


async def test_an_approved_native_call_runs_once(
    env: WorkflowEnvironment, tmp_path: Path, verified: None
) -> None:
    assert verified is None
    effects = tmp_path / "effects.jsonl"
    turns = [
        [
            "run",
            [
                fx.SYSTEM_INIT,
                *fx.native_round(
                    "msg_n",
                    "toolu_x",
                    "Bash",
                    {"command": "echo hi"},
                    writes={"hello.txt": "hello"},
                ),
            ],
        ],
        [CONTINUE_PROMPT, fx.answer("Done.")],
        ["Continue.", fx.answer("Done.")],
    ]
    async with _service(
        env,
        tmp_path,
        [ClaudeNative],
        {
            "turns": turns,
            "state": str(tmp_path / "state.json"),
            "effects": str(effects),
            "log": str(tmp_path / "log.jsonl"),
            "trigger_settle": 0,
        },
        enable_native_tools=True,
    ) as (client, task_queue, _cli_path):
        handle = await client.start_workflow(
            ClaudeNative.run, True, id=task_queue, task_queue=task_queue
        )
        assert await handle.result() == "Done."
        await histories.record("native_approve", handle)

    assert _effects(effects) == ["toolu_x"]


async def test_a_tool_call_repeated_after_an_approval_runs_again(
    env: WorkflowEnvironment, tmp_path: Path, verified: None
) -> None:
    """The approved resume is another run of the same message; its identical call is a new
    call, not the earlier run's call re-issued."""
    assert verified is None
    weather = ("get_weather", {"city": "Paris"})
    turns = [
        [
            "weather twice",
            [
                fx.SYSTEM_INIT,
                *fx.tool_round("msg_t1", [("toolu_w1", *weather)]),
                *fx.native_round("msg_n", "toolu_x", "Bash", {"command": "echo hi"}),
            ],
        ],
        [
            "Continue.",
            [*fx.tool_round("msg_t2", [("toolu_w2", *weather)]), *fx.answer("Done.")],
        ],
    ]
    async with _service(
        env,
        tmp_path,
        [ClaudeNativeWithTool],
        {
            "turns": turns,
            "state": str(tmp_path / "state.json"),
            "effects": str(tmp_path / "effects.jsonl"),
        },
        enable_native_tools=True,
    ) as (client, task_queue, _cli_path):
        calls = await client.execute_workflow(
            ClaudeNativeWithTool.run, id=task_queue, task_queue=task_queue
        )

    assert calls == 2
    assert _effects(tmp_path / "effects.jsonl") == ["toolu_x"]


async def test_a_denied_native_call_does_not_run(
    env: WorkflowEnvironment, tmp_path: Path, verified: None
) -> None:
    assert verified is None
    effects = tmp_path / "effects.jsonl"
    turns = [
        [
            "run",
            [
                fx.SYSTEM_INIT,
                *fx.native_round("msg_n", "toolu_x", "Bash", {"command": "echo hi"}),
            ],
        ],
        [CONTINUE_PROMPT, fx.answer("Asked.")],
        ["Continue.", fx.answer("Asked.")],
    ]
    async with _service(
        env,
        tmp_path,
        [ClaudeNative],
        {
            "turns": turns,
            "state": str(tmp_path / "state.json"),
            "effects": str(effects),
            "log": str(tmp_path / "log.jsonl"),
        },
        enable_native_tools=True,
    ) as (client, task_queue, _cli_path):
        reply = await client.execute_workflow(
            ClaudeNative.run, False, id=task_queue, task_queue=task_queue
        )

    assert reply == "Asked."
    assert _effects(effects) == []


async def test_a_denial_reason_reaches_claude(
    env: WorkflowEnvironment, tmp_path: Path, verified: None
) -> None:
    assert verified is None
    log = tmp_path / "log.jsonl"
    turns = [
        [
            "run",
            [
                fx.SYSTEM_INIT,
                *fx.native_round("msg_n", "toolu_x", "Bash", {"command": "echo hi"}),
            ],
        ],
        ["Continue.", fx.answer("Asked.")],
    ]
    async with _service(
        env,
        tmp_path,
        [ClaudeNativeDenied],
        {"turns": turns, "state": str(tmp_path / "state.json"), "log": str(log)},
        enable_native_tools=True,
    ) as (client, task_queue, _cli_path):
        await client.execute_workflow(
            ClaudeNativeDenied.run,
            "Use the staging host.",
            id=task_queue,
            task_queue=task_queue,
        )

    hooks = [
        hook
        for line in log.read_text().splitlines()
        for hook in json.loads(line)["hooks"]
    ]
    assert [(h["decisions"], h["reasons"]) for h in hooks if h["id"] == "toolu_x"][
        -1
    ] == (["deny"], ["Use the staging host."])


@pytest.mark.parametrize(
    "tool",
    ["Agent", "Task", "Skill", "Workflow", "Monitor", "EnterWorktree", "CronCreate"],
)
async def test_an_orchestration_tool_fails_the_workflow(
    env: WorkflowEnvironment, tmp_path: Path, verified: None, tool: str
) -> None:
    """Same shape as an unsupported OpenAI tool: the workflow fails instead of hanging on a call
    this integration cannot record."""
    assert verified is None
    async with _service(
        env,
        tmp_path,
        [ClaudeNativeTool],
        {"turn": fx.COMPLETED_ONLY},
        enable_native_tools=True,
    ) as (client, task_queue, _cli_path):
        with pytest.raises(WorkflowFailureError) as info:
            await client.execute_workflow(
                ClaudeNativeTool.run,
                tool,
                id=task_queue,
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=60),
            )

    cause = info.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == CLAUDE_TURN_REJECTED_ERROR_TYPE and cause.non_retryable
    assert tool in str(cause) and "cannot be made durable" in str(cause)


async def test_a_killed_approved_bash_call_is_not_run_again(
    env: WorkflowEnvironment, tmp_path: Path, verified: None
) -> None:
    """The CLI dies after the approved effect. Shared lease and session store would let a retry
    land on another worker; the effect must still have run once, and the message must fail."""
    assert verified is None
    effects = tmp_path / "effects.jsonl"
    turns = [
        [
            "run",
            [
                fx.SYSTEM_INIT,
                *fx.native_round(
                    "msg_n",
                    "toolu_x",
                    "Bash",
                    {"command": "echo hi"},
                    writes={"hello.txt": "hello"},
                ),
            ],
        ],
        [CONTINUE_PROMPT, fx.answer("Done.")],
        ["Continue.", fx.answer("Done.")],
    ]
    async with _service(
        env,
        tmp_path,
        [ClaudeNative],
        {
            "turns": turns,
            "state": str(tmp_path / "state.json"),
            "effects": str(effects),
            "log": str(tmp_path / "log.jsonl"),
            "crash_at": "after_tool_run",
            "crash_mode": "die",
            "crash_prompt": "Continue",
            "trigger_settle": 0,
            "trigger_dir": str(tmp_path / "triggers"),
        },
        enable_native_tools=True,
        session_store=_SessionStore(),
        lease_store=FileLeaseStore(tmp_path / "leases"),
        disconnect_timeout=timedelta(seconds=2),
    ) as (client, task_queue, _cli_path):
        with pytest.raises(WorkflowFailureError) as info:
            await client.execute_workflow(
                ClaudeNative.run,
                True,
                id=task_queue,
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=90),
            )

    cause = info.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.non_retryable
    assert _effects(effects) == ["toolu_x"]


async def test_a_turn_timeout_fails_the_workflow(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    async with _service(
        env,
        tmp_path,
        [ClaudeBounded],
        {"turn": fx.STREAMING_UNTIL_INTERRUPT, "hang_after_prompt": True},
        disconnect_timeout=timedelta(seconds=1),
    ) as (client, task_queue, _cli_path):
        with pytest.raises(WorkflowFailureError) as info:
            await client.execute_workflow(
                ClaudeBounded.run, id=task_queue, task_queue=task_queue
            )

    cause = info.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == CLAUDE_TURN_TIMEOUT_ERROR_TYPE


class _SessionStore:
    """Enough of a SessionStore for an API-error resume to be allowed."""

    def __init__(self) -> None:
        self.rows: dict[Any, list[Any]] = {}

    async def append(self, key: Any, entries: list[Any]) -> None:
        self.rows.setdefault(str(key), []).extend(entries)

    async def load(self, key: Any) -> list[Any] | None:
        return self.rows.get(str(key))


async def test_an_api_error_is_resumed(
    tmp_path: Path,
) -> None:
    async with await WorkflowEnvironment.start_time_skipping() as skipping:
        async with _service(
            skipping,
            tmp_path,
            [ClaudeResume],
            {
                "plans": [
                    ["What's the weather?", [{"text": "Sunny."}]],
                ],
                "crash_at": "before_prompt_persisted",
                "crash_mode": "api_error",
                "crash_prompt": "What's the weather?",
                "trigger_settle": 0,
                "trigger_dir": str(tmp_path / "triggers"),
            },
            session_store=_SessionStore(),
        ) as (client, task_queue, _cli_path):
            reply = await client.execute_workflow(
                ClaudeResume.run, id=task_queue, task_queue=task_queue
            )

    assert reply == "Sunny."


async def test_a_crashed_cli_resumes_without_a_shared_lease_store(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """A CLI that exited on its own took its lease with it, so the worker's default host-local
    lease store is enough to resume the turn from the store."""
    async with _service(
        env,
        tmp_path,
        [ClaudeResume],
        {
            "plans": [["What's the weather?", [{"text": "Sunny."}]]],
            "crash_at": "before_prompt_persisted",
            "crash_mode": "die",
            "crash_prompt": "What's the weather?",
            "trigger_settle": 0,
            "trigger_dir": str(tmp_path / "triggers"),
        },
        session_store=_SessionStore(),
    ) as (client, task_queue, _cli_path):
        reply = await client.execute_workflow(
            ClaudeResume.run, id=task_queue, task_queue=task_queue
        )

    assert reply == "Sunny."
    assert (tmp_path / "triggers" / "before_prompt_persisted").exists()


async def test_a_run_whose_workspace_was_not_persisted_is_still_charged(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    async with _service(
        env,
        tmp_path,
        [ClaudeSpend],
        {"turn": text_turn("Hi.")},
        workspace_store=_FailingStore(tmp_path / "store"),
    ) as (client, task_queue, _cli_path):
        failure, cost = await client.execute_workflow(
            ClaudeSpend.run, id=task_queue, task_queue=task_queue
        )

    assert (failure, cost) == (CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE, 0.0123)


async def test_a_result_that_lands_with_the_timeout_is_still_charged(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """The run's result and the expired timer reach the workflow in one activation: the turn
    times out, and the session still pays for the answer its transcript now holds."""
    client = kw.client(env)
    cli, _ = _cli(
        tmp_path / "cli",
        turn=[*START, {"fake": "sleep", "seconds": 6}, *answer("Late.")],
    )
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"

    def workflow_worker() -> Worker:
        return Worker(
            client,
            task_queue=task_queue,
            workflows=[ClaudeSpend],
            plugins=[ClaudeAgentPlugin(register_activities=False)],
        )

    async with Worker(
        client,
        task_queue=task_queue,
        plugins=[
            ClaudeAgentPlugin(cli_path=cli, workspace_root=tmp_path / "workspaces")
        ],
    ):

        async def happened(event: str) -> bool:
            async for e in handle.fetch_history_events():
                if e.WhichOneof("attributes") == event:
                    return True
            return False

        async with workflow_worker():
            handle = await client.start_workflow(
                ClaudeSpend.run, 3.0, id=task_queue, task_queue=task_queue
            )
            while not await happened("activity_task_scheduled_event_attributes"):
                await asyncio.sleep(0.05)
        # No workflow task runs until both the timer and the Activity have completed.
        while not await happened("activity_task_completed_event_attributes"):
            await asyncio.sleep(0.1)
        assert await happened("timer_fired_event_attributes")
        async with workflow_worker():
            failure, cost = await handle.result()

    assert (failure, cost) == (CLAUDE_TURN_TIMEOUT_ERROR_TYPE, 0.0123)


async def test_a_settled_turn_is_counted(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    buffer = MetricBuffer(1000)
    client = await Client.connect(
        env.client.service_client.config.target_host,
        namespace=env.client.namespace,
        data_converter=pydantic_data_converter,
        runtime=Runtime(telemetry=TelemetryConfig(metrics=buffer)),
    )
    cli, _ = _cli(tmp_path / "cli", turn=text_turn("Sunny."))
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeResume],
        plugins=[
            ClaudeAgentPlugin(cli_path=cli, workspace_root=tmp_path / "workspaces")
        ],
    ):
        await client.execute_workflow(
            ClaudeResume.run, id=task_queue, task_queue=task_queue
        )

    keys = ("task_queue", "workflow_type", "outcome", "model")
    counted = [
        (u.value, {k: u.attributes.get(k) for k in keys})
        for u in buffer.retrieve_updates()
        if u.metric.name == "claude_turns"
    ]
    assert counted == [
        (
            1,
            {
                "task_queue": task_queue,
                "workflow_type": "ClaudeResume",
                "outcome": "success",
                "model": "default",
            },
        )
    ]


async def test_sandbox_violations_reach_the_result_and_the_meter(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    buffer = MetricBuffer(1000)
    client = await Client.connect(
        env.client.service_client.config.target_host,
        namespace=env.client.namespace,
        data_converter=pydantic_data_converter,
        runtime=Runtime(telemetry=TelemetryConfig(metrics=buffer)),
    )
    bash_result = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "tool_use_id": "toolu_bash",
                    "type": "tool_result",
                    "content": "curl: (56) CONNECT tunnel failed, response 403\n"
                    "<sandbox_violations>\nnetwork example.com:443\n"
                    "network example.com:443\n</sandbox_violations>",
                }
            ],
        },
        "parent_tool_use_id": None,
        "session_id": SESSION,
        "uuid": "u-bash",
    }
    cli, _ = _cli(tmp_path / "cli", turn=[*START, bash_result, *answer("Blocked.")])
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeResume],
        plugins=[
            ClaudeAgentPlugin(cli_path=cli, workspace_root=tmp_path / "workspaces")
        ],
    ):
        handle = await client.start_workflow(
            ClaudeResume.run, id=task_queue, task_queue=task_queue
        )
        await handle.result()

    results = [
        json.loads(e.activity_task_completed_event_attributes.result.payloads[0].data)
        async for e in handle.fetch_history_events()
        if e.HasField("activity_task_completed_event_attributes")
    ]
    assert [r.get("sandbox_violations") for r in results] == [2]
    counted = [
        (u.value, u.attributes.get("model"))
        for u in buffer.retrieve_updates()
        if u.metric.name == "claude_sandbox_violations"
    ]
    assert counted == [(2, "default")]


async def test_a_killed_write_is_restored_from_the_workspace_snapshot(
    env: WorkflowEnvironment, tmp_path: Path, verified: None
) -> None:
    """A second worker, with the first disk gone, restores the killed write from the snapshot.

    The follow-up does not ``--resume``: there is no shared SessionStore, and the scripted CLI
    would accept ``--resume`` with no file, which the real CLI rejects.
    """
    assert verified is None
    effects = tmp_path / "effects.jsonl"
    store = SharedDirWorkspaceStore(tmp_path / "store")
    leases = FileLeaseStore(tmp_path / "leases")
    root_a, root_b = tmp_path / "disk-a", tmp_path / "disk-b"
    turns = [
        [
            "write",
            [
                fx.SYSTEM_INIT,
                *fx.native_round(
                    "msg_n",
                    "toolu_x",
                    "Bash",
                    {"command": "printf hello > hello.txt"},
                    writes={"hello.txt": "hello"},
                ),
            ],
        ],
        [CONTINUE_PROMPT, fx.answer("wrote")],
        ["Continue.", fx.answer("wrote")],
        ["next", text_turn("restored")],
    ]
    scenario = {
        "turns": turns,
        "state": str(tmp_path / "state.json"),
        "effects": str(effects),
        "log": str(tmp_path / "log.jsonl"),
        "crash_at": "after_tool_run",
        "crash_mode": "die",
        "crash_prompt": "Continue",
        "trigger_settle": 0,
        "trigger_dir": str(tmp_path / "triggers"),
    }
    client = kw.client(env)
    cli, _ = _cli(tmp_path / "cli", **scenario)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"

    def plugin(root: Path) -> ClaudeAgentPlugin:
        return ClaudeAgentPlugin(
            cli_path=cli,
            workspace_root=root,
            enable_native_tools=True,
            workspace_store=store,
            lease_store=leases,
            disconnect_timeout=timedelta(seconds=2),
        )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeFailover],
        plugins=[plugin(root_a)],
    ):
        handle = await client.start_workflow(
            ClaudeFailover.run, id=task_queue, task_queue=task_queue
        )
        for _ in range(100):
            if await handle.query(ClaudeFailover.held):
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("the workflow did not hold after the killed write")

    shutil.rmtree(root_a)
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeFailover],
        plugins=[plugin(root_b)],
    ):
        await handle.signal(ClaudeFailover.proceed)
        assert await handle.result() == "restored"

    files = list(root_b.rglob("hello.txt"))
    assert [p.read_text() for p in files] == ["hello"]
    assert _effects(effects) == ["toolu_x"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    follow = next(row for row in rows if row["prompt"] == "next")
    assert not any(arg.startswith("--resume=") for arg in follow["argv"])


async def test_every_message_continues_the_same_claude_session(
    env: WorkflowEnvironment, tmp_path: Path
):
    async with _chat(env, tmp_path, session_mode=True) as chat:
        first = await chat.execute_update(ClaudeChat.chat, "hello")
        second = await chat.execute_update(ClaudeChat.chat, "again")
        await histories.record("session_chat", chat)

    assert (first, second) == ("turn 1: hello", "turn 2: again")


async def test_claude_calls_a_tool_the_workflow_runs(
    env: WorkflowEnvironment, tmp_path: Path
):
    turn = fx.tool_turn([("toolu_1", "get_weather", {"city": "Paris"})])
    async with _chat(env, tmp_path, turn=turn) as chat:
        reply = await chat.execute_update(ClaudeChat.chat_with_tool, "weather?")

        assert reply == "All done."
        assert await chat.query(ClaudeChat.weather_calls) == 1
        await histories.record("tool_turn", chat)


async def test_continue_as_new_resumes_the_same_session_transcript(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """Turn 1 completes, continue-as-new carries only the session snapshot, and turn 2 resumes
    that session after the worker's transcript file is gone. ``turn 2`` is the transcript's user
    count, so the result exists only when turn 1's transcript was restored from the store."""
    log = tmp_path / "cli.jsonl"
    async with _service(
        env,
        tmp_path,
        [ClaudeContinue],
        {"session_mode": True, "log": str(log)},
        session_store=InMemorySessionStore(),
    ) as (client, task_queue, _cli_path):
        handle = await client.start_workflow(
            ClaudeContinue.run, id=task_queue, task_queue=task_queue
        )
        for _ in range(100):
            if await handle.query(ClaudeContinue.first):
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("turn 1 did not finish")
        assert await handle.query(ClaudeContinue.first) == "turn 1: hello"

        transcripts = list((tmp_path / "claude-config").rglob("*.jsonl"))
        assert transcripts
        for path in transcripts:
            path.unlink()

        await handle.signal(ClaudeContinue.proceed)
        assert await handle.result() == "turn 2: again"

        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert [row["prompt"] for row in rows] == ["hello", "again"]
        assert (session_id := _flag(rows[0]["argv"], "--session-id"))
        assert _flag(rows[1]["argv"], "--resume") == session_id
        assert rows[0]["config_dir"] != rows[1]["config_dir"]

        first_run = handle.first_execution_run_id
        assert first_run is not None
        first = await client.get_workflow_handle(
            handle.id, run_id=first_run
        ).fetch_history()
        continued = next(
            event.workflow_execution_continued_as_new_event_attributes
            for event in first.events
            if event.HasField("workflow_execution_continued_as_new_event_attributes")
        )
        (carried,) = await client.data_converter.decode(
            list(continued.input.payloads), [ClaudeSessionState]
        )
        assert carried == ClaudeSessionState(session_id=session_id, started=True)
        second = await client.get_workflow_handle(
            handle.id, run_id=continued.new_execution_run_id
        ).fetch_history()
        replayer = Replayer(
            workflows=[ClaudeContinue],
            plugins=[ClaudeAgentPlugin(register_activities=False)],
        )
        await replayer.replay_workflow(first)
        await replayer.replay_workflow(second)


@pytest.mark.parametrize("cap", [1.0, None])
async def test_a_session_spend_cap_holds_across_continue_as_new(
    env: WorkflowEnvironment, tmp_path: Path, cap: float | None
) -> None:
    """Each run is one message. The cap lowers each message's budget to what the session has
    left, and once the session's reported spend reaches it the next message fails without an
    Activity. With no cap every message runs on its own budget."""
    log = tmp_path / "cli.jsonl"

    def costing(total: float) -> list[Any]:
        return [*fx.START, fx.result(text="ok", total_cost_usd=total)]

    turns = [["one", costing(0.6)], ["two", costing(1.1)], ["three", costing(1.2)]]
    async with _service(
        env, tmp_path, [ClaudeCapped], {"turns": turns, "log": str(log)}
    ) as (client, task_queue, _cli_path):
        handle = await client.start_workflow(
            ClaudeCapped.run,
            args=[["one", "two", "three"], cap],
            id=task_queue,
            task_queue=task_queue,
        )
        result = await handle.result()
        last = await client.get_workflow_handle(handle.id).fetch_history()

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    budgets = [
        float(row["argv"][row["argv"].index("--max-budget-usd") + 1]) for row in rows
    ]
    scheduled = [
        e for e in last.events if e.HasField("activity_task_scheduled_event_attributes")
    ]
    if cap is None:
        assert result == "ok"
        assert [row["prompt"] for row in rows] == ["one", "two", "three"]
        assert budgets == [5.0, 5.0, 5.0]
        return
    assert result == CLAUDE_TURN_BUDGET_ERROR_TYPE
    assert [row["prompt"] for row in rows] == ["one", "two"]
    assert budgets == pytest.approx([1.0, 0.4])
    assert scheduled == []


def _schema(argv: list[str]) -> Any:
    return json.loads(argv[argv.index("--json-schema") + 1])


def _stored_prompt(store: Path, prompt: str) -> bool:
    for path in store.rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if (entry.get("message") or {}).get("content") == prompt:
                return True
    return False


async def test_continue_as_new_resumes_a_typed_result_from_the_stored_transcript(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """Turn 2 resumes the stored transcript after continue-as-new and returns a second Forecast.

    That Forecast is indexed by the transcript's user count, so a fresh file cannot produce it.
    """
    log = tmp_path / "cli.jsonl"
    store = tmp_path / "store"
    async with _service(
        env,
        tmp_path,
        [ClaudeStructuredContinue],
        {
            "session_mode": True,
            # Indexed by the transcript's user count, same proof as ``turn 2: again``.
            "structured_outputs": [
                {"city": "Paris", "temp_f": 72},
                {"city": "Lyon", "temp_f": 61},
            ],
            "log": str(log),
        },
        session_store=cast(Any, FileSessionStore(store)),
    ) as (client, task_queue, _cli_path):
        handle = await client.start_workflow(
            ClaudeStructuredContinue.run, id=task_queue, task_queue=task_queue
        )
        for _ in range(100):
            if await handle.query(ClaudeStructuredContinue.first):
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("turn 1 did not finish")
        assert await handle.query(ClaudeStructuredContinue.first) == Forecast(
            city="Paris", temp_f=72
        )
        assert _stored_prompt(store, "forecast")

        transcripts = list((tmp_path / "claude-config").rglob("*.jsonl"))
        assert transcripts
        for path in transcripts:
            path.unlink()

        await handle.signal(ClaudeStructuredContinue.proceed)
        assert await handle.result() == Forecast(city="Lyon", temp_f=61)

        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert [row["prompt"] for row in rows] == ["forecast", "again"]
        assert (session_id := _flag(rows[0]["argv"], "--session-id"))
        assert _flag(rows[1]["argv"], "--resume") == session_id
        schema = Forecast.model_json_schema()
        assert [_schema(row["argv"]) for row in rows] == [schema, schema]
        assert rows[0]["config_dir"] != rows[1]["config_dir"]

        first_run = handle.first_execution_run_id
        assert first_run is not None
        first = await client.get_workflow_handle(
            handle.id, run_id=first_run
        ).fetch_history()
        continued = next(
            event.workflow_execution_continued_as_new_event_attributes
            for event in first.events
            if event.HasField("workflow_execution_continued_as_new_event_attributes")
        )
        assert len(continued.input.payloads) == 1
        body = json.loads(continued.input.payloads[0].data)
        assert (
            body == ClaudeSessionState(session_id=session_id, started=True).model_dump()
        )
        (carried,) = await client.data_converter.decode(
            list(continued.input.payloads), [ClaudeSessionState]
        )
        assert carried == ClaudeSessionState(session_id=session_id, started=True)
        second = await client.get_workflow_handle(
            handle.id, run_id=continued.new_execution_run_id
        ).fetch_history()
        replayer = Replayer(
            workflows=[ClaudeStructuredContinue],
            plugins=[ClaudeAgentPlugin(register_activities=False)],
        )
        await replayer.replay_workflow(first)
        await replayer.replay_workflow(second)


COMMAND = {"command": "echo hi"}
CALL = {"id": "toolu_x", "name": "Bash", "input": COMMAND}


def _prompts(path: Path) -> list[str]:
    return [
        json.loads(line)["prompt"]
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _state(tmp_path: Path) -> Path:
    dirs = [
        path
        for path in (tmp_path / "workspaces" / ".native-tool-state").iterdir()
        if path.is_dir()
    ]
    assert len(dirs) == 1
    return dirs[0]


def _budget_scenario(tmp_path: Path) -> dict[str, Any]:
    """One plans deferral that reports a full turn and $0.50, then a settle prompt."""
    return {
        "plans": [
            [
                "run",
                [
                    {
                        "native": "Bash",
                        "input": COMMAND,
                        "writes": {"ran.txt": "ran"},
                        "deferred_result": {"num_turns": 1, "total_cost_usd": 0.5},
                    },
                    {"text": "ran"},
                ],
            ],
            ["settle", [{"text": "settled"}]],
        ],
        "state": str(tmp_path / "state.json"),
        "effects": str(tmp_path / "effects.jsonl"),
        "log": str(tmp_path / "log.jsonl"),
    }


@pytest.mark.parametrize("mode", ["max_turns", "max_budget_usd"])
async def test_a_deferral_that_uses_up_the_budget_is_not_resumed(
    env: WorkflowEnvironment, tmp_path: Path, verified: None, mode: str
) -> None:
    """The deferral's own result spends the budget, so the call is denied as abandoned, not run."""
    assert verified is None
    async with _service(
        env,
        tmp_path,
        [ClaudeNativeSafety],
        _budget_scenario(tmp_path),
        enable_native_tools=True,
    ) as (client, task_queue, _cli_path):
        reply = await client.execute_workflow(
            ClaudeNativeSafety.run, mode, id=task_queue, task_queue=task_queue
        )

    assert reply == f"{CLAUDE_TURN_BUDGET_ERROR_TYPE}|non-retryable|settled"
    assert _prompts(tmp_path / "log.jsonl") == ["run", "settle"]
    assert _effects(tmp_path / "effects.jsonl") == []
    assert list((tmp_path / "workspaces").rglob("ran.txt")) == []
    state = _state(tmp_path)
    assert _native.native_calls(state) == []
    # Settled in the transcript before the CLI started: the hook never saw it again.
    assert _native.ledger(state) == []
    [transcript] = (tmp_path / "claude-config").rglob("*.jsonl")
    answers = [
        block
        for line in transcript.read_text().splitlines()
        for block in _native.content_blocks(json.loads(line))
        if block.get("type") == "tool_result"
    ]
    assert [(a["content"][0]["text"], a.get("is_error")) for a in answers] == [
        (NATIVE_ABANDONED_REASON, True)
    ]


def _redefer_turn() -> list[Any]:
    """The same ``deferred_tool_use`` result a plans defer emits, with nothing run."""
    return [
        *fx.START,
        fx.result(
            text="",
            stop_reason="tool_deferred",
            terminal_reason="tool_deferred",
            num_turns=1,
            total_cost_usd=0.01,
            deferred_tool_use=CALL,
        ),
    ]


async def test_a_resume_that_defers_the_same_call_again_does_not_run_it(
    env: WorkflowEnvironment, tmp_path: Path, verified: None
) -> None:
    """The approved resume's result defers ``toolu_x`` again instead of applying the decision."""
    assert verified is None
    effects = tmp_path / "effects.jsonl"
    async with _service(
        env,
        tmp_path,
        [ClaudeNativeSafety],
        {
            "turns": [["run", _redefer_turn()], ["Continue.", _redefer_turn()]],
            "effects": str(effects),
            "log": str(tmp_path / "log.jsonl"),
        },
        enable_native_tools=True,
    ) as (client, task_queue, _cli_path):
        with pytest.raises(WorkflowFailureError) as info:
            await client.execute_workflow(
                ClaudeNativeSafety.run,
                "redefer",
                id=task_queue,
                task_queue=task_queue,
            )

    cause = info.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == CLAUDE_NATIVE_TOOL_UNSAFE_ERROR_TYPE and cause.non_retryable
    assert _prompts(tmp_path / "log.jsonl") == ["run", "Continue."]
    assert _effects(effects) == []
    state = _state(tmp_path)
    assert _native.native_calls(state) == []
    assert _native.ledger(state) == []
    [decision] = list(state.glob("decisions/*.json"))
    assert json.loads(decision.read_text())["decision"] == "approved"


# Pytest discovers fixtures by name.
_ = _offline_cli
