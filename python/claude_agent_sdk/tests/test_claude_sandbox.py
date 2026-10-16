"""Provider-neutral sandbox turns: the real Activity and SDK client, a fake runtime.

Local subprocess execution is the default and is not this file. These tests opt in with
``ClaudeTurnActivity(sandbox=...)``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import claude_agent_sdk
import pytest

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudePartialEvent,
    ClaudeSession,
    ClaudeSessionState,
    FileLeaseStore,
)
from temporalio.claude_agent_sdk import sandbox as sandbox_module
from temporalio.claude_agent_sdk._activity import ClaudeTurnActivity
from temporalio.claude_agent_sdk._lease import SessionLeaseLost
from temporalio.claude_agent_sdk._models import (
    CLAUDE_SANDBOX_LOST_ERROR_TYPE,
    CLAUDE_TURN_CONFIG_ERROR_TYPE,
    CLAUDE_TURN_REJECTED_ERROR_TYPE,
    CLAUDE_TURN_TIMEOUT_ERROR_TYPE,
    CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE,
    RESUMABLE_DETAIL,
    ClaudeDeferredTool,
    ClaudeNativeCall,
    ClaudeNativeDecision,
    ClaudeTurnInput,
)
from temporalio.claude_agent_sdk._native import (
    SUBPROCESS_ENV_SCRUB,
    NativeToolHooks,
    native_calls,
    sandbox_settings,
)
from temporalio.claude_agent_sdk._session import (
    _record_progress,
    _resumable,
    _unsettled_native_call,
)
from temporalio.claude_agent_sdk.sandbox import SandboxError, SandboxLost, SandboxTurn
from temporalio.claude_agent_sdk.testing import FakeClaudeCLI, result, text_round
from temporalio.client import WorkflowFailureError
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.exceptions import ApplicationError, CancelledError
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from .helpers import fixtures as fx
from .helpers import kill_worker as kw
from .helpers.fake_sandbox import CLI_VERSION, FakeSandbox, FakeSandboxRuntime
from .helpers.workflows import (
    ClaudeChat,
    ClaudeSandboxContinue,
    ClaudeSandboxLive,
    ClaudeSandboxSession,
    ClaudeStream,
    ClaudeStructured,
    Forecast,
)

pytestmark = fx.POSIX_ONLY


@pytest.fixture(autouse=True)
def _skip_version_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))


@pytest.fixture(autouse=True)
def _verified_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sandbox_module,
        "NATIVE_TOOLS_VERIFIED",
        frozenset({(claude_agent_sdk.__version__, CLI_VERSION)}),
    )


def _runtime(tmp_path: Path, cli: FakeClaudeCLI) -> FakeSandboxRuntime:
    return FakeSandboxRuntime(cli.cli_path, str(tmp_path / "workspace"))


def _activity(
    runtime: FakeSandboxRuntime, tmp_path: Path, **kw: Any
) -> ClaudeTurnActivity:
    kw.setdefault("workspace_root", tmp_path / "host")
    return ClaudeTurnActivity(sandbox=runtime, **kw)


def _input(**overrides: Any) -> ClaudeTurnInput:
    overrides.setdefault("session_id", fx.SESSION)
    return ClaudeTurnInput(prompt="Hello", **overrides)


async def _until_the_cli_runs(runtime: FakeSandboxRuntime) -> None:
    for _ in range(200):
        if (
            runtime.sandboxes
            and len(runtime.sandboxes[-1].execs) > 1
            and runtime.sandboxes[-1].processes
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the sandboxed CLI did not start")


async def test_a_turn_runs_in_a_sandbox_and_returns_its_snapshot(
    tmp_path: Path,
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)

    result_turn = await ActivityEnvironment().run(
        _activity(runtime, tmp_path).run_turn, _input()
    )

    assert result_turn.text == "Hello from the completed message."
    assert result_turn.sandbox_snapshot == "snap-1"
    (sandbox,) = runtime.sandboxes
    assert sandbox.session_id == fx.SESSION
    assert sandbox.started_from is None
    assert runtime.snapshots == [(sandbox, "snap-1")]
    assert sandbox.terminated == 1
    argv, env = sandbox.execs[-1]
    assert argv[:3] == ("sh", "-c", sandbox_module.LAUNCH_SCRIPT)
    assert argv[4] == f"{runtime.workspace_root}/{fx.SESSION}"
    assert argv[5] == cli.cli_path
    assert f"--session-id={fx.SESSION}" in argv
    assert f"--resume={fx.SESSION}" not in argv
    assert env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"
    assert "PATH" not in env


async def test_the_next_turn_starts_a_fresh_sandbox_from_the_snapshot(
    tmp_path: Path,
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    activity = _activity(runtime, tmp_path)
    first = await ActivityEnvironment().run(activity.run_turn, _input())

    await ActivityEnvironment().run(
        activity.run_turn, _input(resume=True, sandbox_snapshot=first.sandbox_snapshot)
    )

    second = runtime.sandboxes[1]
    assert second.started_from == first.sandbox_snapshot
    assert f"--resume={fx.SESSION}" in second.execs[-1][0]
    assert runtime.snapshots[-1][0] is second


async def test_a_lost_attempts_sandbox_is_terminated_and_snapshots_stay(
    tmp_path: Path,
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    lost = await runtime.start(session_id=fx.SESSION, snapshot="snap-kept")
    other = await runtime.start(session_id="someone-else", snapshot=None)
    runtime.snapshots.append((lost, "snap-kept"))

    await ActivityEnvironment().run(
        _activity(runtime, tmp_path).run_turn, _input(sandbox_snapshot="snap-kept")
    )

    assert lost.terminated == 1
    assert other.terminated == 0
    assert "snap-kept" in [snapshot for _sandbox, snapshot in runtime.snapshots]
    assert runtime.sandboxes[-1].started_from == "snap-kept"


async def test_a_failed_turn_still_reports_the_snapshot(tmp_path: Path) -> None:
    cli = FakeClaudeCLI(tmp_path / "cli", turn=fx.error_turn(), exit_code=1)
    runtime = _runtime(tmp_path, cli)

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(_activity(runtime, tmp_path).run_turn, _input())

    assert failed.value.details[0]["sandbox_snapshot"] == "snap-1"
    assert runtime.sandboxes[0].terminated == 1


async def test_a_snapshot_failure_keeps_the_old_id(tmp_path: Path) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    runtime.snapshot_error = RuntimeError("snapshot service down")

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(
            _activity(runtime, tmp_path).run_turn, _input(sandbox_snapshot="im-old")
        )

    assert failed.value.type == CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE
    assert failed.value.non_retryable
    assert "keeps snapshot im-old" in str(failed.value)
    assert runtime.sandboxes[0].terminated == 1
    assert runtime.snapshots == []


async def test_a_cli_that_will_not_exit_is_not_snapshotted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox_module, "_EXIT_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(sandbox_module, "_TERM_GRACE_SECONDS", 0.2)
    cli = FakeClaudeCLI(
        tmp_path / "cli",
        turn=fx.COMPLETED_ONLY,
        hang_on_eof=True,
        ignore_sigterm=True,
    )
    runtime = _runtime(tmp_path, cli)

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(_activity(runtime, tmp_path).run_turn, _input())

    assert "would not exit" in str(failed.value)
    assert runtime.snapshots == []
    assert runtime.sandboxes[0].terminated >= 1


async def test_a_cli_that_lingers_after_eof_is_stopped_and_snapshotted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox_module, "_EXIT_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(sandbox_module, "_TERM_GRACE_SECONDS", 0.2)
    cli = FakeClaudeCLI(tmp_path / "cli", turn=fx.COMPLETED_ONLY, hang_on_eof=True)
    runtime = _runtime(tmp_path, cli)

    await ActivityEnvironment().run(_activity(runtime, tmp_path).run_turn, _input())

    assert len(runtime.snapshots) == 1
    assert len(runtime.sandboxes[0].execs) == 3


async def test_cancellation_terminates_the_sandbox_and_takes_no_snapshot(
    tmp_path: Path,
) -> None:
    cli = FakeClaudeCLI(
        tmp_path / "cli", turn=fx.STREAMING_UNTIL_INTERRUPT, hang_after_prompt=True
    )
    runtime = _runtime(tmp_path, cli)
    env = ActivityEnvironment()
    activity = _activity(
        runtime,
        tmp_path,
        interrupt_timeout=timedelta(milliseconds=200),
        drain_timeout=timedelta(milliseconds=200),
        disconnect_timeout=timedelta(milliseconds=200),
    )
    task = asyncio.ensure_future(env.run(activity.run_turn, _input()))
    await _until_the_cli_runs(runtime)

    env.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runtime.sandboxes[0].terminated >= 1
    assert runtime.snapshots == []


async def test_a_dropped_stdin_is_a_lost_sandbox(tmp_path: Path) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    runtime.fail_writes = True
    lease = FileLeaseStore(tmp_path / "leases", shared=True)
    activity = _activity(runtime, tmp_path, lease_store=lease)

    with pytest.raises(ApplicationError) as lost:
        await ActivityEnvironment().run(
            activity.run_turn, _input(sandbox_snapshot="snap-0")
        )

    assert lost.value.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
    assert not lost.value.non_retryable
    assert runtime.sandboxes[0].terminated >= 1
    assert runtime.snapshots == []
    assert lost.value.details[0]["sandbox_snapshot"] is None
    assert lost.value.details[0][RESUMABLE_DETAIL] is True
    session = ClaudeSession(fx.SESSION)
    session.state.sandbox_snapshot = "snap-0"
    assert _resumable(session, lost.value)
    assert session.state.sandbox_snapshot == "snap-0"

    runtime.fail_writes = False
    resumed = await ActivityEnvironment().run(
        activity.run_turn,
        _input(resume=True, resumes_lost_attempt=True, sandbox_snapshot="snap-0"),
    )
    assert resumed.sandbox_snapshot == "snap-1"
    assert runtime.sandboxes[-1].started_from == "snap-0"
    assert runtime.sandboxes[-1].generation == 2


async def test_a_sandbox_dropped_before_the_cli_starts_names_the_old_snapshot(
    tmp_path: Path,
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    runtime.exec_error = RuntimeError("Modal Sandbox is shutting down")

    with pytest.raises(ApplicationError) as lost:
        await ActivityEnvironment().run(
            _activity(runtime, tmp_path).run_turn,
            _input(resume=True, sandbox_snapshot="snap-0"),
        )

    assert lost.value.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
    assert not lost.value.non_retryable
    assert "shutting down" in str(lost.value)
    assert "snap-0" in str(lost.value)
    assert runtime.sandboxes[0].terminated == 1
    assert runtime.snapshots == []


async def test_a_cli_that_is_not_the_verified_version_is_refused(
    tmp_path: Path,
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    runtime.cli_version = "9.9.9 (Claude Code)\n"

    with pytest.raises(ApplicationError) as refused:
        await ActivityEnvironment().run(_activity(runtime, tmp_path).run_turn, _input())

    assert refused.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE
    assert "9.9.9" in str(refused.value)
    assert runtime.sandboxes[0].terminated == 1
    assert len(runtime.sandboxes[0].execs) == 1


@pytest.mark.parametrize(
    ("worker_opts_in", "runtime_supports", "error_type", "hint"),
    [
        (True, False, CLAUDE_TURN_CONFIG_ERROR_TYPE, "vm"),
        (False, True, CLAUDE_TURN_REJECTED_ERROR_TYPE, "enable_native_tools"),
    ],
    ids=["runtime-without-native-tools", "worker-not-opted-in"],
)
async def test_native_tools_are_refused_before_a_sandbox_starts(
    tmp_path: Path,
    worker_opts_in: bool,
    runtime_supports: bool,
    error_type: str,
    hint: str,
) -> None:
    runtime = _runtime(tmp_path, fx.completed_cli(tmp_path))
    runtime.supports_native_tools = runtime_supports
    activity = _activity(runtime, tmp_path, enable_native_tools=worker_opts_in)

    with pytest.raises(ApplicationError) as refused:
        await ActivityEnvironment().run(
            activity.run_turn, _input(native_tools=["Bash"])
        )

    assert refused.value.type == error_type
    assert refused.value.non_retryable
    assert hint in str(refused.value)
    assert runtime.sandboxes == []


async def test_native_tools_run_when_the_runtime_allows_them(tmp_path: Path) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)

    result_turn = await ActivityEnvironment().run(
        _activity(runtime, tmp_path, enable_native_tools=True).run_turn,
        _input(native_tools=["Bash"]),
    )

    assert result_turn.text == "Hello from the completed message."
    remote = Path(runtime.cwd(fx.SESSION)).parent / ".native-tool-state" / fx.SESSION
    assert (remote / "_native_hook.py").is_file()
    (argv,) = [
        argv
        for argv, _env in runtime.sandboxes[0].execs
        if argv[:3] == ("sh", "-c", sandbox_module.LAUNCH_SCRIPT)
    ]
    # can_use_tool admits the calls the hook allowed; without it every asked call is denied.
    assert argv[argv.index("--permission-prompt-tool") + 1] == "stdio"
    assert runtime.sandboxes[0].terminated == 1


@pytest.mark.parametrize(
    ("image_env", "denied"),
    [
        ({"CLAUDE_CONFIG_DIR": "/claude-config"}, "/claude-config"),
        ({"HOME": "/home/agent"}, "/home/agent/.claude"),
        ({"CLAUDE_CONFIG_DIR": "relative/config", "HOME": "/home/agent"}, None),
    ],
)
async def test_the_sandbox_config_dir_is_denied_to_native_tools(
    tmp_path: Path, image_env: dict[str, str], denied: str | None
) -> None:
    """The config directory is where the sandbox resolves it (secrets and image environment
    included), not a runtime attribute; a relative one is not denied."""
    runtime = _runtime(tmp_path, fx.completed_cli(tmp_path))
    runtime.image_env = image_env

    await ActivityEnvironment().run(
        _activity(runtime, tmp_path, enable_native_tools=True).run_turn,
        _input(native_tools=["Bash"]),
    )

    (argv,) = [
        argv
        for argv, _env in runtime.sandboxes[0].execs
        if argv[:3] == ("sh", "-c", sandbox_module.LAUNCH_SCRIPT)
    ]
    settings = json.loads(argv[argv.index("--settings") + 1])
    deny = set(settings["permissions"]["deny"])
    if denied is None:
        assert not any(
            "relative/config" in rule or "agent/.claude" in rule for rule in deny
        )
    else:
        assert {f"Read(/{denied})", f"Edit(/{denied}/**)"} <= deny


@pytest.mark.parametrize("conflict", ["session_store", "workspace_store"])
def test_a_sandbox_cannot_be_combined_with_another_session_authority(
    tmp_path: Path, conflict: str
) -> None:
    runtime = FakeSandboxRuntime("claude", str(tmp_path))
    store = object()

    with pytest.raises(ValueError, match="cannot be combined"):
        if conflict == "session_store":
            ClaudeTurnActivity(sandbox=runtime, session_store=store)  # type: ignore[arg-type]
        else:
            ClaudeTurnActivity(sandbox=runtime, workspace_store=store)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be combined"):
        if conflict == "session_store":
            ClaudeAgentPlugin(sandbox=runtime, session_store=store)  # type: ignore[arg-type]
        else:
            ClaudeAgentPlugin(sandbox=runtime, workspace_store=store)  # type: ignore[arg-type]


def test_the_transport_implements_the_sdk_transport_interface() -> None:
    abstract = claude_agent_sdk.Transport.__abstractmethods__
    assert all(hasattr(sandbox_module.SandboxTransport, name) for name in abstract)


class _OneWriteAtATime:
    """A process whose stdin breaks if two writes overlap, as Modal's does."""

    def __init__(self) -> None:
        self.busy = False
        self.overlaps = 0
        self.written: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.overlaps += self.busy
        self.busy = True
        await asyncio.sleep(0.01)
        self.written.append(data)
        self.busy = False

    async def end_input(self) -> None:
        await self.write(b"")

    async def _no_output(self) -> AsyncIterator[bytes]:
        items: list[bytes] = []
        for item in items:
            yield item

    def stdout(self) -> AsyncIterator[bytes]:
        return self._no_output()

    stderr = stdout


async def test_writes_to_the_cli_stdin_are_serialized(tmp_path: Path) -> None:
    process = _OneWriteAtATime()

    class _Sandbox:
        async def exec(self, argv: Any, *, env: Any = None) -> _OneWriteAtATime:
            del argv, env
            return process

    transport = sandbox_module.SandboxTransport(
        _Sandbox(),  # type: ignore[arg-type]
        "claude",
        claude_agent_sdk.ClaudeAgentOptions(cwd=tmp_path),
    )
    await transport.connect()
    await asyncio.gather(
        *(transport.write(f"{i}\n") for i in range(5)), transport.end_input()
    )

    assert process.overlaps == 0
    assert len(process.written) == 6


async def test_a_corrupt_message_from_the_sandboxed_cli_loses_the_sandbox(
    tmp_path: Path,
) -> None:
    """The SDK's own transport raises on such a line; dropping it could drop a tool call."""

    class _Corrupt(_OneWriteAtATime):
        async def _lines(self) -> AsyncIterator[bytes]:
            yield b'{"type": "system"}\n{"type": "assist\n'

        def stdout(self) -> AsyncIterator[bytes]:
            return self._lines()

        async def wait(self) -> int:
            return 0

    class _Sandbox:
        async def exec(self, argv: Any, *, env: Any = None) -> _Corrupt:
            del argv, env
            return _Corrupt()

    transport = sandbox_module.SandboxTransport(
        _Sandbox(),  # type: ignore[arg-type]
        "claude",
        claude_agent_sdk.ClaudeAgentOptions(cwd=tmp_path),
    )
    await transport.connect()
    seen: list[dict[str, Any]] = []
    with pytest.raises(SandboxLost):
        async for message in transport.read_messages():
            seen.append(message)

    assert seen == [{"type": "system"}]
    assert transport.lost


def test_session_state_from_before_snapshots_still_loads() -> None:
    old = ClaudeSessionState.model_validate({"version": 1, "session_id": fx.SESSION})

    assert old.version == 1
    assert old.sandbox_snapshot is None
    assert "sandbox_snapshot" not in ClaudeSessionState().model_dump()
    assert ClaudeSessionState(sandbox_snapshot="im-1").model_dump()[
        "sandbox_snapshot"
    ] == ("im-1")


def test_an_unset_snapshot_stays_out_of_the_activity_payload() -> None:
    dumped = ClaudeTurnInput(prompt="hi").model_dump(mode="json")

    assert "sandbox_snapshot" not in dumped
    assert (
        ClaudeTurnInput(prompt="hi", sandbox_snapshot="im-1").model_dump()[
            "sandbox_snapshot"
        ]
        == "im-1"
    )


async def test_a_sandbox_killed_mid_turn_is_retried_from_the_last_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawn = asyncio.create_subprocess_exec
    spawning, release, spawned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def delayed_spawn(*argv: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        cli = argv[:3] == ("sh", "-c", sandbox_module.LAUNCH_SCRIPT)
        if cli:
            spawning.set()
            await release.wait()
        process = await spawn(*argv, **kwargs)
        if cli:
            spawned.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    cli = FakeClaudeCLI(
        tmp_path / "cli", turn=fx.STREAMING_UNTIL_INTERRUPT, hang_after_prompt=True
    )
    runtime = _runtime(tmp_path, cli)
    lease = FileLeaseStore(tmp_path / "leases", shared=True)
    activity = _activity(runtime, tmp_path, lease_store=lease)
    env = ActivityEnvironment()
    task = asyncio.ensure_future(
        env.run(activity.run_turn, _input(sandbox_snapshot="im-old"))
    )
    try:
        await asyncio.wait_for(spawning.wait(), timeout=10)
        ready = asyncio.create_task(_until_the_cli_runs(runtime))
        await asyncio.sleep(0)
        reported_ready_before_spawn = ready.done()
    finally:
        release.set()
    await asyncio.wait_for(spawned.wait(), timeout=10)
    await asyncio.wait_for(ready, timeout=10)

    await runtime.sandboxes[0].terminate()
    with pytest.raises(ApplicationError) as lost:
        await task

    assert lost.value.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
    assert not lost.value.non_retryable
    assert "im-old" in str(lost.value)
    assert lost.value.details[0]["sandbox_snapshot"] is None
    session = ClaudeSession(fx.SESSION)
    session.state.sandbox_snapshot = "im-old"
    assert _resumable(session, lost.value)
    assert session.state.sandbox_snapshot == "im-old"
    assert not reported_ready_before_spawn

    fx.completed_cli(tmp_path)
    resumed = await ActivityEnvironment().run(
        activity.run_turn,
        _input(resume=True, resumes_lost_attempt=True, sandbox_snapshot="im-old"),
    )
    assert resumed.sandbox_snapshot == "snap-1"
    assert runtime.sandboxes[-1].started_from == "im-old"


def test_a_sandboxed_cli_omits_the_subprocess_scrub() -> None:
    settings, env = sandbox_settings(
        {"ANTHROPIC_API_KEY": ""}, ["ANTHROPIC_*"], scrub=False
    )

    assert env == {}
    assert settings["credentials"]["envVars"] == [
        {"name": "ANTHROPIC_API_KEY", "mode": "deny"}
    ]


def test_a_sandboxed_turn_denies_secret_names_the_worker_cannot_see(
    tmp_path: Path,
) -> None:
    """The runtime's secrets are not in the worker's environment: deny them by name."""
    hooks = NativeToolHooks(tmp_path)
    turn = _input(native_tools=["Bash"])
    runtime = FakeSandboxRuntime("claude", str(tmp_path))

    def denied(**kw: Any) -> set[str]:
        options = _activity(runtime, tmp_path, **kw)._options(turn, None, hooks)
        assert SUBPROCESS_ENV_SCRUB not in options.env
        sandbox: Any = options.sandbox
        return {e["name"] for e in sandbox["credentials"]["envVars"]}

    assert denied() >= {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "AWS_SESSION_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "OPENAI_API_KEY",
    }
    assert denied(native_env_denylist=["STRIPE_KEY", "GH_*"]) == {
        "STRIPE_KEY",
        "GH_TOKEN",
    }


async def test_the_cli_pid_is_recorded_where_native_tools_cannot_write(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "work" / "sid"
    process = await asyncio.create_subprocess_exec(
        "sh", "-c", sandbox_module.LAUNCH_SCRIPT, "sh", str(cwd), "true"
    )
    assert await process.wait() == 0
    pidfile = Path(sandbox_module.cli_pidfile(str(cwd)))
    assert (
        pidfile.parent == Path(sandbox_module.sandbox_state_dir(str(cwd), "x")).parent
    )
    assert int(pidfile.read_text()) == process.pid
    assert list(cwd.iterdir()) == []


def test_importing_the_package_does_not_import_modal() -> None:
    code = (
        "import sys\n"
        "import temporalio.claude_agent_sdk\n"
        "import temporalio.claude_agent_sdk.sandbox\n"
        "assert 'modal' not in sys.modules\n"
        "assert 'temporalio.claude_agent_sdk.sandbox.modal' not in sys.modules\n"
    )
    subprocess_run = __import__("subprocess").run
    completed = subprocess_run(
        [sys.executable, "-c", code], check=False, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr.decode()


async def test_a_second_worker_resumes_the_committed_snapshot(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    client = kw.client(env)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"

    def plugin() -> ClaudeAgentPlugin:
        return ClaudeAgentPlugin(
            sandbox=runtime, workspace_root=tmp_path / "host", cli_path=cli.cli_path
        )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeSandboxSession],
        plugins=[plugin()],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            ClaudeSandboxSession.run, id=task_queue, task_queue=task_queue
        )
        for _ in range(100):
            if await handle.query(ClaudeSandboxSession.snapshot):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("the first turn did not commit a snapshot")
        assert await handle.query(ClaudeSandboxSession.snapshot) == "snap-1"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeSandboxSession],
        plugins=[plugin()],
        max_cached_workflows=0,
    ):
        await handle.signal(ClaudeSandboxSession.proceed)
        text = await handle.result()

    first, second, reply_1, reply_2 = text.split("|")
    assert first == "snap-1"
    assert second == "snap-2"
    assert reply_1 == reply_2 == "Hello from the completed message."
    assert runtime.sandboxes[1].started_from == "snap-1"
    assert (
        f"--resume={runtime.sandboxes[1].session_id}"
        in runtime.sandboxes[1].execs[-1][0]
    )


async def test_continue_as_new_carries_the_sandbox_snapshot(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    client = kw.client(env)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeSandboxContinue],
        plugins=[
            ClaudeAgentPlugin(
                sandbox=runtime, workspace_root=tmp_path / "host", cli_path=cli.cli_path
            )
        ],
    ):
        handle = await client.start_workflow(
            ClaudeSandboxContinue.run, id=task_queue, task_queue=task_queue
        )
        for _ in range(100):
            if await handle.query(ClaudeSandboxContinue.snapshot):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("the first turn did not commit a snapshot")
        await handle.signal(ClaudeSandboxContinue.proceed)
        assert await handle.result() == "snap-2"

    first_run = handle.first_execution_run_id
    assert first_run is not None
    history = await client.get_workflow_handle(
        handle.id, run_id=first_run
    ).fetch_history()
    continued = next(
        event.workflow_execution_continued_as_new_event_attributes
        for event in history.events
        if event.HasField("workflow_execution_continued_as_new_event_attributes")
    )
    (carried,) = await client.data_converter.decode(
        list(continued.input.payloads), [ClaudeSessionState]
    )
    assert carried.sandbox_snapshot == "snap-1"
    assert runtime.sandboxes[1].started_from == "snap-1"
    replayer = Replayer(
        workflows=[ClaudeSandboxContinue],
        plugins=[ClaudeAgentPlugin(register_activities=False)],
    )
    await replayer.replay_workflow(history)


async def test_streaming_works_in_a_sandbox_and_the_result_stays_authoritative(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    turn = [
        *fx.START,
        *text_round("msg_stream", ["Hello ", "world"], input_tokens=4, output_tokens=2),
        result(text="Hello world"),
    ]
    cli = FakeClaudeCLI(tmp_path / "cli", turn=turn)
    runtime = _runtime(tmp_path, cli)
    client = kw.client(env)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeStream],
        plugins=[
            ClaudeAgentPlugin(
                sandbox=runtime, workspace_root=tmp_path / "host", cli_path=cli.cli_path
            )
        ],
    ):
        handle = await client.start_workflow(
            ClaudeStream.run, "hi", id=task_queue, task_queue=task_queue
        )
        stream = WorkflowStreamClient.create(client, handle.id)
        published: list[ClaudePartialEvent] = []

        async def collect() -> None:
            async for item in stream.subscribe(
                ["claude"],
                from_offset=0,
                result_type=ClaudePartialEvent,
                poll_cooldown=timedelta(milliseconds=20),
            ):
                published.append(item.data)
                if item.data.kind == "message_stop":
                    return

        await asyncio.wait_for(asyncio.create_task(collect()), timeout=20)
        await handle.signal(ClaudeStream.finish)
        assert await handle.result() == "Hello world"

    assert [event.text for event in published if event.text] == ["Hello ", "world"]
    assert runtime.sandboxes[0].terminated == 1


async def test_structured_output_works_in_a_sandbox(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    cli = FakeClaudeCLI(tmp_path / "cli", turn=fx.STRUCTURED)
    runtime = _runtime(tmp_path, cli)
    client = kw.client(env)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeStructured],
        plugins=[
            ClaudeAgentPlugin(
                sandbox=runtime, workspace_root=tmp_path / "host", cli_path=cli.cli_path
            )
        ],
    ):
        forecast = await client.execute_workflow(
            ClaudeStructured.run, 30, id=task_queue, task_queue=task_queue
        )

    assert forecast == Forecast(city="Paris", temp_f=72)
    assert runtime.snapshots


async def test_the_workflow_tool_bridge_works_in_a_sandbox(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    cli = FakeClaudeCLI(
        tmp_path / "cli",
        turn=fx.tool_turn([("toolu_1", "get_weather", {"city": "Paris"})]),
    )
    runtime = _runtime(tmp_path, cli)
    client = kw.client(env)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeChat],
        plugins=[
            ClaudeAgentPlugin(
                sandbox=runtime, workspace_root=tmp_path / "host", cli_path=cli.cli_path
            )
        ],
    ):
        handle = await client.start_workflow(
            ClaudeChat.run, id=task_queue, task_queue=task_queue
        )
        reply = await handle.execute_update(ClaudeChat.chat_with_tool, "weather?")
        calls = await handle.query(ClaudeChat.weather_calls)
        await handle.terminate()

    assert reply == "All done."
    assert calls == 1
    assert runtime.sandboxes[0].terminated == 1


async def test_a_workflow_timeout_terminates_the_sandbox(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    cli = FakeClaudeCLI(tmp_path / "cli", turn=[{"fake": "sleep", "seconds": 30}])
    runtime = _runtime(tmp_path, cli)
    client = kw.client(env)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeSandboxLive],
        plugins=[
            ClaudeAgentPlugin(
                sandbox=runtime,
                workspace_root=tmp_path / "host",
                cli_path=cli.cli_path,
                interrupt_timeout=timedelta(milliseconds=200),
                drain_timeout=timedelta(milliseconds=200),
                disconnect_timeout=timedelta(milliseconds=200),
            )
        ],
    ):
        with pytest.raises(WorkflowFailureError) as failed:
            await client.execute_workflow(
                ClaudeSandboxLive.run, 8, id=task_queue, task_queue=task_queue
            )

    assert isinstance(failed.value.cause, ApplicationError)
    assert failed.value.cause.type == CLAUDE_TURN_TIMEOUT_ERROR_TYPE
    assert runtime.sandboxes[0].terminated >= 1
    assert runtime.snapshots == []


async def test_cancelling_the_workflow_terminates_the_sandbox(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    cli = FakeClaudeCLI(tmp_path / "cli", turn=[{"fake": "sleep", "seconds": 30}])
    runtime = _runtime(tmp_path, cli)
    client = kw.client(env)
    task_queue = f"claude-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeSandboxLive],
        plugins=[
            ClaudeAgentPlugin(
                sandbox=runtime,
                workspace_root=tmp_path / "host",
                cli_path=cli.cli_path,
                interrupt_timeout=timedelta(milliseconds=200),
                drain_timeout=timedelta(milliseconds=200),
                disconnect_timeout=timedelta(milliseconds=200),
            )
        ],
    ):
        handle = await client.start_workflow(
            ClaudeSandboxLive.run, 60, id=task_queue, task_queue=task_queue
        )
        await _until_the_cli_runs(runtime)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as failed:
            await handle.result()

    assert isinstance(failed.value.cause, CancelledError)
    assert runtime.sandboxes[0].terminated >= 1
    assert runtime.snapshots == []


async def test_a_lost_sandbox_without_a_shared_lease_is_not_resumed(
    tmp_path: Path,
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    runtime.fail_writes = True

    with pytest.raises(ApplicationError) as lost:
        await ActivityEnvironment().run(
            _activity(runtime, tmp_path).run_turn, _input(sandbox_snapshot="snap-0")
        )

    assert lost.value.details[0][RESUMABLE_DETAIL] is False
    assert runtime.snapshots == []
    session = ClaudeSession(fx.SESSION)
    session.state.sandbox_snapshot = "snap-0"
    assert not _resumable(session, lost.value)
    assert session.state.sandbox_snapshot == "snap-0"
    with pytest.raises(ApplicationError) as rejected:
        await ActivityEnvironment().run(
            _activity(runtime, tmp_path).run_turn,
            _input(resumes_lost_attempt=True, sandbox_snapshot="snap-0"),
        )
    assert rejected.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE
    assert rejected.value.non_retryable
    assert "lease store is not shared" in str(rejected.value)


def _share(older: FakeSandboxRuntime, newer: FakeSandboxRuntime) -> None:
    sandboxes: list[Any] = []
    snapshots: list[Any] = []
    older.sandboxes = newer.sandboxes = sandboxes
    older.snapshots = newer.snapshots = snapshots


async def _start(
    runtime: FakeSandboxRuntime, generation: int, *, lose_after_create: bool
) -> SandboxTurn:
    seen = 0

    async def owns() -> None:
        nonlocal seen
        seen += 1
        if lose_after_create and seen >= 2:
            raise SessionLeaseLost(f"generation {generation} was superseded")

    return await SandboxTurn.start(
        runtime,
        session_id=fx.SESSION,
        snapshot="snap-0",
        generation=generation,
        owns=owns,
    )


async def test_both_sweeps_finish_before_create_and_only_the_newer_sandbox_remains(
    tmp_path: Path,
) -> None:
    cli = fx.completed_cli(tmp_path)
    older = _runtime(tmp_path / "older", cli)
    newer = _runtime(tmp_path / "newer", cli)
    _share(older, newer)
    older_sweep, newer_sweep = asyncio.Event(), asyncio.Event()
    older_swept, newer_swept = asyncio.Event(), asyncio.Event()
    older_creating, newer_creating = asyncio.Event(), asyncio.Event()
    older_create, newer_create = asyncio.Event(), asyncio.Event()
    older.sweep_started, newer.sweep_started = older_sweep, newer_sweep
    older.sweep_release, newer.sweep_release = older_swept, newer_swept
    older.create_started, newer.create_started = older_creating, newer_creating
    older.create_release, newer.create_release = older_create, newer_create

    older_turn = asyncio.create_task(_start(older, 1, lose_after_create=True))
    newer_turn = asyncio.create_task(_start(newer, 2, lose_after_create=False))
    await older_sweep.wait()
    await newer_sweep.wait()
    older_swept.set()
    newer_swept.set()
    await older_creating.wait()
    await newer_creating.wait()
    older_create.set()
    newer_create.set()

    with pytest.raises(SessionLeaseLost):
        await older_turn
    winner = await newer_turn
    live = [sandbox for sandbox in older.sandboxes if not sandbox.terminated]
    assert [sandbox.generation for sandbox in live] == [2]
    assert await winner.snapshot() == "snap-1"
    assert len(older.snapshots) == 1
    await winner.terminate()


async def test_an_old_list_returning_after_a_newer_create_does_not_kill_it(
    tmp_path: Path,
) -> None:
    cli = fx.completed_cli(tmp_path)
    older = _runtime(tmp_path / "older", cli)
    newer = _runtime(tmp_path / "newer", cli)
    _share(older, newer)
    older_sweep, older_swept = asyncio.Event(), asyncio.Event()
    older.sweep_started, older.sweep_release = older_sweep, older_swept

    older_turn = asyncio.create_task(_start(older, 1, lose_after_create=True))
    await older_sweep.wait()
    winner = await _start(newer, 2, lose_after_create=False)
    older_swept.set()

    with pytest.raises(SessionLeaseLost):
        await older_turn
    live = [sandbox for sandbox in older.sandboxes if not sandbox.terminated]
    assert [sandbox.generation for sandbox in live] == [2]
    assert await winner.snapshot() == "snap-1"
    assert len(older.snapshots) == 1
    await winner.terminate()


def _remote_calls(runtime: FakeSandboxRuntime) -> Path:
    return Path(runtime.cwd(fx.SESSION)).parent / ".native-tool-state" / fx.SESSION


def _seed_remote_call(runtime: FakeSandboxRuntime) -> Path:
    directory = _remote_calls(runtime)
    directory.mkdir(parents=True)
    (directory / "calls.json").write_text(
        json.dumps(
            {
                "version": 1,
                "calls": [
                    {
                        "tool_use_id": "toolu_remote",
                        "tool_name": "Bash",
                        "key": "k",
                        "status": "started",
                    }
                ],
            }
        )
    )
    return directory


async def test_a_failed_startup_pull_does_not_erase_the_snapshot_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    directory = _seed_remote_call(runtime)

    async def unreadable(*_args: Any, **_kwargs: Any) -> None:
        raise SandboxError("the sandbox native ledger could not be read")

    monkeypatch.setattr(sandbox_module, "pull_state", unreadable)

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(
            _activity(runtime, tmp_path, enable_native_tools=True).run_turn,
            _input(native_tools=["Bash"]),
        )

    assert runtime.snapshots == []
    assert failed.value.details[0]["sandbox_snapshot"] is None
    assert (
        json.loads((directory / "calls.json").read_text())["calls"][0]["tool_use_id"]
        == "toolu_remote"
    )


async def test_a_failed_final_pull_does_not_snapshot_and_keeps_the_call_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    directory = _seed_remote_call(runtime)

    async def unreadable(*_args: Any, **_kwargs: Any) -> None:
        raise SandboxError("the sandbox native ledger could not be read")

    monkeypatch.setattr("temporalio.claude_agent_sdk._activity.pull_state", unreadable)
    decision = ClaudeNativeDecision(
        call=ClaudeDeferredTool(
            tool_use_id="toolu_approved", name="Bash", input={"command": "true"}
        ),
        approved=True,
    )

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(
            _activity(runtime, tmp_path, enable_native_tools=True).run_turn,
            _input(resume=True, native_tools=["Bash"], native_decision=decision),
        )

    assert runtime.snapshots == []
    assert failed.value.details[0]["sandbox_snapshot"] is None
    assert failed.value.details[0][RESUMABLE_DETAIL] is False
    ids = {call["tool_use_id"] for call in failed.value.details[0]["native_calls"]}
    assert "toolu_remote" in ids
    assert "toolu_approved" in ids
    pushed = json.loads((directory / "calls.json").read_text())["calls"]
    assert {call["tool_use_id"] for call in pushed} >= {"toolu_remote"}


@pytest.mark.parametrize(
    ("flag", "detail"),
    [("fail_wait", "could not be observed"), ("fail_stdout", "output stream failed")],
)
async def test_a_process_observation_failure_does_not_snapshot(
    tmp_path: Path, flag: str, detail: str
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    setattr(runtime, flag, True)

    with pytest.raises(ApplicationError) as lost:
        await ActivityEnvironment().run(
            _activity(runtime, tmp_path).run_turn, _input(sandbox_snapshot="snap-0")
        )

    assert lost.value.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
    assert detail in str(lost.value)
    assert lost.value.details[0]["sandbox_snapshot"] is None
    assert runtime.snapshots == []
    assert runtime.sandboxes[0].terminated >= 1


def _approved(name: str, tool_input: dict[str, str]) -> ClaudeNativeDecision:
    return ClaudeNativeDecision(
        call=ClaudeDeferredTool(
            tool_use_id=f"toolu_{name}", name=name, input=tool_input
        ),
        approved=True,
    )


async def _lose_the_sandbox_and_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **turn: Any
) -> ApplicationError:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    runtime.fail_writes = True
    lease = FileLeaseStore(tmp_path / "leases", shared=True)

    async def unreadable(*_args: Any, **_kwargs: Any) -> None:
        raise SandboxError("the sandbox native ledger could not be read")

    monkeypatch.setattr("temporalio.claude_agent_sdk._activity.pull_state", unreadable)
    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(
            _activity(
                runtime, tmp_path, lease_store=lease, enable_native_tools=True
            ).run_turn,
            _input(native_tools=["Bash", "Read"], **turn),
        )
    assert runtime.snapshots == []
    return failed.value


async def test_a_lost_sandbox_stays_lost_when_the_ledger_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lost = await _lose_the_sandbox_and_the_ledger(tmp_path, monkeypatch)

    assert lost.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
    assert lost.details[0][RESUMABLE_DETAIL] is True
    assert lost.details[0]["ledger_unreadable"] is True
    assert lost.details[0]["sandbox_snapshot"] is None
    assert lost.details[0]["native_calls"] == []
    session = ClaudeSession(fx.SESSION)
    assert _resumable(session, lost)


async def test_a_lost_sandbox_with_a_repeatable_effect_stays_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lost = await _lose_the_sandbox_and_the_ledger(
        tmp_path,
        monkeypatch,
        resume=True,
        native_decision=_approved("Read", {"file_path": "f"}),
    )

    assert lost.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
    assert lost.details[0][RESUMABLE_DETAIL] is True
    assert lost.details[0]["ledger_unreadable"] is True
    assert lost.details[0]["sandbox_snapshot"] is None
    started = [
        call
        for call in lost.details[0]["native_calls"]
        if call["tool_use_id"] == "toolu_Read"
    ]
    assert [call["status"] for call in started] == ["started"]


async def test_a_lost_sandbox_with_an_unsettled_effect_stays_a_sandbox_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lost = await _lose_the_sandbox_and_the_ledger(
        tmp_path,
        monkeypatch,
        resume=True,
        native_decision=_approved("Bash", {"command": "true"}),
    )

    assert lost.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
    assert lost.details[0][RESUMABLE_DETAIL] is True
    assert lost.details[0]["ledger_unreadable"] is True
    assert lost.details[0]["sandbox_snapshot"] is None
    started = [
        call
        for call in lost.details[0]["native_calls"]
        if call["tool_use_id"] == "toolu_Bash"
    ]
    assert [call["status"] for call in started] == ["started"]


async def test_an_approved_effect_stays_started_until_a_snapshot_holds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker lost before the snapshot leaves a sandbox without the effect: the ledger it
    reported must make the workflow fail the message, not resume and refuse the call as run."""
    command = {"command": "echo hi > hi.txt"}
    cli = FakeClaudeCLI(
        tmp_path / "cli",
        turn=[
            *fx.native_round("msg_n", "toolu_Bash", "Bash", command),
            *fx.answer("Done."),
        ],
        effects=str(tmp_path / "effects.jsonl"),
    )
    runtime = _runtime(tmp_path, cli)
    host = tmp_path / "host" / ".native-tool-state" / fx.SESSION
    at_snapshot: list[list[ClaudeNativeCall]] = []
    snapshot = FakeSandbox.snapshot

    async def observed(sandbox: FakeSandbox) -> str:
        at_snapshot.append(
            [ClaudeNativeCall.model_validate(c) for c in native_calls(host)]
        )
        return await snapshot(sandbox)

    monkeypatch.setattr(FakeSandbox, "snapshot", observed)
    turn = _input(
        resume=True,
        native_tools=["Bash"],
        native_decision=_approved("Bash", command),
    )

    result_turn = await ActivityEnvironment().run(
        _activity(runtime, tmp_path, enable_native_tools=True).run_turn, turn
    )

    assert result_turn.text == "Done."
    assert (tmp_path / "effects.jsonl").read_text().count("toolu_Bash") == 1
    assert [(c.tool_use_id, c.status) for c in result_turn.native_calls] == [
        ("toolu_Bash", "finished")
    ]
    (before,) = at_snapshot
    assert [(c.tool_use_id, c.status) for c in before] == [("toolu_Bash", "started")]
    session = ClaudeSession(fx.SESSION)
    _record_progress(session, before, None)
    assert _unsettled_native_call(session, turn) == turn.native_decision.call  # type: ignore[union-attr]


async def test_heartbeat_names_snapshot_holding_finished_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold teardown after snapshot commit and inspect a real Activity heartbeat."""
    command = {"command": "echo hi > hi.txt"}
    cli = FakeClaudeCLI(
        tmp_path / "cli",
        turn=[
            *fx.native_round("msg_n", "toolu_Bash", "Bash", command),
            *fx.answer("Done."),
        ],
        effects=str(tmp_path / "effects.jsonl"),
    )
    runtime = _runtime(tmp_path, cli)
    teardown = asyncio.Event()
    release = asyncio.Event()
    heartbeat = asyncio.Event()
    beats: list[dict[str, Any]] = []
    terminate = FakeSandbox.terminate

    async def held(sandbox: FakeSandbox) -> None:
        teardown.set()
        await asyncio.wait_for(release.wait(), 5)
        await terminate(sandbox)

    def observe(details: dict[str, Any]) -> None:
        if teardown.is_set():
            beats.append(details)
            heartbeat.set()

    monkeypatch.setattr(FakeSandbox, "terminate", held)
    env = ActivityEnvironment()
    env.on_heartbeat = observe
    env.info = dataclasses.replace(env.info, heartbeat_timeout=timedelta(seconds=1))
    task = asyncio.create_task(
        env.run(
            _activity(runtime, tmp_path, enable_native_tools=True).run_turn,
            _input(
                resume=True,
                sandbox_snapshot="snap-old",
                native_tools=["Bash"],
                native_decision=_approved("Bash", command),
            ),
        )
    )
    try:
        await asyncio.wait_for(teardown.wait(), 10)
        await asyncio.wait_for(heartbeat.wait(), 4)
        assert runtime.snapshots[-1][1] == "snap-1"
        assert [(c["tool_use_id"], c["status"]) for c in beats[-1]["native_calls"]] == [
            ("toolu_Bash", "finished")
        ]
        assert beats[-1]["sandbox_snapshot"] == "snap-1", beats[-1]
    finally:
        release.set()
        await asyncio.wait_for(task, 5)


@pytest.mark.parametrize("code", [2, 137, 143])
async def test_a_native_state_probe_other_than_absent_does_not_overwrite(
    tmp_path: Path, code: int
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    runtime.probe_exit = code
    directory = _seed_remote_call(runtime)
    before = (directory / "calls.json").read_text()

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(
            _activity(runtime, tmp_path, enable_native_tools=True).run_turn,
            _input(native_tools=["Bash"]),
        )

    assert failed.value.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
    assert f"exit {code}" in str(failed.value)
    assert failed.value.details[0]["sandbox_snapshot"] is None
    assert runtime.snapshots == []
    assert (directory / "calls.json").read_text() == before
    assert not any(
        "rm -rf" in part
        for sandbox in runtime.sandboxes
        for parts, _env in sandbox.execs
        for part in parts
    )


async def test_a_sandboxed_cli_that_exits_without_a_result_is_not_resumed(
    tmp_path: Path,
) -> None:
    """The snapshot holds the crashed run's transcript, and a sandboxed attempt cannot plan
    around it: starting over it is refused as in use, and resuming re-sends the prompt."""
    cli = FakeClaudeCLI(tmp_path / "cli", turn=[fx.SYSTEM_INIT], exit_code=1)
    runtime = _runtime(tmp_path, cli)
    lease = FileLeaseStore(tmp_path / "leases", shared=True)
    activity = _activity(runtime, tmp_path, lease_store=lease)

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(activity.run_turn, _input())

    assert failed.value.details[0]["sandbox_snapshot"] == "snap-1"
    assert failed.value.details[0][RESUMABLE_DETAIL] is False
    assert not _resumable(ClaudeSession(fx.SESSION), failed.value)


def _native_run(
    tmp_path: Path, *, crash: bool = False
) -> tuple[FakeSandboxRuntime, ClaudeTurnInput, ClaudeTurnActivity]:
    command = {"command": "echo hi > hi.txt"}
    effect = fx.native_round("msg_n", "toolu_Bash", "Bash", command)
    cli = FakeClaudeCLI(
        tmp_path / "cli",
        turn=effect if crash else [*effect, *fx.answer("Done.")],
        effects=str(tmp_path / "effects.jsonl"),
        exit_code=1 if crash else 0,
    )
    runtime = _runtime(tmp_path, cli)
    turn = _input(
        resume=True, native_tools=["Bash"], native_decision=_approved("Bash", command)
    )
    return runtime, turn, _activity(runtime, tmp_path, enable_native_tools=True)


async def test_a_snapshot_failure_after_a_run_reports_its_started_calls(
    tmp_path: Path,
) -> None:
    runtime, turn, activity = _native_run(tmp_path)
    runtime.snapshot_error = SandboxError("snapshot service down")

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(activity.run_turn, turn)

    assert failed.value.type == CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE
    calls = failed.value.details[0]["native_calls"]
    assert [(c["tool_use_id"], c["status"]) for c in calls] == [
        ("toolu_Bash", "started")
    ]


async def test_a_snapshot_failure_after_a_failed_run_reports_its_started_calls(
    tmp_path: Path,
) -> None:
    runtime, turn, activity = _native_run(tmp_path, crash=True)
    runtime.snapshot_error = SandboxError("snapshot service down")

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(activity.run_turn, turn)

    details = failed.value.details[0]
    assert details["sandbox_snapshot"] is None
    assert [(c["tool_use_id"], c["status"]) for c in details["native_calls"]] == [
        ("toolu_Bash", "started")
    ]


async def test_a_sandbox_lost_after_its_result_still_reports_the_spend(
    tmp_path: Path,
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli)
    runtime.fail_wait = True

    with pytest.raises(ApplicationError) as lost:
        await ActivityEnvironment().run(
            _activity(runtime, tmp_path).run_turn, _input(sandbox_snapshot="snap-0")
        )

    assert lost.value.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
    details = lost.value.details[0]
    assert details["cost_usd"] > 0
    assert details["cumulative_cost"] is False


_ = (_skip_version_probe, _verified_pair)
