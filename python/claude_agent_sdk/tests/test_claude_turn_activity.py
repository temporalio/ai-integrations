"""Turn Activity lifecycle against the fake CLI."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import claude_agent_sdk
import pytest
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from temporalio import activity as temporal_activity
from temporalio.claude_agent_sdk import _activity as _activity_module
from temporalio.claude_agent_sdk import _launcher, _native
from temporalio.claude_agent_sdk._activity import (
    _CLI_ENV,
    _PGID_DIR_NAME,
    NONESSENTIAL_TRAFFIC_ENV,
    ClaudeTurnActivity,
    _CliGroup,
    _leased,
    _stderr_tail,
    sandbox_violations,
)
from temporalio.claude_agent_sdk._bridge import ToolBridgeClient
from temporalio.claude_agent_sdk._launcher import PGID_FILE_ENV
from temporalio.claude_agent_sdk._lease import FileLeaseStore, LeaseRecord
from temporalio.claude_agent_sdk._models import (
    CLAUDE_CLI_UNTRACKED_ERROR_TYPE,
    CLAUDE_RESULT_ERROR_TYPE,
    CLAUDE_SESSION_LEASE_LOST_ERROR_TYPE,
    CLAUDE_TURN_REJECTED_ERROR_TYPE,
    CONTINUE_PROMPT,
    RESUMABLE_DETAIL,
    ClaudeBridgeBatch,
    ClaudeDeferredTool,
    ClaudeNativeCall,
    ClaudeNativeDecision,
    ClaudeToolDeclaration,
    ClaudeToolRequest,
    ClaudeTurnInput,
    ClaudeTurnOptions,
    TokenUsage,
    mcp_tool_name,
    tool_call_key,
)
from temporalio.claude_agent_sdk._native import NATIVE_STATE_DIR_NAME
from temporalio.claude_agent_sdk._native_hook import (
    ALREADY_RAN,
    UNKNOWN_OUTCOME,
    call_key,
)
from temporalio.claude_agent_sdk._session import (
    ClaudeSession,
    ClaudeToolRecord,
    _ToolBridge,
    recorded_tool_calls,
)
from temporalio.claude_agent_sdk.testing import FakeClaudeCLI
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from .conftest import scripted_cli as _cli
from .helpers import fixtures as fx

# A raise, unlike a marker, also tells type checkers that the POSIX calls below are unreachable.
if sys.platform == "win32":
    raise pytest.skip.Exception(fx.POSIX_ONLY_REASON, allow_module_level=True)


@pytest.fixture(autouse=True)
def _skip_version_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")


def _input(**overrides: Any) -> ClaudeTurnInput:
    return ClaudeTurnInput(prompt="What's the weather?", **overrides)


def _activity(cli_path: str | None, **kw: Any) -> ClaudeTurnActivity:
    return ClaudeTurnActivity(cli_path=cli_path, **kw)


async def _until_cli(record: Path) -> None:
    """Wait until the fake CLI has emitted the prompt and is inside the turn."""
    ready = Path(str(record) + ".ready")
    for _ in range(500):
        if ready.exists():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"the CLI did not reach the turn; record {record}")


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_turn_returns_the_result_and_the_cli_is_gone(
    tmp_path: Path,
) -> None:
    cli, record = _cli(tmp_path, turn=fx.TOOL_ROUND_TRIP)

    result = await ActivityEnvironment().run(_activity(cli).run_turn, _input())

    assert result.text == "Let me check.It's sunny."
    assert result.session_id == fx.SESSION
    assert result.usage == TokenUsage(
        input_tokens=55,
        output_tokens=27,
        thought_tokens=4,
        cached_tokens=5,
        total_tokens=82,
    )
    assert not _process_alive(json.loads(record.read_text())["pid"])


async def test_cli_runs_with_every_capability_disabled(tmp_path: Path) -> None:
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    options = ClaudeTurnOptions(
        model="claude-test", system_prompt="Be brief.", max_turns=3
    )

    await ActivityEnvironment().run(_activity(cli).run_turn, _input(options=options))

    argv = json.loads(record.read_text())["argv"]
    flag = lambda name: argv[argv.index(name) + 1]
    assert flag("--tools") == ""
    assert flag("--permission-mode") == "dontAsk"
    assert "--setting-sources=" in argv
    assert "--strict-mcp-config" in argv
    assert "--include-partial-messages" in argv
    assert "--allowedTools" not in argv and "--mcp-config" not in argv
    assert (flag("--model"), flag("--system-prompt"), flag("--max-turns")) == (
        "claude-test",
        "Be brief.",
        "3",
    )


def test_nonessential_cli_traffic_is_off_unless_the_worker_env_sets_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = lambda: ClaudeTurnActivity()._options(_input(), None, None)
    monkeypatch.delenv(NONESSENTIAL_TRAFFIC_ENV, raising=False)
    assert options().env == {NONESSENTIAL_TRAFFIC_ENV: "1", **_CLI_ENV}
    monkeypatch.setenv(NONESSENTIAL_TRAFFIC_ENV, "")  # empty: the CLI's traffic back on
    assert options().env == _CLI_ENV, (
        "the worker's own value must reach the CLI unchanged"
    )


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("ok", 0),
        ("out\n<sandbox_violations>\nnet a.com\nnet b.com\n\n</sandbox_violations>", 2),
        (
            "<sandbox_violations>\nx\n</sandbox_violations>\n"
            "<sandbox_violations>\ny\nz\n</sandbox_violations>",
            3,
        ),
        ("<sandbox_violations>\nunclosed\n", 0),
        ("<sandbox_violations>\n</sandbox_violations>", 0),
    ],
)
def test_sandbox_violations_counts_entries(text: str, count: int) -> None:
    assert sandbox_violations(text) == count


def test_the_cli_never_inherits_claude_codes_own_restart_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker env that turns on Claude Code's interrupted-turn re-run or its resume prompt
    does not reach the CLI the SDK starts with that env, and background tasks stay off."""
    monkeypatch.setenv("CLAUDE_CODE_RESUME_INTERRUPTED_TURN", "1")
    monkeypatch.setenv("CLAUDE_CODE_RESUME_PROMPT", "Continue from where you left off.")
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", "0")
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "0")

    env = {**os.environ, **ClaudeTurnActivity()._options(_input(), None, None).env}

    assert env["CLAUDE_CODE_RESUME_INTERRUPTED_TURN"] == "0"
    assert env["CLAUDE_CODE_RESUME_PROMPT"] == ""
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    # A managed CLAUDE.md loads despite setting_sources=[]; auto memory and claude.ai
    # connectors live outside the turn.
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"


_FORWARDED_OPTIONS = frozenset(
    {
        "allowed_tools",
        "cli_path",
        "cwd",
        "effort",
        "env",
        "max_budget_usd",
        "max_turns",
        "mcp_servers",
        "model",
        "output_format",
        "resume",
        "session_id",
        "session_store",
        "session_store_flush",
        "stderr",
        "system_prompt",
        "thinking",
    }
)
_FIXED_OPTIONS = frozenset(
    {
        "can_use_tool",
        "hooks",
        "include_partial_messages",
        "max_buffer_size",
        "permission_mode",
        "plugins",
        "sandbox",
        "setting_sources",
        "settings",
        "skills",
        "strict_mcp_config",
        "tools",
        "verbatim_prompts",
    }
)
# Left at the SDK default, pinned: an SDK that changes one changes what every turn runs with.
_IGNORED_OPTIONS: dict[str, Any] = {
    "add_dirs": [],
    "agents": None,
    "betas": [],
    "continue_conversation": False,
    "debug_stderr": sys.stderr,
    "disallowed_tools": [],
    "enable_file_checkpointing": False,
    "extra_args": {},
    "fallback_model": None,
    "fork_session": False,
    "forward_subagent_text": False,
    "include_hook_events": False,
    "load_timeout_ms": 60000,
    "max_thinking_tokens": None,
    "permission_prompt_tool_name": None,
    "resume_drops_turn": None,
    "resume_session_at": None,
    "task_budget": None,
    "user": None,
}


def test_every_sdk_option_is_classified() -> None:
    """A field the SDK adds or drops fails here until it is classified as forwarded from the
    turn, fixed by the Activity, or deliberately left at the SDK default, pinned here."""
    fields = {f.name for f in dataclasses.fields(claude_agent_sdk.ClaudeAgentOptions)}
    assert fields == _FORWARDED_OPTIONS | _FIXED_OPTIONS | _IGNORED_OPTIONS.keys()
    built = ClaudeTurnActivity()._options(_input(), None, None)
    default = claude_agent_sdk.ClaudeAgentOptions()
    assert {
        name: getattr(default, name) for name in _IGNORED_OPTIONS
    } == _IGNORED_OPTIONS
    assert {name: getattr(built, name) for name in _IGNORED_OPTIONS} == _IGNORED_OPTIONS


async def test_a_message_over_the_sdks_one_mib_default_reaches_the_turn(
    tmp_path: Path,
) -> None:
    text = "x" * (2 * 1024 * 1024)
    cli, _ = _cli(tmp_path, turn=[*fx.START, fx.result(text=text)])

    result = await ActivityEnvironment().run(_activity(cli).run_turn, _input())

    assert result.text == text


async def test_error_result_then_nonzero_exit_is_one_non_retryable_failure(
    tmp_path: Path,
) -> None:
    cli, _ = _cli(tmp_path, turn=fx.API_ERROR, exit_code=1)

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(_activity(cli).run_turn, _input())

    err = info.value
    assert err.type == CLAUDE_RESULT_ERROR_TYPE and err.non_retryable
    assert err.details[0]["terminal_reason"] == "api_error"
    assert err.details[0]["api_error_status"] == 529


@pytest.mark.parametrize(
    "kind",
    [
        "authentication_failed",
        "billing_error",
        "rate_limit",
        "invalid_request",
        "server_error",
        "unknown",
    ],
)
async def test_the_sdk_error_kind_is_in_the_failure(tmp_path: Path, kind: str) -> None:
    turn = [
        *fx.START,
        fx.assistant(None, [{"type": "text", "text": "API Error"}], error=kind),
        fx.result(text="API Error", is_error=True, terminal_reason="api_error"),
    ]
    cli, _ = _cli(tmp_path, turn=turn, exit_code=1)

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(_activity(cli).run_turn, _input())

    assert info.value.details[0]["error_kind"] == kind
    assert kind in str(info.value)


@pytest.mark.parametrize(
    ("kind", "resumable"), [("authentication_failed", False), ("server_error", True)]
)
async def test_a_cli_that_dies_after_a_fatal_error_kind_is_not_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, resumable: bool
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    turn = [
        *fx.START,
        fx.assistant(None, [{"type": "text", "text": "API Error"}], error=kind),
    ]
    cli, _ = _cli(tmp_path, turn=turn, exit_code=1)
    activity = _activity(
        cli, workspace_root=tmp_path / "ws", session_store=_TranscriptStore([])
    )

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(
            activity.run_turn, _input(session_id=fx.SESSION)
        )

    assert info.value.details[0]["error_kind"] == kind
    assert info.value.details[0][RESUMABLE_DETAIL] is resumable


async def test_a_crashed_cli_reports_a_bounded_stderr_tail(tmp_path: Path) -> None:
    lines = [f"\x1b[31mline {i}\x1b[0m" for i in range(8)]
    cli, _ = _cli(tmp_path, turn=fx.START, exit_code=1, stderr="\n".join(lines) + "\n")

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(_activity(cli).run_turn, _input())

    tail = info.value.details[0]["stderr_tail"]
    assert tail == "\n".join(f"line {i}" for i in range(3, 8))
    assert tail in str(info.value)
    assert len(_stderr_tail(["x" * 2000] * 5)) == 600


async def test_the_result_reports_model_usage_denials_and_api_time(
    tmp_path: Path,
) -> None:
    usage = {"claude-test": {"inputTokens": 3, "outputTokens": 6, "costUSD": 0.01}}
    denial = {"tool_name": "Bash", "tool_use_id": "toolu_d", "tool_input": {}}
    turn = [
        *fx.START,
        fx.result(text="hi", modelUsage=usage, permission_denials=[denial]),
    ]
    cli, _ = _cli(tmp_path, turn=turn)

    result = await ActivityEnvironment().run(_activity(cli).run_turn, _input())

    assert result.model_usage == usage
    assert result.permission_denials == [denial]


async def test_num_turns_counts_this_runs_model_rounds_not_the_clis_total(
    tmp_path: Path,
) -> None:
    # A resumed real CLI reports counts that grow with the transcript (35, 42, 49 ... or 107),
    # which would use up the message-wide max_turns after a few approved native calls.
    turn = [*fx.COMPLETED_ONLY[:-1], fx.result(text="hi", num_turns=107)]
    cli, _ = _cli(tmp_path, turn=turn)

    result = await ActivityEnvironment().run(_activity(cli).run_turn, _input())

    assert result.num_turns == 1


async def test_a_zero_total_cost_falls_back_to_the_model_usage_costs(
    tmp_path: Path,
) -> None:
    usage = {"a": {"costUSD": 0.01}, "b": {"costUSD": 0.02}}
    turn = [*fx.START, fx.result(total_cost_usd=0.0, modelUsage=usage)]
    cli, _ = _cli(tmp_path, turn=turn)

    result = await ActivityEnvironment().run(_activity(cli).run_turn, _input())

    assert result.total_cost_usd == pytest.approx(0.03)
    assert result.duration_api_ms == 1100
    bare = result.model_copy(update={"model_usage": None, "permission_denials": None})
    assert not {"model_usage", "permission_denials"} & bare.model_dump().keys()


async def test_structured_output_passes_through_untouched(tmp_path: Path) -> None:
    cli, record = _cli(tmp_path, turn=fx.STRUCTURED)
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}

    result = await ActivityEnvironment().run(
        _activity(cli).run_turn, _input(options=ClaudeTurnOptions(output_schema=schema))
    )

    assert result.structured_output == {"city": "Paris", "temp_f": 72}
    argv = json.loads(record.read_text())["argv"]
    assert json.loads(argv[argv.index("--json-schema") + 1]) == schema


async def test_discriminators_and_tuples_reach_the_cli_in_a_schema_it_accepts(
    tmp_path: Path,
) -> None:
    """Claude Code 2.1.294 rejects ``discriminator`` and ``prefixItems`` as unknown keywords,
    and the API rejects the draft-07 array form of ``items``."""
    cli, record = _cli(tmp_path, turn=fx.STRUCTURED)
    pet = {
        "oneOf": [{"$ref": "#/$defs/Cat"}],
        "discriminator": {"propertyName": "kind"},
    }
    named = {"type": "string", "enum": [{"discriminator": 1}]}
    defs = {"Cat": {"type": "object", "properties": {"kind": {"const": "cat"}}}}
    schema: dict[str, Any] = {
        "$defs": defs,
        "type": "object",
        "properties": {
            "pet": pet,
            "point": {
                "type": "array",
                "prefixItems": [{"type": "integer"}],
                "minItems": 1,
            },
            "discriminator": named,
        },
    }

    await ActivityEnvironment().run(
        _activity(cli).run_turn, _input(options=ClaudeTurnOptions(output_schema=schema))
    )

    argv = json.loads(record.read_text())["argv"]
    sent = json.loads(argv[argv.index("--json-schema") + 1])
    assert sent["properties"] == {
        "pet": {"oneOf": [{"$ref": "#/$defs/Cat"}]},
        "point": {
            "type": "array",
            "items": {"anyOf": [{"type": "integer"}]},
            "minItems": 1,
        },
        "discriminator": named,
    }
    assert sent["$defs"] == defs


@pytest.mark.parametrize(
    "overrides",
    [
        {"tools": [ClaudeToolDeclaration(name="get_weather")]},
        {"bridge_token": "t"},
        {
            "tools": [ClaudeToolDeclaration(name="a"), ClaudeToolDeclaration(name="a")],
            "bridge_token": "t",
        },
        {"tools": [ClaudeToolDeclaration(name="bad name")], "bridge_token": "t"},
        {"native_auto_allow": ["Bash"]},
        {"session_id": "../../etc"},
        {"session_id": fx.SESSION.replace("-", "")},
        {"resume": True},
    ],
)
async def test_malformed_turn_inputs_are_rejected_before_the_cli_starts(
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(_activity(cli).run_turn, _input(**overrides))

    assert (
        info.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE and info.value.non_retryable
    )
    assert not record.exists(), "no CLI process was started"


async def test_cancellation_interrupts_drains_and_tears_down(tmp_path: Path) -> None:
    cli, record = _cli(
        tmp_path,
        turn=fx.STREAMING_UNTIL_INTERRUPT,
        hang_after_prompt=True,
        interrupt_result=fx.INTERRUPTED_RESULT,
    )
    env = ActivityEnvironment()
    task = asyncio.create_task(env.run(_activity(cli).run_turn, _input()))
    await _until_cli(record)

    env.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not _process_alive(json.loads(record.read_text())["pid"])


async def test_cancelled_turn_with_a_stubborn_cli_is_still_killed(
    tmp_path: Path,
) -> None:
    cli, record = _cli(
        tmp_path,
        turn=fx.STREAMING_UNTIL_INTERRUPT,
        hang_after_prompt=True,
        hang_on_eof=True,
        ignore_sigterm=True,
    )
    env = ActivityEnvironment()
    activity = _activity(
        cli,
        drain_timeout=timedelta(milliseconds=300),
        disconnect_timeout=timedelta(seconds=1),
    )
    task = asyncio.create_task(env.run(activity.run_turn, _input()))
    await _until_cli(record)

    started = time.monotonic()
    env.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert time.monotonic() - started < 5
    assert not _process_alive(json.loads(record.read_text())["pid"])


async def test_cli_that_ignores_eof_and_sigterm_is_killed_within_the_bound(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cli, record = _cli(
        tmp_path, turn=fx.COMPLETED_ONLY, hang_on_eof=True, ignore_sigterm=True
    )
    result = await ActivityEnvironment().run(
        _activity(cli, disconnect_timeout=timedelta(seconds=1)).run_turn, _input()
    )

    assert result.text == "Hello from the completed message."
    # The SDK alone would wait 5s for EOF, then 5s after SIGTERM, before SIGKILL. The kill by
    # the 1s bound is the deterministic proof; wall time is too load-sensitive to assert on.
    assert "disconnect exceeded 1.0s; killing the CLI process group" in caplog.text
    assert not _process_alive(json.loads(record.read_text())["pid"])


def _record(store: FileLeaseStore, session_id: str) -> LeaseRecord:
    record = store.latest(session_id)
    assert record is not None
    return record


async def _gone(pid: int) -> bool:
    """Whether ``pid`` exits within a few seconds (an orphan is reaped by init, not by us)."""
    for _ in range(100):
        if not _process_alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.parametrize("cancelled", [True, False], ids=["cancelled", "completed"])
async def test_what_the_cli_started_dies_with_the_turn(
    tmp_path: Path, cancelled: bool
) -> None:
    # The child stands in for a Bash command or hook: the SDK only ever stops the CLI itself.
    scenario = (
        {
            "turn": fx.STREAMING_UNTIL_INTERRUPT,
            "hang_after_prompt": True,
            "interrupt_result": fx.INTERRUPTED_RESULT,
        }
        if cancelled
        else {"turn": fx.COMPLETED_ONLY}
    )
    cli, record = _cli(tmp_path, child=["sleep", "60"], **scenario)
    env = ActivityEnvironment()
    task = asyncio.create_task(env.run(_activity(cli).run_turn, _input()))
    if cancelled:
        await _until_cli(record)
        env.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task

    pids = json.loads(record.read_text())
    assert not _process_alive(pids["pid"])
    assert await _gone(pids["child_pid"]), "the CLI's child outlived the turn"


async def test_a_process_a_command_detached_dies_with_the_turn(tmp_path: Path) -> None:
    """``nohup x &`` from a Bash command: init adopts ``x`` once the shell exits, so it is no
    longer the CLI's descendant, and it is in the command's session, not the CLI's group.
    ``x`` is not a macOS system binary, whose environment ``ps -E`` cannot read."""
    orphan = tmp_path / "orphan.pid"
    sleeper = shlex.join([sys.executable, "-c", "import time; time.sleep(60)"])
    cli, _ = _cli(
        tmp_path,
        turn=fx.COMPLETED_ONLY,
        child=["sh", "-c", f"{sleeper} >/dev/null 2>&1 & echo $! > {orphan}"],
        detach_child=True,
    )

    await ActivityEnvironment().run(_activity(cli).run_turn, _input())

    pid = int(orphan.read_text())
    try:
        assert await _gone(pid), "the detached process outlived the turn"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


async def test_killing_the_cli_kills_what_it_started_in_a_session_of_its_own(
    tmp_path: Path,
) -> None:
    # Claude Code starts each Bash command in its own session, outside the CLI's group, and only
    # ends them itself when it exits normally; this CLI has to be killed.
    cli, record = _cli(
        tmp_path,
        turn=fx.STREAMING_UNTIL_INTERRUPT,
        hang_after_prompt=True,
        hang_on_eof=True,
        ignore_sigterm=True,
        child=["sleep", "60"],
        detach_child=True,
    )
    env = ActivityEnvironment()
    activity = _activity(
        cli,
        drain_timeout=timedelta(milliseconds=300),
        disconnect_timeout=timedelta(seconds=1),
    )
    task = asyncio.create_task(env.run(activity.run_turn, _input()))
    await _until_cli(record)
    env.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    pids = json.loads(record.read_text())
    assert not _process_alive(pids["pid"])
    assert await _gone(pids["child_pid"]), "the detached child outlived the killed CLI"


class _NoProcess:
    """A transport whose CLI process handle is not where 0.2.165 keeps it (``_process``)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name == "_process":
            raise AttributeError(name)
        return getattr(self._inner, name)


@pytest.fixture
def sdk_process_handle_moved(monkeypatch: pytest.MonkeyPatch) -> None:
    """As if an SDK upgrade renamed the private process handle the Activity could read."""
    connect = claude_agent_sdk.ClaudeSDKClient.connect

    async def connect_and_hide(self: Any, *args: Any, **kwargs: Any) -> None:
        await connect(self, *args, **kwargs)
        self._transport = _NoProcess(self._transport)
        assert _activity_module._cli_process(self) is None

    monkeypatch.setattr(claude_agent_sdk.ClaudeSDKClient, "connect", connect_and_hide)


@pytest.mark.parametrize(
    "cancelled", [True, False], ids=["cancelled", "disconnect-overrun"]
)
async def test_the_cli_is_killed_without_the_sdks_process_handle(
    tmp_path: Path,
    sdk_process_handle_moved: None,
    cancelled: bool,
) -> None:
    assert sdk_process_handle_moved is None
    cli, record = _cli(
        tmp_path,
        turn=fx.STREAMING_UNTIL_INTERRUPT if cancelled else fx.COMPLETED_ONLY,
        hang_after_prompt=cancelled,
        hang_on_eof=True,
        ignore_sigterm=True,
        child=["sleep", "60"],
        detach_child=True,
    )
    activity = _activity(
        cli,
        drain_timeout=timedelta(milliseconds=300),
        disconnect_timeout=timedelta(seconds=1),
    )
    env = ActivityEnvironment()
    task = asyncio.create_task(env.run(activity.run_turn, _input()))
    if cancelled:
        await _until_cli(record)
        env.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert (await task).text == "Hello from the completed message."

    pids = json.loads(record.read_text())
    assert await _gone(pids["pid"]), "the CLI outlived the turn"
    assert await _gone(pids["child_pid"]), "the CLI's detached child outlived the turn"


async def test_a_cli_without_a_recorded_process_group_fails_the_turn_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The launcher reads the real variable name; the Activity now sets another, so no record.
    monkeypatch.setattr(_launcher, "PGID_FILE_ENV", "NOT_THE_LAUNCHERS_VARIABLE")
    cli, record = _cli(
        tmp_path, turn=fx.COMPLETED_ONLY, log=str(tmp_path / "log.jsonl")
    )

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(_activity(cli).run_turn, _input())

    assert (
        info.value.type == CLAUDE_CLI_UNTRACKED_ERROR_TYPE and info.value.non_retryable
    )
    assert "recorded no process group" in str(info.value)
    assert "Claude CLI not killable" in caplog.text
    assert not (tmp_path / "log.jsonl").exists(), "the prompt was never sent"
    assert await _gone(json.loads(record.read_text())["pid"])


def _leftover_group(
    pgid_file: Path, *, ours: bool
) -> tuple[subprocess.Popen, int | None]:
    """A process leading its own group, its pid recorded in ``pgid_file``: a CLI (with the
    launcher's environment, and a Bash-like child in a session of its own) that a crashed attempt
    left, or an unrelated process that reused the recorded pid."""
    pgid_file.parent.mkdir(parents=True, exist_ok=True)
    if not ours:
        leader = subprocess.Popen(["sleep", "60"], start_new_session=True)
        pgid_file.write_text(str(leader.pid))
        return leader, None
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, time; "
            "print(subprocess.Popen(['sleep', '60'], start_new_session=True).pid, flush=True); "
            "time.sleep(60)",
        ],
        start_new_session=True,
        stdout=subprocess.PIPE,
        env={**os.environ, PGID_FILE_ENV: str(pgid_file)},
    )
    stdout = leader.stdout
    assert stdout is not None
    child = int(stdout.readline())
    pgid_file.write_text(str(leader.pid))
    return leader, child


@pytest.mark.parametrize(
    ("ours", "generation"),
    [(True, 0), (False, 0), (True, 2)],
    ids=["crashed-attempt", "reused-pid", "newer-generation"],
)
async def test_a_session_turn_first_kills_the_group_a_crashed_attempt_left(
    tmp_path: Path,
    ours: bool,
    generation: int,
) -> None:
    """The turn holds generation 1. A CLI an earlier generation left is killed; a process that
    reused its pid is not, nor a later generation's CLI (that attempt holds the session now)."""
    cli, _ = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    root = tmp_path / "ws"
    pgid_file = root / _PGID_DIR_NAME / f"{fx.SESSION}.{generation}"
    leader, child = _leftover_group(pgid_file, ours=ours)
    try:
        await ActivityEnvironment().run(
            _activity(cli, workspace_root=root).run_turn,
            _input(session_id=fx.SESSION),
        )

        if ours and generation < 1:
            assert leader.wait(timeout=5) == -9
            assert child is not None
            assert await _gone(child)
        else:
            assert leader.poll() is None, "a process that is not a leftover was killed"
        assert pgid_file.exists() == (generation > 1), "only an earlier record is swept"
        assert not (root / _PGID_DIR_NAME / f"{fx.SESSION}.1").exists(), (
            "the turn's own record is removed when it ends"
        )
    finally:
        leader.kill()
        leader.wait()


@pytest.fixture
def verified_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native tools verified on the installed SDK with CLI 2.1.294 (what the fake reports unless
    told otherwise), on a host with a sandbox backend (the fake never uses it)."""
    monkeypatch.setattr(
        _native,
        "NATIVE_TOOLS_VERIFIED",
        frozenset({(claude_agent_sdk.__version__, "2.1.294")}),
    )
    monkeypatch.setattr(_native, "sandbox_backend_available", lambda: True)
    monkeypatch.setattr(_native, "seccomp_requirement", lambda: None)


def _bundled(monkeypatch: pytest.MonkeyPatch, path: str | None) -> None:
    """Make the SDK's own bundled-CLI lookup find ``path`` (None: a wheel without one)."""
    monkeypatch.setattr(SubprocessCLITransport, "_find_bundled_cli", lambda self: path)


async def _native_turn(tmp_path: Path, cli_path: str | None) -> Any:
    activity = _activity(
        cli_path, workspace_root=tmp_path / "ws", enable_native_tools=True
    )
    turn = _input(session_id=fx.SESSION, native_tools=["Bash"])
    return await ActivityEnvironment().run(activity.run_turn, turn)


async def test_native_tools_run_on_a_verified_pair_through_the_bundled_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified_pair: None,
) -> None:
    assert verified_pair is None
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    _bundled(monkeypatch, cli)

    result = await _native_turn(tmp_path, None)

    assert result.text == "Hello from the completed message."
    assert record.exists(), "the turn ran the CLI the SDK's lookup found"


@pytest.mark.parametrize(
    "custom", [False, True], ids=["bundled-cli", "custom-cli_path"]
)
@pytest.mark.parametrize("version", ["2.1.295", "not a version"])
async def test_native_tools_are_refused_when_the_running_cli_is_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified_pair: None,
    custom: bool,
    version: str,
) -> None:
    assert verified_pair is None
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY, version=version)
    _bundled(monkeypatch, None if custom else cli)

    with pytest.raises(ApplicationError) as info:
        await _native_turn(tmp_path, cli if custom else None)

    assert (
        info.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE and info.value.non_retryable
    )
    message = str(info.value)
    assert "not verified" in message and os.path.realpath(cli) in message
    assert ("2.1.295" if version == "2.1.295" else "version unknown") in message
    assert not record.exists(), "no CLI process was started"


@pytest.mark.parametrize(
    ("resume", "stored", "version"),
    [
        (True, False, "2.1.295"),
        (False, True, "2.1.295"),
        (True, False, "not a version"),
        (False, False, "2.1.295"),
    ],
    ids=["resume", "dedup", "version-unknown", "fresh-turn"],
)
async def test_a_custom_cli_path_resumes_only_a_verified_cli(
    tmp_path: Path, resume: bool, stored: bool, version: str
) -> None:
    """Resume and dedup need a verified CLI version, failing closed; a fresh turn does not check."""
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY, version=version)
    store = _TranscriptStore([{"type": "user"}]) if stored else None
    activity = _activity(cli, workspace_root=tmp_path / "ws", session_store=store)

    if resume or stored:
        with pytest.raises(ApplicationError) as info:
            await ActivityEnvironment().run(
                activity.run_turn,
                _input(session_id=fx.SESSION, resume=resume),
            )
        message = str(info.value)
        assert info.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE
        assert info.value.non_retryable
        assert (
            "2.1.295" in message
            if version == "2.1.295"
            else "version unknown" in message
        )
        assert not record.exists(), "the turn CLI started"
        return

    result = await ActivityEnvironment().run(activity.run_turn, _input())
    assert result.text == "Hello from the completed message."
    assert record.exists()


@pytest.mark.parametrize(
    "stored", [False, True], ids=["own-transcript", "session-store"]
)
async def test_only_a_resume_of_claude_codes_own_transcript_reports_a_cumulative_cost(
    tmp_path: Path, stored: bool
) -> None:
    """Claude Code restores the session's cost from the ``cost-state`` entry it writes into its
    own transcript on exit; a SessionStore's copy has none, so that resume reports its own."""
    cli, _ = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    store = _TranscriptStore([{"type": "user", "uuid": "u"}]) if stored else None
    activity = _activity(cli, workspace_root=tmp_path / "ws", session_store=store)

    result = await ActivityEnvironment().run(
        activity.run_turn, _input(session_id=fx.SESSION, resume=True)
    )

    assert result.cumulative_cost is not stored


class _TranscriptStore:
    """SessionStore stand-in: ``load`` returns the entries a resume would dedup from."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._entries = entries

    async def append(self, _key: Any, _entries: list[Any]) -> None:
        return None

    async def load(self, _key: Any) -> list[dict[str, Any]]:
        return self._entries


async def test_a_tool_result_the_store_holds_twice_answers_one_call(
    tmp_path: Path,
) -> None:
    """A retried mirror batch can land twice in a store that does not dedupe by ``uuid`` (the
    SDK asks it to, but cannot make it), and the CLI's loader keys entries by ``uuid``."""
    call = {
        "type": "tool_use",
        "id": "toolu_w",
        "name": mcp_tool_name("get_weather"),
        "input": {"city": "Paris"},
    }
    answer = {"type": "tool_result", "tool_use_id": "toolu_w", "content": "sunny"}
    result = {"type": "user", "uuid": "u-r", "message": {"content": [answer]}}
    entries = [
        {"type": "user", "uuid": "u-p", "message": {"content": "q"}},
        {"type": "assistant", "uuid": "u-a", "message": {"content": [call]}},
        result,
        result,
    ]
    activity = _activity(
        None, workspace_root=tmp_path / "ws", session_store=_TranscriptStore(entries)
    )

    plan = await activity._plan(_input(session_id=fx.SESSION, prompt_uuid="u-p"))

    assert plan.prior == [
        ("toolu_w", tool_call_key("get_weather", {"city": "Paris"}), True)
    ]


class _SignalRecorder:
    """The workflow handle a ``ToolBridgeClient`` signals; it records each request."""

    def __init__(self) -> None:
        self.requests: list[ClaudeToolRequest] = []

    async def signal(self, _name: str, request: ClaudeToolRequest) -> None:
        self.requests.append(request)


@pytest.mark.parametrize(
    ("accepted", "poll_failed"),
    [(True, False), (False, False), (True, True)],
    ids=["accepted-ack-lost", "undelivered", "poll-failed"],
)
async def test_ambiguous_delivery_does_not_reach_claude_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted: bool, poll_failed: bool
) -> None:
    """Run the real Activity, SDK, MCP handler and fake CLI through a lost acknowledgement."""
    effects: list[str] = []
    ready = asyncio.Event()

    async def get_weather(city: str) -> str:
        effects.append(city)
        return f"authoritative: sunny in {city}"

    backend = _ToolBridge(
        "token", {"get_weather": get_weather}, 4, ClaudeSession("sess"), "msg"
    )

    class Handle:
        async def signal(self, _name: str, request: ClaudeToolRequest) -> None:
            if accepted:
                backend.submit(request)
                await asyncio.gather(*tuple(backend._tasks))
                ready.set()
            if poll_failed:
                return
            raise OSError("signal acknowledgement unavailable")

        async def execute_update(
            self, _name: str, poll: Any, **_kwargs: Any
        ) -> ClaudeBridgeBatch:
            await ready.wait()
            if poll_failed:
                raise OSError("result delivery unavailable")
            return ClaudeBridgeBatch(results=backend.results[poll.after :])

    handle = Handle()
    monkeypatch.setattr(
        temporal_activity,
        "client",
        lambda: SimpleNamespace(get_workflow_handle=lambda _id: handle),
    )
    cli, _ = _cli(
        tmp_path,
        turn=fx.tool_turn([("toolu_ack", "get_weather", {"city": "Paris"})]),
        log=str(tmp_path / "cli.jsonl"),
    )
    failed: ApplicationError | None = None
    try:
        await asyncio.wait_for(
            ActivityEnvironment().run(
                _activity(cli, workspace_root=tmp_path / "ws").run_turn,
                _input(
                    bridge_token="token",
                    tools=[
                        ClaudeToolDeclaration(
                            name="get_weather",
                            description="Weather",
                            input_schema={
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        )
                    ],
                ),
            ),
            10,
        )
    except ApplicationError as exc:
        failed = exc
    finally:
        backend.close()
        await asyncio.gather(*tuple(backend._tasks), return_exceptions=True)

    assert effects == (["Paris"] if accepted else [])
    calls = [
        call
        for line in (
            (tmp_path / "cli.jsonl").read_text().splitlines()
            if (tmp_path / "cli.jsonl").exists()
            else []
        )
        for call in json.loads(line).get("mcp", [])
    ]
    if accepted and not poll_failed:
        assert backend._session.state.tool_record[0].result is not None
        assert calls and calls[0]["result"]["isError"] is False, calls
        assert "authoritative: sunny in Paris" in json.dumps(calls)
    else:
        assert failed is not None, (
            f"Activity completed after fabricating a tool failure: {calls}"
        )
        assert failed.non_retryable
        assert "outcome unknown" in str(failed)
        assert not calls, (
            "Claude received a fabricated result before supervision stopped it"
        )


async def test_a_reissued_parallel_call_takes_the_slot_whose_result_was_lost(
    tmp_path: Path,
) -> None:
    """Two identical parallel calls; the store holds only the second one's result. The resumed
    CLI re-issues the first, which must reach the first call's slot."""
    paris = {"city": "Paris"}

    def call(call_id: str) -> dict[str, Any]:
        block = {
            "type": "tool_use",
            "id": call_id,
            "name": mcp_tool_name("get_weather"),
            "input": paris,
        }
        return {
            "type": "assistant",
            "uuid": f"a-{call_id}",
            "message": {"content": [block]},
        }

    answer = {"type": "tool_result", "tool_use_id": "toolu_2", "content": "sunny"}
    entries = [
        {"type": "user", "uuid": "u-p", "message": {"content": "q"}},
        call("toolu_1"),
        call("toolu_2"),
        {"type": "user", "uuid": "u-r", "message": {"content": [answer]}},
    ]
    activity = _activity(
        None, workspace_root=tmp_path / "ws", session_store=_TranscriptStore(entries)
    )
    plan = await activity._plan(_input(session_id=fx.SESSION, prompt_uuid="u-p"))
    handle = _SignalRecorder()
    bridge = ToolBridgeClient(handle, "token")  # type: ignore[arg-type]
    bridge.number({}, plan.prior)

    calls = [
        asyncio.create_task(bridge.call(f"toolu_re{i}", "get_weather", paris))
        for i in range(2)
    ]
    while len(handle.requests) < 2:
        await asyncio.sleep(0)
    bridge.close("done")
    await asyncio.gather(*calls)

    assert [r.ordinal for r in handle.requests] == [0, 2]


async def test_a_run_that_fails_before_its_state_is_prepared_counts_no_old_rerequests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verified_pair: None
) -> None:
    """The previous run's ledger is deleted only when this run's state is written; a failure
    before that must not report its re-requests again."""
    assert verified_pair is None
    root = tmp_path / "ws"
    state = root / NATIVE_STATE_DIR_NAME / fx.SESSION
    state.mkdir(parents=True)
    (state / "ledger.jsonl").write_text(json.dumps({"rerequest": 1}) + "\n")

    async def unsettled(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the transcript could not be settled")

    monkeypatch.setattr(ClaudeTurnActivity, "_settle_transcript", unsettled)
    cli, _record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(
            _activity(cli, workspace_root=root, enable_native_tools=True).run_turn,
            _input(session_id=fx.SESSION, resume=True, native_tools=["Bash"]),
        )

    assert failed.value.details[0]["native_rerequests"] == 0


def _weather_call(call_id: str) -> dict[str, Any]:
    block = {
        "type": "tool_use",
        "id": call_id,
        "name": mcp_tool_name("get_weather"),
        "input": {"city": "Paris"},
    }
    return {
        "type": "assistant",
        "uuid": f"a-{call_id}",
        "message": {"content": [block]},
    }


def _weather_answer(call_id: str, text: str) -> dict[str, Any]:
    answer = {"type": "tool_result", "tool_use_id": call_id, "content": text}
    return {"type": "user", "uuid": f"r-{call_id}", "message": {"content": [answer]}}


async def _third_attempt_ordinal(
    tmp_path: Path,
    entries: list[dict[str, Any]],
    records: list[ClaudeToolRecord],
    **kw: Any,
) -> int:
    """The ordinal a third attempt gives a new Paris weather call, after numbering from the
    store's transcript and the workflow's call record."""
    input = _input(
        session_id=fx.SESSION, tool_calls=recorded_tool_calls(records) or None, **kw
    )
    activity = _activity(
        None, workspace_root=tmp_path / "ws", session_store=_TranscriptStore(entries)
    )
    plan = await activity._plan(input)
    handle = _SignalRecorder()
    bridge = ToolBridgeClient(handle, "token")  # type: ignore[arg-type]
    bridge.number(
        input.tool_ordinals or {}, plan.prior, input.tool_calls, plan.answered
    )
    call = asyncio.create_task(
        bridge.call("toolu_new", "get_weather", {"city": "Paris"})
    )
    while not handle.requests:
        await asyncio.sleep(0)
    bridge.close("done")
    await call
    return handle.requests[0].ordinal


def _paris_record(
    call_id: str, ordinal: int, aliases: list[str] | None = None
) -> ClaudeToolRecord:
    return ClaudeToolRecord(
        session_id=fx.SESSION,
        message_id="msg",
        tool="get_weather",
        input_key=tool_call_key("get_weather", {"city": "Paris"}),
        ordinal=ordinal,
        tool_use_id=call_id,
        aliases=aliases or [],
    )


async def test_a_third_attempt_numbers_a_new_call_after_an_earlier_reissue(
    tmp_path: Path,
) -> None:
    """Attempt 1's call was patched as interrupted and attempt 2 re-issued it under a new id,
    answered from slot 0; attempt 2 then made a second identical call (slot 1) and was lost
    before storing it. Attempt 3's re-issue of that second call must not take slot 0."""
    entries = [
        {"type": "user", "uuid": "u-p", "message": {"content": "q"}},
        _weather_call("toolu_1"),
        _weather_answer("toolu_1", "[Tool call interrupted: outcome unknown]"),
        _weather_call("toolu_2"),
        _weather_answer("toolu_2", "sunny"),
    ]
    records = [_paris_record("toolu_1", 0, ["toolu_2"]), _paris_record("toolu_3", 1)]

    assert (
        await _third_attempt_ordinal(tmp_path, entries, records, prompt_uuid="u-p") == 1
    )


async def test_a_third_decision_attempt_numbers_after_a_call_before_its_prompt(
    tmp_path: Path,
) -> None:
    """An approved-decision run's first attempt made a call and was lost; attempt 2 anchored on
    the deferred call and stored the resume prompt after that call. Attempt 3 anchors on the
    prompt, so the first call is outside its window, but a new identical call is still new."""
    entries = [
        {"type": "user", "uuid": "u-p", "message": {"content": "q"}},
        _weather_call("toolu_1"),
        _weather_answer("toolu_1", "sunny"),
        {"type": "user", "uuid": "u-resume", "message": {"content": "Continue."}},
    ]
    records = [_paris_record("toolu_1", 0)]

    assert (
        await _third_attempt_ordinal(tmp_path, entries, records, prompt_uuid="u-resume")
        == 1
    )


async def test_a_recorded_call_missing_from_the_store_keeps_its_slot(
    tmp_path: Path,
) -> None:
    """The lost attempt's call reached the workflow but not the store: its re-issue is that
    call, answered from the record, not run again."""
    entries = [{"type": "user", "uuid": "u-p", "message": {"content": "q"}}]
    records = [_paris_record("toolu_1", 0)]

    assert (
        await _third_attempt_ordinal(tmp_path, entries, records, prompt_uuid="u-p") == 0
    )


async def test_a_completed_session_with_an_empty_store_is_rejected_before_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``resume`` with an empty SessionStore is rejected before the CLI; the local file stays."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    # A verified custom CLI would be allowed to resume; the missing transcript must still stop it.
    monkeypatch.setattr(
        _activity_module,
        "NATIVE_TOOLS_VERIFIED",
        frozenset({(claude_agent_sdk.__version__, "2.1.294")}),
    )
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    cwd = tmp_path / "ws" / fx.SESSION
    cwd.mkdir(parents=True)
    local = _activity_module._local_transcript(cwd, fx.SESSION)
    local.parent.mkdir(parents=True)
    local.write_text('{"type": "user"}\n')
    activity = _activity(
        cli, workspace_root=tmp_path / "ws", session_store=_TranscriptStore([])
    )

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(
            activity.run_turn, _input(session_id=fx.SESSION, resume=True)
        )

    assert (
        info.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE and info.value.non_retryable
    )
    assert "no transcript" in str(info.value)
    assert not record.exists(), (
        "the CLI was started for a resume with no shared transcript"
    )
    assert local.exists(), "the local file is not a fresh start to be deleted"


def _allow_verified_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _activity_module,
        "NATIVE_TOOLS_VERIFIED",
        frozenset({(claude_agent_sdk.__version__, "2.1.294")}),
    )


_BASH = ClaudeDeferredTool(
    tool_use_id="toolu_x", name="Bash", input={"command": "echo hi"}
)


async def test_a_continue_prompt_without_a_native_decision_is_still_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resume prompt is not a continuation. Only the typed decision is."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    _allow_verified_resume(monkeypatch)
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    activity = _activity(
        cli, workspace_root=tmp_path / "ws", session_store=_TranscriptStore([])
    )

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(
            activity.run_turn,
            ClaudeTurnInput(prompt="Continue.", session_id=fx.SESSION, resume=True),
        )

    assert info.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE
    assert "no transcript" in str(info.value)
    assert not record.exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"resumes_lost_attempt": True},
        {"resumes_api_error": True},
        {
            "native_calls": [
                ClaudeNativeCall(
                    tool_use_id="toolu_x",
                    tool_name="Bash",
                    key="bash",
                    status="started",
                )
            ]
        },
        {"native_tools": ["Read"]},
        {
            "native_decision": ClaudeNativeDecision(
                call=_BASH, approved=False, reason="no"
            )
        },
    ],
    ids=[
        "lost-attempt",
        "api-error",
        "ledger-already-records-the-call",
        "tool-not-offered",
        "denial-not-recorded",
    ],
)
async def test_a_native_decision_without_pending_state_still_needs_the_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified_pair: None,
    overrides: dict[str, Any],
) -> None:
    """A decision is not enough. Recovery, a recorded call, or a denial the session did not
    keep still fail closed when the SessionStore is empty.
    """
    assert verified_pair is None
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    _allow_verified_resume(monkeypatch)
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    activity = _activity(
        cli,
        workspace_root=tmp_path / "ws",
        session_store=_TranscriptStore([]),
        enable_native_tools=True,
        lease_store=FileLeaseStore(tmp_path / "leases"),
    )
    turn: dict[str, Any] = {
        "session_id": fx.SESSION,
        "resume": True,
        "prompt": "Continue.",
        "native_tools": ["Bash"],
        "native_decision": ClaudeNativeDecision(call=_BASH, approved=True),
    }
    turn.update(overrides)

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(activity.run_turn, ClaudeTurnInput(**turn))

    assert info.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE
    assert "no transcript" in str(info.value)
    assert not record.exists(), "the CLI started without a shared transcript"


@pytest.mark.parametrize(
    "turn",
    [
        {
            "prompt": "Continue.",
            "native_tools": ["Bash"],
            "native_decision": ClaudeNativeDecision(call=_BASH, approved=True),
        },
        {
            "prompt": "Answer the user.",
            "native_tools": ["Bash"],
            "native_denied": [_BASH],
            "native_decision": ClaudeNativeDecision(
                call=_BASH, approved=False, reason="Denied by policy."
            ),
        },
    ],
    ids=["approved", "denied"],
)
async def test_a_pending_native_continuation_resumes_with_an_empty_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified_pair: None,
    turn: dict[str, Any],
) -> None:
    """A deferred call's decision run resumes before its transcript is mirrored."""
    assert verified_pair is None
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    _allow_verified_resume(monkeypatch)
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    cwd = tmp_path / "ws" / fx.SESSION
    cwd.mkdir(parents=True)
    local = _activity_module._local_transcript(cwd, fx.SESSION)
    local.parent.mkdir(parents=True)
    local.write_text('{"type": "user"}\n')
    activity = _activity(
        cli,
        workspace_root=tmp_path / "ws",
        session_store=_TranscriptStore([]),
        enable_native_tools=True,
    )

    result = await ActivityEnvironment().run(
        activity.run_turn,
        ClaudeTurnInput(session_id=fx.SESSION, resume=True, **turn),
    )

    assert result.text == "Hello from the completed message."
    argv = json.loads(record.read_text())["argv"]
    assert any(arg == "--resume" or arg.startswith("--resume=") for arg in argv)
    assert local.exists(), "the continuation is not a fresh start"


async def test_native_tools_are_refused_without_a_sandbox_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified_pair: None,
) -> None:
    """A verified SDK/CLI pair still does not start a CLI when the host has no sandbox."""
    assert verified_pair is None
    monkeypatch.setattr(_native, "sandbox_backend_available", lambda: False)
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)

    with pytest.raises(ApplicationError) as info:
        await _native_turn(tmp_path, cli)

    assert (
        info.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE and info.value.non_retryable
    )
    assert "sandbox" in str(info.value)
    assert not record.exists(), "no CLI process was started"


async def test_orchestration_tools_are_rejected_before_the_cli_starts(
    tmp_path: Path,
    verified_pair: None,
) -> None:
    """Agent, Skill, and Workflow cannot be made durable, so the turn is refused up front."""
    assert verified_pair is None
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(
            _activity(
                cli, workspace_root=tmp_path / "ws", enable_native_tools=True
            ).run_turn,
            _input(
                session_id=fx.SESSION,
                native_tools=["Bash", "Agent", "Skill", "Workflow"],
            ),
        )

    assert (
        info.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE and info.value.non_retryable
    )
    message = str(info.value)
    assert "cannot be made durable" in message
    assert "Agent" in message and "Skill" in message and "Workflow" in message
    assert not record.exists(), "no CLI process was started"


@pytest.mark.parametrize(
    ("status", "reason"),
    [("finished", ALREADY_RAN), ("started", UNKNOWN_OUTCOME)],
)
async def test_a_carried_native_call_is_not_run_when_the_cli_presents_it_again(
    tmp_path: Path,
    verified_pair: None,
    status: str,
    reason: str,
) -> None:
    """A lost attempt's ledger travels with the resume. The same call, re-presented, is denied:
    a finished effect is not repeated, and a started one stays unknown instead of running blind."""
    assert verified_pair is None
    command = {"command": "printf hello > hello.txt"}
    effects = tmp_path / "effects.jsonl"
    log = tmp_path / "log.jsonl"
    deferred = tmp_path / "deferred.json"
    deferred.write_text(
        json.dumps(
            {
                fx.SESSION: {
                    "id": "toolu_x",
                    "name": "Bash",
                    "input": command,
                    "writes": {"hello.txt": "hello"},
                }
            }
        )
    )
    cli, _record = _cli(
        tmp_path,
        turn=fx.COMPLETED_ONLY,
        state=str(deferred),
        effects=str(effects),
        log=str(log),
    )
    call = ClaudeDeferredTool(tool_use_id="toolu_x", name="Bash", input=command)
    root = tmp_path / "ws"

    result = await ActivityEnvironment().run(
        _activity(cli, workspace_root=root, enable_native_tools=True).run_turn,
        _input(
            session_id=fx.SESSION,
            resume=True,
            native_tools=["Bash"],
            native_decision=ClaudeNativeDecision(call=call, approved=True),
            native_calls=[
                ClaudeNativeCall(
                    tool_use_id="toolu_x",
                    tool_name="Bash",
                    key=call_key("Bash", command),
                    run="the-lost-run",
                    status=status,  # type: ignore[arg-type]
                )
            ],
        ),
    )

    assert result.text == "Hello from the completed message."
    assert not effects.exists(), "the re-presented call ran"
    assert list(root.rglob("hello.txt")) == []
    [entry] = [json.loads(line) for line in log.read_text().splitlines()]
    assert entry["hooks"][0]["reasons"] == [reason]


async def test_without_a_bundled_cli_only_turns_without_native_tools_search_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified_pair: None,
) -> None:
    assert verified_pair is None
    _, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    _bundled(monkeypatch, None)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(ApplicationError) as info:
        await _native_turn(tmp_path, None)
    assert info.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE
    assert "never fall back to a `claude` on PATH" in str(info.value)
    assert not record.exists(), "a native turn ran the CLI on PATH"

    await ActivityEnvironment().run(ClaudeTurnActivity().run_turn, _input())
    assert record.exists(), "a turn without native tools uses the SDK's own PATH search"


async def test_each_session_runs_in_its_own_stable_workspace(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    cli, _ = _cli(tmp_path, turn=fx.COMPLETED_ONLY, log=str(log))
    activity = _activity(cli, workspace_root=tmp_path / "ws")
    other = "99999999-2222-4333-8444-555555555555"

    await ActivityEnvironment().run(activity.run_turn, _input(session_id=fx.SESSION))
    await ActivityEnvironment().run(
        activity.run_turn,
        _input(session_id=fx.SESSION, resume=True),
    )
    await ActivityEnvironment().run(activity.run_turn, _input(session_id=other))

    cwds = [json.loads(line)["cwd"] for line in log.read_text().splitlines()]
    root = os.path.realpath(tmp_path / "ws")
    assert cwds == [f"{root}/{fx.SESSION}", f"{root}/{fx.SESSION}", f"{root}/{other}"]


async def test_a_slow_cli_version_probe_heartbeats(tmp_path: Path) -> None:
    """A worker's first turn probes ``cli --version`` before its first heartbeat. A probe slower
    than the heartbeat timeout must not leave the attempt silent, or the server times it out
    with no details and the workflow fails the turn."""
    cli, _ = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    slow = tmp_path / "slow-version"
    slow.write_text(
        f'#!/bin/sh\ncase "$1" in --version) sleep 2;; esac\nexec {shlex.quote(cli)} "$@"\n'
    )
    slow.chmod(0o755)
    beats: list[tuple[Any, ...]] = []
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, heartbeat_timeout=timedelta(seconds=0.6))
    env.on_heartbeat = lambda *details: beats.append(details)

    await asyncio.wait_for(env.run(_activity(str(slow)).run_turn, _input()), timeout=30)

    assert beats and beats[0] == (), beats


async def test_a_turn_that_loses_its_lease_kills_its_cli_and_reports_an_unknown_outcome(
    tmp_path: Path,
) -> None:
    """Another attempt took the session over mid-run: the CLI must die now, not at turn end, and
    the turn must not look like a clean failure a retry could repeat."""
    cli, record = _cli(
        tmp_path,
        turn=fx.STREAMING_UNTIL_INTERRUPT,
        hang_after_prompt=True,
        child=["sleep", "60"],
        detach_child=True,
    )
    store = FileLeaseStore(tmp_path / "leases")
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, heartbeat_timeout=timedelta(seconds=0.6))
    activity = _activity(cli, workspace_root=tmp_path / "ws", lease_store=store)
    task = asyncio.create_task(
        env.run(activity.run_turn, _input(session_id=fx.SESSION))
    )
    await _until_cli(record)
    pids = json.loads(record.read_text())

    assert store.claim(
        fx.SESSION,
        _record(store, fx.SESSION).generation + 1,
        "another-worker",
    )
    with pytest.raises(ApplicationError) as info:
        await asyncio.wait_for(task, timeout=10)

    assert (
        info.value.type == CLAUDE_SESSION_LEASE_LOST_ERROR_TYPE
        and info.value.non_retryable
    )
    assert info.value.details[0]["native_calls"] == []
    assert "workspace_revision" in info.value.details[0]
    assert not _process_alive(pids["pid"])
    assert await _gone(pids["child_pid"]), (
        "the CLI's detached child outlived the lost lease"
    )
    assert _record(store, fx.SESSION).owner == "another-worker"


@pytest.mark.parametrize("cli_started", [False, True])
async def test_a_lost_lease_carries_the_progress_and_is_retryable_only_before_the_cli_started(
    tmp_path: Path,
    cli_started: bool,
) -> None:
    store = FileLeaseStore(tmp_path / "leases")
    progress = {"native_calls": [{"tool_use_id": "toolu_x"}], "workspace_revision": 4}
    group = _CliGroup()
    group.started = cli_started

    async def hold_the_session() -> None:
        holder: Any = SimpleNamespace(details=lambda: progress, lease=None)
        async with _leased(store, _input(session_id=fx.SESSION), group, holder):
            assert store.claim(
                fx.SESSION,
                _record(store, fx.SESSION).generation + 1,
                "other",
            )
            await asyncio.sleep(10)

    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, heartbeat_timeout=timedelta(seconds=0.6))
    with pytest.raises(ApplicationError) as info:
        await asyncio.wait_for(env.run(hold_the_session), timeout=10)

    assert info.value.type == CLAUDE_SESSION_LEASE_LOST_ERROR_TYPE
    assert info.value.non_retryable is cli_started
    assert info.value.details[0] == {"session_id": fx.SESSION, **progress}


async def test_a_stalled_attempt_that_resumes_after_a_takeover_kills_only_its_own_cli(
    tmp_path: Path,
) -> None:
    """A stalls past the lease TTL and B takes over; when A resumes it kills only its own CLI."""
    a_cli, a_record = _cli(
        tmp_path / "a", turn=fx.STREAMING_UNTIL_INTERRUPT, hang_after_prompt=True
    )
    b_cli, b_record = _cli(
        tmp_path / "b", turn=[{"fake": "sleep", "seconds": 3}, *fx.COMPLETED_ONLY]
    )
    root, store = tmp_path / "ws", FileLeaseStore(tmp_path / "leases")
    records = root / _PGID_DIR_NAME

    def env() -> ActivityEnvironment:
        env = ActivityEnvironment()
        env.info = dataclasses.replace(
            env.info, heartbeat_timeout=timedelta(seconds=0.6)
        )
        return env

    a_loop = asyncio.new_event_loop()
    a_outcome: list[BaseException | Any] = []
    frozen, thaw = threading.Event(), threading.Event()

    def run_a() -> None:
        activity = _activity(a_cli, workspace_root=root, lease_store=store)
        try:
            a_outcome.append(
                a_loop.run_until_complete(
                    env().run(activity.run_turn, _input(session_id=fx.SESSION))
                )
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            a_outcome.append(exc)

    def freeze_a() -> None:
        frozen.set()
        thaw.wait()

    a_thread = threading.Thread(target=run_a)
    a_thread.start()
    try:
        await _until(
            lambda: (
                Path(str(a_record) + ".ready").exists()
                and (records / f"{fx.SESSION}.1").exists()
            ),
            timeout=30,
        )
        a_pid = json.loads(a_record.read_text())["pid"]
        a_loop.call_soon_threadsafe(freeze_a)
        await asyncio.to_thread(frozen.wait, 5)
        os.killpg(a_pid, signal.SIGSTOP)

        b_activity = _activity(b_cli, workspace_root=root, lease_store=store)
        b_task = asyncio.create_task(
            env().run(b_activity.run_turn, _input(session_id=fx.SESSION))
        )
        await _until(
            lambda: b_record.exists() and (records / f"{fx.SESSION}.2").exists()
        )
        b_pid = json.loads(b_record.read_text())["pid"]
        assert _process_alive(b_pid)

        thaw.set()
        with contextlib.suppress(
            ProcessLookupError, PermissionError
        ):  # macOS: a zombie group
            os.killpg(a_pid, signal.SIGCONT)
        await asyncio.to_thread(a_thread.join, 10)

        assert (
            await asyncio.wait_for(b_task, 10)
        ).text == "Hello from the completed message."
    finally:
        thaw.set()
        a_thread.join(10)
        if not a_loop.is_running():
            a_loop.close()

    assert not _process_alive(a_pid), "A's CLI outlived A's lost lease"
    # B's sweep killed A's frozen CLI, so A fails on its dead CLI or its lost lease, whichever
    # it sees first; Temporal timed that attempt out before B ran.
    assert isinstance(a_outcome[0], Exception), a_outcome
    assert not list(records.iterdir()), "each attempt removed its own record"


async def _until(predicate: Any, timeout: float = 10) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


class _AnySessionStore:
    async def append(self, _key: Any, _entries: list[Any]) -> None:
        pass

    async def load(self, _key: Any) -> None:
        return None


async def test_resuming_a_lost_attempt_is_refused_without_a_shared_lease_store(
    tmp_path: Path,
) -> None:
    """With a SessionStore other workers can resume the session, and a host-local lease cannot
    see a lost attempt still running there."""
    cli, record = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    activity = _activity(
        cli, workspace_root=tmp_path / "ws", session_store=_AnySessionStore()
    )
    turn = _input(session_id=fx.SESSION, resumes_lost_attempt=True)

    with pytest.raises(ApplicationError, match="lease store is not shared") as info:
        await ActivityEnvironment().run(activity.run_turn, turn)
    assert (
        info.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE and info.value.non_retryable
    )
    assert not record.exists(), "no CLI process was started"


async def test_a_single_worker_without_a_session_store_still_resumes_a_lost_attempt(
    tmp_path: Path,
) -> None:
    """No SessionStore: the transcript lives on this host only, so the local lease covers every
    CLI that can reach it."""
    cli, _ = _cli(tmp_path, turn=fx.COMPLETED_ONLY)
    activity = _activity(cli, workspace_root=tmp_path / "ws")
    turn = _input(session_id=fx.SESSION, resumes_lost_attempt=True)
    assert (await ActivityEnvironment().run(activity.run_turn, turn)).text


async def test_an_api_error_crash_stays_on_the_transcript_and_a_continuation_finishes_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    prompt = "What's the weather?"
    cli = FakeClaudeCLI(
        tmp_path / "cli",
        plans=[[prompt, [{"text": "Sunny."}]]],
        crash_at="before_prompt_persisted",
        crash_mode="api_error",
        crash_prompt=prompt,
        trigger_settle=0,
    )
    activity = ClaudeTurnActivity(cli_path=cli.cli_path, workspace_root=tmp_path / "ws")
    turn = ClaudeTurnInput(prompt=prompt, session_id=fx.SESSION)

    with pytest.raises(ApplicationError) as info:
        await ActivityEnvironment().run(activity.run_turn, turn)

    err = info.value
    assert err.type == CLAUDE_RESULT_ERROR_TYPE and err.non_retryable
    assert err.details[0]["terminal_reason"] == "api_error"
    assert err.details[0]["api_error_status"] == 529
    transcripts = [
        json.loads(line)
        for path in config.rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert any(entry.get("isApiErrorMessage") for entry in transcripts)

    result = await ActivityEnvironment().run(
        activity.run_turn,
        ClaudeTurnInput(prompt=CONTINUE_PROMPT, session_id=fx.SESSION),
    )

    assert result.text == "Sunny."


# Pytest discovers fixtures by name.
_ = _skip_version_probe
