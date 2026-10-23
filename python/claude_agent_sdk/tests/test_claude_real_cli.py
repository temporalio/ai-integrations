"""Opt-in checks of the Claude Code CLI bundled in ``claude-agent-sdk`` 0.2.165. Skipped in CI.

``CLAUDE_REAL_CLI=1`` runs the local ``--version`` probes. ``CLAUDE_REAL_CLI_NETWORK=1`` adds
the missing-session ``--resume``, which may contact Anthropic even without credentials, and,
with ``ANTHROPIC_API_KEY``, the real turns.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import claude_agent_sdk
import pytest
from pydantic import BaseModel, Field

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeSessionState,
    FileLeaseStore,
    FileSessionStore,
    _native,
)
from temporalio.claude_agent_sdk._activity import _PGID_DIR_NAME, ClaudeTurnActivity
from temporalio.claude_agent_sdk._cli import (
    VERSION_PROBE_TIMEOUT_SECONDS,
    CLIResolutionError,
    cli_version,
    resolve_cli,
)
from temporalio.claude_agent_sdk._models import (
    CLAUDE_RESULT_ERROR_TYPE,
    READ_ONLY_NATIVE_TOOLS,
    RESUMABLE_DETAIL,
    ClaudeTurnInput,
    ClaudeTurnOptions,
)
from temporalio.claude_agent_sdk._native import (
    NATIVE_TOOLS_VERIFIED,
    sandbox_backend_available,
)
from temporalio.client import (
    WorkflowContinuedAsNewError,
    WorkflowFailureError,
    WorkflowHistory,
)
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Worker

from .coding_assistant.manager import AssistantInput, CodingAssistant
from .coding_assistant.tracker import TicketTracker
from .helpers import kill_worker as kw
from .helpers.workflows import (
    ClaudeNativeDenied,
    ClaudeNativeSequence,
    ClaudeResume,
    ClaudeSandboxFailover,
    ClaudeStopped,
)

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("CLAUDE_REAL_CLI") != "1",
    reason="set CLAUDE_REAL_CLI=1 to check the bundled Claude Code CLI; skipped in CI",
)

PINNED_SDK = "0.2.165"
PINNED_CLI = "2.1.294"
VERSION_LINE = f"{PINNED_CLI} (Claude Code)"

# Provider channels the missing-session resume strips. That result does not
# require them. Unset credentials do not keep this command off the network.
_CREDENTIAL_ENV = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
)


def _bundled_cli() -> str:
    """The wheel's bundled executable. ``native=True`` refuses a ``claude`` on PATH.

    Skip when this install has none.
    """
    try:
        resolved = resolve_cli(None, native=True)
    except CLIResolutionError as exc:
        pytest.skip(f"bundled CLI unavailable: {exc}")
    assert resolved.version == PINNED_CLI, resolved
    assert Path(resolved.path).parent.name == "_bundled"
    return resolved.path


def _child_env(config: Path) -> dict[str, str]:
    """Parent environment with credential channels removed and config isolated."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _CREDENTIAL_ENV and key != "CLAUDECODE"
    }
    env["CLAUDE_CONFIG_DIR"] = str(config)
    return env


def test_bundled_cli_version_matches_the_verified_pair(tmp_path: Path) -> None:
    """``--version`` and ``-v`` print the pinned CLI, the only one native tools are verified for."""
    cli = _bundled_cli()
    assert claude_agent_sdk.__version__ == PINNED_SDK
    assert NATIVE_TOOLS_VERIFIED == frozenset({(PINNED_SDK, PINNED_CLI)})
    assert cli_version(cli) == PINNED_CLI
    env = _child_env(tmp_path / "config")
    for flag in ("--version", "-v"):
        out = subprocess.run(
            [cli, flag],
            capture_output=True,
            text=True,
            timeout=VERSION_PROBE_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
            env=env,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == VERSION_LINE


@pytest.mark.skipif(
    os.environ.get("CLAUDE_REAL_CLI_NETWORK") != "1",
    reason=(
        "set CLAUDE_REAL_CLI_NETWORK=1 to run missing-session resume; "
        "it may contact Anthropic"
    ),
)
def test_resume_of_a_missing_session_reports_missing_session(
    tmp_path: Path,
) -> None:
    """``--resume=<id>`` with no transcript exits 1 and reports that session.

    May contact Anthropic even with credentials stripped. The equals form is what the SDK
    sends: ``--resume`` takes an optional value, so a separate argument would not bind.
    """
    cli = _bundled_cli()
    session_id = str(uuid.uuid4())
    config = tmp_path / "config"
    work = tmp_path / "work"
    config.mkdir()
    work.mkdir()
    message = f"No conversation found with session ID: {session_id}"
    # Non-native resume argv from SubprocessCLITransport._build_command.
    out = subprocess.run(
        [
            cli,
            "--output-format",
            "stream-json",
            "--verbose",
            "--system-prompt",
            "",
            "--tools",
            "",
            "--permission-mode",
            "dontAsk",
            f"--resume={session_id}",
            "--setting-sources=",
            "--strict-mcp-config",
            "--input-format",
            "stream-json",
        ],
        input=json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "hi"},
                "parent_tool_use_id": None,
                "uuid": str(uuid.uuid4()),
            }
        ),
        capture_output=True,
        text=True,
        timeout=VERSION_PROBE_TIMEOUT_SECONDS,
        cwd=work,
        env=_child_env(config),
        check=False,
    )
    assert out.returncode == 1, out.stderr
    assert out.stderr.strip() == message
    lines = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    result = lines[0]
    assert result["type"] == "result"
    assert result["subtype"] == "error_during_execution"
    assert result["is_error"] is True
    assert result["num_turns"] == 0
    assert result["session_id"] == session_id
    assert result["errors"] == [message]
    projects = config / "projects"
    assert not projects.exists() or not any(projects.rglob("*.jsonl"))


class _Room(BaseModel):
    kind: Literal["room"]
    beds: int


class _Suite(BaseModel):
    kind: Literal["suite"]
    floor: int


class _Booking(BaseModel):
    id: uuid.UUID
    at: datetime
    # Pydantic emits `discriminator` and `prefixItems`, which 2.1.294 rejects as given.
    stay: _Room | _Suite = Field(discriminator="kind")
    nights: tuple[int, str]


_MODEL = "claude-haiku-5-5"

_REAL_TURN = pytest.mark.skipif(
    os.environ.get("CLAUDE_REAL_CLI_NETWORK") != "1"
    or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="set CLAUDE_REAL_CLI_NETWORK=1 and ANTHROPIC_API_KEY to run a real turn",
)


@_REAL_TURN
@pytest.mark.timeout(180)
async def test_a_pydantic_schema_still_returns_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pydantic emits ``format: uuid`` and ``format: date-time``, a ``discriminator`` and
    ``prefixItems``; the bundled CLI must still return ``structured_output`` for that schema,
    which the workflow then revalidates."""
    _bundled_cli()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    schema = _Booking.model_json_schema()
    assert {p.get("format") for p in schema["properties"].values()} >= {
        "uuid",
        "date-time",
    }
    assert "discriminator" in schema["properties"]["stay"]
    assert "prefixItems" in schema["properties"]["nights"]
    booking_id = "123e4567-e89b-12d3-a456-426614174000"

    result = await ActivityEnvironment().run(
        ClaudeTurnActivity(workspace_root=tmp_path / "ws").run_turn,
        ClaudeTurnInput(
            prompt=f"Return the booking with id {booking_id} at 2026-10-15T09:00:00Z: "
            "a suite on floor 4, for nights (2, 'weekend').",
            options=ClaudeTurnOptions(model=_MODEL, output_schema=schema),
        ),
    )

    assert result.structured_output is not None, result.text
    booking = _Booking.model_validate(result.structured_output)
    assert str(booking.id) == booking_id
    assert booking.stay == _Suite(kind="suite", floor=4)
    assert booking.nights == (2, "weekend")
    assert result.model_usage and any(
        _MODEL in model for model in result.model_usage
    ), result.model_usage
    assert result.duration_api_ms is not None and result.duration_api_ms > 0


@_REAL_TURN
@pytest.mark.timeout(120)
async def test_prompts_reach_claude_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``@path`` does not read that worker file into the turn, and a leading ``/`` does not
    run a CLI slash command (which answers with no model round)."""
    _bundled_cli()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    token = f"OSPREY-{uuid.uuid4().hex[:8]}"
    secret = tmp_path / "secret.txt"
    secret.write_text(f"{token}\n")
    activity = ClaudeTurnActivity(workspace_root=tmp_path / "ws")

    for prompt in (f"Quote the token in @{secret}, or say NOFILE.", "/cost"):
        result = await ActivityEnvironment().run(
            activity.run_turn,
            ClaudeTurnInput(
                prompt=prompt, options=ClaudeTurnOptions(model=_MODEL, max_turns=1)
            ),
        )
        assert token not in (result.text or ""), result.text
        assert result.num_turns >= 1, result.text


def _events(history: WorkflowHistory, kind: str) -> int:
    field = f"activity_task_{kind}_event_attributes"
    return sum(e.WhichOneof("attributes") == field for e in history.events)


@_REAL_TURN
@pytest.mark.timeout(120)
async def test_an_invalid_api_key_fails_as_authentication_and_is_not_resumed(
    env: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI reports ``authentication_failed``; with a SessionStore that would otherwise
    resume an API error, the message fails after its one Activity."""
    _bundled_cli()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    # Only this process's CLI sees it; the real key is never logged.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "invalid")
    client = kw.client(env)
    task_queue = f"claude-real-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeResume],
        plugins=[
            ClaudeAgentPlugin(
                workspace_root=tmp_path / "ws",
                session_store=cast(Any, FileSessionStore(tmp_path / "store")),
                lease_store=FileLeaseStore(tmp_path / "leases"),
            )
        ],
    ):
        handle = await client.start_workflow(
            ClaudeResume.run, id=task_queue, task_queue=task_queue
        )
        with pytest.raises(WorkflowFailureError) as failed:
            await asyncio.wait_for(handle.result(), timeout=90)
        history = await handle.fetch_history()

    cause: BaseException | None = failed.value
    while cause is not None and not (
        isinstance(cause, ApplicationError) and cause.type == CLAUDE_RESULT_ERROR_TYPE
    ):
        cause = cause.__cause__
    assert isinstance(cause, ApplicationError), failed.value
    details = cause.details[0]
    assert details["error_kind"] == "authentication_failed", details
    assert details[RESUMABLE_DETAIL] is False, details
    assert _events(history, "scheduled") == 1


@_REAL_TURN
@pytest.mark.timeout(240)
async def test_a_denial_reason_reaches_claude_and_the_call_does_not_run(
    env: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``approve`` returns ``(False, reason)`` for a real Bash call. Claude repeats the reason's
    token, the command never ran, and the result lists the denial."""
    _bundled_cli()
    assert sandbox_backend_available(), "no Seatbelt or bubblewrap on this host"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    token = f"ORCHID-{uuid.uuid4().hex[:8]}"
    marker = f"ran-{uuid.uuid4().hex[:8]}.txt"
    client = kw.client(env)
    task_queue = f"claude-real-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeNativeDenied],
        plugins=[
            ClaudeAgentPlugin(workspace_root=tmp_path / "ws", enable_native_tools=True)
        ],
    ):
        result = await client.execute_workflow(
            ClaudeNativeDenied.run,
            args=[
                f"{token}: use the staging host instead.",
                f"Use the Bash tool to run exactly: touch {marker}\n"
                "If the call is denied, reply with the denial reason word for word.",
                _MODEL,
            ],
            id=task_queue,
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=200),
        )

    assert token in (result.text or ""), result.text
    assert not list(tmp_path.rglob(marker))
    assert result.permission_denials and any(
        d.get("tool_name") == "Bash" for d in result.permission_denials
    ), result.permission_denials


@_REAL_TURN
@pytest.mark.timeout(240)
@pytest.mark.parametrize("store", [False, True], ids=["local", "session-store"])
async def test_a_message_after_a_stopped_native_call_answers_its_own_prompt(
    env: WorkflowEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store: bool,
) -> None:
    """The first message is cancelled while its Bash call awaits approval. The next message
    settles that call without running it and its reply answers its own prompt."""
    _bundled_cli()
    assert sandbox_backend_available(), "no Seatbelt or bubblewrap on this host"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    marker = f"ran-{uuid.uuid4().hex[:8]}.txt"
    token = f"TULIP-{uuid.uuid4().hex[:8]}"
    client = kw.client(env)
    task_queue = f"claude-real-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeStopped],
        plugins=[
            ClaudeAgentPlugin(
                workspace_root=tmp_path / "ws",
                enable_native_tools=True,
                session_store=cast(Any, FileSessionStore(tmp_path / "store"))
                if store
                else None,
                lease_store=FileLeaseStore(tmp_path / "leases") if store else None,
            )
        ],
    ):
        result = await client.execute_workflow(
            ClaudeStopped.run,
            args=[
                f"Use the Bash tool to run exactly: touch {marker}",
                f"Reply with only this word: {token}",
                _MODEL,
                True,
            ],
            id=task_queue,
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=200),
        )

    assert not list(tmp_path.rglob(marker))
    assert token in (result["text"] or ""), result
    assert result["cost_usd"] > 0, result


@_REAL_TURN
@pytest.mark.timeout(420)
@pytest.mark.parametrize("store", [False, True], ids=["local", "session-store"])
async def test_native_calls_approved_after_an_earlier_one_ran_all_run(
    env: WorkflowEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store: bool,
) -> None:
    """2.1.294 keeps an answered call's ``hook_deferred_tool`` marker and forks the transcript at
    it, and answers a prompt sent with an approved resume after the next deferral
    (anthropics/claude-code#97196). Later deferred calls, in the same message and the next, must
    still be deferred, approved once each, and run. Claude may repeat a touch; each repeat is
    its own approved call."""
    _bundled_cli()
    assert sandbox_backend_available(), "no Seatbelt or bubblewrap on this host"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    a, b, c = (f"{n}-{uuid.uuid4().hex[:8]}.txt" for n in ("one", "two", "three"))
    client = kw.client(env)
    task_queue = f"claude-real-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeNativeSequence],
        plugins=[
            ClaudeAgentPlugin(
                workspace_root=tmp_path / "ws",
                enable_native_tools=True,
                session_store=cast(Any, FileSessionStore(tmp_path / "store"))
                if store
                else None,
                lease_store=FileLeaseStore(tmp_path / "leases") if store else None,
            )
        ],
    ):
        result = await client.execute_workflow(
            ClaudeNativeSequence.run,
            args=[
                [
                    f"Use the Bash tool to run exactly: touch {a}\nAfter it finishes, use "
                    f"the Bash tool again, in a separate call, to run exactly: touch {b}\n"
                    "Then reply with only: done",
                    f"Use the Bash tool to run exactly: touch {c}\nThen reply with only: done",
                ],
                _MODEL,
            ],
            id=task_queue,
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=400),
        )

    # The workflow returning at all means neither message hit max_turns.
    ran = {name for name in (a, b, c) if list((tmp_path / "ws").rglob(name))}
    assert ran == {a, b, c}, result
    assert len(result["approved"]) == len(set(result["approved"])) >= 3, result
    assert sorted(result["finished"]) == sorted(result["approved"]), result
    commands = [i.get("command", "") for i in result["inputs"]]
    assert all(any(n in cmd for n in (a, b, c)) for cmd in commands), commands
    assert not result["denials"], result


@_REAL_TURN
@pytest.mark.timeout(180)
async def test_an_auto_allowed_read_runs_in_one_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``native_auto_allow`` lets a real Read run without a deferral; the ledger records it."""
    _bundled_cli()
    assert sandbox_backend_available(), "no Seatbelt or bubblewrap on this host"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    token = f"HERON-{uuid.uuid4().hex[:8]}"
    session_id = str(uuid.uuid4())
    workspace = tmp_path / "ws"
    (workspace / session_id).mkdir(parents=True)
    (workspace / session_id / "token.txt").write_text(f"{token}\n")

    result = await ActivityEnvironment().run(
        ClaudeTurnActivity(workspace_root=workspace, enable_native_tools=True).run_turn,
        ClaudeTurnInput(
            prompt="Read token.txt in the current directory with the Read tool and reply "
            "with its contents only.",
            session_id=session_id,
            native_tools=["Read"],
            native_auto_allow=sorted(READ_ONLY_NATIVE_TOOLS),
            options=ClaudeTurnOptions(model=_MODEL, max_turns=3),
        ),
    )

    assert result.deferred is None
    assert token in (result.text or ""), result.text
    assert [(c.tool_name, c.status) for c in result.native_calls] == [
        ("Read", "finished")
    ], result.native_calls


@_REAL_TURN
@pytest.mark.timeout(180)
async def test_native_tools_with_an_output_schema_return_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate hook matches every tool, so it must pass Claude Code's synthetic
    ``StructuredOutput`` call instead of denying it as a tool the message did not ask for."""
    _bundled_cli()
    assert sandbox_backend_available(), "no Seatbelt or bubblewrap on this host"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    token = f"HERON-{uuid.uuid4().hex[:8]}"
    session_id = str(uuid.uuid4())
    workspace = tmp_path / "ws"
    (workspace / session_id).mkdir(parents=True)
    (workspace / session_id / "token.txt").write_text(f"{token}\n")
    schema = {
        "type": "object",
        "properties": {"token": {"type": "string"}},
        "required": ["token"],
    }

    result = await ActivityEnvironment().run(
        ClaudeTurnActivity(workspace_root=workspace, enable_native_tools=True).run_turn,
        ClaudeTurnInput(
            prompt="Read token.txt in the current directory with the Read tool and return "
            "its contents as the token.",
            session_id=session_id,
            native_tools=["Read"],
            native_auto_allow=sorted(READ_ONLY_NATIVE_TOOLS),
            options=ClaudeTurnOptions(model=_MODEL, max_turns=4, output_schema=schema),
        ),
    )

    assert result.deferred is None
    assert result.structured_output == {"token": token}, result.text


@_REAL_TURN
@pytest.mark.timeout(180)
async def test_an_auto_allowed_read_outside_the_cwd_waits_for_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``native_auto_allow`` covers the session's directory only; a worker file outside it
    defers to ``approve`` and its contents never reach the turn."""
    _bundled_cli()
    assert sandbox_backend_available(), "no Seatbelt or bubblewrap on this host"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    token = f"HERON-{uuid.uuid4().hex[:8]}"
    secret = (tmp_path / "secret.txt").resolve()
    secret.write_text(f"{token}\n")
    session_id = str(uuid.uuid4())
    (tmp_path / "ws" / session_id).mkdir(parents=True)

    result = await ActivityEnvironment().run(
        ClaudeTurnActivity(
            workspace_root=tmp_path / "ws", enable_native_tools=True
        ).run_turn,
        ClaudeTurnInput(
            prompt=f"Read {secret} with the Read tool and reply with its contents only.",
            session_id=session_id,
            native_tools=["Read"],
            native_auto_allow=["Read"],
            options=ClaudeTurnOptions(model=_MODEL, max_turns=3),
        ),
    )

    assert result.deferred is not None and result.deferred.name == "Read", result
    assert token not in (result.text or ""), result.text
    assert result.native_calls == []


@_REAL_TURN
@pytest.mark.timeout(300)
async def test_an_approved_write_cannot_reach_the_native_tool_state(
    env: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write is not sandboxed: a deny rule, not the approval every call here gets, keeps it off
    the state the gate hook decides from. The rule refuses the call before the hook sees it."""
    _bundled_cli()
    assert sandbox_backend_available(), "no Seatbelt or bubblewrap on this host"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    token = f"EGRET-{uuid.uuid4().hex[:8]}"
    workspace = (tmp_path / "ws").resolve()
    target = workspace / ".native-tool-state" / f"probe-{token}.json"
    client = kw.client(env)
    task_queue = f"claude-real-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClaudeNativeSequence],
        plugins=[ClaudeAgentPlugin(workspace_root=workspace, enable_native_tools=True)],
    ):
        result = await client.execute_workflow(
            ClaudeNativeSequence.run,
            args=[
                [
                    f"Use the Write tool to write the text {token} to the file {target}. "
                    "Do not use any other path. Then reply with only: done"
                ],
                _MODEL,
                ["Write"],
            ],
            id=task_queue,
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=280),
        )

    state = target.parent
    assert not target.exists()
    assert not list(workspace.rglob(f"*{token}*"))
    assert not [
        i for i in result["inputs"] if state.name in str(i.get("file_path", ""))
    ], result
    assert any(
        d.get("tool_name") == "Write"
        and state.name in str(d.get("tool_input", {}).get("file_path", ""))
        for d in result["denials"]
    ), result


@_REAL_TURN
@pytest.mark.timeout(180)
async def test_a_read_the_gate_hook_never_decided_does_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate hook that exits 1 (a non-blocking error, like a timeout) decides nothing; 2.1.294
    would then run a Read in the cwd unasked. The ask rule sends it to ``can_use_tool``."""
    _bundled_cli()
    assert sandbox_backend_available(), "no Seatbelt or bubblewrap on this host"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    undecided = tmp_path / "undecided.py"
    undecided.write_text("import sys\nsys.exit(1)\n")
    monkeypatch.setattr(_native, "HOOK_SCRIPT", undecided)
    token = f"HERON-{uuid.uuid4().hex[:8]}"
    session_id = str(uuid.uuid4())
    workspace = tmp_path / "ws"
    (workspace / session_id).mkdir(parents=True)
    (workspace / session_id / "token.txt").write_text(f"{token}\n")

    try:
        result = await ActivityEnvironment().run(
            ClaudeTurnActivity(
                workspace_root=workspace, enable_native_tools=True
            ).run_turn,
            ClaudeTurnInput(
                prompt="Read token.txt in the current directory with the Read tool and "
                "reply with its contents only.",
                session_id=session_id,
                native_tools=["Read"],
                options=ClaudeTurnOptions(model=_MODEL, max_turns=3),
            ),
        )
        assert result.permission_denials, "the Read reached can_use_tool and was denied"
    except ApplicationError as exc:
        # Claude may retry the denied Read until the turn limit instead of giving up.
        assert exc.details[0]["terminal_reason"] == "max_turns", exc

    transcripts = list((tmp_path / "config").rglob(f"{session_id}.jsonl"))
    assert transcripts and not any(token in t.read_text() for t in transcripts)
    assert _native.native_calls(workspace / ".native-tool-state" / session_id) == []


@_REAL_TURN
@pytest.mark.timeout(420)
async def test_a_lost_local_worker_resumes_from_the_session_store(
    env: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker A is SIGKILLed mid tool call; B resumes from the SessionStore; the tool ran once."""
    _bundled_cli()
    disk_a, disk_b = tmp_path / "disk-a", tmp_path / "disk-b"
    workspace = (tmp_path / "workspace").resolve()
    store, leases = tmp_path / "store", tmp_path / "leases"
    for path in (disk_a, disk_b, workspace):
        path.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(disk_b))
    session_id = str(uuid.uuid4())
    client = kw.client(env)
    task_queue = f"claude-real-{uuid.uuid4().hex[:8]}"
    ready = tmp_path / "worker-a-ready"
    spec = {
        "target": env.client.service_client.config.target_host,
        "namespace": env.client.namespace,
        "task_queue": task_queue,
        "workflow": ClaudeSandboxFailover.__name__,
        "cli": None,
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
            "PYTHONPATH": str(_PLUGIN_ROOT),
        },
        start_new_session=True,
    )
    cli_pgid = 0
    try:

        async def worker_a_ready() -> bool:
            assert proc.poll() is None, "worker A exited before it was ready"
            return ready.exists()

        await kw.until(worker_a_ready, timeout=60, interval=0.5)
        handle = await client.start_workflow(
            ClaudeSandboxFailover.run,
            session_id,
            id=task_queue,
            task_queue=task_queue,
        )

        async def tool_called() -> bool:
            return await handle.query(ClaudeSandboxFailover.calls) >= 1

        await kw.until(tool_called, timeout=180, interval=0.5)
        (pgid_file,) = (workspace / _PGID_DIR_NAME).glob(f"{session_id}.*")
        cli_pgid = int(pgid_file.read_text())
        assert kw.group_alive(cli_pgid)

        kw.killpg(proc.pid)
        proc.wait(timeout=10)

        async def cli_gone() -> bool:
            return not kw.group_alive(cli_pgid)

        await kw.until(cli_gone, timeout=15, interval=0.5)
        shutil.rmtree(disk_a)

        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ClaudeSandboxFailover],
            max_cached_workflows=0,
            plugins=[
                ClaudeAgentPlugin(
                    workspace_root=workspace,
                    session_store=cast(Any, FileSessionStore(store)),
                    lease_store=FileLeaseStore(leases),
                    disconnect_timeout=timedelta(seconds=1),
                )
            ],
        ):
            await handle.signal(ClaudeSandboxFailover.release)
            text = await asyncio.wait_for(handle.result(), timeout=240)
            calls = await handle.query(ClaudeSandboxFailover.calls)
        history = await handle.fetch_history()
    finally:
        kw.killpg(proc.pid)
        if cli_pgid:
            kw.killpg(cli_pgid)
        if proc.poll() is None:
            proc.wait(timeout=10)

    assert "bluebird" in text.lower(), text
    assert "sunny" in text.lower(), text
    assert calls == 1
    assert _events(history, "completed") == 2, "the first turn, then the resumed second"
    assert _events(history, "timed_out") == 1, "A's attempt, lost with its worker"


@_REAL_TURN
@pytest.mark.timeout(400)
async def test_the_coding_assistant_example_runs_on_the_real_cli(
    env: WorkflowEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The example agent (``tests/coding_assistant``) on haiku, two messages through planner,
    coder and reviewer: the planner reads the ticket through the tracker Activity, the policy
    approves the coder's edit of the planned file and keeps it from deleting, the coder comments
    on the ticket, each message returns a typed review, and the workflow continues as new.
    Worker loss is covered offline, in ``test_claude_coding_assistant.py``."""
    _bundled_cli()
    assert sandbox_backend_available(), "no Seatbelt or bubblewrap on this host"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    tickets = tmp_path / "tickets"
    tickets.mkdir()
    (tickets / "ENG-42.json").write_text(
        json.dumps({"id": "ENG-42", "title": "add() subtracts", "file": "app.py"})
    )
    session = ClaudeSessionState(session_id=str(uuid.uuid4()), max_cost_usd=1.0)
    repo = tmp_path / "ws" / str(session.session_id)
    (repo / "build").mkdir(parents=True)
    (repo / "app.py").write_text("def add(a, b):\n    return a - b\n")
    client = kw.client(env)
    workflow_id = f"claude-real-{uuid.uuid4().hex[:8]}"
    tracker = TicketTracker(tickets)
    async with Worker(
        client,
        task_queue=workflow_id,
        workflows=[CodingAssistant],
        activities=[tracker.fetch_ticket, tracker.add_comment],
        plugins=[
            ClaudeAgentPlugin(workspace_root=tmp_path / "ws", enable_native_tools=True)
        ],
    ):
        started = await client.start_workflow(
            CodingAssistant.run,
            AssistantInput(session=session, model=_MODEL),
            id=workflow_id,
            task_queue=workflow_id,
            execution_timeout=timedelta(seconds=380),
        )
        first = client.get_workflow_handle(workflow_id, run_id=started.result_run_id)
        fix = await first.execute_update(
            CodingAssistant.ask, "Fix the bug ticket ENG-42 describes."
        )
        await first.execute_update(
            CodingAssistant.ask,
            "Use the Bash tool to run exactly: rm -rf build",
        )
        decisions = await first.query(CodingAssistant.decisions)
        cost = await first.query(CodingAssistant.cost_usd)
        with pytest.raises(WorkflowContinuedAsNewError):
            await first.result(follow_runs=False)
        await client.get_workflow_handle(workflow_id).terminate()

    assert "Edit approved" in decisions, decisions
    assert "a + b" in (repo / "app.py").read_text()
    # Claude may decline the delete itself or ask and be denied; either way it must not run.
    assert (repo / "build").is_dir(), "the delete ran"
    assert any("app.py" in f for f in fix.files), fix
    assert (tickets / "ENG-42.comments.jsonl").read_text().strip(), "no comment posted"
    assert 0 < cost <= 1.0
