"""What keeps Claude Code's own tools inside the plugin's guarantees.

A command or MCP tool that runs as its own Activity must run once, with its approval,
in the Worker's environment, and talk to no model; nothing a tool prints, a hook in
``extra_options`` answers, or a settings file sets may change that. The tests that
came with a fix failed before it; the fault tests check paths that already held.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import claude_agent_sdk
import pytest
from claude_agent_sdk import HookMatcher

from temporalio.activity import ActivityCancellationDetails
from temporalio.claude_agent_sdk import (
    ClaudeAgentSdkRunner,
    DeferredCall,
    FileSessionStore,
    SegmentInput,
    ToolSpec,
    ToolStepInput,
    _defer_hook,
    _runner,
    _stand_in,
)
from temporalio.claude_agent_sdk._models import (
    TOOL_CALL_INTERRUPTED,
    TOOL_CALL_NOT_RUN,
)
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment
from tests.engine_tools.policy import shell_policy
from tests.helpers.fake_messages_api import (
    FakeMessagesAPI,
    engine_env,
    history_of,
    start_with_policy,
)
from tests.helpers.processes import alive
from tests.test_crash import wait_until

pytestmark = pytest.mark.timeout(240)
TOOLS = [ToolSpec("count", "Count one step.", {"type": "object"})]


def make_runner(
    tmp_path: Path, api: FakeMessagesAPI, cfg: str = "cfg", **kw: Any
) -> ClaudeAgentSdkRunner:
    (tmp_path / "work").mkdir(exist_ok=True)
    return ClaudeAgentSdkRunner(
        cwd=str(tmp_path / "work"), env=engine_env(api, str(tmp_path / cfg)), **kw
    )


async def pause_at_bash(runner: ClaudeAgentSdkRunner, command: str) -> Any:
    out = await runner.run(
        SegmentInput(
            session_id=str(uuid.uuid4()),
            prompt=f"run: {command}",
            tools=TOOLS,
            builtin_tools=["Bash"],
            transcript=[],
            tool_activities=["Bash"],
        ),
        1,
    )
    assert out.deferred is not None and out.deferred.kind == "engine", out
    return out


def step_for(out: Any) -> ToolStepInput:
    return ToolStepInput(
        session_id=out.session_id,
        checkpoint=out.checkpoint,
        call=out.deferred,
        tools=TOOLS,
        builtin_tools=["Bash"],
        transcript=out.transcript_add,
    )


def lines(path: Path) -> list[str]:
    return path.read_text().split() if path.exists() else []


# ---- a tool step runs once, talks to no model, and keeps the Worker's environment ----


def fail_after_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a tool step's engine fail after its call ran: the turn goes on after the
    call, and the stand-in model refuses the request that follows."""
    served = _stand_in._Handler.do_POST

    def refuse_after_the_call(handler: Any) -> None:
        call = handler.stand_in.call_for(handler.headers.get("x-api-key", ""))
        if call is None or not call.served:
            return served(handler)
        handler.rfile.read(int(handler.headers.get("content-length", 0)))
        error = {"type": "invalid_request_error", "message": "stand-in refuses"}
        handler._send({"type": "error", "error": error}, status=400)

    monkeypatch.setattr(_runner, "STEP_MAX_TURNS", 2)
    monkeypatch.setattr(_stand_in._Handler, "do_POST", refuse_after_the_call)


async def test_a_tool_step_keeps_its_result_when_the_engine_fails_afterwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model call after the tool fails: the call ran, so the step returns its
    result instead of failing, which would make Temporal run the command again."""
    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    try:
        out = await pause_at_bash(runner, "echo ran >> effects.log && echo local")
        fail_after_the_call(monkeypatch)
        outcome = await runner.run_tool_step(step_for(out), 1)
    finally:
        api.stop()
    assert outcome.content == "local" and not outcome.is_error
    assert outcome.entry is None  # a command's result goes as a message, as usual
    assert lines(effects) == ["ran"]


async def test_in_an_activity_a_kept_result_is_logged_with_the_engines_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The same, as the tool step Activity: the Worker's log says what happened."""
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    try:
        out = await pause_at_bash(runner, "echo local")
        fail_after_the_call(monkeypatch)
        with caplog.at_level(logging.WARNING):
            outcome = await ActivityEnvironment().run(
                runner.run_tool_step, step_for(out), 1
            )
    finally:
        api.stop()
    assert outcome.content == "local" and not outcome.is_error
    kept = [
        r.getMessage()
        for r in caplog.records
        if "and its result is kept" in r.getMessage()
    ]
    assert len(kept) == 1 and kept[0].startswith(
        f"Tool call {out.deferred.id} (Bash) ran, and its result is kept, but the "
        "engine failed after it"
    )


async def test_a_tool_step_whose_engine_breaks_before_the_result_may_have_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Here the command ran, but the engine broke just before the step got its result.
    The hook had let the call run, so the step fails as ``ToolCallInterrupted``: the
    Workflow then gives it no other attempt (unless the tool is in
    ``repeatable_tools``) and tells Claude the call may have run. It never makes up a
    result. A step run again anyway (a repeatable tool) runs the command again."""
    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    real = claude_agent_sdk.query  # the function the runner calls

    async def breaks_before_the_result(*args: Any, **kwargs: Any) -> Any:
        async for message in real(*args, **kwargs):
            if isinstance(message, claude_agent_sdk.UserMessage) and any(
                isinstance(block, claude_agent_sdk.ToolResultBlock)
                for block in message.content
            ):
                raise ConnectionResetError("the engine went away")
            yield message

    try:
        out = await pause_at_bash(runner, "echo ran >> effects.log && echo local")
        monkeypatch.setattr(_runner, "query", breaks_before_the_result)
        with pytest.raises(ApplicationError) as failed:
            await runner.run_tool_step(step_for(out), 1)
        assert failed.value.type == TOOL_CALL_INTERRUPTED
        assert failed.value.message == "the engine went away"
        assert isinstance(failed.value.__cause__, ConnectionResetError)
        assert lines(effects) == ["ran"]
        monkeypatch.setattr(_runner, "query", real)
        outcome = await runner.run_tool_step(step_for(out), 2)  # Temporal's retry
    finally:
        api.stop()
    assert outcome.content == "local" and not outcome.is_error
    assert lines(effects) == ["ran", "ran"]


async def test_a_tool_step_that_stopped_says_its_call_did_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelled or timed out before the call ran: the hook denies it, and the step
    fails for good (Claude learns the call did not run; it is not run again)."""
    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    real = _runner._hook_folder

    def stopped() -> str:
        folder = real()
        Path(folder, "stop").touch()  # what the runner writes on a cancel or timeout
        return folder

    try:
        out = await pause_at_bash(runner, "echo ran >> effects.log")
        monkeypatch.setattr(_runner, "_hook_folder", stopped)
        with pytest.raises(ApplicationError) as failed:
            await runner.run_tool_step(step_for(out), 1)
    finally:
        api.stop()
    assert failed.value.non_retryable and failed.value.type == TOOL_CALL_NOT_RUN
    assert failed.value.message.startswith(
        f"Claude Code did not run tool call {out.deferred.id} (Bash) in its step: "
    )
    assert _defer_hook.STOPPED in failed.value.message
    assert lines(effects) == []


@pytest.mark.parametrize("where", ["user settings", "global config"])
async def test_settings_files_cannot_send_a_tool_steps_model_calls_elsewhere(
    tmp_path: Path, where: str
) -> None:
    """Provider settings in Claude Code's settings files would win over the Worker's
    environment; a tool step's model calls must still reach only the stand-in."""
    api = start_with_policy(shell_policy)
    provider = start_with_policy(shell_policy)  # where the settings point
    cfg = tmp_path / "cfg-x"
    cfg.mkdir()
    env = {"ANTHROPIC_BASE_URL": provider.base_url}
    extra: dict[str, Any] = {}
    if where == "global config":
        (cfg / ".claude.json").write_text(json.dumps({"env": env}))
    else:
        (cfg / "settings.json").write_text(json.dumps({"env": env}))
        extra = {"setting_sources": ["user"]}
    plain = make_runner(tmp_path, api)
    configured = make_runner(tmp_path, api, cfg="cfg-x", extra_options=extra)
    try:
        out = await pause_at_bash(plain, "echo local")
        outcome = await configured.run_tool_step(step_for(out), 1)
    finally:
        api.stop()
        provider.stop()
    assert outcome.content == "local"
    assert provider.requests == []
    assert configured._stand_in.requests >= 1  # type: ignore[reportPrivateUsage]


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows ignores case in variable names: NO_PROXY and no_proxy are one",
)
async def test_a_command_in_a_tool_step_sees_the_workers_environment(
    tmp_path: Path,
) -> None:
    """The step's engine talks to the stand-in, but the command gets the Worker's
    provider settings back (a script that calls the Anthropic API works as in a
    segment), and proxy exceptions are merged, not replaced."""
    api = start_with_policy(shell_policy)
    url = api.base_url
    runner = make_runner(tmp_path, api)
    runner._env["no_proxy"] = "internal.example"  # type: ignore[reportPrivateUsage]
    command = 'echo "url=$ANTHROPIC_BASE_URL key=$ANTHROPIC_API_KEY np=$no_proxy"'
    try:
        out = await pause_at_bash(runner, command)
        outcome = await runner.run_tool_step(step_for(out), 1)
    finally:
        api.stop()
    assert outcome.content == (
        f"url={url} key=sk-ant-fake-not-real np=internal.example"
    )


@pytest.mark.parametrize("words", [_defer_hook.STOPPED, _defer_hook.STEP_ONLY])
async def test_a_command_that_prints_the_hooks_words_still_ran(
    tmp_path: Path, words: str
) -> None:
    """The step decides "did not run" from the hook's own records, not from the
    output, which a command (or an MCP tool) controls."""
    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    try:
        out = await pause_at_bash(runner, f"echo ran >> effects.log && echo '{words}'")
        outcome = await runner.run_tool_step(step_for(out), 1)
        # Inside a segment too: such output does not stop the step.
        inside = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt=f"run: echo '{words}'",
                tools=TOOLS,
                builtin_tools=["Bash"],
                transcript=[],
                tool_activities=[],
            ),
            1,
        )
    finally:
        api.stop()
    assert outcome.content == words and lines(effects) == ["ran"]
    assert not inside.is_error and inside.result == words


@pytest.mark.parametrize("activities", [[], ["Bash"]], ids=["segment", "tool-step"])
async def test_every_command_starts_in_the_working_directory(
    tmp_path: Path, activities: list[str]
) -> None:
    """Each segment and tool step is a new engine run, so a `cd` cannot carry over;
    the plugin makes that the rule everywhere, and tells Claude."""
    (tmp_path / "work" / "sub").mkdir(parents=True)
    ref: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        bash = [h for h in history if h.name == "Bash"]
        if not bash:
            return [ref[0].call("Bash", {"command": "cd sub && echo 1 > here.txt"})]
        if len(bash) == 1:
            return [ref[0].call("Bash", {"command": "echo 2 > here.txt"})]
        return [{"type": "text", "text": "FINAL"}]

    api = FakeMessagesAPI(decide).start()
    ref.append(api)
    runner = make_runner(tmp_path, api)
    transcript: list[dict[str, Any]] = []
    inp = SegmentInput(
        session_id=str(uuid.uuid4()),
        prompt="go",
        tools=TOOLS,
        builtin_tools=["Bash"],
        transcript=[],
        tool_activities=activities,
    )
    try:
        out = await runner.run(inp, 1)
        for index in range(1, 4):
            assert not out.is_error, out.error
            transcript = transcript[: out.transcript_keep] + out.transcript_add
            if out.deferred is None:
                break
            assert out.checkpoint is not None
            outcome = await runner.run_tool_step(
                ToolStepInput(
                    session_id=out.session_id,
                    checkpoint=out.checkpoint,
                    call=out.deferred,
                    tools=TOOLS,
                    builtin_tools=["Bash"],
                    transcript=list(transcript),
                ),
                1,
            )
            inp = SegmentInput(
                session_id=out.session_id,
                prompt=None,
                tools=TOOLS,
                builtin_tools=["Bash"],
                checkpoint=out.checkpoint,
                injected={out.deferred.id: outcome},
                transcript=list(transcript),
                tool_activities=activities,
                segment_index=index,
            )
            out = await runner.run(inp, 1)
    finally:
        api.stop()
    assert out.result == "FINAL"
    work = tmp_path / "work"
    # The second command did not start in sub/, where the first one went.
    assert (work / "sub" / "here.txt").read_text().split() == ["1"]
    assert (work / "here.txt").read_text().split() == ["2"]
    systems = [json.dumps(r.get("system")) for r in api.requests]
    assert systems and all(_runner.SHELL_HINT[:40] in s for s in systems)


# ---- calls that must wait for the Workflow never run inside the segment ----


async def test_a_subagent_cannot_run_a_command_that_runs_as_an_activity(
    tmp_path: Path,
) -> None:
    """With tool_approvals=["Bash"], a subagent's command would run with no approval;
    it is denied with a hint, so commands only ever run as Activities."""
    effects = tmp_path / "work" / "effects.log"
    task = "Run the cleanup command."
    ref: list[FakeMessagesAPI] = []
    seen: list[str] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, texts, history = history_of(body)
        if any(task in t for t in texts) and not any(
            t.startswith("Clean up.") for t in texts
        ):  # the subagent's conversation
            if history:
                seen.extend(str(h.content) for h in history)
                return [{"type": "text", "text": "done"}]
            return [ref[0].call("Bash", {"command": "echo ran >> effects.log"})]
        if not any(h.name == "Agent" for h in history):
            return [
                ref[0].call(
                    "Agent",
                    {
                        "description": "cleanup",
                        "prompt": task,
                        "subagent_type": "general-purpose",
                    },
                )
            ]
        return [{"type": "text", "text": "FINAL"}]

    api = FakeMessagesAPI(decide).start()
    ref.append(api)
    runner = make_runner(tmp_path, api)
    try:
        out = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="Clean up.",
                tools=TOOLS,
                builtin_tools=["Bash", "Agent"],
                transcript=[],
                tool_activities=["Bash"],
            ),
            1,
        )
    finally:
        api.stop()
    assert out.result == "FINAL" and lines(effects) == []
    assert seen and "only the main agent can call it" in seen[0]
    # The subagent's own steps are not part of the conversation the Workflow keeps;
    # its report is, as the Agent call's result.
    calls = [
        block.get("name")
        for entry in out.transcript_add
        for block in (entry.get("message") or {}).get("content") or []
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    assert calls == ["Agent"]


async def test_a_broken_hook_cannot_let_a_command_run_inside_the_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the hook gives no answer (it crashed), the command is not pre-approved, so
    Claude Code refuses a command that changes something instead of running it."""
    monkeypatch.setattr(
        _runner,
        "_hook_entry",
        lambda: {
            "type": "command",
            "command": sys.executable,
            "args": ["-c", "import sys; sys.exit(1)"],
        },
    )
    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    try:
        out = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="run: echo ran >> effects.log",
                tools=TOOLS,
                builtin_tools=["Bash"],
                transcript=[],
                tool_activities=["Bash"],
            ),
            1,
        )
    finally:
        api.stop()
    assert out.deferred is None and lines(effects) == []


@pytest.mark.parametrize("order", [["refund"], ["look_up", "refund"]])
async def test_a_hook_in_extra_options_cannot_decide_on_durable_calls(
    tmp_path: Path, order: list[str]
) -> None:
    """A deny from your own hook was overridden when the call came second in a
    message (the Workflow ran it with the paused one). Such hooks now stop the step
    with an error that says what to change, before any call runs."""

    async def no_refunds(event: Any, tool_use_id: Any, context: Any) -> Any:
        del tool_use_id, context
        if event.get("tool_name") == "mcp__durable__refund":
            output = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "refunds are not allowed",
            }
            return {"hookSpecificOutput": output}
        return {}

    ref: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        uses, _, _ = history_of(body)
        if not uses:
            return [ref[0].tool_use(name, {"order": "A-1"}) for name in order]
        return [{"type": "text", "text": "FINAL"}]

    api = FakeMessagesAPI(decide).start()
    ref.append(api)
    runner = make_runner(
        tmp_path,
        api,
        extra_options={
            "hooks": {"PreToolUse": [HookMatcher(matcher=None, hooks=[no_refunds])]}
        },
    )
    tools = [ToolSpec(n, n, {"type": "object"}) for n in ("look_up", "refund")]
    try:
        out = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="Refund A-1.",
                tools=tools,
                transcript=[],
            ),
            1,
        )
    finally:
        api.stop()
    assert out.is_error and out.error is not None
    assert "deny on mcp__durable__refund" in out.error
    assert out.deferred is None and out.siblings == []


# ---- the Worker's disk, options and the stand-in ----


async def test_a_session_already_on_disk_is_left_alone(tmp_path: Path) -> None:
    """Claude Code refuses a session id that already has a transcript; the runner
    used to remove that transcript afterwards as if it were its own."""
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    sid = str(uuid.uuid4())
    transcript, folder = runner._local_copy(sid)  # type: ignore[reportPrivateUsage]
    folder.mkdir(parents=True)
    entry = {
        "type": "user",
        "uuid": str(uuid.uuid4()),
        "parentUuid": None,
        "sessionId": sid,
        "message": {"role": "user", "content": "a session of someone else"},
    }
    transcript.write_text(json.dumps(entry) + "\n")
    (folder / "kept.txt").write_text("kept")
    try:
        try:
            await runner.run(
                SegmentInput(session_id=sid, prompt="run: echo hi", tools=TOOLS),
                1,
            )
        except Exception:  # the engine refuses the id; whatever it does, the files stay
            pass
    finally:
        api.stop()
    assert transcript.exists() and (folder / "kept.txt").exists()


def test_the_command_env_file_is_a_posix_shell_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claude Code runs it in Git Bash on Windows too, so it has LF line endings
    there as well, and it holds names only: the values travel in the environment.
    It runs quietly, even with a name in the Worker's environment no shell takes."""
    monkeypatch.setenv("TCA_NOT-A-SHELL-NAME", "1")
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        cwd=str(tmp_path / "work"), env={"ANTHROPIC_API_KEY": "x"}
    )
    env = runner._step_env(  # type: ignore[reportPrivateUsage]
        {"ANTHROPIC_BASE_URL": "https://example.invalid", "TCA_HOOK_DIR": "x"},
        str(tmp_path),
        DeferredCall("toolu_1", "Bash", {}, kind="engine"),
        "the step's key",
    )
    script = (tmp_path / "command_env.sh").read_bytes()
    assert env["CLAUDE_ENV_FILE"] == str(tmp_path / "command_env.sh")
    assert b"\r" not in script
    assert b'export ANTHROPIC_BASE_URL="$TCA_KEEP_ANTHROPIC_BASE_URL"' in script
    assert b"example.invalid" not in script
    assert env["TCA_KEEP_ANTHROPIC_BASE_URL"] == "https://example.invalid"
    unset = script.decode().splitlines()[-1].split()
    assert unset[0] == "unset" and set(unset[1:]) >= {
        "TCA_ALLOW_ID",
        "TCA_HOOK_DIR",
        "TCA_KEEP_ANTHROPIC_BASE_URL",
    }
    assert "TCA_NOT-A-SHELL-NAME" not in unset
    assert env["ANTHROPIC_API_KEY"] == "the step's key"
    shell = shutil.which("sh")
    if shell is not None:  # Git Bash's sh on Windows, when it is on PATH
        ran = subprocess.run(
            [shell, "-c", '. "$CLAUDE_ENV_FILE" && echo "$ANTHROPIC_BASE_URL" && env'],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert ran.returncode == 0 and ran.stderr == "", ran.stderr
        first, *rest = ran.stdout.splitlines()
        assert first == "https://example.invalid"  # the Worker's own value
        assert "the step's key" not in ran.stdout
        plugins = [n.split("=", 1)[0] for n in rest if n.startswith("TCA_")]
        assert [n for n in plugins if n.isidentifier()] == []


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_your_own_env_file_runs_after_the_workers_values_are_back(
    tmp_path: Path,
) -> None:
    """As Claude Code runs a ``CLAUDE_ENV_FILE`` of your own in a segment: last, so
    it can set what it likes, and a relative name is found from the working
    directory (not looked up in PATH)."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "user_env.sh").write_text(
        "export FROM_USER=1\nexport ANTHROPIC_BASE_URL=https://user.invalid\n"
    )
    runner = ClaudeAgentSdkRunner(
        cwd=str(work),
        env={"ANTHROPIC_API_KEY": "x", "CLAUDE_ENV_FILE": "user_env.sh"},
    )
    env = runner._step_env(  # type: ignore[reportPrivateUsage]
        {
            "ANTHROPIC_BASE_URL": "https://worker.invalid",
            "CLAUDE_ENV_FILE": "user_env.sh",
        },
        str(tmp_path),
        DeferredCall("toolu_1", "Bash", {}, kind="engine"),
        "the step's key",
    )
    ran = subprocess.run(
        [str(shutil.which("sh")), "-c", '. "$CLAUDE_ENV_FILE" && env'],
        env={**os.environ, **env},
        cwd=str(tmp_path),  # not the working directory: the path must not depend on it
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ran.returncode == 0 and ran.stderr == "", ran.stderr
    seen = [line for line in ran.stdout.splitlines() if "=" in line]
    assert "FROM_USER=1" in seen and "ANTHROPIC_BASE_URL=https://user.invalid" in seen
    assert "CLAUDE_ENV_FILE=user_env.sh" in seen  # the Worker's own value is back


def test_file_checkpointing_is_refused(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    with pytest.raises(ValueError, match="enable_file_checkpointing"):
        ClaudeAgentSdkRunner(
            cwd=str(tmp_path / "work"),
            env={"ANTHROPIC_API_KEY": "x"},
            extra_options={"enable_file_checkpointing": True},
        )


def post_to(
    stand_in: _stand_in.StandInModel,
    body: bytes,
    key: str | None,
    length: int | str | None = None,
) -> int:
    """POST ``body`` to the stand-in (saying it is ``length`` bytes long); the status."""
    headers = {"x-api-key": key} if key else {}
    if length is not None:
        headers["content-length"] = str(length)
    request = urllib.request.Request(
        f"{stand_in.base_url}/v1/messages", data=body, method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return int(response.status)
    except urllib.error.HTTPError as err:
        return err.code


BASH_CALL = {
    "type": "tool_use",
    "id": "toolu_recorded",
    "name": "Bash",
    "input": {"command": "echo hi", "description": "run"},
}
OFFERS_BASH = json.dumps({"model": "m", "tools": [{"name": "Bash"}]}).encode()


def test_the_stand_in_refuses_a_body_that_is_not_json() -> None:
    stand_in = _stand_in.StandInModel()
    call = stand_in.serve(BASH_CALL)
    assert post_to(stand_in, b"not json", call.key) == 400
    assert stand_in.requests == 1 and call.served == call.other == 0


@pytest.mark.parametrize("key", [None, "sk-ant-guess"], ids=["no-key", "wrong-key"])
def test_the_stand_in_answers_only_its_own_engines(key: str | None) -> None:
    """Any program on the machine can reach 127.0.0.1. Without the key of a running
    tool step it gets 401 at once: the stand-in does not wait for the body it says it
    sends. A step's key works only while the step runs."""
    stand_in = _stand_in.StandInModel()
    call = stand_in.serve(BASH_CALL)
    assert post_to(stand_in, b"{}", key, length=1000) == 401
    assert stand_in.requests == 0
    assert post_to(stand_in, b'{"model": "x", "messages": []}', call.key) == 200
    assert stand_in.requests == 1
    stand_in.done(call)
    assert post_to(stand_in, OFFERS_BASH, call.key, length=1000) == 401
    assert stand_in.requests == 1 and call.served == 0


def test_the_stand_in_ends_a_refused_connection_without_a_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused request gets its answer, then the stand-in ends its side and drops
    the rest of the body. A socket closed with unread data is reset instead, and on
    Windows the client can then lose the answer before it reads it (it did)."""
    # The stand-in waits for the client to close; a slow machine must not outlast it.
    monkeypatch.setattr(_stand_in, "LINGER_SECONDS", 30.0, raising=False)
    stand_in = _stand_in.StandInModel()
    stand_in.serve(BASH_CALL)
    port = int(stand_in.base_url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(b"POST /v1/messages HTTP/1.1\r\ncontent-length: 1000\r\n\r\n")
        answer = b""
        while chunk := sock.recv(65536):  # ends only when the stand-in ends its side
            answer += chunk
        assert answer.startswith(b"HTTP/1.1 401 ")
        for _ in range(5):  # the rest of the body: dropped, not answered with a reset
            sock.sendall(b"x" * 200)
            time.sleep(0.05)
    assert b"\r\nconnection: close\r\n" in answer.lower()
    assert stand_in.requests == 0


def test_the_stand_in_reads_no_body_over_its_limit_or_of_a_negative_length() -> None:
    """Both answer at once (before, the stand-in waited for the body that never came)."""
    stand_in = _stand_in.StandInModel()
    key = stand_in.serve(BASH_CALL).key
    assert post_to(stand_in, b"{}", key, _stand_in.MAX_BODY_BYTES + 1) == 413
    assert post_to(stand_in, b"{}", key, -1) == 400
    assert post_to(stand_in, b"{}", key, "two") == 400


def ask(
    stand_in: _stand_in.StandInModel,
    path: str,
    body: bytes | None,
    key: str,
    method: str = "POST",
) -> tuple[int, Any]:
    """Send a request with a step's key; its status and JSON answer."""
    request = urllib.request.Request(
        f"{stand_in.base_url}{path}",
        data=body,
        method=method,
        headers={"x-api-key": key},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return int(response.status), json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_the_stand_in_answers_with_the_steps_call_once() -> None:
    """The first model call that offers the call's tool gets the call, exactly as
    recorded; every other model call gets "ok", and is counted. Token counting,
    other paths and GET get small answers; a body that is not an object is refused."""
    stand_in = _stand_in.StandInModel()
    call = stand_in.serve(BASH_CALL)
    status, answer = ask(stand_in, "/v1/messages", b'{"model": "m"}', call.key)
    assert status == 200 and answer["content"] == [
        {"type": "text", "text": _stand_in.ANSWER}
    ]  # it does not offer Bash: a side call of the engine
    status, answer = ask(stand_in, "/v1/messages", OFFERS_BASH, call.key)
    assert status == 200 and answer["model"] == "m"
    assert answer["content"] == [BASH_CALL] and answer["stop_reason"] == "tool_use"
    status, answer = ask(stand_in, "/v1/messages", OFFERS_BASH, call.key)
    assert answer["content"] == [{"type": "text", "text": _stand_in.ANSWER}]
    assert answer["stop_reason"] == "end_turn"
    assert (call.served, call.other) == (1, 2)
    assert ask(stand_in, "/v1/messages/count_tokens", b"{}", call.key) == (
        200,
        {"input_tokens": 1},
    )
    assert ask(stand_in, "/v1/other", b"{}", call.key) == (200, {})
    assert ask(stand_in, "/", None, call.key, method="GET") == (200, {})
    status, refused = ask(stand_in, "/v1/messages", b"[1, 2]", call.key)
    assert status == 400 and refused["error"]["message"] == "not a JSON object"
    assert stand_in.requests == 6  # the POSTs with the step's key


def test_a_request_in_flight_when_the_step_ends_never_gets_its_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step that ended says whether its engine was sent the call (``done``). A
    request whose key was checked just before the step ended still gets only the
    short text: the call is never sent after that answer, so the answer stays true."""
    stand_in = _stand_in.StandInModel()
    call = stand_in.serve(BASH_CALL)
    found = stand_in.call_for(call.key)
    assert found is call
    assert stand_in.done(call) is False  # not sent: the step reports "not run"
    monkeypatch.setattr(stand_in, "call_for", lambda key: found)  # checked before
    status, answer = ask(stand_in, "/v1/messages", OFFERS_BASH, call.key)
    assert status == 200
    assert answer["content"] == [{"type": "text", "text": _stand_in.ANSWER}]
    assert call.served == 0
    other = _stand_in.StandInModel()
    sent = other.serve(BASH_CALL)
    assert ask(other, "/v1/messages", OFFERS_BASH, sent.key)[1]["content"] == [
        BASH_CALL
    ]
    assert other.done(sent) is True  # sent: the call may have run


def test_the_stand_in_streams_the_steps_call_as_the_api_would() -> None:
    """Claude Code streams its model calls: the call arrives as one tool_use block,
    its input as JSON, and the message stops for the tool."""
    stand_in = _stand_in.StandInModel()
    call = stand_in.serve(BASH_CALL)
    body = json.dumps({"model": "m", "stream": True, "tools": [{"name": "Bash"}]})
    request = urllib.request.Request(
        f"{stand_in.base_url}/v1/messages",
        data=body.encode(),
        method="POST",
        headers={"x-api-key": call.key},
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        events = [
            json.loads(line[len("data: ") :])
            for line in response.read().decode().splitlines()
            if line.startswith("data: ")
        ]
    assert [e["type"] for e in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1]["content_block"] == {**BASH_CALL, "input": {}}
    assert json.loads(events[2]["delta"]["partial_json"]) == BASH_CALL["input"]
    assert events[4]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("credential", ["key helper", "login", "token variables"])
async def test_a_tool_steps_engine_sends_the_stand_ins_key_whatever_the_worker_has(
    tmp_path: Path, credential: str
) -> None:
    """The stand-in answers only its own key. Other credentials on the Worker (a key
    helper in the settings, a Claude login, token variables) must not take its place,
    or every tool step's model calls would be refused."""
    api = start_with_policy(shell_policy)
    cfg = tmp_path / "cfg-x"
    cfg.mkdir()
    env = engine_env(api, str(cfg))
    extra: dict[str, Any] = {}
    if credential == "key helper":
        helper = tmp_path / "helper.py"
        helper.write_text("print('sk-ant-from-a-helper')\n", encoding="utf-8")
        command = f'"{Path(sys.executable).as_posix()}" "{helper.as_posix()}"'
        (cfg / "settings.json").write_text(json.dumps({"apiKeyHelper": command}))
        extra = {"setting_sources": ["user"]}
    elif credential == "login":
        login = {
            "accessToken": "sk-ant-oat01-not-real",
            "refreshToken": "sk-ant-ort01-not-real",
            "expiresAt": int(time.time() + 86400) * 1000,
            "scopes": ["user:inference", "user:profile"],
            "subscriptionType": "max",
        }
        (cfg / ".credentials.json").write_text(json.dumps({"claudeAiOauth": login}))
    else:
        env["ANTHROPIC_AUTH_TOKEN"] = "sk-ant-from-a-variable"
        env["CLAUDE_CODE_OAUTH_TOKEN"] = "sk-ant-oat01-from-a-variable"
    plain = make_runner(tmp_path, api)
    configured = ClaudeAgentSdkRunner(
        cwd=str(tmp_path / "work"), env=env, extra_options=extra
    )
    try:
        out = await pause_at_bash(plain, "echo local")
        outcome = await configured.run_tool_step(step_for(out), 1)
    finally:
        api.stop()
    assert outcome.content == "local" and not outcome.is_error
    assert configured._stand_in.requests >= 1  # type: ignore[reportPrivateUsage]


@pytest.mark.parametrize("where", ["tool-step", "segment"])
async def test_commands_see_none_of_the_plugins_variables(
    tmp_path: Path, where: str
) -> None:
    """The hook's folder and the ids it keeps are the plugin's business: a command
    cannot read them, whether it runs as its own Activity or inside the segment."""
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    command = "env | grep '^TCA_' || echo none"
    try:
        if where == "tool-step":
            out = await pause_at_bash(runner, command)
            seen = (await runner.run_tool_step(step_for(out), 1)).content
        else:
            out = await runner.run(
                SegmentInput(
                    session_id=str(uuid.uuid4()),
                    prompt=f"run: {command}",
                    tools=TOOLS,
                    builtin_tools=["Bash"],
                    transcript=[],
                    tool_activities=[],
                ),
                1,
            )
            assert not out.is_error, out.error
            seen = out.result
    finally:
        api.stop()
    assert seen == "none"


async def test_a_tool_step_on_a_worker_set_up_the_other_way_did_not_run(
    tmp_path: Path,
) -> None:
    """Like a segment, a tool step that lands on a Worker that keeps conversations
    elsewhere fails in a way the Workflow retries, until a matching Worker takes it:
    the call did not run (``ToolCallNotRun``, retryable)."""
    api = start_with_policy(shell_policy)
    held = make_runner(tmp_path, api)
    stored = make_runner(
        tmp_path, api, session_store=FileSessionStore(tmp_path / "sessions")
    )
    try:
        out = await pause_at_bash(held, "echo hi")
        no_conversation = step_for(out)
        no_conversation.transcript = None
        for runner, step, words in [
            (stored, step_for(out), "kept in its Workflow"),
            (held, no_conversation, "session store this runner"),
        ]:
            with pytest.raises(ApplicationError) as failed:
                await runner.run_tool_step(step, 1)
            assert failed.value.type == TOOL_CALL_NOT_RUN
            assert not failed.value.non_retryable
            assert words in failed.value.message
            assert not failed.value.message.endswith("Retrying.")
        with pytest.raises(ApplicationError) as failed:  # a real mismatch is final
            wrong = step_for(out)
            wrong.checkpoint = str(uuid.uuid4())
            await held.run_tool_step(wrong, 1)
        assert failed.value.type == TOOL_CALL_NOT_RUN and failed.value.non_retryable
    finally:
        api.stop()


# ---- a tool step that cannot end with its Worker, or whose Worker shuts down ----


def refuse_locks(fd: int) -> None:
    """Like some network and FUSE file systems."""
    del fd
    raise OSError(errno.ENOLCK, "No locks available")


@pytest.mark.parametrize(
    "cause",
    [
        pytest.param(
            "launcher",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="Windows starts the engine directly"
            ),
        ),
        "lock",
        "job",
    ],
)
async def test_a_tool_step_whose_engine_could_not_end_with_its_worker_did_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    """By default (``engine_cleanup="required"``) the step stops before Claude Code
    runs the call, and says the call did not run, so the Workflow tries it again
    (on another Worker, or this one once fixed). The job object of Windows is
    simulated, on every system: its engine had started, and is ended."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    before = _runner._child_pids()  # earlier tests' processes, if any
    started: set[int] = set()

    def outside_the_job() -> dict[int, str]:
        failed: dict[int, str] = {}
        for child in subprocess_cli._ACTIVE_CHILDREN:
            pid = getattr(child, "pid", None)
            if isinstance(pid, int) and pid not in before:
                started.add(pid)
                failed[pid] = _runner._job_problem("cannot join (Windows error 5)")
        return failed

    try:
        out = await pause_at_bash(runner, "echo ran >> effects.log")
        if cause == "launcher":
            script = tmp_path / "launch"
            script.write_text("#!/bin/sh\nexit 0\n")  # not executable
            monkeypatch.setattr(_runner, "_launch_scripts", {})
            monkeypatch.setattr(_runner, "_write_launch_script", lambda _: str(script))
        elif cause == "lock":
            monkeypatch.setattr(_runner, "_lock_file", refuse_locks)
        else:
            monkeypatch.setattr(_runner, "_engines_end_with_worker", outside_the_job)
            monkeypatch.setattr(_runner, "_JOB_POLL", True)  # as on Windows
        with pytest.raises(ApplicationError) as failed:
            await runner.run_tool_step(step_for(out), 1)
    finally:
        api.stop()
    assert failed.value.type == TOOL_CALL_NOT_RUN and not failed.value.non_retryable
    assert {"launcher": "cannot run", "lock": "cannot lock files", "job": "error 5"}[
        cause
    ] in failed.value.message
    assert lines(effects) == []
    if cause == "job":
        assert started, "the engine had started"
        await wait_until(lambda: not [pid for pid in started if alive(pid)], 15)
    else:
        assert runner._stand_in.requests == 0  # Claude Code never started


async def run_until_cancelled(
    runner: ClaudeAgentSdkRunner,
    step: ToolStepInput,
    ready: Any,
    shutting_down: bool,
    **details: bool,
) -> Any:
    """Run ``step`` as an Activity, cancel it once ``ready()`` (with ``details``, after
    the Worker began to shut down if ``shutting_down``), and return what the Activity
    returned or raised."""
    env = ActivityEnvironment()
    task = asyncio.ensure_future(env.run(runner.run_tool_step, step, 1))
    deadline = time.monotonic() + 120
    while not ready():
        assert time.monotonic() < deadline and not task.done()
        await asyncio.sleep(0.05)
    if shutting_down:
        env.worker_shutdown()
    env.cancel(cancellation_details=ActivityCancellationDetails(**details))
    try:
        return await asyncio.wait_for(task, 60)
    except BaseException as err:  # noqa: BLE001
        return err


@pytest.mark.parametrize("when", ["before the engine started", "during the call"])
async def test_a_tool_step_cut_by_its_workers_shutdown_says_whether_its_call_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    """A Worker that shuts down cancels its steps. Before the hook let the call run
    (here: before Claude Code started), the call did not run (the Workflow tries it
    again elsewhere); after, it may have run (it is not run again)."""
    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    real = claude_agent_sdk.query
    starting = [False]

    async def slow_to_start(*args: Any, **kwargs: Any) -> Any:
        starting[0] = True
        await asyncio.sleep(60)
        async for message in real(*args, **kwargs):
            yield message

    try:
        command = "echo ran >> effects.log; sleep 60"
        out = await pause_at_bash(runner, command)
        if when == "before the engine started":
            monkeypatch.setattr(_runner, "query", slow_to_start)
            ready: Any = lambda: starting[0]  # noqa: E731
        else:
            ready = lambda: lines(effects) == ["ran"]  # noqa: E731
        raised = await run_until_cancelled(
            runner, step_for(out), ready, True, worker_shutdown=True
        )
    finally:
        api.stop()
    assert isinstance(raised, ApplicationError), raised
    assert raised.message == "Its Worker shut down."
    early = when == "before the engine started"
    assert raised.type == (TOOL_CALL_NOT_RUN if early else TOOL_CALL_INTERRUPTED)
    assert not raised.non_retryable
    assert lines(effects) == ([] if early else ["ran"])


async def test_a_call_a_settings_rule_let_run_is_never_reported_as_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook gave no answer (it could not start), so Claude Code's own check
    decided, and an allow rule from a settings file Claude Code trusts (here the
    engine's own; a managed policy would do the same) let the command run. The Worker
    then shuts down while it runs. There is no ``allowed`` record, yet the step says
    the call may have run (it did), so the Workflow never runs it again. (Claude
    Code ignores allow rules in a project's own settings until its folder is
    trusted.)"""
    effects = tmp_path / "work" / "effects.log"
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    real_folder = _runner._hook_folder  # type: ignore[reportPrivateUsage]

    def with_an_allow_rule() -> str:
        folder = real_folder()
        settings = Path(folder, "settings.json")
        content = json.loads(settings.read_text(encoding="utf-8"))
        content["permissions"] = {"allow": ["Bash"]}
        settings.write_text(json.dumps(content), encoding="utf-8")
        return folder

    try:
        out = await pause_at_bash(runner, "echo ran >> effects.log; sleep 60")
        monkeypatch.setattr(
            _runner,
            "_hook_entry",
            lambda: {
                "type": "command",
                "command": sys.executable,
                "args": ["-c", "import sys; sys.exit(1)"],
            },
        )
        monkeypatch.setattr(_runner, "_hook_folder", with_an_allow_rule)
        raised = await run_until_cancelled(
            runner,
            step_for(out),
            lambda: lines(effects) == ["ran"],
            True,
            worker_shutdown=True,
        )
    finally:
        api.stop()
    assert isinstance(raised, ApplicationError), raised
    assert raised.type == TOOL_CALL_INTERRUPTED  # it ran: never "did not run"
    assert lines(effects) == ["ran"]


async def test_a_tool_step_keeps_its_result_when_its_worker_shuts_down_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The call ran and the step has its result; the Worker shuts down while the
    engine finishes: the step returns the result, so the call is not reported as
    interrupted."""
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    real = claude_agent_sdk.query
    has_result = [False]

    async def hangs_after_the_result(*args: Any, **kwargs: Any) -> Any:
        async for message in real(*args, **kwargs):
            yield message
            if isinstance(message, claude_agent_sdk.UserMessage) and any(
                isinstance(block, claude_agent_sdk.ToolResultBlock)
                for block in message.content
            ):
                has_result[0] = True
                await asyncio.sleep(60)

    try:
        out = await pause_at_bash(runner, "echo local")
        monkeypatch.setattr(_runner, "query", hangs_after_the_result)
        kept = await run_until_cancelled(
            runner, step_for(out), lambda: has_result[0], True, worker_shutdown=True
        )
    finally:
        api.stop()
    assert not isinstance(kept, BaseException), kept
    assert kept.content == "local" and not kept.is_error


@pytest.mark.parametrize(
    ("shutting_down", "details"),
    [
        (False, {"cancel_requested": True}),
        (True, {"cancel_requested": True}),
        (True, {"timed_out": True}),
    ],
    ids=["workflow", "workflow-during-shutdown", "timeout-during-shutdown"],
)
async def test_a_tool_step_cancelled_for_another_reason_stays_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shutting_down: bool,
    details: dict[str, bool],
) -> None:
    """Only a cancel because the Worker shuts down becomes a failure that says
    whether the call ran: a cancel the Workflow requested, even while the Worker
    shuts down, or a timeout, keeps its meaning."""
    api = start_with_policy(shell_policy)
    runner = make_runner(tmp_path, api)
    real = claude_agent_sdk.query
    starting = [False]

    async def slow_to_start(*args: Any, **kwargs: Any) -> Any:
        starting[0] = True
        await asyncio.sleep(60)
        async for message in real(*args, **kwargs):
            yield message

    try:
        out = await pause_at_bash(runner, "echo local")
        monkeypatch.setattr(_runner, "query", slow_to_start)
        raised = await run_until_cancelled(
            runner, step_for(out), lambda: starting[0], shutting_down, **details
        )
    finally:
        api.stop()
    assert isinstance(raised, asyncio.CancelledError), raised
