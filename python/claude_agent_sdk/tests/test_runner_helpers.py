"""The runner's small decisions, one by one, without an engine.

How it reads a transcript (the calls a paused message holds, where their results go),
how it guards your own hooks, which errors mean "the session moved", and what it says
when the engine or the launcher cannot be checked.
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import HookMatcher, ToolResultBlock, UserMessage

from temporalio.claude_agent_sdk import ClaudeAgentSdkRunner, DeferredCall, ToolOutcome
from temporalio.claude_agent_sdk import _defer_hook as hook
from temporalio.claude_agent_sdk import _launcher as launcher
from temporalio.claude_agent_sdk import _runner as runner
from temporalio.exceptions import ApplicationError

PREFIX = runner.PREFIX


def assistant(uid: str, *calls: tuple[str, str]) -> dict[str, Any]:
    blocks = [
        {"type": "tool_use", "id": i, "name": n, "input": {"n": 1}} for i, n in calls
    ]
    return {"type": "assistant", "uuid": uid, "message": {"content": blocks}}


def marker(uid: str, paused: str, kind: str = "hook_deferred_tool") -> dict[str, Any]:
    return {
        "type": "attachment",
        "uuid": uid,
        "attachment": {"type": kind, "toolUseID": paused},
    }


def denials(uid: str, *ids: str) -> dict[str, Any]:
    blocks = [
        {
            "type": "tool_result",
            "tool_use_id": i,
            "content": hook.NOT_RUN,
            "is_error": True,
        }
        for i in ids
    ]
    return {"type": "user", "uuid": uid, "message": {"content": blocks}}


# ---- reading a transcript ----


def test_the_last_entry_is_the_last_transcript_entry() -> None:
    cost = {"type": "cost-state", "totalCostUSD": 1.0}
    side = {"type": "user", "uuid": "s", "isSidechain": True}
    assert runner._last_entry([]) is None
    assert runner._last_entry([cost, side, {"type": "user"}]) is None  # no uuid
    assert runner._last_entry([{"type": "user", "uuid": "a"}, cost, side]) == "a"


def test_only_a_user_entry_with_result_blocks_answers_calls() -> None:
    assert runner._result_ids(denials("u", "t1", "t2")) == ["t1", "t2"]
    assert runner._result_ids({"type": "user", "message": {"content": "text"}}) == []
    assert runner._result_ids({"type": "assistant", "message": {"content": []}}) == []
    assert runner._result_ids("not an entry") == []


def test_the_other_durable_calls_of_a_paused_message_are_found() -> None:
    """Only durable calls the hook denied as "not run yet", each once."""
    entries = [
        assistant(
            "a1",
            ("toolu_1", PREFIX + "count"),
            ("toolu_2", PREFIX + "count"),
            ("toolu_3", "Bash"),
            ("toolu_4", PREFIX + "count"),
        ),
        assistant("a2", ("toolu_2", PREFIX + "count")),  # the same block again
        marker("m1", "toolu_1"),
        denials("u1", "toolu_2", "toolu_3", "toolu_4"),
    ]
    records = {
        hook.denial_name("toolu_2"): "not_run",
        hook.denial_name("toolu_3"): "not_run",
        hook.denial_name("toolu_4"): "stopped",  # denied for another reason: never run
    }
    found = runner._siblings(entries, "toolu_1", records)
    assert [(c.id, c.name, c.input) for c in found] == [("toolu_2", "count", {"n": 1})]
    assert runner._siblings(entries, "toolu_9", records) == []  # no such marker


def test_results_of_the_other_calls_move_before_the_paused_calls_marker() -> None:
    """With their real results (an error stays an error), the input left as it is."""
    entries = [
        marker("h1", "toolu_1", kind="hook_success"),
        marker("m1", "toolu_1"),
        denials("u1", "toolu_2", "toolu_3"),
    ]
    before = copy.deepcopy(entries)
    results = {
        "toolu_2": ToolOutcome(content="boom", is_error=True),
        "toolu_3": ToolOutcome(content={"n": 3}),
    }
    moved = runner._deliver(entries, "m1", results)
    assert moved is not None
    resumed, delivered = moved
    assert delivered == {"toolu_2", "toolu_3"}
    assert [e["uuid"] for e in resumed] == ["u1", "h1", "m1"]  # from the very start
    two, three = resumed[0]["message"]["content"]
    assert two["content"] == "boom" and two["is_error"] is True
    assert three["content"] == '{"n": 3}' and "is_error" not in three
    assert resumed[0]["toolUseResult"] == '{"n": 3}'
    assert entries == before
    assert runner._deliver(entries, "h1", results) is None  # not the deferral marker
    assert runner._deliver(entries[:2], "m1", results) is None  # nothing after it


# ---- your own hooks ----


@pytest.mark.parametrize(
    ("output", "decision"),
    [
        ({"hookSpecificOutput": {"permissionDecision": "deny"}}, "deny"),
        ({"hookSpecificOutput": {"permissionDecision": ""}}, None),
        ({"decision": "block"}, "block"),
        ({"decision": "approve"}, "approve"),
        ({"decision": "maybe"}, None),
        ({"continue": False}, "stop"),
        ({"continue": True}, None),
        ({}, None),
        (None, None),
        ("deny", None),
    ],
)
def test_what_counts_as_a_hooks_decision(output: Any, decision: str | None) -> None:
    assert runner._decision_of(output) == decision


async def test_your_hooks_cannot_decide_on_calls_the_workflow_decides() -> None:
    """Their decision is dropped and recorded (the step then fails); on other tools
    it stands."""

    async def deny(input_data: Any, tool_use_id: Any, context: Any) -> Any:
        del tool_use_id, context
        return {"decision": "block", "reason": f"no {input_data['tool_name']}"}

    async def silent(input_data: Any, tool_use_id: Any, context: Any) -> Any:
        del input_data, tool_use_id, context
        return {"systemMessage": "seen"}

    violations: list[str] = []
    hooks = {"PreToolUse": [HookMatcher(hooks=[deny, silent]), "kept as it is"]}
    guarded = runner._guard_hooks(
        hooks, lambda name: name.startswith(PREFIX), violations
    )
    first = guarded["PreToolUse"][0]
    assert guarded["PreToolUse"][1] == "kept as it is"
    durable = {"tool_name": PREFIX + "refund"}
    assert await first.hooks[0](durable, "t1", None) == {}
    assert await first.hooks[1](durable, "t1", None) == {"systemMessage": "seen"}
    read = {"tool_name": "Read"}
    assert await first.hooks[0](read, "t2", None) == {
        "decision": "block",
        "reason": "no Read",
    }
    assert violations == [f"block on {PREFIX}refund"]
    for no_tool_hooks in ({"PostToolUse": [HookMatcher(hooks=[deny])]}, None, "x"):
        assert runner._guard_hooks(no_tool_hooks, lambda name: True, violations) is (
            no_tool_hooks
        )


def test_a_stopped_step_is_seen_only_in_the_hooks_own_denial() -> None:
    def said(*blocks: ToolResultBlock) -> bool:
        return runner._hook_said_stopped(UserMessage(content=list(blocks)))

    labelled = f"PreToolUse:Bash hook error: {hook.STOPPED}"
    assert said(ToolResultBlock("t", hook.STOPPED, True))
    assert said(
        ToolResultBlock("t", "fine", False), ToolResultBlock("u", labelled, True)
    )
    assert not said(ToolResultBlock("t", hook.STOPPED, False))  # not an error
    assert not said(ToolResultBlock("t", f"{hook.STOPPED} (echo)", True))
    assert not runner._hook_said_stopped(UserMessage(content=hook.STOPPED))


def test_a_result_from_the_workflow_is_read_either_way() -> None:
    """The data converter may hand the runner a dict instead of a ToolOutcome."""
    outcome = ToolOutcome(content="x", is_error=True)
    assert runner._as_outcome(outcome) is outcome
    assert runner._as_outcome({"content": "x", "is_error": 1}) == outcome
    assert runner._as_outcome({"content": None}) == ToolOutcome(content=None)


def test_a_moved_session_is_found_anywhere_in_the_error_chain() -> None:
    moved = runner._SessionMoved("continued after the checkpoint")
    try:
        try:
            raise moved
        except runner._SessionMoved as err:
            raise RuntimeError("the SDK wrapped it") from err
    except RuntimeError as wrapped:
        assert runner._moved(wrapped)
    try:
        try:
            raise moved
        except runner._SessionMoved:
            raise ValueError("raised while handling it")
    except ValueError as context:
        assert runner._moved(context)
    assert not runner._moved(RuntimeError("something else"))


# ---- options, and checks of the engine ----


def test_every_option_the_plugin_needs_is_named_at_once() -> None:
    with pytest.raises(ValueError) as refused:
        runner._check_extra_options(
            {"model": "x", "resume": "y", "mcp_servers": ["docs"]}
        )
    assert str(refused.value) == (
        "extra_options cannot set: model (use DurableClaudeAgent(model=...) or "
        "ClaudeAgentSdkRunner(model=...)); resume (the plugin sets it to pause and "
        "resume sessions); mcp_servers (pass a dict of servers; they are merged)"
    )


def test_a_runner_without_a_login_that_survives_resumes_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in runner.ENV_AUTH:
        monkeypatch.delenv(name, raising=False)
    with pytest.warns(UserWarning, match="no API key, cloud provider"):
        ClaudeAgentSdkRunner(cwd=str(tmp_path), env={"CLAUDE_CODE_USE_BEDROCK": "0"})
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        ClaudeAgentSdkRunner(cwd=str(tmp_path), env={"CLAUDE_CODE_USE_VERTEX": "1"})
    assert not [w for w in seen if "no API key" in str(w.message)]


async def test_an_engine_that_cannot_run_reports_no_version(tmp_path: Path) -> None:
    missing = str(tmp_path / "no-claude-here")
    configured = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), cli_path=missing, env={"ANTHROPIC_API_KEY": "x"}
    )
    assert configured._engine_path() == missing
    assert await configured._engine_version() is None
    assert configured._versions == {missing: None}  # checked once


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows starts the engine directly"
)
async def test_a_launcher_that_cannot_run_stops_the_step_and_is_checked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For example a temporary folder mounted noexec. By default the engine must end
    with its Worker, so the step stops (retryable), and the next step checks again:
    once the folder is fixed, the launcher is used."""
    script = tmp_path / "launch"
    script.write_text("#!/bin/sh\nexit 0\n")  # not executable
    written: list[str] = []

    def write(engine: str) -> str:
        written.append(engine)
        return str(script)

    monkeypatch.setattr(runner, "_launch_scripts", {})
    monkeypatch.setattr(runner, "_write_launch_script", write)
    configured = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), cli_path="/bin/true", env={"ANTHROPIC_API_KEY": "x"}
    )
    for _ in range(2):
        with pytest.raises(ApplicationError) as raised:
            await configured._launch_path()
        assert raised.value.type == runner.CLEANUP_UNAVAILABLE
        assert not raised.value.non_retryable
        assert f"cannot run {script} (" in str(raised.value)
    assert written == ["/bin/true", "/bin/true"] and runner._launch_scripts == {}
    script.write_text(  # the folder allows running programs again
        f'#!/bin/sh\n[ "$1" = {launcher.CHECK} ] && echo ok\n'
    )
    script.chmod(0o700)
    assert await configured._launch_path() == str(script)
    assert runner._launch_scripts == {"/bin/true": str(script)}


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows starts the engine directly"
)
async def test_a_launcher_that_cannot_run_falls_back_with_a_warning_in_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``engine_cleanup="best_effort"``: the engine starts directly, with a warning
    that says so, and the script is checked once."""
    script = tmp_path / "launch"
    script.write_text("#!/bin/sh\nexit 0\n")  # not executable
    written: list[str] = []

    def write(engine: str) -> str:
        written.append(engine)
        return str(script)

    monkeypatch.setattr(runner, "_launch_scripts", {})
    monkeypatch.setattr(runner, "_write_launch_script", write)
    monkeypatch.setattr(runner, "_warned", set())
    configured = ClaudeAgentSdkRunner(
        cwd=str(tmp_path),
        cli_path="/bin/true",
        env={"ANTHROPIC_API_KEY": "x"},
        engine_cleanup="best_effort",
    )
    with pytest.warns(
        UserWarning, match=f"cannot run {script} .*Going on anyway, because engine"
    ):
        assert await configured._launch_path() is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # checked once: no second warning
        assert await configured._launch_path() is None
    assert written == ["/bin/true"] and runner._launch_scripts == {"/bin/true": None}


async def test_a_required_runner_checks_a_launcher_a_best_effort_one_gave_up_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runners in one Worker process share the checked scripts: a failure that a
    best-effort runner accepted never lets a required one start without the check."""
    monkeypatch.setattr(runner, "_launch_scripts", {"/bin/true": None})
    monkeypatch.setattr(
        runner, "_write_launch_script", lambda engine: str(tmp_path / "missing")
    )
    configured = ClaudeAgentSdkRunner(
        cwd=str(tmp_path), cli_path="/bin/true", env={"ANTHROPIC_API_KEY": "x"}
    )
    if sys.platform == "win32":
        assert await configured._launch_path() is None
        return
    with pytest.raises(ApplicationError):
        await configured._launch_path()


def test_engine_cleanup_must_be_required_or_best_effort(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="engine_cleanup"):
        ClaudeAgentSdkRunner(
            cwd=str(tmp_path),
            env={"ANTHROPIC_API_KEY": "x"},
            engine_cleanup="off",  # type: ignore[arg-type]
        )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows ignores case: NO_PROXY and no_proxy are one",
)
def test_a_tool_steps_model_calls_go_to_the_stand_in_without_a_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """127.0.0.1 joins the Worker's proxy exceptions once, in both spellings."""
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    for given, merged in (
        ("internal, 127.0.0.1,,internal", "internal,127.0.0.1"),
        ("internal", "internal,127.0.0.1"),
        ("", "127.0.0.1"),
    ):
        configured = ClaudeAgentSdkRunner(
            cwd=str(tmp_path), env={"ANTHROPIC_API_KEY": "x", "NO_PROXY": given}
        )
        hook_dir = tmp_path / "step"
        hook_dir.mkdir(exist_ok=True)
        call = DeferredCall("toolu_1", "Bash", {"command": "ls"}, kind="engine")
        env = configured._step_env(
            dict(configured._env), str(hook_dir), call, "the step's key"
        )
        assert env["NO_PROXY"] == env["no_proxy"] == merged
        assert env["ANTHROPIC_BASE_URL"] == configured._stand_in.base_url
        assert env["ANTHROPIC_API_KEY"] == "the step's key"
        assert env["TCA_ALLOW_ID"] == "toolu_1"
        assert env["TCA_ALLOW_INPUT"] == hook.input_digest({"command": "ls"})
        assert env["ENABLE_TOOL_SEARCH"] == "false"  # every tool in the request
        assert env["DISABLE_AUTO_COMPACT"] == "1"  # no summary asked of the stand-in
