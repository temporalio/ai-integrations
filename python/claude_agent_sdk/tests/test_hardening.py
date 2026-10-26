"""Cases an independent review found: they failed before the fixes they test.

- ``extra_options`` could replace what the plugin relies on (an ``env`` without the
  hook's variables let a built-in tool run twice; ``mcp_servers`` removed the durable
  tools).
- Tool names Claude Code rewrites (a dot, a space, non-ASCII) could never run.
- A working directory with decomposed Unicode or emoji gives Claude Code and the SDK
  two session keys, so every segment retried forever.
- A cancelled or timed-out segment's engine could still run a built-in tool.
- On Windows, Claude Code runs hook commands with Git Bash (or PowerShell), which
  mangled the hook's path, so the hook never ran; now it starts without a shell.
- ``FileSessionStore`` could read half a line while an append was being written.
- With background tasks on, a background subagent kept the engine running after it
  paused, calling the model hundreds of times.
- ``FileSessionStore`` cut long file names (mixing transcripts) and missed other
  Workers' writes.
- Two agents asking for live output failed with a message about Workflow Streams.
- ``FileSessionStore`` appended after the half line a dying writer left, so the next
  load failed; gave keys that differ only in characters it replaced one file; and
  kept the entry ids of a failed write, so the SDK's retry skipped them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import errno
import gc
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import warnings
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import SessionKey

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    DeferredCall,
    DurableClaudeAgent,
    DurableTool,
    FileSessionStore,
    SegmentInput,
    ToolOutcome,
    ToolSpec,
    ToolStepInput,
    _defer_hook,
    _runner,
    _session_store,
    activity_as_tool,
    make_segment_activity,
    make_tool_step_activity,
)
from temporalio.claude_agent_sdk.testing import ScriptedClaude
from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import CancelledError
from temporalio.testing import ActivityEnvironment
from temporalio.worker import Worker
from tests.endless.activities import count
from tests.endless.policy import count_policy
from tests.hardening.workflows import TwoLiveAgentsWorkflow, WriterWorkflow
from tests.helpers.fake_messages_api import (
    PREFIX,
    FakeMessagesAPI,
    engine_env,
    start_with_policy,
)
from tests.helpers.workers import FAIL_FAST

pytestmark = pytest.mark.timeout(120)
AUTH = {"ANTHROPIC_API_KEY": "sk-ant-fake-not-real"}


def _runner_in(tmp_path: Path, **kwargs: Any) -> ClaudeAgentSdkRunner:
    return ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store"),
        cwd=str(tmp_path),
        env=AUTH,
        **kwargs,
    )


# ---- options the plugin relies on ----


@pytest.mark.parametrize(
    "key", ["model", "tools", "max_turns", "settings", "resume", "cwd", "session_store"]
)
def test_extra_options_cannot_replace_what_the_plugin_sets(
    tmp_path: Path, key: str
) -> None:
    with pytest.raises(ValueError, match=f"cannot set: {key}"):
        _runner_in(tmp_path, extra_options={key: "x"})


def test_extra_args_cannot_pass_the_same_engine_flags(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="extra_args"):
        _runner_in(tmp_path, extra_options={"extra_args": {"--resume": "abc"}})
    _runner_in(tmp_path, extra_options={"extra_args": {"debug-to-stderr": None}})


@pytest.mark.parametrize(
    "flag",
    [
        "--permission-mode",
        "permission-mode",
        "--dangerously-skip-permissions",
        "--allow-dangerously-skip-permissions",
        "--permission-prompt-tool",
    ],
)
def test_extra_args_cannot_approve_tool_calls(tmp_path: Path, flag: str) -> None:
    """A tool step replaces the permission mode and the permission prompt tool, so
    that only its hook approves its call: a flag in extra_args would win over them."""
    with pytest.raises(ValueError, match="set permission_mode or"):
        _runner_in(tmp_path, extra_options={"extra_args": {flag: "bypassPermissions"}})
    # The same through extra_options is fine: a tool step replaces it.
    _runner_in(tmp_path, extra_options={"permission_mode": "bypassPermissions"})


@pytest.mark.parametrize("one_at_a_time", [False, True], ids=["default", "one-call"])
def test_extra_options_can_set_a_default_prompt_and_settings(
    tmp_path: Path, one_at_a_time: bool
) -> None:
    preset = {"type": "preset", "preset": "claude_code"}
    runner = _runner_in(
        tmp_path,
        extra_options={
            "system_prompt": preset,
            "setting_sources": ["project"],
            "session_store_flush": "eager",
        },
        one_tool_at_a_time=one_at_a_time,
    )
    hint = _runner.ONE_TOOL_HINT

    def prompt_for(agent_prompt: str | None) -> Any:
        inp = SegmentInput(
            session_id="s", prompt="hi", tools=[], system_prompt=agent_prompt
        )
        options = runner._engine_options(  # type: ignore[reportPrivateUsage]
            inp, {}, "s", False, None, None, str(tmp_path / "hook"), "server", []
        )
        assert options["setting_sources"] == ["project"]
        assert options["session_store_flush"] == "eager"
        return options["system_prompt"]

    if one_at_a_time:  # Claude Code's own prompt, or the agent's, with the hint
        assert prompt_for(None) == {**preset, "append": hint}
        assert prompt_for("You are a refund agent.") == (
            f"You are a refund agent.\n\n{hint}"
        )
    else:  # several calls in one message all run, so no hint by default
        assert prompt_for(None) == preset
        assert prompt_for("You are a refund agent.") == "You are a refund agent."


@pytest.mark.parametrize(
    ("extra", "expected"),
    [({}, "default"), ({"permission_mode": "acceptEdits"}, "acceptEdits")],
    ids=["unset", "set"],
)
def test_the_permission_mode_is_always_set(
    tmp_path: Path, extra: dict[str, Any], expected: str
) -> None:
    """Claude Code 2.1.285+ uses auto mode when none is set (it blocked a subagent)."""
    runner = _runner_in(tmp_path, extra_options=extra)
    inp = SegmentInput(session_id="s", prompt="hi", tools=[])
    options = runner._engine_options(  # type: ignore[reportPrivateUsage]
        inp, {}, "s", False, None, None, str(tmp_path / "hook"), "server", []
    )
    assert options["permission_mode"] == expected


def test_extra_options_cannot_take_the_durable_servers_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="durable"):
        _runner_in(tmp_path, extra_options={"mcp_servers": {"durable": {}}})


def test_extra_options_are_merged_with_the_plugins_own(tmp_path: Path) -> None:
    docs = {"type": "http", "url": "http://127.0.0.1:1/mcp"}
    runner = _runner_in(
        tmp_path,
        extra_options={
            "env": {"FROM_EXTRA": "1", "TCA_HOOK_DIR": "not this"},
            "mcp_servers": {"docs": docs},
            "allowed_tools": ["mcp__docs__search"],
            "permission_mode": "default",
        },
    )
    inp = SegmentInput(
        session_id=str(uuid.uuid4()),
        prompt="hi",
        tools=[ToolSpec("count", "Count one step.", {"type": "object"})],
    )
    hook_dir = str(tmp_path / "hook")
    small = runner._engine_options(  # type: ignore[reportPrivateUsage]
        inp, {}, inp.session_id, False, None, None, hook_dir, "server", []
    )
    env = small["env"]
    assert (
        env["FROM_EXTRA"] == "1"
        and env["ANTHROPIC_API_KEY"] == AUTH["ANTHROPIC_API_KEY"]
    )
    assert env["TCA_HOOK_DIR"] == hook_dir  # the hook's variables always win
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert small["mcp_servers"] == {"docs": docs, "durable": "server"}
    assert small["allowed_tools"] == [PREFIX + "count", "mcp__docs__search"]
    assert small["permission_mode"] == "default"
    assert small["max_buffer_size"] == _runner.MIN_BUFFER_BYTES
    # A result the engine echoes as one long line still fits.
    big = ToolOutcome("x" * (9 * 1024 * 1024))
    large = runner._engine_options(  # type: ignore[reportPrivateUsage]
        inp, {"toolu_1": big}, inp.session_id, True, None, None, hook_dir, "server", []
    )
    # Tools that run as Activities are never pre-approved: the hook answers for them.
    inp.builtin_tools, inp.tool_activities = ["Bash", "Read"], ["Bash", "mcp__*"]
    shell = runner._engine_options(  # type: ignore[reportPrivateUsage]
        inp, {}, inp.session_id, False, None, None, hook_dir, "server", []
    )
    assert shell["allowed_tools"] == [PREFIX + "count", "Read"]
    assert shell["env"]["CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR"] == "1"
    assert shell["system_prompt"] == _runner.SHELL_HINT
    assert large["max_buffer_size"] >= 8 * 9 * 1024 * 1024


# ---- tool names ----


@pytest.mark.parametrize(
    "name", ["orders.lookup", "look up order", "zwróć_zamówienie", "x" * 51]
)
def test_tool_names_claude_code_would_rewrite_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match="Tool name"):
        activity_as_tool(count, name=name)
    tool = DurableTool(activity=count, name=name, description="d", input_schema={})
    with pytest.raises(ValueError, match="Tool name"):
        DurableClaudeAgent(tools=[tool])


def test_a_plain_function_is_not_a_tool() -> None:
    async def not_an_activity(args: dict[str, Any]) -> dict[str, Any]:
        return args

    with pytest.raises(TypeError, match="@activity.defn"):
        activity_as_tool(not_an_activity)


# ---- working directories ----


@pytest.mark.parametrize(
    "name",
    [unicodedata.normalize("NFD", "wörk"), "emoji-\U0001f600"],
    ids=["decomposed", "emoji"],
)
def test_directories_with_two_session_keys_are_refused(
    tmp_path: Path, name: str
) -> None:
    work = tmp_path / name
    work.mkdir()
    with pytest.raises(ValueError, match="different session keys"):
        ClaudeAgentSdkRunner(
            session_store=FileSessionStore(tmp_path / "store"), cwd=str(work), env=AUTH
        )


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_a_directory_given_through_a_symlink_is_fine(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store"),
        cwd=str(tmp_path / "link"),
        env=AUTH,
    )


def test_a_composed_non_ascii_directory_is_fine(tmp_path: Path) -> None:
    work = tmp_path / unicodedata.normalize("NFC", "wörk")
    work.mkdir()
    ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store"), cwd=str(work), env=AUTH
    )


# ---- the hook after the step stopped ----

DURABLE_CALL = {"tool_name": PREFIX + "count", "tool_use_id": "toolu_1"}
BUILTIN_CALL = {"tool_name": "Write", "tool_use_id": "toolu_2"}


def _decision(event: dict[str, Any]) -> str:
    return str(_defer_hook.decide(event).get("permissionDecision", "none"))


def test_the_hook_stops_builtin_tools_once_the_step_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("TCA_HOOK_DIR", str(run_dir))
    monkeypatch.delenv("TCA_ANSWERED_IDS", raising=False)
    assert _decision(BUILTIN_CALL) == "none"  # a running step: built-ins run normally
    (run_dir / "stop").touch()  # the Activity was cancelled or timed out
    assert _decision(BUILTIN_CALL) == "deny"
    assert _decision(DURABLE_CALL) == "defer"  # ends the run; no one will run it
    assert [p.name for p in (run_dir / "denied").iterdir()] == [
        BUILTIN_CALL["tool_use_id"]
    ]
    shutil.rmtree(run_dir)  # the run ended, or the engine cannot see the folder
    assert _decision(BUILTIN_CALL) == "deny" and _decision(DURABLE_CALL) == "deny"


def test_the_hook_starts_without_a_shell() -> None:
    entry = _runner._hook_entry()  # type: ignore[reportPrivateUsage]
    assert entry["command"] == sys.executable  # exec form: nothing to quote
    hook = str(Path(_runner.__file__).with_name("_defer_hook.py"))
    assert entry["args"] == ["-I", "-S", hook]  # standard library only


async def _pause_once(runner: ClaudeAgentSdkRunner) -> Any:
    spec = ToolSpec("count", "Count one step.", {"type": "object"})
    inp = SegmentInput(session_id=str(uuid.uuid4()), prompt="count to 1", tools=[spec])
    return await runner.run(inp, 1)


async def test_a_hook_path_a_shell_would_mangle_still_pauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spaces, "$" and backticks: a shell would split or expand them."""
    folder = tmp_path / "dir with space $HOME `x` (1)"
    folder.mkdir()
    hook = folder / "_defer_hook.py"
    shutil.copy(Path(_runner.__file__).with_name("_defer_hook.py"), hook)
    entry = {"type": "command", "command": sys.executable, "args": [str(hook)]}
    monkeypatch.setattr(_runner, "_hook_entry", lambda: entry)
    api = start_with_policy(count_policy)
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store"),
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
    )
    try:
        out = await _pause_once(runner)
    finally:
        api.stop()
    assert not out.is_error, out.error
    assert out.deferred is not None and out.deferred.name == "count"
    assert runner.stub_calls == 0


async def test_a_hook_that_cannot_see_its_folder_fails_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For example an engine in a sandbox with its own /tmp: fail, never answer without the tool."""
    hook = Path(_runner.__file__).with_name("_defer_hook.py")
    blind = (
        "import os, runpy, sys; os.environ['TCA_HOOK_DIR'] = sys.argv[1]; "
        "runpy.run_path(sys.argv[2], run_name='__main__')"
    )
    missing = str(tmp_path / "not-visible")
    entry = {
        "type": "command",
        "command": sys.executable,
        "args": ["-c", blind, missing, str(hook)],
    }
    monkeypatch.setattr(_runner, "_hook_entry", lambda: entry)
    holder: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        if _results(body):
            return [{"type": "text", "text": "I could not count."}]
        return [holder[0].tool_use("count", {"n": 1})]

    api = FakeMessagesAPI(decide)
    holder.append(api)
    api.start()
    (tmp_path / "work").mkdir()
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store"),
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
    )
    try:
        out = await _pause_once(runner)
    finally:
        api.stop()
    assert out.is_error and "could not see this step's folder" in (out.error or "")
    assert runner.stub_calls == 0


def test_the_hook_reads_utf8_whatever_the_code_page(tmp_path: Path) -> None:
    """A Windows code page must not garble or reject a tool input that is not ASCII."""
    event = {
        "tool_name": PREFIX + "count",
        "tool_use_id": "toolu_1",
        "tool_input": {"note": "Łódź, zwróć 49,99 €"},
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    hook = Path(_runner.__file__).with_name("_defer_hook.py")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "TCA_HOOK_DIR": str(run_dir)}
    env.pop("PYTHONUTF8", None)
    done = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(event, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        env=env,
        check=True,
    )
    decision = json.loads(done.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "defer"
    assert (run_dir / "paused_call").read_text(encoding="utf-8") == "toolu_1"


# ---- FileSessionStore ----


def _key(project: str, session: str) -> SessionKey:
    return SessionKey(project_key=project, session_id=session)


def _entry(uid: str) -> Any:
    return {"type": "user", "uuid": uid}


async def test_the_file_store_keeps_long_keys_apart(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    first, second = _key("p" * 240, "session-a"), _key("p" * 240, "session-b")
    await store.append(first, [_entry("1")])
    await store.append(second, [_entry("2")])
    assert [e.get("uuid") for e in await store.load(first) or []] == ["1"]
    assert [e.get("uuid") for e in await store.load(second) or []] == ["2"]


async def test_the_file_store_ignores_a_line_still_being_written(
    tmp_path: Path,
) -> None:
    store = FileSessionStore(tmp_path)
    key = _key("project", "session")
    await store.append(key, [_entry("1")])
    (path,) = tmp_path.glob("*.jsonl")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "user", "uuid": "2", "message": {"conte')  # half a line
    assert [e.get("uuid") for e in await store.load(key) or []] == ["1"]


async def test_the_file_store_sees_another_workers_writes(tmp_path: Path) -> None:
    one, two = FileSessionStore(tmp_path), FileSessionStore(tmp_path)
    key = _key("project", "session")
    await one.append(key, [_entry("1")])
    await two.append(key, [_entry("2")])
    await one.append(key, [_entry("2"), _entry("3")])  # "2" is already stored
    assert [e.get("uuid") for e in await one.load(key) or []] == ["1", "2", "3"]


async def test_the_file_store_appends_after_a_half_written_line(
    tmp_path: Path,
) -> None:
    store = FileSessionStore(tmp_path)
    key = _key("project", "session")
    await store.append(key, [_entry("1")])
    (path,) = tmp_path.glob("*.jsonl")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "user", "uuid": "2", "message": {"conte')  # writer died
    await store.append(key, [_entry("3")])
    assert [e.get("uuid") for e in await store.load(key) or []] == ["1", "3"]
    assert path.read_text(encoding="utf-8").endswith("}\n")  # the half line is gone


@pytest.mark.parametrize("before", [[], ["1"], ["1", "2"]])
async def test_the_file_store_cuts_a_half_line_longer_than_one_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, before: list[str]
) -> None:
    """It looks back for the last newline a block at a time (small blocks here),
    and an empty file or a file that is all one half line are fine too."""
    monkeypatch.setattr(_session_store, "_BLOCK", 7)
    store = FileSessionStore(tmp_path)
    key = _key("project", "session")
    path = store._path(key)
    whole = "".join(json.dumps(_entry(uid)) + "\n" for uid in before)
    path.write_text(whole + '{"type": "user", "uuid": "9", "text": "' + "x" * 50)
    await store.append(key, [_entry("3")])
    assert [e.get("uuid") for e in await store.load(key) or []] == [*before, "3"]
    path.write_text("")
    await store.append(key, [_entry("4")])
    assert [e.get("uuid") for e in await store.load(key) or []] == ["4"]


async def test_the_file_store_writes_nothing_for_a_batch_it_has(
    tmp_path: Path,
) -> None:
    """Entries are told apart by their uuid; one without a uuid is always new."""
    store = FileSessionStore(tmp_path)
    key = _key("project", "session")
    summary: Any = {"type": "summary"}
    await store.append(key, [_entry("1"), summary])
    path = store._path(key)
    size = path.stat().st_size
    await store.append(key, [_entry("1")])
    assert path.stat().st_size == size  # nothing new, nothing written
    await store.append(key, [summary])
    loaded = await store.load(key) or []
    assert [e.get("uuid", e.get("type")) for e in loaded] == ["1", "summary", "summary"]
    assert await store.load(_key("project", "nobody")) is None


async def test_the_file_store_remembers_only_its_newest_transcripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcript it forgot is read again from its file: nothing is stored twice."""
    monkeypatch.setattr(_session_store, "_CACHED_FILES", 2)
    store = FileSessionStore(tmp_path)
    for session in ("a", "b", "c"):
        await store.append(_key("project", session), [_entry(session)])
    assert [path.name[0] for path in store._seen] == ["b", "c"]
    await store.append(_key("project", "a"), [_entry("a"), _entry("a2")])
    loaded = await store.load(_key("project", "a")) or []
    assert [e.get("uuid") for e in loaded] == ["a", "a2"]
    assert [path.name[0] for path in store._seen] == ["c", "a"]


class _DiskFillsUp:
    """``os`` for the store module, except that a write stops after 10 bytes."""

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    @staticmethod
    def write(fd: int, data: bytes) -> int:
        os.write(fd, bytes(data)[:10])
        raise OSError(errno.ENOSPC, "No space left on device")


async def test_the_file_store_writes_a_failed_batch_when_it_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileSessionStore(tmp_path)
    key = _key("project", "session")
    await store.append(key, [_entry("1")])
    with monkeypatch.context() as patch:
        patch.setattr(_session_store, "os", _DiskFillsUp())
        with pytest.raises(OSError):
            await store.append(key, [_entry("2"), _entry("3")])
    await store.append(key, [_entry("2"), _entry("3")])  # the SDK retries the batch
    assert [e.get("uuid") for e in await store.load(key) or []] == ["1", "2", "3"]


_DIE_IN_THE_MIDDLE_OF_AN_APPEND = """
import asyncio, os, sys
from temporalio.claude_agent_sdk import FileSessionStore, _session_store

class DiesHalfway:
    def __getattr__(self, name):
        return getattr(os, name)

    @staticmethod
    def write(fd, data):
        os.write(fd, bytes(data)[:10])
        os._exit(3)  # the Worker process dies in the middle of the write

_session_store.os = DiesHalfway()
key = {"project_key": "project", "session_id": "session"}
asyncio.run(FileSessionStore(sys.argv[1]).append(key, [{"type": "user", "uuid": "2"}]))
"""


async def test_the_file_store_recovers_from_a_worker_that_died_mid_append(
    tmp_path: Path,
) -> None:
    store = FileSessionStore(tmp_path)
    key = _key("project", "session")
    await store.append(key, [_entry("1")])
    died = subprocess.run(
        [sys.executable, "-c", _DIE_IN_THE_MIDDLE_OF_AN_APPEND, str(tmp_path)],
        timeout=60,
    )
    assert died.returncode == 3
    await store.append(key, [_entry("3")])
    assert [e.get("uuid") for e in await store.load(key) or []] == ["1", "3"]


async def test_a_stuck_transcript_lock_holds_up_only_that_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileSessionStore(tmp_path)
    stuck, other = _key("project", "stuck"), _key("project", "other")
    monkeypatch.setattr(_session_store, "_LOCK_WAIT", 3.0)
    with _session_store._locked(store._path(stuck)):  # a process stalled holding it
        waiting = asyncio.create_task(store.append(stuck, [_entry("1")]))
        await asyncio.sleep(0.2)  # that append now waits for the lock
        await asyncio.wait_for(store.append(other, [_entry("2")]), timeout=2.0)
        assert [e.get("uuid") for e in await store.load(other) or []] == ["2"]
        with pytest.raises(OSError) as raised:
            await waiting
        assert not isinstance(raised.value, TimeoutError)  # so the SDK retries it
    await store.append(stuck, [_entry("1")])  # the retry, once the lock is free
    assert [e.get("uuid") for e in await store.load(stuck) or []] == ["1"]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (  # the review's case: "/" and "_" became the same character
            SessionKey(
                project_key="p",
                session_id="s",
                subpath="subagents/workflows/run-1/agent-nested",
            ),
            SessionKey(
                project_key="p",
                session_id="s",
                subpath="subagents/workflows_run-1/agent-nested",
            ),
        ),
        (  # the "__" between the parts could belong to either part
            SessionKey(project_key="a__b", session_id="c"),
            SessionKey(project_key="a", session_id="b__c"),
        ),
        (  # a main transcript and a subagent transcript
            SessionKey(project_key="p", session_id="s__x"),
            SessionKey(project_key="p", session_id="s", subpath="x"),
        ),
        (  # letters outside ASCII became "_"
            SessionKey(project_key="p", session_id="zażółć"),
            SessionKey(project_key="p", session_id="za____"),
        ),
        (  # only the case differs: one file on macOS and Windows
            SessionKey(project_key="p", session_id="Session"),
            SessionKey(project_key="p", session_id="session"),
        ),
    ],
    ids=[
        "subpath-separator",
        "part-separator",
        "main-and-subagent",
        "non-ascii",
        "case",
    ],
)
async def test_the_file_store_keeps_keys_apart_whatever_their_characters(
    tmp_path: Path, first: SessionKey, second: SessionKey
) -> None:
    store = FileSessionStore(tmp_path)
    await store.append(first, [_entry("1")])
    await store.append(second, [_entry("2")])
    assert [e.get("uuid") for e in await store.load(first) or []] == ["1"]
    assert [e.get("uuid") for e in await store.load(second) or []] == ["2"]


# ---- real engine: background tasks, and built-in tools after a cancel ----

SUBTASK = "SUBTASK: write a note with the Write tool"


def _first_user_text(body: dict[str, Any]) -> str:
    for message in body.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        blocks = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )
        for block in blocks or []:
            text = str(block.get("text", "")) if block.get("type") == "text" else ""
            if text and not text.lstrip().startswith("<system-reminder>"):
                return text
    return ""


def _results(body: dict[str, Any]) -> list[tuple[str, bool]]:
    return [
        (str(block.get("tool_use_id")), bool(block.get("is_error")))
        for message in body.get("messages", [])
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]


async def test_a_background_subagent_does_not_keep_the_engine_running(
    tmp_path: Path,
) -> None:
    """Claude delegates a note to a subagent, then calls a durable tool itself."""
    work = tmp_path / "work"
    work.mkdir()
    model_calls: list[str] = []
    holder: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        api = holder[0]
        results = _results(body)
        if SUBTASK in _first_user_text(body):
            model_calls.append("subagent")
            if results:
                return [{"type": "text", "text": "note written"}]
            note = {"file_path": str(work / "note.txt"), "content": "x"}
            return [
                {
                    "type": "tool_use",
                    "id": api.next_id("toolu_write"),
                    "name": "Write",
                    "input": note,
                }
            ]
        model_calls.append("main")
        if not any(tid.startswith("toolu_task") for tid, _ in results):
            task = {
                "description": "write a note",
                "prompt": SUBTASK,
                "subagent_type": "general-purpose",
            }
            return [
                {
                    "type": "tool_use",
                    "id": api.next_id("toolu_task"),
                    "name": "Agent",
                    "input": task,
                }
            ]
        if not any(tid.startswith("toolu_fake") and not err for tid, err in results):
            return [api.tool_use("count", {"n": 1})]
        return [{"type": "text", "text": "FINAL"}]

    api = FakeMessagesAPI(decide)
    holder.append(api)
    api.start()
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store"),
        cwd=str(work),
        env=engine_env(api, str(tmp_path / "cfg")),
    )
    spec = ToolSpec("count", "Count one step.", {"type": "object"})
    builtin = ["Agent", "Write"]
    session = str(uuid.uuid4())
    try:
        first = await asyncio.wait_for(
            runner.run(
                SegmentInput(
                    session_id=session,
                    prompt="Delegate a note, then count.",
                    tools=[spec],
                    builtin_tools=builtin,
                ),
                1,
            ),
            90,
        )
        assert not first.is_error, first.error
        assert first.deferred is not None and first.deferred.name == "count"
        second = await runner.run(
            SegmentInput(
                session_id=first.session_id,
                prompt=None,
                tools=[spec],
                builtin_tools=builtin,
                checkpoint=first.checkpoint,
                injected={first.deferred.id: ToolOutcome({"n": 1})},
                segment_index=1,
            ),
            1,
        )
    finally:
        api.stop()
    assert second.result == "FINAL", second
    assert (work / "note.txt").exists() and "subagent" in model_calls
    assert len(model_calls) < 12, model_calls  # it looped hundreds of times before
    assert api.errors == [] and runner.stub_calls == 0


class _CancelSeen(logging.Handler):
    """Set when the Worker hands a cancellation to a running Activity."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.event = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("Cancelling activity"):
            self.event.set()


@pytest.fixture
def cancel_seen() -> Iterator[_CancelSeen]:
    handler = _CancelSeen()
    logger = logging.getLogger("temporalio.worker._activity")
    level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(level)


@pytest.mark.usefixtures("shop_dir")
async def test_a_cancelled_step_runs_no_more_builtin_tools(
    client: Client, tmp_path: Path, cancel_seen: _CancelSeen
) -> None:
    """The model answers with a Write only after the cancel reached the Activity."""
    work = tmp_path / "work"
    work.mkdir()
    target = work / "after_cancel.txt"
    arrived, release = threading.Event(), threading.Event()
    numbers = itertools.count(1)
    holder: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        del body
        if next(numbers) > 1:
            return [{"type": "text", "text": "done"}]
        arrived.set()
        release.wait(60)
        write = {"file_path": str(target), "content": "written after the cancel"}
        return [
            {
                "type": "tool_use",
                "id": holder[0].next_id("toolu_write"),
                "name": "Write",
                "input": write,
            }
        ]

    api = FakeMessagesAPI(decide)
    holder.append(api)
    api.start()
    hook_log = tmp_path / "hook.log"
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store"),
        cwd=str(work),
        env={**engine_env(api, str(tmp_path / "cfg")), "TCA_HOOK_LOG": str(hook_log)},
    )
    queue = f"cancel-{uuid.uuid4().hex[:8]}"
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[WriterWorkflow],
            activities=[count],
            plugins=[ClaudeAgentPlugin(runner, heartbeat_every=0.5)],
            **FAIL_FAST,
        ):
            handle = await client.start_workflow(
                WriterWorkflow.run, "count to 1", id=queue, task_queue=queue
            )
            assert await asyncio.to_thread(arrived.wait, 60)
            await handle.cancel()
            assert await asyncio.to_thread(cancel_seen.event.wait, 30)
            release.set()  # the engine gets the Write while the SDK shuts it down
            with pytest.raises(WorkflowFailureError) as failure:
                await handle.result()
            assert isinstance(failure.value.cause, CancelledError)
            await asyncio.sleep(7)  # longer than the SDK waits for the engine to exit
    finally:
        release.set()
        api.stop()
    assert not target.exists()
    decisions = hook_log.read_text().split("\n") if hook_log.exists() else []
    writes = [line for line in decisions if line.startswith("toolu_write")]
    assert all(line.endswith(" deny") for line in writes), decisions


# ---- heartbeats ----


@pytest.mark.usefixtures("shop_dir")
async def test_heartbeats_follow_the_segments_timeout(
    client: Client, tmp_path: Path
) -> None:
    """A Worker set to heartbeat every 30 s still beats often enough for a 2 s timeout."""
    queue = f"beat-{uuid.uuid4().hex[:8]}"
    runner = ScriptedClaude(count_policy, tmp_path / "fake", think_seconds=4)
    async with Worker(
        client,
        task_queue=queue,
        workflows=[WriterWorkflow],
        activities=[count],
        plugins=[ClaudeAgentPlugin(runner, heartbeat_every=30)],
        **FAIL_FAST,
    ):
        answer = await client.execute_workflow(
            WriterWorkflow.run, "count to 1", id=queue, task_queue=queue
        )
    assert answer == "counted to 1"


@pytest.mark.parametrize(
    ("timeout", "every", "gap"),
    [(None, 0.1, 0.1), (0.3, 30.0, 0.1)],
    ids=["no-heartbeat-timeout", "three-per-timeout"],
)
async def test_steps_heartbeat_while_they_run(
    timeout: float | None, every: float, gap: float
) -> None:
    """Every ``heartbeat_every`` seconds, or three times per heartbeat timeout when
    that is shorter: a segment with its index, a tool step with its call's id. The
    typical gap is checked (the median), so a slow machine cannot fail it, but twice
    the gap would."""
    beats: list[tuple[float, Any]] = []
    env = ActivityEnvironment()
    env.on_heartbeat = lambda *details: beats.append((time.monotonic(), details[0]))
    if timeout is not None:
        env.info = dataclasses.replace(
            env.info, heartbeat_timeout=timedelta(seconds=timeout)
        )

    async def slow(_: dict[str, Any]) -> str:
        await asyncio.sleep(1.0)
        return "done"

    def median_gap() -> float:
        times = [t for t, _ in beats]
        gaps = sorted(b - a for a, b in itertools.pairwise(times))
        return gaps[len(gaps) // 2]

    runner = ScriptedClaude(
        count_policy, think_seconds=1.0, engine_tools={"Bash": slow}
    )
    segment = make_segment_activity(runner, heartbeat_every=every)
    inp = SegmentInput(
        session_id="s", prompt="count to 1", tools=[], transcript=[], segment_index=4
    )
    out = await env.run(segment, inp)
    assert out.deferred is not None and len(beats) >= 5
    assert {d for _, d in beats} == {4}
    assert median_gap() < gap * 1.5
    beats.clear()
    step = make_tool_step_activity(runner, heartbeat_every=every)
    call = DeferredCall(id="toolu_7", name="Bash", input={}, kind="engine")
    outcome = await env.run(
        step, ToolStepInput(session_id="s", checkpoint="c", call=call)
    )
    assert outcome.content == "done" and len(beats) >= 5
    assert {d for _, d in beats} == {"toolu_7"}
    assert median_gap() < gap * 1.5


# ---- live output ----


async def test_two_agents_cannot_both_use_live_output(
    client: Client, tmp_path: Path
) -> None:
    queue = f"two-live-{uuid.uuid4().hex[:8]}"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[TwoLiveAgentsWorkflow],
        activities=[count],
        plugins=[ClaudeAgentPlugin(ScriptedClaude(count_policy, tmp_path / "fake"))],
    ):
        handle = await client.start_workflow(
            TwoLiveAgentsWorkflow.run, "count to 1", id=queue, task_queue=queue
        )
        message = ""
        for _ in range(100):
            for event in (await handle.fetch_history()).events:
                if event.HasField("workflow_task_failed_event_attributes"):
                    message = (
                        event.workflow_task_failed_event_attributes.failure.message
                    )
            if message:
                break
            await asyncio.sleep(0.2)
        await handle.terminate()
    with warnings.catch_warnings():  # the failed initialization never ran the Workflow
        warnings.simplefilter("ignore", RuntimeWarning)
        gc.collect()
    assert "Only one DurableClaudeAgent per Workflow can use live_output" in message
