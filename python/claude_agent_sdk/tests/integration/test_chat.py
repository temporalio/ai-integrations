"""ClaudeSDKClient in a Workflow, with the bundled Claude Code in a sandbox that can be lost."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from temporalio import activity
from temporalio.api.enums.v1 import EventType
from temporalio.claude_agent_sdk import (
    ExecResult,
    Sandbox,
)
from temporalio.claude_agent_sdk._activities import (
    SandboxActivities,
)
from temporalio.claude_agent_sdk._payloads import (
    WATCH_OUTPUT,
    LaunchRequest,
    SandboxSession,
    StdinWrite,
)
from temporalio.claude_agent_sdk._sandbox_cli import SandboxCli
from temporalio.client import (
    WorkflowFailureError,
)
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment
from temporalio.worker import (
    ActivityInboundInterceptor,
    ExecuteActivityInput,
    Interceptor,
)
from tests.helpers.harness import (
    ANSWERS,
    CLAUDE,
    WRITE_IT_MESSAGES,
    Hang,
    Harness,
    example_chat,
    say,
)
from tests.helpers.workflows import (
    EndInputWorkflow,
    Hook,
    OneShotWorkflow,
    bash,
    tool_use,
    until,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not CLAUDE.is_file(),
    reason="needs sh and the Claude Agent SDK's bundled Claude Code",
)

# What Claude Code tells the model about a call an interrupt stopped.
REJECTED = (
    "The user doesn't want to proceed with this tool use. The tool use was rejected "
    "(eg. if it was a file edit, the new_string was NOT written to the file). STOP "
    "what you are doing and wait for the user to tell you how to proceed."
)


def tool_results(messages: list[Any]) -> list[Any]:
    return [block for m in messages if m[0] == "UserMessage" for block in m[1:]]


def asked_about(messages: list[Any]) -> list[str]:
    return [m[1] for m in messages if m[0] == "Ask"]


def results(messages: list[Any]) -> list[Any]:
    """Each turn's result: its text, and its subtype if it failed."""
    return [m[1:] for m in messages if m[0] == "Result"]


# ANSWERS and "two". "two" first: a turn after an interrupted one has the tool's result too.
WRITE_IT_AND_TWO = {"two": "Two.", **ANSWERS}


def notes(harness: Harness) -> Path:
    """notes.txt in the workspace."""
    return Path(harness.sandbox.workspace) / "notes.txt"


def conversation(request: dict[str, Any]) -> list[Any]:
    """The prompts, tool calls, tool results, and answers a model request carries."""
    seen: list[Any] = []
    for message in request["messages"]:
        content = message["content"]
        blocks = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )
        for block in blocks:
            if block["type"] == "tool_use":
                seen.append(["tool_use", block["id"]])
            elif block["type"] == "tool_result":
                seen.append(["tool_result", block["tool_use_id"]])
            elif (
                block["type"] == "text"
                and message["role"] != "system"
                and not block["text"].startswith("<")
            ):
                seen.append([message["role"], block["text"]])
    return seen


def intercept(
    harness: Harness,
    around: Callable[
        [ExecuteActivityInput, Callable[[], Awaitable[Any]]], Awaitable[Any]
    ],
) -> None:
    """Run each Activity of the harness's Workers through ``around``, which runs it with its second argument."""

    class Inbound(ActivityInboundInterceptor):
        async def execute_activity(self, input: ExecuteActivityInput) -> Any:
            return await around(input, lambda: self.next.execute_activity(input))

    class Intercept(Interceptor):
        def intercept_activity(
            self, next: ActivityInboundInterceptor
        ) -> ActivityInboundInterceptor:
            return Inbound(next)

    harness.interceptors.append(Intercept())


async def test_each_prompt_is_a_turn_of_one_conversation(harness: Harness) -> None:
    answers = {**ANSWERS, "bye": "Bye."}

    messages = await harness.run(answers, ["write it", "bye"])

    # Each of the CLI's messages reaches the Claude Agent SDK once.
    assert [m for m in messages if m[0] != "Ask"] == [
        *WRITE_IT_MESSAGES,
        ["AssistantMessage", ["Bye."]],
        ["Result", "Bye."],
    ]
    assert harness.asked == ["write it", "tool_result:toolu_1", "bye"]
    assert conversation(harness.requests[-1]) == [
        ["user", "write it"],
        ["tool_use", "toolu_1"],
        ["tool_result", "toolu_1"],
        ["assistant", "Wrote it."],
        ["user", "bye"],
    ]


async def test_a_denied_tool_does_not_run_and_the_model_reads_why(
    harness: Harness,
) -> None:
    messages = await harness.run(WRITE_IT_AND_TWO, ["write it"], approve=False)

    assert asked_about(messages) == ["toolu_1"]
    assert tool_results(messages) == [["toolu_1", "A person denied it", True]]
    assert not notes(harness).exists()


async def test_a_lost_watch_of_the_output_goes_on_where_it_was(
    harness: Harness,
) -> None:
    lost: list[str] = []

    async def lose_the_first_watch(
        _watch: ExecuteActivityInput, execute: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Fail the first watch of the CLI's output, as if its Worker were lost, once the API is asked."""
        if activity.info().activity_type != WATCH_OUTPUT or lost:
            return await execute()
        lost.append("watch")
        watching = asyncio.ensure_future(execute())
        await until(lambda: harness.asked)
        await asyncio.sleep(1.5)  # Past a flush of the output.
        watching.cancel()
        await asyncio.wait([watching])
        raise RuntimeError("The Worker was lost")

    intercept(harness, lose_the_first_watch)

    messages = await harness.run(ANSWERS, ["write it"], decide_after=3)

    assert lost
    assert [m for m in messages if m[0] != "Ask"] == WRITE_IT_MESSAGES
    # Nothing started again.
    assert harness.asked == ["write it", "tool_result:toolu_1"]
    assert len(harness.cli_pids()) == 1


async def test_the_agent_may_not_write_the_session_transcripts(
    harness: Harness,
) -> None:
    """It may read them, and write the config dir's other files, such as memory."""
    config = harness.sandbox.config
    memory = config / "projects" / "key" / "memory" / "x.md"

    def transcript() -> Path:
        # Claude Code writes it a little after its stdout.
        while not (found := list(config.glob("projects/*/*.jsonl"))):
            time.sleep(0.05)
        return found[0]

    def write(tool_use_id: str, path: Path) -> dict[str, Any]:
        return tool_use(tool_use_id, "Write", file_path=str(path), content="x")

    answers = {
        "protect": lambda _: tool_use("toolu_1", "Read", file_path=str(transcript())),
        "tool_result:toolu_1": lambda _: write("toolu_2", transcript()),
        "tool_result:toolu_2": write("toolu_3", memory),
        "tool_result:toolu_3": "Done.",
    }

    messages = await harness.run(
        answers, ["protect"], allowed_tools=["Read", "Write", "Edit"]
    )

    errors = {
        tool_use_id: bool(is_error)
        for tool_use_id, _, is_error in tool_results(messages)
    }
    assert errors == {"toolu_1": False, "toolu_2": True, "toolu_3": False}
    assert memory.read_text() == "x"


async def test_a_cli_that_cannot_run_fails_the_workflow(harness: Harness) -> None:
    harness.sandbox.set_cli("echo bad flag >&2; exit 2")

    with pytest.raises(WorkflowFailureError) as raised:
        await harness.run({}, ["one"])

    error: BaseException | None = raised.value.cause
    while error is not None and not isinstance(error, ApplicationError):
        error = error.__cause__
    assert isinstance(error, ApplicationError)
    assert (error.type, error.non_retryable) == ("ClaudeCodeExited", True)
    assert "bad flag" in error.message
    # The sandbox started for it is ended.
    assert not harness.sandbox.path.exists()


class Activities:
    """The Activities that launch a CLI and write its stdin, run as ActivityEnvironment runs them."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.activities = SandboxActivities({"local": lambda _: sandbox})
        self.environment = ActivityEnvironment()

    async def launch(self, request: LaunchRequest) -> SandboxSession:
        return await self.environment.run(self.activities.launch, request)

    async def write_stdin(self, write: StdinWrite) -> None:
        await self.environment.run(self.activities.write_stdin, write)


async def started_cli(
    harness: Harness, cli: str = "exec cat"
) -> tuple[Activities, LaunchRequest]:
    """Activities on the harness's sandbox, and the request to launch a CLI that runs ``cli``: by default, one that echoes its stdin."""
    sandbox = harness.sandbox
    sandbox.set_cli(cli)
    await sandbox.start(env={})
    session = SandboxSession("local", "")
    return Activities(sandbox), LaunchRequest(session, [])


def running(text: str) -> list[str]:
    """The processes whose arguments have ``text``."""
    ps = subprocess.run(
        ["ps", "-A", "-o", "args="], capture_output=True, text=True, check=True
    )
    return [line for line in ps.stdout.splitlines() if text in line]


async def test_a_retried_launch_or_write_starts_and_writes_once(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = harness.sandbox
    activities, launch = await started_cli(harness)

    session = await activities.launch(launch)
    assert await activities.launch(launch) == session
    first = '{"n": 1}\n{"n": 2}\n'
    write = StdinWrite(session, 1, first)
    await activities.write_stdin(write)
    await activities.write_stdin(write)
    stdin = Path(sandbox.files.cli.stdin)
    assert (stdin / "1.jsonl").read_text() == first
    # A write that fails is retried, and writes the same file.
    write_files = sandbox.write_files

    async def fail_once(files: Mapping[str, tuple[bytes, int]], **kwargs: Any) -> None:
        del files, kwargs
        monkeypatch.setattr(sandbox, "write_files", write_files)
        raise RuntimeError("The connection to the sandbox broke")

    monkeypatch.setattr(sandbox, "write_files", fail_once)
    write = StdinWrite(session, 2, '{"n": 3}\n')
    with pytest.raises(RuntimeError, match="broke"):
        await activities.write_stdin(write)
    await activities.write_stdin(write)
    # Attempts that run together, as a retry can with an earlier attempt that
    # outlived it, write the file whole.
    write = StdinWrite(session, 3, '{"n": 4}\n')
    await asyncio.gather(*(activities.write_stdin(write) for _ in range(5)))
    assert sorted(path.name for path in stdin.iterdir()) == [
        "1.jsonl",
        "2.jsonl",
        "3.jsonl",
    ]
    assert (stdin / "3.jsonl").read_text() == write.text

    # The CLI echoes its stdin: each line, once, in the order of the files.
    stdout = Path(sandbox.files.cli.stdout)
    await until(lambda: len(stdout.read_text().splitlines()) >= 4)
    await asyncio.sleep(0.5)
    lines = stdout.read_text().splitlines()
    assert lines == ['{"n": 1}', '{"n": 2}', '{"n": 3}', '{"n": 4}']
    assert len(harness.cli_pids()) == 1
    # A retry after the CLI exited leaves the exit to watching its output.
    await harness.kill_cli_and_wait()
    assert await activities.launch(launch) == session
    # A write after the CLI exited is no error: watching its output finds the exit.
    await activities.write_stdin(StdinWrite(session, 4, "{}\n"))
    # Nothing that fed the CLI's stdin is left running.
    await until(lambda: not running(sandbox.files.cli.dir))


async def test_a_failed_exec_is_retried_and_not_taken_for_the_clis_exit(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exec can fail on its own, such as when the connection to the sandbox breaks."""
    sandbox = harness.sandbox
    activities, launch = await started_cli(harness)
    exec = sandbox.exec

    async def fail(argv: Sequence[str], **kwargs: Any) -> ExecResult:
        del argv, kwargs
        return ExecResult(1, b"", b"connection reset")

    monkeypatch.setattr(sandbox, "exec", fail)
    # A plain exception, which the Activity retries: not ClaudeCodeExited.
    with pytest.raises(RuntimeError, match="connection reset"):
        await activities.launch(launch)
    monkeypatch.setattr(sandbox, "exec", exec)
    session = await activities.launch(launch)
    cli = SandboxCli(sandbox, session)
    monkeypatch.setattr(sandbox, "exec", fail)
    # No exit is recorded, so watching the output is retried.
    assert await cli.exit_status() is None
    monkeypatch.setattr(sandbox, "exec", exec)
    await harness.kill_cli_and_wait()
    assert await cli.exit_status() == 137


async def test_the_example_chat_answers_each_message(harness: Harness) -> None:
    harness.answers = ANSWERS
    async with example_chat(harness.client, harness.factory) as handle:
        assert await say(handle, "write it") == "Wrote it."
        assert await say(handle, "read it") == "Read it."
        assert notes(harness).read_text() == "hi\n"
        assert harness.asked == [*ANSWERS]


async def test_a_denial_that_interrupts_ends_the_turn(harness: Harness) -> None:
    messages = await harness.run(
        WRITE_IT_AND_TWO, ["write it", "two"], approve=False, deny_interrupt=True
    )

    assert asked_about(messages) == ["toolu_1"]
    assert results(messages) == [[None, "error_during_execution"], ["Two."]]
    assert harness.asked == ["write it", "two"]
    assert not notes(harness).exists()


def delegate(harness: Harness) -> dict[str, Any]:
    """Answers for "delegate", which runs a foreground subagent whose tool call writes ``ran``."""
    agent = tool_use(
        "toolu_a",
        "Agent",
        description="Write",
        prompt="SUBTASK",
        subagent_type="general-purpose",
        run_in_background=False,
    )
    return {
        "delegate": agent,
        "SUBTASK": bash("toolu_b", f"echo x >> {harness.root / 'ran'}"),
        "tool_result:toolu_b": "Not asked.",
        "tool_result:toolu_a": "Stopped.",
    }


def agent_id(body: dict[str, Any]) -> str:
    """The subagent ID in the request's last tool result, from the Agent call."""
    result = [m for m in body["messages"] if m["role"] == "user"][-1]
    found = re.search(r"agentId: (\w+)", json.dumps(result))
    assert found is not None
    return found.group(1)


async def test_a_denial_that_interrupts_in_a_subagent_stops_the_subagent(
    harness: Harness,
) -> None:
    messages = await harness.run(
        delegate(harness),
        ["delegate"],
        approve=False,
        deny_interrupt=True,
        tools=["Agent", "Bash"],
        allowed_tools=["Agent"],
    )

    assert asked_about(messages) == ["toolu_b"]
    assert not (harness.root / "ran").exists()
    # The subagent asked the model nothing after the denied call; the main turn went on.
    assert harness.asked == ["delegate", "SUBTASK", "tool_result:toolu_a"]
    [[_, _, is_error]] = [b for b in tool_results(messages) if b[0] == "toolu_a"]
    assert is_error
    assert messages[-1] == ["Result", "Stopped."]
    assert len(harness.cli_pids()) == 1


async def test_a_subagents_later_turn_gets_its_own_steps(harness: Harness) -> None:
    """SendMessage resumes a finished subagent for a second turn of its own."""

    def resume(body: dict[str, Any]) -> dict[str, Any]:
        # Claude Code writes the subagent's transcript, which SendMessage reads,
        # a little after its result.
        time.sleep(0.5)
        message = {"to": agent_id(body), "message": "AGAIN", "summary": "Again"}
        return tool_use("toolu_m", "SendMessage", **message)

    answers = {
        **delegate(harness),
        "AGAIN": "Sub two.",
        "SUBTASK": "Sub one.",
        "tool_result:toolu_a": resume,
        "tool_result:toolu_m": "Done.",
    }

    messages = await harness.run(
        answers, ["delegate"], allowed_tools=["Agent", "SendMessage"]
    )

    # The subagent's second turn asked the model.
    assert ["AssistantMessage", ["Sub two."]] in messages


async def test_the_models_task_stop_is_recorded_so_the_workflow_stops_waiting(
    harness: Harness,
) -> None:
    """The model stops a background subagent while its tool call waits on a person."""

    def stop(body: dict[str, Any]) -> Hang:
        return Hang(tool_use("toolu_s", "TaskStop", task_id=agent_id(body)))

    answers = delegate(harness)
    answers["delegate"]["input"]["run_in_background"] = True
    answers |= {"tool_result:toolu_a": stop, "tool_result:toolu_s": "Stopped."}

    messages = await harness.run(
        answers,
        ["delegate"],
        decide_after=600,
        release_when=lambda: "toolu_b" in harness.pending,
        allowed_tools=["Agent", "TaskStop"],
    )

    assert asked_about(messages) == ["toolu_b"]
    assert messages[-1] == ["Result", "Stopped."]
    assert not (harness.root / "ran").exists()


async def test_an_interrupt_while_a_person_decides_ends_the_turn(
    harness: Harness,
) -> None:
    messages = await harness.run(
        WRITE_IT_AND_TWO,
        ["write it", "two"],
        decide_after=3600,
        interrupt_when=lambda: harness.pending,
    )

    assert [
        m for m in messages if m[0] != "AssistantMessage" or m[1][0] != "toolu_1"
    ] == [
        ["Ask", "toolu_1"],
        ["Cancelled", "toolu_1"],
        ["UserMessage", ["toolu_1", REJECTED, True]],
        ["UserMessage", ["[Request interrupted by user for tool use]"]],
        ["Result", None, "error_during_execution"],
        ["AssistantMessage", ["Two."]],
        ["Result", "Two."],
    ]
    assert harness.asked == ["write it", "two"]


async def test_ending_the_input_stops_the_cli_once_its_prompts_are_done(
    harness: Harness,
) -> None:
    harness.answers = {"one": Hang()}
    async with harness.start(EndInputWorkflow, ["one"]) as handle:
        await until(lambda: "one" in harness.asked)
        await asyncio.sleep(1)
        sandbox = harness.sandbox.path
        assert sandbox.exists()
        harness.release.set()  # "one" is answered, then the CLI stops.
        await until(lambda: not sandbox.exists())
        await handle.signal(EndInputWorkflow.close)
        kinds = await handle.result()

    assert kinds == ["assistant", "result"]
    assert harness.asked == ["one"]


def decision(behavior: str, reason: str | None = None) -> dict[str, Any]:
    """A PreToolUse hook's output with a permission decision."""
    output = {"hookEventName": "PreToolUse", "permissionDecision": behavior}
    if reason is not None:
        output["permissionDecisionReason"] = reason
    return {"hookSpecificOutput": output}


async def test_a_hook_deny_refuses_the_call_without_asking(harness: Harness) -> None:
    answers = WRITE_IT_AND_TWO
    hooks = [Hook("PreToolUse", decision("deny", "No writes"), matcher="Bash")]

    messages = await harness.run(answers, ["write it"], hooks=hooks)

    assert asked_about(messages) == []
    denied = "PreToolUse:Bash hook error: No writes"
    assert tool_results(messages) == [["toolu_1", denied, True]]
    assert not notes(harness).exists()
    [hook_input] = harness.hook_inputs
    assert hook_input["tool_use_id"] == "toolu_1"


async def test_a_hook_that_raised_has_no_opinion(harness: Harness) -> None:
    hook = Hook("PreToolUse", decision("deny", "No."), fail=True)

    messages = await harness.run(WRITE_IT_AND_TWO, ["write it"], hooks=[hook])

    # The call is asked about, as if there were no hook.
    assert asked_about(messages) == ["toolu_1"]
    assert notes(harness).exists()


async def test_a_hook_out_of_time_stops_the_call(harness: Harness) -> None:
    hook = Hook("PreToolUse", decision("allow"), wait=5, timeout=1)

    messages = await harness.run(WRITE_IT_AND_TWO, ["write it"], hooks=[hook])

    # As in Claude Code, the call does not run.
    assert asked_about(messages) == []
    [[_, text, is_error]] = tool_results(messages)
    assert is_error
    assert "did not respond before its timeout" in text
    assert not notes(harness).exists()


async def test_an_interrupt_while_a_hook_runs_ends_the_turn(harness: Harness) -> None:
    hooks = [Hook("PreToolUse", decision("allow"), wait=3600)]

    messages = await harness.run(
        WRITE_IT_AND_TWO,
        ["write it", "two"],
        interrupt_when=lambda: harness.hook_inputs,
        hooks=hooks,
    )

    assert results(messages) == [[None, "error_during_execution"], ["Two."]]
    assert not notes(harness).exists()
    assert len(harness.cli_pids()) == 1


async def test_an_sdk_mcp_tool_runs_once_in_the_workflow(harness: Harness) -> None:
    answers = {
        "echo it": tool_use("toolu_1", "mcp__app__echo", text="hi"),
        "tool_result:toolu_1": "Echoed.",
    }

    messages = await harness.run(answers, ["echo it"], sdk_tools=True)

    # It is asked about as Claude Code asks about an MCP tool.
    assert [m for m in messages if m[0] in ("Ask", "Echo")] == [
        ["Ask", "toolu_1"],
        ["Echo", "hi"],
    ]
    assert tool_results(messages) == [
        ["toolu_1", [{"type": "text", "text": "echo: hi"}], None]
    ]
    assert messages[-1] == ["Result", "Echoed."]
    assert harness.asked == ["echo it", "tool_result:toolu_1"]


async def test_each_line_of_output_is_a_signal_of_its_own(harness: Harness) -> None:
    messages = await harness.run(ANSWERS, ["write it"])

    assert [m for m in messages if m[0] != "Ask"] == WRITE_IT_MESSAGES
    assert harness.handle is not None
    history = await harness.handle.fetch_history()
    signals = [
        json.loads(
            e.workflow_execution_signaled_event_attributes.input.payloads[0].data
        )
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_SIGNALED
    ]
    outputs = [s for s in signals if isinstance(s, dict) and "line" in s]
    assert outputs
    assert all(json.loads(output["line"])["type"] for output in outputs)
    # Each starts where the one before it ended.
    assert [o["start"] for o in outputs[1:]] == [o["end"] for o in outputs[:-1]]


def background(command: str) -> dict[str, Any]:
    """Answers for "start it", which runs ``command`` in the background; its end wakes the agent."""
    return {
        # First: the harness answers with the first key the last user message
        # has, and the notification's text has the other keys too.
        "<task-notification>": "Finished.",
        "start it": tool_use(
            "toolu_1", "Bash", command=command, run_in_background=True
        ),
        "tool_result:toolu_1": "Started.",
    }


async def test_the_sdks_query_function_ends_once_claude_code_is_done(
    harness: Harness,
) -> None:
    """As without Temporal, a background task finishes first, and its end starts another turn."""
    messages = await harness.execute(
        OneShotWorkflow, "start it", ["Bash"], answers=background("sleep 2")
    )

    assert results(messages) == [["Started."], ["Finished."]]


async def test_a_turn_claude_code_starts_on_its_own_while_idle_is_watched(
    harness: Harness,
) -> None:
    """A background task's end starts a turn, whose tool call asks the Workflow."""
    answers = {
        **background("sleep 2"),
        "<task-notification>": bash("toolu_2", "echo hi > notes.txt"),
        "tool_result:toolu_2": "Finished.",
    }

    messages = await harness.run(answers, ["start it"], follow_ups=1)

    assert asked_about(messages) == ["toolu_2"]
    assert results(messages) == [["Started."], ["Finished."]]
    assert len(harness.cli_pids()) == 1
