"""ClaudeSDKClient in a Workflow, with the bundled Claude Code on the Worker."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from temporalio.api.common.v1 import WorkflowExecution
from temporalio.api.enums.v1 import EventType
from temporalio.api.workflowservice.v1 import ResetWorkflowExecutionRequest
from temporalio.claude_agent_sdk import ClaudeAgentSDKPlugin
from temporalio.client import WorkflowFailureError, WorkflowHandle
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.exceptions import ApplicationError, CancelledError
from temporalio.worker import Replayer, Worker
from tests.helpers.harness import (
    ANSWERS,
    CLAUDE,
    WRITE_IT_MESSAGES,
    Harness,
    example_chat,
    say,
)
from tests.helpers.workflows import (
    Chat,
    ChatWorkflow,
    CliRunWorkflow,
    ForkingWorkflow,
    Hook,
    OneShotWorkflow,
    ResumingWorkflow,
    StreamingWorkflow,
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
    return harness.workspace / "notes.txt"


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
    # One CLI serves both turns.
    assert len(harness.cli_pids()) == 1


async def test_a_denied_tool_does_not_run_and_the_model_reads_why(
    harness: Harness,
) -> None:
    messages = await harness.run(ANSWERS, ["write it"], approve=False)

    assert asked_about(messages) == ["toolu_1"]
    assert tool_results(messages) == [["toolu_1", "A person denied it", True]]
    assert not notes(harness).exists()


async def until_a_person_is_asked(handle: WorkflowHandle[Any, Any]) -> None:
    """Wait until the chat waits on a person to decide a tool call."""
    while not await handle.query(ChatWorkflow.pending_approvals):
        await asyncio.sleep(0.05)


async def chat_until_a_worker_is_lost(
    harness: Harness, chat: Chat, plugin: ClaudeAgentSDKPlugin
) -> WorkflowHandle[Any, Any]:
    """Start the chat on a Worker with ``plugin``, and shut that Worker down once a tool call waits for a person."""
    task_queue = str(uuid.uuid4())
    async with Worker(
        harness.client,
        task_queue=task_queue,
        workflows=[ChatWorkflow],
        plugins=[plugin],
    ):
        handle = await harness.client.start_workflow(
            ChatWorkflow.run, chat, id=task_queue, task_queue=task_queue
        )
        await until_a_person_is_asked(handle)
    return handle


# The next Activity waits 30 seconds for the lost Worker, and so does stopping Claude Code.
@pytest.mark.timeout(150)
async def test_a_workflow_whose_worker_died_fails_with_a_way_to_resume(
    harness: Harness,
) -> None:
    harness.answers = ANSWERS
    chat = Chat(["write it"], str(harness.cli_path))
    handle = await chat_until_a_worker_is_lost(harness, chat, ClaudeAgentSDKPlugin())

    plugin = ClaudeAgentSDKPlugin()
    async with Worker(
        harness.client,
        task_queue=handle.id,
        workflows=[ChatWorkflow],
        plugins=[plugin],
    ):
        with pytest.raises(WorkflowFailureError) as raised:
            await handle.result()
    error = raised.value.cause
    assert isinstance(error, ApplicationError)
    assert error.type == "CLIConnectionError"
    assert "options.resume" in error.message
    replayer = Replayer(workflows=[ChatWorkflow], plugins=[plugin])
    await replayer.replay_workflow(await handle.fetch_history())


# The next Activity waits 30 seconds for the lost Worker, and so does stopping Claude Code.
@pytest.mark.timeout(150)
async def test_a_workflow_resumes_the_session_its_lost_worker_cut_off_mid_turn(
    harness: Harness,
) -> None:
    harness.answers = ANSWERS
    task_queue = str(uuid.uuid4())
    args = ["write it", str(harness.cli_path)]
    async with Worker(
        harness.client,
        task_queue=task_queue,
        workflows=[ResumingWorkflow],
        plugins=[ClaudeAgentSDKPlugin()],
    ):
        handle = await harness.client.start_workflow(
            ResumingWorkflow.run, args=args, id=task_queue, task_queue=task_queue
        )
        while not await handle.query(ResumingWorkflow.pending_approvals):
            await asyncio.sleep(0.05)

    async with Worker(
        harness.client,
        task_queue=task_queue,
        workflows=[ResumingWorkflow],
        plugins=[ClaudeAgentSDKPlugin()],
    ):
        # The resumed Claude Code re-runs the turn, so it asks about the tool call again.
        while await handle.query(ResumingWorkflow.launch_count) < 2 or not (
            await handle.query(ResumingWorkflow.pending_approvals)
        ):
            await asyncio.sleep(0.1)
        await handle.execute_update(ResumingWorkflow.decide, "toolu_1")
        assert await handle.result() == "Wrote it."

    assert notes(harness).read_text() == "hi\n"
    assert harness.asked.count("write it") == 2


async def test_a_cli_that_cannot_run_fails_the_workflow(harness: Harness) -> None:
    harness.set_cli("echo bad flag >&2; exit 2")

    with pytest.raises(WorkflowFailureError) as raised:
        await harness.run({}, ["one"])

    error = raised.value.cause
    assert isinstance(error, ApplicationError)
    # As in the Claude Agent SDK, writing the prompt can see the exit first.
    assert error.type in ("ProcessError", "CLIConnectionError")
    assert "exit code: 2" in error.message
    assert harness.running() == []


async def test_stderr_and_settings_reach_claude_code_as_in_the_sdk(
    harness: Harness,
) -> None:
    """options.stderr gets each line: without trailing whitespace, blank lines skipped, and the last one even without a newline. options.sandbox merges into options.settings."""
    settings = json.dumps({"permissions": {"deny": ["Bash"]}})
    harness.set_cli(
        f'printf "%s\\n" "$@" > "{harness.root}/args"; '
        "echo one >&2; echo >&2; sleep 0.5; printf 'two  ' >&2"
    )

    async with harness.start(CliRunWorkflow, str(harness.cli_path), settings) as handle:
        lines = await handle.result()

    assert lines == ["one", "two"]
    args = (harness.root / "args").read_text().splitlines()
    assert json.loads(args[args.index("--settings") + 1]) == {
        "permissions": {"deny": ["Bash"]},
        "sandbox": {"enabled": True},
    }


async def test_the_example_chat_answers_each_message(harness: Harness) -> None:
    harness.answers = ANSWERS
    async with example_chat(harness.client) as handle:
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


async def test_cancelling_the_workflow_while_a_person_decides_stops_claude_code(
    harness: Harness,
) -> None:
    harness.answers = ANSWERS
    chat = Chat(["write it"], str(harness.cli_path))
    async with harness.start(ChatWorkflow, chat) as handle:
        await until(lambda: harness.asked)
        await until_a_person_is_asked(handle)
        assert len(harness.running()) == 1

        await handle.cancel()

        with pytest.raises(WorkflowFailureError) as raised:
            await handle.result()
    assert isinstance(raised.value.cause, CancelledError)

    history = await handle.fetch_history()
    assert (
        history.events[-1].event_type
        == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_CANCELED
    )
    stops = [
        event
        for event in history.events
        if event.activity_task_scheduled_event_attributes.activity_type.name
        == "claude-agent-sdk-close"
    ]
    assert len(stops) == 1
    assert harness.running() == []
    assert not notes(harness).exists()


async def reset(
    harness: Harness, handle: WorkflowHandle[Any, Any], task: int
) -> WorkflowHandle[Any, Any]:
    """Reset the Workflow to the end of its Workflow Task number ``task``, from 0, and return its new run."""
    finished = [
        event.event_id
        async for event in handle.fetch_history_events()
        if event.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_COMPLETED
    ]
    response = await harness.client.workflow_service.reset_workflow_execution(
        ResetWorkflowExecutionRequest(
            namespace=harness.client.namespace,
            workflow_execution=WorkflowExecution(workflow_id=handle.id),
            reason="test",
            workflow_task_finish_event_id=finished[task],
            request_id=str(uuid.uuid4()),
        )
    )
    return harness.client.get_workflow_handle(handle.id, run_id=response.run_id)


async def test_a_workflow_reset_to_its_start_runs_a_new_claude_code(
    harness: Harness,
) -> None:
    """Such as one reset while a person decides: the new run asks again."""
    harness.answers = ANSWERS
    chat = Chat(["write it"], str(harness.cli_path))
    async with harness.start(ChatWorkflow, chat) as handle:
        await until_a_person_is_asked(handle)

        new_run = await reset(harness, handle, 0)

        await until_a_person_is_asked(new_run)
    assert len(harness.cli_pids()) == 2


async def test_a_workflow_reset_after_its_claude_code_closed_fails_with_a_way_to_resume(
    harness: Harness,
) -> None:
    """Such as a finished chat reset to its first turn, whose Claude Code is gone."""
    harness.answers = {"hi": "Hello."}
    chat = Chat(["hi"], str(harness.cli_path))
    async with harness.start(ChatWorkflow, chat) as handle:
        await handle.result()

        # The second Workflow Task, after the launch.
        new_run = await reset(harness, handle, 1)

        with pytest.raises(WorkflowFailureError) as raised:
            await new_run.result()
    error = raised.value.cause
    assert isinstance(error, ApplicationError)
    assert error.type == "CLIConnectionError"
    assert "options.resume" in error.message


async def test_a_fork_continues_a_session_under_a_new_id_and_leaves_it_as_it_was(
    harness: Harness,
) -> None:
    conversations: dict[str, str] = {}

    def record(name: str, text: str) -> Callable[[dict[str, Any]], str]:
        def answer(request: dict[str, Any]) -> str:
            conversations[name] = json.dumps(request["messages"])
            return text

        return answer

    harness.answers = {
        "remember blue": "Remembered.",
        "which color": record("fork", "Blue."),
        "anything else": record("original", "No."),
    }
    async with harness.start(
        ForkingWorkflow, "remember blue", "which color", "anything else"
    ) as handle:
        original, fork, *ran_in = await handle.result()

    assert ran_in == [original, fork, original]
    # The fork has the original's history; the original never gets the fork's turn.
    assert "remember blue" in conversations["fork"]
    assert "remember blue" in conversations["original"]
    assert "which color" not in conversations["original"]


def decision(behavior: str, reason: str | None = None) -> dict[str, Any]:
    """A PreToolUse hook's output with a permission decision."""
    output = {"hookEventName": "PreToolUse", "permissionDecision": behavior}
    if reason is not None:
        output["permissionDecisionReason"] = reason
    return {"hookSpecificOutput": output}


async def test_a_hook_deny_refuses_the_call_without_asking(harness: Harness) -> None:
    hooks = [Hook("PreToolUse", decision("deny", "No writes"), matcher="Bash")]

    messages = await harness.run(ANSWERS, ["write it"], hooks=hooks)

    assert asked_about(messages) == []
    denied = "PreToolUse:Bash hook error: No writes"
    assert tool_results(messages) == [["toolu_1", denied, True]]
    assert not notes(harness).exists()
    [hook_input] = harness.hook_inputs
    assert hook_input["tool_use_id"] == "toolu_1"


async def test_a_hook_that_raised_has_no_opinion(harness: Harness) -> None:
    hook = Hook("PreToolUse", decision("deny", "No."), fail=True)

    messages = await harness.run(ANSWERS, ["write it"], hooks=[hook])

    # The call is asked about, as if there were no hook.
    assert asked_about(messages) == ["toolu_1"]
    assert notes(harness).exists()


async def test_a_hook_out_of_time_stops_the_call(harness: Harness) -> None:
    hook = Hook("PreToolUse", decision("allow"), wait=5, timeout=1)

    messages = await harness.run(ANSWERS, ["write it"], hooks=[hook])

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
    harness.answers = background("sleep 2")
    async with harness.start(OneShotWorkflow, "start it", ["Bash"]) as handle:
        messages = await handle.result()

    assert results(messages) == [["Started."], ["Finished."]]
    # Ending the input stopped the CLI.
    assert harness.running() == []


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


async def test_partial_messages_go_to_the_stream_and_whole_ones_to_the_workflow(
    harness: Harness,
) -> None:
    harness.answers = {"hi": "Hello."}
    async with harness.start(StreamingWorkflow, "hi") as handle:
        stream = WorkflowStreamClient.create(harness.client, handle.id)
        deltas: list[str] = []
        async for item in stream.subscribe("events", poll_cooldown=timedelta(0)):
            event = item.data["event"]
            if event["type"] == "content_block_delta":
                deltas.append(event["delta"]["text"])
            if event["type"] == "message_stop":
                break
        await handle.signal(StreamingWorkflow.finish)
        messages = await handle.result()

    assert deltas == ["Hello."]
    # The Workflow gets no StreamEvent.
    assert messages == [["AssistantMessage", ["Hello."]], ["Result", "Hello."]]
