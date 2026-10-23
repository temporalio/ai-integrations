"""ClaudeSDKClient in a Workflow, with a scripted Claude Code in an in-memory sandbox."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from temporalio.api.enums.v1 import EventType
from temporalio.claude_agent_sdk import (
    Sandbox,
    SandboxContext,
    SandboxError,
    SandboxFactory,
)
from temporalio.claude_agent_sdk._output import OutputSender
from temporalio.claude_agent_sdk._payloads import (
    CLOSE,
    CONNECT,
    LAUNCH,
    OUTPUT_KINDS,
    WATCH_OUTPUT,
    WRITE_STDIN,
    CliOutput,
)
from temporalio.client import WorkflowFailureError
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.exceptions import ApplicationError, CancelledError
from tests.helpers.memory_sandbox import MemoryHarness, flag
from tests.helpers.workflows import (
    PARTIAL,
    Chat,
    ChatWorkflow,
    CloseFirstWorkflow,
    OneShotWorkflow,
    ReadToEndWorkflow,
    bash,
    until,
)

# A turn that runs a Bash call, and one that answers at once.
WRITE_IT = {
    "write it": bash("toolu_1", "echo hi > notes.txt"),
    "tool_result:toolu_1": "Wrote it.",
}
WRITE_IT_AND_TWO = {"two": "Two.", **WRITE_IT}


async def test_each_prompt_is_a_turn_of_one_conversation(
    harness: MemoryHarness,
) -> None:
    messages = await harness.run({"one": "One.", "two": "Two."}, ["one", "two"])

    assert messages == [
        ["AssistantMessage", ["One."]],
        ["Result", "One."],
        ["AssistantMessage", ["Two."]],
        ["Result", "Two."],
    ]
    [args] = harness.sandbox.cli_args
    assert flag(args, "--resume") is None
    assert harness.sandbox.asked == ["one", "two"]


async def test_an_approved_tool_runs_and_the_model_reads_its_result(
    harness: MemoryHarness,
) -> None:
    messages = await harness.run(WRITE_IT, ["write it"])

    assert messages == [
        ["AssistantMessage", ["toolu_1", "Bash", {"command": "echo hi > notes.txt"}]],
        ["Ask", "toolu_1"],
        ["UserMessage", ["toolu_1", "Ran Bash", False]],
        ["AssistantMessage", ["Wrote it."]],
        ["Result", "Wrote it."],
    ]
    assert harness.sandbox.ran == [["Bash", {"command": "echo hi > notes.txt"}]]


async def test_a_denied_tool_does_not_run_and_the_model_reads_why(
    harness: MemoryHarness,
) -> None:
    messages = await harness.run(WRITE_IT, ["write it"], approve=False)

    assert ["UserMessage", ["toolu_1", "A person denied it", True]] in messages
    assert harness.sandbox.ran == []


async def test_an_allowed_tool_runs_without_asking(harness: MemoryHarness) -> None:
    messages = await harness.run(WRITE_IT, ["write it"], allowed_tools=["Bash"])

    assert ["Ask", "toolu_1"] not in messages
    assert harness.sandbox.ran == [["Bash", {"command": "echo hi > notes.txt"}]]


async def test_a_cli_that_cannot_run_fails_the_workflow(
    harness: MemoryHarness,
) -> None:
    harness.sandbox.fail = "bad flag"

    with pytest.raises(WorkflowFailureError) as raised:
        await harness.run({}, ["one"])

    error = _application_error(raised.value)
    assert (error.type, error.non_retryable) == ("ClaudeCodeExited", True)
    assert "bad flag" in error.message
    # The sandbox started for it is ended.
    assert harness.sandbox._boot is None


async def test_a_cli_that_exits_nonzero_fails_the_workflow_without_a_retry(
    harness: MemoryHarness,
) -> None:
    """As the Claude Agent SDK's own transport raises ProcessError. A retry can't change a recorded exit."""
    harness.sandbox.crash_before = "one"

    with pytest.raises(WorkflowFailureError) as raised:
        await harness.execute(ReadToEndWorkflow, "one", answers={"one": "One."})

    error = _application_error(raised.value)
    assert (error.type, error.non_retryable) == ("ClaudeCodeExited", True)
    assert "status 137" in error.message


async def test_a_lost_cli_is_watched_again_as_the_retry_policy_allows(
    harness: MemoryHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = harness.sandbox
    # Lost, as with its sandbox: no exit is recorded.
    monkeypatch.setattr(sandbox, "crash", sandbox.stop)
    sandbox.crash_before = "one"
    sandbox.answers = {"one": "One."}

    async with harness.start(ReadToEndWorkflow, "one") as handle:
        while True:
            description = await handle.describe()
            pending = description.raw_description.pending_activities
            if any(
                a.activity_type.name == WATCH_OUTPUT and a.attempt > 1 for a in pending
            ):
                break
            await asyncio.sleep(0.1)
        await handle.terminate()


async def test_a_sandbox_error_fails_only_that_activity_and_its_retry_builds_the_sandbox_again(
    harness: MemoryHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = harness.sandbox
    built: list[SandboxContext] = []
    built_when_lost: list[int] = []

    def build(context: SandboxContext) -> Sandbox:
        built.append(context)
        return sandbox

    harness.factory = build
    write_files = sandbox.write_files

    async def lose_a_write(
        files: Mapping[str, tuple[bytes, int]], **kwargs: Any
    ) -> None:
        if not built_when_lost and any(path.endswith(".jsonl") for path in files):
            built_when_lost.append(len(built))
            raise SandboxError("Sandbox is gone")
        await write_files(files, **kwargs)

    monkeypatch.setattr(sandbox, "write_files", lose_a_write)

    messages = await harness.run({"one": "One."}, ["one"])

    assert messages == [["AssistantMessage", ["One."]], ["Result", "One."]]
    assert built_when_lost
    assert len(built) > built_when_lost[0]
    # The same CLI went on: no new CLI.
    assert len(sandbox.cli_args) == 1
    assert sandbox.asked == ["one"]


async def test_the_sdks_query_function_ends_once_claude_code_is_done(
    harness: MemoryHarness,
) -> None:
    messages = await harness.execute(OneShotWorkflow, "one", answers={"one": "One."})

    assert messages == [["AssistantMessage", ["One."]], ["Result", "One."]]
    assert len(harness.sandbox.cli_args) == 1


@pytest.mark.parametrize("factory", ["function", "async function"])
async def test_a_factory_builds_the_sandbox_for_each_activity(
    harness: MemoryHarness, factory: str
) -> None:
    """A factory builds the sandbox for each Activity, from the Workflow it runs for."""
    built: list[SandboxContext] = []

    def build(context: SandboxContext) -> Sandbox:
        built.append(context)
        return harness.sandbox

    async def build_async(context: SandboxContext) -> Sandbox:
        return build(context)

    factories: dict[str, SandboxFactory] = {
        "function": build,
        "async function": build_async,
    }
    harness.factory = factories[factory]

    messages = await harness.execute(OneShotWorkflow, "one", answers={"one": "One."})

    assert messages == [["AssistantMessage", ["One."]], ["Result", "One."]]
    assert harness.handle is not None
    [context] = set(built)
    assert context == SandboxContext(
        "local",
        harness.client.namespace,
        harness.handle.id,
        harness.run_id,
        harness.run_id,
    )


async def test_a_workflow_whose_sandbox_never_starts_can_be_cancelled(
    harness: MemoryHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing the transport cancels starting the sandbox."""
    starts: list[Mapping[str, str]] = []

    async def start(*, env: Mapping[str, str]) -> None:
        starts.append(env)
        raise RuntimeError("No sandbox today")

    monkeypatch.setattr(harness.sandbox, "start", start)

    async with harness.start(ChatWorkflow, Chat(["one"])) as handle:
        await until(lambda: len(starts) >= 2)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as raised:
            await asyncio.wait_for(handle.result(), 20)

    assert isinstance(raised.value.cause, CancelledError)


async def test_closing_with_messages_not_yet_read_ends_them_cleanly(
    harness: MemoryHarness,
) -> None:
    """Closing cancels the Activities that carry the CLI's input and output, which is no failure."""
    kinds = await harness.execute(CloseFirstWorkflow, "one", answers={"one": "One."})

    assert kinds == ["assistant", "result"]


async def test_each_activity_finds_a_lost_worker_soon(harness: MemoryHarness) -> None:
    """Following the output also retries soon: the Claude Agent SDK waits on it."""
    await harness.run({"one": "One."}, ["one"])

    assert harness.handle is not None
    history = await harness.handle.fetch_history()
    scheduled = {
        e.activity_task_scheduled_event_attributes.activity_type.name: (
            e.activity_task_scheduled_event_attributes
        )
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    }
    assert set(scheduled) == {CONNECT, LAUNCH, WRITE_STDIN, WATCH_OUTPUT, CLOSE}
    for attributes in scheduled.values():
        assert attributes.heartbeat_timeout.ToTimedelta() == timedelta(seconds=10)
    retry = scheduled[WATCH_OUTPUT].retry_policy
    assert retry.maximum_interval.ToTimedelta() == timedelta(seconds=5)


async def test_the_temporal_ui_says_what_each_step_does(harness: MemoryHarness) -> None:
    await harness.run({"one": "One."}, ["one"])

    assert harness.handle is not None
    history = await harness.handle.fetch_history()
    activities = [
        json.loads(e.user_metadata.summary.data)
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]
    signals = {
        e.workflow_execution_signaled_event_attributes.signal_name
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_SIGNALED
    }
    assert set(activities) == {
        "Start sandbox",
        "Start Claude Code",
        "Stream output",
        "Send request",
        "Send prompt",
        "Stop sandbox",
    }
    assert signals <= set(OUTPUT_KINDS)


async def test_partial_messages_go_to_the_stream_and_not_the_workflow(
    harness: MemoryHarness,
) -> None:
    """The Workflow still gets the complete assistant message and the result."""
    harness.sandbox.answers = {"one": "One."}
    # follow_ups: the Workflow waits for a turn that never comes, so it stays up to be read.
    chat = Chat(["one"], partial=True, follow_ups=1)
    async with harness.start(ChatWorkflow, chat) as handle:
        stream = WorkflowStreamClient.create(harness.client, handle.id)

        async def events() -> list[Any]:
            items: list[Any] = []
            async for item in stream.subscribe(PARTIAL):
                items.append(item.data)
                if len(items) == 2:
                    return items
            return items

        published = await asyncio.wait_for(events(), 30)
        messages: list[Any] = []
        while ["Result", "One."] not in messages:
            messages = await handle.query(ChatWorkflow.received)
            await asyncio.sleep(0.05)
        await handle.terminate()

    assert [item["type"] for item in published] == ["stream_event"] * 2
    assert [item["event"]["index"] for item in published] == [0, 1]
    assert messages == [["AssistantMessage", ["One."]], ["Result", "One."]]
    [args] = harness.sandbox.cli_args
    assert "--include-partial-messages" in args


def assistant(*content: dict[str, Any]) -> dict[str, Any]:
    return {"type": "assistant", "message": {"content": list(content)}}


def ndjson(*messages: dict[str, Any]) -> bytes:
    return "".join(json.dumps(message) + "\n" for message in messages).encode()


class FakeWorkflow:
    """Records each Signal the sender sends, and what was recorded before it."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.signals: list[tuple[str, CliOutput]] = []
        self.events = events if events is not None else []

    async def signal(self, signal: str, output: CliOutput) -> None:
        self.events.append(f"signal {signal}")
        self.signals.append((signal, output))


class FakeStream:
    """A stream client that records its publishes and flushes in ``events``."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.flush_fails = False

    def topic(self, name: str) -> Any:
        def publish(message: dict[str, Any]) -> None:
            self.events.append("publish")
            self.published.append((name, message))

        return SimpleNamespace(publish=publish)

    async def flush(self) -> None:
        self.events.append("flush")
        if self.flush_fails:
            raise TimeoutError("The stream is gone")


async def test_each_line_is_a_signal_named_and_summarized() -> None:
    """So the Temporal UI shows what each Signal of stdout carries."""
    workflow = FakeWorkflow()
    lines = [
        assistant({"type": "text", "text": "Listing"}),
        assistant({"type": "tool_use", "name": "Bash", "input": {}}),
        {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
        {"type": "stream_event"},
        {"type": "result", "subtype": "success"},
        {
            "type": "control_request",
            "request": {"subtype": "can_use_tool", "tool_name": "Bash"},
        },
    ]

    await OutputSender(cast(Any, workflow), 0).add(ndjson(*lines))

    assert [(name, output.summary) for name, output in workflow.signals] == [
        ("assistant", "assistant: text"),
        ("tool_use", "tool_use: Bash"),
        ("tool_result", "tool_result"),
        ("other", "stream_event"),
        ("result", "result: success"),
        ("control_request", "control_request: can_use_tool Bash"),
    ]
    assert {name for name, _ in workflow.signals} <= set(OUTPUT_KINDS)


async def test_a_line_goes_as_soon_as_it_is_complete_from_the_offset_given() -> None:
    workflow = FakeWorkflow()
    first, second = ndjson({"type": "a"}), ndjson({"type": "b"})
    output = OutputSender(cast(Any, workflow), 100)

    await output.add(first[:5])
    assert workflow.signals == []
    await output.add(first[5:] + second[:3])
    [(_, sent)] = workflow.signals
    assert (sent.start, sent.end, sent.line) == (100, 100 + len(first), '{"type": "a"}')
    await output.add(second[3:])

    assert [(o.start, o.end) for _, o in workflow.signals] == [
        (100, 100 + len(first)),
        (100 + len(first), 100 + len(first) + len(second)),
    ]
    assert (output.start, output.end) == (100 + len(first + second),) * 2


async def test_stdout_read_again_replaces_the_partial_line() -> None:
    """add_rest gets the file from byte ``end``, which repeats what was a partial line."""
    workflow = FakeWorkflow()
    line = ndjson({"type": "a"})
    output = OutputSender(cast(Any, workflow), 0)

    await output.add(line[:5])
    await output.add_rest(line)

    [(_, sent)] = workflow.signals
    assert (sent.start, sent.end) == (0, len(line))


async def test_a_published_line_is_flushed_before_the_next_signal() -> None:
    events: list[str] = []
    workflow, stream = FakeWorkflow(events), FakeStream(events)
    delta = {"type": "stream_event", "event": {"type": "content_block_delta"}}
    first, answer, last = (
        assistant({"type": "thinking"}),
        assistant({"type": "text"}),
        {"type": "result"},
    )
    output = OutputSender(cast(Any, workflow), 0, cast(Any, stream), "partial")

    await output.add(ndjson(first, delta, delta))

    # The offset stays before the published lines, which may not be flushed yet.
    assert events == ["signal assistant", "publish", "publish"]
    assert output.start == len(ndjson(first))
    assert output.end == len(ndjson(first, delta, delta))

    await output.add(ndjson(answer, last))

    assert events == [
        "signal assistant",
        "publish",
        "publish",
        "flush",
        "signal assistant",
        "signal result",
    ]
    assert stream.published == [("partial", delta)] * 2
    assert [name for name, _ in workflow.signals] == [
        "assistant",
        "assistant",
        "result",
    ]
    # The Signal after the published lines starts before them: the Workflow skips them.
    assert workflow.signals[1][1].start == len(ndjson(first))
    assert output.start == output.end == len(ndjson(first, delta, delta, answer, last))


async def test_the_offset_does_not_pass_a_published_line_that_failed_to_flush() -> None:
    events: list[str] = []
    workflow, stream = FakeWorkflow(events), FakeStream(events)
    stream.flush_fails = True
    delta = {"type": "stream_event"}
    output = OutputSender(cast(Any, workflow), 0, cast(Any, stream), "partial")

    with pytest.raises(TimeoutError):
        await output.add(ndjson(delta, {"type": "result"}))

    assert workflow.signals == []
    assert output.start == 0


def _application_error(error: BaseException) -> ApplicationError:
    """The application error at the root of ``error``."""
    cause: BaseException | None = error
    while cause is not None and not isinstance(cause, ApplicationError):
        cause = cause.__cause__
    assert isinstance(cause, ApplicationError)
    return cause
