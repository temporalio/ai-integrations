"""Continue-as-new state carry for long-running Deep Agents.

``run_deep_agent(continue_as_new_after=...)`` keeps a long conversation from
bloating workflow history: once the current turn finishes past the threshold and
there is still pending work, it snapshots the accumulated messages and continues
into a fresh run. (The legacy result cache is retired for new executions —
see ``deepagents.retire-result-cache``; ``test_state_snapshot_roundtrip`` below
covers the _serde plumbing that only the legacy replay branch still uses.) Most tests use a plain
fake agent (no LangChain needed) so they boot a real Temporal server and exercise
the continue-as-new machinery end to end. One drives a real Deep Agent with a
scripted model, because its transcript holds LangChain message objects that must
cross the boundary as messages.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.common import RetryPolicy
from temporalio.testing import WorkflowEnvironment

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)
from temporalio import workflow
from temporalio.deepagents import DeepAgentsPlugin, _serde, run_deep_agent
from temporalio.deepagents.workflow import _CACHE_KEY, _merge_snapshot
from temporalio.worker import Replayer, Worker


class FakeAgent:
    """A stand-in compiled agent that appends a step and reports a todo.

    It is *not* a LangChain object — it just satisfies the ``ainvoke`` shape
    ``run_deep_agent`` drives, so the continue-as-new path can be tested without
    a model provider or the LangChain import tree.
    """

    async def ainvoke(self, input: Any) -> dict:
        messages = list(input.get("messages", [])) if isinstance(input, dict) else []
        messages = [*messages, "step"]
        done = messages.count("step") >= 3
        return {
            "messages": messages,
            "todos": [
                {"content": "work", "status": "completed" if done else "pending"}
            ],
        }


@workflow.defn
class ContinueAsNewWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        # Threshold of 1 means: continue-as-new as soon as there is pending work,
        # which the fake agent reports until it has appended 3 steps.
        return await run_deep_agent(
            FakeAgent(),
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )


@pytest.mark.asyncio
async def test_can_threshold_and_cache(env: WorkflowEnvironment) -> None:
    plugin = DeepAgentsPlugin()
    async with Worker(
        env.client,
        task_queue="da-can",
        workflows=[ContinueAsNewWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ContinueAsNewWorkflow.run,
            {"messages": ["start"]},
            id=f"da-can-{uuid.uuid4()}",
            task_queue="da-can",
        )
        result = await handle.result()

    assert result["messages"] == ["start", "step", "step", "step"], result
    assert result["todos"][0]["status"] == "completed"


def test_state_snapshot_roundtrip() -> None:
    # LEGACY-branch plumbing (deepagents.retire-result-cache unpatched): a
    # carried cache rehydrates to the same hits during replay of pre-change
    # histories. Delete alongside the patch's deprecate_patch cleanup.
    _serde.set_result_cache({})
    key = _serde.cache_key("model", "fake:model", [["m"], []])
    _serde.cache_put(key, {"dumped": "message"})
    snapshot = _serde.result_cache_snapshot()
    assert snapshot and key in snapshot

    # Simulate the continued run: a fresh cache seeded from the snapshot.
    _serde.set_result_cache(dict(snapshot))
    hit, value = _serde.cache_lookup(key)
    assert hit and value == {"dumped": "message"}


@workflow.defn
class SuggestedCanWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        # No continue_as_new_after: the default follows the server's own
        # is_continue_as_new_suggested() signal.
        return await run_deep_agent(
            FakeAgent(),
            input,
            state_snapshot=state_snapshot,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("suggested", [False, True])
async def test_can_defaults_to_server_suggestion(
    env: WorkflowEnvironment, suggested: bool
) -> None:
    """The default follows the SDK suggestion and carries state through real runs.

    Supply the SDK's suggestion directly instead of changing server configuration
    or generating thousands of events to reach its default threshold.
    """
    plugin = DeepAgentsPlugin()
    with patch.object(
        workflow.Info, "is_continue_as_new_suggested", return_value=suggested
    ) as suggestion:
        async with Worker(
            env.client,
            task_queue="da-can-suggested",
            workflows=[SuggestedCanWorkflow],
            plugins=[plugin],
        ):
            handle = await env.client.start_workflow(
                SuggestedCanWorkflow.run,
                {"messages": ["start"]},
                id=f"da-can-suggested-{uuid.uuid4()}",
                task_queue="da-can-suggested",
            )
            result = await handle.result()
        suggestion.assert_called()

    steps = 3 if suggested else 1
    assert result["messages"] == ["start", *(["step"] * steps)], result
    assert result["todos"][0]["status"] == ("completed" if suggested else "pending")
    first = env.client.get_workflow_handle(
        handle.id, run_id=handle.first_execution_run_id
    )
    desc = await first.describe()
    assert desc.status == (
        WorkflowExecutionStatus.CONTINUED_AS_NEW
        if suggested
        else WorkflowExecutionStatus.COMPLETED
    )


def test_merge_snapshot_preserves_new_input_messages() -> None:
    # External resume: a saved snapshot plus a NEW user message composes —
    # carried history first, the new message after. (Replace semantics here
    # would silently drop the user's latest message.)
    merged = _merge_snapshot(
        {"messages": ["new question"], "config": {"k": "v"}},
        {"messages": ["old q", "old a"]},
    )
    assert merged["messages"] == ["old q", "old a", "new question"]
    assert merged["config"] == {"k": "v"}

    # Non-Mapping input: a bare prompt appends after the carried history.
    merged = _merge_snapshot("new question", {"messages": ["old q", "old a"]})
    assert merged["messages"] == ["old q", "old a", "new question"]


def test_merge_snapshot_internal_carry_has_no_duplicates() -> None:
    # The driver strips messages from the carried input, so the internal
    # continue-as-new path resumes from the snapshot alone.
    merged = _merge_snapshot({"config": {"k": "v"}}, {"messages": ["start", "step"]})
    assert merged["messages"] == ["start", "step"]
    assert merged["config"] == {"k": "v"}


class BareInputAgent:
    """ainvoke-shaped agent for a BARE-STRING input: first turn folds the
    prompt into the transcript; finishes after three steps."""

    async def ainvoke(self, input: Any) -> dict:
        if isinstance(input, dict):
            messages = list(input.get("messages", []))
        else:
            messages = [input]
        messages = [*messages, "step"]
        done = messages.count("step") >= 3
        return {
            "messages": messages,
            "todos": [
                {"content": "work", "status": "completed" if done else "pending"}
            ],
        }


@workflow.defn
class BareInputCanWorkflow:
    @workflow.run
    async def run(self, input: str, state_snapshot: dict | None = None) -> dict:
        # A STR-typed run signature: the carried input must decode as str
        # after every continue-as-new, or the workflow stalls on task retry.
        return await run_deep_agent(
            BareInputAgent(),
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )


@pytest.mark.asyncio
async def test_bare_string_input_survives_continue_as_new(
    env: WorkflowEnvironment,
) -> None:
    """A bare-prompt input with a str-typed run signature crosses multiple
    continue-as-new boundaries: the type survives (no decode failure) and the
    prompt appears exactly once in the final transcript."""
    plugin = DeepAgentsPlugin()
    async with Worker(
        env.client,
        task_queue="da-can-bare",
        workflows=[BareInputCanWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            BareInputCanWorkflow.run,
            "start",
            id=f"da-can-bare-{uuid.uuid4()}",
            task_queue="da-can-bare",
        )
        result = await handle.result()

    assert result["messages"] == ["start", "step", "step", "step"], result
    first = env.client.get_workflow_handle(
        handle.id, run_id=handle.first_execution_run_id
    )
    desc = await first.describe()
    assert desc.status == WorkflowExecutionStatus.CONTINUED_AS_NEW, desc.status


@pytest.mark.asyncio
async def test_can_args_do_not_carry_messages(env: WorkflowEnvironment) -> None:
    """Continue-as-new carries one transcript and produces replayable histories."""
    import json

    plugin = DeepAgentsPlugin()
    async with Worker(
        env.client,
        task_queue="da-can-args",
        workflows=[ContinueAsNewWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ContinueAsNewWorkflow.run,
            {"messages": ["start"], "config": {"k": "v"}},
            id=f"da-can-args-{uuid.uuid4()}",
            task_queue="da-can-args",
        )
        await handle.result()
        first = env.client.get_workflow_handle(
            handle.id, run_id=handle.first_execution_run_id
        )
        hist = await first.fetch_history()
        can_events = [
            e
            for e in hist.events
            if e.HasField("workflow_execution_continued_as_new_event_attributes")
        ]
        assert can_events, "first run did not continue-as-new"
        payloads = can_events[
            0
        ].workflow_execution_continued_as_new_event_attributes.input.payloads
        carried_input = json.loads(payloads[0].data)
        snapshot = json.loads(payloads[1].data)

    assert "messages" not in carried_input, carried_input
    assert carried_input.get("config") == {"k": "v"}
    assert snapshot["messages"], snapshot
    assert _CACHE_KEY not in snapshot

    replayer = Replayer(workflows=[ContinueAsNewWorkflow], plugins=[DeepAgentsPlugin()])
    await replayer.replay_workflow(hist)
    await replayer.replay_workflow(await handle.fetch_history())


class DiskCountingBackend:
    """Each read appends to a log and reports the total (disk state survives
    sandbox re-imports, replays, and continue-as-new)."""

    def __init__(self, root: str) -> None:
        self._log = Path(root) / "reads.log"

    def read(self, _file_path: str) -> str:
        with self._log.open("a") as f:
            f.write("r\n")
        return f"read:{len(self._log.read_text().splitlines())}"


class NoMessagesAgent:
    """Returns an EMPTY transcript with pending todos on the first turn, then
    answers. Turn tracking lives on disk via an activity-backed counter — the
    agent object is re-created each run/replay, so in-memory state cannot
    distinguish turns."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    async def ainvoke(self, input: Any) -> dict:
        turn = int((await self._backend.read("turn")).split(":")[1])
        if turn == 1:
            return {"messages": [], "todos": [{"content": "w", "status": "pending"}]}
        messages = list(input.get("messages", [])) if isinstance(input, dict) else []
        return {
            "messages": [*messages, "answered"],
            "todos": [{"content": "w", "status": "completed"}],
        }


@workflow.defn
class EmptyTranscriptCanWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        from temporalio.deepagents import TemporalBackend

        backend = TemporalBackend(
            DiskCountingBackend(input["root"]),
            activity_options={
                "start_to_close_timeout": timedelta(seconds=30),
                "retry_policy": RetryPolicy(maximum_attempts=1),
            },
        )
        return await run_deep_agent(
            NoMessagesAgent(backend),
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )


@pytest.mark.asyncio
async def test_empty_transcript_can_preserves_prompt(
    env: WorkflowEnvironment, tmp_path: Any
) -> None:
    """A turn ending with pending todos and an EMPTY transcript still carries
    the conversation across a REAL continue-as-new boundary."""
    plugin = DeepAgentsPlugin()
    async with Worker(
        env.client,
        task_queue="da-can-empty",
        workflows=[EmptyTranscriptCanWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await env.client.start_workflow(
            EmptyTranscriptCanWorkflow.run,
            {"messages": ["the question"], "root": str(tmp_path)},
            id=f"da-can-empty-{uuid.uuid4()}",
            task_queue="da-can-empty",
        )
        result = await handle.result()
        first = env.client.get_workflow_handle(
            handle.id, run_id=handle.first_execution_run_id
        )
        desc = await first.describe()

    assert desc.status == WorkflowExecutionStatus.CONTINUED_AS_NEW, desc.status
    assert result["messages"] == ["the question", "answered"], result


class _CrossBoundaryAgent:
    """ainvoke-shaped driver issuing the SAME read every run; reports pending
    until the second read has observably executed."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    async def ainvoke(self, input: Any) -> dict:
        out = await self._backend.read("state.txt")
        done = int(out.split(":")[1]) >= 2
        return {
            "messages": [*list(input.get("messages", [])), out],
            "todos": [
                {"content": "work", "status": "completed" if done else "pending"}
            ],
        }


@workflow.defn
class CrossBoundaryOpWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        from temporalio.deepagents import TemporalBackend

        backend = TemporalBackend(
            DiskCountingBackend(input["root"]),
            # No retries: the disk counter increments per activity ATTEMPT, so
            # a retry would skew the exact read-count assertions.
            activity_options={
                "start_to_close_timeout": timedelta(seconds=30),
                "retry_policy": RetryPolicy(maximum_attempts=1),
            },
        )
        return await run_deep_agent(
            _CrossBoundaryAgent(backend),
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )


@pytest.mark.asyncio
async def test_identical_op_reruns_across_continue_as_new(
    env: WorkflowEnvironment, tmp_path: Any
) -> None:
    """An identical backend op issued on BOTH sides of a continue-as-new
    boundary executes on both sides. Under the legacy carried cache the
    post-boundary call was served the pre-boundary result (the continued run
    resumes from the transcript — it never re-executes prior dispatches, so
    a carried hit could only ever be stale)."""
    plugin = DeepAgentsPlugin()
    async with Worker(
        env.client,
        task_queue="da-cross-boundary",
        workflows=[CrossBoundaryOpWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await env.client.start_workflow(
            CrossBoundaryOpWorkflow.run,
            {"messages": [], "root": str(tmp_path)},
            id=f"da-cross-boundary-{uuid.uuid4()}",
            task_queue="da-cross-boundary",
        )
        result = await handle.result()

    # The read really executed in the continued run: disk shows two reads and
    # the carried transcript holds each run's distinct observation.
    assert (tmp_path / "reads.log").read_text().splitlines() == ["r", "r"]
    assert result["messages"][-2:] == ["read:1", "read:2"], result
    # The chain really crossed a boundary.
    first = env.client.get_workflow_handle(
        handle.id, run_id=handle.first_execution_run_id
    )
    desc = await first.describe()
    assert desc.status == WorkflowExecutionStatus.CONTINUED_AS_NEW, desc.status


# The tests above drive fake agents whose "messages" are strings. A real Deep
# Agent returns LangChain message objects, and those must survive the plugin's
# payload converter on their way into the next run.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

with workflow.unsafe.imports_passed_through():
    from langchain.agents.middleware import TodoListMiddleware
    from langchain_core.messages import AIMessage

from temporalio.deepagents.testing import mock_model_provider


def _write_todos(status: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "research", "status": status}]},
                "id": call_id,
            }
        ],
    )


# Run 1 leaves its todo pending, so the driver continues-as-new; run 2 completes
# it. mock_model_provider hands these out one model call at a time.
REAL_AGENT_RESPONSES: list[Any] = [
    _write_todos("pending", "todos-1"),
    "first turn",
    _write_todos("completed", "todos-2"),
    "done",
]


@workflow.defn
class RealAgentCanWorkflow:
    @workflow.run
    async def run(
        self, input: dict, state_snapshot: dict | None = None
    ) -> list[list[Any]]:
        # deepagents 0.7 leaves todos out of the default stack; add them so the
        # driver sees pending work.
        agent = create_deep_agent(model="fake:model", middleware=[TodoListMiddleware()])
        result = await run_deep_agent(
            agent,
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )
        return [
            [
                m.type,
                m.content,
                [c["id"] for c in getattr(m, "tool_calls", [])],
                getattr(m, "tool_call_id", None),
            ]
            for m in result["messages"]
        ]


@pytest.mark.asyncio
async def test_real_agent_messages_survive_continue_as_new(
    env: WorkflowEnvironment,
) -> None:
    """The continued run gets the transcript back as LangChain messages.

    The plugin is on the client, as in the README, so the snapshot crosses the
    boundary through the plugin's payload converter. A message that loses its
    ``type`` there makes the continued run fail every workflow task, so wait a
    bounded time and report that failure.
    """
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(REAL_AGENT_RESPONSES))
    config = env.client.config()
    config["plugins"] = [plugin]
    client = Client(**config)
    async with Worker(
        client,
        task_queue="da-can-real",
        workflows=[RealAgentCanWorkflow],
    ):
        handle = await client.start_workflow(
            RealAgentCanWorkflow.run,
            {"messages": [{"role": "user", "content": "Research."}]},
            id=f"da-can-real-{uuid.uuid4()}",
            task_queue="da-can-real",
        )
        try:
            result = await asyncio.wait_for(handle.result(), timeout=30)
        except asyncio.TimeoutError:
            async for e in handle.fetch_history_events():
                if e.HasField("workflow_task_failed_event_attributes"):
                    failure = e.workflow_task_failed_event_attributes.failure
                    pytest.fail(f"continued run is failing: {failure.message}")
            raise
        first = client.get_workflow_handle(
            handle.id, run_id=handle.first_execution_run_id
        )
        desc = await first.describe()
        first_history = await first.fetch_history()
        last_history = await handle.fetch_history()

    assert desc.status == WorkflowExecutionStatus.CONTINUED_AS_NEW, desc.status
    assert [m[0] for m in result] == ["human", "ai", "tool", "ai", "ai", "tool", "ai"]
    assert result[0][1] == "Research."
    # The first run's tool call and its result keep their ids across the boundary.
    assert result[1][2] == ["todos-1"]
    assert result[2][3] == "todos-1"
    assert result[-1][1] == "done"

    replayer = Replayer(
        workflows=[RealAgentCanWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=mock_model_provider(["unused"]))],
    )
    await replayer.replay_workflow(first_history)
    await replayer.replay_workflow(last_history)


def test_carried_messages_round_trip_and_other_values_pass_through() -> None:
    """Only serialized messages are loaded back; other values are left as is."""
    from langchain_core.documents import Document
    from langchain_core.load import dumpd
    from langchain_core.messages import HumanMessage, ToolMessage

    class OwnMessage(HumanMessage):
        """A subclass ``load`` cannot rebuild, so it is not serialized."""

    tool_message = ToolMessage("ok", tool_call_id="t1")
    other = [
        OwnMessage(content="mine"),
        "a string",
        {"role": "user", "content": "q"},
        # What older code wrote: a message dict without its type.
        {"content": "Research.", "id": "m1"},
        dumpd(Document(page_content="not a message")),
        {"lc": 1, "type": "constructor", "id": ["x"], "kwargs": {}},
    ]
    dumped = _serde.dump_carried_messages([tool_message, *other])
    assert dumped[1:] == other
    loaded = _serde.load_carried_messages(dumped)
    assert isinstance(loaded[0], ToolMessage)
    assert loaded[0].tool_call_id == "t1"
    assert loaded[1:] == other
