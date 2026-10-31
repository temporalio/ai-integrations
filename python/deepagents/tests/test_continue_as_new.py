"""Continue-as-new state carry for long-running Deep Agents.

``run_deep_agent(continue_as_new_after=...)`` keeps a long conversation from
bloating workflow history: once the current turn finishes past the threshold and
there is still pending work, it snapshots the accumulated messages and continues
into a fresh run. (The legacy result cache is retired for new executions —
see ``deepagents.retire-result-cache``; ``test_state_snapshot_roundtrip`` below
covers the _serde plumbing that only the legacy replay branch still uses.) Most tests use a plain
fake agent (no LangChain needed) so they boot a real Temporal server and exercise
the continue-as-new machinery end to end. One drives a real Deep Agent with a
scripted model, because its files live in graph state that a fake agent does
not have.
"""

from __future__ import annotations

import sys
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from temporalio.client import WorkflowExecutionStatus
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


# The tests above drive fake agents with no state beyond messages and todos. A
# real Deep Agent's default StateBackend keeps its files in graph state.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

with workflow.unsafe.imports_passed_through():
    from langchain.agents.middleware import TodoListMiddleware
    from langchain_core.messages import AIMessage

from temporalio.deepagents.testing import mock_model_provider


def _tool_calls(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"{name}-{i}"}
            for i, (name, args) in enumerate(calls)
        ],
    )


# Run 1 writes a new note, overwrites the seeded file and leaves its todo
# pending, so the driver continues-as-new; run 2 reads both back.
FILES_RESPONSES: list[Any] = [
    _tool_calls(
        ("write_file", {"file_path": "/notes.md", "content": "draft"}),
        ("write_file", {"file_path": "/seed.md", "content": "new"}),
        ("write_todos", {"todos": [{"content": "notes", "status": "pending"}]}),
    ),
    "first turn",
    _tool_calls(
        ("read_file", {"file_path": "/notes.md"}),
        ("read_file", {"file_path": "/seed.md"}),
        ("write_todos", {"todos": [{"content": "notes", "status": "completed"}]}),
    ),
    "done",
]


@workflow.defn
class FilesCanWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        agent = create_deep_agent(model="fake:model", middleware=[TodoListMiddleware()])
        result = await run_deep_agent(
            agent,
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )
        reads = [
            str(m.content)
            for m in result["messages"]
            if m.type == "tool" and m.name == "read_file"
        ]
        return {"reads": reads, "files": sorted(result.get("files") or {})}


@pytest.mark.asyncio
async def test_state_files_survive_continue_as_new(env: WorkflowEnvironment) -> None:
    """Files as the agent left them before the boundary are readable after it.

    The input seeds ``/seed.md``, which the agent overwrites in the first run,
    so the continued run must see the agent's version, not the input's.
    """
    from deepagents.backends.utils import create_file_data

    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(FILES_RESPONSES))
    async with Worker(
        env.client,
        task_queue="da-can-files",
        workflows=[FilesCanWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            FilesCanWorkflow.run,
            {
                "messages": [{"role": "user", "content": "Take notes."}],
                "files": {"/seed.md": create_file_data("old")},
            },
            id=f"da-can-files-{uuid.uuid4()}",
            task_queue="da-can-files",
        )
        result = await handle.result()
        first = env.client.get_workflow_handle(
            handle.id, run_id=handle.first_execution_run_id
        )
        desc = await first.describe()
        first_history = await first.fetch_history()
        last_history = await handle.fetch_history()

    assert desc.status == WorkflowExecutionStatus.CONTINUED_AS_NEW, desc.status
    notes, seed = result["reads"]
    assert "draft" in notes, result
    assert "new" in seed and "old" not in seed, result
    assert result["files"] == ["/notes.md", "/seed.md"], result

    replayer = Replayer(
        workflows=[FilesCanWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=mock_model_provider(["unused"]))],
    )
    await replayer.replay_workflow(first_history)
    await replayer.replay_workflow(last_history)


class EmptiesFilesAgent:
    """Reports which files it was given, then leaves its files state empty.

    Two turns: the first leaves a todo pending so the driver continues as new.
    """

    async def ainvoke(self, input: Any) -> dict:
        messages = list(input.get("messages", []))
        seen = sorted(input.get("files") or {})
        messages = [*messages, f"saw {seen}"]
        done = len(messages) >= 3
        return {
            "messages": messages,
            "files": {},
            "todos": [{"content": "w", "status": "completed" if done else "pending"}],
        }


@workflow.defn
class EmptiesFilesCanWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        return await run_deep_agent(
            EmptiesFilesAgent(),
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )


@pytest.mark.asyncio
async def test_emptied_files_stay_empty_across_continue_as_new(
    env: WorkflowEnvironment,
) -> None:
    """Input files the agent removed do not come back with the re-sent input."""
    async with Worker(
        env.client,
        task_queue="da-can-empty-files",
        workflows=[EmptiesFilesCanWorkflow],
        plugins=[DeepAgentsPlugin()],
    ):
        result = await env.client.execute_workflow(
            EmptiesFilesCanWorkflow.run,
            {"messages": ["start"], "files": {"/seed.md": "x"}},
            id=f"da-can-empty-files-{uuid.uuid4()}",
            task_queue="da-can-empty-files",
        )

    assert result["messages"] == ["start", "saw ['/seed.md']", "saw []"], result


def test_merge_files_composes_like_merge_snapshot() -> None:
    """Carried files go into the next input; new input files win per path."""
    from temporalio.deepagents.workflow import _merge_files

    snapshot = {"messages": ["q"], "files": {"/a": "carried", "/b": "carried"}}
    merged = _merge_files({"messages": ["q"], "files": {"/b": "new"}}, snapshot)
    assert merged["files"] == {"/a": "carried", "/b": "new"}
    assert merged["messages"] == ["q"]

    # A bare input becomes a messages input so the files have somewhere to go.
    assert _merge_files("q", snapshot) == {
        "messages": ["q"],
        "files": snapshot["files"],
    }
    assert _merge_files(["q", "r"], snapshot)["messages"] == ["q", "r"]

    # New input whose files value is not a mapping: the carried files are used.
    merged = _merge_files({"messages": ["q"], "files": ["/b"]}, snapshot)
    assert merged["files"] == snapshot["files"]

    # No files, or a value that is not a mapping: the input is left alone.
    no_files: list[Any] = [None, {}, ["/a"]]
    for files in no_files:
        assert _merge_files({"messages": ["q"]}, {"files": files}) == {
            "messages": ["q"]
        }
