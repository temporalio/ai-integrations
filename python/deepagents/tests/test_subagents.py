"""Sub-agents inherit the activity seams.

Deep Agents builds sub-agents as separate graphs, but they inherit the parent's
``model`` object and tools by default. Because the plugin substitutes the model
*object*, every sub-agent's model call routes through an activity without any
per-sub-agent wiring. This runs a real agent configured with a sub-agent and
asserts model calls still land on the activity seam.

The exact number of hops depends on ``deepagents`` internals we do not pin, so
the assertion is the robust invariant: the configured agent runs and at least one
model call went through an activity.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.testing import WorkflowEnvironment

pytest.importorskip("deepagents")
pytest.importorskip("langchain_core")

from temporalio import workflow  # noqa: E402
from temporalio.client import WorkflowHistory  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402
from temporalio.workflow import NondeterminismError  # noqa: E402
from tests.helpers.deepagents import count_scheduled_activities  # noqa: E402

# Resolve framework symbols through the imported module to avoid confusing
# the top-level package with this same-named test directory.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent
StateBackend = pytest.importorskip("deepagents.backends").StateBackend
SummarizationMiddleware = pytest.importorskip(
    "deepagents.middleware"
).SummarizationMiddleware

with workflow.unsafe.imports_passed_through():
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import AIMessage

    from temporalio.deepagents import DeepAgentsPlugin
    from temporalio.deepagents.testing import FakeModel, mock_model_provider

INVOKE_MODEL = "deepagents.invoke_model"


@workflow.defn
class SubAgentWorkflow:
    @workflow.run
    async def run(self, question: str) -> str:
        agent = create_deep_agent(
            model="anthropic:claude-sonnet-4-5",
            system_prompt="You coordinate research.",
            subagents=[
                {
                    "name": "researcher",
                    "description": "Researches a topic in depth.",
                    "system_prompt": "You research topics.",
                }
            ],
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        return result["messages"][-1].content


@pytest.mark.asyncio
async def test_subagent_calls_route_to_activities(env: WorkflowEnvironment) -> None:
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["Coordinated answer."]),
    )
    async with Worker(
        env.client,
        task_queue="da-subagent",
        workflows=[SubAgentWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            SubAgentWorkflow.run,
            "Investigate the topic.",
            id=f"da-subagent-{uuid.uuid4()}",
            task_queue="da-subagent",
        )
        out = await handle.result()

    assert out
    counts = await count_scheduled_activities(handle)
    # A model instance shared with the sub-agent means model calls are activities.
    assert counts[INVOKE_MODEL] >= 1, counts


# ---------------------------------------------------------------------------
# Parallel sub-agents keep their own model results on replay
# ---------------------------------------------------------------------------
#
# Every model call is the same Activity type and Temporal does not compare
# Activity inputs on replay, so with sequence Activity ids a recorded result is
# matched by position. If a change alters the order in which two parallel
# sub-agents schedule their model calls, replay must fail loudly instead of
# handing each sub-agent the other's recorded answer.


def _lanes_tasks(
    description_a: str, description_b: str, subagents: tuple[str, str] = ("a", "b")
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "id": "task-a",
                "args": {"description": description_a, "subagent_type": subagents[0]},
            },
            {
                "name": "task",
                "id": "task-b",
                "args": {"description": description_b, "subagent_type": subagents[1]},
            },
        ],
    )


# With the same task text for both sub-agents, only the sub-agent's name tells
# their model calls apart.
_LANES_TASK_TEXT = {
    "different": ("Ask lane a.", "Ask lane b."),
    "same": ("Ask a lane.", "Ask a lane."),
    # Large enough that Deep Agents' filesystem middleware evicts each task
    # prompt to a file named with ``uuid4`` and puts that path in the message.
    "oversized": ("Ask lane a. " * 25000, "Ask lane b. " * 25000),
    # Both tasks go to one sub-agent and differ only in a UUID the caller
    # supplies, which must stay part of the identity.
    "customers": (
        "Look up customer 11111111-1111-4111-8111-111111111111.",
        "Look up customer 22222222-2222-4222-8222-222222222222.",
    ),
}
# The sub-agents the two tasks go to, when not lanes a and b.
_LANES_SUBAGENTS = {"customers": ("b", "b")}
_LANES_HISTORY = (
    Path(__file__).parent / "histories" / "parallel_subagents_sequence_ids.json"
)
_PATCH_ID = "deepagents.parallel-model-call-ids"


def _lanes_provider(task_text: str = "different") -> Any:
    """A model provider with one script per model name, so lanes stay apart."""
    scripts: dict[str, list[Any]] = {
        "fake:main": [
            _lanes_tasks(
                *_LANES_TASK_TEXT[task_text],
                _LANES_SUBAGENTS.get(task_text, ("a", "b")),
            ),
            "Both answered.",
        ],
        "fake:a": ["A-answer"],
        "fake:b": ["B-answer"],
    }
    cursors: dict[str, int] = {}

    def provider(name: str) -> FakeModel:
        i = cursors.get(name, 0)
        cursors[name] = i + 1
        script = scripts[name]
        return FakeModel([script[i % len(script)]])

    return provider


class _LaneAWaitsForB(AgentMiddleware):
    """Main-agent middleware: when ``wait``, task a starts only once lane b's model call is in flight.

    Stands in for any change that reorders parallel sub-agents' model calls (a
    LangGraph or Deep Agents upgrade, an ``await`` added in middleware). It
    adds no command of its own.
    """

    def __init__(self, wait: bool) -> None:
        super().__init__()
        self.wait = wait
        self.b_started = False

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        if self.wait and request.tool_call["id"] == "task-a":
            await workflow.wait_condition(lambda: self.b_started)
        return await handler(request)


class _MarkLaneBStarted(AgentMiddleware):
    """Lane b's middleware: tells the main agent's middleware its model call started."""

    def __init__(self, gate: _LaneAWaitsForB) -> None:
        super().__init__()
        self.gate = gate

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self.gate.b_started = True
        return await handler(request)


async def _run_lanes(reorder: bool) -> list[str]:
    gate = _LaneAWaitsForB(wait=reorder)
    agent = create_deep_agent(
        model="fake:main",
        subagents=[
            {
                "name": "a",
                "description": "Lane a.",
                "system_prompt": "Answer.",
                "model": "fake:a",
            },
            {
                "name": "b",
                "description": "Lane b.",
                "system_prompt": "Answer.",
                "model": "fake:b",
                "middleware": [_MarkLaneBStarted(gate)],
            },
        ],
        middleware=[gate],
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Ask both lanes."}]}
    )
    return sorted(
        f"{m.tool_call_id}={m.content}" for m in result["messages"] if m.type == "tool"
    )


@workflow.defn(name="ParallelLanesWorkflow")
class ParallelLanesWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        return await _run_lanes(reorder=False)


@workflow.defn(name="ParallelLanesWorkflow")
class ReorderedParallelLanesWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        return await _run_lanes(reorder=True)


@workflow.defn
class SingleAgentWorkflow:
    @workflow.run
    async def run(self) -> str:
        agent = create_deep_agent(model="fake:main")
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Just answer."}]}
        )
        return str(result["messages"][-1].content)


async def _replay(workflow_cls: type, history: WorkflowHistory) -> None:
    # Replay never calls the model; the provider only satisfies the plugin.
    plugin = DeepAgentsPlugin(model_provider=_lanes_provider())
    await Replayer(workflows=[workflow_cls], plugins=[plugin]).replay_workflow(history)


async def _model_ids_and_markers(handle: Any) -> tuple[list[str], list[str]]:
    ids, markers = [], []
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            attrs = event.activity_task_scheduled_event_attributes
            if attrs.activity_type.name == INVOKE_MODEL:
                ids.append(attrs.activity_id)
        if event.HasField("marker_recorded_event_attributes"):
            details = event.marker_recorded_event_attributes.details
            markers += [p.data.decode() for v in details.values() for p in v.payloads]
    return ids, markers


@pytest.mark.asyncio
@pytest.mark.parametrize("task_text", ["different", "same", "oversized", "customers"])
async def test_reordered_parallel_subagents_fail_replay(
    env: WorkflowEnvironment, task_text: str
) -> None:
    async with Worker(
        env.client,
        task_queue=f"da-parallel-lanes-{task_text}",
        workflows=[ParallelLanesWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_lanes_provider(task_text))],
    ):
        handle = await env.client.start_workflow(
            ParallelLanesWorkflow.run,
            id=f"da-parallel-lanes-{uuid.uuid4()}",
            task_queue=f"da-parallel-lanes-{task_text}",
        )
        answer_a = "B-answer" if task_text == "customers" else "A-answer"
        assert await handle.result() == [f"task-a={answer_a}", "task-b=B-answer"]
        history = await handle.fetch_history()
    await _replay(ParallelLanesWorkflow, history)
    with pytest.raises(NondeterminismError, match="Activity id"):
        await _replay(ReorderedParallelLanesWorkflow, history)

    ids, markers = await _model_ids_and_markers(handle)
    # The overlap recorded the patch marker (so the sequential test's check
    # for its absence can see one).
    assert [m for m in markers if _PATCH_ID in m]
    # The second of the two overlapping lane calls is named; sequential calls are not.
    assert [i.startswith("deepagents.model:") for i in ids] == [
        False,
        False,
        True,
        False,
    ]


@pytest.mark.asyncio
async def test_sequential_model_calls_keep_sequence_ids(
    env: WorkflowEnvironment,
) -> None:
    async with Worker(
        env.client,
        task_queue="da-single-agent",
        workflows=[SingleAgentWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=mock_model_provider(["Done."]))],
    ):
        handle = await env.client.start_workflow(
            SingleAgentWorkflow.run,
            id=f"da-single-agent-{uuid.uuid4()}",
            task_queue="da-single-agent",
        )
        assert await handle.result() == "Done."
    ids, markers = await _model_ids_and_markers(handle)
    assert ids == ["1"]
    assert not [m for m in markers if _PATCH_ID in m]


@pytest.mark.asyncio
async def test_sequence_id_history_still_replays() -> None:
    """A history recorded before ``deepagents.parallel-model-call-ids`` replays.

    Fixture provenance: ``ParallelLanesWorkflow`` run on a local dev server with
    ai-integrations ``main`` at 63d7aec, whose model Activities all have
    sequence ids. Delete this test and the fixture together when the patch id
    goes through ``workflow.deprecate_patch``.
    """
    history = WorkflowHistory.from_json("parallel-lanes", _LANES_HISTORY.read_text())
    await _replay(ParallelLanesWorkflow, history)


# Summarization offloads a sub-agent's older messages to a history file whose
# name and ``## Summarized at <now>`` heading change on every run; an unchanged
# workflow whose sub-agent read that file back must still replay.
_HISTORY_FILE = re.compile(r"/conversation_history/session_[0-9a-f]{32}\.md")


class _ReadsSummarizedHistory(FakeModel):
    """Lists files until summarization offloads its history, reads that file, then answers."""

    answer: str = ""

    def __init__(self, answer: str) -> None:
        super().__init__(["unused"], answer=answer)

    def _respond(self, messages: list[Any]) -> AIMessage:
        if any(
            m.type == "tool" and "Summarized at" in str(m.content) for m in messages
        ):
            return AIMessage(content=self.answer)
        paths = [p for m in messages for p in _HISTORY_FILE.findall(str(m.content))]
        tool_call_id = f"call-{uuid.uuid4().hex[:8]}"
        if paths:
            call = {"name": "read_file", "args": {"file_path": paths[0]}}
        else:
            call = {"name": "ls", "args": {"path": "/"}}
        return AIMessage(content="", tool_calls=[{**call, "id": tool_call_id}])

    def _generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    async def _agenerate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        return self._generate(messages)


def _summarized_lanes_provider() -> Any:
    main_calls = 0

    def provider(name: str) -> FakeModel:
        nonlocal main_calls
        if name == "fake:main":
            main_calls += 1
            if main_calls == 1:
                return FakeModel([_lanes_tasks("Ask lane a.", "Ask lane b.")])
            return FakeModel(["Both answered."])
        if name == "fake:summarizer":
            return FakeModel(["Listed the root."])
        return _ReadsSummarizedHistory(f"{name}-answer")

    return provider


@workflow.defn
class SummarizedLanesWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        def lane(name: str) -> dict[str, Any]:
            return {
                "name": name,
                "description": f"Lane {name}.",
                "system_prompt": "Answer.",
                "model": f"fake:{name}",
                "middleware": [
                    SummarizationMiddleware(
                        "fake:summarizer",
                        backend=StateBackend(),
                        trigger=("messages", 8),
                        keep=("messages", 2),
                    )
                ],
            }

        agent = create_deep_agent(model="fake:main", subagents=[lane("a"), lane("b")])
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Ask both lanes."}]}
        )
        return sorted(
            f"{m.tool_call_id}={m.content}"
            for m in result["messages"]
            if m.type == "tool"
        )


@pytest.mark.asyncio
async def test_parallel_subagents_reading_summarized_history_replay(
    env: WorkflowEnvironment,
) -> None:
    async with Worker(
        env.client,
        task_queue="da-summarized-lanes",
        workflows=[SummarizedLanesWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_summarized_lanes_provider())],
    ):
        handle = await env.client.start_workflow(
            SummarizedLanesWorkflow.run,
            id=f"da-summarized-lanes-{uuid.uuid4()}",
            task_queue="da-summarized-lanes",
        )
        assert await handle.result() == [
            "task-a=fake:a-answer",
            "task-b=fake:b-answer",
        ]
        history = await handle.fetch_history()

    # The lanes' calls overlapped, so some were named; none of the named ones
    # carried the summarized history a lane read back, which reads differently
    # on replay.
    named_inputs = [
        b"".join(
            p.data for p in e.activity_task_scheduled_event_attributes.input.payloads
        )
        for e in history.events
        if e.HasField("activity_task_scheduled_event_attributes")
        and e.activity_task_scheduled_event_attributes.activity_type.name
        == INVOKE_MODEL
        and e.activity_task_scheduled_event_attributes.activity_id.startswith(
            "deepagents.model:"
        )
    ]
    assert named_inputs
    assert not [data for data in named_inputs if b"## Summarized at" in data]
    await Replayer(
        workflows=[SummarizedLanesWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_summarized_lanes_provider())],
    ).replay_workflow(history)


# ---------------------------------------------------------------------------
# The call identity itself
# ---------------------------------------------------------------------------


def _identity(messages: list[Any]) -> str | None:
    from temporalio.deepagents._model import TemporalModel

    return TemporalModel._call_identity(messages)


def _turn(result: str, *, ids: bool = False) -> list[Any]:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    call = {"name": "ls", "args": {"path": "/"}, "id": "tc-1"}
    return [
        SystemMessage(content="You help."),
        HumanMessage(content="List files.", id="h-1" if ids else None),
        AIMessage(content="", tool_calls=[call], id="a-1" if ids else None),
        ToolMessage(content=result, tool_call_id="tc-1"),
        AIMessage(content="", tool_calls=[{**call, "id": "tc-2"}]),
        ToolMessage(content="same last step", tool_call_id="tc-2"),
    ]


def test_call_identity_covers_the_whole_transcript() -> None:
    """Two calls that differ only before their latest step are told apart."""
    assert _identity(_turn("one")) != _identity(_turn("two"))


def test_call_identity_ignores_message_ids_and_system_prompt() -> None:
    """Message ids (uuid4, not replay-stable) and system prompts are left out."""
    from langchain_core.messages import SystemMessage

    edited = [SystemMessage(content="Edited prompt."), *_turn("one")[1:]]
    assert _identity(_turn("one")) == _identity(_turn("one", ids=True))
    assert _identity(_turn("one")) == _identity(edited)


def test_call_identity_ignores_offloaded_file_names() -> None:
    """Random names offloading middleware writes into messages are left out.

    Deep Agents' filesystem middleware evicts a large human message to a file
    named with ``uuid4`` and records the path in the content and in
    ``additional_kwargs["lc_evicted_to"]``; summarization does the same with a
    ``session_<uuid4 hex>`` file. Neither replays identically.
    """
    import uuid as _uuid

    from langchain_core.messages import HumanMessage

    def offloaded(preview: str) -> list[Any]:
        path = f"/conversation_history/{_uuid.uuid4()}.md"
        session = f"/conversation_history/session_{_uuid.uuid4().hex}.md"
        return [
            HumanMessage(
                content=f"{preview} The full content was saved to {path}.",
                additional_kwargs={"lc_evicted_to": path},
            ),
            HumanMessage(content=f"Summary. History saved to {session}."),
            *_turn("one")[2:],
        ]

    assert _identity(offloaded("Ask lane a.")) == _identity(offloaded("Ask lane a."))
    assert _identity(offloaded("Ask lane a.")) != _identity(offloaded("Ask lane b."))


def test_call_identity_keeps_uuids_in_request_data() -> None:
    """A UUID the caller supplies, outside a middleware-generated path, counts."""
    from langchain_core.messages import HumanMessage

    def lookup(customer: str) -> list[Any]:
        call = {"name": "lookup", "args": {"customer": customer}, "id": "tc-1"}
        return [
            HumanMessage(content=f"Look up customer {customer}."),
            AIMessage(content="", tool_calls=[call]),
        ]

    first = "11111111-1111-4111-8111-111111111111"
    second = "22222222-2222-4222-8222-222222222222"
    assert _identity(lookup(first)) != _identity(lookup(second))
    # Only the name middleware gave the file is normalized, not the folder's
    # other contents.
    assert _identity(
        [HumanMessage(content=f"Read /conversation_history/{first}.txt")]
    ) != _identity([HumanMessage(content=f"Read /conversation_history/{second}.txt")])


def test_call_identity_falls_back_after_reading_an_offloaded_file() -> None:
    """A call that read back a file offloading middleware wrote keeps its sequence id.

    On replay the middleware writes the file under a new name, and
    summarization with a new ``## Summarized at`` heading, so what the tool
    returned differs from the original run.
    """
    from langchain_core.messages import ToolMessage

    def read(path: str) -> list[Any]:
        call = {"name": "read_file", "args": {"file_path": path}, "id": "tc-1"}
        return [
            *_turn("one")[1:],
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(
                content="## Summarized at 2026-11-03T13:19:49+00:00",
                tool_call_id="tc-1",
            ),
        ]

    session = "/conversation_history/session_" + "0" * 32 + ".md"
    assert _identity(read(session)) is None
    assert _identity(read("/large_tool_results/unknown-0123abcd")) is None
    # A large tool result offloaded under its tool call id replays identically.
    assert _identity(read("/large_tool_results/tc-1")) is not None


def test_call_identity_falls_back_when_the_transcript_is_not_stable() -> None:
    """A value dumpd can only repr, or JSON cannot hold, keeps the sequence id."""

    class Opaque:
        pass

    with_repr = [
        *_turn("one"),
        AIMessage(content="x", additional_kwargs={"o": Opaque()}),
    ]
    mixed_keys = [
        *_turn("one"),
        AIMessage(content="x", additional_kwargs={1: "a", "b": 2}),
    ]
    assert _identity(_turn("one")) is not None
    assert _identity(with_repr) is None
    assert _identity(mixed_keys) is None


def test_call_identity_falls_back_when_dumping_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from temporalio.deepagents import _serde

    def broken(_obj: Any) -> Any:
        raise RecursionError("too deep")

    monkeypatch.setattr(_serde, "dump_object", broken)
    assert _identity(_turn("one")) is None


def test_caller_activity_id_is_kept() -> None:
    """A caller that sets ``activity_id`` keeps it; the identity is not asked."""
    from temporalio.deepagents.workflow import _model_activity_id, _ModelDispatches

    state = _ModelDispatches(in_flight=1, named=True)

    def never() -> str:
        raise AssertionError("identity computed for a caller-set activity_id")

    assert _model_activity_id(state, never, {"activity_id": "mine"}) is None
    assert _model_activity_id(state, lambda: "x", {}) == "x"
    assert _model_activity_id(state, lambda: "x", {}) == "x#2"
    assert _model_activity_id(state, lambda: None, {}) is None
