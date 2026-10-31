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

import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.testing import WorkflowEnvironment

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)

pytest.importorskip("deepagents")
pytest.importorskip("langchain_core")

from temporalio import workflow  # noqa: E402
from temporalio.client import WorkflowHistory  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402
from temporalio.workflow import NondeterminismError  # noqa: E402
from tests.helpers.deepagents import count_scheduled_activities  # noqa: E402

# Bind deepagents symbols off the module importorskip returns: a static
# `from deepagents import ...` cannot resolve on Python 3.10 (deepagents
# needs >= 3.11), and with the package absent the type checkers mis-resolve
# the name against this same-named test directory.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

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


def _lanes_tasks(description_a: str, description_b: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "id": "task-a",
                "args": {"description": description_a, "subagent_type": "a"},
            },
            {
                "name": "task",
                "id": "task-b",
                "args": {"description": description_b, "subagent_type": "b"},
            },
        ],
    )


# With the same task text for both sub-agents, only the sub-agent's name tells
# their model calls apart.
_LANES_TASK_TEXT = {
    "different": ("Ask lane a.", "Ask lane b."),
    "same": ("Ask a lane.", "Ask a lane."),
}
_LANES_HISTORY = (
    Path(__file__).parent / "histories" / "parallel_subagents_sequence_ids.json"
)
_PATCH_ID = "deepagents.parallel-model-call-ids"


def _lanes_provider(task_text: str = "different") -> Any:
    """A model provider with one script per model name, so lanes stay apart."""
    scripts: dict[str, list[Any]] = {
        "fake:main": [_lanes_tasks(*_LANES_TASK_TEXT[task_text]), "Both answered."],
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
    """Main-agent middleware: when ``wait``, lane a starts only once lane b's model call is in flight.

    Stands in for any change that reorders parallel sub-agents' model calls (a
    LangGraph or Deep Agents upgrade, an ``await`` added in middleware). It
    adds no command of its own.
    """

    def __init__(self, wait: bool) -> None:
        super().__init__()
        self.wait = wait
        self.b_started = False

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        if self.wait and request.tool_call["args"].get("subagent_type") == "a":
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
@pytest.mark.parametrize("task_text", ["different", "same"])
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
        assert await handle.result() == ["task-a=A-answer", "task-b=B-answer"]
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
