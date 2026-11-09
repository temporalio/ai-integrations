"""Compare the checkpoint driver with native LangGraph invocation semantics."""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import copy
import dataclasses
import time
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
from deepagents import create_deep_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallLimitMiddleware,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import NotRequired

from temporalio import workflow
from temporalio.deepagents import DeepAgentsWorkflowError, _checkpoint, _serde
from temporalio.deepagents.testing import FakeModel


class _ContinueAsNew(BaseException):
    def __init__(self, args: list[Any]) -> None:
        super().__init__()
        self.input, self.snapshot = args


@pytest.fixture(autouse=True)
def workflow_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate Temporal's boundary decisions without making provider calls."""
    monkeypatch.setattr(workflow, "uuid4", lambda: uuid.UUID(int=42))
    monkeypatch.setattr(
        workflow,
        "info",
        lambda: SimpleNamespace(
            is_continue_as_new_suggested=lambda: False,
            get_current_history_length=lambda: 100,
        ),
    )

    def continue_as_new(*, args: list[Any]) -> None:
        raise _ContinueAsNew(args)

    monkeypatch.setattr(workflow, "continue_as_new", continue_as_new)


def _tool_reply(index: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": f"ls-{index}"}],
    )


async def _drive(
    factory: Callable[[], Any],
    *,
    rollover: bool,
    input: Any = None,
    snapshot: Any = None,
) -> tuple[Any, list[Any]]:
    if input is None:
        input = {"messages": [{"role": "user", "content": "Go."}]}
    carries: list[Any] = []
    for _ in range(20):
        try:
            result = await _checkpoint.run(
                factory(),
                input,
                continue_as_new_after=1 if rollover else None,
                state_snapshot=snapshot,
            )
            return result, carries
        except _ContinueAsNew as continued:
            converter = _serde.build_data_converter(None)
            input, snapshot = await converter.decode(
                await converter.encode([continued.input, continued.snapshot])
            )
            carries.append(snapshot)
    pytest.fail("The driver repeated work across 20 rollovers.")


@pytest.mark.parametrize("tools", [0, 1, 4, 8])
@pytest.mark.parametrize("rollover", [False, True])
@pytest.mark.parametrize("limit_offset", [-1, 0])
async def test_exact_recursion_budget_matches_ainvoke(
    tools: int, rollover: bool, limit_offset: int
) -> None:
    replies = [_tool_reply(index) for index in range(tools)] + ["Done."]
    limit = 3 + 2 * tools + limit_offset
    direct_model = FakeModel(replies)
    direct = create_deep_agent(model=direct_model).with_config(
        {"recursion_limit": limit}
    )
    model = FakeModel(replies)

    def factory() -> Any:
        return create_deep_agent(model=model).with_config({"recursion_limit": limit})

    if limit_offset < 0:
        with pytest.raises(GraphRecursionError):
            await direct.ainvoke({"messages": [{"role": "user", "content": "Go."}]})
        with pytest.raises(DeepAgentsWorkflowError, match="recursion limit"):
            await _drive(factory, rollover=rollover)
    else:
        expected = await direct.ainvoke(
            {"messages": [{"role": "user", "content": "Go."}]}
        )
        actual, carries = await _drive(factory, rollover=rollover)
        assert actual["messages"][-1].content == expected["messages"][-1].content
        assert model._cursor == direct_model._cursor == tools + 1
        assert bool(carries) == (rollover and tools > 0)


@pytest.mark.parametrize("rollover", [False, True])
async def test_invocation_model_limit_survives_boundaries(rollover: bool) -> None:
    model = FakeModel([_tool_reply(0)])

    def factory() -> Any:
        return create_deep_agent(
            model=model,
            middleware=[ModelCallLimitMiddleware(run_limit=3, exit_behavior="end")],
        )

    _, carries = await _drive(factory, rollover=rollover)
    assert model._cursor == 3
    assert bool(carries) == rollover


async def test_new_invocation_resets_untracked_model_limit() -> None:
    model = FakeModel([_tool_reply(index) for index in range(4)])
    saver = InMemorySaver()
    agent = create_deep_agent(
        model=model,
        checkpointer=saver,
        middleware=[ModelCallLimitMiddleware(run_limit=2, exit_behavior="end")],
    ).with_config({"configurable": {"thread_id": "conversation"}})
    await _drive(lambda: agent, rollover=False)
    await _drive(lambda: agent, rollover=False)
    assert model._cursor == 4


@pytest.mark.parametrize("rollover", [False, True])
async def test_bound_checkpoint_id_does_not_repeat_work(rollover: bool) -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    saver = InMemorySaver()
    agent = create_deep_agent(model=model, checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": "history"}}
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Go."}]},
        config,
        interrupt_before=["model"],
    )
    saved = await saver.aget_tuple(config)
    assert saved is not None
    agent = agent.with_config(saved.config)
    original_config = copy.deepcopy(agent.config)
    try:
        result = await _checkpoint.run(
            agent,
            None,
            continue_as_new_after=1 if rollover else None,
            state_snapshot=None,
        )
    except _ContinueAsNew as continued:
        result, _ = await _drive(
            lambda: create_deep_agent(model=model, checkpointer=InMemorySaver()),
            rollover=rollover,
            input=continued.input,
            snapshot=continued.snapshot,
        )
    assert model._cursor == 2
    assert result["messages"][-1].content == "Done."
    assert agent.config == original_config


async def test_uuid_thread_id_survives_rollover() -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    thread_id = uuid.UUID(int=73)

    def factory() -> Any:
        return create_deep_agent(model=model, checkpointer=InMemorySaver()).with_config(
            {"configurable": {"thread_id": thread_id}}
        )

    result, carries = await _drive(factory, rollover=True)
    assert result["messages"][-1].content == "Done."
    assert model._cursor == 2
    assert carries[0][_checkpoint.SNAPSHOT_KEY]["thread_id"] == str(thread_id)


async def test_checkpoint_ordering_survives_worker_clock_skew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langgraph.checkpoint.base.id as checkpoint_id

    model = FakeModel([_tool_reply(0), _tool_reply(1), "Done."])

    def factory() -> Any:
        return create_deep_agent(model=model)

    with pytest.raises(_ContinueAsNew) as caught:
        await _checkpoint.run(
            factory(),
            {"messages": [{"role": "user", "content": "Go."}]},
            continue_as_new_after=1,
            state_snapshot=None,
        )
    clock = time.time_ns() - 5_000_000_000
    monkeypatch.setattr(checkpoint_id, "_last_v6_timestamp", None)
    monkeypatch.setattr(checkpoint_id, "time", SimpleNamespace(time_ns=lambda: clock))
    result, _ = await _drive(
        factory,
        rollover=True,
        input=caught.value.input,
        snapshot=caught.value.snapshot,
    )
    assert model._cursor == 3
    assert [message.type for message in result["messages"]] == [
        "human",
        "ai",
        "tool",
        "ai",
        "tool",
        "ai",
    ]
    assert result["messages"][-1].content == "Done."


class Author(BaseModel):
    """A state type reachable only through a generic field."""

    name: str


class Report(BaseModel):
    """Nested typed state that must survive serialization."""

    authors: list[Author]


class ReportState(AgentState):
    report: NotRequired[Report]


class ReportMiddleware(AgentMiddleware):
    state_schema = ReportState

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return {"report": Report(authors=[Author(name="Ada")])}

    async def abefore_model(self, state: Any, runtime: Any) -> None:
        assert isinstance(state["report"], Report)
        assert isinstance(state["report"].authors[0], Author)


async def test_nested_schema_types_survive_rollover() -> None:
    model = FakeModel([_tool_reply(0), _tool_reply(1), "Done."])

    def factory() -> Any:
        return create_deep_agent(model=model, middleware=[ReportMiddleware()])

    result, carries = await _drive(factory, rollover=True)
    assert len(carries) == 2
    assert result["report"].authors[0].name == "Ada"


class ManagedState(TypedDict):
    remaining: RemainingSteps
    value: int


async def test_managed_channels_do_not_break_serialization() -> None:
    def model(state: ManagedState) -> dict[str, int]:
        return {"value": state["value"] + 1}

    def route(state: ManagedState) -> str:
        return "tools" if state["value"] < 3 else END

    def factory() -> Any:
        graph = StateGraph(ManagedState)
        graph.add_node("model", model)
        graph.add_node("tools", lambda state: {})
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", route)
        graph.add_edge("tools", "model")
        return graph.compile()

    result, carries = await _drive(factory, rollover=True, input={"value": 0})
    assert result["value"] == 3
    assert len(carries) == 2


class ExtraState(AgentState):
    extra: Any


class ExtraMiddleware(AgentMiddleware):
    state_schema = ExtraState


@pytest.mark.parametrize("rollover", [False, True])
async def test_local_state_is_preserved_or_rejected_explicitly(rollover: bool) -> None:
    @dataclasses.dataclass
    class LocalValue:
        text: str

    model = FakeModel([_tool_reply(0), "Done."])

    def factory() -> Any:
        return create_deep_agent(model=model, middleware=[ExtraMiddleware()])

    input = {
        "messages": [{"role": "user", "content": "Go."}],
        "extra": LocalValue(text="value"),
    }
    if rollover:
        with pytest.raises(DeepAgentsWorkflowError, match="Checkpoint state type"):
            await _drive(factory, rollover=True, input=input)
    else:
        result, _ = await _drive(factory, rollover=False, input=input)
        assert isinstance(result["extra"], LocalValue)


async def test_input_files_are_not_duplicated_in_rollover_payloads() -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    input = {
        "messages": [{"role": "user", "content": "Go."}],
        "files": {
            "/large.txt": {
                "content": ["x" * 1_100_000],
                "created_at": "2026-01-01",
                "modified_at": "2026-01-01",
            }
        },
        "configuration": "retained",
    }
    original = copy.deepcopy(input)
    with pytest.raises(_ContinueAsNew) as caught:
        await _checkpoint.run(
            create_deep_agent(model=model),
            input,
            continue_as_new_after=1,
            state_snapshot=None,
        )
    assert caught.value.input == {"configuration": "retained"}
    converter = _serde.build_data_converter(None)
    payloads = await converter.encode([caught.value.input, caught.value.snapshot])
    assert sum(len(payload.data) for payload in payloads) < 2 * 1024 * 1024
    assert input == original


async def test_legacy_transcript_and_files_seed_new_driver() -> None:
    snapshot = {
        "messages": _serde.dump_carried_messages([HumanMessage(content="Old.")]),
        "files": {
            "/old.txt": {
                "content": ["old"],
                "created_at": "2026-01-01",
                "modified_at": "2026-01-01",
            }
        },
    }
    model = FakeModel([_tool_reply(0), "Done."])
    result, carries = await _drive(
        lambda: create_deep_agent(model=model),
        rollover=True,
        input={"messages": [{"role": "user", "content": "New."}]},
        snapshot=snapshot,
    )
    assert carries
    assert [message.content for message in result["messages"][:2]] == ["Old.", "New."]
    assert "/old.txt" in result["files"]


async def test_hitl_configuration_fails_before_model_call() -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    agent = create_deep_agent(model=model, interrupt_on={"ls": True})
    with pytest.raises(DeepAgentsWorkflowError, match="Application interrupts require"):
        await _drive(lambda: agent, rollover=False)
    assert model._cursor == 0


async def test_hitl_resume_uses_callers_saver_after_rollover() -> None:
    model = FakeModel(
        [
            _tool_reply(0),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "/notes.md"},
                        "id": "read",
                    }
                ],
            ),
            _tool_reply(2),
            "Done.",
        ]
    )
    agents = []

    def factory() -> Any:
        agent = create_deep_agent(
            model=model, checkpointer=InMemorySaver(), interrupt_on={"read_file": True}
        ).with_config({"configurable": {"thread_id": "approval"}})
        agents.append(agent)
        return agent

    result, carries = await _drive(factory, rollover=True)
    assert carries and result.get("__interrupt__")
    caller = agents[-1]
    assert await caller.aget_state({"configurable": {"thread_id": "approval"}})
    try:
        result = await _checkpoint.run(
            caller,
            Command(resume={"decisions": [{"type": "approve"}]}),
            continue_as_new_after=1,
            state_snapshot=None,
        )
    except _ContinueAsNew as continued:
        result, more_carries = await _drive(
            factory, rollover=True, input=continued.input, snapshot=continued.snapshot
        )
        assert len(more_carries) == 0
    else:
        pytest.fail("The resumed invocation did not roll over.")
    assert result["messages"][-1].content == "Done."
    assert model._cursor == 4


@pytest.mark.parametrize("mode", ["saver", "namespace", "recursion"])
async def test_invalid_driver_configuration_fails_cleanly(mode: str) -> None:
    model = FakeModel(["Unused."])
    agent = create_deep_agent(
        model=model, checkpointer=False if mode == "saver" else None
    )
    if mode == "namespace":
        agent = agent.with_config({"configurable": {"checkpoint_ns": "nested"}})
    elif mode == "recursion":
        agent = agent.with_config({"recursion_limit": -1})
    with pytest.raises(DeepAgentsWorkflowError):
        await _drive(lambda: agent, rollover=False)
    assert model._cursor == 0


@pytest.mark.parametrize(
    "field", ["checkpoint", "thread_id", "metadata", "remaining_steps"]
)
async def test_malformed_snapshot_fails_before_more_model_work(field: str) -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    with pytest.raises(_ContinueAsNew) as caught:
        await _checkpoint.run(
            create_deep_agent(model=model),
            {"messages": [{"role": "user", "content": "Go."}]},
            continue_as_new_after=1,
            state_snapshot=None,
        )
    snapshot = caught.value.snapshot
    del snapshot[_checkpoint.SNAPSHOT_KEY][field]
    with pytest.raises(
        DeepAgentsWorkflowError, match="Invalid Deep Agents checkpoint snapshot"
    ):
        await _drive(
            lambda: create_deep_agent(model=model),
            rollover=True,
            snapshot=snapshot,
        )
    assert model._cursor == 1


async def test_agent_programming_errors_keep_their_original_type() -> None:
    class BrokenMiddleware(AgentMiddleware):
        async def abefore_model(self, state: Any, runtime: Any) -> None:
            raise ValueError("agent programming error")

    model = FakeModel(["Unused."])
    with pytest.raises(ValueError, match="agent programming error"):
        await _drive(
            lambda: create_deep_agent(model=model, middleware=[BrokenMiddleware()]),
            rollover=False,
        )
    assert model._cursor == 0
