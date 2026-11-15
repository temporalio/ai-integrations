"""Compare the checkpoint driver with native LangGraph invocation semantics."""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import copy
import dataclasses
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum, StrEnum
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict, cast

import pytest
from deepagents import create_deep_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallLimitMiddleware,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.types import Command
from pydantic import BaseModel, Field, PrivateAttr
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
    if input is None and snapshot is None:
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
    native_saver = InMemorySaver()
    native = create_deep_agent(
        name="test-agent", model=FakeModel(replies), checkpointer=native_saver
    ).with_config({"configurable": {"thread_id": "native"}})
    await native.ainvoke({"messages": [{"role": "user", "content": "Go."}]})
    native_checkpoint = await native_saver.aget_tuple(native.config or {})
    assert native_checkpoint is not None
    limit = native_checkpoint.metadata.get("step", -1) + 1 + limit_offset
    direct_model = FakeModel(replies)
    direct = create_deep_agent(name="test-agent", model=direct_model).with_config(
        {"recursion_limit": limit}
    )
    model = FakeModel(replies)

    def factory() -> Any:
        return create_deep_agent(name="test-agent", model=model).with_config(
            {"recursion_limit": limit}
        )

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
            name="test-agent",
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
        name="test-agent",
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
    agent = create_deep_agent(name="test-agent", model=model, checkpointer=saver)
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
            lambda: create_deep_agent(
                name="test-agent", model=model, checkpointer=InMemorySaver()
            ),
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
        return create_deep_agent(
            name="test-agent", model=model, checkpointer=InMemorySaver()
        ).with_config({"configurable": {"thread_id": thread_id}})

    result, carries = await _drive(factory, rollover=True)
    assert result["messages"][-1].content == "Done."
    assert model._cursor == 2
    assert carries[0][_checkpoint.SNAPSHOT_KEY]["thread_id"] == str(thread_id)


async def test_checkpoint_ordering_survives_worker_clock_skew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langgraph.checkpoint.base.id as checkpoint_id

    model = FakeModel([_tool_reply(0), _tool_reply(1), "Done.", "Next done."])

    agents = []

    def factory() -> Any:
        agent = create_deep_agent(
            name="test-agent", model=model, checkpointer=InMemorySaver()
        ).with_config({"configurable": {"thread_id": "clock-skew"}})
        agents.append(agent)
        return agent

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
    second = await _checkpoint.run(
        agents[-1],
        {"messages": [{"role": "user", "content": "Next."}]},
        continue_as_new_after=None,
        state_snapshot=None,
    )
    assert model._cursor == 4
    assert second["messages"][-2].content == "Next."
    assert second["messages"][-1].content == "Next done."


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
        return create_deep_agent(
            name="test-agent", model=model, middleware=[ReportMiddleware()]
        )

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
        return graph.compile(name="test-agent")

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
        return create_deep_agent(
            name="test-agent", model=model, middleware=[ExtraMiddleware()]
        )

    input = {
        "messages": [{"role": "user", "content": "Go."}],
        "extra": LocalValue(text="value"),
    }
    with pytest.raises(DeepAgentsWorkflowError, match="Checkpoint state type"):
        await _drive(factory, rollover=rollover, input=input)
    assert model._cursor == 0


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
            create_deep_agent(name="test-agent", model=model),
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
        lambda: create_deep_agent(name="test-agent", model=model),
        rollover=True,
        input={"messages": [{"role": "user", "content": "New."}]},
        snapshot=snapshot,
    )
    assert carries
    assert [message.content for message in result["messages"][:2]] == ["Old.", "New."]
    assert "/old.txt" in result["files"]


async def test_hitl_configuration_fails_before_model_call() -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    agent = create_deep_agent(name="test-agent", model=model, interrupt_on={"ls": True})
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
            name="test-agent",
            model=model,
            checkpointer=InMemorySaver(),
            interrupt_on={"read_file": True},
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
            continue_as_new_input={},
            continue_as_new_after=1,
            state_snapshot=None,
        )
    except _ContinueAsNew as continued:
        result, more_carries = await _drive(
            factory, rollover=True, input=continued.input, snapshot=continued.snapshot
        )
        assert len(more_carries) == 1
    else:
        pytest.fail("The resumed invocation did not roll over.")
    assert result["messages"][-1].content == "Done."
    assert model._cursor == 4


@pytest.mark.parametrize("mode", ["saver", "namespace", "recursion"])
async def test_invalid_driver_configuration_fails_cleanly(mode: str) -> None:
    model = FakeModel(["Unused."])
    agent = create_deep_agent(
        name="test-agent", model=model, checkpointer=False if mode == "saver" else None
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
            create_deep_agent(name="test-agent", model=model),
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
            lambda: create_deep_agent(name="test-agent", model=model),
            rollover=True,
            snapshot=snapshot,
            input=caught.value.input,
        )
    assert model._cursor == 1


async def test_agent_programming_errors_keep_their_original_type() -> None:
    class BrokenMiddleware(AgentMiddleware):
        async def abefore_model(self, state: Any, runtime: Any) -> None:
            raise ValueError("agent programming error")

    model = FakeModel(["Unused."])
    with pytest.raises(ValueError, match="agent programming error"):
        await _drive(
            lambda: create_deep_agent(
                name="test-agent", model=model, middleware=[BrokenMiddleware()]
            ),
            rollover=False,
        )
    assert model._cursor == 0


async def _snapshot(model: FakeModel, **config: Any) -> _ContinueAsNew:
    agent = create_deep_agent(name="test-agent", model=model).with_config(**config)
    with pytest.raises(_ContinueAsNew) as caught:
        await _checkpoint.run(
            agent,
            {"messages": [{"role": "user", "content": "Go."}]},
            continue_as_new_after=1,
            state_snapshot=None,
        )
    return caught.value


async def test_snapshot_metadata_drops_configuration_secrets() -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    carry = await _snapshot(
        model,
        configurable={"tenant_api_key": "configuration-secret"},
        metadata={"api_key": "metadata-secret"},
    )
    saved = carry.snapshot[_checkpoint.SNAPSHOT_KEY]
    assert set(saved["metadata"]) <= {
        "source",
        "step",
        "parents",
        "counters_since_delta_snapshot",
    }
    assert "configuration-secret" not in str(carry.snapshot)
    assert "metadata-secret" not in str(carry.snapshot)
    result, _ = await _drive(
        lambda: create_deep_agent(name="test-agent", model=model),
        rollover=True,
        input=carry.input,
        snapshot=carry.snapshot,
    )
    assert result["messages"][-1].content == "Done."


@pytest.mark.parametrize("reuse", [False, True])
async def test_snapshot_rejects_new_input_or_an_existing_thread(reuse: bool) -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    carry = await _snapshot(model)
    agent = create_deep_agent(
        name="test-agent", model=model, checkpointer=InMemorySaver()
    ).with_config({"configurable": {"thread_id": "resume"}})
    if reuse:
        await _checkpoint.run(
            agent,
            carry.input,
            continue_as_new_after=None,
            state_snapshot=carry.snapshot,
        )
    cursor = model._cursor
    with pytest.raises(
        DeepAgentsWorkflowError, match="empty thread" if reuse else "new input"
    ):
        await _checkpoint.run(
            agent,
            carry.input
            if reuse
            else {"messages": [{"role": "user", "content": "New turn."}]},
            continue_as_new_after=1,
            state_snapshot=carry.snapshot,
        )
    assert model._cursor == cursor


async def test_snapshots_are_bound_to_a_named_agent() -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    carry = await _snapshot(model)
    with pytest.raises(_checkpoint.CheckpointMismatchError, match="name changed"):
        await _checkpoint.run(
            create_deep_agent(name="writer", model=model),
            carry.input,
            continue_as_new_after=1,
            state_snapshot=carry.snapshot,
        )
    with pytest.raises(DeepAgentsWorkflowError, match="unique, stable agent name"):
        await _checkpoint.run(
            create_deep_agent(model=model),
            {},
            continue_as_new_after=1,
            state_snapshot=None,
        )
    assert model._cursor == 1


async def test_checkpoint_snapshot_is_rejected_by_legacy_driver() -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    carry = await _snapshot(model)
    from temporalio.deepagents.workflow import _merge_snapshot

    agent = create_deep_agent(name="test-agent", model=model)
    with pytest.raises(ValueError, match="message type"):
        await agent.ainvoke(_merge_snapshot(carry.input, carry.snapshot))
    assert model._cursor == 1


@pytest.mark.parametrize("compressible", [False, True])
async def test_large_snapshots_are_compressed_or_fail_explicitly(
    compressible: bool,
) -> None:
    import base64
    import random

    content = (
        "x" * 2_200_000
        if compressible
        else base64.b64encode(random.Random(17).randbytes(1_600_000)).decode()
    )
    model = FakeModel([_tool_reply(0), "Done."])
    factory = lambda: create_deep_agent(name="test-agent", model=model)
    input = {"messages": [{"role": "user", "content": content}]}
    if compressible:
        result, carries = await _drive(factory, rollover=True, input=input)
        assert result["messages"][0].content == content
        payloads = await _serde.build_data_converter(None).encode([carries[0]])
        assert sum(p.ByteSize() for p in payloads) < 100_000
    else:
        with pytest.raises(DeepAgentsWorkflowError, match="payload limit"):
            await _drive(factory, rollover=True, input=input)
        assert model._cursor == 1


@pytest.mark.parametrize("thread_id", [0, ""])
async def test_falsy_thread_ids_support_interrupts(thread_id: Any) -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    agent = create_deep_agent(
        name="test-agent",
        model=model,
        checkpointer=InMemorySaver(),
        interrupt_on={"ls": True},
    ).with_config({"configurable": {"thread_id": thread_id}})
    result = await _checkpoint.run(
        agent,
        {"messages": [{"role": "user", "content": "Go."}]},
        continue_as_new_after=None,
        state_snapshot=None,
    )
    assert result["__interrupt__"]
    result = await _checkpoint.run(
        agent,
        Command(resume={"decisions": [{"type": "approve"}]}),
        continue_as_new_input={},
        continue_as_new_after=None,
        state_snapshot=None,
    )
    assert result["messages"][-1].content == "Done."


async def test_caller_saver_requires_a_thread_id() -> None:
    model = FakeModel(["Unused."])
    with pytest.raises(
        DeepAgentsWorkflowError, match="requires configurable.thread_id"
    ):
        await _drive(
            lambda: create_deep_agent(
                name="test-agent", model=model, checkpointer=InMemorySaver()
            ),
            rollover=False,
        )
    assert model._cursor == 0


async def test_command_resume_requires_workflow_input() -> None:
    model = FakeModel(["Unused."])
    with pytest.raises(DeepAgentsWorkflowError, match="Command resumes require"):
        await _checkpoint.run(
            create_deep_agent(name="test-agent", model=model),
            Command(resume="yes"),
            continue_as_new_after=1,
            state_snapshot=None,
        )
    assert model._cursor == 0


async def test_serialization_and_deepcopy_errors_fail_before_model_work() -> None:
    class Uncopyable:
        def __deepcopy__(self, memo: Any) -> Any:
            raise TypeError("Cannot copy")

    model = FakeModel(["Unused."])
    for value, message in [
        (SimpleNamespace(text="unsupported"), "not serializable"),
        (Uncopyable(), "cannot be copied"),
    ]:
        with pytest.raises(DeepAgentsWorkflowError, match=message):
            await _drive(
                lambda: create_deep_agent(
                    name="test-agent", model=model, middleware=[ExtraMiddleware()]
                ),
                rollover=False,
                input={"messages": [], "extra": value},
            )
    assert model._cursor == 0


class MappingKey(Enum):
    VALUE = 1


class StringKey(StrEnum):
    VALUE = "key"


@pytest.mark.parametrize(
    "key",
    [
        uuid.UUID(int=1),
        datetime.fromtimestamp(0, UTC),
        MappingKey.VALUE,
        StringKey.VALUE,
    ],
)
def test_lossy_mapping_keys_fail_explicitly(key: Any) -> None:
    serde = _checkpoint._serializer(
        create_deep_agent(name="test-agent", model=FakeModel(["Unused."]))
    )
    with pytest.raises(DeepAgentsWorkflowError):
        _checkpoint._round_trip(serde, {key: "value"})


def test_tuple_state_uses_native_msgpack_array_semantics() -> None:
    serde = _checkpoint._serializer(
        create_deep_agent(name="test-agent", model=FakeModel(["Unused."]))
    )
    _checkpoint._round_trip(serde, {"tuple": (1, "two", (3,))})


async def test_strict_msgpack_preserves_schema_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    await test_nested_schema_types_survive_rollover()


async def test_breakpoint_rollover_does_not_report_callback_errors() -> None:
    from langchain_core.callbacks import AsyncCallbackHandler

    errors = []

    class Handler(AsyncCallbackHandler):
        async def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
            errors.append(error)

    model = FakeModel([_tool_reply(0), _tool_reply(1), "Done."])
    result, carries = await _drive(
        lambda: create_deep_agent(name="test-agent", model=model).with_config(
            {"callbacks": [Handler()]}
        ),
        rollover=True,
    )
    assert len(carries) == 2
    assert result["messages"][-1].content == "Done."
    assert errors == []


async def test_declined_interrupt_configuration_runs_normally() -> None:
    model = FakeModel([_tool_reply(0), "Done."])
    result, _ = await _drive(
        lambda: create_deep_agent(
            name="test-agent", model=model, interrupt_on={"ls": False}
        ),
        rollover=True,
    )
    assert result["messages"][-1].content == "Done."


async def test_hitl_subclass_is_validated_before_paid_work() -> None:
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    class Approval(HumanInTheLoopMiddleware):
        pass

    model = FakeModel([_tool_reply(0), "Done."])
    with pytest.raises(DeepAgentsWorkflowError, match="Application interrupts require"):
        await _drive(
            lambda: create_deep_agent(
                name="test-agent",
                model=model,
                middleware=[Approval(interrupt_on={"ls": True})],
            ),
            rollover=False,
        )
    assert model._cursor == 0


class LossyModel(BaseModel):
    """A schema-declared model with intentionally excluded runtime fields."""

    public: int = 1
    hidden: int = Field(default=0, exclude=True)
    _private: str = PrivateAttr(default="initial")


@pytest.mark.parametrize("private", [False, True])
def test_excluded_and_private_pydantic_state_cannot_be_silently_reset(
    private: bool,
) -> None:
    value = LossyModel(hidden=0 if private else 9)
    if private:
        value._private = "changed"
    serde = _checkpoint._CheckpointSerializer(
        allowed_msgpack_modules={(LossyModel.__module__, LossyModel.__name__)}
    )
    with pytest.raises(DeepAgentsWorkflowError, match="loses"):
        _checkpoint._round_trip(serde, value)


class TransientState(AgentState):
    counter: Annotated[NotRequired[int], UntrackedValue]
    resource: Annotated[NotRequired[Any], UntrackedValue]
    cache: Annotated[NotRequired[str], UntrackedValue]


class TransientMiddleware(AgentMiddleware):
    state_schema = TransientState

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        import asyncio

        return {"counter": 0, "resource": asyncio.Lock(), "cache": "x" * 100_000}

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any]:
        return {"counter": state.get("counter", 0) + 1}


async def test_untracked_resources_are_kept_in_run_and_filtered_from_carry() -> None:
    model = FakeModel([_tool_reply(0), _tool_reply(1), "Done."])
    result, carries = await _drive(
        lambda: create_deep_agent(
            name="test-agent", model=model, middleware=[TransientMiddleware()]
        ),
        rollover=True,
    )
    assert result["counter"] == 3
    assert "resource" not in result
    assert "cache" not in result
    assert len(carries) == 2
    assert all(len(str(carry)) < 100_000 for carry in carries)


@pytest.mark.parametrize("offset", [-1, 0])
async def test_recursion_budget_includes_nodes_after_the_last_model(
    offset: int,
) -> None:
    def increment(state: ManagedState) -> dict[str, int]:
        return {"value": state["value"] + 1}

    def factory(saver: Any = None) -> Any:
        graph = StateGraph(ManagedState)
        graph.add_node("model", increment)
        graph.add_node("tools", lambda state: {})
        graph.add_node("tail", increment)
        graph.add_edge(START, "model")
        graph.add_conditional_edges(
            "model", lambda state: "tools" if state["value"] < 3 else "tail"
        )
        graph.add_edge("tools", "model")
        graph.add_edge("tail", END)
        return graph.compile(name="test-agent", checkpointer=saver)

    saver = InMemorySaver()
    native = factory(saver).with_config({"configurable": {"thread_id": "native"}})
    expected = await native.ainvoke({"value": 0})
    saved = await saver.aget_tuple(native.config)
    assert saved is not None
    limit = saved.metadata.get("step", -1) + 1 + offset
    limited = lambda: factory().with_config({"recursion_limit": limit})
    if offset < 0:
        with pytest.raises(GraphRecursionError):
            await limited().ainvoke({"value": 0})
        with pytest.raises(DeepAgentsWorkflowError, match="recursion limit"):
            await _drive(limited, rollover=True, input={"value": 0})
    else:
        result, carries = await _drive(limited, rollover=True, input={"value": 0})
        assert result == expected
        assert len(carries) == 2


async def test_nested_graph_inherits_the_original_recursion_limit() -> None:
    def increment(state: ManagedState) -> dict[str, int]:
        return {"value": state["value"] + 1}

    child = StateGraph(ManagedState)
    for index in range(8):
        child.add_node(str(index), lambda state: {})
        child.add_edge(START if index == 0 else str(index - 1), str(index))
    child.add_edge("7", END)
    subgraph = child.compile()

    async def tools(state: ManagedState) -> dict:
        if state["value"] == 4:
            await subgraph.ainvoke(cast(ManagedState, cast(object, {"value": 0})))
        return {}

    def factory() -> Any:
        graph = StateGraph(ManagedState)
        graph.add_node("model", increment)
        graph.add_node("tools", tools)
        graph.add_edge(START, "model")
        graph.add_conditional_edges(
            "model", lambda state: "tools" if state["value"] < 5 else END
        )
        graph.add_edge("tools", "model")
        return graph.compile(name="test-agent").with_config({"recursion_limit": 10})

    expected = await factory().ainvoke({"value": 0})
    result, carries = await _drive(factory, rollover=True, input={"value": 0})
    assert result == expected
    assert len(carries) == 4


@pytest.mark.parametrize("caller_saver", [False, True])
async def test_handbuilt_graph_preserves_langgraphs_default_limit(
    caller_saver: bool,
) -> None:
    native_limits: list[int] = []

    def model(state: ManagedState, config: RunnableConfig) -> dict[str, int]:
        native_limits.append(
            state["remaining"] + config.get("metadata", {})["langgraph_step"]
        )
        return {"value": state["value"] + 1}

    graph = StateGraph(ManagedState)
    graph.add_node("model", model)
    graph.add_node("tools", lambda state: {})
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model", lambda state: "tools" if state["value"] < 2 else END
    )
    graph.add_edge("tools", "model")
    native = graph.compile()
    await native.ainvoke(cast(ManagedState, cast(object, {"value": 0})))
    agent = graph.compile(
        name="test-agent", checkpointer=InMemorySaver() if caller_saver else None
    )
    if caller_saver:
        agent = agent.with_config({"configurable": {"thread_id": "default-limit"}})
    with pytest.raises(_ContinueAsNew) as caught:
        await _checkpoint.run(
            agent, {"value": 0}, continue_as_new_after=1, state_snapshot=None
        )
    saved = caught.value.snapshot[_checkpoint.SNAPSHOT_KEY]
    assert saved["remaining_steps"] == native_limits[0] - (
        saved["metadata"]["step"] + 1
    )


@pytest.mark.parametrize("rollover", [False, True])
async def test_managed_remaining_steps_match_native(rollover: bool) -> None:
    observed: list[int] = []

    def model(state: ManagedState) -> dict[str, int]:
        observed.append(state["remaining"])
        return {"value": state["value"] + 1}

    def factory() -> Any:
        graph = StateGraph(ManagedState)
        graph.add_node("model", model)
        graph.add_node("tools", lambda state: {})
        graph.add_edge(START, "model")
        graph.add_conditional_edges(
            "model", lambda state: "tools" if state["value"] < 3 else END
        )
        graph.add_edge("tools", "model")
        return graph.compile(name="test-agent").with_config({"recursion_limit": 8})

    expected = await factory().ainvoke({"value": 0})
    native_steps = observed.copy()
    observed.clear()
    result, _ = await _drive(factory, rollover=rollover, input={"value": 0})
    assert result == expected
    assert observed == native_steps


async def test_driver_saver_does_not_retain_intermediate_subagent_histories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    savers = []

    class RecordingSaver(_checkpoint._DriverSaver):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            savers.append(self)

    monkeypatch.setattr(_checkpoint, "_DriverSaver", RecordingSaver)
    submodel = FakeModel([*[_tool_reply(index) for index in range(30)], "Sub done."])
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Research.",
                            "subagent_type": "researcher",
                        },
                        "id": "task",
                    }
                ],
            ),
            "Done.",
        ]
    )
    factory = lambda: create_deep_agent(
        name="test-agent",
        model=model,
        subagents=[
            {
                "name": "researcher",
                "description": "Research",
                "system_prompt": "Research.",
                "model": submodel,
            }
        ],
    )
    result, _ = await _drive(factory, rollover=False)
    assert result["messages"][-1].content == "Done."
    assert submodel._cursor == 31
    nested = [
        checkpoint
        async for checkpoint in savers[0].alist(None)
        if checkpoint.config["configurable"].get("checkpoint_ns")
    ]
    assert len(nested) <= 3
