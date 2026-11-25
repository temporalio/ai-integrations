"""Graph checkpoints preserve an unfinished turn without todo-based decisions."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any
from unittest.mock import patch

import pytest
from typing_extensions import NotRequired, TypedDict

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError, WorkflowHistory
from temporalio.deepagents import (
    DeepAgentsPlugin,
    DeepAgentsWorkflowError,
    run_deep_agent,
)
from temporalio.deepagents.testing import mock_model_provider
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    TodoListMiddleware,
)
from langchain.agents.middleware.types import OmitFromInput
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel


class ResearchNote(BaseModel):
    """A typed middleware value that must survive the snapshot converter."""

    title: str


class LifecycleState(AgentState):
    started: Annotated[NotRequired[int], OmitFromInput]
    model_steps: Annotated[NotRequired[int], OmitFromInput]
    finished: Annotated[NotRequired[int], OmitFromInput]
    note: Annotated[NotRequired[ResearchNote], OmitFromInput]


class LifecycleMiddleware(AgentMiddleware):
    state_schema = LifecycleState

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return {
            "started": state.get("started", 0) + 1,
            "note": ResearchNote(title="research"),
        }

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any]:
        assert isinstance(state["note"], ResearchNote)
        return {"model_steps": state.get("model_steps", 0) + 1}

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any]:
        return {"finished": state.get("finished", 0) + 1}


@workflow.defn
class CheckpointAgentWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        middleware: list[Any] = [LifecycleMiddleware()]
        if input.get("todos"):
            middleware.append(TodoListMiddleware())
        agent = create_deep_agent(
            name="test-agent", model="fake:model", middleware=middleware
        )
        result = await run_deep_agent(
            agent,
            input,
            continue_as_new_after=input.get("limit"),
            state_snapshot=state_snapshot,
        )
        return {
            "messages": [[m.type, m.content] for m in result["messages"]],
            "reads": [
                m.content
                for m in result["messages"]
                if getattr(m, "tool_call_id", None) in {"read-1", "read-2"}
            ],
            "started": result["started"],
            "model_steps": result["model_steps"],
            "finished": result["finished"],
            "note": result["note"].title,
            "todos": result.get("todos"),
            "files": result.get("files"),
        }


def _call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def _responses(todos: bool) -> list[Any]:
    first = _call(
        "write_file", {"file_path": "/notes.md", "content": "draft"}, "write-1"
    )
    if todos:
        first.tool_calls.append(
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "research", "status": "pending"}]},
                "id": "todo-1",
                "type": "tool_call",
            }
        )
    return [
        first,
        _call("read_file", {"file_path": "/notes.md"}, "read-1"),
        _call(
            "write_file", {"file_path": "/notes.md", "content": "updated"}, "write-2"
        ),
        _call("read_file", {"file_path": "/notes.md"}, "read-2"),
        "Done.",
    ]


async def _histories(
    client: Client, workflow_id: str, run_id: str | None
) -> list[WorkflowHistory]:
    assert run_id is not None
    histories = []
    while True:
        history = await client.get_workflow_handle(
            workflow_id, run_id=run_id
        ).fetch_history()
        histories.append(history)
        continued = next(
            (
                e.workflow_execution_continued_as_new_event_attributes
                for e in history.events
                if e.HasField("workflow_execution_continued_as_new_event_attributes")
            ),
            None,
        )
        if continued is None:
            return histories
        run_id = continued.new_execution_run_id


async def _replay(workflow_type: type, histories: list[WorkflowHistory]) -> None:
    replayer = Replayer(workflows=[workflow_type], plugins=[DeepAgentsPlugin()])
    for history in histories:
        await replayer.replay_workflow(history)


async def _run(
    env: WorkflowEnvironment,
    workflow_type: Any,
    input: Any = None,
    *,
    responses: list[Any] | None = None,
    provider: Callable[[str], Any] | None = None,
    replay: bool = True,
    start_signal: str | None = None,
) -> tuple[Any, list[WorkflowHistory]]:
    """Run with cache eviction and collect and replay every run in the chain."""
    plugin = DeepAgentsPlugin(
        model_provider=provider or mock_model_provider(responses or ["Unused."])
    )
    client_config = env.client.config()
    client_config["plugins"] = [plugin]
    client = Client(**client_config)
    task_queue = str(uuid.uuid4())
    async with Worker(
        client, task_queue=task_queue, workflows=[workflow_type], max_cached_workflows=0
    ):
        handle = await client.start_workflow(
            workflow_type.run,
            args=[] if input is None else [input],
            id=task_queue,
            task_queue=task_queue,
            start_signal=start_signal,
        )
        result = await handle.result()
    histories = await _histories(client, handle.id, handle.first_execution_run_id)
    if replay:
        await _replay(workflow_type, histories)
    return result, histories


@pytest.mark.parametrize("todos", [False, True])
async def test_graph_state_and_lifecycle_cross_multiple_runs(
    env: WorkflowEnvironment, todos: bool
) -> None:
    result, histories = await _run(
        env,
        CheckpointAgentWorkflow,
        {
            "messages": [{"role": "user", "content": "Research."}],
            "todos": todos,
            "limit": 1,
        },
        responses=_responses(todos),
    )
    assert len(histories) >= 3
    assert result["started"] == result["finished"] == 1
    assert result["model_steps"] == 5
    assert result["note"] == "research"
    assert result["messages"][0] == ["human", "Research."]
    assert result["messages"][-1] == ["ai", "Done."]
    assert len(result["messages"]) == (11 if todos else 10)
    reads = result["reads"]
    assert len(reads) == 2
    assert "draft" in reads[0] and "updated" in reads[1]
    if todos:
        assert result["todos"][0]["status"] == "pending"
    else:
        assert result["todos"] is None


async def test_completed_answer_returns_even_past_limit(
    env: WorkflowEnvironment,
) -> None:
    result, histories = await _run(
        env,
        CheckpointAgentWorkflow,
        {"messages": [{"role": "user", "content": "Hello."}], "limit": 1},
        responses=["Done."],
    )
    assert result["model_steps"] == result["started"] == result["finished"] == 1
    assert len(histories) == 1


@pytest.mark.parametrize("suggested", [False, True])
async def test_checkpoint_driver_follows_server_suggestion(
    env: WorkflowEnvironment, suggested: bool
) -> None:
    with patch.object(
        workflow.Info, "is_continue_as_new_suggested", return_value=suggested
    ):
        result, histories = await _run(
            env,
            CheckpointAgentWorkflow,
            {"messages": [{"role": "user", "content": "Research."}]},
            responses=_responses(False),
        )
    assert (len(histories) > 1) == suggested
    assert result["model_steps"] == 5


@workflow.defn
class InterruptedAgentWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        agent = create_deep_agent(
            name="test-agent",
            model="fake:model",
            middleware=[LifecycleMiddleware()],
            checkpointer=InMemorySaver(),
            interrupt_on={"read_file": True},
        ).with_config({"configurable": {"thread_id": "approval"}})
        result = await run_deep_agent(
            agent,
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )
        return {
            "interrupted": bool(result.get("__interrupt__")),
            "finished": result.get("finished", 0),
        }


async def test_application_interrupt_is_returned_to_caller(
    env: WorkflowEnvironment,
) -> None:
    result, histories = await _run(
        env,
        InterruptedAgentWorkflow,
        {"messages": [{"role": "user", "content": "Read."}]},
        responses=[_call("read_file", {"file_path": "/notes.md"}, "read-1")],
    )
    assert result == {"interrupted": True, "finished": 0}
    assert len(histories) == 1


@workflow.defn
class RecursionBudgetWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> str:
        agent = create_deep_agent(name="test-agent", model="fake:model").with_config(
            {"recursion_limit": 8}
        )
        try:
            await run_deep_agent(
                agent, input, continue_as_new_after=1, state_snapshot=state_snapshot
            )
        except DeepAgentsWorkflowError:
            return "limited"
        return "unexpected completion"


async def test_recursion_budget_survives_rollovers(env: WorkflowEnvironment) -> None:
    result, histories = await _run(
        env,
        RecursionBudgetWorkflow,
        {"messages": [{"role": "user", "content": "Loop."}]},
        responses=[_call("ls", {"path": "/"}, "loop")],
    )
    assert result == "limited"
    assert 1 < len(histories) < 8


@workflow.defn
class SubagentCheckpointWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> list:
        agent = create_deep_agent(
            name="test-agent",
            model="fake:main",
            subagents=[
                {
                    "name": "researcher",
                    "description": "Researches a topic.",
                    "system_prompt": "Research.",
                    "model": "fake:sub",
                }
            ],
        )
        result = await run_deep_agent(
            agent, input, continue_as_new_after=1, state_snapshot=state_snapshot
        )
        return [[m.type, m.content] for m in result["messages"]]


async def test_subagent_finishes_before_parent_rollover(
    env: WorkflowEnvironment,
) -> None:
    main = mock_model_provider(
        [
            _call(
                "task",
                {"description": "Research.", "subagent_type": "researcher"},
                "task-1",
            ),
            "Main done.",
        ]
    )
    sub = mock_model_provider([_call("ls", {"path": "/"}, "ls-1"), "Sub done."])
    provider_calls: list[str] = []

    def provider(name: str) -> Any:
        provider_calls.append(name)
        return sub(name) if name == "fake:sub" else main(name)

    result, histories = await _run(
        env,
        SubagentCheckpointWorkflow,
        {"messages": [{"role": "user", "content": "Delegate."}]},
        provider=provider,
    )
    assert provider_calls == ["fake:main", "fake:sub", "fake:sub", "fake:main"]
    assert result == [
        ["human", "Delegate."],
        ["ai", ""],
        ["tool", "Sub done."],
        ["ai", "Main done."],
    ]
    assert len(histories) == 2


class LegacyTodoAgent:
    async def ainvoke(self, input: dict) -> dict:
        messages = [*input.get("messages", []), "step"]
        return {
            "messages": messages,
            "todos": [
                {
                    "content": "work",
                    "status": "done" if len(messages) == 3 else "pending",
                }
            ],
        }


@workflow.defn
class LegacyTodoCheckpointWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        return await run_deep_agent(
            LegacyTodoAgent(),
            input,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )


@pytest.mark.parametrize("legacy", [False, True])
async def test_todo_rollover_is_only_used_by_legacy_histories(
    env: WorkflowEnvironment, legacy: bool
) -> None:
    original_patched = workflow.patched

    def before_checkpoint_change(patch_id: str) -> bool:
        if legacy and patch_id == "deepagents.checkpoint-continue-as-new":
            return False
        return original_patched(patch_id)

    with patch.object(workflow, "patched", before_checkpoint_change):
        result, histories = await _run(
            env, LegacyTodoCheckpointWorkflow, {"messages": []}, replay=False
        )
    assert result["messages"] == ["step"] * (3 if legacy else 1)
    assert len(histories) == (3 if legacy else 1)
    # Replay outside the patch to prove old histories select the legacy driver.
    await _replay(LegacyTodoCheckpointWorkflow, histories)


@workflow.defn
class StaticBreakpointWorkflow:
    @workflow.run
    async def run(self) -> list:
        agent = create_deep_agent(name="test-agent", model="fake:model").copy(
            {"interrupt_before_nodes": ["model"]}
        )
        result = await run_deep_agent(
            agent,
            {"messages": [{"role": "user", "content": "Pause."}]},
            continue_as_new_after=1,
        )
        return [[m.type, m.content] for m in result["messages"]]


async def test_caller_breakpoint_is_not_automatically_resumed(
    env: WorkflowEnvironment,
) -> None:
    def no_model_call(name: str) -> Any:
        raise AssertionError(f"Caller breakpoint was bypassed: {name}")

    result, _ = await _run(env, StaticBreakpointWorkflow, provider=no_model_call)
    assert result == [["human", "Pause."]]


@pytest.mark.parametrize(
    "snapshot, message",
    [
        ({"version": 2}, "Unsupported Deep Agents checkpoint snapshot"),
        (
            {"version": 1, "graph": {"nodes": [], "channels": []}},
            "The graph's nodes, state channels or name changed since its checkpoint",
        ),
    ],
)
async def test_incompatible_checkpoint_is_rejected(
    snapshot: dict, message: str
) -> None:
    from temporalio.deepagents._checkpoint import SNAPSHOT_KEY, run
    from temporalio.deepagents.testing import FakeModel

    agent = create_deep_agent(name="test-agent", model=FakeModel(["unused"]))
    error_type = (
        DeepAgentsWorkflowError if snapshot.get("version") == 2 else RuntimeError
    )
    with pytest.raises(error_type, match=message):
        await run(
            agent, {}, continue_as_new_after=1, state_snapshot={SNAPSHOT_KEY: snapshot}
        )


@workflow.defn
class ApprovalAfterRolloverWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> str:
        agent = create_deep_agent(
            name="test-agent",
            model="fake:model",
            checkpointer=InMemorySaver(),
            interrupt_on={"read_file": True},
        ).with_config({"configurable": {"thread_id": workflow.info().run_id}})
        result = await run_deep_agent(
            agent, input, continue_as_new_after=1, state_snapshot=state_snapshot
        )
        while result.get("__interrupt__"):
            assert (await agent.aget_state(agent.config or {})).next
            result = await run_deep_agent(
                agent,
                Command(resume={"decisions": [{"type": "approve"}]}),
                continue_as_new_input=input,
                continue_as_new_after=1,
            )
        return result["messages"][-1].content


async def test_approval_after_rollover_resumes_and_continues_again(
    env: WorkflowEnvironment,
) -> None:
    result, histories = await _run(
        env,
        ApprovalAfterRolloverWorkflow,
        {"messages": [{"role": "user", "content": "Read."}]},
        responses=[
            _call("ls", {"path": "/"}, "ls-1"),
            _call("read_file", {"file_path": "/notes.md"}, "read-1"),
            _call("read_file", {"file_path": "/notes.md"}, "read-2"),
            _call("ls", {"path": "/"}, "ls-2"),
            "Approved.",
        ],
    )
    assert result == "Approved."
    assert len(histories) == 5


@workflow.defn
class InvalidSignatureWorkflow:
    @workflow.run
    async def run(self, input: dict) -> Any:
        return await run_deep_agent(
            create_deep_agent(name="test-agent", model="fake:model"), input
        )


class TranscriptSnapshot(TypedDict):
    messages: list[dict[str, str]]


@workflow.defn
class TranscriptSnapshotWorkflow:
    @workflow.run
    async def run(
        self, input: dict, state_snapshot: TranscriptSnapshot | None = None
    ) -> Any:
        return await run_deep_agent(
            create_deep_agent(name="test-agent", model="fake:model"),
            input,
            state_snapshot=state_snapshot,
            continue_as_new_after=1,
        )


async def test_snapshot_annotation_must_preserve_the_checkpoint(
    env: WorkflowEnvironment,
) -> None:
    with pytest.raises(WorkflowFailureError) as caught:
        await _run(
            env,
            TranscriptSnapshotWorkflow,
            {"messages": [{"role": "user", "content": "Go."}]},
            responses=[_call("ls", {"path": "/"}, "ls"), "Done."],
        )
    cause = caught.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == DeepAgentsWorkflowError.TYPE
    assert "state_snapshot parameter must decode the Deep Agents checkpoint" in str(
        cause
    )


@workflow.defn
class InvalidConfigurationWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> Any:
        mode = input["mode"]
        agent = create_deep_agent(
            name="test-agent",
            model="fake:model",
            checkpointer=False if mode == "saver" else None,
        ).with_config({"recursion_limit": 8})
        return await run_deep_agent(
            agent,
            {**input, "messages": [{"role": "user", "content": "Go."}]},
            continue_as_new_after=0 if mode == "threshold" else 1,
            state_snapshot=state_snapshot,
        )


@pytest.mark.parametrize("mode", ["signature", "saver", "threshold", "recursion"])
async def test_driver_errors_fail_workflow_execution(
    env: WorkflowEnvironment, mode: str
) -> None:
    calls = []
    looping_provider = mock_model_provider([_call("ls", {"path": "/"}, "loop")])

    def provider(name: str) -> Any:
        calls.append(name)
        return looping_provider(name)

    with pytest.raises(WorkflowFailureError) as caught:
        await _run(
            env,
            InvalidSignatureWorkflow
            if mode == "signature"
            else InvalidConfigurationWorkflow,
            {"mode": mode},
            provider=provider,
        )
    cause = caught.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == DeepAgentsWorkflowError.TYPE
    assert cause.non_retryable
    assert bool(calls) == (mode == "recursion")


@workflow.defn
class HandlerAgentWorkflow:
    def __init__(self) -> None:
        self.completed = 0

    @workflow.run
    async def run(self, _input: str) -> int:
        await workflow.wait_condition(lambda: self.completed == 2)
        return self.completed

    async def _turn(self, message: str) -> str:
        agent = create_deep_agent(model="fake:model")
        result = await run_deep_agent(
            agent,
            {"messages": [{"role": "user", "content": message}]},
            continue_as_new_after=1,
        )
        self.completed += 1
        return result["messages"][-1].content

    @workflow.update
    async def chat(self, message: str) -> str:
        return await self._turn(message)

    @workflow.signal
    async def chat_signal(self, message: str) -> None:
        await self._turn(message)


async def test_signal_and_update_turns_finish_without_rolling_over(
    env: WorkflowEnvironment,
) -> None:
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider([_call("ls", {"path": "/"}, "ls"), "Done."])
    )
    async with Worker(
        env.client,
        task_queue="checkpoint-handlers",
        workflows=[HandlerAgentWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            HandlerAgentWorkflow.run,
            "start",
            id=f"checkpoint-handlers-{uuid.uuid4()}",
            task_queue="checkpoint-handlers",
        )
        assert (
            await handle.execute_update(HandlerAgentWorkflow.chat, "Update.") == "Done."
        )
        await handle.signal(HandlerAgentWorkflow.chat_signal, "Signal.")
        assert await handle.result() == 2
    histories = await _histories(env.client, handle.id, handle.first_execution_run_id)
    assert len(histories) == 1
    await _replay(HandlerAgentWorkflow, histories)


class WaitForSignalMiddleware(AgentMiddleware):
    def __init__(self, finish_handler: Callable[[], Awaitable[None]]) -> None:
        self.finish_handler = finish_handler

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        await self.finish_handler()


@workflow.defn
class ActiveSignalWorkflow:
    def __init__(self) -> None:
        self.signal_started = False
        self.agent_started = False

    @workflow.signal
    async def wait_for_agent(self) -> None:
        self.signal_started = True
        await workflow.wait_condition(lambda: self.agent_started)

    async def finish_handler(self) -> None:
        self.agent_started = True
        await workflow.wait_condition(workflow.all_handlers_finished)

    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> str:
        if state_snapshot is None:
            await workflow.wait_condition(lambda: self.signal_started)
            assert not workflow.all_handlers_finished()
        result = await run_deep_agent(
            create_deep_agent(
                name="test-agent",
                model="fake:model",
                middleware=[WaitForSignalMiddleware(self.finish_handler)],
            ),
            input,
            state_snapshot=state_snapshot,
            continue_as_new_after=1,
        )
        return result["messages"][-1].content


async def test_rollover_resumes_after_active_signal_handler_finishes(
    env: WorkflowEnvironment,
) -> None:
    result, histories = await _run(
        env,
        ActiveSignalWorkflow,
        {"messages": [{"role": "user", "content": "Go."}]},
        responses=[_call("ls", {"path": "/"}, "ls"), "Done."],
        start_signal="wait_for_agent",
    )
    assert result == "Done."
    assert len(histories) == 2


@workflow.defn
class TypedInputWorkflow:
    @workflow.run
    async def run(self, question: str, state_snapshot: dict | None = None) -> str:
        agent = create_deep_agent(name="assistant", model="fake:model")
        result = await run_deep_agent(
            agent,
            None
            if state_snapshot
            else {"messages": [{"role": "user", "content": question}]},
            continue_as_new_input=question,
            continue_as_new_after=1,
            state_snapshot=state_snapshot,
        )
        return result["messages"][-1].content


async def test_explicit_workflow_input_retains_its_type_across_rollover(
    env: WorkflowEnvironment,
) -> None:
    result, histories = await _run(
        env,
        TypedInputWorkflow,
        "Go.",
        responses=[_call("ls", {"path": "/"}, "ls"), "Done."],
    )
    assert result == "Done."
    assert len(histories) == 2
