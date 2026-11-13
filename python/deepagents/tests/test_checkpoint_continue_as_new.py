"""Graph checkpoints preserve an unfinished turn without todo-based decisions."""

from __future__ import annotations

import uuid
from typing import Annotated, Any
from unittest.mock import patch

import pytest
from typing_extensions import NotRequired

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
        agent = create_deep_agent(model="fake:model", middleware=middleware)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("todos", [False, True])
async def test_graph_state_and_lifecycle_cross_multiple_runs(
    env: WorkflowEnvironment, todos: bool
) -> None:
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(_responses(todos)))
    client_config = env.client.config()
    client_config["plugins"] = [plugin]
    client = Client(**client_config)
    async with Worker(
        client,
        task_queue=f"checkpoint-state-{todos}",
        workflows=[CheckpointAgentWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            CheckpointAgentWorkflow.run,
            {
                "messages": [{"role": "user", "content": "Research."}],
                "todos": todos,
                "limit": 1,
            },
            id=f"checkpoint-state-{uuid.uuid4()}",
            task_queue=f"checkpoint-state-{todos}",
        )
        result = await handle.result()
        histories = await _histories(client, handle.id, handle.first_execution_run_id)
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
    replayer = Replayer(
        workflows=[CheckpointAgentWorkflow], plugins=[DeepAgentsPlugin()]
    )
    for history in histories:
        await replayer.replay_workflow(history)


@pytest.mark.asyncio
async def test_completed_answer_returns_even_past_limit(
    env: WorkflowEnvironment,
) -> None:
    async with Worker(
        env.client,
        task_queue="checkpoint-completed",
        workflows=[CheckpointAgentWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=mock_model_provider(["Done."]))],
    ):
        handle = await env.client.start_workflow(
            CheckpointAgentWorkflow.run,
            {"messages": [{"role": "user", "content": "Hello."}], "limit": 1},
            id=f"checkpoint-completed-{uuid.uuid4()}",
            task_queue="checkpoint-completed",
        )
        result = await handle.result()
    assert result["model_steps"] == result["started"] == result["finished"] == 1
    assert (
        len(await _histories(env.client, handle.id, handle.first_execution_run_id)) == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("suggested", [False, True])
async def test_checkpoint_driver_follows_server_suggestion(
    env: WorkflowEnvironment, suggested: bool
) -> None:
    with patch.object(
        workflow.Info, "is_continue_as_new_suggested", return_value=suggested
    ):
        async with Worker(
            env.client,
            task_queue=f"checkpoint-suggestion-{suggested}",
            workflows=[CheckpointAgentWorkflow],
            plugins=[
                DeepAgentsPlugin(model_provider=mock_model_provider(_responses(False)))
            ],
        ):
            handle = await env.client.start_workflow(
                CheckpointAgentWorkflow.run,
                {"messages": [{"role": "user", "content": "Research."}]},
                id=f"checkpoint-suggestion-{uuid.uuid4()}",
                task_queue=f"checkpoint-suggestion-{suggested}",
            )
            result = await handle.result()
        histories = await _histories(
            env.client, handle.id, handle.first_execution_run_id
        )
        for history in histories:
            await Replayer(
                workflows=[CheckpointAgentWorkflow], plugins=[DeepAgentsPlugin()]
            ).replay_workflow(history)
    assert (len(histories) > 1) == suggested
    assert result["model_steps"] == 5


@workflow.defn
class InterruptedAgentWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> dict:
        agent = create_deep_agent(
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


@pytest.mark.asyncio
async def test_application_interrupt_is_returned_to_caller(
    env: WorkflowEnvironment,
) -> None:
    async with Worker(
        env.client,
        task_queue="checkpoint-interrupt",
        workflows=[InterruptedAgentWorkflow],
        plugins=[
            DeepAgentsPlugin(
                model_provider=mock_model_provider(
                    [
                        _call("read_file", {"file_path": "/notes.md"}, "read-1"),
                    ]
                )
            )
        ],
    ):
        handle = await env.client.start_workflow(
            InterruptedAgentWorkflow.run,
            {"messages": [{"role": "user", "content": "Read."}]},
            id=f"checkpoint-interrupt-{uuid.uuid4()}",
            task_queue="checkpoint-interrupt",
        )
        assert await handle.result() == {"interrupted": True, "finished": 0}
    assert (
        len(await _histories(env.client, handle.id, handle.first_execution_run_id)) == 1
    )


@workflow.defn
class RecursionBudgetWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> str:
        agent = create_deep_agent(model="fake:model").with_config(
            {"recursion_limit": 8}
        )
        try:
            await run_deep_agent(
                agent, input, continue_as_new_after=1, state_snapshot=state_snapshot
            )
        except DeepAgentsWorkflowError:
            return "limited"
        return "unexpected completion"


@pytest.mark.asyncio
async def test_recursion_budget_survives_rollovers(env: WorkflowEnvironment) -> None:
    async with Worker(
        env.client,
        task_queue="checkpoint-budget",
        workflows=[RecursionBudgetWorkflow],
        plugins=[
            DeepAgentsPlugin(
                model_provider=mock_model_provider(
                    [
                        _call("ls", {"path": "/"}, "loop"),
                    ]
                )
            )
        ],
    ):
        handle = await env.client.start_workflow(
            RecursionBudgetWorkflow.run,
            {"messages": [{"role": "user", "content": "Loop."}]},
            id=f"checkpoint-budget-{uuid.uuid4()}",
            task_queue="checkpoint-budget",
        )
        assert await handle.result() == "limited"
    histories = await _histories(env.client, handle.id, handle.first_execution_run_id)
    assert 1 < len(histories) < 8


@workflow.defn
class SubagentCheckpointWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> list:
        agent = create_deep_agent(
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


@pytest.mark.asyncio
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

    async with Worker(
        env.client,
        task_queue="checkpoint-subagent",
        workflows=[SubagentCheckpointWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=provider)],
    ):
        handle = await env.client.start_workflow(
            SubagentCheckpointWorkflow.run,
            {"messages": [{"role": "user", "content": "Delegate."}]},
            id=f"checkpoint-subagent-{uuid.uuid4()}",
            task_queue="checkpoint-subagent",
        )
        result = await handle.result()
    assert provider_calls == ["fake:main", "fake:sub", "fake:sub", "fake:main"]
    assert result == [
        ["human", "Delegate."],
        ["ai", ""],
        ["tool", "Sub done."],
        ["ai", "Main done."],
    ]
    histories = await _histories(env.client, handle.id, handle.first_execution_run_id)
    assert len(histories) == 2
    for history in histories:
        await Replayer(
            workflows=[SubagentCheckpointWorkflow], plugins=[DeepAgentsPlugin()]
        ).replay_workflow(history)


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


@pytest.mark.asyncio
async def test_pre_checkpoint_histories_use_legacy_driver(
    env: WorkflowEnvironment,
) -> None:
    original_patched = workflow.patched

    def before_checkpoint_change(patch_id: str) -> bool:
        if patch_id == "deepagents.checkpoint-continue-as-new":
            return False
        return original_patched(patch_id)

    with patch.object(workflow, "patched", before_checkpoint_change):
        async with Worker(
            env.client,
            task_queue="checkpoint-legacy",
            workflows=[LegacyTodoCheckpointWorkflow],
            plugins=[DeepAgentsPlugin()],
        ):
            handle = await env.client.start_workflow(
                LegacyTodoCheckpointWorkflow.run,
                {"messages": []},
                id=f"checkpoint-legacy-{uuid.uuid4()}",
                task_queue="checkpoint-legacy",
            )
            assert (await handle.result())["messages"] == ["step"] * 3
    histories = await _histories(env.client, handle.id, handle.first_execution_run_id)
    assert len(histories) == 3
    for history in histories:
        await Replayer(
            workflows=[LegacyTodoCheckpointWorkflow], plugins=[DeepAgentsPlugin()]
        ).replay_workflow(history)


@pytest.mark.asyncio
async def test_plain_agent_todos_do_not_trigger_rollover(
    env: WorkflowEnvironment,
) -> None:
    async with Worker(
        env.client,
        task_queue="checkpoint-plain",
        workflows=[LegacyTodoCheckpointWorkflow],
        plugins=[DeepAgentsPlugin()],
    ):
        handle = await env.client.start_workflow(
            LegacyTodoCheckpointWorkflow.run,
            {"messages": []},
            id=f"checkpoint-plain-{uuid.uuid4()}",
            task_queue="checkpoint-plain",
        )
        assert (await handle.result())["messages"] == ["step"]
    assert (
        len(await _histories(env.client, handle.id, handle.first_execution_run_id)) == 1
    )


@workflow.defn
class StaticBreakpointWorkflow:
    @workflow.run
    async def run(self) -> list:
        agent = create_deep_agent(model="fake:model").copy(
            {"interrupt_before_nodes": ["model"]}
        )
        result = await run_deep_agent(
            agent,
            {"messages": [{"role": "user", "content": "Pause."}]},
            continue_as_new_after=1,
        )
        return [[m.type, m.content] for m in result["messages"]]


@pytest.mark.asyncio
async def test_caller_breakpoint_is_not_automatically_resumed(
    env: WorkflowEnvironment,
) -> None:
    def no_model_call(name: str) -> Any:
        raise AssertionError(f"Caller breakpoint was bypassed: {name}")

    async with Worker(
        env.client,
        task_queue="checkpoint-breakpoint",
        workflows=[StaticBreakpointWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=no_model_call)],
    ):
        assert await env.client.execute_workflow(
            StaticBreakpointWorkflow.run,
            id=f"checkpoint-breakpoint-{uuid.uuid4()}",
            task_queue="checkpoint-breakpoint",
        ) == [["human", "Pause."]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot, message",
    [
        ({"version": 2}, "Unsupported Deep Agents checkpoint snapshot"),
        (
            {"version": 1, "graph": {"nodes": [], "channels": []}},
            "The graph's nodes or state channels changed since its checkpoint",
        ),
    ],
)
async def test_incompatible_checkpoint_is_rejected(
    snapshot: dict, message: str
) -> None:
    from temporalio.deepagents._checkpoint import SNAPSHOT_KEY, run
    from temporalio.deepagents.testing import FakeModel

    agent = create_deep_agent(model=FakeModel(["unused"]))
    with pytest.raises(DeepAgentsWorkflowError, match=message):
        await run(
            agent, {}, continue_as_new_after=1, state_snapshot={SNAPSHOT_KEY: snapshot}
        )


@workflow.defn
class ApprovalAfterRolloverWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> str:
        agent = create_deep_agent(
            model="fake:model",
            checkpointer=InMemorySaver(),
            interrupt_on={"read_file": True},
        ).with_config({"configurable": {"thread_id": "approval"}})
        result = await run_deep_agent(
            agent, input, continue_as_new_after=1, state_snapshot=state_snapshot
        )
        if result.get("__interrupt__"):
            result = await run_deep_agent(
                agent,
                Command(resume={"decisions": [{"type": "approve"}]}),
                continue_as_new_after=1,
            )
        return result["messages"][-1].content


@pytest.mark.asyncio
async def test_approval_after_rollover_resumes_and_continues_again(
    env: WorkflowEnvironment,
) -> None:
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(
            [
                _call("ls", {"path": "/"}, "ls-1"),
                _call("read_file", {"file_path": "/notes.md"}, "read-1"),
                _call("ls", {"path": "/"}, "ls-2"),
                "Approved.",
            ]
        )
    )
    async with Worker(
        env.client,
        task_queue="checkpoint-approval-resume",
        workflows=[ApprovalAfterRolloverWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await env.client.start_workflow(
            ApprovalAfterRolloverWorkflow.run,
            {"messages": [{"role": "user", "content": "Read."}]},
            id=f"checkpoint-approval-{uuid.uuid4()}",
            task_queue="checkpoint-approval-resume",
        )
        assert await handle.result() == "Approved."
    histories = await _histories(env.client, handle.id, handle.first_execution_run_id)
    assert len(histories) == 3
    for history in histories:
        await Replayer(
            workflows=[ApprovalAfterRolloverWorkflow], plugins=[DeepAgentsPlugin()]
        ).replay_workflow(history)


@workflow.defn
class InvalidSignatureWorkflow:
    @workflow.run
    async def run(self, input: dict) -> Any:
        return await run_deep_agent(create_deep_agent(model="fake:model"), input)


@workflow.defn
class InvalidConfigurationWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> Any:
        mode = input["mode"]
        agent = create_deep_agent(
            model="fake:model", checkpointer=False if mode == "saver" else None
        ).with_config({"recursion_limit": 8})
        return await run_deep_agent(
            agent,
            {**input, "messages": [{"role": "user", "content": "Go."}]},
            continue_as_new_after=0 if mode == "threshold" else 1,
            state_snapshot=state_snapshot,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["signature", "saver", "threshold", "recursion"])
async def test_driver_errors_fail_workflow_execution(
    env: WorkflowEnvironment, mode: str
) -> None:
    calls = []
    looping_provider = mock_model_provider([_call("ls", {"path": "/"}, "loop")])

    def provider(name: str) -> Any:
        calls.append(name)
        return looping_provider(name)

    workflow_class = (
        InvalidSignatureWorkflow
        if mode == "signature"
        else InvalidConfigurationWorkflow
    )
    async with Worker(
        env.client,
        task_queue=f"checkpoint-invalid-{mode}",
        workflows=[workflow_class],
        plugins=[DeepAgentsPlugin(model_provider=provider)],
    ):
        handle = await env.client.start_workflow(
            workflow_class.run,
            {"mode": mode},
            id=f"checkpoint-invalid-{mode}-{uuid.uuid4()}",
            task_queue=f"checkpoint-invalid-{mode}",
        )
        with pytest.raises(WorkflowFailureError) as caught:
            await handle.result()
    cause = caught.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == DeepAgentsWorkflowError.TYPE
    assert cause.non_retryable
    assert bool(calls) == (mode == "recursion")
