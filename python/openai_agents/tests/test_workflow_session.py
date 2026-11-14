"""Tests for workflow-owned OpenAI Agents conversation memory."""

import uuid
from typing import Any

import pytest
from agents import (
    Agent,
    AgentOutputSchemaBase,
    Handoff,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    Runner,
    Session,
    SessionSettings,
    Tool,
    TResponseInputItem,
)

from temporalio import workflow
from temporalio.client import Client
from temporalio.openai_agents import WorkflowSession
from temporalio.openai_agents.testing import (
    AgentEnvironment,
    ResponseBuilders,
    TestModel,
)
from temporalio.worker import Replayer
from tests.helpers import new_worker


class RecordingModel(TestModel):
    """Record the input passed to each model activity."""

    def __init__(self) -> None:
        """Return one response per agent turn."""
        responses = iter(
            [
                ResponseBuilders.output_message("Hello, Ada."),
                ResponseBuilders.output_message("Your name is Ada."),
            ]
        )
        super().__init__(lambda: next(responses))
        self.inputs: list[str | list[TResponseInputItem]] = []

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> ModelResponse:
        """Record the input before producing the next response."""
        self.inputs.append(input)
        return await super().get_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            **kwargs,
        )


@workflow.defn
class SessionOperationsWorkflow:
    @workflow.run
    async def run(self) -> None:
        session = WorkflowSession("memory", session_settings=SessionSettings(limit=2))
        assert isinstance(session, Session)
        items: list[TResponseInputItem] = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
        await session.add_items(items)
        assert await session.get_items() == items[-2:]
        assert await session.get_items(1) == items[-1:]
        assert await session.get_items(0) == []
        assert await session.pop_item() == items[-1]
        assert await session.get_items() == items[:2]
        await session.clear_session()
        assert await session.get_items() == []
        assert await session.pop_item() is None


@workflow.defn
class MultiTurnSessionWorkflow:
    @workflow.run
    async def run(self) -> tuple[str, str]:
        agent = Agent(name="Assistant")
        session = WorkflowSession("conversation")
        first = await Runner.run(agent, "My name is Ada.", session=session)
        first_turn = await session.get_items()
        assert first_turn[0] == {"role": "user", "content": "My name is Ada."}
        assert len(first_turn) >= 2
        second = await Runner.run(agent, "What is my name?", session=session)
        all_items = await session.get_items()
        assert len(all_items) > len(first_turn)
        assert all_items[: len(first_turn)] == first_turn
        return str(first.final_output), str(second.final_output)


async def test_workflow_session_operations(client: Client) -> None:
    async with AgentEnvironment(register_activities=False) as env:
        client = env.applied_on_client(client)
        async with new_worker(client, SessionOperationsWorkflow) as worker:
            await client.execute_workflow(
                SessionOperationsWorkflow.run,
                id=f"session-operations-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )


async def test_workflow_session_runner_and_replay(client: Client) -> None:
    model = RecordingModel()
    async with AgentEnvironment(model=model) as env:
        client = env.applied_on_client(client)
        async with new_worker(client, MultiTurnSessionWorkflow) as worker:
            handle = await client.start_workflow(
                MultiTurnSessionWorkflow.run,
                id=f"multi-turn-session-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            assert await handle.result() == ("Hello, Ada.", "Your name is Ada.")
            assert len(model.inputs) == 2
            assert isinstance(model.inputs[1], list)
            assert model.inputs[1][0] == {"role": "user", "content": "My name is Ada."}

        await Replayer(
            workflows=[MultiTurnSessionWorkflow],
            plugins=[env.openai_agents_plugin],
        ).replay_workflow(await handle.fetch_history())


def test_workflow_session_requires_workflow() -> None:
    with pytest.raises(RuntimeError, match="created inside a workflow"):
        WorkflowSession("outside")
