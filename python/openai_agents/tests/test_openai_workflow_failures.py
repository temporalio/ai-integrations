import asyncio
from uuid import uuid4

import pytest
from agents import Agent, ModelResponse, Runner, function_tool
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from pydantic import BaseModel

from temporalio import workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowFailureError, WorkflowUpdateFailedError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents._failure_interceptor import AgentFailureInterceptor
from temporalio.openai_agents.testing import (
    AgentEnvironment,
    ResponseBuilders,
    TestModel,
)
from temporalio.worker import Replayer, Worker
from tests.helpers import assert_event_subsequence


class Answer(BaseModel):
    answer: str


@function_tool
def get_answer() -> str:
    return "42"


@workflow.defn
class AgentFailureWorkflow:
    def __init__(self) -> None:
        self.finished = False

    @workflow.run
    async def run(self, mode: str) -> str:
        if mode == "update":
            await workflow.wait_condition(lambda: self.finished)
            return "finished"
        try:
            return await self._ask()
        except (ModelBehaviorError, MaxTurnsExceeded) as err:
            if mode == "catch":
                return type(err).__name__
            raise

    @workflow.update
    async def ask(self) -> str:
        return await self._ask()

    async def _ask(self) -> str:
        agent = Agent(name="A", output_type=Answer, tools=[get_answer])
        result = await Runner.run(agent, "What is the answer?", max_turns=1)
        return result.final_output_as(Answer).answer

    @workflow.signal
    def finish(self) -> None:
        self.finished = True


@pytest.mark.parametrize("mode", ["fail", "catch", "update"])
@pytest.mark.parametrize("failure_type", ["ModelBehaviorError", "MaxTurnsExceeded"])
async def test_agent_workflow_failure(
    client: Client, mode: str, failure_type: str
) -> None:
    response = (
        ResponseBuilders.output_message('{"answer": 42}')
        if failure_type == "ModelBehaviorError"
        else ResponseBuilders.tool_call("{}", "get_answer")
    )
    model_calls = 0

    def respond() -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return response

    async with AgentEnvironment(model=TestModel(respond)) as agents_env:
        client = agents_env.applied_on_client(client)
        task_queue = str(uuid4())
        async with Worker(
            client, task_queue=task_queue, workflows=[AgentFailureWorkflow]
        ):
            handle = await client.start_workflow(
                AgentFailureWorkflow.run,
                mode,
                id=str(uuid4()),
                task_queue=task_queue,
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            async with asyncio.timeout(15):
                if mode == "catch":
                    assert await handle.result() == failure_type
                else:
                    if mode == "update":
                        with pytest.raises(WorkflowUpdateFailedError) as update_failed:
                            await handle.execute_update(AgentFailureWorkflow.ask)
                        cause = update_failed.value.cause
                        await handle.signal(AgentFailureWorkflow.finish)
                        assert await handle.result() == "finished"
                    else:
                        with pytest.raises(WorkflowFailureError) as workflow_failed:
                            await handle.result()
                        cause = workflow_failed.value.cause
                    assert isinstance(cause, ApplicationError)
                    assert cause.type == failure_type
                    assert not cause.non_retryable
        await Replayer(
            workflows=[AgentFailureWorkflow], plugins=[agents_env.openai_agents_plugin]
        ).replay_workflow(await handle.fetch_history())
    assert model_calls == (2 if mode == "fail" else 1)


async def test_upgrade_fails_workflow_stuck_on_invalid_output(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = TestModel.returning_responses(
        [ResponseBuilders.output_message('{"answer": 42}')]
    )
    async with AgentEnvironment(model=model) as agents_env:
        plugin = agents_env.openai_agents_plugin
        client = agents_env.applied_on_client(client)
        task_queue = str(uuid4())
        with monkeypatch.context() as old_worker:
            # Simulate an older worker without the terminal agent error interceptor.
            assert plugin.interceptors is not None
            old_worker.setattr(
                plugin,
                "interceptors",
                [
                    interceptor
                    for interceptor in plugin.interceptors
                    if not isinstance(interceptor, AgentFailureInterceptor)
                ],
            )
            async with Worker(
                client, task_queue=task_queue, workflows=[AgentFailureWorkflow]
            ):
                handle = await client.start_workflow(
                    AgentFailureWorkflow.run,
                    "fail",
                    id=str(uuid4()),
                    task_queue=task_queue,
                )
                await assert_event_subsequence(
                    handle,
                    [
                        EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED,
                        EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED,
                    ],
                )

        async with Worker(
            client, task_queue=task_queue, workflows=[AgentFailureWorkflow]
        ):
            async with asyncio.timeout(20):
                with pytest.raises(WorkflowFailureError) as raised:
                    await handle.result()
            cause = raised.value.cause
            assert isinstance(cause, ApplicationError)
            assert cause.type == "ModelBehaviorError"
            assert not cause.non_retryable
