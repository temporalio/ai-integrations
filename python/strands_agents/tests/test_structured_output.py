import asyncio
from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field
from strands.types.event_loop import StopReason
from strands.types.exceptions import (
    MaxTokensReachedException,
    StructuredOutputException,
)

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.worker import Replayer, Worker
from tests.mock_model import MockModel


class PersonInfo(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    occupation: str = Field(description="Occupation of the person")


def person_agent() -> TemporalAgent:
    return TemporalAgent(
        model="mock",
        start_to_close_timeout=timedelta(seconds=15),
        structured_output_model=PersonInfo,
    )


@workflow.defn
class StructuredOutputWorkflow:
    def __init__(self) -> None:
        self.agent = person_agent()

    @workflow.run
    async def run(self, prompt: str) -> PersonInfo:
        result = await self.agent.invoke_async(prompt)
        assert isinstance(result.structured_output, PersonInfo)
        return result.structured_output


async def test_structured_output(client: Client):
    task_queue = f"test_structured_output-{uuid4()}"
    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [
                    {
                        "name": "PersonInfo",
                        "input": {
                            "name": "John Smith",
                            "age": 30,
                            "occupation": "software engineer",
                        },
                    },
                ]
            )
        }
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StructuredOutputWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StructuredOutputWorkflow.run,
            "John Smith is a 30 year-old software engineer",
            id=f"test_structured_output_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == PersonInfo(
            name="John Smith", age=30, occupation="software engineer"
        )

    await Replayer(
        workflows=[StructuredOutputWorkflow],
        plugins=[plugin],
    ).replay_workflow(await handle.fetch_history())


@workflow.defn
class StructuredOutputFailureWorkflow:
    @workflow.run
    async def run(self, catch_error: bool) -> str:
        try:
            await person_agent().invoke_async("Tell me about John.")
        except (StructuredOutputException, MaxTokensReachedException) as err:
            if catch_error:
                return type(err).__name__
            raise
        raise AssertionError("Expected the agent to reject the model response")


@pytest.mark.parametrize("catch_error", [False, True])
@pytest.mark.parametrize(
    ("stop_reason", "failure_type"),
    [
        ("end_turn", "StructuredOutputException"),
        ("max_tokens", "MaxTokensReachedException"),
    ],
)
async def test_structured_output_failure(
    client: Client, stop_reason: StopReason, failure_type: str, catch_error: bool
) -> None:
    task_queue = str(uuid4())
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel(["Plain text"] * 3, stop_reason=stop_reason)}
    )
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StructuredOutputFailureWorkflow],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            StructuredOutputFailureWorkflow.run,
            catch_error,
            id=str(uuid4()),
            task_queue=task_queue,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        async with asyncio.timeout(15):
            if catch_error:
                assert await handle.result() == failure_type
            else:
                with pytest.raises(WorkflowFailureError) as raised:
                    await handle.result()
                assert isinstance(raised.value.cause, ApplicationError)
                assert raised.value.cause.type == failure_type
                assert raised.value.cause.non_retryable
    await Replayer(
        workflows=[StructuredOutputFailureWorkflow], plugins=[plugin]
    ).replay_workflow(await handle.fetch_history())
