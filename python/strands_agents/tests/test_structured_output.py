from collections.abc import AsyncIterable
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field
from strands.types.streaming import StreamEvent

from temporalio import workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.worker import Replayer, Worker
from tests.mock_model import MockModel


class PersonInfo(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    occupation: str = Field(description="Occupation of the person")


@workflow.defn
class StructuredOutputWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=15),
            structured_output_model=PersonInfo,
        )

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


async def test_structured_output_missing_fails_workflow(client: Client):
    """A model that never gives structured output fails the Workflow.

    Strands raises StructuredOutputException in Workflow code from model answers
    already in History, so a retried Workflow Task would fail the same way forever.
    """
    task_queue = f"test_structured_output_missing-{uuid4()}"
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel(["I'd rather answer in prose."] * 5)}
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
            id=f"test_structured_output_missing_{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()

    cause = err.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "StructuredOutputException"
    history = await handle.fetch_history()
    assert not any(
        event.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED
        for event in history.events
    )

    await Replayer(
        workflows=[StructuredOutputWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


class MaxTokensModel(MockModel):
    """Answers with text that stops at the model's token limit."""

    async def stream(self, *_args: Any, **_kwargs: Any) -> AsyncIterable[StreamEvent]:
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": "The answer is"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "max_tokens"}}


@workflow.defn
class MaxTokensWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="max_tokens",
            start_to_close_timeout=timedelta(seconds=15),
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        return str(await self.agent.invoke_async(prompt))


async def test_max_tokens_fails_workflow(client: Client):
    """A model answer cut off at its token limit fails the Workflow."""
    task_queue = f"test_max_tokens-{uuid4()}"
    plugin = StrandsPlugin(models={"max_tokens": lambda: MaxTokensModel([])})

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MaxTokensWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MaxTokensWorkflow.run,
            "Tell me a long story",
            id=f"test_max_tokens_{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()

    cause = err.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "MaxTokensReachedException"
