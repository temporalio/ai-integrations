import asyncio
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from datetime import timedelta
from typing import Any, TypeVar
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field
from strands.types.exceptions import StructuredOutputException
from strands.types.streaming import StreamEvent

import temporalio.worker
from temporalio import workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import (
    Client,
    WorkflowExecutionStatus,
    WorkflowFailureError,
    WorkflowHandle,
    WorkflowUpdateFailedError,
)
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.strands_agents import (
    StrandsPlugin,
    TemporalAgent,
    _temporal_agent,
    _workflow_failure,
)
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


T = TypeVar("T")


async def assert_eventually(
    fn: Callable[[], Awaitable[T]], timeout: timedelta = timedelta(seconds=10)
) -> T:
    deadline = time.monotonic() + timeout.total_seconds()
    while True:
        try:
            return await fn()
        except AssertionError:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(0.2)


def _client_with(client: Client, plugin: StrandsPlugin) -> Client:
    config = client.config()
    config["plugins"] = [*config["plugins"], plugin]
    return Client(**config)


async def test_structured_output_missing_follows_workflow_retry_policy(client: Client):
    """With the plugin on the Client, a Workflow retry policy starts another run.

    The plugin's failure converter serializes the Workflow failure, so this covers
    that it stays retryable on the Workflow side.
    """
    task_queue = f"test_structured_output_retry-{uuid4()}"
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel(["I'd rather answer in prose."] * 10)}
    )
    client = _client_with(client, plugin)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StructuredOutputWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StructuredOutputWorkflow.run,
            "John Smith is a 30 year-old software engineer",
            id=f"test_structured_output_retry_{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(milliseconds=100), maximum_attempts=2
            ),
        )
        with pytest.raises(WorkflowFailureError) as err:
            await handle.result()

    cause = err.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "StructuredOutputException"
    assert not cause.non_retryable

    first_run = client.get_workflow_handle(
        handle.id, run_id=handle.first_execution_run_id
    )
    first_history = await first_run.fetch_history()
    failed = first_history.events[-1]
    assert failed.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_FAILED
    second_run_id = (
        failed.workflow_execution_failed_event_attributes.new_execution_run_id
    )
    assert second_run_id

    second_run = client.get_workflow_handle(handle.id, run_id=second_run_id)
    second_history = await second_run.fetch_history()
    started = second_history.events[0].workflow_execution_started_event_attributes
    assert started.attempt == 2
    assert (
        second_history.events[-1].event_type
        == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_FAILED
    )


class _EscapedExceptionRecorder(temporalio.worker.Interceptor):
    """Records the exceptions that escape the Workflow's run method on this Worker.

    Worker interceptors wrap the plugin's, so it sees what the plugin lets through.
    """

    def __init__(self) -> None:
        self.escaped: list[BaseException] = []

    def workflow_interceptor_class(
        self, input: temporalio.worker.WorkflowInterceptorClassInput
    ) -> type[temporalio.worker.WorkflowInboundInterceptor]:
        escaped = self.escaped

        class _Inbound(temporalio.worker.WorkflowInboundInterceptor):
            async def execute_workflow(
                self, input: temporalio.worker.ExecuteWorkflowInput
            ) -> Any:
                try:
                    return await self.next.execute_workflow(input)
                except BaseException as e:
                    escaped.append(e)
                    raise

        return _Inbound


@workflow.defn
class StructuredOutputUpdateWorkflow:
    """Runs the agent in an Update handler while the Workflow waits."""

    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=15),
            structured_output_model=PersonInfo,
        )

    @workflow.run
    async def run(self) -> None:
        await workflow.wait_condition(lambda: False)

    @workflow.update
    async def ask(self, prompt: str) -> PersonInfo:
        result = await self.agent.invoke_async(prompt)
        assert isinstance(result.structured_output, PersonInfo)
        return result.structured_output


async def test_structured_output_missing_in_update_fails_only_the_update(
    client: Client,
):
    """An exception that escapes an Update handler fails the Update, not the Workflow."""
    task_queue = f"test_structured_output_update-{uuid4()}"
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel(["I'd rather answer in prose."] * 5)}
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StructuredOutputUpdateWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StructuredOutputUpdateWorkflow.run,
            id=f"test_structured_output_update_{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        with pytest.raises(WorkflowUpdateFailedError) as err:
            await handle.execute_update(
                StructuredOutputUpdateWorkflow.ask,
                "John Smith is a 30 year-old software engineer",
            )
        cause = err.value.cause
        assert isinstance(cause, ApplicationError)
        assert cause.type == "StructuredOutputException"
        assert (await handle.describe()).status == WorkflowExecutionStatus.RUNNING
        await handle.terminate()


@workflow.defn(name="StructuredOutputWorkflow")
class FixedStructuredOutputWorkflow:
    """StructuredOutputWorkflow after a code fix that catches the exception."""

    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=15),
            structured_output_model=PersonInfo,
        )

    @workflow.run
    async def run(self, prompt: str) -> PersonInfo:
        try:
            result = await self.agent.invoke_async(prompt)
        except StructuredOutputException:
            return PersonInfo(name="unknown", age=0, occupation="unknown")
        assert isinstance(result.structured_output, PersonInfo)
        return result.structured_output


async def test_structured_output_missing_upgrade_keeps_retrying_run_recoverable(
    client: Client, monkeypatch: pytest.MonkeyPatch
):
    """A run that was retrying its Workflow Task before the upgrade keeps doing so.

    The run starts on a Worker without the patch, as on an older release, and its
    Workflow Task fails on the missing structured output. An upgraded Worker must
    not fail the Workflow, so that a Workflow code fix can still recover the run.
    """
    task_queue = f"test_structured_output_upgrade-{uuid4()}"
    plugin = StrandsPlugin(
        models={"mock": lambda: MockModel(["I'd rather answer in prose."] * 5)}
    )

    async def pending_workflow_task_attempt(handle: WorkflowHandle) -> int:
        description = await handle.describe()
        assert description.status == WorkflowExecutionStatus.RUNNING
        return description.raw_description.pending_workflow_task.attempt

    with monkeypatch.context() as m:
        # A Worker from before the patch never records its marker.
        for module in (_temporal_agent, _workflow_failure):
            m.setattr(module, "agent_loop_errors_fail_workflow", lambda: False)
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
                id=f"test_structured_output_upgrade_{uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=60),
            )

            async def failed_once() -> int:
                attempt = await pending_workflow_task_attempt(handle)
                assert attempt >= 2
                return attempt

            old_attempt = await assert_eventually(failed_once)

    recorder = _EscapedExceptionRecorder()
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StructuredOutputWorkflow],
        plugins=[plugin],
        interceptors=[recorder],
        max_cached_workflows=0,
    ):

        async def failed_on_upgraded_worker() -> None:
            # The upgraded Worker itself replays the run and gets the original
            # exception, not an ApplicationError that would fail the Workflow.
            assert any(
                isinstance(e, StructuredOutputException) for e in recorder.escaped
            )
            assert not any(isinstance(e, ApplicationError) for e in recorder.escaped)
            attempt = await pending_workflow_task_attempt(handle)
            assert attempt > old_attempt

        await assert_eventually(
            failed_on_upgraded_worker, timeout=timedelta(seconds=20)
        )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[FixedStructuredOutputWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        result = await handle.result()
    assert result.name == "unknown"
