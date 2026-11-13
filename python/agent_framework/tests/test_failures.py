"""Error classification, retry exhaustion and workflow failure types."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

import pytest
from agent_framework import (
    Agent,
    FunctionInvocationContext,
    FunctionMiddleware,
    Message,
    MiddlewareFailure,
    tool,
)

from temporalio import workflow
from temporalio.agent_framework import (
    AgentFrameworkPlugin,
    AgentFrameworkWorkflowError,
    TemporalChatClient,
)
from temporalio.agent_framework._activities import (
    ChatClientActivities,
    InvokeModelInput,
    InvokeModelStreamingInput,
    classify_provider_error,
)
from temporalio.agent_framework.testing import (
    FakeChatClient,
    fake_chat_client,
    function_call_response,
    text_response,
)
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.testing import ActivityEnvironment
from tests.helpers.agent_framework import (
    INVOKE_MODEL,
    StatusError,
    count_scheduled_activities,
    new_worker,
    started_activities,
    workflow_id,
)


def _wrapped(error: Exception) -> Exception:
    # Agent Framework raises ChatClientException from the SDK error.
    outer = RuntimeError(f"service failed to complete the prompt: {error}")
    outer.__cause__ = error
    return outer


@pytest.mark.parametrize(
    ("status", "non_retryable"),
    [
        (400, True),
        (401, True),
        (403, True),
        (404, True),
        (408, False),
        (409, False),
        (429, False),
        (500, False),
        (503, False),
    ],
)
def test_status_classification(status: int, non_retryable: bool) -> None:
    translated = classify_provider_error(_wrapped(StatusError(status)))
    assert translated is not None
    assert translated.type == "StatusError"
    assert translated.non_retryable is non_retryable
    assert translated.next_retry_delay is None


def test_retry_hints() -> None:
    ms = classify_provider_error(
        _wrapped(StatusError(429, {"retry-after-ms": "250", "retry-after": "7"}))
    )
    assert ms is not None and ms.next_retry_delay == timedelta(milliseconds=250)
    seconds = classify_provider_error(_wrapped(StatusError(503, {"Retry-After": "7"})))
    assert seconds is not None and seconds.next_retry_delay == timedelta(seconds=7)
    forced_off = classify_provider_error(
        _wrapped(StatusError(503, {"x-should-retry": "false"}))
    )
    assert forced_off is not None and forced_off.non_retryable
    forced_on = classify_provider_error(
        _wrapped(StatusError(400, {"x-should-retry": "true"}))
    )
    assert forced_on is not None and not forced_on.non_retryable
    assert classify_provider_error(ConnectionError("reset")) is None


@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "streaming"])
async def test_activity_classifies_provider_errors(streaming: bool) -> None:
    error = StatusError(401)
    activities = ChatClientActivities({"model": lambda: FakeChatClient([error])})
    messages = [Message(role="user", contents=["hi"]).to_dict()]
    env = ActivityEnvironment()
    with pytest.raises(ApplicationError) as excinfo:
        if streaming:
            await env.run(
                activities.invoke_model_streaming,
                InvokeModelStreamingInput(model="model", messages=messages),
            )
        else:
            await env.run(
                activities.invoke_model,
                InvokeModelInput(model="model", messages=messages),
            )
    assert excinfo.value.type == "StatusError"
    assert excinfo.value.non_retryable
    assert excinfo.value.__cause__ is error


async def test_error_without_status_stays_retryable() -> None:
    error = ConnectionError("connection reset by the fake provider")
    activities = ChatClientActivities({"model": lambda: FakeChatClient([error])})
    with pytest.raises(ConnectionError) as excinfo:
        await ActivityEnvironment().run(
            activities.invoke_model,
            InvokeModelInput(
                model="model",
                messages=[Message(role="user", contents=["hi"]).to_dict()],
            ),
        )
    assert excinfo.value is error


@workflow.defn
class OneCallWorkflow:
    @workflow.run
    async def run(self, max_attempts: int) -> str:
        client = TemporalChatClient(
            "model",
            retry_policy=RetryPolicy(
                maximum_attempts=max_attempts,
                initial_interval=timedelta(milliseconds=10),
            ),
        )
        return (await Agent(client=client).run("hello")).text


async def _fail(client: Client, status: int, max_attempts: int) -> tuple[Any, Any]:
    plugin = AgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client([StatusError(status)])}
    )
    async with new_worker(client, OneCallWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            OneCallWorkflow.run,
            max_attempts,
            id=workflow_id("fail"),
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()
    return excinfo.value, handle


async def test_retry_exhausted(client: Client) -> None:
    error, handle = await _fail(client, 500, max_attempts=2)
    cause = error.cause
    assert isinstance(cause, ActivityError)
    failure = cause.__cause__
    assert isinstance(failure, ApplicationError)
    assert failure.type == "StatusError" and not failure.non_retryable
    assert await count_scheduled_activities(handle) == {INVOKE_MODEL: 1}
    (started,) = await started_activities(handle)
    assert started.attempt == 2


async def test_non_retryable_status_fails_once(client: Client) -> None:
    error, handle = await _fail(client, 401, max_attempts=5)
    failure = error.cause.__cause__ if error.cause else None
    assert isinstance(failure, ApplicationError)
    assert failure.type == "StatusError" and failure.non_retryable
    (started,) = await started_activities(handle)
    assert started.attempt == 1


@workflow.defn
class WorkflowErrorWorkflow:
    @workflow.run
    async def run(self) -> None:
        raise AgentFrameworkWorkflowError("deliberate failure", {"detail": 1})


async def test_workflow_error_type(client: Client) -> None:
    plugin = AgentFrameworkPlugin()
    async with new_worker(client, WorkflowErrorWorkflow, plugin=plugin) as worker:
        with pytest.raises(WorkflowFailureError) as excinfo:
            await client.execute_workflow(
                WorkflowErrorWorkflow.run,
                id=workflow_id("workflow-error"),
                task_queue=worker.task_queue,
            )
    cause = excinfo.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "AgentFrameworkWorkflowError"
    assert cause.non_retryable
    assert cause.details == ({"detail": 1},)


class _FailClosed(FunctionMiddleware):
    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        raise MiddlewareFailure("policy forbids this tool")


@workflow.defn
class MiddlewareFailureWorkflow:
    @workflow.run
    async def run(self) -> str:
        @tool
        async def noop() -> str:
            """Do nothing."""
            return "nothing"

        agent = Agent(
            client=TemporalChatClient("model"), tools=[noop], middleware=[_FailClosed()]
        )
        return (await agent.run("go")).text


async def test_framework_exception_fails_workflow(client: Client) -> None:
    plugin = AgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [function_call_response("noop", {}, call_id="c1"), text_response("x")]
            )
        }
    )
    async with new_worker(client, MiddlewareFailureWorkflow, plugin=plugin) as worker:
        with pytest.raises(WorkflowFailureError) as excinfo:
            await client.execute_workflow(
                MiddlewareFailureWorkflow.run,
                id=workflow_id("middleware-failure"),
                task_queue=worker.task_queue,
            )
    cause = excinfo.value.cause
    assert isinstance(cause, ApplicationError)
    # Registered through the AgentFrameworkException base, the framework's own
    # error fails the workflow instead of retrying the workflow task forever.
    assert cause.type == "MiddlewareFailure"


@pytest.mark.parametrize(
    ("status", "level"), [(401, logging.ERROR), (429, logging.WARNING)]
)
async def test_provider_error_log_level(
    caplog: pytest.LogCaptureFixture, status: int, level: int
) -> None:
    activities = ChatClientActivities(
        {"model": lambda: FakeChatClient([StatusError(status)])}
    )
    with caplog.at_level(logging.WARNING), pytest.raises(ApplicationError):
        await ActivityEnvironment().run(
            activities.invoke_model,
            InvokeModelInput(
                model="model",
                messages=[Message(role="user", contents=["hi"]).to_dict()],
            ),
        )
    records = [r for r in caplog.records if "StatusError" in r.getMessage()]
    # A failure that is not retried logs at ERROR; one that is retried at WARNING.
    assert records and records[-1].levelno == level
