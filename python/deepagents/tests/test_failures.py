"""Error handling: retry classification and the workflow-failure type.

Two dep-free paths run against a real server: the HTTP-error → Temporal retry
translation, and that a raised :class:`DeepAgentsWorkflowError` surfaces to the
client as a non-retryable failure with a stable ``ApplicationError.type`` (never
a stringified peer exception). The model-instance validation error needs
LangChain and guards on its import. A cancelled ``run_deep_agent`` must close
the workflow as cancelled, not wrap the cancellation in that failure type.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import timedelta

import pytest

from temporalio.testing import WorkflowEnvironment

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)
from temporalio import activity, workflow
from temporalio.client import WorkflowExecutionStatus, WorkflowFailureError
from temporalio.deepagents import (
    DeepAgentsPlugin,
    DeepAgentsWorkflowError,
    activity_as_tool,
    run_deep_agent,
)
from temporalio.deepagents._activity import _translate_api_error
from temporalio.deepagents.testing import mock_model_provider
from temporalio.exceptions import ApplicationError, CancelledError
from temporalio.worker import Worker
from tests.helpers.deepagents import count_scheduled_activities

# Bound off importorskip, as in test_tools.py, so type checkers do not resolve
# the name against this same-named test directory.
create_deep_agent = pytest.importorskip("deepagents").create_deep_agent


class _FakeResponse:
    def __init__(self, headers: dict) -> None:
        self.headers = headers


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int, headers: dict | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = _FakeResponse(headers or {})


def test_error_classification() -> None:
    # 429 is retryable and honors the upstream Retry-After.
    err = _translate_api_error(_FakeHTTPError(429, {"retry-after": "7"}))
    assert isinstance(err, ApplicationError)
    assert err.non_retryable is False
    assert err.next_retry_delay == timedelta(seconds=7)

    # 400 is a client error: non-retryable.
    e400 = _translate_api_error(_FakeHTTPError(400))
    assert isinstance(e400, ApplicationError)
    assert e400.non_retryable is True

    # 503 is retryable by default...
    e503 = _translate_api_error(_FakeHTTPError(503))
    assert isinstance(e503, ApplicationError)
    assert e503.non_retryable is False
    # ...unless the server explicitly says not to.
    forced = _translate_api_error(_FakeHTTPError(503, {"x-should-retry": "false"}))
    assert isinstance(forced, ApplicationError)
    assert forced.non_retryable is True

    # retry-after-ms wins over retry-after when both are present.
    ms = _translate_api_error(
        _FakeHTTPError(429, {"retry-after-ms": "250", "retry-after": "7"})
    )
    assert isinstance(ms, ApplicationError)
    assert ms.next_retry_delay == timedelta(milliseconds=250)

    # A non-HTTP exception is not recognized, so the caller falls through.
    assert _translate_api_error(ValueError("nope")) is None


@workflow.defn
class FailingWorkflow:
    @workflow.run
    async def run(self) -> None:
        raise DeepAgentsWorkflowError("deliberate non-retryable failure")


@pytest.mark.asyncio
async def test_workflow_failure_type(env: WorkflowEnvironment) -> None:
    plugin = DeepAgentsPlugin()
    async with Worker(
        env.client,
        task_queue="da-fail",
        workflows=[FailingWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            FailingWorkflow.run,
            id=f"da-fail-{uuid.uuid4()}",
            task_queue="da-fail",
        )
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()

    cause = excinfo.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "deepagents.DeepAgentsWorkflowError"
    assert cause.non_retryable is True


@activity.defn
async def wait_tool(value: str) -> str:
    while True:
        activity.heartbeat(value)
        await asyncio.sleep(0.1)


@workflow.defn
class CancelledAgentWorkflow:
    @workflow.run
    async def run(self, input: dict) -> str:
        tool = activity_as_tool(wait_tool, start_to_close_timeout=timedelta(seconds=30))
        agent = create_deep_agent(model="fake:model", tools=[tool])
        result = await run_deep_agent(agent, input)
        return str(result["messages"][-1].content)


@pytest.mark.asyncio
async def test_cancelled_run_deep_agent_closes_cancelled(
    env: WorkflowEnvironment,
) -> None:
    from langchain_core.messages import AIMessage

    call = {"name": "wait_tool", "args": {"value": "go"}, "id": "wait-1"}
    responses = [AIMessage(content="", tool_calls=[call]), AIMessage(content="done")]
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(responses))
    task_queue = f"da-cancel-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[CancelledAgentWorkflow],
        activities=[wait_tool],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            CancelledAgentWorkflow.run,
            {"messages": [{"role": "user", "content": "Call wait_tool."}]},
            id=task_queue,
            task_queue=task_queue,
        )
        for _ in range(100):
            if (await count_scheduled_activities(handle))["wait_tool"]:
                break
            await asyncio.sleep(0.1)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()

    # The cancellation is not wrapped in DeepAgentsWorkflowError.
    assert isinstance(excinfo.value.cause, CancelledError)
    assert (await handle.describe()).status == WorkflowExecutionStatus.CANCELED


@pytest.mark.asyncio
async def test_unwrappable_model_instance() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    # A string is auto-wrapped; a TemporalModel passes through; a live model
    # instance is rejected at the workflow boundary with the typed failure.
    from temporalio.deepagents import TemporalModel
    from temporalio.deepagents._model import _wrap_model_arg

    assert isinstance(_wrap_model_arg("anthropic:claude"), TemporalModel)
    tm = TemporalModel(model="anthropic:claude")
    assert _wrap_model_arg(tm) is tm
    with pytest.raises(DeepAgentsWorkflowError):
        _wrap_model_arg(FakeListChatModel(responses=["hi"]))
