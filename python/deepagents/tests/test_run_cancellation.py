"""Cancellation closes new runs correctly while old failure paths replay."""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from unittest.mock import patch

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage

from temporalio import activity, workflow
from temporalio.client import WorkflowExecutionStatus, WorkflowFailureError
from temporalio.deepagents import (
    DeepAgentsPlugin,
    DeepAgentsWorkflowError,
    activity_as_tool,
    run_deep_agent,
)
from temporalio.deepagents.testing import mock_model_provider
from temporalio.exceptions import ApplicationError, CancelledError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from tests.helpers.deepagents import count_scheduled_activities


@activity.defn
async def cancellation_wait(value: str) -> str:
    while True:
        activity.heartbeat(value)
        await asyncio.sleep(0.1)


@activity.defn
async def cancellation_cleanup() -> str:
    return "cleaned up"


@workflow.defn
class CancelledRunWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> str:
        waiting = activity_as_tool(
            cancellation_wait,
            activity_options=workflow.ActivityConfig(
                start_to_close_timeout=timedelta(seconds=30)
            ),
        )
        agent = create_deep_agent(
            name="cancelled-run", model="fake:model", tools=[waiting]
        )
        try:
            result = await run_deep_agent(agent, input, state_snapshot=state_snapshot)
        except DeepAgentsWorkflowError:
            if not input["cleanup"]:
                raise
            return await workflow.execute_activity(
                cancellation_cleanup, start_to_close_timeout=timedelta(seconds=10)
            )
        return str(result["messages"][-1].content)


@pytest.mark.parametrize("old_behavior", [False, True])
@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("checkpoint_driver", [False, True])
async def test_run_cancellation_and_replay(
    env: WorkflowEnvironment,
    old_behavior: bool,
    cleanup: bool,
    checkpoint_driver: bool,
) -> None:
    call = {"name": "cancellation_wait", "args": {"value": "go"}, "id": "wait-1"}
    responses = [AIMessage(content="", tool_calls=[call]), AIMessage(content="done")]
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider(responses))
    task_queue = f"run-cancel-{uuid.uuid4()}"
    original_patched = workflow.patched

    def selected_behavior(patch_id: str) -> bool:
        if old_behavior and patch_id == "deepagents.raise-cancellation":
            return False
        if (
            not checkpoint_driver
            and patch_id == "deepagents.checkpoint-continue-as-new"
        ):
            return False
        return original_patched(patch_id)

    with patch.object(workflow, "patched", selected_behavior):
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[CancelledRunWorkflow],
            activities=[cancellation_wait, cancellation_cleanup],
            plugins=[plugin],
            max_cached_workflows=0,
        ):
            handle = await env.client.start_workflow(
                CancelledRunWorkflow.run,
                {
                    "messages": [{"role": "user", "content": "Wait."}],
                    "cleanup": cleanup,
                },
                id=task_queue,
                task_queue=task_queue,
            )
            for _ in range(100):
                if (await count_scheduled_activities(handle))["cancellation_wait"]:
                    break
                await asyncio.sleep(0.1)
            else:
                pytest.fail("The waiting tool was never scheduled.")
            await handle.cancel()
            if old_behavior and cleanup:
                assert await handle.result() == "cleaned up"
                expected_status = WorkflowExecutionStatus.COMPLETED
            else:
                with pytest.raises(WorkflowFailureError) as raised:
                    await handle.result()
                if old_behavior:
                    cause = raised.value.cause
                    assert isinstance(cause, ApplicationError)
                    assert cause.type == DeepAgentsWorkflowError.TYPE
                    expected_status = WorkflowExecutionStatus.FAILED
                else:
                    assert isinstance(raised.value.cause, CancelledError)
                    expected_status = WorkflowExecutionStatus.CANCELED

    assert (await handle.describe()).status == expected_status
    history = await handle.fetch_history()
    await Replayer(
        workflows=[CancelledRunWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(history)
