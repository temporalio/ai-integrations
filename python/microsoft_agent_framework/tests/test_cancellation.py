"""Cancelling the workflow while an activity tool or a model call runs."""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta

import pytest
from agent_framework import Agent

from temporalio import activity, workflow
from temporalio.api.enums.v1 import PendingActivityState
from temporalio.client import (
    Client,
    WorkflowExecutionStatus,
    WorkflowFailureError,
    WorkflowHandle,
)
from temporalio.exceptions import CancelledError
from temporalio.microsoft_agent_framework import (
    MicrosoftAgentFrameworkPlugin,
    TemporalChatClient,
)
from temporalio.microsoft_agent_framework.testing import (
    fake_chat_client,
    function_call_response,
    text_response,
)
from temporalio.microsoft_agent_framework.workflow import activity_as_tool
from tests.helpers.microsoft_agent_framework import (
    INVOKE_MODEL,
    SlowFakeChatClient,
    new_worker,
    workflow_id,
)


@activity.defn
async def slow_lookup(key: str) -> str:
    """Look a key up, slowly."""
    await asyncio.sleep(20)
    return key


@workflow.defn
class CancelWorkflow:
    @workflow.run
    async def run(self, mode: str) -> str:
        tools = (
            [
                activity_as_tool(
                    slow_lookup, start_to_close_timeout=timedelta(seconds=60)
                )
            ]
            if mode == "tool"
            else []
        )
        agent = Agent(
            client=TemporalChatClient(
                "model", start_to_close_timeout=timedelta(seconds=60)
            ),
            tools=tools,
        )
        # Agent Framework catches Exception around a tool; a cancelled workflow
        # must still stop, so the cancellation has to arrive as CancelledError.
        return (await agent.run("go")).text


async def _wait_for_running_activity(
    handle: WorkflowHandle, activity_type: str
) -> None:
    # ActivityTaskStarted only reaches history when the activity ends, so the
    # running activity shows up in the execution's pending activities instead.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        description = await handle.describe()
        for pending in description.raw_description.pending_activities:
            if (
                pending.activity_type.name == activity_type
                and pending.state == PendingActivityState.PENDING_ACTIVITY_STATE_STARTED
            ):
                return
        await asyncio.sleep(0.1)
    raise AssertionError(f"{activity_type} did not start")


async def _cancel_and_check(
    client: Client, plugin: MicrosoftAgentFrameworkPlugin, mode: str, running: str
) -> None:
    async with new_worker(
        client, CancelWorkflow, plugin=plugin, activities=[slow_lookup]
    ) as worker:
        handle = await client.start_workflow(
            CancelWorkflow.run,
            mode,
            id=workflow_id(f"cancel-{mode}"),
            task_queue=worker.task_queue,
        )
        await _wait_for_running_activity(handle, running)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()
        assert isinstance(excinfo.value.cause, CancelledError)
        assert (await handle.describe()).status == WorkflowExecutionStatus.CANCELED


async def test_cancel_during_activity_tool(client: Client) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "slow_lookup", {"key": "k"}, call_id="call-1"
                    ),
                    text_response("never reached"),
                ]
            )
        }
    )
    await _cancel_and_check(client, plugin, "tool", running="slow_lookup")


async def test_cancel_during_model_call(client: Client) -> None:
    slow = SlowFakeChatClient(["never reached"], delay=20)
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": lambda: slow})
    await _cancel_and_check(client, plugin, "model", running=INVOKE_MODEL)
