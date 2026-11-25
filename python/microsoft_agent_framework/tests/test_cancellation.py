"""Cancelling the workflow while an activity tool, an MCP tool or a model call runs."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import timedelta

import pytest
from agent_framework import Agent, FunctionInvocationConfiguration
from mcp import Client as MCPClient

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
    TemporalMCPToolProvider,
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

# Constructing MCPServer imports rich, which reads random at import.
with workflow.unsafe.imports_passed_through():
    from tests.echo_mcp_server import server

CALL_TOOL = "temporalio.mcp.echo.call-tool"


@activity.defn
async def slow_lookup(key: str) -> str:
    """Look a key up, slowly."""
    await asyncio.sleep(20)
    return key


@dataclass
class CancelInput:
    mode: str
    # With concurrent invocation the loop runs the tool calls of a response
    # under asyncio.gather, whose own cancellation re-raises in the workflow
    # task whatever the tool returned. Sequential invocation awaits the tool
    # from the workflow task itself, so the cancelled Activity reaches the
    # tool as an ActivityError and only the tool decides whether the run stops.
    concurrent_tools: bool = True


@workflow.defn
class CancelWorkflow:
    @workflow.run
    async def run(self, input: CancelInput) -> str:
        tools = (
            [
                activity_as_tool(
                    slow_lookup, start_to_close_timeout=timedelta(seconds=60)
                )
            ]
            if input.mode == "tool"
            else []
        )
        providers = (
            [
                TemporalMCPToolProvider(
                    "echo", start_to_close_timeout=timedelta(seconds=60)
                )
            ]
            if input.mode == "mcp"
            else []
        )
        agent = Agent(
            client=TemporalChatClient(
                "model",
                start_to_close_timeout=timedelta(seconds=60),
                function_invocation_configuration=FunctionInvocationConfiguration(
                    allow_concurrent_invocation=input.concurrent_tools
                ),
            ),
            tools=tools,
            context_providers=providers,
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
    client: Client,
    plugin: MicrosoftAgentFrameworkPlugin,
    input: CancelInput,
    running: str,
) -> None:
    async with new_worker(
        client, CancelWorkflow, plugin=plugin, activities=[slow_lookup]
    ) as worker:
        handle = await client.start_workflow(
            CancelWorkflow.run,
            input,
            id=workflow_id(f"cancel-{input.mode}"),
            task_queue=worker.task_queue,
        )
        await _wait_for_running_activity(handle, running)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()
        assert isinstance(excinfo.value.cause, CancelledError)
        assert (await handle.describe()).status == WorkflowExecutionStatus.CANCELED


@pytest.mark.parametrize(
    "concurrent_tools", [True, False], ids=["gather", "sequential"]
)
async def test_cancel_during_activity_tool(
    client: Client, concurrent_tools: bool
) -> None:
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
    await _cancel_and_check(
        client, plugin, CancelInput("tool", concurrent_tools), running="slow_lookup"
    )


async def test_cancel_during_model_call(client: Client) -> None:
    slow = SlowFakeChatClient(["never reached"], delay=20)
    plugin = MicrosoftAgentFrameworkPlugin(chat_clients={"model": lambda: slow})
    await _cancel_and_check(client, plugin, CancelInput("model"), running=INVOKE_MODEL)


@pytest.mark.parametrize(
    "concurrent_tools", [True, False], ids=["gather", "sequential"]
)
async def test_cancel_during_mcp_call(client: Client, concurrent_tools: bool) -> None:
    plugin = MicrosoftAgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response("wait", {"seconds": 20}, call_id="call-1"),
                    text_response("never reached"),
                ]
            )
        },
        mcp_clients={"echo": lambda: MCPClient(server)},
    )
    await _cancel_and_check(
        client, plugin, CancelInput("mcp", concurrent_tools), running=CALL_TOOL
    )
