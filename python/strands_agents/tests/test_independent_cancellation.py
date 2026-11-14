"""Activity cancellation without tool-task cancellation remains a tool result."""

import asyncio
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import Tool
from strands import Agent
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent
from strands.tools.executors import SequentialToolExecutor

from temporalio import activity
from temporalio.exceptions import ActivityError, CancelledError
from temporalio.mcp import TemporalMCPClient
from temporalio.strands_agents._temporal_mcp_tool import TemporalMCPTool
from temporalio.strands_agents.workflow import activity_as_tool
from tests.mock_model import MockModel


@activity.defn(name="cancelled_tool")
async def _cancelled_tool() -> None:
    raise AssertionError("The activity boundary is mocked")


class _CaughtCancellation(HookProvider):
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._consume_cancel)

    async def _consume_cancel(self, event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] != "cancelled_tool":
            return
        # A caught cancellation leaves the task's cancellation count nonzero.
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass


@pytest.mark.parametrize("kind", ["activity", "mcp"])
@pytest.mark.parametrize("sequential", [False, True], ids=["concurrent", "sequential"])
@pytest.mark.parametrize(
    "prior_cancel", [False, True], ids=["no-prior-cancel", "caught-cancel"]
)
async def test_independent_activity_cancellation_reports_tool_error(
    kind: str, sequential: bool, prior_cancel: bool
) -> None:
    # Inject the error at the activity boundary: the dev server rejects an
    # activity failing with CanceledFailureInfo without a cancel request.
    error = ActivityError(
        "Activity task canceled",
        scheduled_event_id=1,
        started_event_id=2,
        identity="test-worker",
        activity_type="cancelled_tool",
        activity_id="tool-1",
        retry_state=None,
    )
    error.__cause__ = CancelledError("Cancelled independently")
    if kind == "activity":
        tool = activity_as_tool(
            _cancelled_tool, start_to_close_timeout=timedelta(seconds=15)
        )
        target = "temporalio.strands_agents._temporal_activity_tool.workflow.execute_activity"
    else:
        tool = TemporalMCPTool(
            TemporalMCPClient("cancelled"),
            Tool(name="cancelled_tool", input_schema={"type": "object"}),
        )
        target = "temporalio.mcp.TemporalMCPClient.call_tool"
    agent = Agent(
        model=MockModel([{"name": "cancelled_tool", "input": {}}, "Done!"]),
        tools=[tool],
        hooks=[_CaughtCancellation()] if prior_cancel else [],
        tool_executor=SequentialToolExecutor() if sequential else None,
        callback_handler=None,
    )
    with patch(target, new=AsyncMock(side_effect=error)) as operation:
        assert (
            str(await asyncio.wait_for(agent.invoke_async("Use the tool"), 5))
            == "Done!\n"
        )
        operation.assert_awaited_once()

    results = [
        block["toolResult"]
        for message in agent.messages
        for block in message["content"]
        if "toolResult" in block
    ]
    assert results == [
        {
            "toolUseId": "mock-tool-1",
            "status": "error",
            "content": [{"text": "Cancelled independently"}],
        }
    ]
