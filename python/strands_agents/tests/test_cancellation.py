"""Workflow cancellation propagates through activity and MCP tools."""

import asyncio
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from mcp.types import Tool
from strands.tools.executors import SequentialToolExecutor

from temporalio import activity, workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowExecutionStatus, WorkflowFailureError
from temporalio.exceptions import CancelledError
from temporalio.mcp import TemporalMCPClient
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents._temporal_mcp_tool import TemporalMCPTool
from temporalio.strands_agents.workflow import activity_as_tool
from temporalio.worker import Replayer, Worker
from temporalio.workflow import ActivityCancellationType
from tests.common import get_activities
from tests.mock_model import MockModel


class _BlockingTool:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.finish = asyncio.Event()
        self.finished = asyncio.Event()

    async def _wait(self) -> None:
        self.started.set()
        try:
            while not self.finish.is_set():
                activity.heartbeat()
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            self.cancelled.set()
            # Hold cleanup open to check WAIT_CANCELLATION_COMPLETED.
            await self.finish.wait()
            raise
        finally:
            self.finished.set()

    @activity.defn(name="blocking_tool")
    async def run(self) -> str:
        """Wait until the test releases the tool."""
        await self._wait()
        return "finished"

    @activity.defn(name="temporalio.mcp.blocked.call-tool")
    async def call_tool(self, _request: dict[str, Any]) -> dict[str, Any]:
        await self._wait()
        return {"content": [{"type": "text", "text": "finished"}]}


@workflow.defn
class _ToolCancellationWorkflow:
    @workflow.run
    async def run(
        self,
        kind: str,
        sequential: bool,
        cancellation_type: ActivityCancellationType,
    ) -> str:
        if kind == "activity":
            tools = [
                activity_as_tool(
                    _BlockingTool.run,
                    start_to_close_timeout=timedelta(seconds=30),
                    heartbeat_timeout=timedelta(seconds=1),
                    cancellation_type=cancellation_type,
                )
            ]
        else:
            tools = [
                TemporalMCPTool(
                    TemporalMCPClient(
                        "blocked",
                        activity_config={
                            "start_to_close_timeout": timedelta(seconds=30),
                            "heartbeat_timeout": timedelta(seconds=1),
                            "cancellation_type": cancellation_type,
                        },
                    ),
                    Tool(name="blocking_tool", input_schema={"type": "object"}),
                )
            ]
        agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=15),
            tools=tools,
            tool_executor=SequentialToolExecutor() if sequential else None,
        )
        return str(await agent.invoke_async("Call the blocking tool."))


@pytest.mark.parametrize("kind", ["activity", "mcp"])
@pytest.mark.parametrize("sequential", [False, True], ids=["concurrent", "sequential"])
@pytest.mark.parametrize(
    "cancellation_type",
    list(ActivityCancellationType),
    ids=lambda value: value.name.lower(),
)
async def test_cancel_during_tool_call(
    client: Client,
    kind: str,
    sequential: bool,
    cancellation_type: ActivityCancellationType,
) -> None:
    blocked = _BlockingTool()
    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [{"name": "blocking_tool", "input": {}}, "Unexpected continuation"]
            )
        }
    )
    task_queue = f"test_tool_cancellation-{uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[_ToolCancellationWorkflow],
        activities=[blocked.run, blocked.call_tool],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            _ToolCancellationWorkflow.run,
            args=[kind, sequential, cancellation_type],
            id=f"test_tool_cancellation_{uuid4()}",
            task_queue=task_queue,
        )
        try:
            await asyncio.wait_for(blocked.started.wait(), 10)
            await handle.cancel()
            if (
                cancellation_type
                == ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
            ):
                await asyncio.wait_for(blocked.cancelled.wait(), 5)
                assert (
                    await handle.describe()
                ).status == WorkflowExecutionStatus.RUNNING
                blocked.finish.set()

            with pytest.raises(WorkflowFailureError) as raised:
                await asyncio.wait_for(handle.result(), 5)
            assert isinstance(raised.value.cause, CancelledError)
            assert (await handle.describe()).status == WorkflowExecutionStatus.CANCELED

            history = await handle.fetch_history()
            assert get_activities(history) == [
                "strands_get_model_config",
                "invoke_model",
                "blocking_tool"
                if kind == "activity"
                else "temporalio.mcp.blocked.call-tool",
            ]
            event_types = [event.event_type for event in history.events]
            assert (EventType.EVENT_TYPE_ACTIVITY_TASK_CANCELED in event_types) == (
                cancellation_type
                == ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
            )
            assert (
                EventType.EVENT_TYPE_ACTIVITY_TASK_CANCEL_REQUESTED in event_types
            ) == (cancellation_type != ActivityCancellationType.ABANDON)
        finally:
            blocked.finish.set()
            if (await handle.describe()).status == WorkflowExecutionStatus.RUNNING:
                await handle.terminate("Cancellation test finished")
            await asyncio.wait_for(blocked.finished.wait(), 5)

    await Replayer(
        workflows=[_ToolCancellationWorkflow], plugins=[plugin]
    ).replay_workflow(history)
