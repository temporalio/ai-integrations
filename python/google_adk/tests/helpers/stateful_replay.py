"""Workflows for histories recorded with stateful MCP operations.

``tests/histories/stateful_mcp_*.json`` were recorded with these workflows on
earlier versions of the plugin, and ``test_stateful_mcp.py`` replays them on
this one.
"""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from typing import cast

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityCancellationType, ActivityConfig

with workflow.unsafe.imports_passed_through():
    from google.adk.events import EventActions
    from google.adk.tools.tool_context import ToolContext

    from temporalio.google_adk import TemporalStatefulMcpToolSet

MCP_SERVER = """
import asyncio

from mcp.server.fastmcp import FastMCP

server = FastMCP("replay")


@server.tool()
async def slow(text: str) -> str:
    await asyncio.sleep(2)
    return text


server.run()
"""


@activity.defn
async def plain_activity() -> str:
    await asyncio.sleep(4)
    return "plain"


@activity.defn
async def follow_up_a() -> str:
    return "a"


@activity.defn
async def follow_up_b() -> str:
    return "b"


def _tool_context() -> ToolContext:
    return cast(
        ToolContext,
        cast(
            object,
            SimpleNamespace(
                tool_confirmation=None,
                function_call_id=None,
                _event_actions=EventActions(),
            ),
        ),
    )


def _toolset(
    host_task_queue: str, wait_cancellation: bool = False
) -> TemporalStatefulMcpToolSet:
    config = ActivityConfig(
        start_to_close_timeout=timedelta(seconds=20),
        retry_policy=RetryPolicy(maximum_attempts=1),
    )
    if wait_cancellation:
        config["cancellation_type"] = (
            ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
        )
    return TemporalStatefulMcpToolSet(
        "stateful_replay",
        config=config,
        server_session_config=ActivityConfig(
            task_queue=host_task_queue, start_to_close_timeout=timedelta(hours=1)
        ),
    )


@workflow.defn
class StatefulMcpGatherWorkflow:
    """A stateful tool call and a plain activity in two gathered branches."""

    @workflow.run
    async def run(self, host_task_queue: str) -> list[str]:
        async with _toolset(host_task_queue) as toolset:
            tools = {t.name: t for t in await toolset.get_tools()}

            async def branch_a() -> str:
                result = await tools["slow"].run_async(
                    args={"text": "a"}, tool_context=_tool_context()
                )
                follow_up = await workflow.execute_activity(
                    follow_up_a,
                    task_queue=host_task_queue,
                    start_to_close_timeout=timedelta(seconds=10),
                )
                return f"{result}:{follow_up}"

            async def branch_b() -> str:
                result = await workflow.execute_activity(
                    plain_activity,
                    task_queue=host_task_queue,
                    start_to_close_timeout=timedelta(seconds=10),
                )
                follow_up = await workflow.execute_activity(
                    follow_up_b,
                    task_queue=host_task_queue,
                    start_to_close_timeout=timedelta(seconds=10),
                )
                return f"{result}:{follow_up}"

            return list(await asyncio.gather(branch_a(), branch_b()))


@workflow.defn
class StatefulMcpCancelWorkflow:
    """A stateful tool call awaited directly, then another activity."""

    @workflow.run
    async def run(self, host_task_queue: str) -> str:
        async with _toolset(host_task_queue) as toolset:
            tools = {t.name: t for t in await toolset.get_tools()}
            result = await tools["slow"].run_async(
                args={"text": "x"}, tool_context=_tool_context()
            )
            await workflow.execute_activity(
                follow_up_a,
                task_queue=host_task_queue,
                start_to_close_timeout=timedelta(seconds=10),
            )
            return str(result)


@workflow.defn
class StatefulMcpCancelWaitWorkflow:
    """StatefulMcpCancelWorkflow with WAIT_CANCELLATION_COMPLETED operations."""

    @workflow.run
    async def run(self, host_task_queue: str) -> str:
        async with _toolset(host_task_queue, wait_cancellation=True) as toolset:
            tools = {t.name: t for t in await toolset.get_tools()}
            result = await tools["slow"].run_async(
                args={"text": "x"}, tool_context=_tool_context()
            )
            await workflow.execute_activity(
                follow_up_a,
                task_queue=host_task_queue,
                start_to_close_timeout=timedelta(seconds=10),
            )
            return str(result)
