"""Workflows for histories recorded with stateful MCP operations.

``tests/histories/stateful-mcp-*-workflow-history.json`` were recorded with
these workflows on earlier versions of the plugin, and
``test_openai_replay.py`` replays them on this one.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityCancellationType, ActivityConfig

with workflow.unsafe.imports_passed_through():
    import temporalio.openai_agents as openai_agents


@activity.defn
async def stateful_replay_plain() -> str:
    await asyncio.sleep(4)
    return "plain"


@activity.defn
async def stateful_replay_follow_up_a() -> str:
    return "a"


@activity.defn
async def stateful_replay_follow_up_b() -> str:
    return "b"


def _server(host_task_queue: str, wait_cancellation: bool = False):
    # No schedule-to-start or schedule-to-close timeout: only the session
    # watches over these operations.
    config = ActivityConfig(
        start_to_close_timeout=timedelta(seconds=20),
        retry_policy=RetryPolicy(maximum_attempts=1),
    )
    if wait_cancellation:
        config["cancellation_type"] = (
            ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
        )
    return openai_agents.workflow.stateful_mcp_server(
        "StatefulReplay",
        config=config,
        server_session_config=ActivityConfig(
            task_queue=host_task_queue, start_to_close_timeout=timedelta(hours=1)
        ),
    )


async def _follow_up(
    activity: Callable[[], Awaitable[str]], host_task_queue: str
) -> str:
    return await workflow.execute_activity(
        activity,
        task_queue=host_task_queue,
        start_to_close_timeout=timedelta(seconds=10),
    )


async def _gather(host_task_queue: str, wait_cancellation: bool) -> list[str]:
    async with _server(host_task_queue, wait_cancellation) as server:

        async def branch_a() -> str:
            await server.call_tool("slow", {})
            return await _follow_up(stateful_replay_follow_up_a, host_task_queue)

        async def branch_b() -> str:
            await workflow.execute_activity(
                stateful_replay_plain,
                task_queue=host_task_queue,
                start_to_close_timeout=timedelta(seconds=10),
            )
            return await _follow_up(stateful_replay_follow_up_b, host_task_queue)

        return list(await asyncio.gather(branch_a(), branch_b()))


@workflow.defn
class StatefulMcpGatherWorkflow:
    """A stateful tool call and a plain activity in two gathered branches."""

    @workflow.run
    async def run(self, host_task_queue: str) -> list[str]:
        return await _gather(host_task_queue, wait_cancellation=False)


@workflow.defn
class StatefulMcpGatherWaitWorkflow:
    """As StatefulMcpGatherWorkflow, with WAIT_CANCELLATION_COMPLETED operations."""

    @workflow.run
    async def run(self, host_task_queue: str) -> list[str]:
        return await _gather(host_task_queue, wait_cancellation=True)


@workflow.defn
class StatefulMcpCancelWorkflow:
    """A stateful tool call awaited directly, then another activity."""

    @workflow.run
    async def run(self, host_task_queue: str) -> str:
        async with _server(host_task_queue) as server:
            await server.call_tool("slow", {})
            return await _follow_up(stateful_replay_follow_up_a, host_task_queue)
