"""Workflows for the hardening tests: built-in tools after a cancel, two live agents."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tests.endless.activities import count

COUNT = activity_as_tool(count, start_to_close_timeout=timedelta(seconds=30))


@workflow.defn
class WriterWorkflow:
    """An agent that may use Claude Code's Write tool, with a short heartbeat timeout."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(
            tools=[COUNT],
            builtin_tools=["Write"],
            segment_heartbeat_timeout=timedelta(seconds=2),
            segment_retry_policy=RetryPolicy(maximum_attempts=1),
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Run one task."""
        return await self.agent.run(prompt)


@workflow.defn
class TwoLiveAgentsWorkflow:
    """Wrong: two agents in one Workflow both ask for live output."""

    def __init__(self) -> None:
        self.first = DurableClaudeAgent(tools=[COUNT], live_output=True)
        self.second = DurableClaudeAgent(tools=[COUNT], live_output=True)

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Never reached: initialization fails."""
        return await self.first.run(prompt)
