"""Workflows for the checks tests: two agents at once, two calls at once, misuse."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tests.checks.activities import quick, slow
    from tests.endless.activities import count

ONCE = RetryPolicy(maximum_attempts=1)
COUNT = activity_as_tool(
    count, start_to_close_timeout=timedelta(seconds=30), retry_policy=ONCE
)
QUICK = activity_as_tool(
    quick, start_to_close_timeout=timedelta(seconds=30), retry_policy=ONCE
)
SLOW = activity_as_tool(
    slow,
    start_to_close_timeout=timedelta(minutes=2),
    heartbeat_timeout=timedelta(seconds=5),
    retry_policy=ONCE,
    cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
)


@workflow.defn
class TwoAgentsWorkflow:
    """Two agents in one Workflow, working at the same time."""

    def __init__(self) -> None:
        self.agents = [
            DurableClaudeAgent(tools=[COUNT], max_segments=None) for _ in range(2)
        ]

    @workflow.run
    async def run(self, first: str, second: str) -> list[str]:
        """Give each agent its task; return both answers."""
        a, b = self.agents
        return list(await asyncio.gather(a.run(first), b.run(second)))

    @workflow.query
    def prompts(self) -> list[list[str]]:
        """The prompts in each agent's conversation."""
        out: list[list[str]] = []
        for agent in self.agents:
            entries = [json.loads(t) for t in agent.state().conversation]
            out.append([e["prompt"] for e in entries if "prompt" in e])
        return out


@workflow.defn
class BothAtOnceWorkflow:
    """Claude calls two tools in one message: one returns at once, one runs on."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(tools=[QUICK, SLOW], max_segments=None)

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Run the task."""
        return await self.agent.run(prompt)

    @workflow.query
    def calls(self) -> list[dict[str, Any]]:
        """The tool calls and their status."""
        return self.agent.tool_calls


@workflow.defn
class MisuseWorkflow:
    """Uses the agent wrongly when asked: a second task while one runs, no prompt."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(tools=[COUNT], max_segments=None)

    @workflow.run
    async def run(self, prompt: str | None) -> str:
        """Run a task (``None``: no prompt, and no unfinished task)."""
        return await self.agent.run(prompt)

    @workflow.update
    async def ask(self, prompt: str) -> str:
        """Start another task while the first one runs."""
        return await self.agent.run(prompt)

    @workflow.query
    def calls(self) -> list[dict[str, Any]]:
        """The tool calls and their status."""
        return self.agent.tool_calls
