"""Workflow coverage for original native batches and interactive tools."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tests.native_batches.activities import record


@workflow.defn
class NativeWorkflow:
    """Run an agent with every enabled native tool managed by Activities."""

    @workflow.init
    def __init__(self, prompt: str, builtins: list[str]) -> None:
        del prompt
        self.agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(record, start_to_close_timeout=timedelta(seconds=30))
            ],
            builtin_tools=builtins,
            segment_heartbeat_timeout=timedelta(seconds=5),
            segment_retry_policy=RetryPolicy(maximum_attempts=2),
            tool_activity_retry_policy=RetryPolicy(maximum_attempts=2),
        )

    @workflow.run
    async def run(self, prompt: str, builtins: list[str]) -> str:
        """Run one task."""
        del builtins
        return await self.agent.run(prompt)

    @workflow.query
    def interactions(self) -> list[dict[str, Any]]:
        """Return pending native interactions."""
        return self.agent.pending_interactions()

    @workflow.query
    def calls(self) -> list[dict[str, Any]]:
        """Return durable calls and their statuses."""
        return self.agent.tool_calls

    @workflow.update
    def respond(self, tool_use_id: str, response: Any) -> bool:
        """Record a validated native interaction answer."""
        return self.agent.respond(tool_use_id, response)

    @respond.validator
    def validate_response(self, tool_use_id: str, response: Any) -> None:
        """Reject malformed or repeated answers before acceptance."""
        self.agent.validate_response(tool_use_id, response)


@workflow.defn
class ConcurrentNativeWorkflow:
    """Run independent native controllers in the same Workflow."""

    def __init__(self) -> None:
        self.agents = [
            DurableClaudeAgent(
                tools=[activity_as_tool(record)],
                builtin_tools=["Read"],
                segment_heartbeat_timeout=timedelta(seconds=5),
                segment_retry_policy=RetryPolicy(maximum_attempts=2),
                tool_activity_retry_policy=RetryPolicy(maximum_attempts=2),
            )
            for _ in range(2)
        ]

    @workflow.run
    async def run(self) -> list[str]:
        """Run both agents concurrently."""
        return list(
            await asyncio.gather(
                self.agents[0].run("read native agent one"),
                self.agents[1].run("read native agent two"),
            )
        )
