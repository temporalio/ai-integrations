"""An agent whose tasks call several tools at once."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool
from temporalio.common import RetryPolicy
from temporalio.exceptions import FailureError

with workflow.unsafe.imports_passed_through():
    from tests.endless.activities import count, publish

FAST = RetryPolicy(initial_interval=timedelta(milliseconds=200), maximum_attempts=5)
TOOL: dict[str, Any] = {
    "start_to_close_timeout": timedelta(seconds=30),
    "retry_policy": FAST,
}


@workflow.defn
class ParallelWorkflow:
    """Runs prompts one after another; a failed task does not stop the next one."""

    @workflow.init
    def __init__(self, prompts: list[str], max_segments: int | None = None) -> None:
        del prompts
        self.agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(count, **TOOL),
                activity_as_tool(publish, needs_approval=True, **TOOL),
            ],
            max_segments=max_segments,
            segment_retry_policy=FAST,
            live_output=True,
        )

    @workflow.run
    async def run(
        self, prompts: list[str], max_segments: int | None = None
    ) -> list[str]:
        """Each prompt's answer, or ``error: <type>: ...``."""
        del max_segments
        answers: list[str] = []
        for prompt in prompts:
            try:
                answers.append(await self.agent.run(prompt))
            except FailureError as err:  # ApplicationError or ActivityError
                answers.append(f"error: {type(err).__name__}: {err.message}")
        return answers

    @workflow.update
    def review(self, tool_use_id: str, approved: bool) -> None:
        """Approve or reject a call."""
        self.agent.decide(tool_use_id, approved)

    @review.validator
    def check_review(self, tool_use_id: str, approved: bool) -> None:
        """Refuse decisions on calls that are not waiting."""
        del approved
        self.agent.validate_decision(tool_use_id)

    @workflow.query
    def pending_approvals(self) -> list[dict[str, Any]]:
        """Calls waiting for a decision."""
        return self.agent.pending_approvals()

    @workflow.query
    def tool_calls(self) -> list[dict[str, Any]]:
        """Every tool call of this run, with its status."""
        return self.agent.tool_calls


@workflow.defn
class SurviveCancelWorkflow:
    """The first task is cancelled while the calls of one message are out; the
    Workflow goes on with a second task."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(count, **TOOL),
                activity_as_tool(publish, needs_approval=True, **TOOL),
            ],
            segment_retry_policy=FAST,
        )

    @workflow.run
    async def run(self, first: str, second: str) -> list[str]:
        """Run both tasks; report how the first one ended."""
        answers: list[str] = []
        try:
            answers.append(await self.agent.run(first))
        except asyncio.CancelledError:
            answers.append("first task cancelled")
        answers.append(await self.agent.run(second))
        return answers

    @workflow.query
    def pending_approvals(self) -> list[dict[str, Any]]:
        """Calls waiting for a decision."""
        return self.agent.pending_approvals()
