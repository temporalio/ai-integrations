"""Agents for the tests of the conversation the Workflow holds."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.claude_agent_sdk import AgentState, DurableClaudeAgent, activity_as_tool
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from tests.endless.activities import count, publish
    from tests.storage.workflows import fetch_document

FAST = RetryPolicy(initial_interval=timedelta(milliseconds=200), maximum_attempts=5)
STEADY = RetryPolicy(
    initial_interval=timedelta(milliseconds=200), backoff_coefficient=1.0
)
"""Retries every 200 ms, without limit: a step waits until the Workers are fixed."""
TOOL: dict[str, Any] = {
    "start_to_close_timeout": timedelta(seconds=60),
    "retry_policy": FAST,
}


@workflow.defn
class PatientWorkflow:
    """Counts, then asks for approval to publish; its steps retry without limit."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(count, **TOOL),
                activity_as_tool(publish, needs_approval=True, **TOOL),
            ],
            segment_retry_policy=STEADY,
            segment_heartbeat_timeout=timedelta(seconds=10),
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Run the task."""
        return await self.agent.run(prompt)

    @workflow.update
    def review(self, tool_use_id: str, approved: bool) -> None:
        """Approve or reject a call."""
        self.agent.decide(tool_use_id, approved)

    @workflow.query
    def pending_approvals(self) -> list[dict[str, Any]]:
        """Calls waiting for a decision."""
        return self.agent.pending_approvals()


@workflow.defn
class HandOverWorkflow:
    """Runs one task, then continues as new by an explicit call."""

    @workflow.init
    def __init__(self, prompt: str | None, state: AgentState | None = None) -> None:
        self.agent = DurableClaudeAgent(
            tools=[activity_as_tool(fetch_document, **TOOL)],
            state=state,
            segment_retry_policy=FAST,
        )

    @workflow.run
    async def run(self, prompt: str | None, state: AgentState | None = None) -> str:
        """Run the task and hand the conversation over, or report what arrived."""
        if state is not None:
            return f"continued with {len(state.conversation)} entries"
        answer = await self.agent.run(prompt)
        try:
            await self.agent.continue_as_new()
        except ApplicationError as err:
            return f"{answer}; {err.message}"
