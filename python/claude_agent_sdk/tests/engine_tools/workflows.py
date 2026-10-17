"""An agent that uses Claude Code's tools (Bash by default), and an MCP server's tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.claude_agent_sdk import AgentState, DurableClaudeAgent, activity_as_tool
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tests.endless.activities import count

FAST = RetryPolicy(initial_interval=timedelta(milliseconds=200), maximum_attempts=5)


@dataclass
class ShellOptions:
    """Test knobs for ShellWorkflow.

    Attributes:
        tool_activities: The agent's ``tool_activities``.
        tool_approvals: The agent's ``tool_approvals``.
        segment_timeout: Seconds each model segment attempt may take.
        repeatable_tools: The agent's ``repeatable_tools``.
        tool_timeout: Seconds each tool step attempt may take.
        tool_retry_initial: The tool steps' retry policy: first interval, seconds.
        tool_retry_backoff: Its backoff coefficient.
        tool_retry_max_interval: Its maximum interval in seconds (0: Temporal's
            default, 100 times the first).
        tool_retry_attempts: Its maximum attempts (0: unlimited).
        builtin_tools: The agent's Claude Code tools.
        state: The agent's state to start from (for example its session and its
            folder's first snapshot, for runners with a workspace store).
    """

    tool_activities: list[str] = field(default_factory=lambda: ["Bash", "mcp__*"])
    tool_approvals: list[str] = field(default_factory=list)
    segment_timeout: float = 120
    repeatable_tools: list[str] = field(default_factory=list)
    tool_timeout: float = 600
    tool_retry_initial: float = 0.2
    tool_retry_backoff: float = 2.0
    tool_retry_max_interval: float = 0
    tool_retry_attempts: int = 5
    builtin_tools: list[str] = field(default_factory=lambda: ["Bash"])
    state: AgentState | None = None


@workflow.defn
class ShellWorkflow:
    """Runs one task with Claude Code's tools (and a durable ``count``)."""

    @workflow.init
    def __init__(self, prompt: str, options: ShellOptions) -> None:
        del prompt
        self.agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(
                    count,
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=FAST,
                )
            ],
            builtin_tools=options.builtin_tools,
            tool_activities=options.tool_activities,
            tool_approvals=options.tool_approvals,
            repeatable_tools=options.repeatable_tools,
            tool_activity_timeout=timedelta(seconds=options.tool_timeout),
            segment_timeout=timedelta(seconds=options.segment_timeout),
            segment_heartbeat_timeout=timedelta(seconds=10),
            segment_retry_policy=FAST,
            tool_activity_retry_policy=RetryPolicy(  # FAST unless a test says otherwise
                initial_interval=timedelta(seconds=options.tool_retry_initial),
                backoff_coefficient=options.tool_retry_backoff,
                maximum_interval=timedelta(seconds=options.tool_retry_max_interval)
                if options.tool_retry_max_interval
                else None,
                maximum_attempts=options.tool_retry_attempts,
            ),
            state=options.state,
        )

    @workflow.run
    async def run(self, prompt: str, options: ShellOptions) -> str:
        """Run the task."""
        del options
        return await self.agent.run(prompt)

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

    @workflow.query
    def agent_state(self) -> AgentState:
        """The agent's state (its session, and its folder's snapshot)."""
        return self.agent.state()

    @workflow.query
    def total_cost_usd(self) -> float:
        """The model cost the segments reported, in USD."""
        return self.agent.total_cost_usd
