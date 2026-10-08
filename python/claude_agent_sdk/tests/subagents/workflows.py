"""Workflows exercising native child calls without caller-owned bridge handlers."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.claude_agent_sdk import (
    AgentState,
    DurableClaudeAgent,
    ToolOutcome,
    activity_as_tool,
)
from temporalio.common import RetryPolicy


@activity.defn
async def child_echo(args: dict[str, Any]) -> ToolOutcome:
    """Return the argument, or a structured tool error."""
    if args.get("error"):
        return ToolOutcome(content="recorded child error", is_error=True)
    if args.get("image"):
        return ToolOutcome(
            blocks=[
                {"type": "text", "text": "recorded child image"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=",
                    },
                },
            ]
        )
    return ToolOutcome(content=args)


@workflow.defn
class NativeWorkflow:
    """Native children call the same approved tools as their parent."""

    def __init__(self) -> None:
        self.agent: DurableClaudeAgent | None = None

    @workflow.run
    async def run(
        self, options: dict[str, Any], state: AgentState | None = None
    ) -> dict[str, Any]:
        self.agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(
                    child_echo,
                    needs_approval=options.get("approval", False),
                    task_queue=options.get("tool_queue"),
                )
            ],
            builtin_tools=["Bash", "PowerShell"]
            if options.get("without_children")
            else ["Agent", "TaskOutput", "Bash", "PowerShell"],
            tool_activities=["Bash", "PowerShell", "mcp__*"],
            tool_activity_task_queue=options.get("tool_queue"),
            tool_approvals=["Bash"] if options.get("approval") else [],
            segment_timeout=timedelta(seconds=options.get("timeout", 100)),
            segment_heartbeat_timeout=timedelta(seconds=3),
            segment_retry_policy=RetryPolicy(maximum_attempts=2),
            tool_activity_retry_policy=RetryPolicy(maximum_attempts=1),
            state=state,
            continue_as_new_args=lambda state: [
                {
                    **options,
                    "again": False,
                    "without_children": options.get("disable_after", False),
                },
                state,
            ],
        )
        result = await self.agent.run("Delegate to a general-purpose subagent.")
        if options.get("again"):
            await self.agent.continue_as_new()
        state = self.agent.state()
        return {"result": result, "calls": self.agent.tool_calls, "state": state}

    @workflow.query
    def state(self) -> AgentState | None:
        """Expose recorded calls even when a native segment cannot be recovered."""
        return self.agent.state() if self.agent else None

    @workflow.query
    def approvals(self) -> list[dict[str, Any]]:
        """Expose the original child arguments while a call waits for approval."""
        return self.agent.pending_approvals() if self.agent else []

    @workflow.signal
    def decide(self, tid: str, approved: bool) -> None:
        """Decide on an original child ID."""
        assert self.agent is not None
        self.agent.decide(tid, approved)
