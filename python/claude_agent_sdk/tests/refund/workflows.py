"""The refund agent: a normal Temporal Workflow with a durable Claude agent inside."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tests.refund.activities import email_customer, issue_refund, look_up_order

FAST_RETRY = RetryPolicy(
    initial_interval=timedelta(milliseconds=500),
    backoff_coefficient=2.0,
    maximum_attempts=6,
)
ORDER = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}},
    "required": ["order_id"],
}
REFUND = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "amount": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["order_id", "amount"],
}
EMAIL = {
    "type": "object",
    "properties": {
        "to": {"type": "string"},
        "subject": {"type": "string"},
        "body": {"type": "string"},
    },
    "required": ["to", "subject", "body"],
}
REFUND_PROMPT = (
    "You handle refund requests for an online store. Steps: 1) Look up the order with "
    "look_up_order. 2) If the item arrived broken, refund the full order total with "
    "issue_refund. 3) Email the customer with email_customer. If a refund is rejected or "
    "the order does not exist, do not retry and do not email; say so. Finish with a "
    "one-line summary."
)
MANAGER = "manager@shop.example"  # the only person allowed to approve refunds here


@workflow.defn
class RefundAgentWorkflow:
    """Handles one refund request; refunds need the manager's approval."""

    def __init__(self) -> None:
        tool_opts: dict[str, Any] = {
            "start_to_close_timeout": timedelta(seconds=10),
            "retry_policy": FAST_RETRY,
        }
        self.agent = DurableClaudeAgent(
            system_prompt=REFUND_PROMPT,
            tools=[
                activity_as_tool(look_up_order, input_schema=ORDER, **tool_opts),
                activity_as_tool(
                    issue_refund, input_schema=REFUND, needs_approval=True, **tool_opts
                ),
                activity_as_tool(email_customer, input_schema=EMAIL, **tool_opts),
            ],
            segment_timeout=timedelta(minutes=5),
            segment_heartbeat_timeout=timedelta(seconds=5),
            segment_retry_policy=FAST_RETRY,
            approvers=[MANAGER],
        )

    @workflow.run
    async def run(self, request: str) -> str:
        """Handle the request."""
        return await self.agent.run(request)

    @workflow.update
    def review(self, tool_use_id: str, approved: bool, approver: str) -> str:
        """Approve or reject a refund."""
        self.agent.decide(tool_use_id, approved, approver)
        return "approved" if approved else "rejected"

    @review.validator
    def check_review(self, tool_use_id: str, approved: bool, approver: str) -> None:
        """Refuse unknown approvers before anything is recorded."""
        del approved
        self.agent.validate_decision(tool_use_id, approver)

    @workflow.query
    def pending_approvals(self) -> list[dict[str, Any]]:
        """Tool calls waiting for a decision."""
        return self.agent.pending_approvals()

    @workflow.query
    def tool_calls(self) -> list[dict[str, Any]]:
        """Every tool call so far."""
        return self.agent.tool_calls

    @workflow.query
    def cost_usd(self) -> float:
        """Model cost so far."""
        return self.agent.total_cost_usd
