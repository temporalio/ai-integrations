"""The example chat the tests drive: ClaudeSDKClient in a Workflow, with Claude Code in a subprocess on the Worker."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    ToolPermissionContext,
    create_sdk_mcp_server,
    tool,
)

from temporalio import activity, workflow
from temporalio.claude_agent_sdk import ClaudeAgentSDKPlugin, TemporalTransport
from temporalio.client import Client
from temporalio.exceptions import ApplicationError
from temporalio.worker import Worker

TASK_QUEUE = "claude-agent-sdk"


@dataclass
class ToolApprovalRequest:
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass
class ToolApproval:
    tool_use_id: str
    approved: bool


@activity.defn
async def roll_dice(sides: int) -> int:
    return random.randint(1, sides)


@tool("roll_dice", "Roll a die with the given number of sides", {"sides": int})
async def roll_dice_tool(args: dict[str, Any]) -> dict[str, Any]:
    # A tool runs in the Workflow, so it calls an Activity for anything nondeterministic.
    result = await workflow.execute_activity(
        roll_dice, args["sides"], start_to_close_timeout=timedelta(seconds=10)
    )
    return {"content": [{"type": "text", "text": str(result)}]}


@workflow.defn
class AgentWorkflow:
    """Chats with Claude. Prompts arrive through ``send_message``."""

    @workflow.init
    def __init__(self) -> None:
        self.client: ClaudeSDKClient | None = None
        self.ended = False
        self.pending: dict[str, ToolApprovalRequest] = {}
        self.decisions: dict[str, bool] = {}

    @workflow.run
    async def run(self) -> None:
        # Reading is allowed outright. Bash and rolling dice wait for a person to
        # approve them, which needs the default permission mode: auto mode decides
        # without asking.
        options = ClaudeAgentOptions(
            tools=["Bash", "Read", "Glob", "Grep"],
            mcp_servers={"dice": create_sdk_mcp_server("dice", tools=[roll_dice_tool])},
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="default",
            can_use_tool=self.approve,
            setting_sources=[],
            model="claude-sonnet-5-5",
            max_turns=10,
        )
        transport = TemporalTransport(options)
        async with ClaudeSDKClient(options=options, transport=transport) as client:
            self.client = client
            await workflow.wait_condition(
                lambda: self.ended and workflow.all_handlers_finished()
            )

    async def approve(
        self, tool_name: str, tool_input: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResult:
        tool_use_id = context.tool_use_id or ""
        self.pending[tool_use_id] = ToolApprovalRequest(
            tool_use_id, tool_name, tool_input
        )
        try:
            await workflow.wait_condition(lambda: tool_use_id in self.decisions)
        finally:
            # Also when Claude Code takes the request back, which cancels this.
            del self.pending[tool_use_id]
        if self.decisions[tool_use_id]:
            return PermissionResultAllow()
        return PermissionResultDeny(message="The user did not approve this tool call")

    @workflow.update
    async def send_message(self, message: str) -> str:
        """Send Claude a message and return its reply, or its errors."""
        # A message can arrive before the client has connected.
        await workflow.wait_condition(lambda: self.client is not None)
        assert self.client is not None
        await self.client.query(message)
        # The turn ends with its result.
        async for response in self.client.receive_response():
            if isinstance(response, ResultMessage):
                return response.result or "\n".join(response.errors or [])
        return ""

    @send_message.validator
    def validate_send_message(self, message: str) -> None:
        del message
        if self.ended:
            raise ApplicationError("The conversation has ended")

    @workflow.query
    def pending_approvals(self) -> list[ToolApprovalRequest]:
        return list(self.pending.values())

    @workflow.update
    def approve_tool(self, approval: ToolApproval) -> None:
        self.decisions[approval.tool_use_id] = approval.approved

    @workflow.update
    def end_conversation(self) -> None:
        """Reject further messages. A message already sent is still answered."""
        self.ended = True


def create_worker(client: Client) -> Worker:
    return Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[roll_dice],
        plugins=[ClaudeAgentSDKPlugin()],
    )
