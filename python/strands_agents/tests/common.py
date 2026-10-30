import asyncio
from datetime import timedelta
from typing import Any

from strands.types.tools import AgentTool, ToolResult

from temporalio import activity
from temporalio.api.enums.v1 import EventType
from temporalio.client import WorkflowHistory
from temporalio.strands_agents._temporal_mcp_client import _CallToolArgs, _MCPToolInfo
from temporalio.strands_agents._temporal_mcp_tool import TemporalMCPTool
from temporalio.strands_agents.workflow import activity_as_tool


def get_activities(history: WorkflowHistory) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]


# Set once a blocking tool activity starts; clear it before each run.
blocking_tool_started = asyncio.Event()


@activity.defn(name="lookup")
async def blocking_lookup(query: str) -> str:
    """Look up a query; runs until the worker shuts down."""
    blocking_tool_started.set()
    await asyncio.Future()
    return query


@activity.defn(name="blocking-mcp-call-tool")
async def blocking_mcp_call_tool(args: _CallToolArgs) -> ToolResult:
    blocking_tool_started.set()
    await asyncio.Future()
    return ToolResult(toolUseId=args.tool_use_id, status="success", content=[])


def blocking_tool(kind: str) -> AgentTool:
    """A "lookup" tool backed by a blocking "activity" or "mcp" tool activity."""
    options: dict[str, Any] = {"start_to_close_timeout": timedelta(seconds=30)}
    if kind == "activity":
        return activity_as_tool(blocking_lookup, **options)
    return TemporalMCPTool(
        "blocking-mcp",
        _MCPToolInfo("lookup", "Look up a query", {"type": "object"}),
        options,
    )
