import asyncio
import zlib
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp.types import Tool
from strands import tool
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
from strands.types.tools import AgentTool

from temporalio import activity
from temporalio.api.common.v1 import Payload
from temporalio.api.enums.v1 import EventType
from temporalio.client import WorkflowHistory
from temporalio.converter import PayloadCodec
from temporalio.mcp import TemporalMCPClient
from temporalio.strands_agents._temporal_mcp_tool import TemporalMCPTool
from temporalio.strands_agents.workflow import activity_as_hook, activity_as_tool


def get_activities(history: WorkflowHistory) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]


def get_heartbeat_timeouts(history: WorkflowHistory) -> list[timedelta | None]:
    return [
        event.activity_task_scheduled_event_attributes.heartbeat_timeout.ToTimedelta()
        or None
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]


def load_history(name: str, workflow_id: str = "legacy-replay") -> WorkflowHistory:
    """Load the recorded history ``tests/histories/<name>.json``."""
    path = Path(__file__).parent / "histories" / f"{name}.json"
    return WorkflowHistory.from_json(workflow_id, path.read_text())


class CompressionCodec(PayloadCodec):
    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload(
                metadata={"encoding": b"binary/zlib"},
                data=zlib.compress(payload.SerializeToString()),
            )
            for payload in payloads
        ]

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload.FromString(zlib.decompress(payload.data)) for payload in payloads
        ]


# Set once a blocking tool or hook activity starts; clear it before each run.
blocking_tool_started = asyncio.Event()


@activity.defn(name="lookup")
async def blocking_lookup(query: str) -> str:
    """Look up a query; runs until the worker shuts down."""
    blocking_tool_started.set()
    await asyncio.Future()
    return query


@activity.defn(name="temporalio.mcp.blocking-mcp.call-tool")
async def blocking_mcp_call_tool(_request: dict[str, Any]) -> dict[str, Any]:
    blocking_tool_started.set()
    await asyncio.Future()
    return {"content": []}


@activity.defn
async def blocking_hook(_tool_name: str) -> None:
    """Runs until the worker shuts down."""
    blocking_tool_started.set()
    await asyncio.Future()


@tool
def lookup(query: str) -> str:
    """Look up a query."""
    return query


class BlockingHook(HookProvider):
    """Runs the blocking hook activity after each tool call."""

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(
            AfterToolCallEvent,
            activity_as_hook(
                blocking_hook,
                activity_input=lambda event: event.tool_use["name"],
                start_to_close_timeout=timedelta(seconds=30),
            ),
        )


def blocking_hooks(kind: str) -> list[HookProvider]:
    """The hooks that go with ``blocking_tool(kind)``."""
    return [BlockingHook()] if kind == "hook" else []


def blocking_tool(kind: str) -> AgentTool:
    """A "lookup" tool that blocks in an "activity" or "mcp" tool activity.

    With "hook", the tool runs in the workflow and ``blocking_hooks`` blocks after it.
    """
    options: dict[str, Any] = {"start_to_close_timeout": timedelta(seconds=30)}
    if kind == "activity":
        return activity_as_tool(blocking_lookup, **options)
    if kind == "hook":
        return lookup
    return TemporalMCPTool(
        TemporalMCPClient(
            "blocking-mcp",
            activity_config={"start_to_close_timeout": timedelta(seconds=30)},
        ),
        Tool(
            name="lookup",
            description="Look up a query",
            input_schema={"type": "object"},
        ),
    )
