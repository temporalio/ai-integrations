"""An MCP server's tools run as Activities: listing, calls, errors and non-text results."""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from locus.agent import Agent
from locus.core.events import ModelChunkEvent
from locus.core.media import has_images
from locus.core.messages import Message, Role, ToolCall
from locus.models.base import ModelResponse
from mcp import Client as MCPClient
from mcp import StdioServerParameters, stdio_client

from temporalio import workflow
from temporalio.client import Client
from temporalio.locus import LocusPlugin, TemporalMCPClient, TemporalModel
from temporalio.worker import Replayer, Worker

LIST_TOOLS = "temporalio.mcp.orders.list-tools"
CALL_TOOL = "temporalio.mcp.orders.call-tool"
OFFERED: list[list[str]] = []


def _orders_factory(calls_file: Path) -> Any:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "orders_mcp_server.py")],
        env={"ORDERS_MCP_CALLS_FILE": str(calls_file)},
    )

    # A default that captures the parameters, as in a loop over servers; the
    # plugin must still call the factory with no arguments.
    def factory(params: StdioServerParameters = params) -> MCPClient:
        return MCPClient(stdio_client(params))

    return factory


class CallsOneTool:
    """Calls the tool the prompt names, then answers with the tool's result."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        del kwargs
        OFFERED.append(sorted(t["function"]["name"] for t in tools or []))
        results = [m for m in messages if m.role == Role.TOOL]
        if results:
            return ModelResponse(
                message=Message.assistant(content=f"Done: {results[-1].content}"),
                stop_reason="stop",
            )
        request = json.loads(
            [m.content for m in messages if m.role == Role.USER][-1] or ""
        )
        call = ToolCall(
            id="call_1", name=request["tool"], arguments=request["arguments"]
        )
        return ModelResponse(
            message=Message.assistant(content=None, tool_calls=[call]),
            stop_reason="tool_calls",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        response = await self.complete(messages, tools, **kwargs)
        yield ModelChunkEvent(content=response.message.content, done=True)


def _model() -> TemporalModel:
    return TemporalModel("m", start_to_close_timeout=timedelta(seconds=30))


@workflow.defn
class McpServersWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        orders = TemporalMCPClient(
            "orders", start_to_close_timeout=timedelta(seconds=30)
        )
        agent = Agent(model=_model(), mcp_servers=[orders])
        return (await agent.arun(prompt)).message or ""


@workflow.defn
class LoadToolsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        orders = TemporalMCPClient(
            "orders",
            prefix="shop",
            allowed_tools=["lookup_order", "cancel_order"],
            start_to_close_timeout=timedelta(seconds=30),
        )
        agent = Agent(model=_model(), tools=await orders.load_tools())
        # Listed once: a second load reuses the first listing.
        await orders.load_tools()
        return (await agent.arun(prompt)).message or ""


@workflow.defn
class ToolResultWorkflow:
    @workflow.run
    async def run(self, name: str) -> dict[str, Any]:
        orders = TemporalMCPClient(
            "orders", start_to_close_timeout=timedelta(seconds=30)
        )
        tools = {tool.name: tool for tool in await orders.load_tools()}
        output = await tools[name].fn(order_id="ORD-7842")
        return {
            "text": str(output),
            "structured": output.structured_content,
            "blocks": [block["type"] for block in output.content_blocks or []],
            "is_error": output.is_error,
            "output_schema": tools[name].output_schema is not None,
        }


async def _run(
    client: Client, tmp_path: Path, workflow_class: Any, argument: str
) -> tuple[Any, list[str], list[str]]:
    calls_file = tmp_path / "calls"
    calls_file.touch()
    plugin = LocusPlugin(
        models={"m": CallsOneTool},
        mcp_clients={"orders": _orders_factory(calls_file)},
    )
    config = client.config()
    config["plugins"] = [plugin]
    client = Client(**config)
    task_queue = f"locus-{uuid.uuid4()}"
    OFFERED.clear()
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[workflow_class],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            workflow_class.run,
            argument,
            id=f"mcp-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        result = await handle.result()
    history = await handle.fetch_history()
    await Replayer(workflows=[workflow_class], plugins=[plugin]).replay_workflow(
        history
    )
    scheduled = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField("activity_task_scheduled_event_attributes")
    ]
    return result, scheduled, calls_file.read_text().split()


def _prompt(tool: str) -> str:
    return json.dumps({"tool": tool, "arguments": {"order_id": "ORD-7842"}})


async def test_mcp_servers_tools_run_as_activities(
    client: Client, tmp_path: Path
) -> None:
    result, scheduled, calls = await _run(
        client, tmp_path, McpServersWorkflow, _prompt("lookup_order")
    )
    assert result == "Done: order ORD-7842: shipped"
    assert scheduled == [
        LIST_TOOLS,
        "locus_model_complete",
        CALL_TOOL,
        "locus_model_complete",
    ]
    # With no Workflow cache every Activity completion replays the run; the
    # server still saw one call.
    assert calls == ["ORD-7842"]
    assert OFFERED[0] == ["cancel_order", "label", "lookup_order", "order_totals"]


async def test_load_tools_with_prefix_and_allowed_tools(
    client: Client, tmp_path: Path
) -> None:
    result, scheduled, calls = await _run(
        client, tmp_path, LoadToolsWorkflow, _prompt("shop_lookup_order")
    )
    assert result == "Done: order ORD-7842: shipped"
    assert scheduled.count(LIST_TOOLS) == 1
    assert calls == ["ORD-7842"]
    assert OFFERED[0] == ["shop_cancel_order", "shop_lookup_order"]


async def test_a_tool_the_server_fails_is_a_tool_error_for_the_model(
    client: Client, tmp_path: Path
) -> None:
    result, scheduled, _ = await _run(
        client, tmp_path, McpServersWorkflow, _prompt("cancel_order")
    )
    assert "order ORD-7842 already shipped" in result
    # The server answered, so the Activity succeeded and was not retried.
    assert scheduled.count(CALL_TOOL) == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "order_totals",
            {
                "structured": {"subtotal": 40.0, "tax": 3.5},
                "blocks": [],
                "is_error": False,
                "output_schema": True,
            },
        ),
        (
            "cancel_order",
            {"structured": None, "blocks": [], "is_error": True},
        ),
        (
            "label",
            {"structured": None, "blocks": ["image"], "is_error": False},
        ),
    ],
)
async def test_tool_results_keep_what_the_server_returned(
    client: Client, tmp_path: Path, name: str, expected: dict[str, Any]
) -> None:
    result, _, _ = await _run(client, tmp_path, ToolResultWorkflow, name)
    for key, value in expected.items():
        assert result[key] == value, key
    if name == "label":
        assert has_images(result["text"])
    if name == "order_totals":
        assert json.loads(result["text"]) == {"subtotal": 40.0, "tax": 3.5}


def test_mcp_client_factories_must_be_callable() -> None:
    with pytest.raises(TypeError, match="not callable"):
        LocusPlugin(
            models={"m": CallsOneTool},
            mcp_clients={"orders": object()},  # type: ignore[dict-item]
        )
