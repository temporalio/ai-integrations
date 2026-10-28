import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from mcp import Client as MCPClient
from mcp import StdioServerParameters, stdio_client
from mcp.types import Tool, ToolAnnotations
from strands.tools.mcp import ToolFilters

from temporalio import workflow
from temporalio.client import Client
from temporalio.mcp import TemporalMCPClient as DurableMCPClient
from temporalio.strands_agents import StrandsPlugin, TemporalAgent, TemporalMCPClient
from temporalio.strands_agents._temporal_mcp_client import _included
from temporalio.strands_agents._temporal_mcp_tool import TemporalMCPTool
from temporalio.worker import Replayer, Worker
from tests.common import get_activities
from tests.mock_model import MockModel

_ECHO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
}


def _tool(
    name: str, annotations: ToolAnnotations | None = None, prefix: str | None = None
) -> TemporalMCPTool:
    # Never called: these tests only read the tool's name, spec and filters.
    return TemporalMCPTool(
        DurableMCPClient("gh"),
        Tool(
            name=name,
            description=f"Run {name}",
            input_schema=_ECHO_SCHEMA,
            annotations=annotations,
        ),
        prefix,
    )


def test_tools_apply_client_prefix() -> None:
    tool = _tool("echo", prefix="gh")

    assert tool.mcp_tool.name == "echo"
    assert tool.tool_name == "gh_echo"
    assert tool.tool_spec["name"] == "gh_echo"


def test_tools_without_prefix_keep_the_server_name() -> None:
    tool = _tool("echo")

    assert tool.tool_name == "echo"
    assert tool.tool_spec["name"] == "echo"
    assert "annotations" not in tool.tool_spec


@pytest.mark.parametrize(
    ("filters", "included"),
    [
        ({"allowed": ["echo"]}, ["echo"]),
        ({"allowed": [re.compile("ot")]}, ["other"]),
        ({"rejected": [lambda tool: tool.tool_name == "gh_echo"]}, ["other"]),
    ],
)
def test_tool_filters_still_match_the_server_name_when_prefixed(
    filters: ToolFilters, included: list[str]
) -> None:
    tools = [_tool("echo", prefix="gh"), _tool("other", prefix="gh")]

    assert [t.mcp_tool.name for t in tools if _included(t, filters)] == included


def test_tools_carry_annotations() -> None:
    annotations = ToolAnnotations(read_only_hint=True, destructive_hint=False)

    assert _tool("echo", annotations, prefix="gh").tool_spec.get("annotations") == {
        "readOnlyHint": True,
        "destructiveHint": False,
    }
    assert "annotations" not in _tool("plain", prefix="gh").tool_spec


def _labeled_client_factory(label: str) -> Any:
    return lambda: MCPClient(
        stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=[str(Path(__file__).parent / "labeled_mcp_server.py"), label],
            )
        )
    )


@workflow.defn
class MCPPrefixWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[
                TemporalMCPClient(
                    "gh",
                    prefix="gh",
                    cache_tools=True,
                    start_to_close_timeout=timedelta(seconds=30),
                ),
                TemporalMCPClient(
                    "gl",
                    prefix="gl",
                    cache_tools=True,
                    start_to_close_timeout=timedelta(seconds=30),
                ),
            ],
        )

    @workflow.run
    async def run(self, prompt: str) -> dict[str, Any]:
        await self.agent.invoke_async(prompt)
        results = [
            block["toolResult"]["content"][0].get("text")
            for message in self.agent.messages
            for block in message["content"]
            if "toolResult" in block
        ]
        return {
            "tools": self.agent.tool_registry.get_all_tools_config(),
            "results": results,
        }


async def test_mcp_prefix_and_annotations(client: Client) -> None:
    """Two servers that expose ``echo`` stay distinct, and calls use the server name."""
    task_queue = f"test_mcp_prefix_and_annotations-{uuid4()}"
    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [
                    {"name": "gh_echo", "input": {"message": "one"}},
                    {"name": "gl_echo", "input": {"message": "two"}},
                    "Done!",
                ]
            )
        },
        mcp_clients={
            "gh": _labeled_client_factory("gh"),
            "gl": _labeled_client_factory("gl"),
        },
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPPrefixWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MCPPrefixWorkflow.run,
            "echo twice",
            id=f"test_mcp_prefix_and_annotations_{uuid4()}",
            task_queue=task_queue,
        )
        result = await handle.result()

    # The model sees the prefixed names and the server's annotations.
    assert sorted(result["tools"]) == ["gh_echo", "gl_echo"]
    for name in ("gh_echo", "gl_echo"):
        assert result["tools"][name]["name"] == name
        assert result["tools"][name]["annotations"] == {
            "readOnlyHint": True,
            "destructiveHint": False,
        }
    # Each call reached its own server, which only knows the tool as ``echo``.
    assert result["results"] == ["gh:one", "gl:two"]

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "strands_get_model_config",
        "temporalio.mcp.gh.list-tools",
        "temporalio.mcp.gl.list-tools",
        "invoke_model",
        "temporalio.mcp.gh.call-tool",
        "invoke_model",
        "temporalio.mcp.gl.call-tool",
        "invoke_model",
    ]

    await Replayer(workflows=[MCPPrefixWorkflow], plugins=[plugin]).replay_workflow(
        history
    )
