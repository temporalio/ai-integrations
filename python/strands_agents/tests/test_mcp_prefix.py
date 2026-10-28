import sys
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp import StdioServerParameters, stdio_client
from mcp.types import Tool, ToolAnnotations
from strands.tools.mcp.mcp_client import MCPClient

from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.strands_agents import StrandsPlugin, TemporalAgent, TemporalMCPClient
from temporalio.strands_agents._temporal_mcp_client import _MCPToolInfo, _tool_infos
from temporalio.strands_agents._temporal_mcp_tool import TemporalMCPTool
from temporalio.worker import Replayer, Worker
from tests.common import get_activities
from tests.mock_model import MockModel

_ECHO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
}


def _client(**kwargs: Any) -> MCPClient:
    # Never connected: _tool_infos only reads the client's prefix and filters.
    return MCPClient(
        lambda: stdio_client(StdioServerParameters(command="unused")), **kwargs
    )


def _tool(name: str, annotations: ToolAnnotations | None = None) -> Tool:
    return Tool(
        name=name,
        description=f"Run {name}",
        inputSchema=_ECHO_SCHEMA,
        annotations=annotations,
    )


def test_tool_infos_apply_client_prefix() -> None:
    infos = _tool_infos(_client(prefix="gh"), [_tool("echo")])

    assert [(info.name, info.agent_name) for info in infos] == [("echo", "gh_echo")]
    tool = TemporalMCPTool("gh", infos[0], {})
    assert tool.tool_name == "gh_echo"
    assert tool.tool_spec["name"] == "gh_echo"


def test_tool_infos_without_prefix_keep_the_server_name() -> None:
    infos = _tool_infos(_client(), [_tool("echo")])

    assert [(info.name, info.agent_name) for info in infos] == [("echo", "echo")]
    tool = TemporalMCPTool("gh", infos[0], {})
    assert tool.tool_name == "echo"
    assert tool.tool_spec["name"] == "echo"
    assert "annotations" not in tool.tool_spec


def test_tool_filters_still_match_the_server_name_when_prefixed() -> None:
    client = _client(prefix="gh", tool_filters={"allowed": ["echo"]})

    infos = _tool_infos(client, [_tool("echo"), _tool("other")])

    assert [(info.name, info.agent_name) for info in infos] == [("echo", "gh_echo")]


def test_tool_infos_carry_annotations() -> None:
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
    infos = _tool_infos(
        _client(prefix="gh"), [_tool("echo", annotations), _tool("plain")]
    )

    assert infos[0].annotations == {"readOnlyHint": True, "destructiveHint": False}
    assert infos[1].annotations is None
    annotated = TemporalMCPTool("gh", infos[0], {}).tool_spec
    assert annotated.get("annotations") == {
        "readOnlyHint": True,
        "destructiveHint": False,
    }
    assert "annotations" not in TemporalMCPTool("gh", infos[1], {}).tool_spec


def test_tool_info_recorded_before_agent_name_still_decodes() -> None:
    """A list-tools result recorded by an older version replays with the old name."""
    payload = pydantic_data_converter.payload_converter.to_payload(
        {"name": "echo", "description": "d", "input_schema": _ECHO_SCHEMA}
    )

    (info,) = pydantic_data_converter.payload_converter.from_payloads(
        [payload], [_MCPToolInfo]
    )

    assert (info.agent_name, info.annotations) == (None, None)
    assert TemporalMCPTool("gh", info, {}).tool_name == "echo"


def _labeled_client_factory(label: str, prefix: str) -> Any:
    return lambda: MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=[str(Path(__file__).parent / "labeled_mcp_server.py"), label],
            )
        ),
        prefix=prefix,
    )


@workflow.defn
class MCPPrefixWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[
                TemporalMCPClient(
                    "gh", cache_tools=True, start_to_close_timeout=timedelta(seconds=30)
                ),
                TemporalMCPClient(
                    "gl", cache_tools=True, start_to_close_timeout=timedelta(seconds=30)
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
            "gh": _labeled_client_factory("gh", "gh"),
            "gl": _labeled_client_factory("gl", "gl"),
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
        "gh-list-tools",
        "gl-list-tools",
        "invoke_model",
        "gh-call-tool",
        "invoke_model",
        "gl-call-tool",
        "invoke_model",
    ]

    await Replayer(workflows=[MCPPrefixWorkflow], plugins=[plugin]).replay_workflow(
        history
    )
