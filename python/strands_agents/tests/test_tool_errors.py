from collections.abc import AsyncIterable
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from mcp.types import Tool
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterToolCallEvent
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolResult, ToolSpec

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.exceptions import ApplicationError
from temporalio.mcp import TemporalMCPClient
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents._temporal_mcp_tool import TemporalMCPTool
from temporalio.strands_agents.workflow import activity_as_tool
from temporalio.worker import Replayer, Worker
from tests.mock_model import MockModel

_MODEL_TOOL_RESULTS: list[ToolResult] = []


class _RecordingModel(MockModel):
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        for message in messages:
            for content in message["content"]:
                if "toolResult" in content:
                    _MODEL_TOOL_RESULTS.append(content["toolResult"])
        async for event in super().stream(
            messages,
            tool_specs,
            system_prompt,
            tool_choice=tool_choice,
            system_prompt_content=system_prompt_content,
            invocation_state=invocation_state,
            **kwargs,
        ):
            yield event


class _ToolResultHook(HookProvider):
    def __init__(self) -> None:
        self.failures: list[dict[str, str | None]] = []

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(AfterToolCallEvent, self._record)

    def _record(self, event: AfterToolCallEvent) -> None:
        if event.result["status"] == "error":
            error = event.exception
            self.failures.append(
                {
                    "text": event.result["content"][0].get("text"),
                    "exception_type": type(error).__name__ if error else None,
                    "exception_message": error.message
                    if isinstance(error, ApplicationError)
                    else None,
                }
            )


@activity.defn
async def failing_activity_tool(location: str) -> None:
    raise ApplicationError(f"Unknown location: {location}", non_retryable=True)


@activity.defn(name="temporalio.mcp.failing-mcp.call-tool")
async def failing_mcp_call_tool(_request: dict[str, Any]) -> None:
    raise ApplicationError("MCP server unavailable", non_retryable=True)


_ELICITATION = {
    "mode": "url",
    "elicitationId": "e1",
    "url": "https://example.com/authorize",
    "message": "Authorize access",
}


# How temporalio-mcp reports an MCPError with code -32042.
@activity.defn(name="temporalio.mcp.elicit-mcp.call-tool")
async def elicit_mcp_call_tool(_request: dict[str, Any]) -> None:
    raise ApplicationError(
        "URL elicitation required",
        -32042,
        {"elicitations": [_ELICITATION]},
        type="MCPProtocolError",
        non_retryable=True,
    )


@workflow.defn
class _FailingToolWorkflow:
    @workflow.run
    async def run(self, kind: str) -> list[dict[str, str | None]]:
        if kind == "activity":
            tools = [
                activity_as_tool(
                    failing_activity_tool,
                    start_to_close_timeout=timedelta(seconds=15),
                )
            ]
        else:
            tools = [
                TemporalMCPTool(
                    TemporalMCPClient(
                        f"{kind}-mcp",
                        activity_config={
                            "start_to_close_timeout": timedelta(seconds=15)
                        },
                    ),
                    Tool(
                        name="list_files",
                        description="List files",
                        input_schema={"type": "object"},
                    ),
                )
            ]
        hook = _ToolResultHook()
        agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=15),
            tools=tools,
            hooks=[hook],
        )
        await agent.invoke_async("Use the tool")
        return hook.failures


def _failure(text: str, message: str | None) -> dict[str, str | None]:
    return {
        "text": text,
        "exception_type": "ApplicationError" if message else None,
        "exception_message": message,
    }


@pytest.mark.parametrize(
    ("kind", "tool_name", "tool_input", "failure"),
    [
        (
            "activity",
            "failing_activity_tool",
            {"location": "Atlantis"},
            _failure("Unknown location: Atlantis", "Unknown location: Atlantis"),
        ),
        (
            "failing",
            "list_files",
            {"path": "/"},
            _failure("MCP server unavailable", "MCP server unavailable"),
        ),
        # Rejected before the call-tool activity runs.
        (
            "failing",
            "list_files",
            "hello",
            _failure("Tool execution failed: tool input must be a JSON object", None),
        ),
        # The model gets the URLs to send the user to, as from Strands' MCPClient.
        (
            "elicit",
            "list_files",
            {"path": "/"},
            _failure(
                "MCP Elicitation required: [URL elicitation required] with data "
                '[{"mode": "url", "message": "Authorize access", '
                '"url": "https://example.com/authorize", "elicitation_id": "e1"}]',
                "URL elicitation required",
            ),
        ),
    ],
)
async def test_failed_activity_tool_reports_cause(
    client: Client,
    kind: str,
    tool_name: str,
    tool_input: Any,
    failure: dict[str, str | None],
) -> None:
    _MODEL_TOOL_RESULTS.clear()
    task_queue = f"test_failed_tool-{uuid4()}"
    plugin = StrandsPlugin(
        models={
            "mock": lambda: _RecordingModel(
                [{"name": tool_name, "input": tool_input}, "Done!"]
            )
        }
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[_FailingToolWorkflow],
        activities=[failing_activity_tool, failing_mcp_call_tool, elicit_mcp_call_tool],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            _FailingToolWorkflow.run,
            kind,
            id=f"test_failed_tool_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == [failure]

    assert len(_MODEL_TOOL_RESULTS) == 1
    assert _MODEL_TOOL_RESULTS[0]["status"] == "error"
    assert _MODEL_TOOL_RESULTS[0]["content"] == [{"text": failure["text"]}]

    await Replayer(workflows=[_FailingToolWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )
