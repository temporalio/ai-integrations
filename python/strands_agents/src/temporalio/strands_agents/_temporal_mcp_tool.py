from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from strands.tools.mcp.mcp_client import MIME_TO_FORMAT
from strands.tools.mcp.mcp_types import MCPToolResult
from strands.types._events import ToolResultEvent
from strands.types.tools import (
    AgentTool,
    ToolGenerator,
    ToolResultContent,
    ToolSpec,
    ToolUse,
)

from temporalio.exceptions import ActivityError

from ._temporal_activity_tool import _activity_error_event

if TYPE_CHECKING:
    from mcp.types import CallToolResult, Tool

    from temporalio.mcp import TemporalMCPClient as DurableMCPClient

# Blob MIME types that Strands' MCPClient hands to the model as text.
_TEXT_MIME_TYPES = (
    "application/json",
    "application/xml",
    "application/javascript",
    "application/yaml",
    "application/x-yaml",
)


class TemporalMCPTool(AgentTool):
    """Workflow-side stub for a single MCP tool; calls it through an activity."""

    def __init__(
        self, client: DurableMCPClient, tool: Tool, prefix: str | None = None
    ) -> None:
        """Bind this tool to its server's durable client and listed MCP tool."""
        super().__init__()
        self._client = client
        self._tool = tool
        self._name = f"{prefix}_{tool.name}" if prefix else tool.name

    @property
    def mcp_tool(self) -> Tool:
        """The MCP tool as the server listed it, as on Strands' ``MCPAgentTool``."""
        return self._tool

    @property
    def tool_name(self) -> str:
        """Name the agent sees: the MCP tool name with any client prefix."""
        return self._name

    @property
    def tool_spec(self) -> ToolSpec:
        """Strands ToolSpec built from the listed MCP tool, as ``MCPAgentTool`` does."""
        spec: ToolSpec = {
            "name": self.tool_name,
            "description": self._tool.description
            or f"Tool which performs {self._tool.name}",
            "inputSchema": {"json": self._tool.input_schema},
        }
        if self._tool.output_schema:
            spec["outputSchema"] = {"json": self._tool.output_schema}
        if self._tool.annotations:
            annotations = self._tool.annotations.model_dump(
                by_alias=True, exclude_none=True
            )
            if annotations:
                spec["annotations"] = annotations
        return spec

    @property
    def tool_type(self) -> str:
        """Tool kind identifier used by Strands."""
        return "temporal_mcp"

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> ToolGenerator:
        """Execute the tool through the server's call-tool activity."""
        try:
            result = await self._client.call_tool(self._tool.name, tool_use["input"])
        except ActivityError as error:
            yield _activity_error_event(tool_use["toolUseId"], error)
            return
        yield ToolResultEvent(_tool_result(tool_use["toolUseId"], result))


def _tool_result(tool_use_id: str, result: CallToolResult) -> MCPToolResult:
    """Map an MCP tool result the way Strands' ``MCPClient`` does."""
    tool_result = MCPToolResult(
        status="error" if result.is_error else "success",
        toolUseId=tool_use_id,
        content=[
            mapped
            for content in result.content
            if (mapped := _tool_result_content(content)) is not None
        ],
    )
    if result.structured_content is not None:
        tool_result["structuredContent"] = result.structured_content
    if result.meta:
        tool_result["metadata"] = result.meta
    tool_result["isError"] = result.is_error
    return tool_result


def _tool_result_content(content: Any) -> ToolResultContent | None:
    """Map one MCP content block; drop the ones Strands' ``MCPClient`` drops.

    This runs in the workflow, so data that does not decode is dropped rather
    than raised.
    """
    from mcp.types import (
        BlobResourceContents,
        EmbeddedResource,
        ImageContent,
        TextContent,
        TextResourceContents,
    )

    if isinstance(content, TextContent):
        return {"text": content.text}
    if isinstance(content, ImageContent):
        return _image(content.mime_type, content.data)
    if isinstance(content, EmbeddedResource):
        resource = content.resource
        if isinstance(resource, TextResourceContents):
            return {"text": resource.text}
        if isinstance(resource, BlobResourceContents):
            mime_type = resource.mime_type or ""
            if (
                mime_type.startswith("text/")
                or mime_type in _TEXT_MIME_TYPES
                or mime_type.endswith(("+json", "+xml"))
            ):
                data = _decode(resource.blob)
                return None if data is None else {"text": data.decode(errors="replace")}
            return _image(mime_type, resource.blob)
    return None


def _image(mime_type: str, data: str) -> ToolResultContent | None:
    image_format = MIME_TO_FORMAT.get(mime_type)
    raw = _decode(data)
    if image_format is None or raw is None:
        return None
    return {"image": {"format": image_format, "source": {"bytes": raw}}}


def _decode(data: str) -> bytes | None:
    try:
        return base64.b64decode(data)
    except ValueError:
        return None
