from typing import Any

from strands.types._events import ToolResultEvent
from strands.types.session import decode_bytes_values
from strands.types.tools import AgentTool, ToolGenerator, ToolResult, ToolSpec, ToolUse

from temporalio import workflow
from temporalio.exceptions import ActivityError

from ._temporal_activity_tool import _activity_error_event
from ._temporal_mcp_client import _CallToolArgs, _MCPToolInfo


class TemporalMCPTool(AgentTool):
    """Workflow-side stub for a single MCP tool; dispatches to an activity."""

    def __init__(
        self,
        server: str,
        info: _MCPToolInfo,
        options: dict[str, Any],
    ) -> None:
        """Bind this tool to a server, its cached info, and activity options."""
        super().__init__()
        self._server = server
        self._info = info
        self._options = options

    @property
    def tool_name(self) -> str:
        """Name the agent sees: the MCP tool name with any client prefix."""
        return self._info.agent_name or self._info.name

    @property
    def tool_spec(self) -> ToolSpec:
        """Strands ToolSpec built from the cached MCP tool info."""
        spec: ToolSpec = {
            "name": self.tool_name,
            "description": self._info.description
            or f"Tool which performs {self._info.name}",
            "inputSchema": {"json": self._info.input_schema},
        }
        if self._info.output_schema:
            spec["outputSchema"] = {"json": self._info.output_schema}
        if self._info.annotations:
            spec["annotations"] = self._info.annotations
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
        """Execute the tool by dispatching to the per-server call-tool activity."""
        try:
            result: ToolResult = await workflow.execute_activity(
                f"{self._server}-call-tool",
                _CallToolArgs(
                    tool_name=self._info.name,
                    arguments=tool_use["input"],
                    tool_use_id=tool_use["toolUseId"],
                ),
                **self._options,
            )
        except ActivityError as error:
            yield _activity_error_event(tool_use["toolUseId"], error)
            return
        yield ToolResultEvent(decode_bytes_values(result))
