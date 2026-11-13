"""MCP tools for an agent, listed and called through ``temporalio-mcp`` Activities."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from agent_framework import (
    AgentSession,
    Content,
    ContextProvider,
    FunctionTool,
    SessionContext,
)
from agent_framework.exceptions import ToolExecutionException

from temporalio.common import Priority, RetryPolicy
from temporalio.microsoft_agent_framework.workflow import (
    ApprovalMode,
    MicrosoftAgentFrameworkWorkflowError,
    _propagate_workflow_cancellation,
)
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityConfig,
    VersioningIntent,
)

if TYPE_CHECKING:
    from mcp.types import CallToolResult, Tool

    from temporalio.mcp import TemporalMCPClient


class TemporalMCPToolProvider(ContextProvider):
    """Gives an agent the tools of an MCP server registered on the worker.

    Pass it as ``Agent(context_providers=[TemporalMCPToolProvider("github")])``.
    ``server`` names an ``mcp.Client`` factory registered with
    ``MicrosoftAgentFrameworkPlugin(mcp_clients=...)``; the factory, transport and
    credentials stay on the worker. Before each run the provider lists the
    server's tools with the ``temporalio.mcp.{server}.list-tools`` Activity and
    contributes one ``FunctionTool`` per tool, with the server's full input
    schema, whose call runs the ``temporalio.mcp.{server}.call-tool`` Activity.
    Agent Framework's own loop decides when to call them, so middleware and
    approvals apply as they do to any tool. A workflow cancelled during a call
    stops: the cancelled Activity reaches the tool as ``asyncio.CancelledError``,
    as with ``activity_as_tool``.

    ``cache_tools=False`` (the default) lists on every run, so a server
    restarted with different tools is picked up; ``True`` lists once per
    provider instance. ``tool_name_prefix``, ``allowed_tools``,
    ``approval_mode`` and ``parse_tool_results`` work as on the framework's
    ``MCPTool``: the agent sees ``{prefix}_{name}``, ``allowed_tools`` matches
    the server's names, ``approval_mode="always_require"`` makes every tool
    ask for approval, and ``parse_tool_results`` replaces the default mapping
    of a ``CallToolResult`` to content.

    The activity options apply to every MCP Activity of this provider;
    ``summary`` defaults to the server name. Without ``start_to_close_timeout``
    or ``schedule_to_close_timeout`` the ``temporalio-mcp`` default of one
    minute applies. The provider sets no ``heartbeat_timeout``:
    ``temporalio-mcp`` 0.3 does not heartbeat its Activities, and its next
    release gives them a 10-second default heartbeat timeout with heartbeats
    at half that.
    """

    def __init__(
        self,
        server: str,
        *,
        cache_tools: bool = False,
        tool_name_prefix: str | None = None,
        allowed_tools: Collection[str] | None = None,
        approval_mode: ApprovalMode | None = None,
        parse_tool_results: Callable[[CallToolResult], str | list[Content]]
        | None = None,
        task_queue: str | None = None,
        schedule_to_close_timeout: timedelta | None = None,
        schedule_to_start_timeout: timedelta | None = None,
        start_to_close_timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL,
        versioning_intent: VersioningIntent | None = None,
        summary: str | None = None,
        priority: Priority = Priority.default,
    ) -> None:
        """Configure the server name, tool naming and filtering, and activity options.

        Raises:
            RuntimeError: If the ``mcp`` extra is not installed.
        """
        try:
            from temporalio.mcp import TemporalMCPClient
        except ImportError as err:
            raise RuntimeError(
                "TemporalMCPToolProvider requires the 'mcp' extra and MCP Python SDK v2; "
                "install temporalio-microsoft-agent-framework[mcp]"
            ) from err
        # Two providers of one server need different prefixes anyway, or their
        # tool names collide; the prefix keeps their provider state apart too.
        source_id = f"temporal_mcp:{server}"
        if tool_name_prefix:
            source_id += f":{tool_name_prefix}"
        super().__init__(source_id)
        self._server = server
        self._cache_tools = cache_tools
        self._prefix = tool_name_prefix
        self._allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self._approval_mode: ApprovalMode | None = approval_mode
        self._parse_tool_results = parse_tool_results
        self._tools: list[FunctionTool] | None = None
        self._client: TemporalMCPClient = TemporalMCPClient(
            server,
            activity_config=ActivityConfig(
                task_queue=task_queue,
                schedule_to_close_timeout=schedule_to_close_timeout,
                schedule_to_start_timeout=schedule_to_start_timeout,
                start_to_close_timeout=start_to_close_timeout,
                retry_policy=retry_policy,
                cancellation_type=cancellation_type,
                versioning_intent=versioning_intent,
                summary=summary if summary is not None else server,
                priority=priority,
            ),
            # This provider decides when to list; the durable client never caches.
            cache_tools_list=False,
        )

    @property
    def server(self) -> str:
        """The MCP server name registered on the worker."""
        return self._server

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """List the server's tools and add them to this run.

        Tools passed to ``agent.run(tools=...)`` are merged after context
        providers run, so a collision with one of those raises the framework's
        own ``ValueError`` instead and fails the workflow task.

        Raises:
            MicrosoftAgentFrameworkWorkflowError: If a listed tool has the name of a
                tool the agent already has, from its own ``tools`` or from
                another provider, or the server lists one name twice.
        """
        tools = await self._load_tools()
        self._check_names(agent, context, tools)
        context.extend_tools(self.source_id, tools)

    async def _load_tools(self) -> list[FunctionTool]:
        if self._cache_tools and self._tools is not None:
            return self._tools
        listed = await self._client.list_tools()
        tools = [
            self._function_tool(tool)
            for tool in listed.tools
            if self._allowed_tools is None or tool.name in self._allowed_tools
        ]
        if self._cache_tools:
            self._tools = tools
        return tools

    def _function_tool(self, tool: Tool) -> FunctionTool:
        server_tool_name = tool.name
        name = (
            f"{self._prefix}_{server_tool_name}" if self._prefix else server_tool_name
        )

        async def call(**arguments: Any) -> str | list[Content]:
            with _propagate_workflow_cancellation():
                result = await self._client.call_tool(server_tool_name, arguments)
            return self._parse(result)

        return FunctionTool(
            name=name,
            description=tool.description or "",
            func=call,
            input_model=dict(tool.input_schema),
            approval_mode=self._approval_mode,
        )

    def _parse(self, result: CallToolResult) -> str | list[Content]:
        parsed = (
            self._parse_tool_results(result)
            if self._parse_tool_results is not None
            else _contents(result)
        )
        if result.is_error:
            # The framework's MCPTool raises the same way, so the loop records
            # a tool error for the model instead of a successful result.
            text = (
                "\n".join(c.text for c in parsed if c.type == "text" and c.text)
                if isinstance(parsed, list)
                else str(parsed)
            )
            raise ToolExecutionException(
                f"MCP tool returned an error: {text or parsed!s}"
            )
        return parsed

    def _check_names(
        self, agent: Any, context: SessionContext, tools: list[FunctionTool]
    ) -> None:
        taken: dict[str, str] = {}
        default_options = getattr(agent, "default_options", None)
        agent_tools = (
            default_options.get("tools") if isinstance(default_options, dict) else None
        )
        for name in _tool_names(agent_tools or []):
            taken[name] = "the agent's own tools"
        for name in _tool_names(context.tools):
            taken.setdefault(name, "another context provider")
        for tool in tools:
            source = taken.get(tool.name)
            if source is not None:
                raise MicrosoftAgentFrameworkWorkflowError(
                    f"MCP server {self._server!r} lists a tool named {tool.name!r}, which the "
                    f"agent already has from {source}. Set tool_name_prefix or allowed_tools "
                    "on TemporalMCPToolProvider."
                )
            taken[tool.name] = f"MCP server {self._server!r}"


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if isinstance(name, str):
            names.append(name)
    return names


def _contents(result: CallToolResult) -> list[Content]:
    """Map a ``CallToolResult`` to content the way the framework's ``MCPTool`` does.

    Structured content wins when the server returns it; otherwise text, image,
    audio, resource links and embedded resources map to their content types.
    """
    from mcp import types

    if result.structured_content is not None:
        return [Content.from_text(json.dumps(result.structured_content, default=str))]
    contents: list[Content] = []
    for item in result.content:
        if isinstance(item, types.TextContent):
            contents.append(Content.from_text(item.text))
        elif isinstance(item, (types.ImageContent, types.AudioContent)):
            contents.append(
                Content.from_uri(
                    uri=f"data:{item.mime_type};base64,{item.data}",
                    media_type=item.mime_type,
                )
            )
        elif isinstance(item, types.ResourceLink):
            contents.append(
                Content.from_uri(uri=str(item.uri), media_type=item.mime_type)
            )
        elif isinstance(item, types.EmbeddedResource):
            resource = item.resource
            if isinstance(resource, types.TextResourceContents):
                contents.append(Content.from_text(resource.text))
            else:
                mime = resource.mime_type or "application/octet-stream"
                blob = resource.blob
                if not blob.startswith("data:"):
                    blob = f"data:{mime};base64,{blob}"
                contents.append(Content.from_uri(uri=blob, media_type=mime))
        else:
            contents.append(Content.from_text(str(item)))
    if not contents:
        contents.append(Content.from_text("null"))
    return contents
