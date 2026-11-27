"""MCP servers whose tools run as Temporal Activities."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from locus.core.media import encode_image
from locus.tools.decorator import Tool
from locus.tools.output import ToolOutput
from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityConfig,
    VersioningIntent,
)

if TYPE_CHECKING:
    from mcp.types import CallToolResult
    from mcp.types import Tool as MCPTool

    from temporalio.mcp import TemporalMCPClient as DurableMCPClient

_INSTALL_HINT = (
    "requires the 'mcp' extra and MCP Python SDK v2; install temporalio-locus[mcp]"
)


class TemporalMCPClient:
    """Workflow-side handle to an MCP server registered on the Worker.

    The ``mcp.Client`` factory lives on the Worker, in
    ``LocusPlugin(mcp_clients={"orders": lambda: Client(...)})``. This handle
    carries the server's name, which selects that factory, and the Activity
    options of its calls. Listing the server's tools runs as the
    ``temporalio.mcp.{server}.list-tools`` Activity and every tool call as a
    ``temporalio.mcp.{server}.call-tool`` Activity, from ``temporalio-mcp``.

    Pass it to a Locus agent as Locus's own ``MCPClient`` is passed, or load
    the tools yourself::

        orders = TemporalMCPClient("orders", start_to_close_timeout=timedelta(seconds=30))
        agent = Agent(model=..., mcp_servers=[orders])
        # or
        agent = Agent(model=..., tools=await orders.load_tools())

    The tools are listed once per handle and reused; pass
    ``cache_tools=False`` to list them on every ``load_tools`` call. Each
    listing records the server's full tool list in Event History.

    ``prefix`` names the tools ``{prefix}_{name}`` for the model.
    ``allowed_tools`` keeps only the named tools, by the server's names.

    The Activity options apply to both Activities. With neither
    ``start_to_close_timeout`` nor ``schedule_to_close_timeout``, a one-minute
    ``start_to_close_timeout`` applies. A tool call's Activity ``summary`` is
    the tool's name unless ``summary`` is given.

    A tool the server reports as failed (``isError``) is handed to the model
    as a tool error. So is a tool call whose Activity fails after its
    retries, as Locus does for any tool that raises.
    """

    def __init__(
        self,
        server: str,
        *,
        prefix: str | None = None,
        allowed_tools: Sequence[str] | None = None,
        cache_tools: bool = True,
        task_queue: str | None = None,
        schedule_to_close_timeout: timedelta | None = None,
        schedule_to_start_timeout: timedelta | None = None,
        start_to_close_timeout: timedelta | None = None,
        heartbeat_timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL,
        versioning_intent: VersioningIntent | None = None,
        summary: str | None = None,
        priority: Priority = Priority.default,
    ) -> None:
        """Configure the server name, tool naming and filtering, and Activity options."""
        self._server = server
        self._prefix = prefix
        self._allowed = None if allowed_tools is None else frozenset(allowed_tools)
        self._summary = summary
        self._config = ActivityConfig(
            task_queue=task_queue,
            schedule_to_close_timeout=schedule_to_close_timeout,
            schedule_to_start_timeout=schedule_to_start_timeout,
            start_to_close_timeout=start_to_close_timeout,
            retry_policy=retry_policy,
            cancellation_type=cancellation_type,
            versioning_intent=versioning_intent,
            priority=priority,
        )
        if heartbeat_timeout is not None:
            # Left out, temporalio-mcp applies its own default.
            self._config["heartbeat_timeout"] = heartbeat_timeout
        self._lister = self._durable(summary, cache_tools_list=cache_tools)

    @property
    def server(self) -> str:
        """MCP server name registered on the Worker."""
        return self._server

    @property
    def label(self) -> str:
        """The name Locus logs this server under."""
        return f"mcp:{self._server}"

    def _durable(
        self, summary: str | None, *, cache_tools_list: bool
    ) -> DurableMCPClient:
        try:
            from temporalio.mcp import TemporalMCPClient as DurableMCPClient
        except ImportError as err:
            raise RuntimeError(f"TemporalMCPClient {_INSTALL_HINT}") from err
        config = ActivityConfig(**self._config)
        if summary is not None:
            config["summary"] = summary
        return DurableMCPClient(
            self._server, activity_config=config, cache_tools_list=cache_tools_list
        )

    async def load_tools(self, metadata: Mapping[str, Any] | None = None) -> list[Tool]:
        """List the server's tools in an Activity and return them as Locus tools.

        ``metadata`` is the run's metadata, which Locus passes when it attaches
        ``mcp_servers``. It is not sent to the Worker.
        """
        del metadata
        listed = await self._lister.list_tools()
        return [
            self._tool(tool)
            for tool in listed.tools
            if self._allowed is None or tool.name in self._allowed
        ]

    def _tool(self, tool: MCPTool) -> Tool:
        client = self._durable(self._summary or tool.name, cache_tools_list=False)
        name = tool.name

        async def call(**arguments: Any) -> str:
            return _tool_output(await client.call_tool(name, arguments))

        return _locus_tool(
            name=f"{self._prefix}_{name}" if self._prefix else name,
            description=tool.description or "",
            parameters=dict(tool.input_schema),
            fn=call,
            output_schema=tool.output_schema,
        )


def _locus_tool(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any],
    fn: Callable[..., Any],
    output_schema: dict[str, Any] | None,
) -> Tool:
    # The server's input schema is the tool's parameters as they are, so the
    # model sees the server's argument names.
    fields: dict[str, Any] = {}
    if "output_schema" in Tool.model_fields and isinstance(output_schema, dict):
        fields["output_schema"] = output_schema
    return Tool(
        name=name, description=description, parameters=parameters, fn=fn, **fields
    )


def _tool_output(result: CallToolResult) -> ToolOutput:
    """An MCP tool result as Locus hands it to the model and the application."""
    parts: list[str] = []
    blocks: list[dict[str, Any]] = []
    for item in result.content:
        if item.type != "text":
            blocks.append(
                item.model_dump(mode="json", by_alias=True, exclude_none=True)
            )
        parts.append(_block_text(item))
    structured = result.structured_content
    structured = structured if isinstance(structured, dict) else None
    if parts:
        text = "\n".join(parts)
    elif structured is not None:
        # Not every server mirrors its structured content as text, and a model
        # shown an empty result reads it as "nothing found".
        text = json.dumps(structured, default=str)
    else:
        text = ""
    return ToolOutput(
        text,
        structured_content=structured,
        content_blocks=blocks or None,
        is_error=result.is_error is True,
    )


def _block_text(item: Any) -> str:
    """What the model reads for one content block, as Locus's MCP client renders it."""
    if item.type == "text":
        return str(item.text)
    if item.type == "image":
        try:
            return encode_image(
                base64.b64decode(item.data, validate=True), item.mime_type
            )
        except (binascii.Error, ValueError):
            return f"[image: {item.mime_type}]"
    if item.type == "audio":
        return f"[audio: {item.mime_type}]"
    if item.type == "resource":
        text = getattr(item.resource, "text", None)
        if isinstance(text, str):
            return text
        mime = getattr(item.resource, "mime_type", None) or "binary"
        return f"[resource: {item.resource.uri} ({mime})]"
    if item.type == "resource_link":
        return f"[resource link: {item.name} {item.uri}]"
    return f"[{item.type}]"
