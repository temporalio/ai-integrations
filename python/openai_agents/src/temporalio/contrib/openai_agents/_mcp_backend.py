# pyright: reportUnusedClass=false, reportUnusedFunction=false

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Any, cast

from agents import UserError
from agents.mcp import MCPServer
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ReadResourceResult,
    RequestParamsMeta,
    Tool,
)
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from temporalio.contrib.mcp._backend import (
    _NOT_SUPPLIED,
    _FactoryInvoker,
    _MCPBackendFactory,
)
from temporalio.exceptions import ApplicationError

_MCPServerFactory = Callable[[], MCPServer] | Callable[[Any], MCPServer]


class _OpenAIMCPServerBackend:
    """Adapt an OpenAI Agents MCPServer to durable MCP Activities."""

    def __init__(self, server: MCPServer) -> None:
        self._server = server

    async def __aenter__(self) -> "_OpenAIMCPServerBackend":
        await self._server.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._server.cleanup()

    @property
    def cacheable(self) -> bool:
        session = getattr(self._server, "session", None)
        return getattr(session, "protocol_version", None) in MODERN_PROTOCOL_VERSIONS

    async def list_tools(self) -> list[Tool]:
        try:
            return await self._server.list_tools()
        except UserError as err:
            _raise_repeated_cursor_error(err)
            raise

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        meta: RequestParamsMeta | None,
    ) -> CallToolResult:
        return await self._server.call_tool(
            name, arguments, cast(dict[str, Any] | None, meta)
        )

    async def list_prompts(self) -> ListPromptsResult:
        try:
            return await self._server.list_prompts()
        except UserError as err:
            _raise_repeated_cursor_error(err)
            raise

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None
    ) -> GetPromptResult:
        return await self._server.get_prompt(name, arguments)

    async def list_resources(self) -> ListResourcesResult:
        return await self._list_all("list_resources", "resources", ListResourcesResult)

    async def list_resource_templates(self) -> ListResourceTemplatesResult:
        return await self._list_all(
            "list_resource_templates",
            "resource_templates",
            ListResourceTemplatesResult,
        )

    async def _list_all(self, method: str, field: str, result_type: type[Any]) -> Any:
        values: list[Any] = []
        cursor: str | None = None
        seen_cursors: set[str | None] = set()
        first_result: Any | None = None
        meta: dict[str, Any] | None = None
        ttl_ms: int | None = None
        cache_scope = "public"
        while True:
            result = await getattr(self._server, method)(cursor)
            if first_result is None:
                first_result = result_type.model_validate(result)
            values.extend(getattr(result, field))
            if result.meta is not None:
                if meta is None:
                    meta = {}
                meta.update(result.meta)
            ttl_ms = result.ttl_ms if ttl_ms is None else min(ttl_ms, result.ttl_ms)
            if result.cache_scope == "private":
                cache_scope = "private"
            seen_cursors.add(cursor)
            next_cursor = result.next_cursor
            if next_cursor is None:
                assert first_result is not None
                return first_result.model_copy(
                    update={
                        field: values,
                        "next_cursor": None,
                        "meta": meta,
                        "ttl_ms": ttl_ms,
                        "cache_scope": cache_scope,
                    }
                )
            if next_cursor in seen_cursors:
                raise ApplicationError(
                    "MCP server returned a repeated pagination cursor",
                    type="MCPProtocolError",
                    non_retryable=True,
                )
            cursor = next_cursor

    async def read_resource(self, uri: str) -> ReadResourceResult:
        return await self._server.read_resource(uri)


def _raise_repeated_cursor_error(err: UserError) -> None:
    """Convert the Agents SDK's repeated-cursor error to a terminal failure."""
    if "returned a repeated cursor while listing " in str(err):
        raise ApplicationError(
            "MCP server returned a repeated pagination cursor",
            type="MCPProtocolError",
            non_retryable=True,
        ) from err


def _reject_dynamic_tool_filter(name: str, server: MCPServer) -> None:
    """Fail fast on a worker-side filter that can never be applied.

    A callable ``tool_filter`` needs the run context and agent, which exist only
    in the workflow, so the Agents SDK rejects one on the worker side and the
    list-tools Activity would otherwise retry forever. A static filter needs
    neither and is applied here as usual.
    """
    if callable(getattr(server, "tool_filter", None)):
        raise ApplicationError(
            f"MCP server {name!r} sets a callable tool_filter on its worker-side "
            "MCPServer, which cannot be applied there because the run context "
            "and agent it receives exist only in the workflow; pass the filter "
            "to temporal_mcp_server() instead",
            non_retryable=True,
        )


def _mcp_server_backend_factory(
    name: str, factory: _MCPServerFactory
) -> _MCPBackendFactory:
    invoke = _FactoryInvoker(name, factory)

    def create(argument: Any = _NOT_SUPPLIED) -> _OpenAIMCPServerBackend:
        server = cast(MCPServer, invoke(argument))
        _reject_dynamic_tool_filter(name, server)
        return _OpenAIMCPServerBackend(server)

    return create
