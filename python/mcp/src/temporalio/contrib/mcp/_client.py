# pyright: reportUnusedClass=false, reportUnusedFunction=false

from collections.abc import Callable
from types import TracebackType
from typing import Any, TypeVar, cast

from mcp import Client
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    ReadResourceResult,
    RequestParamsMeta,
)
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from temporalio.contrib.mcp._backend import (
    _NOT_SUPPLIED,
    _FactoryInvoker,
    _MCPBackendFactory,
)
from temporalio.exceptions import ApplicationError

_ListResult = TypeVar(
    "_ListResult",
    ListToolsResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
)


class _MCPClientBackend:
    """Adapt an MCP Python SDK v2 client to the shared Activity backend."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def __aenter__(self) -> "_MCPClientBackend":
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._client.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def cacheable(self) -> bool:
        return self._client.protocol_version in MODERN_PROTOCOL_VERSIONS

    async def _list_all(
        self, method: str, field: str, result_type: type[_ListResult]
    ) -> _ListResult:
        values: list[Any] = []
        cursor: str | None = None
        seen_cursors: set[str | None] = set()
        first_result: _ListResult | None = None
        meta: dict[str, Any] | None = None
        ttl_ms: int | None = None
        cache_scope = "public"
        while True:
            # Workflow history is the durable cache for every operation, so the
            # client-side response cache (server ``ttlMs`` hints) is bypassed:
            # each Activity must observe the server, not a worker-local copy.
            result = result_type.model_validate(
                await getattr(self._client, method)(cursor=cursor, cache_mode="bypass")
            )
            if first_result is None:
                first_result = result
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
                # There is no protocol envelope for an aggregate of multiple
                # pages. Merge metadata in page order and use conservative cache
                # hints so the synthetic result is never fresher or more widely
                # shareable than any page it contains.
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

    async def list_tools(self) -> ListToolsResult:
        return await self._list_all("list_tools", "tools", ListToolsResult)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        meta: RequestParamsMeta | None,
    ) -> CallToolResult:
        return await self._client.call_tool(name, arguments, meta=meta)

    async def list_prompts(self) -> ListPromptsResult:
        return await self._list_all("list_prompts", "prompts", ListPromptsResult)

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None
    ) -> GetPromptResult:
        return await self._client.get_prompt(name, arguments)

    async def list_resources(self) -> ListResourcesResult:
        return await self._list_all("list_resources", "resources", ListResourcesResult)

    async def list_resource_templates(self) -> ListResourceTemplatesResult:
        return await self._list_all(
            "list_resource_templates",
            "resource_templates",
            ListResourceTemplatesResult,
        )

    async def read_resource(self, uri: str) -> ReadResourceResult:
        return await self._client.read_resource(uri, cache_mode="bypass")


_MCPClientFactory = Callable[[], Client] | Callable[[Any], Client]


def _mcp_client_backend_factory(
    name: str, factory: _MCPClientFactory
) -> _MCPBackendFactory:
    """Adapt a public MCP client factory to the shared Activity backend."""
    invoke = _FactoryInvoker(name, factory)

    def create(argument: Any = _NOT_SUPPLIED) -> _MCPClientBackend:
        return _MCPClientBackend(cast(Client, invoke(argument)))

    return create
