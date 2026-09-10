# pyright: reportUnusedClass=false

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, TypeVar

from mcp import MCPError
from mcp.types import (
    HEADER_MISMATCH,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    MISSING_REQUIRED_CLIENT_CAPABILITY,
    PARSE_ERROR,
    UNSUPPORTED_PROTOCOL_VERSION,
    URL_ELICITATION_REQUIRED,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
)
from pydantic import BaseModel, ValidationError

from temporalio import activity
from temporalio.contrib.mcp._activity import (
    _activity_name,
    _CallToolRequest,
    _GetPromptRequest,
    _MCPRequest,
    _ReadResourceRequest,
)
from temporalio.contrib.mcp._backend import _MCPBackend, _MCPBackendFactory
from temporalio.contrib.mcp._pool import _MCPConnectionPool
from temporalio.exceptions import ApplicationError

_Result = TypeVar("_Result")
_ListResult = TypeVar("_ListResult", bound=BaseModel)

logger = logging.getLogger(__name__)

# Upper bound on how long worker shutdown waits for MCP transports to close.
_CLOSE_TIMEOUT_SECONDS = 10.0
# JSON-RPC errors that a retry of the same request cannot fix. INTERNAL_ERROR,
# CONNECTION_CLOSED, REQUEST_TIMEOUT and unknown codes stay retryable.
_NON_RETRYABLE_PROTOCOL_ERRORS = {
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    HEADER_MISMATCH,
    MISSING_REQUIRED_CLIENT_CAPABILITY,
    UNSUPPORTED_PROTOCOL_VERSION,
    URL_ELICITATION_REQUIRED,
}


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _normalize_list_result(
    value: BaseModel | list[Any], result_type: type[_ListResult], field: str
) -> _ListResult:
    """Wrap list-only compatibility backends in an MCP result envelope."""
    if isinstance(value, result_type):
        return value
    return result_type.model_validate({field: value})


class _MCPActivities:
    """Build framework-neutral MCP operation Activities for named backends.

    ``temporalio-openai-agents`` builds on this class and on ``_backend``; a
    change to either is a coordinated release across both packages.
    """

    def __init__(
        self,
        factories: dict[str, _MCPBackendFactory],
        idle_timeout: timedelta | None = timedelta(minutes=5),
    ) -> None:
        self._factories = dict(factories)
        self._pool = _MCPConnectionPool(self._factories, idle_timeout)
        self._run_contexts: dict[asyncio.AbstractEventLoop, int] = {}
        self._abandoned_closes: set[asyncio.Task[None]] = set()
        self.activities = self._build_activities()

    async def _run(
        self,
        server: str,
        request: _MCPRequest,
        operation: Callable[[_MCPBackend], Awaitable[_Result]],
    ) -> _Result:
        try:
            async with self._pool.backend(
                server,
                factory_argument=request.factory_argument,
            ) as backend:
                return await operation(backend)
        except MCPError as err:
            details = (err.code,) if err.data is None else (err.code, err.data)
            raise ApplicationError(
                err.message,
                *details,
                type="MCPProtocolError",
                non_retryable=err.code in _NON_RETRYABLE_PROTOCOL_ERRORS,
            ) from err
        except ValidationError as err:
            raise ApplicationError(
                "MCP server returned an invalid response",
                err.errors(include_url=False, include_input=False),
                type="MCPProtocolError",
                non_retryable=True,
            ) from err

    def _build_activities(self) -> Sequence[Callable[..., Any]]:
        activities: list[Callable[..., Any]] = []
        for server in self._factories:

            @activity.defn(name=_activity_name(server, "list-tools"))
            async def list_tools(
                request: dict[str, Any], server: str = server
            ) -> dict[str, Any]:
                parsed = _MCPRequest(**request)
                result = await self._run(
                    server,
                    parsed,
                    lambda backend: backend.list_tools(),
                )
                return _dump(_normalize_list_result(result, ListToolsResult, "tools"))

            @activity.defn(name=_activity_name(server, "call-tool"))
            async def call_tool(
                request: dict[str, Any], server: str = server
            ) -> dict[str, Any]:
                parsed = _CallToolRequest(**request)
                result = await self._run(
                    server,
                    parsed,
                    lambda backend: backend.call_tool(
                        parsed.name,
                        parsed.arguments,
                        parsed.meta,
                    ),
                )
                return _dump(result)

            @activity.defn(name=_activity_name(server, "list-prompts"))
            async def list_prompts(
                request: dict[str, Any], server: str = server
            ) -> dict[str, Any]:
                parsed = _MCPRequest(**request)
                result = await self._run(
                    server,
                    parsed,
                    lambda backend: backend.list_prompts(),
                )
                return _dump(
                    _normalize_list_result(result, ListPromptsResult, "prompts")
                )

            @activity.defn(name=_activity_name(server, "get-prompt"))
            async def get_prompt(
                request: dict[str, Any], server: str = server
            ) -> dict[str, Any]:
                parsed = _GetPromptRequest(**request)
                result = await self._run(
                    server,
                    parsed,
                    lambda backend: backend.get_prompt(
                        parsed.name,
                        parsed.arguments,
                    ),
                )
                return _dump(result)

            @activity.defn(name=_activity_name(server, "list-resources"))
            async def list_resources(
                request: dict[str, Any], server: str = server
            ) -> dict[str, Any]:
                parsed = _MCPRequest(**request)
                result = await self._run(
                    server,
                    parsed,
                    lambda backend: backend.list_resources(),
                )
                return _dump(
                    _normalize_list_result(result, ListResourcesResult, "resources")
                )

            @activity.defn(name=_activity_name(server, "list-resource-templates"))
            async def list_resource_templates(
                request: dict[str, Any], server: str = server
            ) -> dict[str, Any]:
                parsed = _MCPRequest(**request)
                result = await self._run(
                    server,
                    parsed,
                    lambda backend: backend.list_resource_templates(),
                )
                return _dump(
                    _normalize_list_result(
                        result,
                        ListResourceTemplatesResult,
                        "resource_templates",
                    )
                )

            @activity.defn(name=_activity_name(server, "read-resource"))
            async def read_resource(
                request: dict[str, Any], server: str = server
            ) -> dict[str, Any]:
                parsed = _ReadResourceRequest(**request)
                result = await self._run(
                    server,
                    parsed,
                    lambda backend: backend.read_resource(parsed.uri),
                )
                return _dump(result)

            activities.extend(
                (
                    list_tools,
                    call_tool,
                    list_prompts,
                    get_prompt,
                    list_resources,
                    list_resource_templates,
                    read_resource,
                )
            )
        return activities

    @asynccontextmanager
    async def run_context(self) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        self._run_contexts[loop] = self._run_contexts.get(loop, 0) + 1
        body_completed = False
        try:
            yield
            body_completed = True
        finally:
            remaining = self._run_contexts[loop] - 1
            if remaining:
                self._run_contexts[loop] = remaining
            else:
                self._run_contexts.pop(loop)
                close_task = asyncio.create_task(self._close_if_unused(loop))
                try:
                    await self._finish_close(close_task)
                except asyncio.CancelledError:
                    # Worker context exit cancels its wrapper after the inner
                    # run completes. Finish MCP cleanup without swallowing
                    # cancellation received while the worker was still running.
                    if not body_completed:
                        raise
                    await self._finish_close(close_task)

    async def _close_if_unused(self, loop: asyncio.AbstractEventLoop) -> None:
        """Close this loop's connections unless another Worker has entered."""
        if self._run_contexts.get(loop, 0) == 0:
            # close() detaches the loop's pool generation before its first
            # suspension, so a later entrant cannot inherit closing records.
            await self._pool.close()

    async def _finish_close(self, close_task: asyncio.Task[None]) -> None:
        """Wait a bounded time for the connection pool to close."""
        try:
            await asyncio.wait_for(asyncio.shield(close_task), _CLOSE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            # An unresponsive MCP server must not hang worker shutdown
            # indefinitely, especially now that there is no cancellation left to
            # break out with. Leave the close running, but strongly referenced
            # so it is not garbage collected part way through.
            self._abandoned_closes.add(close_task)
            close_task.add_done_callback(self._abandoned_closes.discard)
            logger.warning(
                "Timed out after %s seconds closing MCP connections; "
                "an MCP server may not have shut down cleanly.",
                _CLOSE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            self._abandoned_closes.add(close_task)
            close_task.add_done_callback(self._abandoned_closes.discard)
            raise
