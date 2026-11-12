# pyright: reportUnusedClass=false

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, TypeVar, cast

from mcp import MCPError
from mcp.client import InputRequiredRoundsExceededError
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
from temporalio.exceptions import ApplicationError
from temporalio.mcp._activity import (
    _activity_name,
    _CallToolRequest,
    _GetPromptRequest,
    _MCPRequest,
    _ReadResourceRequest,
)
from temporalio.mcp._backend import _MCPBackend, _MCPBackendFactory
from temporalio.mcp._pool import _MCPConnectionPool

_Result = TypeVar("_Result")
_ListResult = TypeVar("_ListResult", bound=BaseModel)

logger = logging.getLogger(__name__)

# Upper bound on how long worker shutdown waits for MCP transports to close.
_CLOSE_TIMEOUT_SECONDS = 10.0
# JSON-RPC errors that a retry of the same request cannot fix. INTERNAL_ERROR,
# CONNECTION_CLOSED, REQUEST_TIMEOUT, unknown codes and a lost session (see
# _session_lost) stay retryable.
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


# A server answers HTTP 404 for a session it no longer knows, for example after a
# restart, and the client must start a new session. MCP reports it only as
# INVALID_REQUEST with no data and one of these messages: mcp's streamable HTTP
# client uses "Session terminated" when the 404 body is not a JSON-RPC error,
# and the MCP Python SDK server's body says "Session not found".
_SESSION_LOST_MESSAGES = frozenset({"Session terminated", "Session not found"})


def _session_lost(err: MCPError) -> bool:
    """Whether the server no longer knows the connection's MCP session."""
    return (
        err.code == INVALID_REQUEST
        and err.data is None
        and err.message in _SESSION_LOST_MESSAGES
    )


# The MCP client checks a tool's result against the tool's output schema and
# raises RuntimeError with one of these messages (mcp/client/session.py). It
# also raises InputRequiredRoundsExceededError when a call runs past its round
# limit. A retry of the same request can't fix either.
_REJECTED_RESULT = re.compile(
    r"Tool .+ has an output schema but did not return structured content$"
    r"|Invalid structured content returned by tool "
    r"|Invalid schema for tool ",
    re.DOTALL,
)


def _rejected_result(err: RuntimeError) -> bool:
    """Whether the MCP client rejected a result that the server returned."""
    return isinstance(err, InputRequiredRoundsExceededError) or bool(
        _REJECTED_RESULT.match(str(err))
    )


def _rejected(err: RuntimeError) -> ApplicationError:
    return ApplicationError(str(err), type="MCPProtocolError", non_retryable=True)


def _lone_error(
    group: BaseExceptionGroup[Any],
) -> MCPError | ValidationError | RuntimeError | ApplicationError | None:
    """The error we classify that a nest of ExceptionGroups carries alone, if it does."""
    while len(group.exceptions) == 1:
        (inner,) = group.exceptions
        if isinstance(inner, (MCPError, ValidationError, ApplicationError)):
            return inner
        if isinstance(inner, RuntimeError) and _rejected_result(inner):
            return inner
        if not isinstance(inner, BaseExceptionGroup):
            return None
        group = cast(BaseExceptionGroup[Any], inner)
    return None


def _protocol_error(err: MCPError, has_session: bool) -> ApplicationError:
    details = (err.code,) if err.data is None else (err.code, err.data)
    session_lost = has_session and _session_lost(err)
    return ApplicationError(
        err.message,
        *details,
        type="MCPProtocolError",
        non_retryable=err.code in _NON_RETRYABLE_PROTOCOL_ERRORS and not session_lost,
    )


def _invalid_response(err: ValidationError) -> ApplicationError:
    # A validator's ctx can hold the exception it raised, which the payload
    # converter cannot serialize; pydantic's JSON form renders it as a string.
    return ApplicationError(
        "MCP server returned an invalid response",
        json.loads(err.json(include_url=False, include_input=False)),
        type="MCPProtocolError",
        non_retryable=True,
    )


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


@asynccontextmanager
async def _heartbeating() -> AsyncGenerator[None, None]:
    """Heartbeat at half the Activity's heartbeat timeout while the block runs.

    A cancel reaches a running Activity only in a heartbeat's response. The
    beats come from a timer, not from the server, so a slow request isn't
    mistaken for a lost worker.
    """
    timeout = activity.info().heartbeat_timeout if activity.in_activity() else None
    if not timeout:
        yield
        return

    async def beat(interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            activity.heartbeat()

    task = asyncio.create_task(beat(timeout.total_seconds() / 2))
    try:
        yield
    finally:
        # Not awaited: awaiting it here could swallow a cancel of the Activity.
        task.cancel()


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
        has_session = False
        try:
            async with (
                _heartbeating(),
                self._pool.backend(
                    server,
                    factory_argument=request.factory_argument,
                ) as backend,
            ):
                # Only older-protocol connections have MCP sessions. Each
                # Activity attempt opens its own, so a retry starts a new one.
                has_session = not backend.cacheable
                return await operation(backend)
        except MCPError as err:
            raise _protocol_error(err, has_session) from err
        except BaseExceptionGroup as group:
            # ``Client.__aexit__`` re-raises an error from the ``async with`` body
            # wrapped in task-group ExceptionGroups, which a factory argument or a
            # failed connect lets through.
            inner = _lone_error(group)
            if isinstance(inner, ApplicationError):
                # Already classified, such as a repeated pagination cursor.
                raise inner
            if isinstance(inner, MCPError):
                raise _protocol_error(inner, has_session) from inner
            if isinstance(inner, ValidationError):
                raise _invalid_response(inner) from inner
            if isinstance(inner, RuntimeError):
                raise _rejected(inner) from inner
            raise
        except ValidationError as err:
            raise _invalid_response(err) from err
        except RuntimeError as err:
            if _rejected_result(err):
                raise _rejected(err) from err
            raise

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
    async def run_context(self) -> AsyncGenerator[None, None]:
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
