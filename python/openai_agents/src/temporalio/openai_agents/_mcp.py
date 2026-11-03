import asyncio
import dataclasses
import functools
import inspect
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta
from types import TracebackType
from typing import Any, cast

from agents import AgentBase, RunContextWrapper
from agents.mcp import MCPServer
from agents.mcp.server import RequireApprovalSetting
from mcp import GetPromptResult, ListPromptsResult  # type:ignore
from mcp import Tool as MCPTool  # type:ignore
from mcp.types import CallToolResult  # type:ignore

from temporalio import activity, workflow
from temporalio.api.enums.v1.workflow_pb2 import (
    TIMEOUT_TYPE_HEARTBEAT,
    TIMEOUT_TYPE_SCHEDULE_TO_START,
)
from temporalio.converter import ActivitySerializationContext
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    is_cancelled_exception,
)
from temporalio.openai_agents._mcp_server_checks import (
    _has_dynamic_tool_filter,
    _has_worker_side_require_approval,
    _has_worker_side_tool_guardrails,
    _has_worker_side_tool_meta_resolver,
    _reject_dynamic_tool_filter,
    _reject_worker_side_require_approval,
    _reject_worker_side_tool_guardrails,
    _reject_worker_side_tool_meta_resolver,
)
from temporalio.service import RPCError, RPCStatusCode
from temporalio.worker import PollerBehaviorSimpleMaximum, Worker
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityConfig,
    ActivityHandle,
)

# What the deprecated routes' tool_filter error tells the user to do instead.
_LEGACY_TOOL_FILTER_REMEDY = (
    "use a static filter (create_static_tool_filter) on the factory's server, "
    "or register the factory with OpenAIAgentsPlugin(mcp_servers=...) and pass "
    "the filter to temporal_mcp_server(tool_filter=...)"
)


def _legacy_tool_meta_resolver_remedy(route: str) -> str:
    """What the deprecated routes' tool_meta_resolver error tells the user to do."""
    return (
        f"set tool_meta_resolver on the server {route}() gives the workflow, or "
        "register the factory with OpenAIAgentsPlugin(mcp_servers=...) and pass "
        "it to temporal_mcp_server(tool_meta_resolver=...)"
    )


def _legacy_tool_guardrails_remedy(route: str) -> str:
    """What the deprecated routes' tool guardrails error tells the user to do."""
    return (
        f"set them on the server {route}() gives the workflow, or register the "
        "factory with OpenAIAgentsPlugin(mcp_servers=...) and pass them to "
        "temporal_mcp_server()"
    )


@dataclasses.dataclass
class _StatelessListToolsArguments:
    factory_argument: Any | None


@dataclasses.dataclass
class _StatelessCallToolsArguments:
    tool_name: str
    arguments: dict[str, Any] | None
    factory_argument: Any | None
    meta: dict[str, Any] | None = None


@dataclasses.dataclass
class _StatelessListPromptsArguments:
    factory_argument: Any | None


@dataclasses.dataclass
class _StatelessGetPromptArguments:
    name: str
    arguments: dict[str, Any] | None
    factory_argument: Any | None


class _StatelessMCPServerReference(MCPServer):  # type:ignore[reportUnusedClass]
    def __init__(
        self,
        server: str,
        config: ActivityConfig | None,
        cache_tools_list: bool,
        factory_argument: Any | None = None,
        require_approval: RequireApprovalSetting = None,
    ):
        self._name = server + "-stateless"
        self._config = config or ActivityConfig(
            start_to_close_timeout=timedelta(minutes=1)
        )
        self._cache_tools_list = cache_tools_list
        self._tools = None
        self._factory_argument = factory_argument
        super().__init__(require_approval=require_approval)

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def list_tools(
        self,
        run_context: RunContextWrapper[Any] | None = None,
        agent: AgentBase | None = None,
    ) -> list[MCPTool]:
        if self._tools:
            return self._tools
        tools = await workflow.execute_activity(
            self.name + "-list-tools",
            _StatelessListToolsArguments(self._factory_argument),
            result_type=list[MCPTool],
            **self._config,
        )
        if self._cache_tools_list:
            self._tools = tools
        return tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        return await workflow.execute_activity(
            self.name + "-call-tool-v2",
            _StatelessCallToolsArguments(
                tool_name, arguments, self._factory_argument, meta
            ),
            result_type=CallToolResult,
            **self._config,
        )

    async def list_prompts(self) -> ListPromptsResult:
        return await workflow.execute_activity(
            self.name + "-list-prompts",
            _StatelessListPromptsArguments(self._factory_argument),
            result_type=ListPromptsResult,
            **self._config,
        )

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        return await workflow.execute_activity(
            self.name + "-get-prompt-v2",
            _StatelessGetPromptArguments(name, arguments, self._factory_argument),
            result_type=GetPromptResult,
            **self._config,
        )


class StatelessMCPServerProvider:
    """A stateless MCP server implementation for Temporal workflows.

    .. deprecated:: 1.0.0
        Use ``OpenAIAgentsPlugin(mcp_servers=...)`` and
        ``workflow.temporal_mcp_server()`` instead.

    This class wraps a function to create MCP servers to make them stateless by executing each MCP operation
    as a separate Temporal activity. Each operation (list_tools, call_tool, etc.) will
    connect to the underlying server, execute the operation, and then clean up the connection.

    This approach will not maintain state across calls. If the desired MCPServer needs persistent state in order to
    function, this cannot be used.
    """

    def __init__(
        self,
        name: str,
        server_factory: (Callable[[], MCPServer] | Callable[[Any | None], MCPServer]),
    ):
        """Initialize the stateless temporal MCP server.

        Args:
            name: The name of the MCP server.
            server_factory: A function which will produce MCPServer instances. It should return a new server each time
                so that state is not shared between workflow runs. It may accept a single positional parameter, which
                receives a ``factory_argument`` from the workflow. A callable ``tool_filter``, a ``require_approval``
                policy that can ask for approval, a ``tool_meta_resolver``, or a tool guardrail on the returned server
                is rejected in every MCP operation; a static filter is applied.
        """
        warnings.warn(
            "StatelessMCPServerProvider is deprecated; use "
            "OpenAIAgentsPlugin(mcp_servers=...) and "
            "workflow.temporal_mcp_server() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._server_factory = server_factory

        # Cache whether the server factory needs to be provided with arguments
        sig = inspect.signature(self._server_factory)
        self._server_accepts_arguments = len(sig.parameters) != 0

        self._server_name = name
        self._name = name + "-stateless"
        super().__init__()

    def _create_server(self, factory_argument: Any | None) -> MCPServer:
        if self._server_accepts_arguments:
            server = cast(Callable[[Any | None], MCPServer], self._server_factory)(
                factory_argument
            )
        else:
            server = cast(Callable[[], MCPServer], self._server_factory)()
        _reject_dynamic_tool_filter(
            self._server_name, server, _LEGACY_TOOL_FILTER_REMEDY
        )
        _reject_worker_side_require_approval(
            self._server_name, server, "stateless_mcp_server"
        )
        _reject_worker_side_tool_meta_resolver(
            self._server_name,
            server,
            _legacy_tool_meta_resolver_remedy("stateless_mcp_server"),
        )
        _reject_worker_side_tool_guardrails(
            self._server_name,
            server,
            _legacy_tool_guardrails_remedy("stateless_mcp_server"),
        )
        return server

    @property
    def name(self) -> str:
        """Get the server name."""
        return self._name

    def _get_activities(self) -> Sequence[Callable]:
        @activity.defn(name=self.name + "-list-tools")
        async def list_tools(
            args: _StatelessListToolsArguments | None = None,
        ) -> list[MCPTool]:
            server = self._create_server(args.factory_argument if args else None)
            try:
                await server.connect()
                return await server.list_tools()
            finally:
                await server.cleanup()

        @activity.defn(name=self.name + "-call-tool-v2")
        async def call_tool(args: _StatelessCallToolsArguments) -> CallToolResult:
            server = self._create_server(args.factory_argument)
            try:
                await server.connect()
                return await server.call_tool(args.tool_name, args.arguments, args.meta)
            finally:
                await server.cleanup()

        @activity.defn(name=self.name + "-list-prompts")
        async def list_prompts(
            args: _StatelessListPromptsArguments | None = None,
        ) -> ListPromptsResult:
            server = self._create_server(args.factory_argument if args else None)
            try:
                await server.connect()
                return await server.list_prompts()
            finally:
                await server.cleanup()

        @activity.defn(name=self.name + "-get-prompt-v2")
        async def get_prompt(args: _StatelessGetPromptArguments) -> GetPromptResult:
            server = self._create_server(args.factory_argument)
            try:
                await server.connect()
                return await server.get_prompt(args.name, args.arguments)
            finally:
                await server.cleanup()

        @activity.defn(name=self.name + "-call-tool")
        async def call_tool_deprecated(
            tool_name: str,
            arguments: dict[str, Any] | None,
        ) -> CallToolResult:
            return await call_tool(
                _StatelessCallToolsArguments(tool_name, arguments, None)
            )

        @activity.defn(name=self.name + "-get-prompt")
        async def get_prompt_deprecated(
            name: str,
            arguments: dict[str, Any] | None,
        ) -> GetPromptResult:
            return await get_prompt(_StatelessGetPromptArguments(name, arguments, None))

        return (
            list_tools,
            call_tool,
            list_prompts,
            get_prompt,
            call_tool_deprecated,
            get_prompt_deprecated,
        )


def _handle_worker_failure(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any):
        try:
            return await func(*args, **kwargs)
        except ActivityError as e:
            failure = e.failure
            if failure:
                cause = failure.cause
                if cause:
                    if (
                        cause.timeout_failure_info.timeout_type
                        == TIMEOUT_TYPE_SCHEDULE_TO_START
                    ):
                        raise ApplicationError(
                            "MCP Stateful Server Worker failed to schedule activity.",
                            type="DedicatedWorkerFailure",
                        ) from e
                    if (
                        cause.timeout_failure_info.timeout_type
                        == TIMEOUT_TYPE_HEARTBEAT
                    ):
                        raise ApplicationError(
                            "MCP Stateful Server Worker failed to heartbeat.",
                            type="DedicatedWorkerFailure",
                        ) from e
            raise e

    return wrapper


@dataclasses.dataclass
class _StatefulCallToolsArguments:
    tool_name: str
    arguments: dict[str, Any] | None
    meta: dict[str, Any] | None = None


@dataclasses.dataclass
class _StatefulGetPromptArguments:
    name: str
    arguments: dict[str, Any] | None


@dataclasses.dataclass
class _StatefulServerSessionArguments:
    factory_argument: Any | None


# Seconds between heartbeats of the ``-server-session`` activity.
_HEARTBEAT_INTERVAL = 30.0

# Patch marker of runs whose operations without a schedule-to-start or
# schedule-to-close timeout follow the session activity.
_SESSION_WATCH_PATCH = "openai-agents-stateful-mcp-session-watch"

# Seconds the ``-server-session`` activity waits for the server to show its
# first heartbeat.
_STORE_TIMEOUT = 30.0

# Seconds an attempt that connected no MCP server waits for heartbeats still
# being encoded before it clears its marker.
_CLOSE_TIMEOUT = 10.0

# Describe errors that mean the worker's client lacks permission.
_DESCRIBE_PERMISSION_ERRORS = {
    RPCStatusCode.PERMISSION_DENIED,
    RPCStatusCode.UNAUTHENTICATED,
}

# Describe errors that retrying cannot fix.
_DESCRIBE_REFUSALS = {
    RPCStatusCode.PERMISSION_DENIED,
    RPCStatusCode.UNAUTHENTICATED,
    RPCStatusCode.UNIMPLEMENTED,
    RPCStatusCode.NOT_FOUND,
    RPCStatusCode.INVALID_ARGUMENT,
}

# Heartbeat details of a ``-server-session`` attempt that connected its MCP
# server. Heartbeat details carry over to the next attempt, which must not open
# a new, empty session in its place.
_SESSION_OPENED = "mcp-session-opened"
# Heartbeat details of an attempt that has not connected yet. The worker sends
# heartbeats later, so an attempt can be lost after connecting but before its
# marker reaches the server.
_SESSION_MAY_BE_OPEN = "mcp-session-may-be-open"


class _SessionHeartbeats:
    """Heartbeat details of one ``-server-session`` attempt.

    Only the worker sends them (``activity.heartbeat()``), so cancels, pauses
    and resets reach the attempt as before. The worker sends its latest
    details with the attempt's failure, but it can receive heartbeats out of
    order.

    The plugin cannot wait for a heartbeat, so every heartbeat says a session
    may be open until the attempt ends without one, and the server is not
    connected until the Temporal server shows that. Both markers refuse a
    retry, so only the heartbeat that clears them must not be overtaken.
    """

    def __init__(self) -> None:
        info = activity.info()
        self._client = activity.client()
        self._workflow_id = info.workflow_id or ""
        self._run_id = info.workflow_run_id
        self._activity_id = info.activity_id
        self._attempt = info.attempt
        self._data_converter = self._client.data_converter.with_context(
            ActivitySerializationContext(
                namespace=info.namespace,
                workflow_id=info.workflow_id,
                workflow_type=info.workflow_type,
                activity_type=info.activity_type,
                activity_id=info.activity_id,
                activity_task_queue=info.task_queue,
                is_local=info.is_local,
            )
        )
        self._may_be_open = True
        self._opened = False
        # The worker encodes each heartbeat in its own task, so they can reach
        # it out of order.
        self._sending: set[asyncio.Task[Any]] = set()
        # The worker sends an attempt's first heartbeat at once.
        self.beat()

    def beat(self) -> None:
        """Sends the current details."""
        if self._opened:
            details = [_SESSION_OPENED]
        elif self._may_be_open:
            details = [_SESSION_MAY_BE_OPEN]
        else:
            details = []
        before = asyncio.all_tasks()
        activity.heartbeat(*details)
        for task in asyncio.all_tasks() - before:
            self._sending.add(task)
            task.add_done_callback(self._sending.discard)

    async def wait_until_stored(self) -> None:
        """Waits until the Temporal server holds this attempt's first heartbeat.

        Until then, a lost attempt's retry would find no marker and open a
        new, empty session.
        """
        handle = self._client.get_workflow_handle(
            self._workflow_id, run_id=self._run_id
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _STORE_TIMEOUT
        last_error: RPCError | None = None

        async def _read_back() -> None:
            nonlocal last_error
            delay = 0.1
            while True:
                try:
                    description = await handle.describe(
                        rpc_timeout=timedelta(
                            seconds=max(deadline - loop.time(), 0.001)
                        )
                    )
                except RPCError as err:
                    if err.status in _DESCRIBE_REFUSALS:
                        status = f"{err.status.name}: {err.message}"
                        if err.status in _DESCRIBE_PERMISSION_ERRORS:
                            reason = (
                                f"the server refused it ({status}). Give the "
                                "worker's client permission to describe "
                                "workflows in this namespace."
                            )
                        else:
                            reason = (
                                "the server could not describe this workflow "
                                f"run ({status})."
                            )
                        raise ApplicationError(
                            "The stateful MCP server session activity confirms "
                            "its first heartbeat with DescribeWorkflowExecution, "
                            f"and {reason}",
                            type="DedicatedWorkerFailure",
                            non_retryable=True,
                        ) from err
                    last_error = err
                else:
                    pending = next(
                        (
                            pending
                            for pending in description.raw_description.pending_activities
                            if pending.activity_id == self._activity_id
                        ),
                        None,
                    )
                    if pending is None or pending.attempt != self._attempt:
                        # A later attempt or the workflow's end replaced this
                        # one, and the details on the server are not this
                        # attempt's.
                        raise ApplicationError(
                            "This stateful MCP server session activity attempt "
                            "is no longer the current one."
                        )
                    try:
                        details = await self._data_converter.decode(
                            pending.heartbeat_details.payloads
                        )
                    except Exception as err:
                        raise ApplicationError(
                            "The stateful MCP server session activity could not "
                            "decode its own heartbeat details.",
                            type="DedicatedWorkerFailure",
                            non_retryable=True,
                        ) from err
                    if _SESSION_MAY_BE_OPEN in details or _SESSION_OPENED in details:
                        return
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)

        # The deadline also bounds a describe that the client keeps retrying,
        # and the decoding of the details.
        try:
            await asyncio.wait_for(_read_back(), _STORE_TIMEOUT)
        except asyncio.TimeoutError:
            raise ApplicationError(
                "The server did not show the stateful MCP server session "
                f"activity's first heartbeat within {_STORE_TIMEOUT:g} seconds."
            ) from last_error

    def record_opened(self) -> None:
        """Records that the MCP server is connected."""
        self._opened = True
        self.beat()

    async def close(self) -> None:
        """Ends the attempt. Its failure carries the last details to a retry."""
        if self._may_be_open and not self._opened:
            # This attempt connected no server, so a retry may connect again.
            # Send that last, after every earlier heartbeat: one still being
            # encoded would otherwise replace it. A codec may never finish, so
            # after _CLOSE_TIMEOUT, or when the wait is interrupted, cancel the
            # heartbeats still being encoded and clear all the same: the clear
            # has no details, so it skips the codec, and the worker waits only
            # for an activity's last heartbeat.
            try:
                if self._sending:
                    await asyncio.wait(set(self._sending), timeout=_CLOSE_TIMEOUT)
            finally:
                # Only "may be open" heartbeats can still be pending here.
                for task in list(self._sending):
                    task.cancel()
                self._may_be_open = False
                self.beat()


def _earlier_attempt_failure(details: Sequence[Any]) -> ApplicationError | None:
    """The failure for a retry whose earlier attempt opened, or may have opened, a session."""
    if _SESSION_OPENED in details:
        reason = "opened an MCP session"
    elif _SESSION_MAY_BE_OPEN in details:
        reason = (
            "ended (it lost its worker, timed out, or was reset) without "
            "confirming that it opened no MCP session"
        )
    else:
        return None
    return ApplicationError(
        f"An earlier attempt of this stateful MCP server session activity {reason}, "
        "and Temporal cannot recreate an MCP session's state.",
        type="DedicatedWorkerFailure",
        non_retryable=True,
    )


def _cancel_requests() -> int:
    """Pending cancellation requests of the current task (Python 3.11 and later)."""
    cancelling = getattr(asyncio.current_task(), "cancelling", None)
    return cancelling() if cancelling else 0


def _session_ended_failure(session: ActivityHandle) -> ApplicationError:
    cause = None if session.cancelled() else session.exception()
    if session.cancelled() or (cause is not None and is_cancelled_exception(cause)):
        message = (
            "The stateful MCP server session activity was cancelled, as when its "
            "async with block exits. Use the server inside an open "
            "stateful_mcp_server() block."
        )
    else:
        message = (
            "The stateful MCP server session activity has ended, so no worker "
            "serves this server's operations."
        )
    failure = ApplicationError(message, type="DedicatedWorkerFailure")
    failure.__cause__ = cause
    return failure


class _StatefulMCPServerReference(MCPServer, AbstractAsyncContextManager):  # type:ignore[reportUnusedClass]
    def __init__(
        self,
        server: str,
        config: ActivityConfig | None,
        server_session_config: ActivityConfig | None,
        factory_argument: Any | None,
        require_approval: RequireApprovalSetting = None,
    ):
        self._name = server + "-stateful"
        self._config = config or ActivityConfig(
            start_to_close_timeout=timedelta(minutes=1),
            schedule_to_start_timeout=timedelta(seconds=30),
        )
        self._server_session_config = server_session_config or ActivityConfig(
            start_to_close_timeout=timedelta(hours=1),
        )
        self._connect_handle: ActivityHandle | None = None
        self._factory_argument = factory_argument
        super().__init__(require_approval=require_approval)

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        self._config["task_queue"] = self.name + "@" + workflow.info().run_id
        self._connect_handle = workflow.start_activity(
            self.name + "-server-session",
            _StatefulServerSessionArguments(self._factory_argument),
            **self._server_session_config,
        )

    async def cleanup(self) -> None:
        """Cancels the session activity and waits for it to end.

        If the session activity failed with ``DedicatedWorkerFailure``, raises
        ``ApplicationError(type="DedicatedWorkerFailure")`` with the session's
        ``ActivityError`` as its cause. If the block's last timer fired in the
        workflow task that brought the session's end, ahead of it, raises it
        with no cause, whatever the session's error, since the SDK then drops
        the session's result. A cancellation wins over this failure only when
        it leaves the ``async with`` block: ``cleanup()`` called directly after
        a workflow cancellation raises it.
        """
        if self._connect_handle:
            session = self._connect_handle
            coro = session.get_coro()
            started = not (
                inspect.iscoroutine(coro)
                and inspect.getcoroutinestate(coro) == inspect.CORO_CREATED
            )
            cancels = _cancel_requests()
            session.cancel()
            try:
                await session
            except asyncio.CancelledError:
                if (
                    not started
                    or not session.cancelled()
                    or _cancel_requests() > cancels
                ):
                    raise
                # The block's last timer fired in the workflow task that
                # brought the session's end, ahead of it, and cancelling the
                # handle dropped the session's result.
                raise ApplicationError(
                    "The stateful MCP server session activity has ended, so no "
                    "worker serves this server's operations.",
                    type="DedicatedWorkerFailure",
                ) from None
            except Exception as e:
                if is_cancelled_exception(e):
                    pass
                elif (
                    isinstance(e, ActivityError)
                    and isinstance(e.cause, ApplicationError)
                    and e.cause.type == "DedicatedWorkerFailure"
                ):
                    raise _session_ended_failure(session) from e
                else:
                    raise

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await self.cleanup()
        except ApplicationError:
            # A cancellation leaving the block wins over the session's failure.
            if exc_value is not None and is_cancelled_exception(exc_value):
                return
            raise

    async def _execute_operation(
        self, activity: str, args: list[Any], result_type: type
    ) -> Any:
        """Runs an operation activity on the session's task queue.

        Only the session activity's worker serves that queue. A
        schedule_to_start_timeout or schedule_to_close_timeout already ends an
        operation once that worker is gone. Without either, the operation would
        wait forever, so fail it with DedicatedWorkerFailure once the session
        activity has ended. Only runs with the patch marker do this: runs
        recorded before it schedule the operation as earlier versions did, so
        their histories replay.
        """
        session = self._connect_handle
        assert session
        if (
            self._config.get("schedule_to_start_timeout")
            or self._config.get("schedule_to_close_timeout")
            or not workflow.patched(_SESSION_WATCH_PATCH)
        ):
            return await workflow.execute_activity(
                activity, args=args, result_type=result_type, **self._config
            )
        if session.done():
            raise _session_ended_failure(session)
        operation = workflow.start_activity(
            activity, args=args, result_type=result_type, **self._config
        )
        if (
            self._config.get("cancellation_type")
            != ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
        ):
            # Await the operation itself, as ``workflow.execute_activity`` does,
            # so the workflow resumes in the same order as on earlier versions
            # and the SDK handles a cancellation that arrives with the result.
            # If the session ends first, cancel the operation and fail.
            cancelled_by_session = False

            def cancel_operation(_: Any) -> None:
                nonlocal cancelled_by_session
                if not operation.done():
                    cancelled_by_session = True
                    operation.cancel()

            session.add_done_callback(cancel_operation)
            # A count, not a flag: it stays up after code catches a cancellation.
            cancels = _cancel_requests()
            try:
                return await operation
            except BaseException as err:
                if (
                    cancelled_by_session
                    and is_cancelled_exception(err)
                    and _cancel_requests() <= cancels
                ):
                    raise _session_ended_failure(session)
                raise
            finally:
                session.remove_done_callback(cancel_operation)
        try:
            await workflow.wait(
                [operation, session], return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            if operation.done():
                # Cancelled in the activation that completed the operation:
                # keep the cancellation, as awaiting the operation would.
                raise
            # The operation is not awaited directly, so pass the cancellation on
            # and wait for it as its cancellation type says.
            operation.cancel()
            return await operation
        if operation.done():
            return operation.result()
        # Fail without waiting for the operation: the worker that ran it may be
        # gone. A cancellation that waits for that worker to acknowledge it would
        # also keep the workflow from being evicted, so the operation is left to
        # its timeout instead. A later session on this task queue refuses it.
        operation.add_done_callback(lambda op: op.cancelled() or op.exception())
        raise _session_ended_failure(session)

    @_handle_worker_failure
    async def list_tools(
        self,
        run_context: RunContextWrapper[Any] | None = None,
        agent: AgentBase | None = None,
    ) -> list[MCPTool]:
        if not self._connect_handle:
            raise ApplicationError(
                "Stateful MCP Server not connected. Call connect first."
            )
        return await self._execute_operation(
            self.name + "-list-tools", [], list[MCPTool]
        )

    @_handle_worker_failure
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        if not self._connect_handle:
            raise ApplicationError(
                "Stateful MCP Server not connected. Call connect first."
            )
        return await self._execute_operation(
            self.name + "-call-tool-v2",
            [_StatefulCallToolsArguments(tool_name, arguments, meta)],
            CallToolResult,
        )

    @_handle_worker_failure
    async def list_prompts(self) -> ListPromptsResult:
        if not self._connect_handle:
            raise ApplicationError(
                "Stateful MCP Server not connected. Call connect first."
            )
        return await self._execute_operation(
            self.name + "-list-prompts", [], ListPromptsResult
        )

    @_handle_worker_failure
    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        if not self._connect_handle:
            raise ApplicationError(
                "Stateful MCP Server not connected. Call connect first."
            )
        return await self._execute_operation(
            self.name + "-get-prompt-v2",
            [_StatefulGetPromptArguments(name, arguments)],
            GetPromptResult,
        )


class StatefulMCPServerProvider:
    """A stateful MCP server implementation for Temporal workflows.

    .. deprecated:: 1.0.0
        Use ``OpenAIAgentsPlugin(mcp_servers=...)`` and
        ``workflow.temporal_mcp_server()`` instead.

    This class wraps a function to create MCP servers to maintain a persistent connection throughout
    the workflow execution. It creates a dedicated worker that stays connected to
    the MCP server and processes operations on a dedicated task queue.

    This approach will allow the MCPServer to maintain state across calls if needed, but the caller
    will have to handle cases where the dedicated worker fails, as Temporal is unable to seamlessly
    recreate any lost state in that case. It is discouraged to use this approach unless necessary.

    Handling dedicated worker failure will entail catching ApplicationError with type "DedicatedWorkerFailure",
    which is also raised if the session activity is retried after it may have opened a session.
    Depending on the usage pattern, the caller will then have to either restart from the point at which the Stateful
    server was needed or handle continuing from that loss of state in some other way.
    """

    def __init__(
        self,
        name: str,
        server_factory: Callable[[Any | None], MCPServer],
    ):
        """Initialize the stateful temporal MCP server.

        Args:
            name: The name of the MCP server.
            server_factory: A function which will produce MCPServer instances. It should return a new server each time
                so that state is not shared between workflow runs. It receives an optional ``factory_argument`` from the
                workflow. A callable ``tool_filter``, a ``require_approval`` policy that can ask for approval, a
                ``tool_meta_resolver``, or a tool guardrail on the returned server is rejected in every MCP operation;
                a static filter is applied.
        """
        warnings.warn(
            "StatefulMCPServerProvider is deprecated; use "
            "OpenAIAgentsPlugin(mcp_servers=...) and "
            "workflow.temporal_mcp_server() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._server_factory = server_factory
        self._server_name = name
        self._name = name + "-stateful"
        self._connect_handle: ActivityHandle | None = None
        self._servers: dict[str, MCPServer] = {}
        self._session_scheduled_times: dict[str, datetime] = {}
        super().__init__()

    @property
    def name(self) -> str:
        """Get the server name."""
        return self._name

    def _get_activities(self) -> Sequence[Callable]:
        def _server_id():
            return self.name + "@" + (activity.info().workflow_run_id or "")

        def _server() -> MCPServer:
            server_id = _server_id()
            if (
                activity.info().scheduled_time
                < self._session_scheduled_times[server_id]
            ):
                # Every server block of a run uses the same task queue. An
                # operation first scheduled before this session belongs to an
                # earlier block, which has ended, and must not run against this
                # server.
                raise ApplicationError(
                    "This operation belongs to an earlier stateful MCP server "
                    "session of this workflow run, which has ended.",
                    type="DedicatedWorkerFailure",
                    non_retryable=True,
                )
            server = self._servers[server_id]
            _reject_dynamic_tool_filter(
                self._server_name, server, _LEGACY_TOOL_FILTER_REMEDY
            )
            _reject_worker_side_require_approval(
                self._server_name, server, "stateful_mcp_server"
            )
            _reject_worker_side_tool_meta_resolver(
                self._server_name,
                server,
                _legacy_tool_meta_resolver_remedy("stateful_mcp_server"),
            )
            _reject_worker_side_tool_guardrails(
                self._server_name,
                server,
                _legacy_tool_guardrails_remedy("stateful_mcp_server"),
            )
            return server

        @activity.defn(name=self.name + "-list-tools")
        async def list_tools() -> list[MCPTool]:
            return await _server().list_tools()

        @activity.defn(name=self.name + "-call-tool")
        async def call_tool_deprecated(
            tool_name: str, arguments: dict[str, Any] | None
        ) -> CallToolResult:
            return await _server().call_tool(tool_name, arguments)

        @activity.defn(name=self.name + "-call-tool-v2")
        async def call_tool(args: _StatefulCallToolsArguments) -> CallToolResult:
            return await _server().call_tool(args.tool_name, args.arguments, args.meta)

        @activity.defn(name=self.name + "-list-prompts")
        async def list_prompts() -> ListPromptsResult:
            return await _server().list_prompts()

        @activity.defn(name=self.name + "-get-prompt")
        async def get_prompt_deprecated(
            name: str, arguments: dict[str, Any] | None
        ) -> GetPromptResult:
            return await _server().get_prompt(name, arguments)

        @activity.defn(name=self.name + "-get-prompt-v2")
        async def get_prompt(args: _StatefulGetPromptArguments) -> GetPromptResult:
            return await _server().get_prompt(args.name, args.arguments)

        async def heartbeat_every(delay: float, heartbeats: _SessionHeartbeats) -> None:
            """Heartbeat every ``delay`` seconds until cancelled."""
            while True:
                await asyncio.sleep(delay)
                heartbeats.beat()

        @activity.defn(name=self.name + "-server-session")
        async def connect(
            args: _StatefulServerSessionArguments | None = None,
        ) -> None:
            server_id = self.name + "@" + (activity.info().workflow_run_id or "")
            failure = _earlier_attempt_failure(activity.info().heartbeat_details)
            if failure:
                raise failure
            heartbeats = _SessionHeartbeats()
            interval = _HEARTBEAT_INTERVAL
            heartbeat_timeout = activity.info().heartbeat_timeout
            if heartbeat_timeout:
                # Heartbeat well within a shorter heartbeat timeout too.
                interval = min(interval, heartbeat_timeout.total_seconds() / 2)
            heartbeat_task = asyncio.create_task(heartbeat_every(interval, heartbeats))
            try:
                # No server is built before the Temporal server holds the first
                # heartbeat, which says a session may be open.
                await heartbeats.wait_until_stored()
                # A cancellation, such as a worker shutdown, doesn't interrupt
                # the read-back, and Python 3.10 and 3.11's asyncio.wait_for can
                # drop one.
                if activity.is_cancelled():
                    raise asyncio.CancelledError()
                if server_id in self._servers:
                    raise ApplicationError(
                        "Cannot connect to an already running server. Use a distinct name if running multiple servers in one workflow."
                    )
                await _run_session(server_id, heartbeats, args)
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                await heartbeats.close()

        async def _run_session(
            server_id: str,
            heartbeats: _SessionHeartbeats,
            args: _StatefulServerSessionArguments | None,
        ) -> None:
            server = self._server_factory(args.factory_argument if args else None)
            # A server with a callable tool filter, a worker-side approval
            # policy, a tool_meta_resolver or tool guardrails is never
            # connected. The dedicated worker still runs, and
            # each of its operations fails with the same error, so the workflow
            # sees that error at its first MCP call whatever its
            # schedule-to-start timeout.
            has_filter = _has_dynamic_tool_filter(server)
            rejected = (
                has_filter
                or _has_worker_side_require_approval(server)
                or _has_worker_side_tool_meta_resolver(server)
                or _has_worker_side_tool_guardrails(server)
            )
            self._servers[server_id] = server
            self._session_scheduled_times[server_id] = activity.info().scheduled_time
            try:
                try:
                    if not rejected:
                        await server.connect()
                        # Nor does every connect() observe it: close this server
                        # instead of serving.
                        if activity.is_cancelled():
                            raise asyncio.CancelledError()
                        heartbeats.record_opened()

                    worker = Worker(
                        activity.client(),
                        task_queue=server_id,
                        activities=[
                            list_tools,
                            call_tool,
                            list_prompts,
                            get_prompt,
                            call_tool_deprecated,
                            get_prompt_deprecated,
                        ],
                        activity_task_poller_behavior=PollerBehaviorSimpleMaximum(1),
                    )

                    await worker.run()
                finally:
                    if not rejected:
                        await server.cleanup()
            finally:
                del self._servers[server_id]
                del self._session_scheduled_times[server_id]

        return (connect,)
