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
    _reject_dynamic_tool_filter,
    _reject_worker_side_require_approval,
)
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
                receives a ``factory_argument`` from the workflow. A callable ``tool_filter`` or a ``require_approval``
                policy that can ask for approval on the returned server is rejected in every MCP operation; a static
                filter is applied.
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
        delay = 0.1
        handle = self._client.get_workflow_handle(
            self._workflow_id, run_id=self._run_id
        )
        while True:
            description = await handle.describe()
            pending = next(
                (
                    pending
                    for pending in description.raw_description.pending_activities
                    if pending.activity_id == self._activity_id
                ),
                None,
            )
            if pending is None or pending.attempt != self._attempt:
                # A later attempt or the workflow's end replaced this one, and
                # the details on the server are not this attempt's.
                raise ApplicationError(
                    "This stateful MCP server session activity attempt is no "
                    "longer the current one."
                )
            details = await self._data_converter.decode(
                pending.heartbeat_details.payloads
            )
            if _SESSION_MAY_BE_OPEN in details or _SESSION_OPENED in details:
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)

    def record_opened(self) -> None:
        """Records that the MCP server is connected."""
        self._opened = True
        self.beat()

    async def close(self) -> None:
        """Ends the attempt. Its failure carries the last details to a retry."""
        if self._may_be_open and not self._opened:
            # This attempt connected no server, so a retry may connect again.
            # Send that last, after every earlier heartbeat: one still being
            # encoded would otherwise replace it. If this wait is interrupted,
            # the marker stays and the retry is refused.
            if self._sending:
                await asyncio.wait(set(self._sending))
            self._may_be_open = False
            self.beat()


def _earlier_attempt_failure(details: Sequence[Any]) -> ApplicationError | None:
    """The failure for a retry whose earlier attempt opened, or may have opened, a session."""
    if _SESSION_OPENED in details:
        reason = "opened an MCP session"
    elif _SESSION_MAY_BE_OPEN in details:
        reason = "was lost after its first heartbeat and may have opened an MCP session"
    else:
        return None
    return ApplicationError(
        f"An earlier attempt of this stateful MCP server session activity {reason}, "
        "and Temporal cannot recreate its state.",
        type="DedicatedWorkerFailure",
        non_retryable=True,
    )


def _session_ended_failure(session: ActivityHandle) -> ApplicationError:
    failure = ApplicationError(
        "The stateful MCP server session activity has ended, so no worker serves "
        "this server's operations.",
        type="DedicatedWorkerFailure",
    )
    if not session.cancelled():
        failure.__cause__ = session.exception()
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
        if self._connect_handle:
            self._connect_handle.cancel()
            try:
                await self._connect_handle
            except Exception as e:
                if is_cancelled_exception(e):
                    pass
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
        await self.cleanup()

    async def _execute_operation(
        self, activity: str, args: list[Any], result_type: type
    ) -> Any:
        """Runs an operation activity on the session's task queue.

        Only the session activity's worker serves that queue. A
        schedule_to_start_timeout or schedule_to_close_timeout already ends an
        operation once that worker is gone. Without either, the operation would
        wait forever, so fail it with DedicatedWorkerFailure once the session
        activity has ended.
        """
        session = self._connect_handle
        assert session
        if self._config.get("schedule_to_start_timeout") or self._config.get(
            "schedule_to_close_timeout"
        ):
            return await workflow.execute_activity(
                activity, args=args, result_type=result_type, **self._config
            )
        if session.done():
            raise _session_ended_failure(session)
        operation = workflow.start_activity(
            activity, args=args, result_type=result_type, **self._config
        )
        try:
            await workflow.wait(
                [operation, session], return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            if operation.done():
                raise
            # The operation is not awaited directly, so pass the cancellation on
            # and wait for it as its cancellation type says.
            operation.cancel()
            return await operation
        if operation.done():
            return operation.result()
        # Fail without waiting for the operation: the worker that ran it may be
        # gone. A cancellation that waits for that worker to acknowledge it would
        # also keep the workflow from being evicted, so such an operation is left
        # to its timeout instead. A later session on this task queue refuses it.
        if (
            self._config.get("cancellation_type")
            != ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
        ):
            operation.cancel()
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

    Handling dedicated worker failure will entail catching ApplicationError with type "DedicatedWorkerFailure".
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
                workflow. A callable ``tool_filter`` or a ``require_approval`` policy that can ask for approval on the
                returned server is rejected in every MCP operation; a static filter is applied.
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
            if server_id in self._servers:
                raise ApplicationError(
                    "Cannot connect to an already running server. Use a distinct name if running multiple servers in one workflow."
                )
            heartbeats = _SessionHeartbeats()
            interval = _HEARTBEAT_INTERVAL
            heartbeat_timeout = activity.info().heartbeat_timeout
            if heartbeat_timeout:
                # Heartbeat well within a shorter heartbeat timeout too.
                interval = min(interval, heartbeat_timeout.total_seconds() / 2)
            heartbeat_task = asyncio.create_task(heartbeat_every(interval, heartbeats))
            try:
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
            # A server with a callable tool filter or a worker-side approval
            # policy is never connected. The dedicated worker still runs, and
            # each of its operations fails with the same error, so the workflow
            # sees that error at its first MCP call whatever its
            # schedule-to-start timeout.
            has_filter = _has_dynamic_tool_filter(server)
            rejected = has_filter or _has_worker_side_require_approval(server)
            self._servers[server_id] = server
            self._session_scheduled_times[server_id] = activity.info().scheduled_time
            try:
                try:
                    if not rejected:
                        await heartbeats.wait_until_stored()
                        await server.connect()
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
