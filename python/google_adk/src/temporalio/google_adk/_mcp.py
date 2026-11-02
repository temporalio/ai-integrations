import asyncio
import contextvars
import functools
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from types import TracebackType
from typing import Any, Callable

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import EventActions
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.genai.types import FunctionDeclaration

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
from temporalio.worker import PollerBehaviorSimpleMaximum, Worker
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityConfig,
    ActivityHandle,
)


@dataclass
class _GetToolsArguments:
    factory_argument: Any | None


@dataclass
class _ToolResult:
    name: str
    description: str
    is_long_running: bool
    custom_metadata: dict[str, Any] | None
    function_declaration: FunctionDeclaration | None


@dataclass
class TemporalToolContext:
    """Context for tools running within Temporal workflows.

    Provides access to tool confirmation and event actions for ADK integration.
    """

    tool_confirmation: ToolConfirmation | None
    function_call_id: str | None
    event_actions: EventActions

    def request_confirmation(
        self,
        *,
        hint: str | None = None,
        payload: Any | None = None,
    ) -> None:
        """Requests confirmation for the given function call.

        Args:
          hint: A hint to the user on how to confirm the tool call.
          payload: The payload used to confirm the tool call.
        """
        if not self.function_call_id:
            raise ValueError("function_call_id is not set.")
        self.event_actions.requested_tool_confirmations[self.function_call_id] = (
            ToolConfirmation(
                hint=hint or "",
                payload=payload,
            )
        )


@dataclass
class _CallToolResult:
    result: Any
    tool_context: TemporalToolContext


@dataclass
class _CallToolArguments:
    factory_argument: Any | None
    name: str
    arguments: dict[str, Any]
    tool_context: TemporalToolContext


class TemporalMcpToolSetProvider:
    """Provider for creating Temporal-aware MCP toolsets.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Manages the creation of toolset activities and handles tool execution
    within Temporal workflows.

    This provider is *stateless*: every ``list-tools``/``call-tool`` activity
    invocation builds a fresh ``McpToolset`` via ``toolset_factory``, runs the
    operation, and always closes the toolset in a ``finally`` block. This means
    ``factory_argument`` is honored on every single call (no cross-workflow
    connection sharing or silent mis-routing) and no MCP session or stdio
    subprocess is leaked. State is not maintained across calls; if a persistent
    connection is required, use :class:`TemporalStatefulMcpToolSetProvider`.
    """

    def __init__(
        self,
        name: str,
        toolset_factory: Callable[[Any | None], McpToolset],
    ) -> None:
        """Initializes the toolset provider.

        Args:
            name: Name prefix for the generated activities.
            toolset_factory: Factory function that creates McpToolset instances.
                It should return a new toolset each time so that no state is
                shared between workflow runs.
        """
        super().__init__()
        self._name = name
        self._toolset_factory = toolset_factory

    def _get_activities(self) -> Sequence[Callable]:
        @activity.defn(name=self._name + "-list-tools")
        async def get_tools(
            args: _GetToolsArguments,
        ) -> list[_ToolResult]:
            # Build a fresh toolset per call, honoring this call's
            # ``factory_argument``, and always close it so no MCP session
            # (or stdio subprocess) leaks. See issue #1663.
            toolset = self._toolset_factory(args.factory_argument)
            try:
                tools = await toolset.get_tools()
                return [
                    _ToolResult(
                        tool.name,
                        tool.description,
                        tool.is_long_running,
                        tool.custom_metadata,
                        tool._get_declaration(),
                    )
                    for tool in tools
                ]
            finally:
                await toolset.close()

        @activity.defn(name=self._name + "-call-tool")
        async def call_tool(
            args: _CallToolArguments,
        ) -> _CallToolResult:
            toolset = self._toolset_factory(args.factory_argument)
            try:
                tools = await toolset.get_tools()

                tool_match = [tool for tool in tools if tool.name == args.name]
                if len(tool_match) == 0:
                    raise ApplicationError(
                        f"Unable to find matching mcp tool by name: {args.name}"
                    )
                if len(tool_match) > 1:
                    raise ApplicationError(
                        f"Found multiple MCP tools with the same name: {args.name}"
                    )
                tool = tool_match[0]

                # We cannot provide a full-fledged ToolContext so we need to provide only what is needed by the tool
                result = await tool.run_async(
                    args=args.arguments,
                    tool_context=args.tool_context,  #  type:ignore
                )
                return _CallToolResult(result=result, tool_context=args.tool_context)
            finally:
                await toolset.close()

        return get_tools, call_tool


class _TemporalTool(BaseTool):
    def __init__(
        self,
        set_name: str,
        factory_argument: Any | None,
        config: ActivityConfig | None,
        declaration: FunctionDeclaration | None,
        *,
        name: str,
        description: str,
        is_long_running: bool = False,
        custom_metadata: dict[str, Any] | None = None,
    ):
        super().__init__(
            name=name,
            description=description,
            is_long_running=is_long_running,
            custom_metadata=custom_metadata,
        )
        self._set_name = set_name
        self._factory_argument = factory_argument
        self._config = config or ActivityConfig(
            start_to_close_timeout=timedelta(minutes=1)
        )
        self._declaration = declaration

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        return self._declaration

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        result: _CallToolResult = await workflow.execute_activity(
            self._set_name + "-call-tool",
            _CallToolArguments(
                self._factory_argument,
                self.name,
                arguments=args,
                tool_context=TemporalToolContext(
                    tool_confirmation=tool_context.tool_confirmation,
                    function_call_id=tool_context.function_call_id,
                    event_actions=tool_context._event_actions,
                ),
            ),
            result_type=_CallToolResult,
            **self._config,
        )

        # We need to propagate any event actions back to the main context
        tool_context._event_actions = result.tool_context.event_actions
        return result.result


class TemporalMcpToolSet(BaseToolset):
    """Temporal-aware MCP toolset implementation.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Executes MCP tools as Temporal activities, providing proper isolation
    and execution guarantees within workflows.
    """

    def __init__(
        self,
        name: str,
        config: ActivityConfig | None = None,
        factory_argument: Any | None = None,
        not_in_workflow_toolset: Callable[[Any | None], McpToolset] | None = None,
    ):
        """Initializes the Temporal MCP toolset.

        .. warning::
            Do not pass secrets, credentials, or API keys through ``factory_argument``. It
            is an activity argument, so it is recorded in workflow history and, without a
            payload codec, visible in the web UI. Resolve credentials worker-side inside the
            toolset factory instead.

        Args:
            name: Name of the toolset (used for activity naming).
            config: Optional activity configuration.
            factory_argument: Optional argument passed to ``toolset_factory``.
                Must not contain secrets.
            not_in_workflow_toolset: Optional factory that returns the
                underlying ``McpToolset`` to use when this wrapper executes
                outside ``workflow.in_workflow()``, such as local ADK runs.
                This is not needed during normal workflow execution, but
                ``get_tools()`` raises ``ValueError`` outside a workflow if it
                is omitted.
        """
        super().__init__()
        self._name = name
        self._factory_argument = factory_argument
        self._config = config or ActivityConfig(
            start_to_close_timeout=timedelta(minutes=1)
        )
        self._not_in_workflow_toolset = not_in_workflow_toolset

    async def get_tools(
        self, readonly_context: ReadonlyContext | None = None
    ) -> list[BaseTool]:
        """Retrieves available tools from the MCP toolset.

        Args:
            readonly_context: Optional readonly context (unused in this implementation).

        Returns:
            List of available tools wrapped as Temporal activities.
        """
        # If executed outside a workflow, like when doing local adk runs, use the mcp server directly
        if not workflow.in_workflow():
            if self._not_in_workflow_toolset is None:
                raise ValueError(
                    "Attempted to use TemporalMcpToolSet outside a workflow, but "
                    "no not_in_workflow_toolset was provided. Either use "
                    "McpToolSet directly or pass a factory that returns the "
                    "underlying McpToolset for non-workflow execution."
                )
            return await self._not_in_workflow_toolset(None).get_tools(readonly_context)

        tool_results: list[_ToolResult] = await workflow.execute_activity(
            self._name + "-list-tools",
            _GetToolsArguments(self._factory_argument),
            result_type=list[_ToolResult],
            **self._config,
        )
        return [
            _TemporalTool(
                set_name=self._name,
                factory_argument=self._factory_argument,
                config=self._config,
                declaration=tool_result.function_declaration,
                name=tool_result.name,
                description=tool_result.description,
                is_long_running=tool_result.is_long_running,
                custom_metadata=tool_result.custom_metadata,
            )
            for tool_result in tool_results
        ]


def _handle_worker_failure(func: Callable) -> Callable:
    """Surface dedicated-worker failures on the run-scoped task queue.

    A schedule-to-start timeout means the dedicated ``-server-session`` worker
    never picked the activity up (it is gone); a heartbeat timeout means it
    died mid-session. Either way Temporal cannot recreate the in-memory toolset
    state, so we re-raise as an ``ApplicationError`` of type
    ``"DedicatedWorkerFailure"`` for the caller to handle.

    """

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


def _session_ended_failure(session: ActivityHandle) -> ApplicationError:
    failure = ApplicationError(
        "The stateful MCP session activity has ended, so no worker serves this "
        "toolset's operations.",
        type="DedicatedWorkerFailure",
    )
    if not session.cancelled():
        failure.__cause__ = session.exception()
    return failure


async def _execute_operation(
    session: ActivityHandle, activity: str, *args: Any, **kwargs: Any
) -> Any:
    """Runs a list or call activity on the session's task queue.

    Only the session activity's worker serves that queue, so once the session
    has ended, an operation would wait until its schedule-to-start timeout,
    or forever without one. Fail it with ``DedicatedWorkerFailure`` instead.
    """
    if session.done():
        raise _session_ended_failure(session)
    operation = workflow.start_activity(activity, *args, **kwargs)
    try:
        await workflow.wait([operation, session], return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # The operation is not awaited directly, so pass the cancellation on
        # and wait for it as its cancellation type says.
        operation.cancel()
        return await operation
    if operation.done():
        return operation.result()
    # Fail without waiting for the operation: the worker that ran it may be
    # gone. A cancellation that waits for that worker to acknowledge it would
    # also keep the workflow from being evicted, so such an operation is left
    # to its timeout instead. It stays on this session's task queue, which a
    # later block does not serve (see ``_session_task_queue``).
    if (
        kwargs.get("cancellation_type")
        != ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
    ):
        operation.cancel()
    operation.add_done_callback(lambda op: op.cancelled() or op.exception())
    raise _session_ended_failure(session)


@dataclass
class _StatefulServerSessionArguments:
    factory_argument: Any | None
    # Which ``async with`` block of the run, for this toolset name, the session
    # serves. Earlier versions did not send it.
    block: int = 1


def _session_task_queue(server_id: str, block: int) -> str:
    """The task queue of one toolset block's operations.

    An operation can outlive its block's session, so each block of a run with
    the same name serves its own queue and never runs an earlier block's
    operations. The first block keeps the queue earlier versions used.
    """
    return server_id if block == 1 else f"{server_id}#{block}"


# Toolset blocks opened by each running workflow, by toolset name. A workflow's
# event loop lasts as long as its instance, so replay counts the same blocks.
_blocks_opened: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, int]] = (
    weakref.WeakKeyDictionary()
)


def _next_block(name: str) -> int:
    opened = _blocks_opened.setdefault(asyncio.get_running_loop(), {})
    opened[name] = opened.get(name, 0) + 1
    return opened[name]


# Seconds between heartbeats of the ``-server-session`` activity.
_HEARTBEAT_INTERVAL = 30.0

# Heartbeat details of a ``-server-session`` attempt whose toolset opened an MCP
# session. Heartbeat details carry over to the next attempt, which must not open
# a new, empty session in its place.
_SESSION_OPENED = "mcp-session-opened"
# Heartbeat details of an attempt that has not opened a session yet. The worker
# sends heartbeats later, so an attempt can be lost after opening a session but
# before its marker reaches the server.
_SESSION_MAY_BE_OPEN = "mcp-session-may-be-open"


class _SessionHeartbeats:
    """Heartbeat details of one ``-server-session`` attempt.

    Only the worker sends them (``activity.heartbeat()``), so cancels, pauses
    and resets reach the attempt as before. The worker sends its latest
    details with the attempt's failure, but it can receive heartbeats out of
    order.

    The plugin cannot wait for a heartbeat, so every heartbeat says a session
    may be open until the attempt ends without one, and no session opens until
    the server shows that. Both markers refuse a retry, so only the heartbeat
    that clears them must not be overtaken.
    """

    def __init__(self) -> None:
        info = activity.info()
        # The marker is recorded from list and call activities, so keep this
        # activity's context for ``activity.heartbeat()``.
        self._context = contextvars.copy_context()
        self._client = activity.client()
        self._workflow_id = info.workflow_id or ""
        self._run_id = info.workflow_run_id
        self._activity_id = info.activity_id
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
        self._stored = False
        self._store_lock = asyncio.Lock()
        self._may_be_open = True
        self._opened = False
        self._closed = False
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
        self._context.run(activity.heartbeat, *details)
        for task in asyncio.all_tasks() - before:
            self._sending.add(task)
            task.add_done_callback(self._sending.discard)

    async def wait_until_stored(self) -> None:
        """Waits until the server holds this attempt's first heartbeat.

        Until then, a lost attempt's retry would find no marker and open a
        new, empty session.
        """
        delay = 0.1
        async with self._store_lock:
            while not self._stored:
                if self._closed:
                    raise _stopping_error()
                handle = self._client.get_workflow_handle(
                    self._workflow_id, run_id=self._run_id
                )
                description = await handle.describe()
                for pending in description.raw_description.pending_activities:
                    if pending.activity_id != self._activity_id:
                        continue
                    details = await self._data_converter.decode(
                        pending.heartbeat_details.payloads
                    )
                    self._stored = (
                        _SESSION_MAY_BE_OPEN in details or _SESSION_OPENED in details
                    )
                if not self._stored:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 2.0)

    def record_opened(self) -> None:
        """Records that the toolset opened an MCP session."""
        if self._opened:
            return
        if self._closed:
            raise _stopping_error()
        self._opened = True
        self.beat()

    async def close(self) -> None:
        """Ends the attempt. Its failure carries the last details to a retry."""
        self._closed = True
        if self._may_be_open and not self._opened:
            # This attempt opened no session, so a retry may build the toolset.
            # Send that last, after every earlier heartbeat: one still being
            # encoded would otherwise replace it. If this wait is interrupted,
            # the marker stays and the retry is refused.
            if self._sending:
                await asyncio.wait(set(self._sending))
            self._may_be_open = False
            self.beat()


def _stopping_error() -> ApplicationError:
    return ApplicationError(
        "The stateful MCP session activity is stopping, so it cannot open a new "
        "MCP session."
    )


def _earlier_attempt_failure(details: Sequence[Any]) -> ApplicationError | None:
    """The failure for a retry whose earlier attempt opened, or may have opened, a session."""
    if _SESSION_OPENED in details:
        reason = "opened an MCP session"
    elif _SESSION_MAY_BE_OPEN in details:
        reason = "was lost after its first heartbeat and may have opened an MCP session"
    else:
        return None
    return ApplicationError(
        f"An earlier attempt of this stateful MCP session activity {reason}, "
        "and Temporal cannot recreate its state.",
        type="DedicatedWorkerFailure",
        non_retryable=True,
    )


def _record_opened_sessions(
    toolset: McpToolset, heartbeats: _SessionHeartbeats
) -> bool:
    """Records the session marker whenever the toolset gets an MCP session.

    ADK's ``McpToolset`` and its tools get their session from
    ``MCPSessionManager.create_session()`` before every request. Returns
    ``False`` when the toolset has no session manager to observe.
    """
    manager = getattr(toolset, "_mcp_session_manager", None)
    if manager is None:
        return False
    create_session = manager.create_session

    async def create_session_and_record(*args: Any, **kwargs: Any) -> Any:
        await heartbeats.wait_until_stored()
        session = await create_session(*args, **kwargs)
        heartbeats.record_opened()
        return session

    manager.create_session = create_session_and_record
    return True


@dataclass
class _RunningToolset:
    toolset: McpToolset
    heartbeats: _SessionHeartbeats
    observes_sessions: bool

    async def get(self) -> McpToolset:
        if not self.observes_sessions:
            # No session to observe: record the marker before the first
            # operation instead.
            await self.heartbeats.wait_until_stored()
            self.heartbeats.record_opened()
        return self.toolset


@dataclass
class _StatefulCallToolArguments:
    name: str
    arguments: dict[str, Any]
    tool_context: TemporalToolContext


class TemporalStatefulMcpToolSetProvider:
    """Provider for a stateful, pooled MCP toolset backed by a dedicated worker.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Unlike :class:`TemporalMcpToolSetProvider` (which builds and closes a fresh
    ``McpToolset`` per activity call), this provider maintains a single
    ``McpToolset`` for the lifetime of a workflow run. The workflow-side
    :class:`TemporalStatefulMcpToolSet` starts a dedicated ``-server-session``
    activity on a task queue scoped to that specific run
    (``name@run_id``); that activity builds the toolset once via
    ``toolset_factory(factory_argument)``, holds it open, and runs a nested
    :py:class:`temporalio.worker.Worker` serving the ``-list-tools``/``-call-tool``
    activities off the same run-scoped queue. The toolset is closed when the
    workflow cancels the session activity on cleanup.

    Because state lives in a dedicated worker's memory, the caller must handle
    the case where that worker fails, as Temporal cannot seamlessly recreate the
    lost toolset. Failure surfaces as an ``ApplicationError`` with
    ``type="DedicatedWorkerFailure"``. Prefer the stateless
    :class:`TemporalMcpToolSetProvider` unless a persistent connection is
    genuinely required.
    """

    def __init__(
        self,
        name: str,
        toolset_factory: Callable[[Any | None], McpToolset],
    ) -> None:
        """Initializes the stateful toolset provider.

        Args:
            name: Name prefix for the generated activities. It is suffixed with
                ``-stateful`` so it never collides with a stateless provider of
                the same base name registered on the same worker.
            toolset_factory: Factory function that creates McpToolset instances.
                It should return a new toolset each time so that no state is
                shared between workflow runs.
        """
        super().__init__()
        self._name = name + "-stateful"
        self._toolset_factory = toolset_factory
        self._toolsets: dict[str, _RunningToolset] = {}

    @property
    def name(self) -> str:
        """The activity-name prefix (base name with the ``-stateful`` suffix)."""
        return self._name

    def _get_activities(self) -> Sequence[Callable]:
        def _server_id() -> str:
            return self._name + "@" + (activity.info().workflow_run_id or "")

        @activity.defn(name=self._name + "-list-tools")
        async def get_tools() -> list[_ToolResult]:
            toolset = await self._toolsets[_server_id()].get()
            tools = await toolset.get_tools()
            return [
                _ToolResult(
                    tool.name,
                    tool.description,
                    tool.is_long_running,
                    tool.custom_metadata,
                    tool._get_declaration(),
                )
                for tool in tools
            ]

        @activity.defn(name=self._name + "-call-tool")
        async def call_tool(args: _StatefulCallToolArguments) -> _CallToolResult:
            toolset = await self._toolsets[_server_id()].get()
            tools = await toolset.get_tools()

            tool_match = [tool for tool in tools if tool.name == args.name]
            if len(tool_match) == 0:
                raise ApplicationError(
                    f"Unable to find matching mcp tool by name: {args.name}"
                )
            if len(tool_match) > 1:
                raise ApplicationError(
                    f"Found multiple MCP tools with the same name: {args.name}"
                )
            tool = tool_match[0]

            # We cannot provide a full-fledged ToolContext so we need to provide only what is needed by the tool
            result = await tool.run_async(
                args=args.arguments,
                tool_context=args.tool_context,  #  type:ignore
            )
            return _CallToolResult(result=result, tool_context=args.tool_context)

        async def heartbeat_every(delay: float, heartbeats: _SessionHeartbeats) -> None:
            """Heartbeat every ``delay`` seconds until cancelled."""
            while True:
                await asyncio.sleep(delay)
                heartbeats.beat()

        @activity.defn(name=self._name + "-server-session")
        async def connect(
            args: _StatefulServerSessionArguments | None = None,
        ) -> None:
            server_id = self._name + "@" + (activity.info().workflow_run_id or "")
            failure = _earlier_attempt_failure(activity.info().heartbeat_details)
            if failure:
                raise failure
            if server_id in self._toolsets:
                raise ApplicationError(
                    "Cannot connect to an already running toolset. Use a distinct "
                    "name if running multiple stateful toolsets in one workflow."
                )

            # Created only once the duplicate-connect check has passed, and
            # cancelled in the outermost ``finally`` below so it is torn down
            # on every exit path -- including a ``toolset.close()`` failure,
            # which would otherwise leave it heartbeating forever after this
            # activity has already exited.
            heartbeats = _SessionHeartbeats()
            heartbeat_task = asyncio.create_task(
                heartbeat_every(_HEARTBEAT_INTERVAL, heartbeats)
            )
            try:
                toolset = self._toolset_factory(args.factory_argument if args else None)
                try:
                    self._toolsets[server_id] = _RunningToolset(
                        toolset,
                        heartbeats,
                        _record_opened_sessions(toolset, heartbeats),
                    )
                    try:
                        worker = Worker(
                            activity.client(),
                            task_queue=_session_task_queue(
                                server_id, args.block if args else 1
                            ),
                            activities=[get_tools, call_tool],
                            activity_task_poller_behavior=PollerBehaviorSimpleMaximum(
                                1
                            ),
                        )
                        await worker.run()
                    finally:
                        await toolset.close()
                finally:
                    del self._toolsets[server_id]
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                await heartbeats.close()

        return (connect,)


class _TemporalStatefulTool(BaseTool):
    def __init__(
        self,
        set_name: str,
        session: ActivityHandle,
        config: ActivityConfig,
        declaration: FunctionDeclaration | None,
        *,
        name: str,
        description: str,
        is_long_running: bool = False,
        custom_metadata: dict[str, Any] | None = None,
    ):
        super().__init__(
            name=name,
            description=description,
            is_long_running=is_long_running,
            custom_metadata=custom_metadata,
        )
        self._set_name = set_name
        self._session = session
        self._config = config
        self._declaration = declaration

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        return self._declaration

    @_handle_worker_failure
    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        result: _CallToolResult = await _execute_operation(
            self._session,
            self._set_name + "-call-tool",
            _StatefulCallToolArguments(
                self.name,
                arguments=args,
                tool_context=TemporalToolContext(
                    tool_confirmation=tool_context.tool_confirmation,
                    function_call_id=tool_context.function_call_id,
                    event_actions=tool_context._event_actions,
                ),
            ),
            result_type=_CallToolResult,
            **self._config,
        )

        # We need to propagate any event actions back to the main context
        tool_context._event_actions = result.tool_context.event_actions
        return result.result


class TemporalStatefulMcpToolSet(BaseToolset):
    """Workflow-side handle for a stateful MCP toolset.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Use it as an async context manager inside a workflow so the dedicated
    ``-server-session`` worker is started on ``connect`` and torn down on
    ``cleanup``::

        async with TemporalStatefulMcpToolSet("my-tools") as toolset:
            agent = Agent(..., tools=[toolset])
            await runner.run(...)

    The connection (and any MCP subprocess) lives for the duration of the
    ``async with`` block and is reused across every tool call in that block,
    then closed on exit. Callers must be prepared to handle
    ``ApplicationError(type="DedicatedWorkerFailure")`` should the dedicated
    worker die mid-run.
    """

    def __init__(
        self,
        name: str,
        config: ActivityConfig | None = None,
        server_session_config: ActivityConfig | None = None,
        factory_argument: Any | None = None,
        not_in_workflow_toolset: Callable[[Any | None], McpToolset] | None = None,
    ):
        """Initializes the stateful Temporal MCP toolset handle.

        Args:
            name: Base name of the toolset. Must match the ``name`` passed to
                the :class:`TemporalStatefulMcpToolSetProvider` (the
                ``-stateful`` suffix is applied here automatically).
            config: Optional activity configuration for the per-operation
                (``-list-tools``/``-call-tool``) activities. A
                ``schedule_to_start_timeout`` is used to detect a dead
                dedicated worker. Operations also fail with
                ``DedicatedWorkerFailure`` once the session activity has
                ended, without waiting for a cancellation to be
                acknowledged.
            server_session_config: Optional activity configuration for the
                long-running ``-server-session`` activity.
            factory_argument: Optional argument passed once to the toolset
                factory when the session connects.
            not_in_workflow_toolset: Optional factory that returns the
                underlying ``McpToolset`` to use when this wrapper executes
                outside ``workflow.in_workflow()``, such as local ADK runs.
        """
        super().__init__()
        self._name = name + "-stateful"
        self._config = config or ActivityConfig(
            start_to_close_timeout=timedelta(minutes=1),
            schedule_to_start_timeout=timedelta(seconds=30),
        )
        self._server_session_config = server_session_config or ActivityConfig(
            start_to_close_timeout=timedelta(hours=1),
        )
        self._factory_argument = factory_argument
        self._not_in_workflow_toolset = not_in_workflow_toolset
        self._connect_handle: ActivityHandle | None = None

    async def connect(self) -> None:
        """Starts the dedicated ``-server-session`` activity for this run."""
        if not workflow.in_workflow():
            return
        block = _next_block(self._name)
        self._config["task_queue"] = _session_task_queue(
            self._name + "@" + workflow.info().run_id, block
        )
        self._connect_handle = workflow.start_activity(
            self._name + "-server-session",
            _StatefulServerSessionArguments(self._factory_argument, block),
            **self._server_session_config,
        )

    async def cleanup(self) -> None:
        """Cancels the dedicated session activity and awaits its teardown."""
        if self._connect_handle:
            self._connect_handle.cancel()
            try:
                await self._connect_handle
            except Exception as e:
                if not is_cancelled_exception(e):
                    raise
            finally:
                self._connect_handle = None

    async def close(self) -> None:
        """``BaseToolset`` teardown hook; delegates to :meth:`cleanup`."""
        await self.cleanup()

    async def __aenter__(self) -> "TemporalStatefulMcpToolSet":
        """Connects the toolset and returns it for use as a context manager."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Cleans up the toolset when exiting the context manager."""
        await self.cleanup()

    @_handle_worker_failure
    async def get_tools(
        self, readonly_context: ReadonlyContext | None = None
    ) -> list[BaseTool]:
        """Retrieves available tools from the stateful MCP toolset."""
        # If executed outside a workflow, like when doing local adk runs, use the mcp server directly
        if not workflow.in_workflow():
            if self._not_in_workflow_toolset is None:
                raise ValueError(
                    "Attempted to use TemporalStatefulMcpToolSet outside a "
                    "workflow, but no not_in_workflow_toolset was provided. "
                    "Either use McpToolSet directly or pass a factory that "
                    "returns the underlying McpToolset for non-workflow execution."
                )
            return await self._not_in_workflow_toolset(
                self._factory_argument
            ).get_tools(readonly_context)

        if not self._connect_handle:
            raise ApplicationError(
                "Stateful MCP toolset not connected. Use it as an async context "
                "manager (async with ...) or call connect() first."
            )

        tool_results: list[_ToolResult] = await _execute_operation(
            self._connect_handle,
            self._name + "-list-tools",
            result_type=list[_ToolResult],
            **self._config,
        )
        return [
            _TemporalStatefulTool(
                set_name=self._name,
                session=self._connect_handle,
                config=self._config,
                declaration=tool_result.function_declaration,
                name=tool_result.name,
                description=tool_result.description,
                is_long_running=tool_result.is_long_running,
                custom_metadata=tool_result.custom_metadata,
            )
            for tool_result in tool_results
        ]
