"""Workflow side of the Claude Agent SDK integration.

Call :func:`run_claude_turn` from a workflow::

    reply = await run_claude_turn(message.text, session=self._claude, tools=(lookup,))

Safe to import from workflow code: this module never imports ``claude_agent_sdk``.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import sys
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Literal, Self, TypeVar, get_type_hints

from pydantic import (
    BaseModel,
    Field,
    PydanticUserError,
    TypeAdapter,
    ValidationError,
)
from pydantic_core import to_json

from temporalio import workflow
from temporalio.claude_agent_sdk import workflow as claude_workflow
from temporalio.claude_agent_sdk._models import (
    BRIDGE_POLL_UPDATE,
    BRIDGE_REQUEST_SIGNAL,
    CLAUDE_NATIVE_OUTCOME_UNKNOWN_ERROR_TYPE,
    CLAUDE_NATIVE_TOOL_UNSAFE_ERROR_TYPE,
    CLAUDE_RESULT_ERROR_TYPE,
    CLAUDE_SANDBOX_LOST_ERROR_TYPE,
    CLAUDE_SESSION_DEGRADED_ERROR_TYPE,
    CLAUDE_SESSION_LEASE_LOST_ERROR_TYPE,
    CLAUDE_SESSION_LEASED_ERROR_TYPE,
    CLAUDE_SESSION_MIRROR_ERROR_TYPE,
    CLAUDE_STRUCTURED_OUTPUT_ERROR_TYPE,
    CLAUDE_TURN_BUDGET_ERROR_TYPE,
    CLAUDE_TURN_REJECTED_ERROR_TYPE,
    CLAUDE_TURN_TIMEOUT_ERROR_TYPE,
    NATIVE_DEFAULT_MAX_TURNS,
    NEVER_RESUMED_ERROR_KINDS,
    REPEATABLE_EFFECTS,
    RESUMABLE_DETAIL,
    TURN_PROGRESS_SIGNAL,
    ClaudeBridgeBatch,
    ClaudeBridgePoll,
    ClaudeDeferredTool,
    ClaudeNativeCall,
    ClaudeNativeDecision,
    ClaudeToolDeclaration,
    ClaudeToolRequest,
    ClaudeToolResult,
    ClaudeTurnInput,
    ClaudeTurnOptions,
    ClaudeTurnProgress,
    ClaudeTurnResult,
    _omit_if_none,
    merge_native_calls,
    native_tool_effect,
    tool_call_key,
)
from temporalio.claude_agent_sdk._native_hook import UNKNOWN_OUTCOME, call_key
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    CancelledError,
    TimeoutType,
)
from temporalio.exceptions import TimeoutError as TemporalTimeoutError
from temporalio.workflow import ActivityConfig

__all__ = [
    "ClaudeSession",
    "ClaudeSessionState",
    "ClaudeToolRecord",
    "claude_tool_declaration",
    "run_claude_turn",
    "validate_structured_output",
]

OutputT = TypeVar("OutputT", bound=BaseModel)
T = TypeVar("T")
Approve = Callable[[ClaudeDeferredTool], Awaitable[bool | tuple[bool, str]]]
"""A native-call gate: approve, deny, or deny with the reason Claude is told."""


def tool_argument_coercers(tool: Callable[..., Any]) -> dict[str, TypeAdapter[Any]]:
    """One ``TypeAdapter`` per annotated parameter of a tool, to coerce JSON arguments."""
    try:
        module = sys.modules.get(getattr(tool, "__module__", "") or "")
        hints = get_type_hints(tool, globalns=getattr(module, "__dict__", None))
    except Exception:  # noqa: BLE001 - fall back to the raw annotations
        hints = {}
    adapters: dict[str, TypeAdapter[Any]] = {}
    for name, param in inspect.signature(tool).parameters.items():
        annotation = hints.get(name, param.annotation)
        if annotation is not inspect.Parameter.empty:
            adapters[name] = TypeAdapter(annotation)
    return adapters


# The workflow half of the tool bridge (``_bridge.py`` is the Activity half). Each call runs in a
# copy of the context captured when the turn opened the bridge, so concurrent calls' contextvars
# stay apart. A resumed CLI re-issues a lost call under a fresh ``tool_use_id``, so every call is
# recorded by slot (session, message, tool, canonical input, ordinal) and a re-issued call is
# answered from the record instead of running again.


def _call_address(request: ClaudeToolRequest) -> str:
    """The content of one tool call (name and canonical arguments), without its id or ordinal."""
    return tool_call_key(request.name, request.arguments)


_Slot = tuple[str, str, str, str, int]


def _slot(session_id: str, message_id: str, request: ClaudeToolRequest) -> _Slot:
    """The record slot of one call: its session, message, tool, canonical input and ordinal.
    Never its ``tool_use_id``: a resumed attempt re-issues a lost call under a new one.
    """
    return (
        session_id,
        message_id,
        request.name,
        _call_address(request),
        request.ordinal,
    )


class ClaudeToolRecord(BaseModel):
    """One tool call in a session's call record (``ClaudeSessionState.tool_record``):
    the slot it ran in, the ``tool_use_id`` that ran there, and its result once it has one.
    """

    session_id: str
    message_id: str
    tool: str
    input_key: str = Field(
        description="``tool_call_key`` of the tool name and canonical input."
    )
    ordinal: int
    tool_use_id: str
    aliases: list[str] = Field(
        default_factory=list,
        description="Ids a resumed attempt re-issued this call under, answered from this record.",
    )
    result: ClaudeToolResult | None = None

    @property
    def slot(self) -> _Slot:
        """Return this call's identity in the session tool record."""
        return (
            self.session_id,
            self.message_id,
            self.tool,
            self.input_key,
            self.ordinal,
        )


def recorded_tool_calls(
    records: Sequence[ClaudeToolRecord],
) -> dict[str, tuple[str, int]]:
    """Every id ``records`` ran or answered under, with its call key and ordinal
    (``ClaudeTurnInput.tool_calls``).
    """
    return {
        call_id: (r.input_key, r.ordinal)
        for r in records
        for call_id in (r.tool_use_id, *r.aliases)
    }


class _ToolBridge:
    """One Claude turn's tool registry, in-flight calls, call record, and result log."""

    def __init__(
        self,
        token: str,
        tools: dict[str, Callable[..., Awaitable[Any]]],
        max_in_flight: int,
        session: ClaudeSession,
        message_id: str,
    ) -> None:
        assert session.state.session_id is not None
        self.token = token
        self._session = session
        self._message_id = message_id
        # The record holds one message's calls: a new turn's ordinals start again at 0.
        session.state.tool_record = []
        self.results: list[ClaudeToolResult] = []
        self.closed = False
        self._tools = tools
        self._coercers = {
            name: tool_argument_coercers(tool) for name, tool in tools.items()
        }
        self._max_in_flight = max_in_flight
        self._context = contextvars.copy_context()
        self.seen: set[str] = set()
        self._tasks: dict[asyncio.Task[None], None] = {}

    @property
    def records(self) -> list[ClaudeToolRecord]:
        """The message's call record."""
        return self._session.state.tool_record

    def rotate(self) -> None:
        """Give the next attempt its own token, so calls and polls from the lost one are dropped."""
        hub = _BridgeHub.current()
        del hub.bridges[self.token]
        self.token = str(workflow.uuid4())
        hub.bridges[self.token] = self

    def submit(self, request: ClaudeToolRequest) -> None:
        if self.closed:
            return
        session_id = self._session.state.session_id
        assert session_id is not None
        slot = _slot(session_id, self._message_id, request)
        records = self._session.state.tool_record
        # The same call under the id the record holds is that call asked about again (a lost
        # attempt's dangling call, answered under its own id); its ordinal may not match when the
        # resumed run cannot tell where its message began. A different call under a used id
        # stays a duplicate, below.
        same_call = (request.tool_use_id, request.name, _call_address(request))
        if (
            entry := next(
                (
                    r
                    for r in records
                    if r.slot == slot
                    or (r.tool_use_id, r.tool, r.input_key) == same_call
                ),
                None,
            )
        ) is not None:
            if request.tool_use_id not in (entry.tool_use_id, *entry.aliases):
                entry.aliases.append(request.tool_use_id)
            self._spawn(self._serve(request, entry))
            return
        if request.tool_use_id in self.seen:
            # Same id, a different call: the result log is keyed by tool_use_id, so letting a
            # second live call share one would hand it the other's answer.
            self._reject(request, f"duplicate tool call {request.tool_use_id!r}")
            return
        self.seen.add(request.tool_use_id)
        tool = self._tools.get(request.name)
        if tool is None:
            self._reject(
                request, f"{request.name!r} is not a tool declared for this turn"
            )
            return
        if len(self._tasks) >= self._max_in_flight:
            self._reject(
                request,
                f"too many concurrent tool calls (limit {self._max_in_flight}); retry later",
            )
            return
        adapters = self._coercers[request.name]
        try:
            arguments = {
                k: adapters[k].validate_python(v) if k in adapters else v
                for k, v in request.arguments.items()
            }
        except ValidationError as exc:
            self._reject(request, f"invalid arguments for {request.name!r}: {exc}")
            return
        entry = ClaudeToolRecord(
            session_id=session_id,
            message_id=self._message_id,
            tool=request.name,
            input_key=_call_address(request),
            ordinal=request.ordinal,
            tool_use_id=request.tool_use_id,
        )
        records.append(entry)
        self._spawn(self._run(request, tool, arguments, entry))

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro, context=self._context.copy())
        self._tasks[task] = None
        task.add_done_callback(lambda done: self._tasks.pop(done, None))

    def close(self) -> None:
        """End the turn's bridge: wake its poll and cancel calls still running (a cancelled
        turn's tool calls, which the Activity has already answered with errors).
        """
        self.closed = True
        for task in self._tasks:
            task.cancel()

    def _reject(self, request: ClaudeToolRequest, reason: str) -> None:
        self.results.append(
            ClaudeToolResult(
                tool_use_id=request.tool_use_id, content=reason, is_error=True
            )
        )

    async def _serve(self, request: ClaudeToolRequest, entry: ClaudeToolRecord) -> None:
        """Answer a call a retried attempt re-issued with the result of the call it repeats.

        That call can still be running: the worker waiting on it died, and the resumed CLI
        asked again before it finished. Waiting makes that an attach, not a second execution.
        A recorded result is final, errors included (the tool's own retry policy has run).
        """
        await workflow.wait_condition(lambda: entry.result is not None or self.closed)
        result = entry.result
        if result is None or self.closed:
            return
        self.results.append(
            result.model_copy(update={"tool_use_id": request.tool_use_id})
        )

    async def _run(
        self,
        request: ClaudeToolRequest,
        tool: Callable[..., Awaitable[Any]],
        arguments: dict[str, Any],
        entry: ClaudeToolRecord,
    ) -> None:
        try:
            output = await tool(**arguments)
            result = ClaudeToolResult(
                tool_use_id=request.tool_use_id, content=_render(output)
            )
        except Exception as exc:  # noqa: BLE001 - a denied/failed tool is a model-visible result
            result = ClaudeToolResult(
                tool_use_id=request.tool_use_id,
                content=f"Tool {request.name!r} failed: {_failure_message(exc)}",
                is_error=True,
            )
        entry.result = result
        if not self.closed:
            self.results.append(result)


class _BridgeHub:
    """The per-workflow-run owner of the bridge signal and poll update, and of the progress
    signal turn Activities send about the sessions whose turns are running.
    """

    def __init__(self) -> None:
        self.bridges: dict[str, _ToolBridge] = {}
        self.sessions: dict[str, ClaudeSession] = {}

    @classmethod
    def current(cls) -> _BridgeHub:
        hub = getattr(workflow.get_update_handler(BRIDGE_POLL_UPDATE), "__self__", None)
        if isinstance(hub, cls):
            return hub
        hub = cls()
        workflow.set_signal_handler(BRIDGE_REQUEST_SIGNAL, hub.on_request)
        workflow.set_update_handler(
            BRIDGE_POLL_UPDATE, hub.on_poll, validator=hub.validate_poll
        )
        workflow.set_signal_handler(TURN_PROGRESS_SIGNAL, hub.on_progress)
        return hub

    def on_progress(self, progress: ClaudeTurnProgress) -> None:
        """Record what a running turn reported, unless it is from an attempt the turn has since
        replaced: a zombie must not rewind the revision or the ledger the live attempt built on.
        After the turn, its result or error carried the same.
        """
        session = self.sessions.get(progress.session_id)
        if session is None:
            workflow.logger.warning(
                "Dropping Claude turn progress for session %s, which runs no turn",
                progress.session_id,
            )
            return
        if progress.attempt_id != session.attempt_id:
            workflow.logger.warning(
                "Dropping Claude turn progress from superseded attempt %s of session %s",
                progress.attempt_id,
                progress.session_id,
            )
            return
        _record_progress(session, progress.native_calls, progress.workspace_revision)
        _record_mirror_errors(session, progress.mirror_errors)
        # Only a cancelled attempt sends a cost; it returns no result that would charge it.
        _charge(session, progress.cost_usd, progress.cumulative_cost, completed=False)

    def on_request(self, request: ClaudeToolRequest) -> None:
        bridge = self.bridges.get(request.token)
        if bridge is None:
            # Its turn is over (or never existed); that Activity's poll is rejected, which
            # answers the call with an error on the Activity side.
            workflow.logger.warning(
                "Dropping Claude tool request %s for an inactive bridge",
                request.tool_use_id,
            )
            return
        bridge.submit(request)

    def validate_poll(self, poll: ClaudeBridgePoll) -> None:
        if poll.token not in self.bridges:
            raise ApplicationError("Stale or unknown Claude tool bridge token.")

    async def on_poll(self, poll: ClaudeBridgePoll) -> ClaudeBridgeBatch:
        bridge = self.bridges.get(poll.token)
        if bridge is None:
            return ClaudeBridgeBatch(closed=True)
        await workflow.wait_condition(
            lambda: len(bridge.results) > poll.after or bridge.closed
        )
        return ClaudeBridgeBatch(
            results=bridge.results[poll.after :], closed=bridge.closed
        )


@contextmanager
def _open_bridge(
    tools: Sequence[Callable[..., Awaitable[Any]]],
    max_in_flight: int,
    session: ClaudeSession,
    message_id: str,
) -> Iterator[tuple[list[ClaudeToolDeclaration], _ToolBridge | None]]:
    """Register ``tools`` for one turn of message ``message_id`` on ``session``; yield their
    declarations and the turn's bridge, whose call record lives in the session's state.
    """
    if not tools:
        yield [], None
        return
    declarations = [claude_tool_declaration(t) for t in tools]
    names = [d.name for d in declarations]
    if duplicates := sorted({n for n in names if names.count(n) > 1}):
        raise claude_workflow.config_error(
            f"two tools of this turn are named {', '.join(map(repr, duplicates))}"
        )
    hub = _BridgeHub.current()
    bridge = _ToolBridge(
        str(workflow.uuid4()),
        {d.name: t for d, t in zip(declarations, tools, strict=True)},
        max_in_flight,
        session,
        message_id,
    )
    hub.bridges[bridge.token] = bridge
    try:
        yield declarations, bridge
    finally:
        bridge.close()
        del hub.bridges[bridge.token]


def claude_tool_declaration(tool: Callable[..., Any]) -> ClaudeToolDeclaration:
    """Describe an async tool to Claude: name, docstring, and JSON Schema.

    A tool whose parameters have no JSON Schema raises a config error, which fails the
    message rather than the workflow task.
    """
    name = getattr(tool, "__name__", None)
    if not inspect.iscoroutinefunction(tool) or not isinstance(name, str) or not name:
        raise claude_workflow.config_error(
            f"{name or tool!r} is not an async function; pass plain async tools"
        )
    try:
        input_schema = TypeAdapter(tool).json_schema()
    except PydanticUserError as exc:
        raise claude_workflow.config_error(
            f"the parameters of tool {name} have no JSON Schema: {exc}"
        ) from exc
    return ClaudeToolDeclaration(
        name=name,
        description=inspect.getdoc(tool) or "",
        input_schema=input_schema,
    )


def _render(output: Any) -> str:
    return output if isinstance(output, str) else to_json(output).decode()


def _failure_message(exc: BaseException) -> str:
    """The useful text of a tool failure; an ActivityError's own text is only "Activity task
    failed", so its cause's is used instead.
    """
    if isinstance(exc, ActivityError) and exc.cause is not None:
        return _failure_message(exc.cause)
    if isinstance(exc, ApplicationError):
        return exc.message
    return str(exc)


class ClaudeSession:
    """One Claude conversation continued across workflow turns.

    Create it once and pass it to every :func:`run_claude_turn` that should share the
    conversation::

        self._claude = ClaudeSession()
        ...
        reply = await run_claude_turn(message.text, session=self._claude)

    The session id is minted in workflow code (``workflow.uuid4()``) so replay yields the same
    id. Turns run one at a time, queued in arrival order, because two CLI processes must never
    append to one transcript. ``activity_config`` is the default for every turn; a call's own is
    merged over it.

    ``max_cost_usd`` caps the session's total spend across messages: once ``cost_usd`` reaches
    it, each message fails as ``ClaudeTurnBudgetExhausted`` without running, and until then each
    message's ``max_budget_usd`` is lowered to what is left. Claude Code checks the budget
    between model calls, so the message that crosses the cap can overshoot it by one call.

    The session's data lives in :attr:`state` (:class:`ClaudeSessionState`). To carry the
    conversation across continue-as-new, pass :meth:`snapshot` to the new run and rebuild the
    session there with :meth:`restore`.
    """

    def __init__(
        self,
        session_id: str | None = None,
        *,
        started: bool = False,
        activity_config: ActivityConfig | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        """Create a session. See the class docstring for how its fields behave."""
        self.state = ClaudeSessionState(
            session_id=session_id, started=started, max_cost_usd=max_cost_usd
        )
        self.activity_config = activity_config
        # The attempt the workflow is waiting on; a progress signal from any other is dropped.
        self.attempt_id: str | None = None
        self._lock = asyncio.Lock()

    def snapshot(self) -> ClaudeSessionState:
        """The session's state, to pass to a continued-as-new run (call it between turns)."""
        return self.state.model_copy(deep=True)

    @classmethod
    def restore(
        cls, state: ClaudeSessionState, *, activity_config: ActivityConfig | None = None
    ) -> Self:
        """A session continuing the conversation :meth:`snapshot` captured."""
        session = cls(activity_config=activity_config)
        session.state = state.model_copy(deep=True)
        return session


class ClaudeSessionState(BaseModel):
    """A :class:`ClaudeSession` as data, and its continue-as-new input.

    ``activity_config`` is not part of it: the new run passes its own to
    :meth:`ClaudeSession.restore`.

    Attributes:
        session_id: The Claude session UUID; it also names the worker's working directory.
        started: A run on the session completed. A hint: the Activity resumes whenever it
            finds the transcript.
        degraded: Why the ``SessionStore`` is incomplete. While set every turn is refused,
            because resuming from the store would silently lose context; clear it once the
            store is known to be complete.
        native_denied: Denied native calls; the hook refuses repeats of them.
        native_rerequests: Calls made right after denials; the hook caps them.
        cost_usd: The session's spend: each run's own cost as Claude Code reported it, plus what
            lost attempts reported before they were lost.
        restored_cost_usd: The cost Claude Code restores when it resumes the session from its
            own transcript (on the worker or in a sandbox snapshot): the last completed run's
            total. Subtracted from a run that reports that cumulative total.
        max_cost_usd: The cap on ``cost_usd``; raise it here to let a capped session go on.
        native_unresolved: A native call a turn left deferred; the next run settles it as
            expired, not run, because Claude Code will not take a new prompt until it is.
        native_calls: The started-call ledger. Each run merges it into the worker's ledger, so
            a call started on a lost worker is never run again.
        workspace_revision: The ``WorkspaceStore`` revision the next run restores.
        sandbox_snapshot: The remote ``SandboxRuntime`` snapshot the next attempt starts from.
        tool_record: The latest tool turn's calls, which answer a retried attempt's re-issued
            calls.
    """

    version: Literal[1] = 1
    session_id: str | None = None
    started: bool = False
    degraded: str | None = None
    native_denied: list[ClaudeDeferredTool] = []
    native_rerequests: int = 0
    cost_usd: float = 0.0
    restored_cost_usd: float = 0.0
    native_unresolved: ClaudeDeferredTool | None = None
    native_calls: list[ClaudeNativeCall] = []
    workspace_revision: int | None = None
    sandbox_snapshot: str | None = Field(default=None, exclude_if=_omit_if_none)
    tool_record: list[ClaudeToolRecord] = []
    max_cost_usd: float | None = Field(default=None, exclude_if=_omit_if_none)


async def run_claude_turn(
    prompt: str,
    *,
    session: ClaudeSession | None = None,
    options: ClaudeTurnOptions | None = None,
    output_type: type[OutputT] | None = None,
    tools: Sequence[Callable[..., Awaitable[Any]]] = (),
    approve: Approve | None = None,
    max_concurrent_tool_calls: int = 32,
    native_tools: Sequence[str] = (),
    native_auto_allow: Sequence[str] = (),
    activity_config: ActivityConfig | None = None,
    timeout: timedelta | None = None,
    streaming_topic: str | None = None,
    streaming_batch_interval: timedelta = timedelta(milliseconds=100),
    streaming_context: Any = None,
) -> ClaudeTurnResult:
    """Run one Claude turn from a workflow.

    A turn lost with its worker resumes from the plugin's ``session_store`` (or sandbox
    snapshot) when the worker's lease store is shared; tool calls the lost attempt ran are
    answered from the session's record instead of running again. See the README for the full
    durability model.

    Args:
        prompt: The user message.
        session: The conversation to continue; a fresh one-off session when None.
        options: Model, system prompt and per-message caps (``max_turns``, ``max_budget_usd``).
        output_type: Pydantic model Claude's ``structured_output`` is validated into.
        tools: Async functions Claude may call; each call runs in the workflow.
        approve: Decides each deferred native call: ``True``, ``False`` or ``(False, reason)``.
            Absent, every native call is denied.
        max_concurrent_tool_calls: Calls past this are answered with a retryable error.
        native_tools: Claude built-in tools to offer, refused unless the worker opted in.
        native_auto_allow: Read-only native tools that run without asking ``approve``.
        activity_config: Merged over the session's; see :func:`workflow.run_turn`.
        timeout: Bound on the whole message in workflow time, including approval waits.
        streaming_topic: ``WorkflowStream`` topic for advisory, at-least-once
            :class:`ClaudePartialEvent` partials.
        streaming_batch_interval: How long partials are coalesced before the topic or the
            worker's ``partial_sink`` receives them.
        streaming_context: Opaque JSON value the worker's ``partial_sink`` receives with each
            batch of this turn's partials; needs no topic.

    Returns:
        The turn's durable result.
    """
    deadline = _Deadline(timeout)
    session = session or ClaudeSession()
    if streaming_topic is not None and not streaming_topic.strip():
        raise claude_workflow.config_error(
            "streaming_topic must be a non-empty topic name"
        )
    if streaming_batch_interval.total_seconds() < 0:
        raise claude_workflow.config_error(
            "streaming_batch_interval cannot be negative"
        )
    opts = options or ClaudeTurnOptions()
    if output_type is not None:
        if not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):
            raise claude_workflow.config_error(  # type: ignore
                f"output_type must be a pydantic BaseModel subclass, not {output_type!r}"
            )
        try:
            schema = output_type.model_json_schema()
        except PydanticUserError as exc:
            raise claude_workflow.config_error(
                f"output_type {output_type.__name__} has no JSON Schema: {exc}"
            ) from exc
        opts = opts.model_copy(update={"output_schema": schema})
    if native_tools and opts.max_turns is None:
        opts = opts.model_copy(update={"max_turns": NATIVE_DEFAULT_MAX_TURNS})
    await deadline.run(
        session._lock.acquire(),
        on_abandon=lambda taken: taken and session._lock.release(),  # type: ignore
    )
    try:
        if session.state.degraded is not None:
            raise ApplicationError(
                f"Claude session {session.state.session_id} is degraded and will not resume: "
                f"{session.state.degraded}",
                type=CLAUDE_SESSION_DEGRADED_ERROR_TYPE,
                non_retryable=True,
            )
        if session.state.session_id is None:
            session.state.session_id = str(workflow.uuid4())
        if (cap := session.state.max_cost_usd) is not None:
            spent = session.state.cost_usd
            if spent >= cap:
                raise ApplicationError(
                    f"Claude session {session.state.session_id} has spent ${spent} of its "
                    f"max_cost_usd ({cap}); the message was not run.",
                    {"session_cost_usd": spent, "max_cost_usd": cap},
                    type=CLAUDE_TURN_BUDGET_ERROR_TYPE,
                    non_retryable=True,
                )
            if opts.max_budget_usd is None or opts.max_budget_usd > cap - spent:
                opts = opts.model_copy(update={"max_budget_usd": cap - spent})
        budget = _Budget(opts)
        _BridgeHub.current().sessions[session.state.session_id] = session
        message_id = str(workflow.uuid4())
        with _open_bridge(tools, max_concurrent_tool_calls, session, message_id) as (
            declarations,
            bridge,
        ):
            turn = ClaudeTurnInput(
                prompt=prompt,
                prompt_uuid=str(workflow.uuid4()),
                options=opts,
                session_id=session.state.session_id,
                tools=declarations,
                bridge_token=bridge.token if bridge is not None else None,
                native_tools=list(native_tools),
                native_auto_allow=list(native_auto_allow) or None,
                native_denied=list(session.state.native_denied),
                native_rerequests=session.state.native_rerequests,
                native_decision=_settle(session),
                report_progress=True,
                streaming_topic=streaming_topic,
                streaming_batch_interval_ms=(
                    int(streaming_batch_interval.total_seconds() * 1000)
                    if streaming_topic or streaming_context is not None
                    else None
                ),
                streaming_context=streaming_context,
            )
            config = ActivityConfig(
                **{**(session.activity_config or {}), **(activity_config or {})}  # type: ignore
            )
            unresolved = session.state.native_unresolved
            result = await deadline.run(
                _dispatch(session, turn, config, bridge, budget, deadline),
                on_abandon=lambda run: _abandon_run(session, budget, run),
            )
            if session.state.native_unresolved == unresolved:
                session.state.native_unresolved = None
            budget.spend(session, result)
            # Native calls defer one at a time; the session lock stays held from each deferral
            # through its approval to the run that resumes the call.
            while (deferred := result.deferred) is not None:
                resumed = turn.native_decision
                if (
                    resumed is not None
                    and deferred.tool_use_id == resumed.call.tool_use_id
                ):
                    raise ApplicationError(
                        f"Claude deferred the {deferred.name} call {deferred.tool_use_id} "
                        "again instead of applying the decision on it.",
                        type=CLAUDE_NATIVE_TOOL_UNSAFE_ERROR_TYPE,
                        non_retryable=True,
                    )
                if exhausted := budget.exhausted():
                    _deny_abandoned(session, deferred)
                    raise ApplicationError(
                        f"Claude deferred the {deferred.name} call {deferred.tool_use_id} with "
                        f"the turn's {exhausted} used up; it was not resumed.",
                        {"num_turns": budget.turns, "cost_usd": budget.cost_usd},
                        type=CLAUDE_TURN_BUDGET_ERROR_TYPE,
                        non_retryable=True,
                    )
                turn = await deadline.run(
                    _after_deferral(approve, session, deferred, turn),
                    on_abandon=lambda _: _deny_abandoned(session, deferred),
                )
                result = await deadline.run(
                    _dispatch(
                        session,
                        turn.model_copy(update={"options": budget.left()}),
                        config,
                        bridge,
                        budget,
                        deadline,
                    ),
                    on_abandon=lambda run: _abandon_run(session, budget, run),
                )
                budget.spend(session, result)
        if output_type is not None:
            result = result.model_copy(
                update={"output": validate_structured_output(output_type, result)}
            )
    except BaseException as exc:
        cancelled = isinstance(exc, asyncio.CancelledError | CancelledError)
        _count_turn(opts, "cancelled" if cancelled else "failure")
        raise
    finally:
        if session.state.session_id is not None:
            _BridgeHub.current().sessions.pop(session.state.session_id, None)
        session._lock.release()
    _count_turn(opts, "refusal" if result.stop_reason == "refusal" else "success")
    return result


def _count_turn(
    opts: ClaudeTurnOptions,
    outcome: Literal["success", "refusal", "failure", "cancelled"],
) -> None:
    """Count one settled message on the workflow's replay-safe meter."""
    workflow.metric_meter().create_counter(
        "claude_turns", "Claude messages settled, by outcome and model."
    ).add(1, {"outcome": outcome, "model": opts.model or "default"})


async def _dispatch(
    session: ClaudeSession,
    turn: ClaudeTurnInput,
    activity_config: ActivityConfig,
    bridge: _ToolBridge | None = None,
    budget: _Budget | None = None,
    deadline: _Deadline | None = None,
) -> ClaudeTurnResult:
    """Run one turn Activity on ``session``, retrying after a lost attempt or a transient API error.

    A lost attempt that may have run a non-repeatable approved native call is not resumed.
    """
    resumes_lost = resumes_api_error = False
    lost_lease = None
    api_retries = 0
    # The message's earlier runs made these calls; every attempt of this run numbers on from them.
    ordinals: dict[str, int] = {}
    for record in session.state.tool_record if bridge is not None else ():
        ordinals[record.input_key] = max(
            ordinals.get(record.input_key, 0), record.ordinal + 1
        )
    if ordinals:
        turn = turn.model_copy(update={"tool_ordinals": ordinals})
    for attempt in range(1, _RESUME_ATTEMPTS + 1):
        this = _for_attempt(
            turn,
            attempt,
            bridge,
            budget,
            resumes_lost=resumes_lost,
            resumes_api_error=resumes_api_error,
        )
        if lost_lease is not None:
            this = this.model_copy(update={"lost_lease": tuple(lost_lease)})
        try:
            result = await _attempt(session, this, activity_config)
            if result.sandbox_violations:
                workflow.metric_meter().create_counter(
                    "claude_sandbox_violations",
                    "Sandbox violations Claude Code reported in tool results, by model.",
                ).add(
                    result.sandbox_violations,
                    {"model": turn.options.model or "default"},
                )
            return result
        except (ActivityError, ApplicationError) as exc:
            lost = _lost_attempt(exc)
            # Only a heartbeat timeout proves the attempt's lease went a whole TTL unrenewed.
            lost_lease = (_lost_details(exc) or {}).get("lease")
            call = None
            if lost is not None:
                _record_progress(
                    session, lost.get("native_calls"), lost.get("workspace_revision")
                )
                _record_snapshot(session, lost.get("sandbox_snapshot"))
                _record_mirror_errors(session, lost.get("mirror_errors"))
                if budget is not None:
                    budget.turns += lost.get("num_turns") or 0
                    cost = lost.get("cost_usd")
                    if isinstance(cost, int | float):
                        budget.spend(
                            session,
                            ClaudeTurnResult(
                                total_cost_usd=float(cost),
                                cumulative_cost=bool(lost.get("cumulative_cost")),
                            ),
                            completed=False,
                        )
                call = _unsettled_native_call(session, turn)
            if (
                call is not None
                and native_tool_effect(call.name) not in REPEATABLE_EFFECTS
            ):
                _outcome_unknown(session, call)
                raise ApplicationError(
                    f"Claude session {session.state.session_id} lost the attempt running the approved "
                    f"{call.name} call {call.tool_use_id}, and nothing recorded whether it ran; "
                    "it was not resumed and will not be run again unasked.",
                    {
                        "session_id": session.state.session_id,
                        "tool_use_id": call.tool_use_id,
                    },
                    type=CLAUDE_NATIVE_OUTCOME_UNKNOWN_ERROR_TYPE,
                    non_retryable=True,
                ) from exc
            if lost is not None and session.state.degraded is not None:
                raise ApplicationError(
                    f"Claude session {session.state.session_id} lost an attempt after the "
                    f"SessionStore dropped part of the turn ({session.state.degraded}); it was not "
                    "resumed from the incomplete store, and the session will not resume.",
                    {
                        "session_id": session.state.session_id,
                        "mirror_errors": lost.get("mirror_errors"),
                    },
                    type=CLAUDE_SESSION_DEGRADED_ERROR_TYPE,
                    non_retryable=True,
                ) from exc
            if (
                lost is None
                and budget is not None
                and isinstance(exc, ApplicationError)
                and exc.details
                and isinstance(exc.details[0], dict)
            ):
                budget.spend(session, _spent(exc))  # a failed run spent too
            delay = (
                None
                if lost is not None
                else _api_error_delay(session, turn, exc, api_retries)
            )
            if (
                attempt == _RESUME_ATTEMPTS
                or (deadline is not None and deadline.abandoned)
                or not (delay is not None or _resumable(session, exc))
            ):
                raise
            if budget is not None and (exhausted := budget.exhausted()):
                raise ApplicationError(
                    f"Claude {'failed' if delay is not None else 'lost'} an attempt of this turn "
                    f"with the turn's {exhausted} used up; it was not resumed.",
                    {"num_turns": budget.turns, "cost_usd": budget.cost_usd},
                    type=CLAUDE_TURN_BUDGET_ERROR_TYPE,
                    non_retryable=True,
                ) from exc
            if (
                delay is not None
                and deadline is not None
                and not deadline.allows(delay)
            ):
                raise
            reason = _failure_message(exc)
            workflow.logger.warning(
                "Claude session %s %s (%s); resuming it%s",
                session.state.session_id,
                "hit a transient API error" if delay is not None else "lost a turn",
                reason,
                f" in {delay}" if delay is not None else "",
            )
            if bridge is not None:
                bridge.rotate()
            # No backoff after a lost attempt: the retry's Activity waits out the lost attempt's
            # lease itself (claiming it at once if it is still at the beat the last heartbeat
            # reported, else watching it for a whole TTL), so a workflow timer would only add
            # latency.
            if delay is not None:
                await workflow.sleep(delay, summary="Claude API error backoff")
                api_retries += 1
            # Only a worker that stopped heartbeating may leave its attempt running. A refusal
            # to start beside a still-held lease leaves that possibility as it was.
            resumes_lost = _lost_details(exc) is not None or (
                resumes_lost
                and isinstance(exc, ApplicationError)
                and exc.type == CLAUDE_SESSION_LEASED_ERROR_TYPE
            )
            resumes_api_error = delay is not None
    raise AssertionError("unreachable: the loop returns or raises")


_API_RETRY_BASE_DELAY = timedelta(seconds=30)
"""The wait before resuming after the first transient API error, doubled for each one after it.
Claude Code has already retried the request itself (10 times by default, backing off to 32 s;
https://code.claude.com/docs/en/env-vars), so the API has been failing for minutes."""

_API_RETRY_MAX_DELAY = timedelta(minutes=5)
"""The longest wait before such a resume. A rate-limit reset further away fails the message
instead of holding the session (and its lock) for it."""


def _api_error_delay(
    session: ClaudeSession, turn: ClaudeTurnInput, exc: BaseException, retries: int
) -> timedelta | None:
    """The backoff before resuming a run that ended on a transient API error, else None."""
    if not isinstance(exc, ApplicationError) or exc.type != CLAUDE_RESULT_ERROR_TYPE:
        return None
    details = exc.details[0] if exc.details and isinstance(exc.details[0], dict) else {}
    if (
        not details.get("session_store")
        or session.state.degraded
        or session.state.session_id is None
    ):
        return None
    status = details.get("api_error_status")
    kind = details.get("error_kind")
    if isinstance(status, int):
        transient = status == 429 or 500 <= status <= 599
    else:
        transient = details.get("terminal_reason") == "api_error" or kind in (
            "rate_limit",
            "server_error",
        )
    if (
        not transient
        or kind in NEVER_RESUMED_ERROR_KINDS
        or _unsettled_native_call(session, turn) is not None
    ):
        return None
    delay = _API_RETRY_BASE_DELAY * 2**retries
    resets_at = details.get("rate_limit_resets_at")
    if isinstance(resets_at, int | float):
        delay = max(delay, timedelta(seconds=resets_at - workflow.now().timestamp()))
    return delay if delay <= _API_RETRY_MAX_DELAY else None


def _spent(exc: ApplicationError) -> ClaudeTurnResult:
    """What a failed run reports it spent, as a result the budget can charge: its result's
    cost, else what the attempt had reported before it failed.
    """
    details = exc.details[0]
    total = details.get("total_cost_usd")
    return ClaudeTurnResult(
        num_turns=details.get("num_turns") or 0,
        total_cost_usd=total if total is not None else details.get("cost_usd") or None,
        cumulative_cost=bool(details.get("cumulative_cost")),
    )


def _lost_details(exc: BaseException) -> dict[str, Any] | None:
    """The last heartbeat details of an attempt whose worker stopped heartbeating, or None for
    any other failure (or a worker that died before its first heartbeat).
    """
    cause = exc.cause if isinstance(exc, ActivityError) else None
    if (
        not isinstance(cause, TemporalTimeoutError)
        or cause.type != TimeoutType.HEARTBEAT
    ):
        return None  # a start-to-close timeout is the turn's own time running out
    details = cause.last_heartbeat_details
    return details[0] if details and isinstance(details[0], dict) else None


def _resumable(session: ClaudeSession, exc: BaseException) -> bool:
    """Whether a lost attempt can be run again, picking up what it wrote to the store."""
    if session.state.degraded is not None or session.state.session_id is None:
        return False
    if isinstance(exc, CancelledError) or isinstance(exc.cause, CancelledError):  # type: ignore
        return False  # the turn was cancelled on purpose; resuming it would undo that
    if isinstance(exc, ApplicationError):
        # Rejected before the CLI ran, so nothing happened. A remote sandbox killed from outside
        # leaves the last committed snapshot. A CLI that exited without a result on a resumable
        # attempt is unfinished, not an answer. Any other failure is the outcome.
        if exc.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE and not exc.non_retryable:
            # Failover needs a shared lease. A host-local store leaves this False.
            details = (
                exc.details[0]
                if exc.details and isinstance(exc.details[0], dict)
                else None
            )
            return bool(details and details.get(RESUMABLE_DETAIL) is True)
        if exc.type in _BEFORE_THE_CLI and not exc.non_retryable:
            return True
        return _cli_died(exc) is not None
    details = _lost_details(exc)
    return details is not None and details.get(RESUMABLE_DETAIL) is True


def _sandbox_lost_details(exc: BaseException) -> dict[str, Any] | None:
    """Details of a turn whose sandbox was lost, else None."""
    if (
        not isinstance(exc, ApplicationError)
        or exc.type != CLAUDE_SANDBOX_LOST_ERROR_TYPE
    ):
        return None
    return exc.details[0] if exc.details and isinstance(exc.details[0], dict) else None


def _lost_attempt(exc: BaseException) -> dict[str, Any] | None:
    """Details of an attempt the session may resume, or None when the failure is the outcome."""
    return _lost_details(exc) or _cli_died(exc) or _sandbox_lost_details(exc)


def _cli_died(exc: BaseException) -> dict[str, Any] | None:
    """Details of a resumable attempt whose CLI exited without a result, else None.

    A result the CLI did produce (including a transient API error) carries ``terminal_reason``,
    ``subtype`` or ``api_error_status`` and is not this: that result is the turn's outcome.
    """
    if not isinstance(exc, ApplicationError) or exc.type != CLAUDE_RESULT_ERROR_TYPE:
        return None
    details = (
        exc.details[0] if exc.details and isinstance(exc.details[0], dict) else None
    )
    if not details or details.get(RESUMABLE_DETAIL) is not True:
        return None
    if (
        details.get("terminal_reason")
        or details.get("subtype")
        or details.get("api_error_status") is not None
    ):
        return None
    return details


def _for_attempt(
    turn: ClaudeTurnInput,
    attempt: int,
    bridge: _ToolBridge | None,
    budget: _Budget | None = None,
    *,
    resumes_lost: bool = False,
    resumes_api_error: bool = False,
) -> ClaudeTurnInput:
    """``turn`` as attempt number ``attempt`` sends it: its own attempt id and bridge token, the
    mark that it resumes a lost attempt that may still be running (the Activity refuses that
    without a lease store every worker shares; a run that ended on an API error is not running)
    or a run that ended on an API error, the message's call record so far, and from the second
    attempt on what the earlier attempts left of the budget.
    """
    update: dict[str, Any] = {
        "attempt": attempt,
        "resumes_lost_attempt": resumes_lost,
        "resumes_api_error": resumes_api_error,
    }
    if attempt > 1 and budget is not None:
        update["options"] = budget.left()
    if bridge is not None:
        update["bridge_token"] = bridge.token
        if calls := recorded_tool_calls(bridge.records):
            update["tool_calls"] = calls
    return turn.model_copy(update=update)


def _record_progress(
    session: ClaudeSession,
    calls: Sequence[ClaudeNativeCall | Mapping[str, Any]] | None,
    workspace_revision: int | None,
) -> None:
    """Fold what a turn Activity reported into the session: ledger entries by their later
    status (a stale report never reopens a settled call), and a newer workspace revision.
    """
    if calls:
        merged = [c.model_dump() for c in session.state.native_calls]
        merge_native_calls(
            merged,
            [c.model_dump() if isinstance(c, BaseModel) else dict(c) for c in calls],
        )
        session.state.native_calls = [
            ClaudeNativeCall.model_validate(c) for c in merged
        ]
    if workspace_revision is not None and workspace_revision > (
        session.state.workspace_revision or 0
    ):
        session.state.workspace_revision = workspace_revision


def _record_snapshot(session: ClaudeSession, snapshot: Any) -> None:
    """Keep a newly committed remote snapshot. A missing or empty id leaves the previous one.

    A lost attempt reports no new snapshot. Replacing the committed id with that would resume
    from nothing.
    """
    if isinstance(snapshot, str) and snapshot:
        session.state.sandbox_snapshot = snapshot


def _record_mirror_errors(session: ClaudeSession, errors: Sequence[str] | None) -> None:
    """Mark the session degraded when a turn reported batches its ``SessionStore`` dropped."""
    if errors:
        session.state.degraded = "; ".join(errors)


def _ledger_status(session: ClaudeSession, tool_use_id: str) -> str | None:
    return next(
        (c.status for c in session.state.native_calls if c.tool_use_id == tool_use_id),
        None,
    )


def _unsettled_native_call(
    session: ClaudeSession, turn: ClaudeTurnInput
) -> ClaudeDeferredTool | None:
    """The approved native call ``turn`` resumes, unless the ledger shows it finished or failed:
    the one native effect a lost attempt of ``turn`` may have had without anyone learning.
    """
    decision = turn.native_decision
    if decision is None or not decision.approved:
        return None
    if _ledger_status(session, decision.call.tool_use_id) in ("finished", "failed"):
        return None
    return decision.call


def _outcome_unknown(session: ClaudeSession, call: ClaudeDeferredTool) -> None:
    """Record ``call`` as possibly run: ``started`` in the session's ledger (added if the lost
    worker never reported it), so the hook on any worker refuses its id and denies a repeat of
    it once with ``UNKNOWN_OUTCOME``, and the next run settles it with that reason.
    """
    if _ledger_status(session, call.tool_use_id) is None:
        session.state.native_calls = [
            *session.state.native_calls,
            ClaudeNativeCall(
                tool_use_id=call.tool_use_id,
                tool_name=call.name,
                key=call_key(call.name, call.input),
                status="started",
            ),
        ]
    session.state.native_unresolved = call


async def _attempt(
    session: ClaudeSession, turn: ClaudeTurnInput, activity_config: ActivityConfig
) -> ClaudeTurnResult:
    """Run the turn Activity once. The session's state as it stands now goes with it:
    ``resume`` (whether a run on the session completed), the ledger and the workspace revision.
    What the Activity reports back, result or failure, is recorded on the session.
    """
    session.attempt_id = turn.attempt_id
    try:
        result = await claude_workflow.run_turn(
            turn.model_copy(
                update={
                    "resume": session.state.started,
                    "native_calls": list(session.state.native_calls),
                    "workspace_revision": session.state.workspace_revision,
                    "sandbox_snapshot": session.state.sandbox_snapshot,
                }
            ),
            activity_config=activity_config,
        )
    except ApplicationError as exc:
        details = exc.details[0] if exc.details else None
        if isinstance(details, dict):
            session.state.native_rerequests += details.get("native_rerequests") or 0
            revision = details.get("workspace_revision")
            _record_progress(session, details.get("native_calls"), revision)
            _record_snapshot(session, details.get("sandbox_snapshot"))
            _record_mirror_errors(session, details.get("mirror_errors"))
        raise
    session.state.started = True
    session.state.native_rerequests += result.native_rerequests
    _record_progress(session, result.native_calls, result.workspace_revision)
    _record_snapshot(session, result.sandbox_snapshot)
    if result.mirror_errors:
        session.state.degraded = "; ".join(result.mirror_errors)
        raise ApplicationError(
            f"Claude session {session.state.session_id}: the SessionStore did not persist the "
            f"whole turn ({session.state.degraded}); the session will not resume elsewhere.",
            {
                "session_id": session.state.session_id,
                "mirror_errors": result.mirror_errors,
                # The run completed, and the session pays for it.
                "num_turns": result.num_turns,
                "total_cost_usd": result.total_cost_usd,
                "cumulative_cost": result.cumulative_cost,
            },
            type=CLAUDE_SESSION_MIRROR_ERROR_TYPE,
            non_retryable=True,
        )
    return result


class _Deadline:
    """One message's ``timeout``, enforced in workflow code (``workflow.now()`` and a timer).

    Each step runs as its own task while the turn waits on it. Temporal shields an awaited
    Activity from task cancellation and returns its result when the Activity completes before
    hearing of the cancel, which would silently drop a cancellation (or a timeout) of the turn.
    Waiting on the step instead keeps both with the turn: on either, the step is cancelled and
    waited for (an Activity until its CLI is gone), ``on_abandon`` gets its result if it had one
    anyway (else None), and the turn ends cancelled or timed out.
    """

    def __init__(self, timeout: timedelta | None) -> None:
        self._timeout = timeout
        self._at: datetime | None = (
            workflow.now() + timeout if timeout is not None else None
        )
        # Temporal swallows the step's task cancellation while it waits out the Activity's,
        # so the step reads this to start no further attempt.
        self.abandoned = False

    async def run(
        self, step: Coroutine[Any, Any, T], *, on_abandon: Callable[[T | None], None]
    ) -> T:
        if self._at is not None and workflow.now() >= self._at:
            step.close()
            on_abandon(None)
            raise self._expired()
        task = asyncio.create_task(step)
        try:
            await workflow.wait_condition(
                task.done,
                timeout=None if self._at is None else self._at - workflow.now(),
                timeout_summary="Claude turn timeout",
            )
        except BaseException as exc:
            self.abandoned = True
            # This activation can carry the step's result along with the timeout or the
            # cancellation. Conditions are checked only once nothing else is ready, so the
            # step takes that result before it is cancelled.
            await workflow.wait_condition(lambda: True)
            task.cancel()
            await asyncio.wait([task])
            on_abandon(
                task.result()
                if not task.cancelled() and task.exception() is None
                else None
            )
            if isinstance(exc, TimeoutError):
                raise self._expired() from None
            raise
        # One activation can hold the step's result and a deadline that has already
        # passed: a time jump, or a workflow task delayed past both. Drop the result.
        if self._at is not None and workflow.now() >= self._at:
            on_abandon(
                task.result()
                if not task.cancelled() and task.exception() is None
                else None
            )
            raise self._expired()
        return task.result()

    def allows(self, wait: timedelta) -> bool:
        """Whether waiting ``wait`` from now still ends before the deadline."""
        return self._at is None or workflow.now() + wait < self._at

    def _expired(self) -> ApplicationError:
        return ApplicationError(
            f"The Claude turn reached its {self._timeout} timeout.",
            type=CLAUDE_TURN_TIMEOUT_ERROR_TYPE,
            non_retryable=True,
        )


class _Budget:
    """One message's ``max_turns`` and ``max_budget_usd``, spent across the runs of its native
    defer -> approve -> resume chain (the CLI applies each only to the run it is given).
    """

    def __init__(self, options: ClaudeTurnOptions) -> None:
        self._options = options
        self.turns = 0
        self.cost_usd = 0.0

    def spend(
        self,
        session: ClaudeSession,
        result: ClaudeTurnResult,
        *,
        completed: bool = True,
    ) -> None:
        """Charge one run. ``completed``: its CLI exited on its own, recording its total in
        the transcript for the next resume to restore; a lost attempt's did not.
        """
        self.turns += result.num_turns
        self.cost_usd += _charge(
            session, result.total_cost_usd, result.cumulative_cost, completed=completed
        )

    def exhausted(self) -> str | None:
        """Which cap the runs so far have used up, if any."""
        opts = self._options
        if opts.max_turns is not None and self.turns >= opts.max_turns:
            return f"max_turns ({opts.max_turns})"
        if opts.max_budget_usd is not None and self.cost_usd >= opts.max_budget_usd:
            return f"max_budget_usd ({opts.max_budget_usd})"
        return None

    def left(self) -> ClaudeTurnOptions:
        """The options for the next run: what the runs so far left of each cap."""
        opts = self._options
        update: dict[str, Any] = {}
        if opts.max_turns is not None:
            update["max_turns"] = opts.max_turns - self.turns
        if opts.max_budget_usd is not None:
            update["max_budget_usd"] = opts.max_budget_usd - self.cost_usd
        return opts.model_copy(update=update)


def _charge(
    session: ClaudeSession,
    total: float | None,
    cumulative: bool,
    *,
    completed: bool,
) -> float:
    """Add one run's own cost to the session's spend and return it. ``total`` is what Claude
    Code reported; ``cumulative`` totals include the cost it restored on resume.
    """
    if total is None:
        return 0.0
    restored = session.state.restored_cost_usd
    cost = total - restored if cumulative and total >= restored else total
    session.state.cost_usd += cost
    if completed:
        session.state.restored_cost_usd = total
    return cost


def _abandon_run(
    session: ClaudeSession, budget: _Budget, run: ClaudeTurnResult | None
) -> None:
    """A run the turn gives up on (cancelled, or past its timeout) that still finished: its
    answer is in the transcript, so it is paid for, and a call it deferred stays unresolved.
    """
    if run is not None:
        budget.spend(session, run)
        _deny_abandoned(session, run.deferred)


def _deny_abandoned(
    session: ClaudeSession, deferred: ClaudeDeferredTool | None
) -> None:
    """Record a deferred call the turn abandons undecided. The transcript still holds it deferred
    and the CLI re-presents it on the session's next resume, where the next run's decision denies
    it as expired (``NATIVE_ABANDONED_REASON``) instead of it coming back as a new approval. No
    one denied it, so it stays out of ``native_denied``: an identical call later goes through
    approval like any other.
    """
    if deferred is None:
        return
    session.state.native_unresolved = deferred


NATIVE_ABANDONED_REASON = (
    "Approval expired: this call was never approved or denied, and it did not run. Do not retry "
    "it or mention it; answer the user's next message."
)
"""What Claude is told about a call an earlier message left deferred, when the next run settles
it."""


def _settle(session: ClaudeSession) -> ClaudeNativeDecision | None:
    """The denial settling the call an earlier message left unresolved: expired, or, when the
    ledger says it may have run on a lost worker, outcome unknown.
    """
    call = session.state.native_unresolved
    if call is None:
        return None
    unknown = _ledger_status(session, call.tool_use_id) in ("started", "reported")
    reason = UNKNOWN_OUTCOME if unknown else NATIVE_ABANDONED_REASON
    return ClaudeNativeDecision(call=call, approved=False, reason=reason, settle=True)


RESUME_PROMPT = "Continue."
"""The prompt that resumes a session after a decision on its deferred native call. Claude Code
re-presents the deferred call to ``PreToolUse`` before the model sees this prompt."""

_BEFORE_THE_CLI = frozenset(
    {
        CLAUDE_TURN_REJECTED_ERROR_TYPE,
        CLAUDE_SESSION_LEASED_ERROR_TYPE,
        CLAUDE_SESSION_LEASE_LOST_ERROR_TYPE,
    }
)
"""Rejections raised before the CLI could run, so nothing happened and a retryable one may simply
be tried again."""

_RESUME_ATTEMPTS = 3
"""How many times one message's turn may be run before it fails: the first attempt plus the
retries that follow a lost worker. Each retry costs the interrupted round, not the message."""


NATIVE_DENY_REASON = "Denied by policy; do not retry this command. Tell the user it was not run and stop."
"""What Claude is told when a native call is denied. Live runs on haiku made no retry after it in
6 of 6 denials, against up to 2 after a bare "denied"; the hook's repeat guard and re-request cap
stop whatever retries remain."""


async def _after_deferral(
    approve: Approve | None,
    session: ClaudeSession,
    deferred: ClaudeDeferredTool,
    turn: ClaudeTurnInput,
) -> ClaudeTurnInput:
    """Ask ``approve`` about a deferred native call and resume the session with the decision.

    Absent ``approve``, or a callback that returns false or raises, denies the call. Both
    outcomes keep the transcript: the resumed run's hook allows the call, or denies it with the
    callback's ``(False, reason)`` reason, else ``NATIVE_DENY_REASON``. A denied call joins the
    session's denied list.
    """
    approved, reason = False, None
    if approve is not None:
        try:
            verdict = await approve(deferred)
            approved, reason = (
                verdict if isinstance(verdict, tuple) else (verdict, None)
            )
            approved = bool(approved)
        except Exception:  # noqa: BLE001 - a failing gate is a denial
            approved, reason = False, None
    if approved:
        decision = ClaudeNativeDecision(call=deferred, approved=True)
    else:
        session.state.native_denied.append(deferred)
        decision = ClaudeNativeDecision(
            call=deferred, approved=False, reason=reason or NATIVE_DENY_REASON
        )
    return turn.model_copy(
        update={
            "prompt": RESUME_PROMPT,
            "prompt_uuid": str(workflow.uuid4()),
            "native_decision": decision,
            "native_denied": list(session.state.native_denied),
            "native_rerequests": session.state.native_rerequests,
        }
    )


def validate_structured_output(
    output_type: type[OutputT], result: ClaudeTurnResult
) -> OutputT:
    """Validate ``result.structured_output`` into ``output_type``, or fail the message.

    Raises a non-retryable ``ApplicationError`` (failing the handler, not the workflow task) when
    Claude returned no structured output or one that does not match the model.
    """
    if result.structured_output is None:
        raise ApplicationError(
            f"Claude returned no structured output for {output_type.__name__}.",
            type=CLAUDE_STRUCTURED_OUTPUT_ERROR_TYPE,
            non_retryable=True,
        )
    try:
        return output_type.model_validate(result.structured_output)
    except ValidationError as exc:
        raise ApplicationError(
            f"Claude's structured output does not match {output_type.__name__}: {exc}",
            type=CLAUDE_STRUCTURED_OUTPUT_ERROR_TYPE,
            non_retryable=True,
        ) from exc
