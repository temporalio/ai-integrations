"""Workflow side: the durable agent loop. Everything here must stay deterministic."""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, NoReturn, cast

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.exceptions import ActivityError, ApplicationError, CancelledError

from ._conversation import PAGE_BYTES, PAYLOAD_LIMIT_BYTES, QUERY, entry_text, page
from ._events import TOPIC, cap_event
from ._models import (
    AgentState,
    ConversationRef,
    DeferredCall,
    NativeCallState,
    NativeRequest,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolSpec,
    ToolStepInput,
)

SEGMENT_ACTIVITY_NAME = "run_claude_segment"
"""Name of the Activity that runs one model segment."""

TOOL_STEP_ACTIVITY_NAME = "run_claude_tool_step"
"""Name of the Activity that runs one Claude Code tool call (a tool step)."""

NATIVE_UPDATE = "__claude_agent_native_tool"
"""Internal Update for registering attempts and running native child calls."""

_OPEN_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}
_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,50}")
"""Tool names Claude Code keeps as they are: ``mcp__durable__<name>`` fits in 64 characters."""
_RECENT_CALLS = 256  # tool calls remembered across runs for the run-once guard
_HANDOVER_BYTES = 1536 * 1024  # live output's share keeps the new run's input small
_STREAM_ITEM_BYTES = 34
"""The JSON around one event the new run's input carries: ``{"data":"","offset":0,
"topic":""}`` and a comma (measured with the default converter)."""
_CHECKS_PATCH = "temporalio-claude-agent-sdk-continue-as-new-checks"
"""Patch ID of the Continue-As-New checks added after the first published version
(766c647): Workflows that version started keep their decisions on replay."""
_CARRY_PATCH = "temporalio-claude-agent-sdk-measured-stream-carry"
"""Patch ID of the check of the new run's input with the live output stream in it."""
_HISTORY_EVENTS = 51_200
_HISTORY_BYTES = 50 * 1024 * 1024
"""Temporal's default limits for one run's history: the server ends a run past them."""
_ROOM_EVENTS = 500
_ROOM_BYTES = 3 * PAYLOAD_LIMIT_BYTES
"""Least room kept for the next step (a result near the payload limit is stored three
times), or twice the largest step so far, whichever is more."""


def _is_cancellation(err: BaseException) -> bool:
    """Whether ``err`` means the Workflow (or the awaited Activity) was cancelled."""
    if isinstance(err, asyncio.CancelledError):
        return True
    return isinstance(err, ActivityError) and isinstance(err.cause, CancelledError)


def _event_bytes(item: Any) -> int:
    """A live output event's size in the new run's input (a stream log item)."""
    return len(item.data) + len(item.topic) + _STREAM_ITEM_BYTES


_ENGINE_ACTIVITY_TOOLS = ("Bash", "PowerShell")
"""Claude Code built-in tools that can run as their own Activities (and MCP tools)."""


def _check_tool_activities(patterns: Sequence[str], approvals: Sequence[str]) -> None:
    """Refuse Claude Code tools that cannot run as their own Activities.

    A tool step answers the engine's model calls with a stand-in, and Claude Code
    checks a file tool call again when the next segment delivers its result: an Edit
    run in a tool step happens, but Claude is told it failed, because the file changed
    since Claude read it (tested). So file tools, tools that call the model themselves
    (such as WebFetch) and subagents stay in the segment.
    """
    for pattern in patterns:
        if pattern not in _ENGINE_ACTIVITY_TOOLS and not pattern.startswith("mcp__"):
            raise ValueError(
                f"tool_activities: {pattern!r} cannot run as its own Activity. Use "
                "'Bash', 'PowerShell', or MCP tool names ('mcp__<server>__<tool>', "
                "patterns allowed); other Claude Code tools stay in the segment."
            )
    for pattern in approvals:
        if not any(fnmatch.fnmatchcase(pattern, p) for p in patterns):
            raise ValueError(
                f"tool_approvals: {pattern!r} does not run as its own Activity, so it "
                "could not wait for a decision. Add it to tool_activities."
            )


def _check_tool_name(name: str) -> None:
    if not _TOOL_NAME.fullmatch(name):
        raise ValueError(
            f"Tool name {name!r}: use 1 to 50 letters, digits, '_' or '-'. Claude Code "
            "renames other tool names, and the call could never run."
        )


class _Conversations:
    """The agents of one Workflow whose conversation the Query serves.

    One per Workflow instance: it lives on the Query handler itself, so it is never
    shared between Workflows that run in the same Worker process.
    """

    def __init__(self) -> None:
        self.agents: dict[str, DurableClaudeAgent] = {}

    def serve(
        self, agent: str, start: int, total: int, limit: int
    ) -> list[dict[str, Any]]:
        """Return a page of an agent's conversation (the segment Activity asks)."""
        key, _, subpath = agent.partition("/")
        held = self.agents.get(key)
        if held is None:
            raise ValueError(f"This Workflow has no agent {agent!r}")
        return held._page(start, total, limit, subpath)  # pyright: ignore[reportPrivateUsage]

    async def native(self, request: NativeRequest) -> ToolOutcome | None:
        """Route an internal child request to its owning agent."""
        held = self.agents.get(request.agent)
        if held is None:
            raise ApplicationError("Unknown Claude agent", non_retryable=True)
        return await held._native(request)  # pyright: ignore[reportPrivateUsage]


def _register(agent: DurableClaudeAgent) -> str:
    """Serve ``agent``'s conversation through the Workflow's Query; return its key.

    Keys count agents in the order they first run, which replays the same way.
    """
    handler = workflow.get_query_handler(QUERY)
    registry = getattr(handler, "__self__", None)
    if not isinstance(registry, _Conversations):
        registry = _Conversations()
        workflow.set_query_handler(QUERY, registry.serve)
        workflow.set_update_handler(NATIVE_UPDATE, registry.native)
    key = str(len(registry.agents))
    registry.agents[key] = agent
    return key


def _handler_context() -> str | None:
    """Describe the Update or Signal handler this code runs in, or None.

    Continuing as new waits until every handler finished, so it can never happen
    while the task itself runs in a handler.
    """
    update = workflow.current_update_info()
    if update is not None:
        return f"the Update handler {update.name!r}"
    task = asyncio.current_task()
    if task is not None and task.get_name().startswith("signal: "):
        return "a Signal handler"
    return None


@dataclass
class DurableTool:
    """A Temporal Activity that Claude can call as a tool.

    Create one with :func:`activity_as_tool`.

    Attributes:
        activity: The Activity function (it takes one dict argument).
        name: Tool name shown to Claude.
        description: Tool description shown to Claude.
        input_schema: JSON Schema of the argument.
        needs_approval: Whether a human must approve each call first.
        start_to_close_timeout: Timeout of each attempt.
        retry_policy: Retry policy of the tool Activity.
        cancellation_type: What cancelling the Workflow does to a running call.
        schedule_to_close_timeout: Timeout of the whole call, retries included.
        heartbeat_timeout: Heartbeat timeout of each attempt.
        task_queue: Task queue of the tool Activity (default: the Workflow's).
    """

    activity: Callable[..., Any]
    name: str
    description: str
    input_schema: dict[str, Any]
    needs_approval: bool = False
    start_to_close_timeout: timedelta = timedelta(minutes=1)
    retry_policy: RetryPolicy | None = None
    cancellation_type: workflow.ActivityCancellationType = (
        workflow.ActivityCancellationType.TRY_CANCEL
    )
    schedule_to_close_timeout: timedelta | None = None
    heartbeat_timeout: timedelta | None = None
    task_queue: str | None = None

    def spec(self) -> ToolSpec:
        """Return what Claude sees about this tool."""
        return ToolSpec(self.name, self.description, self.input_schema)


def activity_as_tool(
    activity_fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
    needs_approval: bool = False,
    start_to_close_timeout: timedelta = timedelta(minutes=1),
    retry_policy: RetryPolicy | None = None,
    cancellation_type: workflow.ActivityCancellationType = (
        workflow.ActivityCancellationType.TRY_CANCEL
    ),
    schedule_to_close_timeout: timedelta | None = None,
    heartbeat_timeout: timedelta | None = None,
    task_queue: str | None = None,
) -> DurableTool:
    """Turn a Temporal Activity (taking one dict argument) into a tool Claude can call.

    Each call runs as its own Activity with ID ``tool-<tool_use_id>``. Pass
    ``activity.info().activity_id`` to external systems as an idempotency key.
    Claude's arguments are not checked against ``input_schema``: validate them in the
    Activity like any other untrusted input.

    Args:
        activity_fn: The Activity function.
        name: Tool name; defaults to the function name.
        description: Tool description; defaults to the function's docstring.
        input_schema: JSON Schema of the argument; defaults to any object.
        needs_approval: Whether a human must approve each call first.
        start_to_close_timeout: Timeout of each attempt.
        retry_policy: Retry policy of the tool Activity.
        cancellation_type: What cancelling the Workflow does to a running call.
            ``WAIT_CANCELLATION_COMPLETED`` waits until the call finishes or
            acknowledges the cancellation (through a heartbeat), so the history
            records what really happened.
        schedule_to_close_timeout: Timeout of the whole call, retries included.
        heartbeat_timeout: Heartbeat timeout of each attempt, for long tools.
        task_queue: Task queue of the tool Activity (default: the Workflow's).

    Returns:
        The tool, to pass to :class:`DurableClaudeAgent`.

    Raises:
        TypeError: If ``activity_fn`` is not a Temporal Activity (``@activity.defn``).
        ValueError: If the tool name is not 1 to 50 letters, digits, ``_`` or ``-``.
    """
    if activity._Definition.from_callable(activity_fn) is None:  # pyright: ignore[reportPrivateUsage]
        raise TypeError(
            f"{getattr(activity_fn, '__name__', activity_fn)!r} is not a Temporal "
            "Activity: decorate it with @activity.defn."
        )
    tool_name = name or activity_fn.__name__
    _check_tool_name(tool_name)
    return DurableTool(
        activity=activity_fn,
        name=tool_name,
        description=description or inspect.getdoc(activity_fn) or activity_fn.__name__,
        input_schema=input_schema or dict(_OPEN_SCHEMA),
        needs_approval=needs_approval,
        start_to_close_timeout=start_to_close_timeout,
        retry_policy=retry_policy,
        cancellation_type=cancellation_type,
        schedule_to_close_timeout=schedule_to_close_timeout,
        heartbeat_timeout=heartbeat_timeout,
        task_queue=task_queue,
    )


class DurableClaudeAgent:
    """Runs a Claude agent loop inside a Temporal Workflow.

    Each model segment is one Activity. Each durable tool call is its own Activity.
    Tools marked ``needs_approval`` wait for :meth:`decide`. The agent runs one task
    at a time. The conversation lives in the Workflow, unless the Worker's runner
    keeps it in a session store; segments read it with a Query.

    Long sessions can continue as new (:meth:`continue_as_new`), carrying an
    :class:`AgentState`; with ``auto_continue_as_new=True`` the agent does it by
    itself, before a step, when the server suggests it.
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        tools: Sequence[DurableTool] = (),
        model: str | None = None,
        max_turns: int | None = None,
        builtin_tools: Sequence[str] = (),
        tool_activities: Sequence[str] = ("Bash", "mcp__*"),
        tool_approvals: Sequence[str] = (),
        tool_activity_timeout: timedelta = timedelta(minutes=10),
        tool_activity_retry_policy: RetryPolicy | None = None,
        tool_activity_task_queue: str | None = None,
        segment_timeout: timedelta = timedelta(minutes=10),
        segment_heartbeat_timeout: timedelta | None = timedelta(seconds=30),
        segment_retry_policy: RetryPolicy | None = None,
        segment_cancellation_type: workflow.ActivityCancellationType = (
            workflow.ActivityCancellationType.TRY_CANCEL
        ),
        max_segments: int | None = 50,
        approvers: Sequence[str] | None = None,
        state: AgentState | None = None,
        auto_continue_as_new: bool = False,
        continue_as_new_after_events: int | None = None,
        continue_as_new_args: Callable[[AgentState], Sequence[Any]] | None = None,
        live_output: bool = False,
        live_output_keep: int = 1000,
        live_output_keep_bytes: int = 256 * 1024,
        live_output_linger: timedelta = timedelta(milliseconds=500),
    ) -> None:
        """Create the agent, usually in the Workflow's ``__init__``.

        Args:
            system_prompt: Optional system prompt.
            tools: The durable tools Claude may call. Names must be unique.
            model: Optional model name.
            max_turns: Optional cap on engine turns within one segment.
            builtin_tools: Claude Code built-in tools to enable inside the engine.
                Those in ``tool_activities`` run as their own Activities; the others
                run inside the segment Activity.
            tool_activities: Claude Code tools that run as their own Activities, like
                durable tools: ``Bash``, ``PowerShell``, and MCP tools (name patterns
                such as ``mcp__github__*``). Each call is an Activity
                ``run_claude_tool_step`` with ID ``tool-<tool_use_id>``. Native
                subagents also run these calls as Activities, with the same approvals.
            tool_approvals: Patterns of ``tool_activities`` whose calls wait for a
                human decision first, like ``needs_approval`` tools. Each must fall
                within a ``tool_activities`` pattern.
            tool_activity_timeout: Timeout of each attempt of such a call.
            tool_activity_retry_policy: Retry policy of such a call. A failing command
                is a result for Claude, not a failed Activity: the Activity fails only
                when the step breaks before it has the call's result. By default
                Temporal retries it without limit, so the command can run again then;
                ``maximum_attempts=1`` runs it at most once.
            tool_activity_task_queue: Task queue for engine tool Activities. Use a
                separate queue when model segments occupy all the segment Worker's
                Activity slots while waiting for native child results.
            segment_timeout: Timeout of each model segment attempt.
            segment_heartbeat_timeout: Heartbeat timeout of each segment attempt, and
                of each tool step. It also bounds how late a cancel reaches a running
                step: the Worker hears of it with a heartbeat, at most every 0.8 x
                this.
            segment_retry_policy: Retry policy of the segment Activity.
            segment_cancellation_type: What cancelling the Workflow does to a running
                segment or tool step. ``WAIT_CANCELLATION_COMPLETED`` waits until the
                engine stopped, so nothing runs after the Workflow reports cancelled.
            max_segments: A task fails after this many segments, counted across
                Continue-As-New, before it runs another tool. None means no limit.
            approvers: If set, only these names may decide on tool calls. The name is
                whatever the caller passes; authenticate callers with Temporal
                (for example API keys and namespace permissions), not with this list.
            state: The state a previous run handed over, or None on first start.
            auto_continue_as_new: Continue as new by itself, before a step (between
                tool calls, or before a task's first step), when the server suggests
                it. The Workflow's run method
                must accept the new arguments (see ``continue_as_new_args``), ``run``
                must be called from that method, and no handler may wait for the
                task (Continue-As-New waits for every handler to finish). If the
                state cannot move to a new run (a long conversation without External
                Storage), the agent keeps going in this run, and the task fails before
                the next step could take the history past Temporal's limits.
            continue_as_new_after_events: Continue as new at this history length
                instead of when the server suggests it.
            continue_as_new_args: Builds the new run's arguments from the state; it
                may be called more than once, so keep it free of side effects.
                Defaults to ``[prompt, state]``, for a ``run(self, prompt, state=None)``.
            live_output: Publish the agent's events (Claude's text, tool calls,
                approvals, the answer) through Workflow Streams; read them with
                ``follow_agent``. Requires creating the agent during the Workflow's
                initialization, where Workflow Streams registers its handlers.
            live_output_keep: Most events carried across Continue-As-New.
            live_output_keep_bytes: Most bytes of events carried across
                Continue-As-New, so the new run's input stays small.
            live_output_linger: How long a task waits after its final event, so
                subscribers receive it before the Workflow closes.

        Raises:
            ValueError: If two tools share a name, a tool name is not 1 to 50 letters,
                digits, ``_`` or ``-``, ``max_segments`` is below 1,
                ``tool_activities`` names a tool that cannot run as its own Activity,
                or a ``tool_approvals`` pattern falls outside ``tool_activities``.
            RuntimeError: If ``live_output`` is on and the Workflow is already
                initialized, or another agent in this Workflow already uses it.
        """
        names = [t.name for t in tools]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Tool names must be unique: {', '.join(duplicates)}")
        for name in names:
            _check_tool_name(name)
        if max_segments is not None and max_segments < 1:
            raise ValueError("max_segments must be at least 1, or None")
        self._tools = {t.name: t for t in tools}
        self._system_prompt = system_prompt
        self._model = model
        self._max_turns = max_turns
        self._builtin_tools = list(builtin_tools)
        _check_tool_activities(tool_activities, tool_approvals)
        self._tool_activities = list(tool_activities)
        self._tool_approvals = list(tool_approvals)
        self._tool_activity_timeout = tool_activity_timeout
        self._tool_activity_retry_policy = tool_activity_retry_policy
        self._tool_activity_task_queue = tool_activity_task_queue
        self._segment_timeout = segment_timeout
        self._segment_heartbeat_timeout = segment_heartbeat_timeout
        self._segment_retry_policy = segment_retry_policy
        self._segment_cancellation_type = segment_cancellation_type
        self._max_segments = max_segments
        self._approvers = set(approvers) if approvers else None
        self._auto_continue = auto_continue_as_new
        self._continue_as_new_after_events = continue_as_new_after_events
        self._continue_as_new_args = continue_as_new_args
        self._decisions: dict[str, bool] = {}
        self._waiting: dict[str, DeferredCall] = {}
        self._calls: dict[str, dict[str, Any]] = {}
        self._outcomes: dict[str, ToolOutcome] = {}  # of unanswered calls that finished
        self._unanswered: list[DeferredCall] = []
        self._running = False
        self._native_segment: int | None = None
        self._native_attempt: tuple[int, str] | None = None
        self._native_tasks: dict[str, asyncio.Task[ToolOutcome]] = {}
        self._state = state if state is not None else AgentState()
        self._key: str | None = None  # names this agent in the conversation Query
        self._sizes = [len(t) for t in self._state.conversation]
        self._warned_handover = False
        self._mark: tuple[int, int] | None = None  # history events and bytes then
        self._largest_step = (0, 0)  # most events and bytes one step added
        self._keep = live_output_keep
        self._keep_bytes = live_output_keep_bytes
        self._linger = live_output_linger
        self._stream: WorkflowStream | None = None
        if live_output:
            if workflow.instance() is not None:
                raise RuntimeError(
                    "With live_output=True, create DurableClaudeAgent while the "
                    "Workflow is being initialized (from its __init__): Workflow "
                    "Streams registers its handlers there."
                )
            try:
                self._stream = WorkflowStream(prior_state=self._state.stream)
            except RuntimeError as err:
                raise RuntimeError(
                    "Only one DurableClaudeAgent per Workflow can use live_output=True "
                    f"(Workflow Streams allows one stream per Workflow): {err}"
                ) from err
            self._topic = self._stream.topic(TOPIC)
        self._state.stream = None  # the stream owns it now

    # ---- what the agent has done ----
    @property
    def total_cost_usd(self) -> float:
        """Model cost reported by the SDK, across all runs, in USD."""
        return self._state.total_cost_usd

    @property
    def segments(self) -> int:
        """Model segments run, across all runs."""
        return self._state.segments

    @property
    def total_tool_calls(self) -> int:
        """Tool calls Claude asked for that the Workflow handled, across all runs.

        Durable tools and Claude Code tools in ``tool_activities``, including calls
        that were rejected or failed.
        """
        return self._state.tool_calls

    @property
    def runs(self) -> int:
        """Workflow runs this agent has used, counting the current one."""
        return self._state.runs

    @property
    def busy(self) -> bool:
        """Whether a task is unfinished (for example, one handed over by Continue-As-New)."""
        return self._state.task_prompt is not None

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Durable tool calls of the current run, with their status."""
        return list(self._calls.values())

    # ---- handlers the user's Workflow exposes ----
    def validate_decision(self, tool_use_id: str, approver: str | None = None) -> None:
        """Refuse a decision that must not be recorded.

        Call it from an Update validator: a refused Update is never written to
        the Workflow history.

        Args:
            tool_use_id: The tool call being decided.
            approver: Who decides.

        Raises:
            ValueError: If no such call is waiting, it was already decided, or the
                approver is not allowed.
        """
        if tool_use_id not in self._waiting:
            raise ValueError(f"No tool call {tool_use_id} is waiting for approval")
        if tool_use_id in self._decisions:
            raise ValueError(f"Tool call {tool_use_id} was already decided")
        if self._approvers is not None and approver not in self._approvers:
            raise ValueError(f"{approver!r} is not allowed to approve tool calls")

    def decide(
        self, tool_use_id: str, approved: bool, approver: str | None = None
    ) -> bool:
        """Record a human decision on a tool call that needs approval.

        The first valid decision counts. Safe to call from a Signal handler: an
        invalid decision is logged and ignored instead of raising.

        Args:
            tool_use_id: The tool call being decided.
            approved: Whether the call may run.
            approver: Who decided, stored with the call.

        Returns:
            Whether the decision was recorded.
        """
        try:
            self.validate_decision(tool_use_id, approver)
        except ValueError as err:
            workflow.logger.warning("Ignored a decision: %s", err)
            return False
        self._decisions[tool_use_id] = approved
        record = self._calls.get(tool_use_id)
        if record is not None and approver:
            record["decided_by"] = approver
        return True

    def pending_approvals(self) -> list[dict[str, Any]]:
        """Return the tool calls waiting for a decision."""
        return [
            {"id": c.id, "name": c.name, "input": c.input}
            | (
                {"child": self._state.native_calls[c.id].child}
                if c.id in self._state.native_calls
                else {}
            )
            for c in self._waiting.values()
        ]

    def state(self) -> AgentState:
        """Return a copy of what the agent has done so far, without the live output.

        To continue as new, use :meth:`continue_as_new`, which also carries the
        live output stream.
        """
        s = self._state
        return AgentState(
            session_id=s.session_id,
            checkpoint=s.checkpoint,
            segment_index=s.segment_index,
            task_prompt=s.task_prompt,
            task_segments=s.task_segments,
            pending=dict(s.pending),
            recent_call_ids=list(s.recent_call_ids),
            segments=s.segments,
            tool_calls=s.tool_calls,
            total_cost_usd=s.total_cost_usd,
            runs=s.runs,
            fork_next=s.fork_next,
            conversation=list(s.conversation),
            external_storage=s.external_storage,
            child_conversations={k: list(v) for k, v in s.child_conversations.items()},
            child_subpaths=list(s.child_subpaths),
            child_checkpoints=dict(s.child_checkpoints),
            native_calls=dict(s.native_calls),
        )

    def _page(
        self, start: int, total: int, limit: int, subpath: str = ""
    ) -> list[dict[str, Any]]:
        """A page of the conversation the Workflow holds (read-only: Query handler).

        A step scheduled with another number of entries is out of date (for example
        an attempt that timed out and still runs): it gets no page.
        """
        texts = (
            self._state.child_conversations.get(subpath, [])
            if subpath
            else self._state.conversation
        )
        if total != len(texts):
            raise ValueError(
                f"The conversation has {len(texts)} entries, but the step that asks "
                f"was scheduled with {total}."
            )
        limit = min(max(limit, 1), PAGE_BYTES)
        sizes = [len(t) for t in texts] if subpath else self._sizes
        return [json.loads(t) for t in page(texts, sizes, start, limit)]

    async def _native(self, request: NativeRequest) -> ToolOutcome | None:
        """Accept a child request once; never regenerate it after segment loss."""
        if not self._running or request.segment != self._native_segment:
            raise ApplicationError(
                "Stopped or stale Claude segment", non_retryable=True
            )
        identity = (request.attempt, request.token)
        calls = self._state.native_calls
        if request.call is None:
            if self._native_attempt == identity:
                return None
            if any(c.segment == request.segment for c in calls.values()):
                raise ApplicationError(
                    "The native child segment was lost after accepting a tool call. "
                    "Its outcomes remain in AgentState.native_calls; native result "
                    "delivery cannot be restored safely. Do not rediscover the call.",
                    type="ClaudeNativeDeliveryLost",
                    non_retryable=True,
                )
            if self._native_attempt and request.attempt <= self._native_attempt[0]:
                raise ApplicationError("Stale Claude attempt", non_retryable=True)
            self._native_attempt = identity
            return None
        if identity != self._native_attempt or not request.child:
            raise ApplicationError(
                "Stale or unidentified child call", non_retryable=True
            )
        call = request.call
        valid = (
            call.name in self._tools
            if call.kind == "durable"
            else call.kind == "engine"
            and any(fnmatch.fnmatchcase(call.name, p) for p in self._tool_activities)
        )
        if not valid:
            raise ApplicationError("Unmanaged child tool", non_retryable=True)
        previous = calls.get(call.id)
        if previous is not None:
            if (
                previous.segment != request.segment
                or previous.child != request.child
                or previous.call != call
            ):
                raise ApplicationError(
                    "Conflicting native call identity", non_retryable=True
                )
            if previous.outcome is not None:
                return previous.outcome
        else:
            if call.id in self._calls or call.id in self._state.recent_call_ids:
                raise ApplicationError("Reused native tool-use ID", non_retryable=True)
            calls[call.id] = NativeCallState(request.segment, request.child, call)
            self._state.tool_calls += 1

            async def execute() -> ToolOutcome:
                outcome = await self._run_tool(call)
                calls[call.id].outcome = outcome
                return outcome

            self._native_tasks[call.id] = asyncio.create_task(execute())
        # Duplicate Update callers share the same execution. Cancelling one caller
        # must not cancel the accepted Activity or another caller's delivery.
        return await asyncio.shield(self._native_tasks[call.id])

    async def _drain_native(self) -> None:
        """Fence new calls, stop outstanding calls, and settle their handlers."""
        self._native_segment = None
        for task in self._native_tasks.values():
            if not task.done():
                task.cancel()
        if self._native_tasks:
            await asyncio.gather(*self._native_tasks.values(), return_exceptions=True)
        self._native_tasks.clear()
        self._native_attempt = None

    # ---- Continue-As-New ----
    def should_continue_as_new(self) -> bool:
        """Whether it is time to continue as new.

        True when the server suggests it (it counts history events, history size
        and Updates), or at ``continue_as_new_after_events`` when that is set, and
        the agent's state fits in the new run's input. Without External Storage, a
        conversation the Workflow holds can outgrow one payload (2 MB by default);
        then this stays False, with a warning in the Worker's log, and the agent
        keeps going in this run (with ``auto_continue_as_new``, until the next step
        could take the history past Temporal's limits: then the task fails with an
        error that says what to change).
        """
        info = workflow.info()
        if self._continue_as_new_after_events is not None:
            due = (
                info.get_current_history_length() >= self._continue_as_new_after_events
            )
        else:
            due = info.is_continue_as_new_suggested()
        if not due:
            return False
        problem = self._handover_problem()
        if problem is None or not self._new_checks():
            return True
        if not self._warned_handover:
            self._warned_handover = True
            workflow.logger.warning("%s The agent keeps going in this run.", problem)
        return False

    def _new_checks(self, patch: str = _CHECKS_PATCH) -> bool:
        """Whether this run makes the checks added after the first published version.

        Called only where they change a decision, so most histories get no marker.
        """
        return workflow.patched(patch)

    def _handover_problem(self, measured: int | None = None) -> str | None:
        """Why the new run's input cannot carry this agent's state, or None if it can.

        Args:
            measured: The new run's input size, if the caller just measured it.
        """
        if self._state.external_storage:
            return None  # large payloads go to the store
        size = sum(self._sizes)  # the conversation alone: at most the state's size
        if size <= PAYLOAD_LIMIT_BYTES:
            size = measured if measured is not None else self._input_bytes(self.state())
            if size <= PAYLOAD_LIMIT_BYTES:
                return None
        return (
            f"The agent's state is {size / 1024 / 1024:.2f} MB, mostly the "
            "conversation: more than the new run's input can carry without External "
            "Storage (Temporal refuses payloads over 2 MB by default). Configure "
            "External Storage on the Client (see the README), or give the runner a "
            "session store."
        )

    async def continue_as_new(self) -> NoReturn:
        """Continue the Workflow as new, carrying this agent's state.

        Call it from the Workflow's run method, for example between the messages of
        a chat, when no handler is waiting for an answer. It waits until no task is
        running (in an Update handler, say), lets running handlers finish, hands
        live output subscribers over to the new run, and calls
        ``workflow.continue_as_new`` with ``continue_as_new_args`` (default
        ``[prompt, state]``).

        Raises:
            ApplicationError: If called from an Update or Signal handler, or if the
                state does not fit in the new run's input (see
                :meth:`should_continue_as_new`).
        """
        self._refuse_in_handler("continue_as_new()")
        await workflow.wait_condition(lambda: not self._running)
        problem = self._handover_problem()
        if problem is not None and self._new_checks():
            raise ApplicationError(problem, non_retryable=True)
        await self._hand_over()

    def _refuse_in_handler(self, what: str) -> None:
        where = _handler_context()
        if where is not None:
            raise ApplicationError(
                f"{what} cannot be used in {where}: continuing as new waits for every "
                "handler to finish, including this one. Use it from the Workflow's "
                "run method.",
                non_retryable=True,
            )

    async def _hand_over(self) -> NoReturn:
        if self._stream is not None:
            self._publish({"type": "continued_as_new", "run": self._state.runs + 1})
            self._stream.detach_pollers()  # subscribers follow to the new run
        await workflow.wait_condition(workflow.all_handlers_finished)
        state = self.state()
        state.runs += 1
        # Measured once, here: a handler that ran meanwhile may have grown the state.
        # Before any codec, which can only overstate it.
        rest = None if state.external_storage else self._input_bytes(state)
        problem = self._handover_problem(rest)
        if problem is not None and self._new_checks():
            raise ApplicationError(problem, non_retryable=True)
        if self._stream is not None:
            problem = self._handover_problem(self._carry_stream(state, rest))
            if problem is not None and self._new_checks(_CARRY_PATCH):
                raise ApplicationError(problem, non_retryable=True)
        workflow.continue_as_new(args=self._new_run_args(state))

    def _carry_stream(self, state: AgentState, rest: int | None) -> int | None:
        """Put the newest live output events that fit into the new run's ``state``.

        Args:
            state: The new run's state, without the stream.
            rest: The new run's input size without the stream, or None when External
                Storage takes large payloads.

        Returns:
            The new run's input size with the stream, or None when not measured.
        """
        assert self._stream is not None
        budget = self._keep_bytes
        if rest is not None:
            # The stream shares the new run's input with everything else in it.
            budget = max(0, min(budget, _HANDOVER_BYTES - rest))
        self._trim_stream(budget)
        state.stream = self._stream.get_state()
        if rest is None:
            return None
        # Measured again as the input carries it (with the stream's publishers, and
        # whatever the converter adds): while over the share, carry fewer events.
        total = self._input_bytes(state)
        while total > _HANDOVER_BYTES and state.stream.log:
            carried = sum(_event_bytes(item) for item in state.stream.log)
            self._trim_stream(carried - (total - _HANDOVER_BYTES))
            state.stream = self._stream.get_state()
            total = self._input_bytes(state)
        return total

    def _input_bytes(self, state: AgentState) -> int:
        """Size of the new run's input with ``state``, before any codec."""
        payloads = workflow.payload_converter().to_payloads(self._new_run_args(state))
        return sum(p.ByteSize() for p in payloads)

    def _new_run_args(self, state: AgentState) -> list[Any]:
        if self._continue_as_new_args is not None:
            return list(self._continue_as_new_args(state))
        return [state.task_prompt, state]

    def _trim_stream(self, max_bytes: int) -> None:
        """Keep only the newest events, within both limits, for the next run."""
        assert self._stream is not None
        snapshot = self._stream.get_state()
        kept = size = 0
        for item in reversed(snapshot.log):
            item_size = _event_bytes(item)
            if kept >= self._keep or size + item_size > max_bytes:
                break
            kept += 1
            size += item_size
        self._stream.truncate(snapshot.base_offset + len(snapshot.log) - kept)

    # ---- the loop ----
    async def run(self, prompt: str | None = None) -> str:
        """Run a task until Claude gives a final answer.

        If a task is unfinished (handed over by Continue-As-New), it continues that
        task, and ``prompt`` is ignored. Otherwise it starts a task with ``prompt``:
        the first task starts the Claude session, later tasks continue it, so the
        agent remembers earlier turns.

        Args:
            prompt: The user's request.

        Returns:
            Claude's final answer.

        Raises:
            ValueError: If there is no prompt and no unfinished task.
            ApplicationError: If another task is running, Claude asks for a tool call
                that already ran, a segment reports an error that retrying cannot fix,
                ``max_segments`` is reached, or ``auto_continue_as_new`` is on and this
                runs in a handler, or the history nearly reached Temporal's limits
                while the state could not move to a new run.
            ActivityError: If a segment's Activity fails for good. Catch
                ``temporalio.exceptions.FailureError`` for both.
        """
        if self._running:
            raise ApplicationError(
                "The agent runs one task at a time; wait for the current task to end.",
                non_retryable=True,
            )
        if self._auto_continue:
            self._refuse_in_handler("run() with auto_continue_as_new")
        if self._key is None:
            self._key = _register(self)
        state = self._state
        if state.task_prompt is None:
            if prompt is None:
                raise ValueError("No prompt, and no unfinished task to continue")
            state.task_prompt = prompt
            state.task_segments = 0
            self._publish({"type": "prompt", "text": prompt})
        if state.session_id is None:
            state.session_id = str(workflow.uuid4())
        self._running = True
        try:
            return await self._loop()
        except BaseException as err:
            if _is_cancellation(err):
                self._end_task("the Workflow was cancelled")
                self._publish({"type": "cancelled"})
                await self._linger_if_live()
            raise
        finally:
            self._running = False
            await self._drain_native()

    async def _loop(self) -> str:
        state = self._state
        # A task sends its prompt with its first segment. A task handed over by
        # Continue-As-New already sent it, unless no segment of it committed.
        send_prompt = state.task_segments == 0
        self._mark = None  # steps are measured within a task, not across idle time
        first = True
        while True:
            if self._auto_continue:
                await self._continue_as_new_or_stop(first_of_task=first)
            first = False
            index = state.segment_index
            self._native_segment = index
            self._native_attempt = None

            def segment_input(index: int = index) -> SegmentInput:
                return SegmentInput(
                    session_id=state.session_id or "",
                    prompt=state.task_prompt if send_prompt else None,
                    tools=[t.spec() for t in self._tools.values()],
                    system_prompt=self._system_prompt,
                    model=self._model,
                    max_turns=self._max_turns,
                    builtin_tools=self._builtin_tools,
                    tool_activities=self._tool_activities,
                    checkpoint=state.checkpoint,
                    injected=dict(state.pending),
                    segment_index=index,
                    live_output=self._stream is not None,
                    fork=state.fork_next,
                    conversation=ConversationRef(
                        query=QUERY,
                        agent=self._key or "",
                        entries=len(state.conversation),
                    ),
                    child_conversations={
                        path: ConversationRef(
                            QUERY, f"{self._key}/{path}", len(entries)
                        )
                        for path, entries in state.child_conversations.items()
                    },
                    child_subpaths=list(state.child_subpaths),
                    child_checkpoints=dict(state.child_checkpoints),
                )

            seg_input = await self._fit_results(segment_input)
            state.segment_index = index + 1
            try:
                seg: SegmentOutput = await workflow.execute_activity(
                    SEGMENT_ACTIVITY_NAME,
                    seg_input,
                    result_type=SegmentOutput,
                    start_to_close_timeout=self._segment_timeout,
                    heartbeat_timeout=self._segment_heartbeat_timeout,
                    retry_policy=self._segment_retry_policy,
                    cancellation_type=self._segment_cancellation_type,
                    summary=f"claude segment {index + 1}",
                )
            except ActivityError as err:
                if not _is_cancellation(err):
                    self._end_task("a model segment failed")
                    self._publish({"type": "error", "error": f"{err}: {err.cause}"})
                    await self._linger_if_live()
                raise
            state.task_segments += 1
            state.segments += 1
            state.total_cost_usd += seg.cost_usd
            if seg.is_error:
                await self._fail(f"Claude run failed: {seg.error}")
            accepted = {
                key for key, call in state.native_calls.items() if call.segment == index
            }
            if accepted != set(seg.native_calls) or any(
                state.native_calls[key].outcome is None for key in accepted
            ):
                await self._fail(
                    "Native child outcomes were not delivered and verified"
                )
            for key in accepted:
                state.native_calls[key].delivered = True
            state.recent_call_ids = [*state.recent_call_ids, *sorted(accepted)][
                -_RECENT_CALLS:
            ]
            completed = [
                key for key, call in state.native_calls.items() if call.delivered
            ]
            for key in completed[:-_RECENT_CALLS]:
                del state.native_calls[key]
            self._native_segment = None
            self._native_tasks.clear()
            self._native_attempt = None
            if seg.checkpoint is None:
                await self._fail(
                    "The segment runner returned no checkpoint, so a retry could not "
                    "continue this session safely."
                )
            mixed = self._conversation_problem(seg)
            if mixed is not None:
                await self._fail(mixed)
            # The segment committed: its tool results and prompt reached Claude.
            send_prompt = False
            state.pending = {}
            state.fork_next = False
            state.session_id = seg.session_id or state.session_id
            state.checkpoint = seg.checkpoint
            state.external_storage = seg.external_storage
            state.child_subpaths = list(seg.child_subpaths)
            state.child_checkpoints = dict(seg.child_checkpoints)
            for path, entries in seg.child_transcripts.items():
                held = state.child_conversations.setdefault(path, [])
                keep = seg.child_keep.get(path, 0)
                if not 0 <= keep <= len(held):
                    await self._fail("Invalid child transcript checkpoint")
                del held[keep:]
                held.extend(entry_text(e) for e in entries)
            if seg.transcript_keep is not None:
                keep = seg.transcript_keep
                added = [entry_text(e) for e in seg.transcript_add]
                del state.conversation[keep:]
                del self._sizes[keep:]
                state.conversation.extend(added)
                self._sizes.extend(len(t) for t in added)
            if seg.deferred is None:
                state.task_prompt = None
                state.task_segments = 0
                result = seg.result or ""
                self._publish({"type": "done", "result": result})
                await self._linger_if_live()
                return result
            # The paused call, and the other durable calls of the same message.
            calls = [seg.deferred, *seg.siblings]
            self._unanswered = calls
            seen: set[str] = set()
            for call in calls:
                if call.id in seen or call.id in self._calls:
                    again = True
                else:
                    again = call.id in state.recent_call_ids
                seen.add(call.id)
                if again:
                    # A tool call runs at most once per agent, whatever Claude asks.
                    await self._fail(
                        f"Claude asked again for tool call {call.id} ({call.name}), "
                        "which already ran. Stopping so it cannot run twice."
                    )
            if (
                self._max_segments is not None
                and state.task_segments >= self._max_segments
            ):
                # Their results could never reach Claude, so the calls do not run.
                which = ", ".join(f"{c.id} ({c.name})" for c in calls)
                plural = "s" if len(calls) > 1 else ""
                await self._fail(
                    f"Stopped after {self._max_segments} segments, before running "
                    f"tool call{plural} {which}."
                )
            outcomes = await self._run_tools(calls)
            state.pending = {c.id: o for c, o in zip(calls, outcomes)}
            self._unanswered = []
            self._outcomes.clear()  # they are pending now
            state.recent_call_ids = [
                *state.recent_call_ids,
                *(c.id for c in calls),
            ][-_RECENT_CALLS:]
            state.tool_calls += len(calls)

    async def _fit_results(self, make: Callable[[], SegmentInput]) -> SegmentInput:
        """The next segment's input, with tool results it can carry.

        Without External Storage, the results go to the segment in its input, which
        is one payload (Temporal refuses one over 2 MB by default, and a Workflow task
        that schedules it fails again and again). When the results of one message do
        not fit together, the largest are replaced, one by one, by a note for Claude:
        the call ran, and its result is in the Workflow's history. If the input still
        does not fit, the task fails.

        Args:
            make: Builds the input from the agent's state.

        Returns:
            The input to schedule.
        """
        state = self._state
        inp = make()
        if state.external_storage or not state.pending:
            return inp
        converter = workflow.payload_converter()

        def size(value: Any) -> int:
            return sum(p.ByteSize() for p in converter.to_payloads([value]))

        total = size(inp)
        if total <= PAYLOAD_LIMIT_BYTES:
            return inp
        sizes = {call_id: size(o) for call_id, o in state.pending.items()}
        for call_id in sorted(sizes, key=lambda c: (-sizes[c], c)):
            if total <= PAYLOAD_LIMIT_BYTES:
                break
            state.pending[call_id] = ToolOutcome(
                content=(
                    f"This tool call ran, but its result ({sizes[call_id] / 1024 / 1024:.1f}"
                    " MB) was too large to deliver to Claude together with the other "
                    "results of the same message without External Storage. It is kept "
                    "in the Workflow's history. Do not call the tool again only to see "
                    "the result; ask for less data next time."
                ),
                is_error=True,
            )
            workflow.logger.warning(
                "The result of tool call %s (%d bytes) was too large to deliver with "
                "the others of its message without External Storage; Claude gets a "
                "note instead.",
                call_id,
                sizes[call_id],
            )
            inp = make()
            total = size(inp)
        if total > PAYLOAD_LIMIT_BYTES and self._new_checks():
            await self._fail(
                f"The next step's input is {total / 1024 / 1024:.2f} MB, more than one "
                "payload carries without External Storage (Temporal refuses payloads "
                "over 2 MB by default). Configure External Storage on the Client (see "
                "the README)."
            )
        return inp

    async def _continue_as_new_or_stop(self, first_of_task: bool) -> None:
        """Before a segment: continue as new when it is time, or stop in good order.

        When the state cannot move to a new run (see :meth:`should_continue_as_new`),
        the agent keeps going in this run until the next step could take the history
        past Temporal's limits; then the task fails with an error that says what to
        change, before the server ends the Workflow.

        Args:
            first_of_task: This is the task's first segment, where the first
                published version did not check (only after tool calls).
        """
        full = self._history_nearly_full()
        due = self.should_continue_as_new()
        if not due and full is None:
            return
        if first_of_task and not self._new_checks():
            return
        if due:
            await self._hand_over()
        if full is None or not self._new_checks():
            return
        problem = self._handover_problem()
        if problem is None:
            await self._hand_over()  # not suggested yet, but this run is nearly full
        await self._fail(
            f"{full}, and the agent cannot continue as new, so the task stops here "
            f"instead of the server ending the Workflow. {problem}"
        )

    def _history_nearly_full(self) -> str | None:
        """Describe a history the next step could take past the limits, or None.

        Measured between the safe points of a task, so the margin grows with the
        largest step: twice that step, and never less than ``_ROOM_EVENTS`` events and
        ``_ROOM_BYTES`` bytes.
        """
        info = workflow.info()
        now = (info.get_current_history_length(), info.get_current_history_size())
        last, self._mark = self._mark, now
        if last is not None:
            self._largest_step = (
                max(self._largest_step[0], now[0] - last[0]),
                max(self._largest_step[1], now[1] - last[1]),
            )
        # A history length set for Continue-As-New past the default limit says the
        # server allows more.
        most_events = max(_HISTORY_EVENTS, self._continue_as_new_after_events or 0)
        events = max(_ROOM_EVENTS, 2 * self._largest_step[0])
        size = max(_ROOM_BYTES, 2 * self._largest_step[1])
        if now[0] + events < most_events and now[1] + size < _HISTORY_BYTES:
            return None
        return (
            f"This Workflow's history ({now[0]:,} events, {now[1] / 1024 / 1024:.1f} "
            f"MB) is close to Temporal's limits ({most_events:,} events, "
            f"{_HISTORY_BYTES // 1024 // 1024} MB by default)"
        )

    def _end_task(self, reason: str) -> None:
        """Forget the stopped task, so the agent can take the next one.

        The next segment continues from the checkpoint (with a session store, in a
        copy of the session, which may hold part of a segment that did not commit).
        A tool call Claude is still waiting for gets an error result, delivered with
        the next task.
        """
        state = self._state
        if self._unanswered:
            state.pending = {
                call.id: self._outcome_after_stop(call, reason)
                for call in self._unanswered
            }
            self._unanswered = []
            self._outcomes.clear()
        state.task_prompt = None
        state.task_segments = 0
        state.fork_next = True
        if state.checkpoint is None:
            state.session_id = None  # nothing committed: the next task starts afresh

    def _outcome_after_stop(self, call: DeferredCall, reason: str) -> ToolOutcome:
        """What Claude learns about a call it is still waiting for when its task stops."""
        outcome = self._outcomes.get(call.id)
        if outcome is not None:
            return outcome  # it finished before the task stopped
        if self._calls.get(call.id, {}).get("status") in ("started", "cancelled"):
            text = (
                f"This tool call was interrupted ({reason}); whether it took "
                "effect is unknown. Check before running it again."
            )
        else:
            text = f"This tool call did not run: {reason}."
        return ToolOutcome(content=text, is_error=True)

    def _conversation_problem(self, seg: SegmentOutput) -> str | None:
        """Refuse a segment that keeps the conversation somewhere else than before."""
        state = self._state
        same_choice = (
            "Every Worker of a task queue needs the same kind of runner (with or "
            "without a session store)."
        )
        if seg.transcript_keep is None:
            if state.conversation:
                return (
                    "The segment runner keeps conversations in a session store, but "
                    f"this Workflow holds this one. {same_choice}"
                )
            return None
        if state.checkpoint is not None and not state.conversation:
            return (
                "The segment runner returned a conversation for the Workflow to hold, "
                f"but this one is kept in a session store. {same_choice}"
            )
        keep = seg.transcript_keep
        if (
            not 0 <= keep <= len(state.conversation)
            or keep + len(seg.transcript_add) == 0
        ):
            return (
                f"The segment runner kept {keep} of {len(state.conversation)} "
                f"conversation entries and added {len(seg.transcript_add)}."
            )
        return None

    async def _fail(self, message: str) -> NoReturn:
        self._end_task(message)
        self._publish({"type": "error", "error": message})
        await self._linger_if_live()
        raise ApplicationError(message, non_retryable=True)

    async def _linger_if_live(self) -> None:
        if self._stream is not None:
            await workflow.sleep(self._linger)  # let subscribers read the last event

    def _publish(self, event: dict[str, Any]) -> None:
        if self._stream is not None:
            self._topic.publish(cap_event({**event, "at": workflow.now().isoformat()}))

    def _tool_step(self, call: DeferredCall) -> ToolStepInput:
        """What the tool step needs: the call, and where its session paused."""
        state = self._state
        return ToolStepInput(
            session_id=state.session_id or "",
            checkpoint=state.checkpoint or "",
            call=call,
            tools=[t.spec() for t in self._tools.values()],
            builtin_tools=self._builtin_tools,
            conversation=ConversationRef(
                query=QUERY, agent=self._key or "", entries=len(state.conversation)
            ),
            native=call.id in state.native_calls,
        )

    async def _run_tools(self, calls: list[DeferredCall]) -> list[ToolOutcome]:
        """Run the calls at once, each its own Activity; return their outcomes in order.

        If one is cancelled (the Workflow is), the others still finish or stop
        before the cancellation goes on.
        """
        if len(calls) == 1:
            return [await self._run_tool(calls[0])]
        done = await asyncio.gather(
            *(self._run_tool(c) for c in calls), return_exceptions=True
        )
        for item in done:
            if isinstance(item, BaseException):
                raise item
        return cast("list[ToolOutcome]", done)

    async def _run_tool(self, call: DeferredCall) -> ToolOutcome:
        record: dict[str, Any] = {
            "id": call.id,
            "name": call.name,
            "input": call.input,
            "status": "started",
        }
        native = self._state.native_calls.get(call.id)
        if native is not None:
            record["child"] = native.child
        self._calls[call.id] = record
        self._publish(
            {
                "type": "tool_call",
                "id": call.id,
                "name": call.name,
                "input": call.input,
                **({"child": native.child} if native is not None else {}),
            }
        )
        outcome = await self._execute_tool(call, record)
        self._outcomes[call.id] = outcome
        self._publish(
            {
                "type": "tool_result",
                "id": call.id,
                "name": call.name,
                "status": record["status"],
                **({"child": native.child} if native is not None else {}),
            }
        )
        return outcome

    async def _execute_tool(
        self, call: DeferredCall, record: dict[str, Any]
    ) -> ToolOutcome:
        engine = call.kind == "engine"
        tool = None if engine else self._tools.get(call.name)
        if not engine and tool is None:
            record["status"] = "unknown tool"
            return ToolOutcome(content=f"Unknown tool: {call.name}", is_error=True)
        if engine:
            needs_approval = any(
                fnmatch.fnmatchcase(call.name, p) for p in self._tool_approvals
            )
        else:
            needs_approval = tool is not None and tool.needs_approval
        if needs_approval:
            self._waiting[call.id] = call
            record["status"] = "waiting for approval"
            self._publish(
                {
                    "type": "approval_needed",
                    "id": call.id,
                    "name": call.name,
                    "input": call.input,
                    **({"child": record["child"]} if "child" in record else {}),
                }
            )
            try:
                await workflow.wait_condition(lambda: call.id in self._decisions)
            finally:
                self._waiting.pop(call.id, None)
            if not self._decisions[call.id]:
                record["status"] = "rejected"
                return ToolOutcome(
                    content="A human reviewer rejected this action. Do not retry it.",
                    is_error=True,
                )
            record["status"] = "started"
        try:
            if tool is None:  # a Claude Code tool: its own Activity, a tool step
                outcome: ToolOutcome = await workflow.execute_activity(
                    TOOL_STEP_ACTIVITY_NAME,
                    self._tool_step(call),
                    result_type=ToolOutcome,
                    activity_id=f"tool-{call.id}",
                    task_queue=self._tool_activity_task_queue,
                    start_to_close_timeout=self._tool_activity_timeout,
                    heartbeat_timeout=self._segment_heartbeat_timeout,
                    retry_policy=self._tool_activity_retry_policy,
                    cancellation_type=self._segment_cancellation_type,
                    summary=f"tool {call.name}",
                )
                record["status"] = "done"
                return outcome
            result = await workflow.execute_activity(
                tool.activity,
                call.input,
                activity_id=f"tool-{call.id}",
                task_queue=tool.task_queue,
                start_to_close_timeout=tool.start_to_close_timeout,
                schedule_to_close_timeout=tool.schedule_to_close_timeout,
                heartbeat_timeout=tool.heartbeat_timeout,
                retry_policy=tool.retry_policy,
                cancellation_type=tool.cancellation_type,
                summary=f"tool {call.name}",
            )
        except ActivityError as err:
            if _is_cancellation(err):
                record["status"] = "cancelled"
                raise  # the Workflow is being cancelled: do not hide it from Claude's loop
            cause = err.cause
            message = (
                cause.message
                if isinstance(cause, ApplicationError)
                else str(cause or err)
            )
            record["status"] = "failed"
            return ToolOutcome(content=f"Tool failed: {message}", is_error=True)
        record["status"] = "done"
        return (
            result if isinstance(result, ToolOutcome) else ToolOutcome(content=result)
        )
