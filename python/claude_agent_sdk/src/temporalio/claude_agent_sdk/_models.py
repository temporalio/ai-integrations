"""Plain data passed between the Workflow and the segment Activity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from temporalio.contrib.workflow_streams import WorkflowStreamState

TOOL_CALL_NOT_RUN = "ToolCallNotRun"
"""Type of the ``ApplicationError`` of a tool step that failed before Claude Code ran
its call: the call did not run, and Claude can call it again."""

TOOL_CALL_INTERRUPTED = "ToolCallInterrupted"
"""Type of the ``ApplicationError`` of a tool step that failed after its call was let
run: the call may have run, in full or in part."""

TRY_AGAIN = "try_again"
"""Key in the details of a ``ToolCallNotRun`` failure: whether the Workflow may run
the call in a new attempt (see ``ToolStepRetry``)."""

WORKSPACE_NOT_FOUND = "WorkspaceNotFound"
"""Type of the ``ApplicationError`` of a step whose agent's workspace snapshot cannot
be restored: an object of it is not in the workspace store. Not retryable."""

WORKSPACE_CORRUPT = "WorkspaceCorrupt"
"""Type of the ``ApplicationError`` of a step whose agent's workspace snapshot cannot
be restored: an object of it does not match its key, or is not valid. Not retryable."""

WORKSPACE_TOO_LARGE = "WorkspaceTooLarge"
"""Type of the ``ApplicationError`` of a step whose agent's folder holds more bytes of
files than ``workspace_max_bytes``. Not retryable."""

WORKSPACE_INVALID = "WorkspaceInvalid"
"""Type of the ``ApplicationError`` of a step whose agent's folder cannot be used: its
id is not a plain name, it is a link or a file, a file in it cannot be read, or a link
in it cannot be created. Not retryable."""


_WORKSPACE_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?")
_WINDOWS_DEVICES = frozenset(
    ["con", "prn", "aux", "nul", "conin$", "conout$"]
    + [f"com{i}" for i in range(10)]
    + [f"lpt{i}" for i in range(10)]
)


def windows_device_name(name: str) -> bool:
    """Whether Windows keeps ``name`` for a device (``con``, ``nul.txt``, ...)."""
    return name.split(".")[0].lower() in _WINDOWS_DEVICES


def workspace_id_ok(value: Any) -> bool:
    """Whether ``value`` can name an agent's folder on any system.

    1 to 128 letters, digits, ``.``, ``_`` or ``-``, starting and ending with a letter
    or a digit, and not a name Windows keeps for a device.
    """
    return (
        isinstance(value, str)
        and _WORKSPACE_ID.fullmatch(value) is not None
        and not windows_device_name(value)
    )


@dataclass
class Workspace:
    """An agent's own folder, and the snapshot of it that the Workflow committed.

    Used by a runner with a ``workspace_store``: the folder is ``<cwd>/<id>`` on every
    Worker, and each segment and tool step first makes it match ``snapshot``.

    Attributes:
        id: The folder's name: the agent's first session id.
        snapshot: The snapshot of the folder that the last completed segment or tool
            step saved (the SHA-256 of its root tree in the store), or None before
            the first one (an empty folder).
    """

    id: str
    snapshot: str | None = None


@dataclass
class ToolSpec:
    """What Claude sees about one durable tool.

    Attributes:
        name: Tool name, unique within the agent.
        description: What the tool does, shown to Claude.
        input_schema: JSON Schema of the tool's single argument.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolOutcome:
    """The stored result of a durable tool call, handed back to Claude.

    Attributes:
        content: The tool's result (any JSON-serializable value).
        is_error: Whether the call failed; Claude sees the content as an error.
        blocks: Content blocks (text, images) to hand back exactly as they are,
            instead of ``content``: what a Claude Code tool returned.
        entry: For an Edit or a Write that ran in a tool step, and for any call that
            waited for its tool step in a warm engine (``warm_engines``): Claude
            Code's own transcript entry for the result, without the result itself
            (that is ``content`` or ``blocks``), or one of the same shape. The next
            segment puts it where the call's result belongs, so Claude Code continues
            from its own record of the call (without it, the segment writes a record
            of the same shape).
        workspace: From a tool step of a runner with a ``workspace_store``: the
            snapshot of the agent's folder after the call. The Workflow takes it as
            the agent's current snapshot; it is not passed to Claude.
    """

    content: Any = None
    is_error: bool = False
    blocks: list[dict[str, Any]] | None = None
    entry: dict[str, Any] | None = None
    workspace: str | None = None


@dataclass
class DeferredCall:
    """A tool call Claude asked for; the Workflow runs it as its own Activity.

    Attributes:
        id: The ``tool_use_id`` Claude assigned to the call.
        name: The durable tool's name, or the Claude Code tool's name.
        input: The call's arguments.
        kind: ``durable`` for a durable tool (an Activity of yours), ``engine`` for a
            Claude Code tool (such as Bash, or an MCP server's tool) that runs as
            its own Activity, ``run_claude_tool_step``.
    """

    id: str
    name: str
    input: dict[str, Any]
    kind: str = "durable"


@dataclass
class ConversationRef:
    """Where a segment finds the conversation the Workflow holds.

    The segment Activity reads it with a Query on its own Workflow, so the
    conversation is not copied into each step's input: the history records only what
    each step adds.

    Attributes:
        query: Name of the Workflow Query that returns the conversation, a page at a
            time.
        agent: The agent's key, the Query's first argument (a Workflow can hold
            several agents).
        entries: How many entries the committed conversation has.
    """

    query: str
    agent: str
    entries: int


@dataclass
class SegmentInput:
    """Input of one model segment: from a prompt or a tool result to the next pause.

    Attributes:
        session_id: The Claude session to start or continue.
        prompt: The user's prompt, when this segment starts a task.
        tools: The durable tools Claude may call.
        system_prompt: Optional system prompt.
        model: Optional model name.
        max_turns: Optional cap on engine turns within the segment.
        builtin_tools: Claude Code built-in tools to enable inside the engine.
        checkpoint: Where the session's last committed segment ended, or None when
            nothing is committed yet (the segment starts a new session).
        injected: Tool results to deliver to Claude, by ``tool_use_id``.
        segment_index: Index of this segment among all the agent's segments.
        live_output: Publish Claude's text to the Workflow's stream while running.
        fork: Continue in a copy of the session that ends at ``checkpoint``, because
            an earlier segment that did not commit may have written to the session.
            Retries (attempt 2 and later) always do this. (Session store only: a
            conversation the Workflow holds is always the committed one.)
        conversation: Where the conversation the Workflow holds can be read. The
            runner reads it when it has no session store.
        transcript: The committed conversation itself, for callers that drive a runner
            directly (tests); the Workflow passes ``conversation`` instead.
        tool_activities: Name patterns of Claude Code tools that pause the segment
            like durable tools, to run as their own Activities.
        workspace: The agent's own folder and its committed snapshot, for a runner
            with a ``workspace_store`` (others ignore it).
    """

    session_id: str
    prompt: str | None
    tools: list[ToolSpec]
    system_prompt: str | None = None
    model: str | None = None
    max_turns: int | None = None
    builtin_tools: list[str] = field(default_factory=list)
    checkpoint: str | None = None
    injected: dict[str, ToolOutcome] = field(default_factory=dict)
    segment_index: int = 0
    live_output: bool = False
    fork: bool = False
    conversation: ConversationRef | None = None
    transcript: list[dict[str, Any]] | None = None
    tool_activities: list[str] = field(default_factory=list)
    workspace: Workspace | None = None


@dataclass
class ToolStepRetry:
    """When a tool step whose call did not run may be tried again.

    For calls outside ``repeatable_tools``, the Workflow runs each attempt as its own
    Activity with one try. The tool step Activity decides from this whether its
    ``ToolCallNotRun`` failure may be tried again, and says so in the failure's
    details: the decision is then in the history, so a later change to the retry
    settings never changes it on replay.

    Attributes:
        maximum_attempts: Attempts in all (0: no limit), from
            ``tool_activity_retry_policy``.
        non_retryable_error_types: Error types not tried again: the step's own, or
            the type of the error that made it fail.
    """

    maximum_attempts: int = 0
    non_retryable_error_types: list[str] = field(default_factory=list)


@dataclass
class ToolStepInput:
    """Input of a tool step: one Claude Code tool call, run as its own Activity.

    Claude Code resumes a private copy of the session that ends before the call; a
    stand-in model on the Worker answers with the call, as Claude sent it, and Claude
    Code runs exactly that call.

    Attributes:
        session_id: The Claude session.
        checkpoint: Where the segment that paused at the call ended.
        call: The call to run.
        tools: The durable tools, as the segment declared them.
        builtin_tools: The Claude Code built-in tools the segment enabled.
        conversation: Where the conversation the Workflow holds can be read.
        transcript: The committed conversation itself, for direct calls (tests).
        attempt: Which attempt at the call this step is, from 1.
        retry: When the step may be tried again if its call did not run; None when
            Temporal's own retries run the step (``repeatable_tools``).
        workspace: The agent's own folder and its committed snapshot, for a runner
            with a ``workspace_store`` (others ignore it).
    """

    session_id: str
    checkpoint: str
    call: DeferredCall
    tools: list[ToolSpec] = field(default_factory=list)
    builtin_tools: list[str] = field(default_factory=list)
    conversation: ConversationRef | None = None
    transcript: list[dict[str, Any]] | None = None
    attempt: int = 1
    retry: ToolStepRetry | None = None
    workspace: Workspace | None = None


@dataclass
class SegmentOutput:
    """Output of one model segment.

    Attributes:
        session_id: The session the segment used (a new one after a fork).
        result: Claude's final answer, when the agent finished.
        deferred: The tool call Claude paused at, when there is one.
        checkpoint: Where this segment's turn ends in the session; the next segment
            receives it. Required for every segment that is not an error.
        cost_usd: Model cost of this segment, as reported by the SDK.
        is_error: Whether the segment failed in a way retrying cannot fix.
        error: A description of that failure.
        transcript_keep: When the Workflow holds the conversation: how many of its
            entries stay as they are. None when the conversation lives in a session
            store.
        transcript_add: The entries that follow them: what this segment added (and any
            entries it rewrote after the first ``transcript_keep``).
        external_storage: Whether the Worker's data converter has External Storage,
            which moves large payloads (such as a long conversation) out of the
            history.
        siblings: The other durable calls Claude made in the same message as
            ``deferred``. They did not run in the engine: the Workflow runs them with
            ``deferred``, and the next segment delivers every result.
        workspace: From a runner with a ``workspace_store``: the snapshot of the
            agent's folder after this segment. None from other runners.
    """

    session_id: str
    result: str | None = None
    deferred: DeferredCall | None = None
    checkpoint: str | None = None
    cost_usd: float = 0.0
    is_error: bool = False
    error: str | None = None
    transcript_keep: int | None = None
    transcript_add: list[dict[str, Any]] = field(default_factory=list)
    external_storage: bool = False
    siblings: list[DeferredCall] = field(default_factory=list)
    workspace: str | None = None


@dataclass
class AgentState:
    """What a new Workflow run needs to continue an agent after Continue-As-New.

    By default the conversation lives in the Workflow, so it moves to the new run here
    (``conversation``). With a session store, it stays in the store, and this stays
    small. Type the Workflow parameter that carries it as ``AgentState | None``.

    Attributes:
        session_id: The Claude session.
        checkpoint: Where the last committed segment ended in the session.
        segment_index: Index of the next segment.
        task_prompt: The prompt of the unfinished task, or None when idle.
        task_segments: Segments used by the unfinished task so far.
        pending: Tool results not yet delivered to Claude, by ``tool_use_id``.
        recent_call_ids: The most recent tool calls that already ran, so none can run
            again in a later run.
        segments: Segments run by this agent across all runs.
        tool_calls: Tool calls the Workflow handled for this agent across all runs
            (durable tools and Claude Code tools in ``tool_activities``).
        total_cost_usd: Model cost reported by the SDK across all runs.
        runs: Workflow runs this agent has used, counting the current one.
        fork_next: The last task stopped early, so the next segment continues in a
            copy of the session that ends at ``checkpoint``.
        stream: The live output stream's state, when live output is on.
        conversation: The conversation, when the Workflow holds it (no session
            store): each transcript entry as JSON text. The Workflow never reads
            inside an entry, and text moves to the next run much faster than nested
            objects.
        external_storage: Whether the Workers reported External Storage, so a large
            conversation can move to the next run.
        workspace: The agent's own folder and its committed snapshot (used by
            runners with a ``workspace_store``); None until the agent's first task.
    """

    session_id: str | None = None
    checkpoint: str | None = None
    segment_index: int = 0
    task_prompt: str | None = None
    task_segments: int = 0
    pending: dict[str, ToolOutcome] = field(default_factory=dict)
    recent_call_ids: list[str] = field(default_factory=list)
    segments: int = 0
    tool_calls: int = 0
    total_cost_usd: float = 0.0
    runs: int = 1
    fork_next: bool = False
    stream: WorkflowStreamState | None = None
    conversation: list[str] = field(default_factory=list)
    external_storage: bool = False
    workspace: Workspace | None = None
