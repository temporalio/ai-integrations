"""Plain data passed between the Workflow and the segment Activity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from temporalio.contrib.workflow_streams import WorkflowStreamState


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
    """

    content: Any = None
    is_error: bool = False
    blocks: list[dict[str, Any]] | None = None
    native_output: dict[str, Any] | None = None
    """Native shell output, used to render a child call's recorded result."""


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
    child_conversations: dict[str, ConversationRef] = field(default_factory=dict)
    child_subpaths: list[str] = field(default_factory=list)
    child_checkpoints: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolStepInput:
    """Input of a tool step: one Claude Code tool call, run as its own Activity.

    The step resumes the session where it paused at the call, and lets Claude Code run
    exactly that call.

    Attributes:
        session_id: The Claude session.
        checkpoint: Where the segment that paused at the call ended.
        call: The call to run.
        tools: The durable tools, as the segment declared them.
        builtin_tools: The Claude Code built-in tools the segment enabled.
        conversation: Where the conversation the Workflow holds can be read.
        transcript: The committed conversation itself, for direct calls (tests).
    """

    session_id: str
    checkpoint: str
    call: DeferredCall
    tools: list[ToolSpec] = field(default_factory=list)
    builtin_tools: list[str] = field(default_factory=list)
    conversation: ConversationRef | None = None
    transcript: list[dict[str, Any]] | None = None
    native: bool = False
    """Replay exactly this accepted child call against a local model."""


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
    child_transcripts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    child_keep: dict[str, int] = field(default_factory=dict)
    child_subpaths: list[str] = field(default_factory=list)
    child_checkpoints: dict[str, str] = field(default_factory=dict)
    native_calls: list[str] = field(default_factory=list)


@dataclass
class NativeRequest:
    """Internal segment-to-Workflow request, fenced by segment and Activity attempt."""

    agent: str
    segment: int
    attempt: int
    token: str
    call: DeferredCall | None = None
    child: str = ""


@dataclass
class NativeCallState:
    """A child call accepted by the Workflow, including an undelivered outcome."""

    segment: int
    child: str
    call: DeferredCall
    outcome: ToolOutcome | None = None
    delivered: bool = False


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
    child_conversations: dict[str, list[str]] = field(default_factory=dict)
    child_subpaths: list[str] = field(default_factory=list)
    child_checkpoints: dict[str, str] = field(default_factory=dict)
    native_calls: dict[str, NativeCallState] = field(default_factory=dict)
