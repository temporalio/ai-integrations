"""Wire types shared by the Workflow and the segment Activity.

Plain dataclasses, so they cross the Activity boundary with Temporal's default payload converter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

CODEX_RUN_SEGMENT_ACTIVITY = "openai_codex.run_segment"
"""Name of the Activity that runs one segment of a Codex turn."""

CODEX_APPROVAL_UPDATE = "openai_codex.approval"
"""Name of the Workflow Update a segment calls to ask for an approval decision."""


@dataclass
class CodexToolSpec:
    """A host tool as Codex sees it (a ``dynamicTools`` entry)."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodexPendingCall:
    """A host-tool call the model made, awaiting execution by the Workflow."""

    call_id: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodexApprovalRequest:
    """Codex asks to run a command or apply a file change; the Workflow decides.

    ``kind`` is ``"command"`` or ``"file_change"``. For a command, ``command`` and ``cwd`` are set; for
    a file change, ``changes`` lists what would change (``path``, ``kind`` and ``diff`` per file).
    ``item_id`` is Codex's id for the call and stays stable for the life of that call.
    """

    kind: str
    item_id: str
    command: str | None = None
    cwd: str | None = None
    reason: str | None = None
    changes: list[dict[str, Any]] = field(default_factory=list)
    # Which segment Activity (and which attempt of it) is asking. A retried segment asks again with a
    # higher attempt, which tells the Workflow that the earlier attempt's questions are dead.
    segment: str | None = None
    attempt: int = 1


@dataclass
class CodexApprovalDecision:
    """The Workflow's answer to a :class:`CodexApprovalRequest`.

    ``approved=False`` makes Codex skip the action; the model is told it was rejected, and ``reason``
    is for your own records. ``interrupt=True`` also ends the turn.
    """

    approved: bool
    reason: str | None = None
    interrupt: bool = False


@dataclass
class CodexTokenUsage:
    """Token accounting for one model interaction; unreported counts are ``None``."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    def plus(self, other: CodexTokenUsage | None) -> CodexTokenUsage:
        """The counts of this and ``other`` added together."""
        values: dict[str, int | None] = {}
        for f in fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name, None)
            values[f.name] = (
                None if mine is None and theirs is None else (mine or 0) + (theirs or 0)
            )
        return CodexTokenUsage(**values)


@dataclass
class CodexSegmentInput:
    """Input of the segment Activity."""

    prompt: str
    thread_id: str | None = None
    rollout_name: str | None = None
    rollout: str = ""
    committed_lines: int = 0
    # call_id -> real tool output, injected into the resumed thread before the turn starts.
    inject: dict[str, str] = field(default_factory=dict)
    tools: list[CodexToolSpec] = field(default_factory=list)
    instructions: str | None = None
    model: str | None = None
    cwd: str | None = None
    sandbox: str = "read-only"
    # Native mode: Codex's own tools (shell, apply_patch, ...) stay on and every action that needs
    # approval is sent to the Workflow as an Update. Host-only mode turns them off.
    native_tools: bool = False
    approval_policy: str = "untrusted"
    # Opaque JSON handed to the Worker's observer factory (see CodexPlugin), if one is configured.
    observer_context: dict[str, Any] | None = None


@dataclass
class CodexSegmentResult:
    """Result of the segment Activity."""

    thread_id: str
    rollout_name: str
    tail: str  # rollout lines this segment appended; the Workflow appends them
    status: str  # "done" | "tool_call"
    final_response: str = ""
    call: CodexPendingCall | None = None
    usage: CodexTokenUsage | None = None  # used by this segment alone
    thread_total: CodexTokenUsage | None = (
        None  # the thread's cumulative counts afterwards
    )


@dataclass
class CodexTurnResult:
    """The outcome of one :meth:`~temporalio.openai_codex.workflow.CodexSession.run` turn."""

    text: str
    usage: CodexTokenUsage | None = None
    segments: int = 1
