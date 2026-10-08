"""Wire types shared by the Workflow and the segment Activity.

Plain dataclasses, so they cross the Activity boundary with Temporal's default payload converter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CODEX_RUN_SEGMENT_ACTIVITY = "openai_codex.run_segment"
"""Name of the Activity that runs one segment of a Codex turn."""


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
class CodexTokenUsage:
    """Token accounting for one model interaction; unreported counts are ``None``."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None


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
    usage: CodexTokenUsage | None = None


@dataclass
class CodexTurnResult:
    """The outcome of one :meth:`~temporalio.openai_codex.workflow.CodexSession.run` turn."""

    text: str
    usage: CodexTokenUsage | None = None
    segments: int = 1
