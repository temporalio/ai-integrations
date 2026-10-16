"""Serializable contracts of the Claude turn Activity.

Plain pydantic, no SDK imports: these cross the workflow/activity boundary and are built in
workflow code, so they must never hold an SDK object. The Activity converts them into
``ClaudeAgentOptions`` on its side.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _omit_if_none(value: Any) -> bool:
    """Leave an unset streaming field out of the activity payload.

    A turn with no topic must serialize exactly as it did before streaming existed, so a running
    workflow's replay still matches its history.
    """
    return value is None


class TokenUsage(BaseModel):
    """Provider-agnostic token accounting for one model interaction or a whole turn.

    Every field is optional: a count the provider did not report stays ``None``. ``total_tokens``
    is the provider's own grand total, not necessarily the sum of the parts.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None


RUN_CLAUDE_TURN_ACTIVITY = "claude_agent.run_turn"
"""Registered name of the turn Activity. Workflows dispatch it by name, never by function."""

CLAUDE_RESULT_ERROR_TYPE = "ClaudeResultError"
"""``ApplicationError.type`` raised when Claude ends the run with an error result."""

CLAUDE_TURN_REJECTED_ERROR_TYPE = "ClaudeTurnRejected"
"""``ApplicationError.type`` for a turn input the Activity refuses before starting the CLI."""

CLAUDE_TURN_CONFIG_ERROR_TYPE = "ClaudeTurnConfigError"
"""``ApplicationError.type`` for a setup mistake caught in workflow code before any Activity is
scheduled (an option ``run_turn`` fixes, a tool that is not an async function, an output type
without a JSON Schema). Non-retryable, so it fails the message, never the
workflow task: a workflow task that raised anything else would be retried forever."""

CLAUDE_SESSION_MIRROR_ERROR_TYPE = "ClaudeSessionMirrorError"
"""``ApplicationError.type`` when the ``SessionStore`` failed to persist part of the turn."""

CLAUDE_SESSION_DEGRADED_ERROR_TYPE = "ClaudeSessionDegraded"
"""``ApplicationError.type`` for a turn refused because its session's store copy is incomplete."""

CLAUDE_CLI_UNTRACKED_ERROR_TYPE = "ClaudeCLIUntracked"
"""``ApplicationError.type`` when the turn's CLI is running but the Activity holds no handle it
can kill it by (the launcher recorded no process group), so it cannot guarantee the CLI ends."""

CLAUDE_NATIVE_TOOL_UNSAFE_ERROR_TYPE = "ClaudeNativeToolUnsafe"
"""``ApplicationError.type`` when the CLI defers a native call again instead of applying the
recorded decision on it; the message fails instead."""

CLAUDE_STRUCTURED_OUTPUT_ERROR_TYPE = "ClaudeStructuredOutputInvalid"
"""``ApplicationError.type`` when Claude returned no structured output, or one that does not
validate into the requested output type."""

CLAUDE_TURN_TIMEOUT_ERROR_TYPE = "ClaudeTurnTimeout"
"""``ApplicationError.type`` when a turn reaches its ``timeout`` (``run_claude_turn``)."""

CLAUDE_TURN_BUDGET_ERROR_TYPE = "ClaudeTurnBudgetExhausted"
"""``ApplicationError.type`` when a native turn's runs have used up its ``max_turns`` or
``max_budget_usd``, so its deferred call is refused instead of resumed."""

CLAUDE_SESSION_LEASED_ERROR_TYPE = "ClaudeSessionLeased"
"""``ApplicationError.type`` when another attempt still holds the session's lease, so this one
refused to start a second CLI on the same transcript. Raised only after waiting about one lease
TTL (the heartbeat timeout) for the holder to stop renewing. Retryable."""

CLAUDE_SANDBOX_LOST_ERROR_TYPE = "ClaudeSandboxLost"
"""``ApplicationError.type`` when the turn's remote sandbox or its CLI was killed from outside.

The filesystem is back at the session's last committed snapshot, so the workflow runs the turn
again from that id. Retryable. This is not Claude Code's local Bash sandbox."""

CLAUDE_SESSION_LEASE_LOST_ERROR_TYPE = "ClaudeSessionLeaseLost"
"""``ApplicationError.type`` when the turn lost its session's lease while its CLI ran (another
attempt took the session over, or renewal could not be confirmed in time). The CLI was killed at
once, so the turn's outcome is unknown: non-retryable, never resumed automatically."""

CLAUDE_NATIVE_OUTCOME_UNKNOWN_ERROR_TYPE = "ClaudeNativeOutcomeUnknown"
"""``ApplicationError.type`` when a turn's worker was lost while an approved native call that is
not repeatable (``Bash``, ``Edit``, ...) may have run, and no record says it finished. The
message fails instead of resuming; the call is never run again unasked."""

CLAUDE_WORKSPACE_RESTORE_ERROR_TYPE = "ClaudeWorkspaceRestoreFailed"
"""``ApplicationError.type`` when the session's recorded workspace revision could not be
restored before the CLI started. Non-retryable; nothing ran."""

CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE = "ClaudeWorkspacePersistFailed"
"""``ApplicationError.type`` when the session's workspace could not be persisted after a run.
Non-retryable; the session keeps its previous revision."""

RESUMABLE_DETAIL = "resumable"
"""Key of the turn Activity's heartbeat details that says whether a lost attempt may be retried:
True only when the worker mirrors the transcript to a SessionStore (eagerly), the turn has a
session, a native-tool turn's lease store is shared, and the store has not dropped a batch of
this attempt. A remote sandbox sets it only when the lease store is shared: a host-local store
does not fail a lost sandbox over. A workflow reads it from the heartbeat timeout of a dead
worker. The same details
carry ``native_calls`` (the started-call ledger), ``workspace_revision``, ``sandbox_snapshot``
(the last committed remote snapshot id, absent or null when the worker runs the CLI locally),
``num_turns`` (model rounds seen so far), ``cost_usd`` (spend reported so far this attempt,
known only once its result arrived; absent on a heartbeat recorded before that field existed),
``cumulative_cost`` (as on :class:`ClaudeTurnResult`), ``mirror_errors`` (the batches
the store dropped), ``error_kind`` (the SDK's ``AssistantMessage.error`` last seen, if any)
and ``lease`` (the session lease's ``[generation, beat]`` once held)."""

NEVER_RESUMED_ERROR_KINDS = frozenset(
    {
        "authentication_failed",
        "billing_error",
        "invalid_request",
        # Claude Code 2.1.294 also reports these, outside the SDK's AssistantMessageError, and
        # stops on them itself. account_on_hold (an OAuth refresh) carries no HTTP status.
        "account_on_hold",
        "model_not_found",
        "oauth_org_not_allowed",
        "verification_required",
    }
)
"""``AssistantMessage.error`` kinds that the same request cannot get past, so an attempt that hit
one is never resumed."""

NATIVE_DEFAULT_MAX_TURNS = 10
"""``max_turns`` for a native turn whose options leave it unset. A broken hook denies every call
and Claude retries until the turn limit, so a native turn is never left unbounded; the release
gates finished within 4 to 8 rounds. ``run_claude_turn`` applies it to the whole message, across
the runs that resume deferred calls. Set ``ClaudeTurnOptions.max_turns`` to override it."""

BRIDGE_MCP_SERVER = "temporal"
"""Name of the in-process MCP server carrying the turn's tools. Claude sees each tool as
``mcp__temporal__<tool>``."""

BRIDGE_REQUEST_SIGNAL = "claude_agent.bridge_request"
"""Signal the turn Activity sends the workflow once per tool call."""

BRIDGE_POLL_UPDATE = "claude_agent.bridge_poll"
"""Update the turn Activity long-polls for tool results; at most one in flight per turn."""

TURN_PROGRESS_SIGNAL = "claude_agent.turn_progress"
"""Signal the turn Activity sends the workflow when its native-call ledger or workspace revision
changes mid-run (:class:`ClaudeTurnProgress`), when the input asks for it."""


CONTINUE_PROMPT = (
    "Continue: your previous attempt at this turn was interrupted by an infrastructure failure. "
    "Carry on from where it stopped, and do not repeat tool calls that already returned a result."
)
"""What a retried attempt sends when the store already holds the turn's prompt. Re-sending the
original prompt there makes the model restart the task and repeat its effects (the real-CLI
probe); the prompt is sent only when the store does not hold it."""


def mcp_tool_name(name: str) -> str:
    """The name Claude knows a bridged tool by (and that ``allowed_tools`` must list)."""
    return f"mcp__{BRIDGE_MCP_SERVER}__{name}"


def tool_call_key(name: str, arguments: dict[str, Any]) -> str:
    """The content of one tool call as a digest: its name and canonical JSON arguments.

    Sorted keys, so the order Claude serializes arguments in never gives one call two keys. The
    ``tool_use_id`` is left out: a resumed attempt re-issues a lost call under a new id.
    """
    canonical = json.dumps(
        {"name": name, "arguments": arguments}, sort_keys=True, default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class ClaudeTurnOptions(BaseModel):
    """The model-facing settings of one Claude turn, chosen in workflow code.

    Deliberately a small subset of ``ClaudeAgentOptions``: everything that would widen what Claude
    can touch (built-in tools, MCP servers, plugins, skills, filesystem settings) is fixed OFF by
    the Activity and not configurable here.
    """

    model: str | None = Field(
        default=None, description="Claude model id; CLI default if None."
    )
    system_prompt: str | None = Field(
        default=None,
        description="Replaces Claude Code's system prompt entirely; empty if None.",
    )
    max_turns: int | None = Field(
        default=None,
        description="Cap on agent-loop rounds. ``run_claude_turn`` applies it to the whole "
        "message, across the runs that resume deferred native calls.",
    )
    max_budget_usd: float | None = Field(
        default=None,
        description="Cap on spend for the turn; across resumes, like ``max_turns``.",
    )
    thinking: dict[str, Any] | None = Field(
        default=None,
        description='A ``ThinkingConfig``: {"type": "adaptive"}, {"type": "enabled", '
        '"budget_tokens": N} or {"type": "disabled"}.',
    )
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema for a structured reply (``output_format``). Set by "
        "``run_claude_turn(output_type=...)``; Claude returns it as ``structured_output``.",
    )


class ToolEffect(StrEnum):
    """What repeating one tool call costs, declared per tool rather than guessed per turn.

    This is the axis the integration actually decides on. Where a tool *runs* is settled
    elsewhere and is not ours to redeclare: a tool's placement comes from the workflow
    (``@agent.tool_defn`` inline, ``@agent.activity_tool_defn`` as an Activity), and a native
    tool runs inside the Claude Code CLI whatever we say. What is NOT settled anywhere is
    whether a lost turn may be run again, which depends on what a repeat of its tool calls would
    duplicate.
    """

    INHERENTLY_SAFE = "inherently_safe"
    """No effect outside the call's own result; repeating it costs nothing. The same claim
    ``@agent.activity_tool_defn(inherently_safe=True)`` makes to the approval gate."""

    IDEMPOTENT = "idempotent"
    """Has effects, but a repeat converges on the same state (a whole-value write, an upsert)."""

    AT_LEAST_ONCE = "at_least_once"
    """A repeat may duplicate the effect. The DEFAULT for anything undeclared: a tool nobody
    classified is a tool nobody thought about, and that is not evidence a repeat is free. The
    cost of being wrong is asymmetric — defaulting unsafe loses a turn the user can retry,
    defaulting safe charges a card twice and nobody sees it."""


REPEATABLE_EFFECTS = frozenset({ToolEffect.INHERENTLY_SAFE, ToolEffect.IDEMPOTENT})
"""The effects a lost turn may be re-run through. Everything else stops a resume."""

NATIVE_TOOL_EFFECTS: dict[str, ToolEffect] = {
    "Read": ToolEffect.INHERENTLY_SAFE,
    "Glob": ToolEffect.INHERENTLY_SAFE,
    "Grep": ToolEffect.INHERENTLY_SAFE,
    "WebFetch": ToolEffect.INHERENTLY_SAFE,
    "WebSearch": ToolEffect.INHERENTLY_SAFE,
    "BashOutput": ToolEffect.INHERENTLY_SAFE,
    # The CLI's own todo list, replaced wholesale: a repeat lands on the same state.
    "TodoWrite": ToolEffect.IDEMPOTENT,
    # Bash's effect is in its argument, not its name, so the name means the worst case.
    # Edit does not survive being applied twice. Write converges alone, but not after a later
    # edit of the same file.
    "Write": ToolEffect.AT_LEAST_ONCE,
    "Bash": ToolEffect.AT_LEAST_ONCE,
    "Edit": ToolEffect.AT_LEAST_ONCE,
    "NotebookEdit": ToolEffect.AT_LEAST_ONCE,
    "KillShell": ToolEffect.AT_LEAST_ONCE,
}
"""What repeating each Claude Code built-in costs, and the only built-ins a turn may request: a
built-in missing from the table is refused. True for the versions in ``NATIVE_TOOLS_VERIFIED``."""


READ_ONLY_NATIVE_TOOLS = frozenset(
    name
    for name, effect in NATIVE_TOOL_EFFECTS.items()
    if effect is ToolEffect.INHERENTLY_SAFE and name not in ("WebFetch", "WebSearch")
)
"""The built-ins ``native_auto_allow`` may name: those with no effect outside their result.
Not WebFetch or WebSearch: a URL or a search query can carry what an earlier call read off the
worker, and neither runs in the Bash sandbox's network allowlist. An auto-allowed Read, Glob or
Grep runs unasked only inside the session's working directory."""


def native_tool_effect(name: str) -> ToolEffect:
    """What repeating a native call costs. An unknown built-in is at-least-once."""
    return NATIVE_TOOL_EFFECTS.get(name, ToolEffect.AT_LEAST_ONCE)


def cli_tools(turn: ClaudeTurnInput) -> list[str]:
    """Built-in tools the CLI must offer: the message's, plus the tool of a resumed call."""
    tools = list(turn.native_tools)
    if turn.native_decision is not None and turn.native_decision.call.name not in tools:
        tools.append(turn.native_decision.call.name)
    return tools


class ClaudeToolDeclaration(BaseModel):
    """A workflow tool exposed to Claude for one turn: what the model sees, nothing it runs.
    The callable and its injections stay in workflow memory, keyed by the bridge token.
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ClaudeToolRequest(BaseModel):
    """One tool call, Activity -> workflow (:data:`BRIDGE_REQUEST_SIGNAL`). Carries only
    what the model chose; the workflow resolves the tool and its injections from the token.
    """

    token: str
    tool_use_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ordinal: int = Field(
        default=0,
        description="How many calls with the same name and arguments the message already made "
        "before this one: those of its earlier runs, those of this run the session's store "
        "holds, and those this attempt issued. A resumed attempt re-issuing a lost call gets "
        "the lost call's ordinal.",
    )


class ClaudeToolResult(BaseModel):
    """The model-visible outcome of one tool call, returned to Claude as an MCP result."""

    tool_use_id: str
    content: str
    is_error: bool = False


class ClaudeBridgePoll(BaseModel):
    """Long-poll for results past ``after`` (:data:`BRIDGE_POLL_UPDATE`)."""

    token: str
    after: int = 0


class ClaudeBridgeBatch(BaseModel):
    """Results appended since the poll's cursor; ``closed`` once the turn's bridge is shut."""

    results: list[ClaudeToolResult] = Field(default_factory=list)
    closed: bool = False


class ClaudeDeferredTool(BaseModel):
    """A native tool call the Activity's ``PreToolUse`` hook deferred: the run stopped before
    executing it, so the workflow can apply its approval policy without holding the CLI.
    """

    tool_use_id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ClaudeNativeDecision(BaseModel):
    """The workflow's decision on a deferred native call, carried by the run that resumes it."""

    call: ClaudeDeferredTool
    approved: bool
    reason: str | None = Field(
        default=None, description="What Claude is told on a denial."
    )
    settle: bool = Field(
        default=False,
        description="A denial of a call an earlier message abandoned, written into the "
        "transcript as the call's error result before the CLI starts instead of resumed, so the "
        "run is an ordinary turn on the message's own prompt.",
    )


NativeCallStatus = Literal["started", "finished", "failed", "reported"]


class ClaudeNativeCall(BaseModel):
    """One entry of a session's started-call ledger (``calls.json``, see ``_native_hook.py``): a
    native call the hook let run, ``started`` until its PostToolUse (``finished``) or
    PostToolUseFailure (``failed``), ``reported`` once a later run refused a repeat of it.
    """

    tool_use_id: str
    tool_name: str | None = None
    key: str
    run: str | None = None
    status: NativeCallStatus
    reported_run: str | None = None


_NATIVE_STATUS_RANK = {"started": 0, "reported": 1, "finished": 2, "failed": 2}


def merge_native_calls(base: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
    """Fold ledger entries ``calls`` into ``base`` in place: unknown ids are appended, and a known
    id keeps the later of the two statuses (started < reported < finished/failed), so a stale
    copy never reopens a settled call.
    """
    by_id = {c["tool_use_id"]: c for c in base}
    for call in calls:
        mine = by_id.get(call["tool_use_id"])
        if mine is None:
            by_id[call["tool_use_id"]] = dict(call)
            base.append(by_id[call["tool_use_id"]])
        elif _NATIVE_STATUS_RANK[call["status"]] > _NATIVE_STATUS_RANK[mine["status"]]:
            mine.update(call)


class ClaudeTurnProgress(BaseModel):
    """What a running turn Activity reports mid-run (:data:`TURN_PROGRESS_SIGNAL`), so the
    workflow knows it even if the worker dies before the result.
    """

    session_id: str
    attempt_id: str | None = Field(
        default=None,
        description="``ClaudeTurnInput.attempt_id`` of the reporting attempt. The workflow drops a "
        "report from any other attempt than the one it is waiting on.",
    )
    native_calls: list[ClaudeNativeCall] = Field(default_factory=list)
    workspace_revision: int | None = None
    mirror_errors: list[str] = Field(
        default_factory=list,
        description="Transcript batches the ``SessionStore`` dropped so far in this attempt; "
        "non-empty marks the session degraded.",
    )
    cost_usd: float | None = Field(
        default=None,
        description="Set only by a cancelled attempt, which returns no result: the spend its "
        "interrupted CLI reported, charged to the session's ``cost_usd``.",
    )
    cumulative_cost: bool = Field(
        default=False,
        description="``cost_usd`` includes the session's earlier runs, as in "
        "``ClaudeTurnResult.cumulative_cost``.",
    )


class ClaudeTurnInput(BaseModel):
    """Input of the turn Activity: one prompt submitted to one Claude agent loop."""

    prompt: str
    options: ClaudeTurnOptions = Field(default_factory=ClaudeTurnOptions)
    session_id: str | None = Field(
        default=None,
        description="Workflow-minted Claude session UUID. Also names the turn's working directory.",
    )
    resume: bool = Field(
        default=False,
        description="The workflow knows a run on this session completed, so its transcript must "
        "exist. The Activity also resumes whenever it finds the transcript (in the SessionStore, "
        "or without one on this worker's disk), and starts a new session otherwise.",
    )
    prompt_uuid: str | None = Field(
        default=None,
        description="Workflow-minted uuid the prompt is sent with; the CLI keeps it on the "
        "transcript entry. A retried attempt finds it in the SessionStore and then sends "
        "``CONTINUE_PROMPT`` instead of the prompt again.",
    )
    tools: list[ClaudeToolDeclaration] = Field(default_factory=list)
    bridge_token: str | None = Field(
        default=None,
        description="Active-turn token for the tool bridge; required with ``tools``.",
    )
    tool_ordinals: dict[str, int] | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Per ``tool_call_key``, the next ordinal after the calls the message's "
        "earlier runs made (its native defer -> approve -> resume chain), so this run numbers "
        "on from them. Omitted from the payload when unset so older histories replay.",
    )
    tool_calls: dict[str, tuple[str, int]] | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="The message's call record as of this attempt: per ``tool_use_id`` that "
        "ran or was answered from the record, its ``tool_call_key`` and ordinal. With the "
        "store's transcript, the Activity numbers new calls after these and re-queues only the "
        "slots no stored result answers. Omitted from the payload when unset.",
    )
    native_tools: list[str] = Field(
        default_factory=list,
        description="Claude built-in tools to enable (e.g. ``Bash``). Refused unless the worker "
        "opted in AND the installed SDK passed the native-tool release gate.",
    )
    native_auto_allow: list[str] | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Native tools from ``READ_ONLY_NATIVE_TOOLS`` the hook allows without a "
        "deferral, after its usual denials; each call is still recorded in the started-call "
        "ledger. Omitted from the payload when unset so older histories replay.",
    )
    native_decision: ClaudeNativeDecision | None = Field(
        default=None,
        description="The decision on the deferred native call this resumed run continues; the "
        "hook allows (or denies) exactly that call.",
    )
    native_denied: list[ClaudeDeferredTool] = Field(
        default_factory=list,
        description="Native calls already denied in the session; the hook denies any repeat.",
    )
    native_rerequests: int = Field(
        default=0, description="Native calls the session already made after a denial."
    )
    resumes_lost_attempt: bool = Field(
        default=False,
        description="This run resumes a turn whose earlier attempt was lost (its worker died or "
        "timed out) and may still be running. Refused unless the worker's lease store can see "
        "that attempt's lease. Such a run does not publish the approved call's ``tool_start`` "
        "again.",
    )
    lost_lease: tuple[int, int] | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="The session lease ``(generation, beat)`` the lost attempt's last heartbeat "
        "reported. Temporal timed that heartbeat out, so a lease still at it has been silent for "
        "a whole heartbeat timeout and is claimed without waiting one more.",
    )
    attempt: int = Field(
        default=1,
        description="Which attempt at this message's turn the run is. ``run_claude_turn`` "
        "schedules each retry as a new Activity, whose own attempt number stays 1.",
    )
    resumes_api_error: bool = Field(
        default=False,
        description="This run resumes a turn whose previous attempt ended on a transient "
        "Anthropic API error.",
    )
    native_calls: list[ClaudeNativeCall] = Field(
        default_factory=list,
        description="The session's started-call ledger as the workflow knows it. Merged into "
        "the worker's ledger before the CLI starts, so a call started on a lost worker is "
        "refused, not re-run, wherever the session runs next.",
    )
    workspace_revision: int | None = Field(
        default=None,
        description="The session's workspace revision the workflow recorded; restored before "
        "the CLI starts when the worker has a ``WorkspaceStore``. None: nothing persisted yet.",
    )
    sandbox_snapshot: str | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Opaque filesystem snapshot id from the worker's ``SandboxRuntime``. The "
        "next attempt's sandbox starts from it. None on a local worker and until a sandboxed "
        "turn has committed one. Omitted from the payload when unset so older histories replay.",
    )
    report_progress: bool = Field(
        default=False,
        description="Signal ledger and workspace changes mid-run (:data:`TURN_PROGRESS_SIGNAL`). "
        "Only for a workflow that handles that signal.",
    )
    streaming_topic: str | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Workflow Stream topic for advisory partials (:class:`ClaudePartialEvent`). "
        "None publishes nothing and adds nothing to this payload. The workflow must host a "
        "``WorkflowStream`` or :func:`workflow.run_turn` fails before the Activity starts.",
    )
    streaming_batch_interval_ms: int | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="How long the Activity coalesces partials before signaling the topic or "
        "calling the worker's ``partial_sink``, in milliseconds. None uses 100.",
    )
    streaming_context: Any = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Opaque JSON value handed to the worker's ``partial_sink`` with each "
        "batch of this turn's partials, such as the caller's own turn or channel id. The plugin never "
        "reads it. Omitted from the payload when unset so older histories replay.",
    )

    @property
    def attempt_id(self) -> str:
        """Names this attempt of this message's turn, across the workflow and the Activity."""
        return f"{self.prompt_uuid}.{self.attempt}"


class ClaudeTurnResult(BaseModel):
    """What the Activity returns once Claude reports a successful result."""

    text: str | None = Field(default=None, description="``ResultMessage.result``.")
    structured_output: Any = Field(
        default=None,
        description="``ResultMessage.structured_output`` as JSON; validated workflow-side.",
    )
    session_id: str | None = None
    num_turns: int = 0
    stop_reason: str | None = None
    terminal_reason: str | None = None
    total_cost_usd: float | None = Field(
        default=None,
        description="``ResultMessage.total_cost_usd``, or the sum of ``model_usage`` costs when "
        "that is missing or zero.",
    )
    cumulative_cost: bool = Field(
        default=False,
        description="``total_cost_usd`` includes the session's earlier runs, which Claude Code "
        "restored from the transcript it resumed (its own file, on the worker or in a sandbox "
        "snapshot). A run resumed from a ``SessionStore`` reports only its own spend.",
    )
    duration_ms: int = 0
    duration_api_ms: int | None = Field(default=None, exclude_if=_omit_if_none)
    usage: TokenUsage | None = Field(
        default=None,
        description="Cumulative usage across every model round of the turn.",
    )
    model_usage: dict[str, dict[str, Any]] | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="``ResultMessage.model_usage``: per-model usage and cost, keyed by model, "
        "with the CLI's camelCase fields.",
    )
    permission_denials: list[dict[str, Any]] | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="``ResultMessage.permission_denials``: the tool calls the CLI refused.",
    )
    sandbox_violations: int | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Entries in the ``<sandbox_violations>`` blocks of this run's tool results; "
        "None when there were none. A signal, not enforcement: Claude Code does not report "
        "every sandbox denial.",
    )
    mirror_errors: list[str] = Field(
        default_factory=list,
        description="``MirrorErrorMessage`` errors: transcript batches the ``SessionStore`` "
        "failed to persist. Non-empty means the store copy of the session is incomplete.",
    )
    deferred: ClaudeDeferredTool | None = Field(
        default=None,
        description="Set when the run stopped on a deferred native tool call.",
    )
    native_rerequests: int = Field(
        default=0,
        description="Native calls this run made after a denial, counted by the hook.",
    )
    native_calls: list[ClaudeNativeCall] = Field(
        default_factory=list,
        description="The worker's started-call ledger after the run.",
    )
    workspace_revision: int | None = Field(
        default=None,
        description="The workspace revision persisted after the run; None without a "
        "``WorkspaceStore``.",
    )
    sandbox_snapshot: str | None = Field(
        default=None,
        exclude_if=_omit_if_none,
        description="Opaque filesystem snapshot id taken after the run. None when the worker "
        "runs the CLI locally, or when the sandbox was lost before a snapshot could be taken.",
    )
    output: Any | None = Field(
        default=None,
        description="Structured output validated into the caller's model. The Activity leaves "
        "this unset; the workflow sets it when ``output_type`` was given.",
    )


class ClaudePartialEvent(BaseModel):
    """One advisory partial from a Claude turn attempt.

    The Activity publishes one of these for each CLI ``StreamEvent`` it consumes, on
    ``ClaudeTurnInput.streaming_topic`` and to the worker's ``partial_sink``. Delivery is at-least-once and is not a durable record:
    a subscriber can see the same ``(attempt_id, sequence)`` more than once, and events from an
    attempt that later fails or is abandoned after its worker is lost can still arrive. ``sequence``
    counts only within one ``attempt_id`` (``prompt_uuid.attempt``), from 0, in consume order. It
    is not an order across attempts. The durable result of the turn is :class:`ClaudeTurnResult`.

    ``text`` is set only for an Anthropic ``text_delta`` and ``thinking`` only for a
    ``thinking_delta``. A signature, tool-input fragment, or completed message is not copied into
    either field. ``parent_tool_use_id`` is set when the event belongs to a Claude-internal
    subagent, so that text is not the turn's own reply.
    """

    attempt_id: str
    sequence: int
    kind: str = Field(
        description="Anthropic stream event type, such as ``content_block_delta``."
    )
    index: int | None = Field(
        default=None, description="Content-block index when the event carries one."
    )
    text: str | None = None
    thinking: str | None = None
    parent_tool_use_id: str | None = None


def partial_event(
    event: dict[str, Any],
    *,
    attempt_id: str,
    sequence: int,
    parent_tool_use_id: str | None = None,
) -> ClaudePartialEvent:
    """One public partial from a CLI stream event dict.

    The dict is the Anthropic event inside a ``StreamEvent``, not the SDK object. Only a
    ``text_delta`` fills ``text`` and only a ``thinking_delta`` fills ``thinking``.
    """
    raw_kind = event.get("type")
    raw_index = event.get("index")
    text: str | None = None
    thinking: str | None = None
    delta = event.get("delta")
    if isinstance(delta, dict):
        if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
            text = delta["text"]
        elif delta.get("type") == "thinking_delta" and isinstance(
            delta.get("thinking"), str
        ):
            thinking = delta["thinking"]
    parent = parent_tool_use_id if isinstance(parent_tool_use_id, str) else None
    return ClaudePartialEvent(
        attempt_id=attempt_id,
        sequence=sequence,
        kind=raw_kind if isinstance(raw_kind, str) else "",
        index=raw_index if type(raw_index) is int else None,
        text=text,
        thinking=thinking,
        parent_tool_use_id=parent,
    )


def anthropic_usage(usage: dict[str, Any] | None) -> TokenUsage | None:
    """Map an Anthropic ``usage`` dict onto :class:`TokenUsage`.

    Anthropic's ``input_tokens`` excludes cache reads and writes; this mapping counts the whole
    prompt (as the OpenAI and Gemini mappings do), so they are added back in. Thinking tokens
    (``output_tokens_details.thinking_tokens``) are a slice of ``output_tokens``, as the OpenAI
    mapping's reasoning tokens are.
    """
    if not usage:
        return None
    fresh = usage.get("input_tokens") or 0
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    output = usage.get("output_tokens")
    details = usage.get("output_tokens_details") or {}
    prompt = fresh + cache_read + cache_write
    return TokenUsage(
        input_tokens=prompt,
        output_tokens=output,
        thought_tokens=details.get("thinking_tokens"),
        cached_tokens=cache_read or None,
        total_tokens=prompt + (output or 0),
    )
