"""The real runner: drives the Claude Agent SDK and its bundled Claude Code engine.

How one segment works:

1. Durable tools are declared to Claude as in-process SDK MCP tools.
2. A settings-file PreToolUse command hook answers "defer" when Claude calls one, or a
   Claude Code tool in ``tool_activities``. The run stops and the SDK returns
   ``ResultMessage.deferred_tool_use``. The Workflow runs the call as a Temporal
   Activity: a durable tool's own Activity, or a tool step (``run_tool_step``).
3. The next segment resumes the session and sends the stored result as a normal
   ``tool_result`` message for that ``tool_use_id``. Claude continues and can pause
   again. In a segment, the hook keeps answering "defer" (never "allow") for these
   calls.

A tool step runs a Claude Code call that paused a segment with Brian Strauch's bounded
native call replay (from his hybrid prototype in temporalio/ai-integrations): Claude
Code resumes a private copy of the conversation that ends before the call, a stand-in
model on 127.0.0.1 (``_stand_in``) answers with the call itself, the exact ``tool_use``
block Claude sent, the hook allows exactly that call, and ``max_turns=1`` ends the turn
after it. So Claude Code runs the call in an ordinary turn and writes its own result
entry, which the step returns for an Edit or a Write (``ToolOutcome.entry``). No real
model is asked, and the command still sees the Worker's own environment.

For an Edit or a Write (``RECORDED_TOOLS``), the next segment puts that entry, or one
of the same shape (``_record``), where the call's result belongs, and Claude Code
continues the turn itself (``CONTINUE_ENV``): Claude sees the result as Claude Code
recorded it, then Claude Code's own line "Continue from where you left off.". Claude
Code does not check the call again. (Such a result sent as a new message after the
pause is checked again: Claude would be told that a file it edited in the step "has
been modified since read", anthropics/claude-code#99041.) The results of other tools
go as a message, as in step 3: Claude Code does not check them again.

Checkpoints make retries clean. After a segment, the runner reads the session back
from the session store and returns where the next segment must continue (the last
transcript entry, or the paused call's deferral marker after parallel calls) as the
segment's checkpoint; the Workflow stores it with the segment's result.

Where the conversation lives:

- By default (no ``session_store``), in the Workflow. The segment reads the committed
  transcript with a Query on its own Workflow, seeds a fresh in-memory session store
  with it (up to the checkpoint), runs, and returns what changed
  (``transcript_keep`` and ``transcript_add``); the Workflow splices it in. Every
  attempt starts from what the Workflow committed, so a retry can never build on
  an unfinished attempt, and any Worker can run any step with no shared storage.
- With a ``session_store``, in the store. Reading the checkpoint back also proves
  the turn reached the store, so a segment only commits what every Worker can
  resume. A segment that runs again (a retry after a crash or a timeout, or the
  first segment after a failed task) cannot trust what an unfinished attempt wrote,
  so it continues in a copy of the session that ends at the checkpoint
  (``fork_session_via_store``), and Claude decides again from there.

Why not "resume and let the hook allow the deferred call"? On that auto-resume path
the engine ignores a later "defer" when the resume also sends a user message, and the
SDK's resume always does (reported: anthropics/claude-code#97196).

Why a copy, and not ``resume_session_at``? Tested with Claude Code 2.1.281: resuming
at the checkpoint of a session that holds a later attempt's entries (in place or with
``fork_session``) makes the engine treat the paused call as interrupted: it answers
the call with an error placeholder and drops the delivered result. In a copy that
ends at the checkpoint, the paused call resumes normally. The same drop happens when a
user row follows the paused call (anthropics/claude-code#97358), which is why the
checkpoint after parallel calls is the deferral marker (see ``_resume_point``).

Parallel calls: the engine keeps one paused call per run, so the hook defers the first
call of a message that runs as an Activity and denies the calls after it. The segment
reports the denied durable calls (``siblings``); the Workflow runs them with the
paused one, and the next segment puts their results where the engine resumes (see
``_deliver``), so Claude sees every call of the message with its result. A denied
Claude Code call keeps its denial, and Claude calls it again. The engine starts a call
as soon as its block arrives, so with a model that takes its time between the calls
of a message, the calls after the paused one come after its pause in the session:
they move back first (see ``_blocks_first``).
"""

from __future__ import annotations

import asyncio
import atexit
import collections
import contextlib
import contextvars
import dataclasses
import fnmatch
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Collection, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    InMemorySessionStore,
    MirrorErrorMessage,
    ResultError,
    ResultMessage,
    SessionStore,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    fork_session_via_store,
    project_key_for_directory,
    query,
    tool,
)
from temporalio import activity
from temporalio.exceptions import ApplicationError

from ._conversation import external_storage_on, read_conversation, too_large
from ._defer_hook import (
    ALLOWED,
    ANSWERED,
    DECISION,
    REASON_KEYS,
    STOPPED,
    TURN_GATE,
    TURN_GO,
    TURN_WAITS,
    WAITING,
    WORKER_LOCK,
    denial_name,
    input_digest,
)
from ._defer_hook import FAILED as HOOK_FAILED
from ._defer_hook import NOT_RUN as NOT_RUN_REASON
from ._events import emit
from ._launcher import CHECK as LAUNCHER_CHECK
from ._launcher import WORKER_PID
from ._models import (
    TOOL_CALL_INTERRUPTED,
    TOOL_CALL_NOT_RUN,
    DeferredCall,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolSpec,
    ToolStepInput,
    Workspace,
)
from ._stand_in import RecordedCall, StandInModel
from ._workspace import DEFAULT_MAX_BYTES, Workspaces, WorkspaceStore

_AGENT_FOLDER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "temporalio_claude_agent_sdk_agent_folder", default=None
)
"""The agent's own folder, while a step of a runner with a workspace store runs (see
``ClaudeAgentSdkRunner._folder``)."""

ENV_AUTH = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
"""Logins that survive resumes.

A Claude app login (Keychain OAuth) does not: when the SDK resumes a session (every
segment after the first) it copies the login without its refresh token, so once the
short-lived access token expires, resumed segments fail with "OAuth session expired".
"""

SERVER = "durable"
PREFIX = f"mcp__{SERVER}__"
ONE_TOOL_HINT = "Call at most one tool per message, then wait for its result before calling another."
"""Asks Claude for one call per message (``one_tool_at_a_time``; off by default)."""

PAUSE_CONTRACT = (
    "Durable tools never run inside the engine (the in-engine tool only returns an "
    "error), so nothing ran outside Temporal. This segment stops instead of "
    "continuing. Durable tools cannot be called from subagents; otherwise, use a "
    "Claude Code version that passes this plugin's test suite."
)

MIN_BUFFER_BYTES = 64 * 1024 * 1024
"""Smallest limit for one message from the engine (the SDK's default is 1 MiB).

The engine echoes each delivered tool result as one JSON line, so a result over the
SDK's default could never reach Claude.
"""

STEP_PROVIDER_ENV = {
    # Provider settings in settings files (user, project, .claude.json, managed) no
    # longer apply, so they cannot send a tool step's model calls elsewhere.
    "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST": "1",
    "ANTHROPIC_AUTH_TOKEN": "",
    "CLAUDE_CODE_OAUTH_TOKEN": "",
    "ANTHROPIC_UNIX_SOCKET": "",
    "CLAUDE_CODE_API_BASE_URL": "",
    "CLAUDE_CODE_USE_BEDROCK": "",
    "CLAUDE_CODE_USE_VERTEX": "",
    "CLAUDE_CODE_USE_FOUNDRY": "",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS": "",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD": "",
    "CLAUDE_CODE_USE_MANTLE": "",
    "CLAUDE_CODE_USE_GATEWAY": "",
}
"""A tool step talks to the local stand-in model only, whatever provider the Worker uses.

``ANTHROPIC_API_KEY`` is the key the stand-in gave the step (``StandInModel.serve``).
"""

RECORDED_TOOLS = ("Edit", "Write")
"""Claude Code tools whose result goes into the conversation as Claude Code's own record
of it (see ``_place``), not as a message.

Claude Code checks a paused Edit or Write again when its result arrives as a message,
finds the file changed since Claude read it (the step changed it), and tells Claude
"File has been modified since read" (anthropics/claude-code#99041). A command's or an
MCP tool's result is not checked again, so it goes as a message, as before.
"""

RESUME = "CLAUDE_CODE_RESUME_INTERRUPTED_TURN"
"""Claude Code's switch to continue an interrupted turn by itself when it starts."""

STEP_ENV = {
    # Every tool is in the stand-in's request, so it can answer with the step's call
    # (Claude Code turns tool search off by itself for a host that is not Anthropic's,
    # unless the Worker's environment turns it on).
    "ENABLE_TOOL_SEARCH": "false",
    # The step's engine never compacts the conversation: it would ask the stand-in
    # for a summary.
    "DISABLE_AUTO_COMPACT": "1",
    # The step's copy of the conversation can end like an interrupted turn: Claude
    # Code must not start a turn of its own beside the one the step's prompt starts.
    RESUME: "",
}
"""More settings of a tool step's engine (commands see the Worker's own values)."""

CONTINUE_ENV = {
    RESUME: "1",
    "CLAUDE_CODE_RESUME_INTERRUPTED_TURN_MAX_AGE_MS": "0",
    "CLAUDE_CODE_RESUME_PROMPT": "",
    "CLAUDE_CODE_RESUME_REASON": "",
}
"""Set for a segment that continues after a tool step's record (see ``_place``).

The conversation then ends with a tool result, which Claude Code takes for a turn that
was interrupted. With these, it continues that turn by itself, with no new message
(tested on Claude Code 2.1.273, 2.1.274 and 2.1.288): Claude sees the result, then
Claude Code's own line "Continue from where you left off.". With a message instead,
Claude Code first closes the turn with a line in Claude's place that Claude did not
write, "No response requested." (tested). ``MAX_AGE_MS=0``: however long the step took, the turn continues
(the Worker's environment could set a maximum age). ``PROMPT`` and ``REASON`` empty:
Claude Code's own line and reason, whatever the Worker's environment says. Other
segments get ``NEW_TURN_ENV``. Commands see the Worker's own values.
"""

NEW_TURN_ENV = {
    RESUME: "",
    "CLAUDE_CODE_RESUME_PROMPT": "",
    "CLAUDE_CODE_RESUME_REASON": "",
}
"""Set for every other segment, whatever the Worker's environment says.

Claude Code does not continue an interrupted turn by itself, and Claude reads Claude
Code's own lines: a new task right after a tool step's result comes after Claude
Code's answer in Claude's place, "No response requested." (tested; Claude Code
2.1.280 and older first add their line for an interrupted turn, "Continue from
where you left off.").
Commands see the Worker's own values.
"""

CONTINUE_START_SECONDS = 120.0
"""How long a continuing segment waits for Claude Code to start the turn by itself,
from the engine's start or its last notice before the turn (see ``_StartWatch``).

Claude Code starts it within seconds (tested). One that does not would wait for a message
forever: the segment ends its input, so the engine exits, and fails (see
``CONTINUE_ATTEMPTS``).
"""

CONTINUE_ATTEMPTS = 3
"""Attempts of a segment whose Claude Code did not continue the turn by itself.

Temporal runs it again until then (a slow start can pass); after that, the task fails
with an error that names the Claude Code version, instead of being retried forever."""

STEP_MAX_TURNS = 1
"""A tool step's turn ends right after its call (a test may let it go on)."""

SAVED_OUTPUT_TAIL = 4096
"""Bytes of the end of an output Claude Code saved to a file, added to the preview."""

ENGINE_ENV = {
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR": "1",
}
"""Set for every engine run.

With background tasks (for example a subagent running in the background), the engine
keeps working after it paused at a durable call. And every shell command starts in the
working directory: a ``cd`` would otherwise carry over inside one engine run but not
to the next one (each segment and tool step is a new run), so a later command could
run somewhere else than Claude expects.
"""

SHELL_HINT = (
    "Each shell command starts in the working directory: a `cd` does not carry over "
    "to the next command, so use absolute paths or `cd <dir> && <command>`."
)
"""Added to the system prompt when Bash or PowerShell is on: Claude Code's own tool
description says the working directory persists, which is not so here."""

_SHELL_TOOLS = ("Bash", "PowerShell")

DEFAULT_PERMISSION_MODE = "default"
"""Permission mode unless ``extra_options`` sets one.

Since Claude Code 2.1.285, a run with no permission mode uses auto mode when telemetry
is off or the provider is Bedrock, Vertex or Foundry. Auto mode asks the model whether
each tool call is safe, and blocked a subagent in this plugin's tests (claude-agent-sdk
0.2.163). Setting the mode keeps every engine version on the same rules: tools in
``allowed_tools`` run, others are refused.
"""

_RESERVED_OPTIONS = {
    "tools": "DurableClaudeAgent(builtin_tools=...)",
    "model": "DurableClaudeAgent(model=...) or ClaudeAgentSdkRunner(model=...)",
    "max_turns": "DurableClaudeAgent(max_turns=...)",
    "max_budget_usd": "ClaudeAgentSdkRunner(max_budget_usd=...)",
    "cwd": "ClaudeAgentSdkRunner(cwd=...)",
    "cli_path": "ClaudeAgentSdkRunner(cli_path=...)",
    "session_store": "ClaudeAgentSdkRunner(session_store=...)",
}
"""``extra_options`` keys the agent or the runner already sets, with where to set them."""

_MANAGED_OPTIONS = frozenset(
    {
        "settings",
        "strict_mcp_config",
        "resume",
        "session_id",
        "continue_conversation",
        "fork_session",
        "resume_session_at",
        "resume_drops_turn",
    }
)
"""``extra_options`` keys the plugin needs for pausing, resuming and checkpoints."""

_MANAGED_FLAGS = frozenset(
    {
        "settings",
        "resume",
        "session-id",
        "continue",
        "fork-session",
        "resume-session-at",
        "mcp-config",
        "strict-mcp-config",
        "system-prompt",
        "tools",
        "allowedTools",
        "allowed-tools",
        "model",
        "max-turns",
    }
)
"""Engine flags the plugin sets, refused in ``extra_options["extra_args"]`` too."""

_APPROVING_FLAGS = frozenset(
    {
        "permission-mode",
        "dangerously-skip-permissions",
        "allow-dangerously-skip-permissions",
        "permission-prompt-tool",
    }
)
"""Engine flags that approve tool calls, refused in ``extra_options["extra_args"]``.

``extra_options`` sets the same with ``permission_mode`` and
``permission_prompt_tool_name``, which a tool step replaces: there, nothing but the
hook approves the call. A flag would come after the step's own and win."""


def _check_extra_options(extra: dict[str, Any]) -> None:
    """Refuse ``extra_options`` that would replace what the plugin relies on.

    ``env``, ``mcp_servers``, ``allowed_tools`` and ``max_buffer_size`` are merged
    with the plugin's own values instead.

    Raises:
        ValueError: If ``extra`` sets a reserved option.
    """
    problems = [
        f"{key} (use {_RESERVED_OPTIONS[key]})"
        for key in sorted(extra)
        if key in _RESERVED_OPTIONS
    ] + [
        f"{key} (the plugin sets it to pause and resume sessions)"
        for key in sorted(extra)
        if key in _MANAGED_OPTIONS
    ]
    flags = extra.get("extra_args") or {}
    problems += [
        f"extra_args[{flag!r}] (the plugin sets that engine flag)"
        for flag in sorted(flags)
        if flag.lstrip("-") in _MANAGED_FLAGS
    ] + [
        f"extra_args[{flag!r}] (set permission_mode or permission_prompt_tool_name "
        "in extra_options instead: a tool step must be able to replace it)"
        for flag in sorted(flags)
        if flag.lstrip("-") in _APPROVING_FLAGS
    ]
    if extra.get("enable_file_checkpointing"):
        problems.append(
            "enable_file_checkpointing (Claude Code cannot combine it with the session "
            "store every engine run uses)"
        )
    servers = extra.get("mcp_servers") or {}
    if not isinstance(servers, dict):
        problems.append("mcp_servers (pass a dict of servers; they are merged)")
    elif SERVER in servers:
        problems.append(f"mcp_servers[{SERVER!r}] (the durable tools use that name)")
    if problems:
        raise ValueError("extra_options cannot set: " + "; ".join(problems))


def _engine_cwd(directory: str) -> str:
    """The working directory as a process started in ``directory`` reports it.

    That is the path the engine derives its session key from: on macOS, the form
    stored on disk, which can be decomposed even when ``directory`` is not.
    """
    code = "import os, sys; sys.stdout.buffer.write(os.fsencode(os.getcwd()))"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=directory,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    return os.fsdecode(out)


def _key_mismatch(directory: str) -> str | None:
    """How Claude Code and the SDK would store this directory's sessions under two keys.

    The SDK derives the key from the NFC form of the real path and counts code
    points; the engine uses its working directory as it is and counts UTF-16 units.
    They agree when the engine's path is exactly the SDK's and has no character
    outside the Basic Multilingual Plane. Otherwise no segment could find its own
    turn in the store, and every one would be retried.

    Returns:
        A description of the difference, or None if the keys agree.
    """
    ours = unicodedata.normalize("NFC", os.path.realpath(directory))
    engine = _engine_cwd(directory)
    if engine != ours:
        return f"the engine sees {engine!r}, the SDK uses {ours!r}"
    if any(ord(char) > 0xFFFF for char in engine):
        return "it has characters outside the Basic Multilingual Plane (such as emoji)"
    return None


MIN_ENGINE_VERSION = (2, 1, 273)
"""Oldest Claude Code engine that keeps every tool result (bundled in claude-agent-sdk 0.2.153).

Tested: when Claude calls two tools in one message, Claude Code 2.1.259 replaces the
result of the paused call with "[Tool result missing due to internal error]" in the
next request, so Claude asks for the same tool again and it would run twice.
"""


def _version(text: str) -> tuple[int, ...] | None:
    parts = text.split()[0].split(".") if text.strip() else []
    if len(parts) < 3 or not all(p.isdigit() for p in parts[:3]):
        return None
    return tuple(int(p) for p in parts[:3])


def _too_old(reported: str | None) -> bool:
    parsed = _version(reported or "")
    return parsed is not None and parsed < MIN_ENGINE_VERSION


def _too_old_output(session_id: str, reported: str) -> SegmentOutput:
    minimum = ".".join(map(str, MIN_ENGINE_VERSION))
    return SegmentOutput(
        session_id=session_id,
        is_error=True,
        error=(
            f"Claude Code {reported} is older than {minimum}. Older engines can drop a "
            "tool result when Claude calls several tools in one message; Claude then "
            "asks for the tool again, so it could run twice. Use claude-agent-sdk "
            f"0.2.153 or newer, or set cli_path to Claude Code {minimum} or newer. "
            "No durable tool ran."
        ),
    )


FINAL_RESULT_ERRORS = frozenset({"error_max_turns", "error_max_budget_usd"})
"""Result subtypes that running the segment again cannot fix."""


_FINAL_TERMINAL_REASONS = frozenset({"prompt_too_long", "image_error"})


def _final_api_error(err: ResultError) -> bool:
    """Whether the same request would be refused again however often it is sent.

    Final: the engine's own verdicts (prompt too long, image error), an invalid
    request (400), an unknown model (404) and a request too large (413): the model,
    tools and conversation come from the segment's input, which a retry reuses.
    Retried: a low credit balance, authentication and permissions (fixed outside the
    request), rate limits, overload and server errors.
    """
    if err.terminal_reason in _FINAL_TERMINAL_REASONS:
        return True
    if err.api_error_status in (404, 413):
        return True
    if err.api_error_status == 400:
        text = f"{err.result or ''} {' '.join(err.errors)} {err}".lower()
        return "credit balance" not in text
    return False


_TRANSCRIPT_TYPES = frozenset({"user", "assistant", "attachment", "system"})
_OPTIONAL_STORE_METHODS = (
    "list_sessions",
    "list_session_summaries",
    "delete",
    "list_subkeys",
)


def _is_transcript(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("uuid"), str)
        and entry.get("type") in _TRANSCRIPT_TYPES
        and not entry.get("isSidechain")
    )


_COST_STATE = "cost-state"


def _is_cost_state(entry: Any) -> bool:
    """Whether ``entry`` is a saved cost total (see ``_without_cost_state``)."""
    return isinstance(entry, dict) and entry.get("type") == _COST_STATE


def _without_cost_state(entries: list[Any]) -> list[Any]:
    """The entries without the session's saved cost totals.

    Claude Code 2.1.277 and newer write the session's running cost at the end of a run
    (a ``cost-state`` entry) and start a resumed run from it, so the run's result
    reports the session's total instead of what the run spent, and the Workflow would
    count earlier steps again. Without the entry every engine reports each run's own
    cost (``max_budget_usd`` stays per run either way, tested on 2.1.287).
    """
    return [e for e in entries if not _is_cost_state(e)]


def _json_copy(value: Any) -> Any:
    """A deep copy of JSON data (conversation entries), made through ``json``.

    As deep as Temporal's own payload converter can carry it: ``copy.deepcopy``
    takes two Python frames per level, so it fails on data nested a few hundred
    levels deep that ``json`` still reads.
    """
    return json.loads(json.dumps(value))


def _last_entry(entries: list[Any]) -> str | None:
    """The uuid of the session's last transcript entry: where the engine resumes."""
    for entry in reversed(entries):
        if _is_transcript(entry):
            return entry["uuid"]
    return None


def _resume_point(entries: list[Any], paused_call: str | None) -> str | None:
    """Where the next segment must continue after this one.

    Normally the last transcript entry. But when Claude sent several durable calls
    at once, the engine writes the denied calls' results after the paused call's
    deferral marker, and a session that ends there no longer resumes the paused
    call (tested: its delivered result is replaced with "[Tool result missing due
    to internal error]"). Then the checkpoint is the marker, so the next segment
    continues in a copy that ends at it. So too when a later block of the paused
    call's message comes after the marker (a slow model, see ``_blocks_first``).
    """
    leaf: str | None = None
    marker: str | None = None
    user_after_marker = False
    for entry in entries:
        if not _is_transcript(entry):
            continue
        leaf = entry["uuid"]
        attachment = entry.get("attachment") if entry["type"] == "attachment" else None
        if (
            paused_call is not None
            and isinstance(attachment, dict)
            and attachment.get("type") == "hook_deferred_tool"
            and attachment.get("toolUseID") == paused_call
        ):
            marker, user_after_marker = leaf, False
        elif marker is not None and entry["type"] in ("user", "assistant"):
            user_after_marker = True
    return marker if marker is not None and user_after_marker else leaf


def _result_ids(entry: Any) -> list[str]:
    """The tool calls a transcript entry answers (a user entry with tool results)."""
    if not isinstance(entry, dict) or entry.get("type") != "user":
        return []
    content = (entry.get("message") or {}).get("content")
    if not isinstance(content, list):
        return []
    return [
        str(block.get("tool_use_id"))
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


def _marker_of(entry: Any) -> str | None:
    """The paused call's id, if ``entry`` is a deferral marker."""
    if not isinstance(entry, dict) or entry.get("type") != "attachment":
        return None
    attachment = entry.get("attachment")
    if isinstance(attachment, dict) and attachment.get("type") == "hook_deferred_tool":
        return str(attachment.get("toolUseID"))
    return None


def _siblings(
    entries: list[Any], paused_call: str, denials: dict[str, str]
) -> list[DeferredCall]:
    """The other durable calls of the paused message, denied after the pause.

    Their denials come after the paused call's deferral marker, and the hook recorded
    each one as "not run" (``denials``): a call denied for any other reason, or by
    anything else, is never run. With a slow model, the calls themselves can come
    after the marker too (see ``_blocks_first``).
    """
    marker = _marker_index(entries, paused_call)
    if marker is None:
        return []
    entries = _blocks_first(entries, marker)
    marker = cast("int", _marker_index(entries, paused_call))
    not_run = REASON_KEYS[NOT_RUN_REASON]
    denied = {
        tid
        for entry in entries[marker + 1 :]
        for tid in _result_ids(entry)
        if denials.get(denial_name(tid)) == not_run
    }
    calls: list[DeferredCall] = []
    for entry in entries[:marker]:
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        for block in (entry.get("message") or {}).get("content") or []:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") in denied
                and str(block.get("name", "")).startswith(PREFIX)
                and all(c.id != block["id"] for c in calls)
            ):
                calls.append(
                    DeferredCall(
                        id=block["id"],
                        name=str(block["name"]).removeprefix(PREFIX),
                        input=dict(block.get("input") or {}),
                    )
                )
    return calls


def _deliver(
    entries: list[Any], checkpoint: str, results: dict[str, ToolOutcome]
) -> tuple[list[Any], set[str]] | None:
    """Put the results of the calls denied after a pause where the engine resumes.

    After a pause in a message with several calls, the session ends with the paused
    call's hook entries (the last is the deferral marker, the checkpoint), followed
    by the denials of the calls after it. Resuming at the marker would drop those
    calls: the engine strips tool calls whose results come after where it resumes.
    So the denials move before the paused call's hook entries, and the denials of
    durable calls get their real results. The engine's own links between entries
    stay as they are (calls that came after the marker, with a slow model, move back
    first: see ``_blocks_first``). Tested on Claude Code 2.1.273 and 2.1.287, with
    built-in and durable calls mixed in one message: Claude sees each call of the
    message with its result (a built-in call denied after the pause keeps the
    denial, and Claude calls it again).

    Returns:
        The entries to resume from, and the calls whose results they now hold; or
        None if the checkpoint is not a deferral marker followed by denials (or by
        later blocks of its message).
    """
    marker = _index_of(entries, checkpoint)
    paused = _marker_of(entries[marker]) if marker is not None else None
    if marker is None or paused is None:
        return None
    ordered = _blocks_first(entries, marker)
    reordered = ordered is not entries
    if reordered:  # a slow model's blocks after the pause, now before it
        entries = ordered
        marker = cast("int", _index_of(entries, checkpoint))
    first = _hooks_start(entries, marker, paused)
    moved, delivered = _denied_after(entries, marker, paused, results)
    if not moved and not reordered:
        return None
    return [*entries[:first], *moved, *entries[first : marker + 1]], delivered


def _hooks_start(entries: list[Any], marker: int, paused: str) -> int:
    """Where the paused call's hook entries start: they end at its deferral marker."""
    first = marker
    while first > 0:
        previous = entries[first - 1]
        attachment = previous.get("attachment") if isinstance(previous, dict) else None
        if not (isinstance(attachment, dict) and attachment.get("toolUseID") == paused):
            break
        first -= 1
    return first


def _index_of(entries: list[Any], uuid_: str) -> int | None:
    """The index of the (first) entry with this uuid, or None."""
    return next(
        (
            i
            for i, e in enumerate(entries)
            if isinstance(e, dict) and e.get("uuid") == uuid_
        ),
        None,
    )


def _marker_index(entries: list[Any], paused: str) -> int | None:
    """The index of a paused call's deferral marker (the last one), or None."""
    for index in range(len(entries) - 1, -1, -1):
        if _marker_of(entries[index]) == paused:
            return index
    return None


def _blocks_first(entries: list[Any], marker: int) -> list[Any]:
    """The session with every block of the paused call's message before its hook entries.

    Claude Code writes each block of a message as an entry of its own, and starts a
    call as soon as its block arrives. So when the model takes its time between the
    calls of one message, the paused call's hook entries and deferral marker come
    before the entries of the message's later calls, the first of them linked to the
    marker. The next segment resumes in a copy that ends at the marker: those calls
    and their results would be lost, and a durable one would not run with the paused
    call. So their entries move back to where Claude Code writes them when the model
    is quick: right after the message's other blocks, each linked to the block before
    it, with the paused call's first hook entry linked to the last block. Then the
    session is as Claude Code has it then (tested with a model that waits 1.5 s after
    each call, on Claude Code 2.1.273, 2.1.274, 2.1.288 and 2.1.295).

    Returns:
        ``entries`` itself when no block of the message comes after its hook entries,
        else a new list (with copies of the entries moved or linked anew).
    """
    paused = _marker_of(entries[marker])
    if paused is None:
        return entries
    first = _hooks_start(entries, marker, paused)
    anchor = entries[first - 1] if first > 0 else None
    message = anchor.get("message") if isinstance(anchor, dict) else None
    found = _calls_of(entries[:first], paused)
    if (
        not isinstance(anchor, dict)
        or anchor.get("type") != "assistant"
        or not isinstance(message, dict)
        or message.get("id") is None
        or found is None
        or found[0]["message"].get("id") != message["id"]
    ):
        return entries  # not a shape this knows: as it is
    late = [
        index
        for index in range(marker + 1, len(entries))
        if isinstance(entries[index], dict)
        and entries[index].get("type") == "assistant"
        and isinstance(entries[index].get("message"), dict)
        and entries[index]["message"].get("id") == message["id"]
    ]
    if not late:
        return entries
    blocks: list[Any] = []
    parent = anchor.get("uuid")
    for index in late:
        block = _json_copy(entries[index])
        block["parentUuid"] = parent
        parent = block.get("uuid")
        blocks.append(block)
    hook = _json_copy(entries[first])
    hook["parentUuid"] = parent
    moved = set(late)
    rest = [e for i, e in enumerate(entries) if i > first and i not in moved]
    return [*entries[:first], *blocks, hook, *rest]


def _denied_after(
    entries: list[Any],
    marker: int,
    paused: str,
    results: dict[str, ToolOutcome],
    calls: Collection[str] | None = None,
) -> tuple[list[Any], set[str]]:
    """The results after a deferral marker, durable ones replaced by their real results.

    Those are the denials of the calls after the paused one (see ``_deliver``).

    Args:
        entries: The session.
        marker: The index of the paused call's deferral marker.
        paused: The paused call's id (its own result is never among them).
        results: The results the Workflow delivers.
        calls: When given, only results of these calls: the calls of the paused
            call's message (entries a later run wrote are not denials).

    Returns:
        The result entries (copies), and the calls whose results they now hold.
    """
    moved: list[Any] = []
    delivered: set[str] = set()
    for entry in entries[marker + 1 :]:
        ids = _result_ids(entry)
        if not ids or paused in ids:
            continue
        if calls is not None and not set(ids) <= set(calls):
            continue
        entry = _json_copy(entry)
        for block in entry["message"]["content"]:
            outcome = results.get(str(block.get("tool_use_id")))
            if block.get("type") != "tool_result" or outcome is None:
                continue
            block["content"] = _result_content(outcome)
            if outcome.is_error:
                block["is_error"] = True
            else:
                block.pop("is_error", None)
            entry["toolUseResult"] = block["content"]
            delivered.add(str(block["tool_use_id"]))
        moved.append(entry)
    return moved, delivered


def _waiting_call_of(entry: Any) -> str | None:
    """The call of an assistant entry that holds exactly one call and nothing else: the
    checkpoint of a pause at a call that waited for its tool step (warm tool calls).

    Any other pause ends at its deferral marker or after it (``_resume_point``).
    """
    if not isinstance(entry, dict) or entry.get("type") != "assistant":
        return None
    content = (entry.get("message") or {}).get("content")
    if not isinstance(content, list) or len(content) != 1:
        return None
    block = content[0]
    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
        return str(block["id"])
    return None


def _hook_span(entries: list[Any], checkpoint: str) -> tuple[int, int, str] | None:
    """Where a pause's hook entries are: (the first, the deferral marker, the call's id).

    The checkpoint is the marker, or a later entry when no user entry followed the
    marker (see ``_resume_point``): then only attachments come between them. After a
    call that waited for its tool step, the checkpoint is the call's own entry, and
    no hook entry follows it: (the entry after it, the call's entry, the call's id).

    Returns:
        None if the checkpoint is not in ``entries``, or is not right after a pause.
    """
    end = next(
        (
            i
            for i in range(len(entries) - 1, -1, -1)
            if isinstance(entries[i], dict) and entries[i].get("uuid") == checkpoint
        ),
        None,
    )
    if end is None:
        return None
    waiting = _waiting_call_of(entries[end])
    if waiting is not None:  # the call waited for its step: no hook entries follow
        return end + 1, end, waiting
    marker = end
    while marker >= 0 and _marker_of(entries[marker]) is None:
        entry = entries[marker]
        if isinstance(entry, dict) and entry.get("type") in ("user", "assistant"):
            return None  # the checkpoint is not right after a pause
        marker -= 1
    if marker < 0:
        return None
    paused = cast("str", _marker_of(entries[marker]))
    return _hooks_start(entries, marker, paused), marker, paused


def _calls_of(
    entries: list[Any], call_id: str
) -> tuple[dict[str, Any], list[str]] | None:
    """The assistant entry with a call, and the ids of every call of its message.

    Claude Code writes each block of a message as an entry of its own, with the
    message's id.
    """
    owner = None
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        message = entry.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(b, dict)
            and b.get("type") == "tool_use"
            and b.get("id") == call_id
            for b in content
        ):
            owner = entry
    if owner is None or not isinstance(owner.get("uuid"), str):
        return None
    message_id = owner["message"].get("id")
    ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        if entry is not owner and (
            message_id is None or message.get("id") != message_id
        ):
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                ids.append(str(block.get("id")))
    return owner, ids


def _recorded_call(entries: list[Any], call_id: str) -> dict[str, Any] | None:
    """The ``tool_use`` block of a call, as the conversation recorded it."""
    found = _calls_of(entries, call_id)
    if found is None:
        return None
    for block in found[0]["message"]["content"]:
        if isinstance(block, dict) and block.get("id") == call_id:
            return _json_copy(block)
    return None


def _step_context(entries: list[Any], first: int) -> list[Any]:
    """The private copy of the conversation that a tool step runs its call in.

    As in Brian Strauch's ``context()`` (hybrid prototype): the conversation before
    the paused call's hook entries, without the calls that have no result there (the
    paused call, and the calls of its message after it), and without entries left
    empty. Calls that ran before keep Claude Code's own results (an Edit needs the
    Read before it). An entry whose parent was left out is linked to that entry's
    own parent; the other links stay (a compaction's start keeps having none). The
    stand-in model then answers with the paused call, so Claude Code runs it as the
    next call of the turn.
    """
    part = entries[:first]
    answered = {tid for entry in part for tid in _result_ids(entry)}
    context: list[Any] = []
    left_out: dict[str, Any] = {}  # uuid -> its parent
    for original in _without_cost_state(part):
        entry = _json_copy(original)
        message = entry.get("message")
        if entry.get("type") == "assistant" and isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                kept = [
                    b
                    for b in content
                    if not (
                        isinstance(b, dict)
                        and b.get("type") == "tool_use"
                        and b.get("id") not in answered
                    )
                ]
                if not kept:
                    if isinstance(entry.get("uuid"), str):
                        left_out[entry["uuid"]] = entry.get("parentUuid")
                    continue
                message["content"] = kept
        parent = entry.get("parentUuid")
        seen: set[Any] = set()
        while parent in left_out and parent not in seen:
            seen.add(parent)
            parent = left_out[parent]
        if parent != entry.get("parentUuid"):
            entry["parentUuid"] = parent
        context.append(entry)
    return context


def _result_record(entry: dict[str, Any]) -> dict[str, Any]:
    """A tool step's record of its result as the step returns it (``ToolOutcome.entry``).

    Without the result itself (the outcome carries it), and without Claude Code's
    metadata of the call (``toolUseResult``): an Edit's or a Write's holds the whole
    file as it was before (``originalFile``), which would put the file in the
    Workflow's history. Claude Code does not need it to go on (tested: two Edits of
    one file, and a Write over a file Claude read).
    """
    message = entry.get("message")
    record = {k: v for k, v in entry.items() if k != "toolUseResult"}
    record["message"] = {
        **(message if isinstance(message, dict) else {}),
        "content": [],
    }
    return _json_copy(record)


def _result_block(call_id: str, outcome: ToolOutcome) -> dict[str, Any]:
    """The ``tool_result`` block that hands a call's result to Claude."""
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": _result_content(outcome),
        "is_error": outcome.is_error,
    }


_RECORD_FIELDS = (
    "isSidechain",
    "userType",
    "entrypoint",
    "cwd",
    "sessionId",
    "version",
    "gitBranch",
)
"""Fields a record without the step's own entry takes from the call's assistant entry:
fields Claude Code writes in the entries of a session (checked on 2.1.273, 2.1.274
and 2.1.288)."""

_RECORD_NAMESPACE = uuid.UUID("5c4f2d0e-8f8e-4c2e-9d55-7a1e0d6b9f21")
"""Names the uuid of such a record after its call, so every attempt makes the same."""


def _record(
    owner: dict[str, Any],
    call_id: str,
    outcome: ToolOutcome,
    before: list[Any],
) -> dict[str, Any]:
    """Claude Code's record of a call's result, with the outcome's result in it.

    The tool step's own record when it returned one (``ToolOutcome.entry``). When it
    did not (the step failed, the call was rejected, or the task stopped first), one
    with the fields Claude Code writes, from the call's own assistant entry (tested:
    Claude Code goes on from such a record as from its own). Either way it is linked
    to the call's assistant entry, as Claude Code links each call's result.
    """
    taken = {e.get("uuid") for e in before if isinstance(e, dict)}
    given = outcome.entry
    if (
        isinstance(given, dict)
        and given.get("type") == "user"
        and isinstance(given.get("uuid"), str)
        and given["uuid"] not in taken
    ):
        record = _json_copy(given)
    else:
        record = {k: owner[k] for k in _RECORD_FIELDS if k in owner}
        record["type"] = "user"
        record["uuid"] = str(uuid.uuid5(_RECORD_NAMESPACE, call_id))
        record["timestamp"] = (
            datetime.now(timezone.utc).isoformat(timespec="milliseconds")[:-6] + "Z"
        )
    message = record.get("message")
    record["message"] = {
        **(message if isinstance(message, dict) else {}),
        "role": "user",
        "content": [_result_block(call_id, outcome)],
    }
    record["parentUuid"] = owner["uuid"]
    record["sourceToolAssistantUUID"] = owner["uuid"]
    return record


def _place(
    entries: list[Any], checkpoint: str, results: dict[str, ToolOutcome]
) -> tuple[list[Any], set[str]] | None:
    """The conversation with Claude Code's record of a paused call's result in it.

    For a call of ``RECORDED_TOOLS``, and for any call that waited for its tool step
    (it has no deferral marker to resume at): the call's hook entries (and everything
    after them) give way to the record (see ``_record``). Then come the denials of the
    calls after the paused one, durable ones with their real results (as in
    ``_deliver``). Claude Code then holds the call as answered, and does not check it
    again; with no new message, it continues the turn by itself (``CONTINUE_ENV``).
    Tested on Claude Code 2.1.273, 2.1.274 and 2.1.288: Edit after Read, Write over a
    file Claude read, an Edit with a durable call in one message.

    Returns:
        The entries to resume from, and the calls whose results they now hold; or
        None if the checkpoint is not right after a pause at such a call, or a result
        would not go into the conversation.
    """
    span = _hook_span(entries, checkpoint)
    if span is None:
        return None
    ordered = _blocks_first(entries, span[1])  # a slow model's calls after the pause
    if ordered is not entries:
        entries = ordered
        span = cast("tuple[int, int, str]", _hook_span(entries, checkpoint))
    first, marker, paused = span
    outcome = results.get(paused)
    found = _calls_of(entries[:first], paused)
    if outcome is None or found is None:
        return None
    owner, calls = found
    tool = next(
        (
            b.get("name")
            for b in owner["message"]["content"]
            if isinstance(b, dict) and b.get("id") == paused
        ),
        None,
    )
    if tool not in RECORDED_TOOLS and _waiting_call_of(entries[marker]) is None:
        return None
    record = _record(owner, paused, outcome, entries[:first])
    moved, delivered = _denied_after(entries, marker, paused, results, calls)
    delivered.add(paused)
    if set(results) - delivered:
        return None  # a result would go as a message too: as before, all of them
    return [*entries[:first], record, *moved], delivered


def _seed(committed: list[Any], checkpoint: str) -> list[Any] | None:
    """The committed conversation up to the checkpoint, where the next run resumes.

    That is all of it, unless the checkpoint is the paused call's deferral marker
    (see ``_resume_point``): then the entries after the marker are left out.

    Returns:
        The entries, or None if the checkpoint is not in the conversation.
    """
    if _last_entry(committed) == checkpoint:
        return committed
    for index in range(len(committed) - 1, -1, -1):
        entry = committed[index]
        if _is_transcript(entry) and entry["uuid"] == checkpoint:
            return committed[: index + 1]
    return None


def _kept(committed: list[Any], entries: list[Any]) -> tuple[int, int]:
    """How much of the committed conversation the session still starts with.

    Saved cost totals do not count: the engine never resumes from one, and those an
    older version left in the conversation stay where they are (every seed and every
    store load leaves them out), so the step sends nothing again because of them.

    Returns:
        The number of committed entries to keep, and the number of the session's
        entries they cover.
    """
    i = j = 0
    while True:
        while i < len(committed) and _is_cost_state(committed[i]):
            i += 1
        while j < len(entries) and _is_cost_state(entries[j]):
            j += 1
        if i == len(committed) or j == len(entries):
            return i, j
        if committed[i] is not entries[j] and committed[i] != entries[j]:
            return i, j
        i += 1
        j += 1


def _attempt_session_id(session_id: str, attempt: int) -> str:
    """The id of a new session: a fresh one for each attempt.

    A failed attempt may have left a partial transcript, in the session store and
    in the engine's own folder.
    """
    if attempt == 1:
        return session_id
    return str(uuid.uuid5(uuid.UUID(session_id), f"attempt-{attempt}"))


class _SessionMoved(Exception):
    """The session no longer ends at the checkpoint (for example after a Workflow reset)."""


def _moved(err: BaseException) -> bool:
    seen: BaseException | None = err
    while seen is not None:
        if isinstance(seen, _SessionMoved):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


class _GuardedStore:
    """Passes the session store to the SDK, without the saved cost totals.

    The engine loads every session it resumes through it, so it never starts from the
    session's running cost (see ``_without_cost_state``); the store itself keeps what
    the engine wrote. With a checkpoint, it also checks the session when it is
    resumed in place: that is right only while the session still ends at the
    checkpoint. The check rides on the load the SDK does anyway, so it costs nothing
    extra.
    """

    def __init__(self, inner: Any, session_id: str, checkpoint: str | None) -> None:
        self._inner = inner
        self._session_id = session_id
        self._checkpoint = checkpoint
        self.written: set[str] = set()
        """The transcript entries the engine wrote to its session (their uuids)."""
        self.last: dict[str, Any] | None = None
        """The last transcript entry the engine wrote to its session."""
        self.calls: dict[str, list[str]] = {}
        """The tool calls of each assistant message the engine wrote (by message id)."""
        self.ran_here: set[str] = set()
        """Calls that waited for their tool step and ran in this engine."""
        self.results: dict[str, dict[str, Any]] = {}
        """The engine's own result entries of calls in ``ran_here`` (a tool step takes
        its call's, and drops the rest)."""

    async def append(self, key: Any, entries: Any) -> None:
        mine = key.get("session_id") == self._session_id and not key.get("subpath")
        if mine and self.ran_here:
            # A call that ran here in its tool step: its result entry goes in as a
            # tool step's record does, without Claude Code's metadata of the call (an
            # Edit's holds the whole file as it was).
            entries = [
                {k: v for k, v in e.items() if k != "toolUseResult"}
                if isinstance(e, dict) and self.ran_here.intersection(_result_ids(e))
                else e
                for e in entries
            ]
        await self._inner.append(key, entries)
        if not mine:
            return
        for entry in entries:
            if not _is_transcript(entry):
                continue
            self.written.add(entry["uuid"])
            self.last = entry
            message = entry.get("message")
            if entry.get("type") == "assistant" and isinstance(message, dict):
                calls = self.calls.setdefault(str(message.get("id")), [])
                for block in message.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        calls.append(str(block.get("id")))
            for call in self.ran_here.intersection(_result_ids(entry)):
                self.results[call] = entry

    async def load(self, key: Any) -> Any:
        entries = await self._inner.load(key)
        if (
            entries
            and self._checkpoint is not None
            and key.get("session_id") == self._session_id
            and not key.get("subpath")
            and _last_entry(entries) != self._checkpoint
        ):
            raise _SessionMoved(
                f"Session {self._session_id} continued after checkpoint "
                f"{self._checkpoint}"
            )
        return _without_cost_state(entries) if entries else entries


def _guarded(inner: Any, session_id: str, checkpoint: str | None) -> Any:
    """``inner`` for the SDK, keeping only the optional methods it has."""
    namespace: dict[str, Any] = {}
    for name in _OPTIONAL_STORE_METHODS:
        # The rule the SDK applies: present, and not the protocol's default.
        present = getattr(inner, name, None) is not None
        if present and getattr(type(inner), name, None) is not getattr(
            SessionStore, name, None
        ):
            namespace[name] = _forward(name)
    cls = type("GuardedSessionStore", (_GuardedStore,), namespace)
    return cls(inner, session_id, checkpoint)


def _forward(name: str) -> Any:
    async def method(self: _GuardedStore, *args: Any, **kwargs: Any) -> Any:
        return await getattr(self._inner, name)(*args, **kwargs)  # type: ignore[reportPrivateUsage]

    method.__name__ = name
    return method


def _hook_entry() -> dict[str, Any]:
    """The PreToolUse command hook (a function so tests can simulate other engines).

    Exec form: Claude Code starts the program directly, with no shell, so no path
    needs quoting. (Through a shell, Git Bash on Windows drops the backslashes of a
    Windows path, and PowerShell, its fallback, needs other quoting.) The hook file
    is run as a plain script: it only needs the standard library, and importing
    this package would add about a second to every durable tool call. Isolated mode
    and no ``site``, like the launcher: nothing from the Worker's environment or its
    ``.pth`` files runs before each decision.
    """
    hook = Path(__file__).with_name("_defer_hook.py")
    return {
        "type": "command",
        "command": sys.executable,
        "args": ["-I", "-S", str(hook)],
    }


_REMOVE_TRIES = 6
"""How often a hook folder that cannot be renamed yet is tried again."""


def _remove_hook_folder(hook_dir: str, tries: int = _REMOVE_TRIES) -> None:
    """Remove a run's hook folder: from then on its hook denies every call.

    Renamed first (atomic), so a hook that runs while the files go never finds the
    folder without its ``stop``, ``worker.lock`` or ``allowed`` file. When it cannot
    be renamed (Windows: an engine still ending has a file in it open), it stays as
    it is, with ``stop`` added, and is tried again every ``EXIT_GRACE_SECONDS`` while
    the event loop runs, and when the Worker exits.
    """
    gone = f"{hook_dir}.gone"
    try:
        os.rename(hook_dir, gone)
    except FileNotFoundError:
        return
    except OSError:
        with contextlib.suppress(OSError):
            Path(hook_dir, "stop").touch()
        if tries == _REMOVE_TRIES:
            atexit.register(shutil.rmtree, hook_dir, True)
        if tries > 1:
            with contextlib.suppress(RuntimeError):  # no event loop runs
                asyncio.get_running_loop().call_later(
                    EXIT_GRACE_SECONDS, _remove_hook_folder, hook_dir, tries - 1
                )
        return
    shutil.rmtree(gone, ignore_errors=True)


WAIT_MARGIN_SECONDS = 30.0
"""How much longer than ``warm_seconds`` the hook of a waiting call may wait. The runner
ends the engine at ``warm_seconds``; the hook's own limit only backs that up, and
Claude Code's time limit for the hook is longer again, so the hook always answers."""

ENTRY_SECONDS = 5.0
"""How long a waiting call's entry may take to reach the session store. Claude Code
writes its transcript every 100 ms in the middle of a turn (``FLUSH_INTERVAL_MS`` in
2.1.274 and 2.1.295); if it takes longer, the call is deferred as without waiting."""

ENTRY_POLL_SECONDS = 0.001
"""How often the runner looks for a waiting call's entry."""

WAITING_POLL_SECONDS = 0.005
"""How often a segment looks for a call that waits in its engine's hook (``waiting``).
The call's entry reaches the session store tens of milliseconds later anyway."""

WAIT_MODES = frozenset({"default", "plan"})
"""Permission modes in which a call may wait for its tool step in its segment's engine.
In these, Claude Code's own permission check refuses the call if the hook ever gives
no answer (a hook that waits long could be killed); in others (``acceptEdits``,
``bypassPermissions``, auto mode) it could let the call run in the segment."""

COST_SECONDS = 5.0
"""How long the runner waits for an engine's running cost (``_engine_cost``)."""


def _hook_folder(wait_seconds: float | None = None) -> str:
    """A new folder with the settings file that registers the hook for every tool.

    With ``wait_seconds`` (warm tool calls): a call may wait in the hook that long for
    its tool step, and the turn after it as long (the ``PostToolBatch`` hook). Claude
    Code's own time limit for these hooks is ``WAIT_MARGIN_SECONDS`` longer, so the
    hook always answers first (a hook that times out gives no answer).
    """
    hook_dir = tempfile.mkdtemp(prefix="tca-hook-")
    pre = _hook_entry()
    hooks: dict[str, Any] = {
        "PreToolUse": [
            # Every tool: built-in calls after a pause are denied too.
            {"matcher": ".*", "hooks": [pre]}
        ]
    }
    if wait_seconds:
        limit = math.ceil(wait_seconds + WAIT_MARGIN_SECONDS)
        pre["timeout"] = limit
        after = _hook_entry()
        after["args"] = [*after["args"], "after-calls"]
        after["timeout"] = limit
        hooks["PostToolBatch"] = [{"matcher": "", "hooks": [after]}]
    settings = {"hooks": hooks}
    Path(hook_dir, "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    return hook_dir


async def _stop_hooks_when_cancelled(hook_dir: str) -> None:
    """Once the segment Activity is cancelled (or timed out), the hook denies every call.

    The SDK gives the engine a few seconds to exit, and a cancel reaches the Worker
    only with a heartbeat, so the engine could otherwise still run a tool after the
    Workflow moved on.
    """
    await activity.wait_for_cancelled()
    try:
        Path(hook_dir, "stop").touch()
    except OSError:
        pass  # the folder is already gone, and the hook denies without it


def _hook_denials(hook_dir: str) -> dict[str, str]:
    """The calls the hook denied in this run: record name (``denial_name``) to reason key."""
    try:
        return {
            path.name: path.read_text(encoding="utf-8").strip()
            for path in Path(hook_dir, "denied").iterdir()
        }
    except OSError:
        return {}


class _Feed:
    """An engine's input: the messages for its turns, then, once ``close`` is called,
    its end.

    The SDK writes each message to the engine as it comes, and closes the engine's
    input when the stream ends; the engine then exits after its turn. A warm engine's
    input stays open between segments: the next segment's message goes in with
    ``put``.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.put(messages)

    def put(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self._queue.put_nowait(message)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        while (message := await self._queue.get()) is not None:
            yield message


class _StartWatch:
    """Ends an engine's input when it does not start its turn in time.

    For an engine that must start its turn by itself (see ``CONTINUE_ENV``): one that
    does not would wait for a message forever. With its input ended, it exits. Each
    notice the engine gives before its turn (a SessionStart hook's events, for
    example) starts the time again: the engine is still getting ready.
    """

    def __init__(self, feed: _Feed, seconds: float) -> None:
        self.fired = False
        """Whether the time ran out, so the input was ended."""
        self._feed = feed
        self._seconds = seconds
        self._timer = asyncio.get_running_loop().call_later(seconds, self._fire)

    def _fire(self) -> None:
        self.fired = True
        self._feed.close()

    def waiting(self) -> None:
        """The engine gave a notice before its turn: the time starts again."""
        if not self.fired and not self._timer.cancelled():
            self._timer.cancel()
            self._timer = asyncio.get_running_loop().call_later(
                self._seconds, self._fire
            )

    def heard(self) -> None:
        """The engine started its turn (or ended): no more time is needed."""
        self._timer.cancel()


def _turn_began(message: Any) -> bool:
    """Whether an engine's message shows that it started a turn (for ``_StartWatch``).

    Claude Code says ``init`` when it starts one. Other notices can come before it,
    while the engine waits for a message: a SessionStart hook's events, for example.
    """
    if isinstance(message, SystemMessage):
        return message.subtype == "init"
    return isinstance(
        message, (AssistantMessage, UserMessage, ResultMessage, StreamEvent)
    )


class _NotContinued(RuntimeError):
    """Claude Code did not continue the turn by itself (see ``CONTINUE_ENV``)."""


def _not_continued(version: str) -> _NotContinued:
    return _NotContinued(
        f"Claude Code {version} did not continue the turn after the result of an Edit "
        f"or a Write: it said nothing for {CONTINUE_START_SECONDS:.0f} seconds before "
        f"the turn. This plugin asks it to with {RESUME} (tested on every Claude "
        "Code release from 2.1.273 to 2.1.295)."
    )


def _not_continued_output(
    session_id: str, err: _NotContinued, attempt: int
) -> SegmentOutput:
    """Run the segment again (raise) until ``CONTINUE_ATTEMPTS``, then stop the task.

    Raises:
        RuntimeError: Before the last attempt (Temporal runs the segment again).
    """
    if attempt < CONTINUE_ATTEMPTS:
        raise RuntimeError(f"{err} Retrying.") from err
    return SegmentOutput(
        session_id=session_id,
        is_error=True,
        error=(
            f"{err} It did not in {attempt} attempts, so the task stops here. Use a "
            "Claude Code version this plugin was tested with, or leave Edit and Write "
            "out of tool_activities."
        ),
    )


async def _as_messages(prompt: Any) -> list[dict[str, Any]]:
    """An engine run's input as user messages.

    A text prompt becomes the message the one-shot query writes for it; the runner's
    own message stream is read as it is. None is no message: Claude Code continues
    the turn by itself (see ``CONTINUE_ENV``).
    """
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return [
            {
                "type": "user",
                "session_id": "",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
            }
        ]
    return [message async for message in prompt]


def _agents_running(client: Any) -> bool:
    """Whether delegated agents the engine started still run (the SDK tracks them)."""
    return bool(getattr(getattr(client, "_query", None), "_inflight_tasks", None))


async def _turn_messages(
    client: Any,
    feed: _Feed,
    resumed: bool,
    *,
    stay: bool = False,
    answered: Collection[str] = (),
    source: AsyncIterator[Any] | None = None,
) -> AsyncGenerator[Any, None]:
    """An engine's messages until it exits. Its input ends once its own turn is over.

    ``source``: where the messages come from, when not straight from the client (an
    engine with an ``_Inbox``; a call that waits for its tool step comes through it
    as a ``_WaitingCall``).

    The one-shot query ends the input at the first result that comes while no
    delegated agent runs. A resumed engine first repeats the result its session
    paused with, before its ``init`` message (tested on Claude Code 2.1.273 and
    2.1.288), so the input ended before the resumed turn even started: hooks and
    permission callbacks from ``extra_options``, which the engine asks over its
    input, were never asked, and Claude Code refused their calls. This ends the
    input at the turn's own result instead, once no agent runs: any result but
    one such repeat (a pause, before ``init``). Like the one-shot query, it reads
    on until the engine exits: the last result counts, and an engine that exits
    after an error result raises ``ResultError``.

    With ``stay``, an engine that paused cleanly stays running instead (it can stay
    warm, see ``warm_engines``): when its turn's result came with no error and no
    task started, at a call other than one of ``answered`` (whose results this run
    brings: a resumed engine repeats such a pause), this returns at that result, and
    the input stays open.
    """
    started = not resumed
    repeated = False
    tasks = False
    ending = False
    async for message in source if source is not None else client.receive_messages():
        yield message
        if ending:
            continue
        if isinstance(message, SystemMessage):
            started = started or message.subtype == "init"
            tasks = tasks or message.subtype == "task_started"
        elif isinstance(message, ResultMessage):
            paused = message.deferred_tool_use
            if (
                not started
                and not repeated
                and paused is not None
                and not message.is_error
            ):
                repeated = True  # the pause the session ended with, repeated
                continue
            if (
                stay
                and not tasks
                and not message.is_error
                and paused is not None
                and paused.id not in answered
            ):
                return  # the engine waits, its input open, for the next segment
            if not _agents_running(client):  # each one wakes the engine again
                feed.close()
                ending = True


async def _disconnect(client: Any) -> None:
    """Stop an engine started through the streaming client, and clean up after it.

    The SDK closes the engine's input, waits for it to exit (and ends it if it does
    not), and removes its resume folder. If that wait is cut short (a second cancel,
    an event loop that closes), the engine still gets SIGTERM and the folder goes.
    """
    process = getattr(getattr(client, "_transport", None), "_process", None)
    resumed = getattr(client, "_materialized", None)
    try:
        with contextlib.suppress(Exception):
            await client.disconnect()
    finally:
        _stop_now(process, resumed)


EXIT_GRACE_SECONDS = 10.0
"""How long an engine that got SIGTERM may still write to its resume folder."""


def _stop_now(process: Any, resumed: Any) -> None:
    """End an engine without waiting: SIGTERM if it still runs; its resume folder goes.

    An engine that is still exiting can write a few lines to the folder after it is
    removed (session metadata, no conversation): it is removed again after
    ``EXIT_GRACE_SECONDS`` while the event loop runs, and when the Worker exits.
    """
    running = process is not None and process.returncode is None
    if running:
        with contextlib.suppress(OSError):
            process.terminate()
    if resumed is None:
        return
    folder = str(resumed.config_dir)
    shutil.rmtree(folder, ignore_errors=True)
    if running:
        with contextlib.suppress(RuntimeError):  # no event loop runs
            asyncio.get_running_loop().call_later(
                EXIT_GRACE_SECONDS, shutil.rmtree, folder, True
            )
        atexit.register(shutil.rmtree, folder, True)


async def _engine_messages(
    options: dict[str, Any], feed: _Feed, resumed: bool
) -> AsyncGenerator[Any, None]:
    """Start an engine, send it the messages of ``feed``, and yield what it says until
    it exits.

    See ``_turn_messages`` for when its input ends.
    """
    client = ClaudeSDKClient(ClaudeAgentOptions(**options))
    try:
        await client.connect(feed.stream())
        async for message in _turn_messages(client, feed, resumed):
            yield message
    finally:
        await _disconnect(client)


async def _connected(
    client: Any, feed: _Feed, resumed: bool, answered: Collection[str]
) -> AsyncGenerator[Any, None]:
    """A new engine's messages, for an engine that may stay warm (``stay``).

    The runner, not this, ends the engine: it may keep it for the next segment.
    """
    await client.connect(feed.stream())
    async for message in _turn_messages(
        client, feed, resumed, stay=True, answered=answered
    ):
        yield message


@dataclasses.dataclass
class _WaitingCall:
    """A call that waits in its engine's hook for its tool step (warm tool calls)."""

    id: str
    digest: str
    """``input_digest`` of the call's input, as the hook saw it."""
    input: dict[str, Any]
    """The call's input as the hook saw it: Claude Code has made a relative
    ``file_path`` absolute by then, as in the input a paused call reports."""


_END = object()
"""The end of an engine's messages, in its ``_Inbox``."""


class _ReaderFailed:
    """The engine's message stream failed (in its ``_Inbox``)."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


class _Inbox:
    """An engine's messages, read by a task of their own, kept until a step takes them.

    An engine whose calls may wait for their tool steps serves a segment, then a tool
    step, then the next segment, each its own Activity: none of them reads the
    engine's stream to its end, so a reader does, and each takes what it needs.
    """

    def __init__(self) -> None:
        self.items: collections.deque[Any] = collections.deque()
        self.event = asyncio.Event()
        self.reader: asyncio.Task[None] | None = None

    def put(self, item: Any) -> None:
        self.items.append(item)
        self.event.set()

    async def wait(self, timeout: float | None) -> None:
        """Wait until something comes, or ``timeout`` passes."""
        self.event.clear()
        if self.items:
            return
        wake = (
            None
            if timeout is None
            else asyncio.get_running_loop().call_later(timeout, self.event.set)
        )
        try:
            await self.event.wait()
        finally:
            if wake is not None:
                wake.cancel()

    def stop(self) -> None:
        if self.reader is not None:
            self.reader.cancel()


async def _read_engine(client: Any, feed: _Feed, inbox: _Inbox) -> None:
    """Connect the engine and keep its messages in ``inbox`` until it exits."""
    try:
        await client.connect(feed.stream())
        async for message in client.receive_messages():
            inbox.put(message)
    except BaseException as err:  # handed to whoever reads the inbox
        inbox.put(_ReaderFailed(err))
        if isinstance(err, asyncio.CancelledError):
            raise
    finally:
        inbox.put(_END)


def _read_waiting(hook_dir: str, seen: str | None = None) -> _WaitingCall | None:
    """The call that waits in the engine's hook (``waiting``), if one does.

    None while its record is being written, and for ``seen`` (already reported).
    """
    try:
        with open(os.path.join(hook_dir, WAITING), encoding="utf-8") as handle:
            call_id = handle.readline().rstrip("\n")
            if not call_id or call_id == seen:
                return None
            digest = handle.readline().rstrip("\n")
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return None
    try:
        tool_input = json.loads(text)
    except ValueError:
        return None  # still being written
    if not isinstance(tool_input, dict) or input_digest(tool_input) != digest:
        return None  # still being written
    return _WaitingCall(call_id, digest, tool_input)


async def _inbox_messages(
    inbox: _Inbox, hook_dir: str | None
) -> AsyncGenerator[Any, None]:
    """An engine's messages from its inbox.

    With ``hook_dir``, also a ``_WaitingCall`` when a call starts to wait in the hook
    (once per call).
    """
    seen: str | None = None
    while True:
        if inbox.items:
            item = inbox.items.popleft()
            if item is _END:
                inbox.items.appendleft(item)  # every later reader sees the end too
                return
            if isinstance(item, _ReaderFailed):
                raise item.error
            yield item
            continue
        if hook_dir is not None:
            waiting = _read_waiting(hook_dir, seen)
            if waiting is not None:
                seen = waiting.id
                yield waiting
                continue
        await inbox.wait(WAITING_POLL_SECONDS if hook_dir is not None else None)


def _decide(hook_dir: str, call_id: str, decision: str) -> None:
    """Give the waiting call's hook the runner's decision (written whole, then renamed)."""
    path = os.path.join(hook_dir, DECISION + denial_name(call_id))
    partial = f"{path}.{uuid.uuid4().hex}.partial"
    with open(partial, "w", encoding="utf-8") as handle:
        handle.write(decision)
    os.replace(partial, path)


async def _mirror_flush(client: Any) -> None:
    """Send what the engine wrote so far to the session store (the SDK batches it)."""
    batcher = getattr(
        getattr(client, "_query", None), "_transcript_mirror_batcher", None
    )
    if batcher is not None:
        await batcher.flush()


async def _engine_cost(client: Any) -> float | None:
    """The engine's running cost in USD: what its next result would report.

    Claude Code reports cost only when a turn ends; a call that waits for its tool step
    is in the middle of one. Its ``get_usage`` control request answers mid-turn too,
    with the same total (tested on Claude Code 2.1.273, 2.1.274, 2.1.288 and 2.1.295);
    ``skip_behaviors`` leaves out its scan of recent sessions. Ask it only once the
    call's entry is in the session store: Claude Code records a model request's cost
    at the end of its stream, and may start a call before that. None if it cannot be
    read.
    """
    send = getattr(getattr(client, "_query", None), "_send_control_request", None)
    if send is None:
        return None
    try:
        reply = await send(
            {"subtype": "get_usage", "skip_behaviors": True}, timeout=COST_SECONDS
        )
    except Exception:
        return None
    session = reply.get("session") if isinstance(reply, dict) else None
    total = session.get("total_cost_usd") if isinstance(session, dict) else None
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return None
    return float(total) if math.isfinite(total) and total >= 0 else None


@dataclasses.dataclass
class _WarmEngine:
    """An engine that paused and stays running for the session's next segment.

    The durable boundary is still the pause: the Workflow committed the checkpoint
    before the engine waits here, so a lost warm engine only means a cold start that
    resumes from the checkpoint.
    """

    client: Any
    """The SDK's streaming client, connected to the engine."""
    feed: _Feed
    """The engine's input: the next segment's message goes here."""
    session_id: str
    checkpoint: str
    """Where the engine paused: the next segment must continue exactly there."""
    paused: str
    """The call it paused at: the next segment must bring exactly its result."""
    shape: str
    """The agent's tools and settings: the next segment must have the same."""
    hook_dir: str
    lock: int | None
    store: Any
    """The session store the engine mirrors into."""
    cost: float
    """The engine's running cost total when its last segment ended, to report each
    segment's own cost."""
    version: str
    ran_inside: list[str]
    violations: list[str]
    buffer: int
    """The engine's ``max_buffer_size``: a bigger result needs a new engine."""
    timer: asyncio.TimerHandle | None = None
    inbox: _Inbox | None = None
    """The engine's messages, when a reader of their own reads them (an engine whose
    calls may wait for their tool steps)."""
    state: str = "paused"
    """``paused``: its turn ended at the durable call ``paused`` (the next segment
    brings its result). ``call_waits``: the call ``paused`` waits in its hook for its
    tool step (which can run it here). ``turn_waits``: that call ran here in its tool
    step, and the turn waits right after it for the next segment."""
    call_name: str = ""
    """The waiting call's tool (``call_waits``)."""
    digest: str = ""
    """``input_digest`` of the waiting call's input, as its hook saw it."""
    outcome: ToolOutcome | None = None
    """What the call returned in its tool step (``turn_waits``): the next segment must
    bring exactly this."""
    view: _GuardedStore | None = None
    """The engine's session store, as the engine sees it: what it wrote there."""
    owner: dict[str, Any] | None = None
    """The waiting call's own entry (``call_waits``): a record of its result is made
    from it."""


def _same_outcome(a: ToolOutcome, b: ToolOutcome) -> bool:
    """Whether two outcomes of a call carry the same result (what Claude sees)."""
    return bool(a.is_error) == bool(b.is_error) and _result_content(
        a
    ) == _result_content(b)


def _forget_local_copy(paths: tuple[Path, ...]) -> None:
    """Remove the engine's own copy of a new session from its config folder.

    Claude Code writes a new session's transcript (and tool outputs too large to show
    Claude in full) under ``<config folder>/projects``; resumed sessions run in a
    temporary folder the SDK removes. With the conversation in the Workflow, that
    copy is never read again, so it does not stay on the Worker's disk. Only a copy
    the engine created is removed (see ``_run_held``), once the engine ended (such an
    engine never stays warm: see ``_parks``).
    """
    if not paths:
        return
    transcript, folder = paths
    try:
        transcript.unlink(missing_ok=True)
    except OSError:
        pass  # best effort, for example a file still open on Windows
    shutil.rmtree(folder, ignore_errors=True)


def _next_hook_turn(hook_dir: str, injected: dict[str, Any]) -> None:
    """Prepare a warm engine's hook folder for its next turn (a new segment).

    The hook allows one paused call per segment and records its denials per segment;
    the calls answered since the engine started are in ``answered`` (its environment
    is fixed when it starts). After a warm tool call (``turn_waits``), the runner lets
    the turn go on right after this.
    """
    with contextlib.suppress(OSError):
        os.remove(os.path.join(hook_dir, "paused_call"))
    # A call that ran in its tool step (warm tool calls): its records go too.
    for name in (WAITING, TURN_GATE, TURN_WAITS, TURN_GO, ALLOWED):
        with contextlib.suppress(OSError):
            os.remove(os.path.join(hook_dir, name))
    for decision in Path(hook_dir).glob(DECISION + "*"):
        with contextlib.suppress(OSError):
            decision.unlink()
    shutil.rmtree(os.path.join(hook_dir, "denied"), ignore_errors=True)
    path = os.path.join(hook_dir, ANSWERED)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("".join(f"{call}\n" for call in injected))


_ending: set[asyncio.Task[None]] = set()
"""Warm engines being ended in the background (kept until done)."""


def _engine_alive(client: Any) -> bool:
    """Whether the engine behind a streaming client still runs."""
    process = getattr(getattr(client, "_transport", None), "_process", None)
    return process is not None and process.returncode is None


# ---- the engine ends with its Worker ----

if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    def _lock_file(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock_file(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JOB: list[Any] = []
    """The job's handle, never closed: Windows closes it when this process ends."""
    _KERNEL32: list[Any] = []

    def _kernel32() -> Any:
        if _KERNEL32:
            return _KERNEL32[0]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        _KERNEL32.append(kernel32)
        return kernel32

    def _in_worker_job(pid: int) -> bool:
        """Whether process ``pid`` is in the Worker's job."""
        if not _JOB:
            return False
        kernel32 = _kernel32()
        process = kernel32.OpenProcess(0x1000, False, pid)  # query limited information
        if not process:
            return False
        try:
            inside = wintypes.BOOL()
            return bool(
                kernel32.IsProcessInJob(process, _JOB[0], ctypes.byref(inside))
                and inside.value
            )
        finally:
            kernel32.CloseHandle(process)

    def _worker_job() -> str | None:
        """Create this Worker process's job object, once.

        Its handle stays open for the life of this process: when the process ends,
        however it ends, Windows closes it and ends every process in the job
        (``KILL_ON_JOB_CLOSE``), with the processes they started (those start in the
        job).

        Returns:
            What went wrong, or None.
        """
        if _JOB:
            return None
        kernel32 = _kernel32()
        job = kernel32.CreateJobObjectW(None, None)
        info = _ExtendedLimits()
        # Kill on close; breakaway allowed, for programs that ask for it.
        info.BasicLimitInformation.LimitFlags = 0x2000 | 0x800
        if not job or not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            return (
                f"cannot create a job object (Windows error {ctypes.get_last_error()})"
            )
        _JOB.append(job)
        return None

    def _end_with_worker(pid: int) -> str | None:
        """Put process ``pid`` in the job that ends it when this Worker process ends.

        The Worker process and its other child processes are not in it.

        Returns:
            What went wrong, or None (the runner decides, see ``engine_cleanup``).
        """
        problem = _worker_job()
        if problem is not None:
            return problem
        kernel32 = _kernel32()
        if _in_worker_job(pid):
            return None
        # Query, quota, terminate.
        process = kernel32.OpenProcess(0x1000 | 0x0100 | 0x0001, False, pid)
        if not process:
            return None  # it has ended
        try:
            if kernel32.AssignProcessToJobObject(_JOB[0], process):
                return None
            error = ctypes.get_last_error()
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(process, ctypes.byref(code)) and (
                code.value != 259  # STILL_ACTIVE
            ):
                return None  # it ended (an engine that failed at its start)
            return f"cannot put Claude Code in the job object (Windows error {error})"
        finally:
            kernel32.CloseHandle(process)

else:
    import fcntl

    def _lock_file(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_file(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

    def _worker_job() -> str | None:
        """Nothing to do here: on Linux and macOS the launcher does it (``_launcher``)."""
        return None

    def _end_with_worker(pid: int) -> str | None:
        """Nothing to do here: on Linux and macOS the launcher does it (``_launcher``)."""
        del pid
        return None


_warned: set[str] = set()


def _warn_once(message: str, key: str) -> None:
    """Warn the first time something goes wrong in this process (one ``key`` each)."""
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, stacklevel=3)


CLEANUP_UNAVAILABLE = "EngineCleanupUnavailable"
"""Type of the ``ApplicationError`` of a step stopped because its engine could not be
made to end with the Worker process (``engine_cleanup="required"``)."""


def _cleanup_problem(problem: str, *, required: bool, key: str) -> None:
    """Stop the step (``engine_cleanup="required"``), or warn once (``"best_effort"``).

    Args:
        problem: What went wrong, and what it means for a Worker process that dies.
        required: Whether the runner requires the engine to end with its Worker.
        key: One warning per key and Worker process.

    Raises:
        ApplicationError: If ``required`` (type ``CLEANUP_UNAVAILABLE``). Retryable:
            another Worker, or this one once fixed, can run the step.
    """
    message = f"temporalio.claude_agent_sdk: {problem}"
    if required:
        raise ApplicationError(
            f'{message} The step stopped, because engine_cleanup is "required" '
            '(engine_cleanup="best_effort" goes on anyway, with a warning).',
            type=CLEANUP_UNAVAILABLE,
        )
    _warn_once(
        f'{message} Going on anyway, because engine_cleanup is "best_effort".',
        key=key,
    )


def _tool_step_failure(
    reason: str, allowed: bool, *, non_retryable: bool
) -> ApplicationError:
    """The error of a failed tool step: whether its call may have run, and why.

    Args:
        reason: Why the step failed.
        allowed: Whether the call may have run: the hook let it run, or nothing
            shows that it did not (see ``_tool_step``).
        non_retryable: Whether retrying cannot help.
    """
    return ApplicationError(
        reason.removesuffix(" Retrying."),
        type=TOOL_CALL_INTERRUPTED if allowed else TOOL_CALL_NOT_RUN,
        non_retryable=non_retryable,
    )


def _cancelled_by_shutdown() -> bool:
    """Whether this Activity was cancelled because its Worker is shutting down.

    Only then: a cancel the Workflow requested, a pause, a reset or a timeout keeps
    its own meaning, even while the Worker shuts down.
    """
    if not activity.in_activity():
        return False
    details = activity.cancellation_details()
    return (
        details is not None
        and details.worker_shutdown
        and not (details.cancel_requested or details.paused or details.reset)
    )


def _job_problem(problem: str) -> str:
    """``problem`` with what it means: the engine would outlive a dead Worker."""
    return (
        f"{problem}: Claude Code would not end with this Worker process, and would "
        "finish its current turn if the process died."
    )


def _child_pids() -> set[int]:
    """The processes the SDK has started in this Worker process.

    It reads the SDK's own list (``_ACTIVE_CHILDREN``); a test checks that the SDK
    still keeps it.
    """
    try:
        from claude_agent_sdk._internal.transport import subprocess_cli

        children = list(subprocess_cli._ACTIVE_CHILDREN)
    except Exception:
        return set()
    return {
        pid
        for pid in (getattr(c, "pid", None) for c in children)
        if isinstance(pid, int)
    }


def _engines_end_with_worker() -> dict[int, str]:
    """Windows: put the Claude Code processes the SDK has started in the Worker's job.

    Every one is tried (``_child_pids``).

    Returns:
        The processes that could not join it, with what went wrong (as
        ``_job_problem`` words it); 0 stands for all of them. Empty elsewhere.
    """
    if sys.platform != "win32":
        return {}
    failed: dict[int, str] = {}
    try:
        for pid in _child_pids():
            problem = _end_with_worker(pid)
            if problem is not None:
                failed[pid] = _job_problem(problem)
    except Exception as err:
        failed[0] = _job_problem(f"cannot put Claude Code in a job object ({err!r})")
    return failed


_JOB_POLL = sys.platform == "win32"
"""Whether ``_JobWatch`` looks while the engine starts (tests set it elsewhere too)."""


class _JobWatch:
    """Windows: a step's engine joins the Worker's job object, or ``engine_cleanup``
    decides. Elsewhere the launcher did it before the engine started.

    It looks every 5 ms while the engine starts, on the Worker's event loop, so the
    engine is in the job within milliseconds, normally long before it starts a
    process of its own (an MCP server, a command); a process it starts before it
    joins (on a loop blocked for longer) stays outside the job. If this step's
    engine cannot join it and cleanup is required, the engine ends at once: the hook
    denies every call from then on, and the engine is terminated, not asked to
    finish its turn. (It cannot tell its engine from one that another step starts at
    the same moment, so such an engine's failure stops this step too; a job that
    fails, fails for every engine of the Worker.)
    """

    def __init__(self, hook_dir: str, *, required: bool) -> None:
        self._hook_dir = hook_dir
        self._required = required
        self._before = _child_pids()  # other steps' engines: not this step's
        self.problems: list[str] = []
        self._poll: asyncio.Future[None] | None = None
        if _JOB_POLL:
            self._poll = asyncio.ensure_future(self._polling())

    async def _polling(self) -> None:
        while True:
            self._check()
            await asyncio.sleep(0.005)

    def _check(self) -> None:
        failed = {
            pid: problem
            for pid, problem in _engines_end_with_worker().items()
            if pid not in self._before
        }
        for problem in failed.values():
            if problem not in self.problems:
                self.problems.append(problem)
        if failed and self._required:
            with contextlib.suppress(OSError):
                Path(self._hook_dir, "stop").touch()
            for pid in failed:
                if pid > 0:
                    with contextlib.suppress(OSError):
                        os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows

    def stop(self) -> None:
        if self._poll is not None:
            self._poll.cancel()

    def joined(self) -> None:
        """At the engine's first message, or when it ended without one.

        Raises:
            ApplicationError: If this step's engine could not join the job and
                cleanup is required.
        """
        self.stop()
        self._check()
        if self.problems:
            _cleanup_problem(self.problems[0], required=self._required, key="job")


def _hold_worker_lock(hook_dir: str, *, required: bool) -> int | None:
    """Lock ``<hook folder>/worker.lock`` for this run: the hook's sign that the Worker lives.

    The hook can take the lock only once this process is gone (the operating system
    releases it), and then denies every call. Python opens files non-inheritable, so
    the engine never holds the lock. Where locks do not work (some network or FUSE
    file systems), ``engine_cleanup`` decides: the step stops (``required``), or the
    file goes, the hook takes the Worker to be there, and a warning says so.

    Returns:
        The file descriptor to pass to ``_release_worker_lock``, or None.

    Raises:
        ApplicationError: If locks do not work and ``required``.
    """
    path = os.path.join(hook_dir, WORKER_LOCK)
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        _lock_file(fd)
        return fd
    except OSError as err:
        if fd is not None:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.remove(path)
        _cleanup_problem(
            f"cannot lock files in {tempfile.gettempdir()} ({err.strerror or err}): "
            "without the lock, the hook cannot tell that a dead Worker's engine "
            "should stop. Use a temporary folder where file locks work (for example, "
            "set TMPDIR to a folder on a local disk).",
            required=required,
            key="lock",
        )
        return None


def _close_call(hook_dir: str) -> bool:
    """End a tool step's chance to run its call; whether the hook let it run.

    Creates the hook's ``allowed`` record (atomic, empty) if the hook has not: from
    then on no hook of this step lets the call run, even while the engine is still
    shutting down. If the record is there, the hook let the call run.
    """
    path = os.path.join(hook_dir, ALLOWED)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except FileExistsError:
        return True
    except OSError:
        # Only a folder that is still there without the record shows that the call
        # never started (a command may have removed the folder, record and all).
        return not os.path.isdir(hook_dir) or os.path.lexists(path)
    return False


def _release_worker_lock(fd: int | None) -> None:
    if fd is None:
        return
    try:
        _unlock_file(fd)
    except OSError:
        pass  # closing releases it too
    os.close(fd)


_LAUNCHER = Path(__file__).with_name("_launcher.py")
_launch_dir: list[str] = []
_launch_scripts: dict[str, str | None] = {}
"""Engine path to its checked launcher script, or None when the script cannot run."""


def _private_folder(path: str) -> bool:
    """Whether ``path`` is still the private folder this process made."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        return False
    getuid = getattr(os, "getuid", None)
    return getuid is None or info.st_uid == getuid()


def _sh_quote(text: str) -> str:
    """``text`` as one word for ``/bin/sh``."""
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _write_launch_script(engine: str) -> str:
    """A ``/bin/sh`` script that starts ``engine`` through the launcher (``_launcher``).

    Isolated mode and no ``site``: the launcher needs only the standard library.
    The SDK's version probe before every start (``-v`` alone) goes straight to the
    engine, which prints its version and exits: no Python in between, about 20 ms
    of every step. Written to a new name and moved in place, so a run never sees
    half a script.
    """
    if not _launch_dir or not _private_folder(_launch_dir[0]):
        # A cleaner of temporary files may remove an old one: never write into a
        # folder of that name that someone else may have made since.
        folder = tempfile.mkdtemp(prefix="tca-launch-")
        atexit.register(shutil.rmtree, folder, True)
        _launch_dir[:] = [folder]
    path = Path(_launch_dir[0], hashlib.sha256(engine.encode()).hexdigest()[:16])
    command = " ".join(
        _sh_quote(part) for part in (sys.executable, "-I", "-S", str(_LAUNCHER), engine)
    )
    probe = f'if [ "$#" -eq 1 ] && [ "$1" = -v ]; then exec {_sh_quote(engine)} -v; fi'
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}")
    temporary.write_text(f'#!/bin/sh\n{probe}\nexec {command} "$@"\n', encoding="utf-8")
    temporary.chmod(0o700)
    os.replace(temporary, path)
    return str(path)


async def _launch_problem(script: str) -> str | None:
    """Why ``script`` cannot start the launcher here, or None if it can."""
    try:
        proc = await asyncio.create_subprocess_exec(
            script,
            LAUNCHER_CHECK,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as err:
        return err.strerror or str(err)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), 30)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return "no answer in 30 seconds"
    if proc.returncode == 0 and out.strip() == b"ok":
        return None
    return f"exit status {proc.returncode}"


def _command_env(
    env: dict[str, str],
    hook_dir: str,
    cwd: str | None,
    overrides: dict[str, str],
    extra: dict[str, str],
) -> dict[str, str]:
    """The engine's environment, with a file that gives each command the Worker's own.

    Claude Code runs ``CLAUDE_ENV_FILE`` before each Bash command, in the command's
    shell. This file puts back what the Worker had for every variable the engine
    gets in ``overrides`` (the values travel in ``TCA_KEEP_*`` variables, never in
    the file), then runs a ``CLAUDE_ENV_FILE`` of your own, as Claude Code would,
    and removes every ``TCA_*`` variable: commands see none of the plugin's own.

    Args:
        env: The engine's environment so far.
        hook_dir: The step's folder, where the file goes.
        cwd: The engine's working directory (a relative ``CLAUDE_ENV_FILE`` of your
            own is found from there, as Claude Code finds it).
        overrides: Variables the engine gets but commands must not.
        extra: The plugin's own variables to add for the engine (and its hook).

    Returns:
        The engine's environment.
    """
    before = {**os.environ, **env}
    path = str(Path(hook_dir, "command_env.sh"))
    overrides = {**overrides, "CLAUDE_ENV_FILE": path}
    keep: dict[str, str] = {}
    lines = [
        "# temporalio.claude_agent_sdk: give the command the Worker's own environment."
    ]
    for name in overrides:
        if before.get(name) is None:
            lines.append(f"unset {name}")
        else:
            keep[f"TCA_KEEP_{name}"] = before[name]
            lines.append(f'export {name}="$TCA_KEEP_{name}"')
    user_file = before.get("CLAUDE_ENV_FILE")
    if user_file:
        user_file = os.path.join(cwd or os.getcwd(), user_file)
        lines += [
            'if [ -f "$TCA_USER_ENV_FILE" ] && [ -r "$TCA_USER_ENV_FILE" ]; then',
            '  . "$TCA_USER_ENV_FILE"',
            "fi",
        ]
    result = {
        **env,
        **overrides,
        **keep,
        **({"TCA_USER_ENV_FILE": user_file} if user_file else {}),
        **extra,
    }
    # Names a shell takes (a stray name in the Worker's environment would print an
    # error into every command's output).
    hidden = sorted(
        n
        for n in {*os.environ, *result}
        if n.startswith("TCA_") and n.isidentifier() and n.isascii()
    )
    lines.append("unset " + " ".join(hidden))
    # Git Bash runs it on Windows too: LF line endings on every platform.
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return result


def _says_only(content: Any, reason: str) -> bool:
    """Whether a tool result is the hook's denial ``reason`` and nothing else.

    Claude Code records the reason as it is (2.1.273), or after
    ``PreToolUse:<tool> hook error: `` (2.1.287).
    """
    text = _text(content).strip() if content is not None else ""
    if text == reason:
        return True
    head, sep, rest = text.partition(" hook error: ")
    return bool(sep) and head.startswith("PreToolUse:") and rest == reason


def _hook_said_stopped(message: UserMessage) -> bool:
    """Whether a tool result in ``message`` is the hook's "step stopped" denial.

    The hook writes no record when it cannot see its folder, so this matches the
    denial itself: an error that is exactly the reason, never a tool's output that
    merely contains it.
    """
    if isinstance(message.content, str):
        return False
    return any(
        isinstance(block, ToolResultBlock)
        and bool(block.is_error)
        and _says_only(block.content, STOPPED)
        for block in message.content
    )


def _refused_ids(denials: Any) -> set[str]:
    """The ids of the calls in a result's ``permission_denials``."""
    if not isinstance(denials, list):
        return set()
    return {
        str(d["tool_use_id"])
        for d in cast("list[Any]", denials)
        if isinstance(d, dict) and d.get("tool_use_id")
    }


def _runs_as_activity(name: str, patterns: list[str]) -> bool:
    """Whether a Claude Code tool runs as an Activity: it matches ``tool_activities``.

    ``name`` can also be a permission rule, such as ``Bash(npm *)``.
    """
    tool = name.split("(", 1)[0].strip()
    return any(fnmatch.fnmatchcase(tool, p) for p in patterns if p)


def _decision_of(output: Any) -> str | None:
    """The decision a PreToolUse hook's output makes, or None if it makes none."""
    if not isinstance(output, dict):
        return None
    specific = output.get("hookSpecificOutput")
    if isinstance(specific, dict) and specific.get("permissionDecision"):
        return str(specific["permissionDecision"])
    if output.get("decision") in ("block", "approve"):
        return str(output["decision"])
    if output.get("continue") is False:
        return "stop"
    return None


def _guard_hooks(hooks: Any, decides_alone: Any, violations: list[str]) -> Any:
    """Wrap ``extra_options["hooks"]`` so they cannot decide on calls the plugin decides.

    A ``PreToolUse`` callback that returns a decision for a tool that runs as an
    Activity (a durable tool, or one of ``tool_activities``) would override the
    plugin's pause or approval: its decision is dropped, recorded in ``violations``,
    and the step then fails (see ``_hook_violation``).
    """
    if not isinstance(hooks, dict) or not hooks.get("PreToolUse"):
        return hooks

    def wrap(callback: Any) -> Any:
        async def guarded(input_data: Any, tool_use_id: Any, context: Any) -> Any:
            output = await callback(input_data, tool_use_id, context)
            name = str((input_data or {}).get("tool_name") or "")
            decision = _decision_of(output)
            if decision is not None and decides_alone(name):
                violations.append(f"{decision} on {name}")
                return {}
            return output

        return guarded

    matchers = [
        dataclasses.replace(m, hooks=[wrap(cb) for cb in m.hooks])
        if isinstance(m, HookMatcher)
        else m
        for m in hooks["PreToolUse"]
    ]
    return {**hooks, "PreToolUse": matchers}


def _hook_violation(violations: list[str]) -> str | None:
    if not violations:
        return None
    return (
        "A PreToolUse hook in extra_options decided on a call that runs as its own "
        f"Activity ({', '.join(sorted(set(violations)))}). The Workflow decides on "
        "those calls (needs_approval, tool_approvals), so make the hook return no "
        "decision for durable tools and the tools in tool_activities. This step "
        "stopped before any such call ran."
    )


def _as_outcome(value: Any) -> ToolOutcome:
    if isinstance(value, ToolOutcome):
        return value
    blocks = value.get("blocks")
    entry = value.get("entry")
    return ToolOutcome(
        content=value.get("content"),
        is_error=bool(value.get("is_error")),
        blocks=list(blocks) if isinstance(blocks, list) else None,
        entry=dict(entry) if isinstance(entry, dict) else None,
    )


_SAVED_OUTPUT = re.compile(
    r"<persisted-output>\r?\nOutput too large \([^)\r\n]*\)\. "
    r"Full output saved to: ([^\r\n]+)\r?\n"
)
"""How Claude Code's preview of a large output starts, naming the file with the rest."""


def _saved_output_tail(content: Any, project_key: str, session_id: str) -> str | None:
    """The end of an output Claude Code saved to a file in this step, if it did.

    Read it while the engine runs: the file is in the folder of the resumed session,
    ``<temp>/claude-resume-*/projects/<project_key>/<session_id>/tool-results``, which
    the SDK removes. Any other path is ignored: the text names it, and a tool's output
    (a command's, or an MCP tool's from an issue body) can contain such text, so a
    file anywhere else is never read into the conversation.
    """
    found = _SAVED_OUTPUT.match(content) if isinstance(content, str) else None
    if found is None:
        return None
    path = Path(found.group(1).strip())
    layout = path.parts[-6:-1]
    expected = ("projects", project_key, session_id, "tool-results")
    if len(layout) != 5 or not layout[0].startswith("claude-resume-"):
        return None
    if layout[1:] != expected or path.suffix != ".txt":
        return None
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None  # not a link to a file elsewhere
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - SAVED_OUTPUT_TAIL))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None


def _result_content(outcome: ToolOutcome) -> Any:
    """A tool result's content for Claude: content blocks as they are, else text."""
    return outcome.blocks if outcome.blocks is not None else _text(outcome.content)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


class ClaudeAgentSdkRunner:
    """Runs each segment with the Claude Agent SDK and its bundled Claude Code engine."""

    def __init__(
        self,
        *,
        session_store: Any = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        cli_path: str | None = None,
        extra_options: dict[str, Any] | None = None,
        one_tool_at_a_time: bool = False,
        model: str | None = None,
        max_budget_usd: float | None = None,
        warm_engines: int = 0,
        warm_seconds: float = 30.0,
        engine_cleanup: Literal["required", "best_effort"] = "required",
        workspace_store: WorkspaceStore | None = None,
        workspace_max_bytes: int | None = DEFAULT_MAX_BYTES,
        workspace_exclude: Sequence[str] = (),
    ) -> None:
        """Create the runner.

        Args:
            session_store: None (the default) keeps each conversation in its
                Workflow: nothing to set up, and any Worker can run any step. Or a
                Claude Agent SDK ``SessionStore`` that every Worker can reach, to
                keep conversations there instead (``FileSessionStore`` is for tests
                and one machine). Every Worker of a task queue needs the same choice.
            cwd: Working directory of the engine. Sessions are keyed by it, so give
                every Worker the same one. With a ``workspace_store``, the folder that
                holds each agent's own folder (``<cwd>/<id>``), where its engine
                works.
            env: Extra environment variables for the engine. The engine also
                inherits the Worker's environment.
            cli_path: Path of a Claude Code executable to use instead of the bundled one.
            extra_options: More ``ClaudeAgentOptions`` fields (for example
                ``permission_mode``, ``agents``, ``hooks``, ``setting_sources``,
                ``thinking``). ``env``, ``mcp_servers`` and ``allowed_tools`` are
                merged with the plugin's; ``system_prompt`` (a string or a preset) is
                the default for agents that set none. Options the agent or the plugin
                sets itself are refused, and so are ``extra_args`` for the same engine
                flags or for flags that approve tool calls (set ``permission_mode``
                here instead).
            one_tool_at_a_time: Ask Claude for one tool call per message. Not needed
                for durable tools: when Claude calls several at once, they all run.
            model: Default model when the agent does not set one.
            max_budget_usd: Cost cap per segment.
            warm_engines: Experimental. How many paused engines this Worker keeps
                running (0, the default: none). When the session's next segment
                runs on this Worker within ``warm_seconds``, at the same checkpoint
                and with only the result of the paused call, the result goes to the
                running engine as its next message: no new engine starts and
                resumes the session (with a local fake model, a tool round was
                about 7 times faster: 100 ms instead of 740 ms). A call of a Claude
                Code tool in ``tool_activities`` waits in the engine for its tool
                step instead of ending the turn: the step, still its own Activity,
                lets it run there when it comes to this Worker, and the turn then
                waits right after the call for the next segment (a round with an
                ``Edit``, a ``Write`` or Bash took about 0.14 to 0.24 s instead of
                1.35 to 1.97 s). Calls wait only in the permission modes ``default``
                and ``plan``, with a finite ``warm_seconds``, and one call per
                message. In any other case (another Worker, a retry, a reset, a
                longer wait) the warm engine ends, a call that waits there never
                runs there, and the step or the segment runs from the checkpoint as
                usual. Each warm engine keeps its process and memory (a few hundred
                MB); ``ClaudeAgentPlugin`` ends them when its Worker stops. No
                engine stays warm for segments with ``max_turns`` or
                ``max_budget_usd``, calls answered in one message (calls in one
                message never wait), or a runner whose ``extra_options`` has hooks,
                a permission or ``stderr`` callback, or in-process MCP servers;
                with the conversation in the Workflow, neither does the engine that
                starts it. A resumed engine runs ``PreToolUse`` hooks again for the
                call whose result it gets; a warm one does not. A warm engine keeps
                Claude Code's own notes between calls, as one Claude Code run does
                (for example, that a file changed on disk since Claude read it).
            warm_seconds: How long a paused engine stays warm.
            engine_cleanup: Whether each engine must end with this Worker process.
                ``"required"`` (the default): if the engine cannot start through its
                launcher (Linux, macOS), cannot join the Worker's job object
                (Windows), or the Worker cannot hold its lock file, the step stops
                with a retryable ``ApplicationError`` of type
                ``EngineCleanupUnavailable`` that says what to fix, before Claude
                Code runs anything (an engine can fail to join the job only after it
                started: the step then ends it at once). ``"best_effort"``: the step
                goes on with a warning, and if the Worker process dies, its engine
                finishes its current turn, built-in tools included.
            workspace_store: None (the default): every agent on this Worker works in
                ``cwd``, on this Worker's disk. Or a ``WorkspaceStore`` that every
                Worker can reach (``FileWorkspaceStore`` for tests, one machine or a
                shared mount): each agent then works in its own folder,
                ``<cwd>/<id>``, that follows it to any Worker. After each segment and
                tool step the runner saves what changed in the folder to the store,
                and the Workflow takes the new snapshot with the step's result; each
                segment and tool step first makes the folder match the agent's
                current snapshot (an attempt that did not complete is rolled back).
                Every Worker of a task queue needs the same store, ``cwd``,
                ``workspace_exclude`` and kind of system: a runner without a store
                refuses the steps of an agent whose folder has a snapshot (Temporal
                tries them again). An agent whose first segment ran without a store
                keeps working in ``cwd``.
            workspace_max_bytes: Most bytes of files an agent's folder may hold (1 GiB
                by default; None: no limit). A step that leaves more fails, not
                retryable (``WorkspaceTooLarge``).
            workspace_exclude: Paths left out of the snapshots, as ``fnmatch``
                patterns: never saved or restored, and left as they are (unless a
                directory that holds them is not in the snapshot). A pattern with no
                ``/``, or only a trailing one, matches a name at any depth
                (``node_modules``, ``*.log``); one with a ``/`` at its start or in its
                middle matches the path from the agent's folder (``build/*``,
                ``/cache``).

        Raises:
            ValueError: If ``extra_options`` sets an option the plugin manages, or
                Claude Code and the SDK would derive different session keys for the
                working directory (for example decomposed Unicode or emoji in its
                path, or a path given in another form than its real one), or
                ``warm_engines`` is negative, or ``warm_seconds`` not positive, or
                ``engine_cleanup`` is not ``"required"`` or ``"best_effort"``, or
                ``workspace_max_bytes`` is not positive, or a ``workspace_exclude``
                pattern is empty.
            TypeError: If ``workspace_store`` has no async ``save`` and ``load``, or
                ``workspace_exclude`` is one string instead of a list of patterns.
        """
        _check_extra_options(extra_options or {})
        if warm_engines < 0 or not warm_seconds > 0:  # NaN too
            raise ValueError("warm_engines must be 0 or more, warm_seconds above 0")
        cleanup = cast("str", engine_cleanup)  # a caller may pass anything
        if cleanup not in ("required", "best_effort"):
            raise ValueError(
                f'engine_cleanup must be "required" or "best_effort", not {cleanup!r}'
            )
        directory = cwd or os.getcwd()
        mismatch = _key_mismatch(directory)
        if mismatch is not None:
            raise ValueError(
                f"cwd {directory!r}: Claude Code and the Claude Agent SDK would derive "
                f"different session keys ({mismatch}), so resumed sessions would not "
                "be found. Pass the path exactly as os.path.realpath gives it, or use "
                "a directory with a plain ASCII path."
            )
        self._workspaces: Workspaces | None = None
        if workspace_store is not None:
            store_methods = [
                getattr(workspace_store, name, None) for name in ("save", "load")
            ]
            if not all(inspect.iscoroutinefunction(m) for m in store_methods):
                raise TypeError(
                    "workspace_store must have async save(key, path) and "
                    "load(key, path) methods (see WorkspaceStore)"
                )
            if isinstance(cast("object", workspace_exclude), str):
                raise TypeError(
                    "workspace_exclude takes a list of patterns, not one string"
                )
            if workspace_max_bytes is not None and not workspace_max_bytes > 0:
                raise ValueError("workspace_max_bytes must be above 0, or None")
            self._workspaces = Workspaces(
                directory,
                workspace_store,
                max_bytes=workspace_max_bytes,
                exclude=list(workspace_exclude),
            )
        self._store = session_store
        self._cwd = cwd
        self._env = env or {}
        self._cli_path = cli_path
        self._extra = extra_options or {}
        self._one_tool = one_tool_at_a_time
        self._model = model
        self._max_budget = max_budget_usd
        self._warm_cap = warm_engines
        self._warm_seconds = warm_seconds
        self._cleanup_required = cleanup == "required"
        self._warm: dict[tuple[str, str], _WarmEngine] = {}
        """Paused engines by session and checkpoint (see ``warm_engines``)."""
        self.stub_calls = 0
        """Durable tools the engine ran itself. Stays 0 while the engine honors defer."""
        self._stand_in = StandInModel()
        self._versions: dict[str, str | None] = {}
        self._launch: str | None = (
            None  # the checked launcher script (``_launch_path``)
        )

        def is_set(name: str) -> bool:
            value = self._env.get(name) or os.environ.get(name, "")
            return value.lower() not in ("", "0", "false", "no")

        if not any(is_set(name) for name in ENV_AUTH):
            warnings.warn(
                "temporalio.claude_agent_sdk: no API key, cloud provider, or "
                "CLAUDE_CODE_OAUTH_TOKEN is set. A Claude app login cannot refresh "
                "itself when a session is resumed (every step after the first), so "
                "long-running agents can fail with 'OAuth session expired'. Set "
                "ANTHROPIC_API_KEY, "
                "use Bedrock or Vertex, or run `claude setup-token` and set "
                "CLAUDE_CODE_OAUTH_TOKEN.",
                stacklevel=2,
            )

    def _engine_path(self) -> str | None:
        """The Claude Code executable the SDK will start (same order the SDK uses)."""
        if self._cli_path:
            return str(self._cli_path)
        import claude_agent_sdk

        name = "claude.exe" if os.name == "nt" else "claude"
        bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / name
        return str(bundled) if bundled.is_file() else shutil.which("claude")

    async def _launch_path(self) -> str | None:
        """The launcher script for this runner's engine, checked (Linux, macOS).

        The SDK starts the engine through it, so the engine ends with this Worker
        process (see ``_launcher``). When the script cannot run (for example a
        temporary folder mounted noexec), ``engine_cleanup`` decides: the step stops,
        and the next one checks again (``"required"``), or the engine starts
        directly, with a warning once (``"best_effort"``).

        Raises:
            ApplicationError: If the script cannot run and cleanup is required.
        """
        if sys.platform == "win32":
            return None
        engine = self._engine_path()
        if engine is None:
            return None  # the SDK reports the missing engine
        known = engine in _launch_scripts
        script = _launch_scripts.get(engine)
        if script is not None and not (
            os.path.isfile(script) and _private_folder(os.path.dirname(script))
        ):
            # A cleaner of temporary files removed it (a Worker idle for days).
            script = _launch_scripts[engine] = _write_launch_script(engine)
        if script is None and (not known or self._cleanup_required):
            # Checked the first time, and again for a runner that requires it.
            script = _write_launch_script(engine)
            problem = await _launch_problem(script)
            if problem is None:
                _launch_scripts[engine] = script
                return script
            if not self._cleanup_required:
                _launch_scripts[engine] = None  # not checked again
            _cleanup_problem(
                f"cannot run {script} ({problem}): without it, Claude Code would not "
                "end with this Worker process, and would finish its current turn if "
                "the process died. Make the temporary folder allow running programs "
                "(for example, set TMPDIR to a folder on a local disk).",
                required=self._cleanup_required,
                key=f"launch {engine}",
            )
            return None
        return script

    async def _prepare_engine(self) -> None:
        """Arrange for engines to end with this Worker process.

        Linux and macOS: the launcher (``_launch_path``). Windows: the Worker's job
        object, created once; each engine joins it as it starts (``_JobWatch``).

        Raises:
            ApplicationError: If that cannot be arranged and cleanup is required,
                before any engine starts.
        """
        self._launch = await self._launch_path()
        problem = _worker_job()
        if problem is not None:
            _cleanup_problem(
                _job_problem(problem), required=self._cleanup_required, key="job"
            )

    async def _engine_version(self) -> str | None:
        """``claude -v`` of the engine, checked once per executable (None if unknown)."""
        path = self._engine_path()
        if path is None:
            return None
        if path not in self._versions:
            reported: str | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    path,
                    "-v",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    out, _ = await asyncio.wait_for(proc.communicate(), 30)
                    reported = out.decode(errors="replace").strip() or None
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            except OSError:
                reported = None
            self._versions[path] = reported
        return self._versions[path]

    async def _start(
        self, inp: SegmentInput, attempt: int, injected: dict[str, ToolOutcome]
    ) -> tuple[str, bool, set[str], str | None, bool]:
        """Where a segment starts in the session store.

        A segment that runs again continues in a copy of the session that ends at
        the checkpoint, because an unfinished attempt may have written after it.

        Returns:
            The session, whether to resume it, the calls whose results are already
            in it (see ``_copy`` and ``_record_copy``), the entry it must still end at
            when the engine resumes it in place (None for a copy), and whether it
            ends with Claude Code's record of a paused call's result (see ``_place``).
        """
        if inp.checkpoint is None:  # nothing committed yet: a new session
            return (
                _attempt_session_id(inp.session_id, attempt),
                False,
                set(),
                None,
                False,
            )
        if attempt == 1 and not inp.fork:
            # Only a tool step's result (``entry``), or an error (a step that failed,
            # a rejected call), can be an Edit's or a Write's, or the result of a call
            # that waited for its step in a warm engine (its step returns a record for
            # any tool: the session has no deferral marker to resume at). Others go as
            # before, with no read of the session here.
            if any(o.entry is not None or o.is_error for o in injected.values()):
                copied = await self._record_copy(inp, injected)
                if copied is not None:
                    return copied[0], True, copied[1], None, True
            return inp.session_id, True, set(), inp.checkpoint, False
        session_id, delivered, recorded = await self._copy(inp, injected)
        return session_id, True, delivered, None, recorded

    async def _record_copy(
        self, inp: SegmentInput, injected: dict[str, ToolOutcome]
    ) -> tuple[str, set[str]] | None:
        """A copy of the session with Claude Code's record of the paused call's result.

        Only when the call is one of ``RECORDED_TOOLS`` (see ``_place``). The record
        takes the place of the pause's hook entries, as in the conversation the
        Workflow holds. (Added to the session itself, after the hook entries, the
        record would leave the pause's deferral marker in the session: a later copy
        of it, after a retry or a pause in a message with several calls, then makes
        Claude Code take the paused call of that copy for interrupted, tested on
        Claude Code 2.1.273.)

        Returns:
            The copy's id, and the calls whose results it holds; or None when the
            session goes on as before: in place, with the results in a message.
        """
        assert inp.checkpoint is not None
        key = {
            "project_key": project_key_for_directory(self._folder()),
            "session_id": inp.session_id,
        }
        entries = cast("list[dict[str, Any]]", await self._store.load(key) or [])
        placed = _place(entries, inp.checkpoint, injected)
        if placed is None:
            return None
        return await self._write_copy(key, placed[0]), placed[1]

    async def _write_copy(self, key: dict[str, Any], seed: list[Any]) -> str:
        """Write ``seed`` to the store as a new session; return its id."""
        copy_id = str(uuid.uuid4())
        await self._store.append(
            {**key, "session_id": copy_id},
            [{**e, "sessionId": copy_id} if "sessionId" in e else e for e in seed],
        )
        return copy_id

    async def _copy(
        self, inp: SegmentInput, injected: dict[str, ToolOutcome]
    ) -> tuple[str, set[str], bool]:
        """A copy of the session that ends at the checkpoint.

        After a pause in a message with several calls, the copy also holds the
        results of the calls denied after the pause (see ``_deliver``), and after a
        pause at an Edit or a Write, Claude Code's record of its result (see
        ``_place``).

        Returns:
            The copy's id, the calls whose results it holds, and whether it ends
            with such a record.
        """
        assert inp.checkpoint is not None
        key = {
            "project_key": project_key_for_directory(self._folder()),
            "session_id": inp.session_id,
        }
        entries = cast("list[dict[str, Any]]", await self._store.load(key) or [])
        placed = _place(entries, inp.checkpoint, injected)
        moved = placed or _deliver(entries, inp.checkpoint, injected)
        if moved is None:
            forked = await fork_session_via_store(
                self._store,
                inp.session_id,
                directory=self._folder(),
                up_to_message_id=inp.checkpoint,
            )
            return forked.session_id, set(), False
        seed, delivered = moved
        return await self._write_copy(key, seed), delivered, placed is not None

    async def _checkpoint(
        self,
        store: Any,
        session_id: str,
        assistant_uuid: str | None,
        paused_call: str | None,
        resumed_at: str | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Where the next segment continues, read back from the store (see ``_resume_point``).

        Reading it back also proves that the turn reached the session store.

        Returns:
            The checkpoint, and the session's entries.

        Raises:
            RuntimeError: If the store does not have the turn (Temporal retries).
        """
        key = {
            "project_key": project_key_for_directory(self._folder()),
            "session_id": session_id,
        }
        entries = cast("list[dict[str, Any]]", await store.load(key) or [])
        stored = assistant_uuid is None or any(
            e.get("uuid") == assistant_uuid for e in entries
        )
        leaf = _resume_point(entries, paused_call)
        if not stored or leaf is None or leaf == resumed_at:  # nothing new stored
            raise RuntimeError(
                f"This segment's turn did not reach the session store (session "
                f"{session_id}), so it could not be continued. Retrying."
            )
        return leaf, entries

    def _may_park(self, inp: SegmentInput) -> bool:
        """Whether this segment's engine may stay warm when it pauses.

        Not with a limit on turns or cost: the engine counts them per run. Not with
        hooks, a permission callback, a ``stderr`` callback or in-process MCP servers
        in ``extra_options``: the SDK runs them in tasks of the segment that started
        the engine, so in a later segment they would see that segment's Activity
        (its info, heartbeats and cancellation).
        """
        servers = self._extra.get("mcp_servers")
        return (
            self._warm_cap > 0
            and inp.max_turns is None
            and self._max_budget is None
            and not self._extra.get("hooks")
            and self._extra.get("can_use_tool") is None
            and self._extra.get("stderr") is None
            and not (
                isinstance(servers, dict)
                and any(
                    isinstance(c, dict) and c.get("type") == "sdk"
                    for c in servers.values()
                )
            )
        )

    def _may_wait(self, inp: SegmentInput) -> bool:
        """Whether a call of a Claude Code tool in ``tool_activities`` may wait for its
        tool step in this segment's engine (warm tool calls).

        Only in an engine that may stay warm, with a finite ``warm_seconds``, and in a
        permission mode where Claude Code refuses the call if its hook ever gives no
        answer (``WAIT_MODES``): a hook that waits long could be killed.
        """
        mode = self._extra.get("permission_mode", DEFAULT_PERMISSION_MODE)
        return (
            bool(inp.tool_activities)
            and self._may_park(inp)
            and math.isfinite(self._warm_seconds)
            and mode in WAIT_MODES
        )

    @staticmethod
    def _shape(inp: SegmentInput) -> str:
        """What an engine is started with that the next segment must share."""
        return json.dumps(
            [
                inp.system_prompt,
                inp.model,
                [[t.name, t.description, t.input_schema] for t in inp.tools],
                inp.builtin_tools,
                inp.tool_activities,
            ],
            sort_keys=True,
            default=str,
        )

    def _take_warm(
        self, inp: SegmentInput, injected: dict[str, ToolOutcome], attempt: int
    ) -> _WarmEngine | None:
        """The warm engine this segment continues, taken from the pool, or None.

        Never for a retry: an earlier attempt may still run somewhere and go on from
        the same checkpoint; a retry starts a new engine, as without warm engines.
        An engine whose call still waits for its tool step (``call_waits``) is never
        taken: its step did not come to this Worker, so the call ran elsewhere.
        """
        if inp.checkpoint is None:
            return None
        warm = self._warm.pop((inp.session_id, inp.checkpoint), None)
        if warm is None:
            return None
        if warm.timer is not None:
            warm.timer.cancel()
        shared = (
            attempt == 1
            and inp.prompt is None
            and not inp.fork
            and self._may_park(inp)
            and set(injected) == {warm.paused}
            and warm.shape == self._shape(inp)
            and _engine_alive(warm.client)
        )
        if warm.state == "turn_waits":
            # The call ran here in its tool step: the segment must bring exactly what
            # the step returned, and the turn must still wait (had its after-calls
            # hook died, it would have gone on without a segment).
            assert warm.outcome is not None and warm.inbox is not None
            went_on = any(
                isinstance(m, (AssistantMessage, ResultMessage))
                for m in warm.inbox.items
            )
            if (
                shared
                and not went_on
                and _same_outcome(injected[warm.paused], warm.outcome)
            ):
                return warm
        elif warm.state == "paused":
            payload = sum(len(_text(_result_content(o))) for o in injected.values())
            if (
                shared
                and all(o.entry is None for o in injected.values())  # no tool step's
                and 8 * payload <= warm.buffer
            ):
                return warm
        self._end_warm(warm)
        return None

    def _park(self, warm: _WarmEngine) -> None:
        """Keep a paused engine for ``warm_seconds``, ending the oldest beyond the cap."""
        key = (warm.session_id, warm.checkpoint)
        old = self._warm.pop(key, None)
        if old is not None:
            self._end_warm(old)
        while self._warm and len(self._warm) >= self._warm_cap:
            self._end_warm(self._warm.pop(next(iter(self._warm))))
        self._warm[key] = warm
        warm.timer = asyncio.get_running_loop().call_later(
            self._warm_seconds, self._expire, key, warm
        )

    def _expire(self, key: tuple[str, str], warm: _WarmEngine) -> None:
        if self._warm.get(key) is warm:
            del self._warm[key]
            self._end_warm(warm)

    @staticmethod
    def _end_warm(warm: _WarmEngine) -> None:
        """End a warm engine in the background; its hook folder and lock go too.

        The folder and the lock go once the ending task is done, also if it was
        cancelled before it ran (an event loop that closes): the engine then gets
        SIGTERM at once.
        """
        if warm.timer is not None:
            warm.timer.cancel()
        if warm.state != "paused":
            # A call or a turn waits in a hook: the hook sees ``stop`` and denies the
            # call (it never runs) or stops the turn (no model request follows).
            # ``turn_gate`` first, so a call denied now does not let the turn go on.
            for name in (TURN_GATE, "stop"):
                with contextlib.suppress(OSError):
                    Path(warm.hook_dir, name).touch()
        process = getattr(getattr(warm.client, "_transport", None), "_process", None)
        resumed = getattr(warm.client, "_materialized", None)
        task = asyncio.ensure_future(_disconnect(warm.client))
        _ending.add(task)

        def ended(task: asyncio.Task[None]) -> None:
            _ending.discard(task)
            _stop_now(process, resumed)
            _release_worker_lock(warm.lock)
            _remove_hook_folder(warm.hook_dir)
            if warm.inbox is not None:
                warm.inbox.stop()

        task.add_done_callback(ended)

    async def _ends_at_checkpoint(self, inp: SegmentInput, warm: _WarmEngine) -> bool:
        """Whether the shared session is still where the warm engine left it.

        A warm engine knows the session as it left it. When a segment from the same
        checkpoint ran on another Worker (a retry, or a reset), the session went on
        there, and only a new engine continues it right (in a copy, ``_SessionMoved``).
        After a call that ran here in its tool step, the session also holds what the
        engine wrote since the checkpoint (the call's result): every entry after the
        checkpoint must be the engine's own.
        """
        key = {
            "project_key": project_key_for_directory(self._folder()),
            "session_id": inp.session_id,
        }
        entries = cast("list[Any]", await self._store.load(key) or [])
        if warm.state != "turn_waits":
            return _last_entry(entries) == inp.checkpoint
        uuids = [e["uuid"] for e in entries if _is_transcript(e)]
        if inp.checkpoint not in uuids or warm.view is None:
            return False
        after = uuids[uuids.index(inp.checkpoint) + 1 :]
        return all(u in warm.view.written for u in after)

    def _end_all_warm(self) -> None:
        """End every warm engine, each in a task of its own (the Worker stopped).

        Nothing waits for them here: ``async with Worker`` cancels the Worker's run
        once it shut down, and a run that still waited would then cancel the code
        that comes after the ``async with`` block.
        """
        while self._warm:
            self._end_warm(self._warm.pop(next(iter(self._warm))))

    def _end_session_warm(self, session_id: str) -> None:
        """End the session's warm engines: its folder had to change under them.

        A call that waits in one never runs there; the step runs it in a new engine.
        """
        for key in [k for k in self._warm if k[0] == session_id]:
            self._end_warm(self._warm.pop(key))

    def _has_folder(self, workspace: Workspace | None, checkpoint: str | None) -> bool:
        """Whether this step runs in the agent's own folder (see ``workspace_store``).

        Not without a workspace store, nor for an agent that started without one: its
        first segment ran on a Worker with no store (so its Workflow has a checkpoint
        but no snapshot), or before the plugin knew of workspaces (no ``workspace``
        in the input). Such an agent keeps working in ``cwd``, as before.
        """
        return (
            self._workspaces is not None
            and workspace is not None
            and (workspace.snapshot is not None or not checkpoint)
        )

    def _check_store_for(self, workspace: Workspace | None) -> None:
        """Refuse a step of an agent whose folder is kept in a workspace store.

        Only on a runner without a store: the step would run in ``cwd``, where the
        agent's files are not.

        Raises:
            RuntimeError: Temporal tries the step again, on a Worker with the store
                once there is one (as when a session store is missing).
        """
        if (
            self._workspaces is None
            and workspace is not None
            and workspace.snapshot is not None
        ):
            raise RuntimeError(
                "This agent's folder is kept in a workspace store, but this Worker's "
                "runner has none, so the step would not see the agent's files. Every "
                "Worker of a task queue needs the same workspace_store. Retrying."
            )

    def _folder(self) -> str | None:
        """The engine's working directory in this segment or tool step.

        With a workspace store, the agent's own folder (set by ``run`` and
        ``run_tool_step`` for the step, and seen by every task it starts); otherwise
        the runner's ``cwd``.
        """
        return _AGENT_FOLDER.get() or self._cwd

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """Run one segment until Claude pauses at a durable tool call or finishes.

        With a workspace store, the agent's folder first matches its committed
        snapshot (see ``workspace_store``), and the segment's output carries the
        snapshot saved after it, unless the output is an error.

        Args:
            inp: The segment input.
            attempt: The Activity attempt number, starting at 1.

        Returns:
            The pause or the final answer, with the segment's checkpoint (and, when
            the Workflow holds the conversation, what changed in it). ``is_error``
            is set when the engine broke the pause contract, or when retrying cannot
            help.

        Raises:
            RuntimeError: If the transcript did not reach the session store, the
                Workflow did not serve its conversation, or the conversation or the
                agent's folder is kept where this runner does not keep it (a session
                store, the Workflow, or a workspace store). Temporal retries the
                segment.
            ApplicationError: Not retryable, if the agent's workspace cannot be
                restored or saved (types ``WorkspaceNotFound``, ``WorkspaceCorrupt``,
                ``WorkspaceTooLarge``, ``WorkspaceInvalid``).
        """
        self._check_store_for(inp.workspace)
        reported = await self._engine_version()
        if reported is not None and _too_old(reported):
            return _too_old_output(inp.session_id, reported)  # before the engine starts
        await self._prepare_engine()
        if not self._has_folder(inp.workspace, inp.checkpoint):
            return await self._run(inp, attempt)
        assert self._workspaces is not None and inp.workspace is not None
        async with self._workspaces.use(inp.workspace) as folder:
            if folder.changed:
                self._end_session_warm(inp.session_id)
            token = _AGENT_FOLDER.set(folder.folder)
            try:
                out = await self._run(inp, attempt)
            finally:
                _AGENT_FOLDER.reset(token)
            if not out.is_error:
                out.workspace = await folder.save()
            return out

    async def _run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """``run``'s work, in the engine's working directory (``_folder``)."""
        injected = {k: _as_outcome(v) for k, v in inp.injected.items()}
        if self._store is None:
            return await self._run_held(inp, injected, attempt)
        if inp.transcript or (
            inp.conversation is not None and inp.conversation.entries
        ):
            raise RuntimeError(
                f"Session {inp.session_id} is kept in its Workflow, but this Worker's "
                "runner has a session store. Every Worker of a task queue needs the "
                "same choice: create this runner without session_store. Retrying."
            )
        warm = self._take_warm(inp, injected, attempt)
        if warm is not None:
            try:
                fresh = await self._ends_at_checkpoint(inp, warm)
            except BaseException:
                self._end_warm(warm)
                raise
            if not fresh:
                self._end_warm(warm)
                warm = None
        if warm is not None:
            return await self._run_engine(
                inp,
                injected,
                warm.session_id,
                True,
                warm.store,
                guard=inp.checkpoint,
                # A call that ran in this engine has its result in it already.
                delivered={warm.paused} if warm.state == "turn_waits" else set(),
                warm=warm,
            )
        session_id, resume, delivered, guard, continuing = await self._start(
            inp, attempt, injected
        )
        if (
            resume
            and len(injected) == len(delivered)
            and inp.prompt is None
            and not continuing
        ):
            return SegmentOutput(
                session_id=session_id,
                is_error=True,
                error="Nothing to send: no tool result and no prompt",
            )
        try:
            try:
                return await self._run_engine(
                    inp,
                    injected,
                    session_id,
                    resume,
                    self._store,
                    guard=guard,
                    delivered=delivered,
                    continuing=continuing,
                )
            except _SessionMoved:
                # The session went on after the checkpoint (after a pause in a
                # message with several calls, or a Workflow reset to an earlier
                # point): continue in a copy that ends there.
                session_id, delivered, continuing = await self._copy(inp, injected)
                return await self._run_engine(
                    inp,
                    injected,
                    session_id,
                    True,
                    self._store,
                    delivered=delivered,
                    continuing=continuing,
                )
        except _NotContinued as err:
            return _not_continued_output(session_id, err, attempt)

    async def _run_held(
        self, inp: SegmentInput, injected: dict[str, ToolOutcome], attempt: int
    ) -> SegmentOutput:
        """Run a segment of a conversation the Workflow holds.

        The engine resumes from a fresh in-memory session store that holds the
        committed conversation up to the checkpoint, so every attempt starts from
        what the Workflow committed.
        """
        committed = await read_conversation(inp)
        warm = self._take_warm(inp, injected, attempt)
        if warm is not None:
            try:
                # The conversation must hold the checkpoint, and no calls answered in
                # the same message (their results go into the conversation, which
                # only a new engine reads).
                clean = (
                    bool(committed)
                    and _seed(committed, warm.checkpoint) is not None
                    and _deliver(committed, warm.checkpoint, injected) is None
                )
            except BaseException:
                self._end_warm(warm)
                raise
            if clean:
                return await self._run_engine(
                    inp,
                    injected,
                    warm.session_id,
                    True,
                    warm.store,
                    committed=committed,
                    # A call that ran in this engine has its result in it already.
                    delivered={warm.paused} if warm.state == "turn_waits" else set(),
                    warm=warm,
                )
            self._end_warm(warm)
        delivered: set[str] = set()  # results the seed already holds
        placed = None  # the seed ends with Claude Code's record of a result
        if inp.checkpoint is None:
            if committed:
                return SegmentOutput(
                    session_id=inp.session_id,
                    is_error=True,
                    error=(
                        f"The Workflow holds {len(committed)} conversation entries but "
                        "no checkpoint, so it is unclear where Claude would continue."
                    ),
                )
            session_id, seed, resume = (
                _attempt_session_id(inp.session_id, attempt),
                [],
                False,
            )
        else:
            if not committed:
                raise RuntimeError(
                    f"The Workflow holds no conversation for session {inp.session_id} "
                    f"(checkpoint {inp.checkpoint}): it is kept in a session store this "
                    "runner does not have, or AgentState.transcript was not carried to "
                    "this run. Give this Worker's runner the session store the "
                    "conversation started with. Retrying."
                )
            placed = _place(committed, inp.checkpoint, injected)
            moved = placed or _deliver(committed, inp.checkpoint, injected)
            found = moved[0] if moved is not None else _seed(committed, inp.checkpoint)
            if found is None:
                return SegmentOutput(
                    session_id=inp.session_id,
                    is_error=True,
                    error=(
                        f"Checkpoint {inp.checkpoint} is not in the conversation the "
                        "Workflow holds, so it is unclear where Claude would continue."
                    ),
                )
            if moved is not None:
                delivered = moved[1]
            session_id, seed, resume = inp.session_id, _without_cost_state(found), True
        continuing = resume and placed is not None
        if (
            resume
            and len(injected) == len(delivered)
            and inp.prompt is None
            and not continuing
        ):
            return SegmentOutput(
                session_id=session_id,
                is_error=True,
                error="Nothing to send: no tool result and no prompt",
            )
        store = InMemorySessionStore()
        if seed:
            key = {
                "project_key": project_key_for_directory(self._folder()),
                "session_id": session_id,
            }
            await store.append(key, seed)  # type: ignore[arg-type]
        # A copy that is already there is not this run's (Claude Code then refuses the
        # session id), so it stays.
        local = self._local_copy(session_id)
        ours = not resume and not any(p.exists() for p in local)
        try:
            return await self._run_engine(
                inp,
                injected,
                session_id,
                resume,
                store,
                committed=committed,
                delivered=delivered,
                local_copy=local if ours else (),
                continuing=continuing,
                resumed_at=_last_entry(seed) if continuing else None,
            )
        except _NotContinued as err:
            return _not_continued_output(session_id, err, attempt)

    async def run_tool_step(self, step: ToolStepInput, attempt: int) -> ToolOutcome:
        """Run one Claude Code tool call that paused a segment, as its own Activity.

        Brian Strauch's bounded native call replay (hybrid prototype): Claude Code
        resumes a private copy of the conversation that ends before the call (in
        memory: a session store is not written, see ``_step_context``). A local
        stand-in model answers with the call itself, the exact ``tool_use`` block
        Claude sent, so no real model is asked; the hook lets exactly that call run;
        and ``max_turns=1`` ends the turn after it. Claude Code runs the call in an
        ordinary turn, as it would have in the segment, and writes its own result
        entry. For an Edit or a Write, the step returns it with the result
        (``ToolOutcome.entry``; one of the same shape if Claude Code wrote none), and
        the next segment continues from it (see ``_place``). Tested on Claude Code
        2.1.273, 2.1.274 and 2.1.288 with Bash, Edit and Write: the tool runs once,
        with the engine's own result, and one model answer.

        No permission mode, ``allowed_tools`` or callback approves a tool in the
        step: the hook's "allow" lets the call run, and only with the input the
        segment reported (which is what an approval saw), as Claude Code is about to
        run it. If deciding fails, the hook denies the call. Only a hook that gives
        no answer at all (it could not start) leaves the call to Claude Code's own
        permission check, which refuses it unless a rule in a settings file allows
        it.

        The command still sees the Worker's own environment (see ``_step_env``), and
        a result the step already has is kept even if the engine fails afterwards,
        so the Activity is not retried for a call that ran.

        With warm engines, the call may wait for this step in its segment's engine on
        this Worker (see ``warm_engines``): for exactly that call, with the input its
        hook saw, the step lets it run there instead (``_step_in_warm``). The engine
        runs it as it would have in the segment, and the same rules hold: the hook
        records the call's start before it runs, and a step that ends first makes
        sure it never does.

        A step that fails says on which side of the call's start it failed: the hook
        writes ``allowed`` just before it lets the call run. Without that record, the
        call did not run only if the engine never got it, or the hook or Claude
        Code's permission check refused it; otherwise it may have run. So the
        Workflow can tell Claude whether the call ran; it gives the step one attempt
        unless the tool is in ``repeatable_tools``.

        With a workspace store, the agent's folder first matches its committed
        snapshot, and the result carries the snapshot saved after the call
        (``ToolOutcome.workspace``). A step that fails saves none: the next step
        rolls back what its call changed in the folder.

        Args:
            step: The call, and where its session paused.
            attempt: The Activity attempt number.

        Returns:
            What the tool returned, as Claude Code would show it to Claude.

        Raises:
            ApplicationError: Of type ``ToolCallNotRun`` if the step failed before
                Claude Code could run the call (it did not run), or
                ``ToolCallInterrupted`` if it failed after the hook let the call run
                (it may have run). Not retryable where retrying cannot help: the
                session did not pause at this call, or the hook refused it (also
                for other input than the segment reported, or when it failed), or
                the agent's workspace cannot be restored or saved. (A hook in
                ``extra_options`` that decides on the call already stopped the task
                in the segment, where the same call came first.)
        """
        del attempt
        allowed = [False]  # whether the call may have run (``_tool_step`` sets it)
        try:
            self._check_store_for(step.workspace)
            if not self._has_folder(step.workspace, step.checkpoint):
                return await self._tool_step(step, allowed)
            assert self._workspaces is not None and step.workspace is not None
            async with self._workspaces.use(step.workspace) as folder:
                if folder.changed:
                    self._end_session_warm(step.session_id)
                token = _AGENT_FOLDER.set(folder.folder)
                try:
                    outcome = await self._tool_step(step, allowed)
                finally:
                    _AGENT_FOLDER.reset(token)
                # The step has the call's result, so the call ran, even if no hook
                # let it (an allow rule in a settings file can): a save that fails
                # now says it may have run, so it never runs again.
                allowed[0] = True
                outcome.workspace = await folder.save()
                return outcome
        except asyncio.CancelledError:
            if not _cancelled_by_shutdown():
                raise  # the Workflow cancelled the step, or it timed out, or ...
            raise _tool_step_failure(
                "Its Worker shut down.", allowed[0], non_retryable=False
            ) from None
        except ApplicationError as err:
            if err.type in (TOOL_CALL_NOT_RUN, TOOL_CALL_INTERRUPTED):
                raise
            raise _tool_step_failure(
                err.message, allowed[0], non_retryable=err.non_retryable
            ) from err
        except Exception as err:
            raise _tool_step_failure(
                str(err) or repr(err), allowed[0], non_retryable=False
            ) from err

    async def _tool_step(self, step: ToolStepInput, allowed: list[bool]) -> ToolOutcome:
        """``run_tool_step``'s work; sets ``allowed[0]`` when the call may have run."""
        await self._prepare_engine()
        waiting = self._take_waiting(step)
        if waiting is not None:
            return await self._step_in_warm(step, waiting, allowed)
        return await self._step_in_new_engine(step, allowed)

    async def _step_in_new_engine(
        self, step: ToolStepInput, allowed: list[bool]
    ) -> ToolOutcome:
        """A tool step in a new engine (see ``run_tool_step``)."""
        call = step.call
        key = {
            "project_key": project_key_for_directory(self._folder()),
            "session_id": step.session_id,
        }
        held = step.transcript is not None or (
            step.conversation is not None and step.conversation.entries > 0
        )
        if self._store is None and not held:
            raise RuntimeError(
                f"Session {step.session_id} is kept in a session store this runner "
                "does not have. Every Worker of a task queue needs the same choice: "
                "give this runner the session store. Retrying."
            )
        if self._store is not None and held:
            raise RuntimeError(
                f"Session {step.session_id} is kept in its Workflow, but this Worker's "
                "runner has a session store. Every Worker of a task queue needs the "
                "same choice: create this runner without session_store. Retrying."
            )
        if self._store is None:
            entries = await read_conversation(step)
        else:
            entries = cast("list[Any]", await self._store.load(key) or [])
        seed = _seed(entries, step.checkpoint)
        span = _hook_span(seed, step.checkpoint) if seed is not None else None
        found = _calls_of(seed[: span[0]], call.id) if seed and span else None
        block = _recorded_call(seed[: span[0]], call.id) if seed and span else None
        if (
            seed is None
            or span is None
            or found is None
            or span[2] != call.id
            or block is None
            or block.get("name") != call.name
        ):
            raise ApplicationError(
                f"Session {step.session_id} did not pause at tool call {call.id} "
                f"({call.name}), so the call cannot run.",
                non_retryable=True,
            )
        store = InMemorySessionStore()
        await store.append(key, _step_context(seed, span[0]))  # type: ignore[arg-type]
        hook_dir = _hook_folder()
        inp = SegmentInput(
            session_id=step.session_id,
            prompt=None,
            tools=step.tools,
            builtin_tools=step.builtin_tools,
            checkpoint=step.checkpoint,
        )
        violations: list[str] = []
        recorded: RecordedCall | None = None
        try:
            recorded = self._stand_in.serve(block)
            options = self._engine_options(
                inp,
                {},
                step.session_id,
                True,
                store,
                None,
                hook_dir,
                self._durable_server(step.tools, []),
                violations,
            )
            options["max_turns"] = STEP_MAX_TURNS  # the turn ends after the call
            # No permission mode or callback approves a tool here: the hook's "allow"
            # lets the call run. (A rule in a settings file could, if the hook gave
            # no answer: so a failed step does not take that for "not run".)
            options["allowed_tools"] = []
            options["permission_mode"] = DEFAULT_PERMISSION_MODE
            options.pop("can_use_tool", None)
            options.pop("permission_prompt_tool_name", None)
            options["env"] = self._step_env(
                options["env"], hook_dir, call, recorded.key
            )
        except BaseException:
            if recorded is not None:
                self._stand_in.done(recorded)
            _remove_hook_folder(hook_dir)
            raise
        result: ToolResultBlock | None = None
        saved: str | None = None
        failure: BaseException | None = None
        denials: dict[str, str] = {}
        lock: int | None = None
        others: list[str] = []  # calls the engine made besides the step's (denied)
        refused: set[str] = set()  # calls Claude Code's own permission check refused
        bounded = False  # the turn ended at max_turns=1, right after the call
        stopper = (
            asyncio.ensure_future(_stop_hooks_when_cancelled(hook_dir))
            if activity.in_activity()
            else None
        )
        job = _JobWatch(hook_dir, required=self._cleanup_required)
        joined = False  # the engine's first message came (``_JobWatch.joined``)
        completed = False  # the engine's run ended by itself, not by a failure
        try:
            lock = _hold_worker_lock(hook_dir, required=self._cleanup_required)
            engine = cast(  # query() is typed as an iterator; it is a generator
                "AsyncGenerator[Any, None]",
                query(prompt="", options=ClaudeAgentOptions(**options)),
            )
            async with contextlib.aclosing(engine) as messages:
                async for message in messages:
                    if not joined:
                        joined = True
                        job.joined()
                    if isinstance(message, AssistantMessage):
                        others += [
                            b.id
                            for b in message.content
                            if isinstance(b, ToolUseBlock) and b.id != call.id
                        ]
                        continue
                    if isinstance(message, ResultMessage):
                        bounded = message.subtype == "error_max_turns"
                        refused |= _refused_ids(message.permission_denials)
                        continue
                    if not isinstance(message, UserMessage) or isinstance(
                        message.content, str
                    ):
                        continue
                    for content in message.content:
                        if (
                            isinstance(content, ToolResultBlock)
                            and content.tool_use_id == call.id
                        ):
                            result = content
                            # The engine's temporary folder still exists: read the
                            # end of an output it saved to a file before it goes.
                            saved = _saved_output_tail(
                                content.content, key["project_key"], step.session_id
                            )
            completed = True
            if not joined:
                joined = True
                job.joined()  # the engine ended without a message
        except asyncio.CancelledError as err:
            if result is None or not _cancelled_by_shutdown():
                raise
            failure = err  # the call ran, then the Worker began to shut down
        except Exception as err:
            if isinstance(err, ResultError) and err.subtype == "error_max_turns":
                bounded = True  # how the turn ends: right after the call
                completed = True
            elif result is None:
                if not joined:
                    joined = True
                    job.joined()  # the engine ended before its first message: why
                raise
            else:
                # The call ran; the engine failed afterwards. Running the step again
                # would run it again.
                failure = err
        finally:
            if stopper is not None:
                stopper.cancel()
            job.stop()
            sent = self._stand_in.done(recorded)
            denials = _hook_denials(hook_dir)
            allowed[0] = _close_call(hook_dir) or (
                # A run that failed with no result: the hook did not let the call
                # run, but if the hook gave no answer, a rule in a settings file (or
                # a managed policy) may have. It did not run only if the engine never
                # got it, or the hook or Claude Code's permission check refused it.
                not completed
                and result is None
                and sent
                and denial_name(call.id) not in denials
                and call.id not in refused
            )
            _release_worker_lock(lock)
            _remove_hook_folder(hook_dir)
        violation = _hook_violation(violations)
        if violation is not None:
            raise ApplicationError(violation, non_retryable=True)
        text = _text(result.content) if result is not None else ""
        if result is None and allowed[0]:
            raise ApplicationError(
                f"Claude Code was let run tool call {call.id} ({call.name}) in its "
                "step, but the step ended without its result."
            )
        if (
            result is None
            or denial_name(call.id) in denials
            or (bool(result.is_error) and _says_only(result.content, STOPPED))
            # The hook failed and could not record its denial: Claude Code refused
            # the call with the hook's reason. Trying again would fail the same way.
            or (call.id in refused and _says_only(result.content, HOOK_FAILED))
        ):
            raise ApplicationError(
                f"Claude Code did not run tool call {call.id} ({call.name}) in its "
                f"step: {text or 'no result'}",
                non_retryable=True,
            )
        if call.id in refused and not allowed[0]:
            # The hook neither allowed nor denied it (it could not start, or failed),
            # so Claude Code's own permission check refused it: it did not run, and a
            # new attempt may run it.
            raise ApplicationError(
                f"Claude Code did not run tool call {call.id} ({call.name}) in its "
                f"step: its hook gave no answer, so its permission check refused it "
                f"({text or 'no result'})."
            )
        # Claude Code's own result entry, from the step's private copy.
        records = [
            e
            for e in cast("list[Any]", await store.load(cast("Any", key)) or [])
            if call.id in _result_ids(e)
        ]
        with recorded.lock:  # a request still in flight could count one more
            asked = recorded.other
        problems = [
            *([f"the engine failed after it ({failure!r})"] if failure else []),
            *([f"it also asked for {', '.join(others)}"] if others else []),
            *([] if bounded else ["its turn did not end right after the call"]),
            *([f"the stand-in model was asked {asked} more time(s)"] if asked else []),
            *(
                []
                if len(records) == 1 or call.name not in RECORDED_TOOLS
                else [f"it wrote {len(records)} result entries for it (not 1)"]
            ),
        ]
        if problems and activity.in_activity():
            activity.logger.warning(
                "Tool call %s (%s) ran, and its result is kept, but %s.",
                call.id,
                call.name,
                "; ".join(problems),
            )
        is_error = bool(result.is_error)
        entry: dict[str, Any] | None = None
        # The call waited for its step in a warm engine: its checkpoint is the call's
        # own entry, with no deferral marker to resume at (see ``_hook_span``).
        waited = span[0] == span[1] + 1
        if call.name in RECORDED_TOOLS or waited:
            # Claude Code's own record, or one of the same shape (see ``_record``),
            # so the next segment never sends an Edit's result as a message.
            own = records[0] if len(records) == 1 else None
            entry = _result_record(
                own
                if own is not None
                else _record(found[0], call.id, ToolOutcome(), seed[: span[0]])
            )
        if isinstance(result.content, list):
            return ToolOutcome(
                blocks=list(result.content), is_error=is_error, entry=entry
            )
        if saved is not None:
            text += (
                "\n\nThe file named above was removed when the step that ran this "
                f"call ended. The output ends with:\n{saved}"
            )
        return ToolOutcome(content=text, is_error=is_error, entry=entry)

    def _step_env(
        self, env: dict[str, str], hook_dir: str, call: DeferredCall, key: str
    ) -> dict[str, str]:
        """The engine's environment in a tool step, and the command's.

        The engine talks to the local stand-in model (``STEP_PROVIDER_ENV``,
        ``ANTHROPIC_BASE_URL``, and ``key``, which the stand-in gave this step), with
        every tool in its request (``STEP_ENV``). Its hook lets only ``call`` run,
        with the input the segment reported. The command gets the Worker's own
        environment back (see ``_command_env``), so a command that calls the
        Anthropic API reaches the Worker's provider as it would in a segment.
        """
        before = {**os.environ, **env}
        hosts: list[str] = []
        for name in ("NO_PROXY", "no_proxy"):
            for host in (before.get(name) or "").split(","):
                if host.strip() and host.strip() not in hosts:
                    hosts.append(host.strip())
        if "127.0.0.1" not in hosts:
            hosts.append("127.0.0.1")
        overrides = {
            **STEP_PROVIDER_ENV,
            **STEP_ENV,
            "ANTHROPIC_API_KEY": key,
            "ANTHROPIC_BASE_URL": self._stand_in.base_url,
            "NO_PROXY": ",".join(hosts),
            "no_proxy": ",".join(hosts),
        }
        return _command_env(
            env,
            hook_dir,
            self._folder(),
            overrides,
            {"TCA_ALLOW_ID": call.id, "TCA_ALLOW_INPUT": input_digest(call.input)},
        )

    def _durable_server(self, tools: list[ToolSpec], ran_inside: list[str]) -> Any:
        """The durable tools as SDK MCP tools.

        The engine never runs them while the hook defers; if it does, the call is
        recorded in ``ran_inside``.
        """

        def make_stub(spec: ToolSpec) -> Any:
            @tool(spec.name, spec.description, spec.input_schema)
            async def stub(args: dict[str, Any]) -> dict[str, Any]:
                del args
                self.stub_calls += 1  # never happens while the hook defers
                ran_inside.append(spec.name)
                return {
                    "content": [
                        {"type": "text", "text": "This tool must run through Temporal."}
                    ],
                    "is_error": True,
                }

            return stub

        return create_sdk_mcp_server(SERVER, tools=[make_stub(t) for t in tools])

    def _local_copy(self, session_id: str) -> tuple[Path, ...]:
        """Where Claude Code writes a new session in its config folder.

        Nothing for an id this runner would never start.
        """
        try:
            uuid.UUID(session_id)  # only ever a session this runner started
        except ValueError:
            return ()
        env = {**self._env, **(self._extra.get("env") or {})}
        config = env.get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR")
        base = Path(config) if config else Path.home() / ".claude"
        folder = (
            Path(unicodedata.normalize("NFC", str(base)))
            / "projects"
            / project_key_for_directory(self._folder())
        )
        return folder / f"{session_id}.jsonl", folder / session_id

    async def _run_engine(
        self,
        inp: SegmentInput,
        injected: dict[str, ToolOutcome],
        session_id: str,
        resume: bool,
        store: Any,
        *,
        guard: str | None = None,
        committed: list[dict[str, Any]] | None = None,
        delivered: set[str] | None = None,
        warm: _WarmEngine | None = None,
        local_copy: tuple[Path, ...] = (),
        continuing: bool = False,
        resumed_at: str | None = None,
    ) -> SegmentOutput:
        """Run the engine once, on the session in ``store``.

        Args:
            inp: The segment input.
            injected: Tool results to deliver.
            session_id: The session to start or resume.
            resume: Whether to resume it.
            store: The session store the SDK reads and writes.
            guard: Resuming requires the session to end at this checkpoint.
            committed: The conversation the Workflow holds, when it holds one: the
                output then says what changed in it.
            delivered: Calls whose results the session already holds (see
                ``_deliver`` and ``_place``); the others go in the message that
                resumes it.
            warm: An engine still running since it paused at the call whose result
                is in ``injected`` (see ``warm_engines``): the results go to it as
                its next message, instead of a new engine resuming the session.
            local_copy: The engine's own copy of the new session it starts, to
                remove once the engine ended (see ``_forget_local_copy``).
            continuing: The session ends with a tool step's result entry (see
                ``_place``). With no prompt, Claude Code continues the turn by
                itself (``CONTINUE_ENV``), and no message is sent.
            resumed_at: The session's last entry before this run, when it is not
                the checkpoint (a run that added nothing after it did not reach the
                store).

        Raises:
            _SessionMoved: If the session does not end at ``guard``.
            RuntimeError: If Claude Code did not continue the turn by itself
                (``CONTINUE_START_SECONDS``); Temporal runs the segment again.
        """
        # No message at all: Claude Code starts the turn by itself (see ``_place``).
        waits = (
            continuing
            and inp.prompt is None
            and warm is None
            and all(k in (delivered or ()) for k in injected)
        )
        options: dict[str, Any] = {}
        # Warm tool calls: in a new engine that may stay warm, a call of a Claude Code
        # tool that runs as an Activity can wait for its tool step (``_keeps_waiting``).
        may_wait = warm is None and self._may_wait(inp) and not local_copy
        wait_seconds = self._warm_seconds + WAIT_MARGIN_SECONDS
        hook_dir = (
            warm.hook_dir
            if warm is not None
            else _hook_folder(wait_seconds if may_wait else None)
        )
        inbox: _Inbox | None = warm.inbox if warm is not None else None
        view: _GuardedStore | None = warm.view if warm is not None else None
        waiting: _WaitingCall | None = None  # the call that waits for its tool step
        waiting_cost = 0.0  # the engine's running cost then
        waiting_entry: dict[str, Any] = {}  # the call's own entry, the checkpoint
        calls_of_message: dict[
            str, list[str]
        ] = {}  # the main agent's calls, by message
        tasks = False  # a task (an agent, a background command) started in this run
        # Durable tools the engine ran itself (must stay empty), and decisions hooks
        # in extra_options tried to make.
        ran_inside: list[str] = warm.ran_inside if warm is not None else []
        violations: list[str] = warm.violations if warm is not None else []
        result: ResultMessage | None = None
        # A warm engine reported its version when it started.
        engine_version = warm.version if warm is not None else "(unknown version)"
        paused_by_hook: str | None = None
        last_assistant: str | None = None
        store_error: str | None = None
        stopped_by_hook = False
        denials: dict[str, str] = {}
        lock: int | None = warm.lock if warm is not None else None
        # The engine's client and input when it may stay warm (see warm_engines).
        client: Any = warm.client if warm is not None else None
        feed: _Feed | None = warm.feed if warm is not None else None
        parked = False
        stopper: asyncio.Future[None] | None = None
        job: _JobWatch | None = None  # a new engine joins the Worker's job (Windows)
        joined = False
        watch: _StartWatch | None = None  # the engine must start the turn (``waits``)
        try:
            if warm is None:
                options = self._engine_options(
                    inp,
                    injected,
                    session_id,
                    resume,
                    store,
                    guard,
                    hook_dir,
                    self._durable_server(inp.tools, ran_inside),
                    violations,
                )
                # A command run inside the segment sees none of the plugin's variables.
                options["env"] = _command_env(
                    options["env"],
                    hook_dir,
                    self._folder(),
                    CONTINUE_ENV if waits else NEW_TURN_ENV,
                    {"TCA_WAIT_SECONDS": str(wait_seconds)} if may_wait else {},
                )
                view = cast("_GuardedStore", options["session_store"])
            else:
                ran_inside.clear()
                violations.clear()
                _next_hook_turn(hook_dir, injected)
            prompt: Any
            in_message = {
                k: v for k, v in injected.items() if k not in (delivered or ())
            }
            if not resume:
                prompt = inp.prompt or ""
            elif in_message:
                prompt = self._user_message(session_id, in_message, inp.prompt)
            else:
                prompt = inp.prompt  # a new task on the session
            # When the Activity is cancelled or times out, deny every later tool call
            # while the SDK shuts the engine down.
            if activity.in_activity():
                stopper = asyncio.ensure_future(_stop_hooks_when_cancelled(hook_dir))
            try:
                messages: AsyncGenerator[Any, None]
                if warm is not None:  # a warm engine: its next message
                    assert feed is not None
                    feed.put(await _as_messages(prompt))
                    if warm.state == "turn_waits":
                        # The call ran here in its tool step, and the turn waits right
                        # after it: let it go on.
                        Path(hook_dir, TURN_GO).touch()
                    messages = _turn_messages(
                        client,
                        feed,
                        False,
                        stay=True,
                        answered=set(injected),
                        source=(
                            _inbox_messages(warm.inbox, hook_dir)
                            if warm.inbox is not None
                            else None
                        ),
                    )
                else:
                    job = _JobWatch(hook_dir, required=self._cleanup_required)
                    lock = _hold_worker_lock(hook_dir, required=self._cleanup_required)
                    first = await _as_messages(prompt)
                    opened: _Feed
                    if may_wait:
                        feed = opened = _Feed(first)
                        client = ClaudeSDKClient(ClaudeAgentOptions(**options))
                        inbox = _Inbox()
                        inbox.reader = asyncio.ensure_future(
                            _read_engine(client, feed, inbox)
                        )
                        messages = _turn_messages(
                            client,
                            feed,
                            resume,
                            stay=True,
                            answered=set(injected),
                            source=_inbox_messages(inbox, hook_dir),
                        )
                    elif self._may_park(inp) and not local_copy:
                        feed = opened = _Feed(first)
                        client = ClaudeSDKClient(ClaudeAgentOptions(**options))
                        messages = _connected(client, feed, resume, set(injected))
                    else:
                        opened = _Feed(first)
                        messages = _engine_messages(options, opened, resume)
                    if waits:
                        watch = _StartWatch(opened, CONTINUE_START_SECONDS)
                async with contextlib.aclosing(messages):
                    async for message in messages:
                        if isinstance(message, _WaitingCall):
                            # From now on the turn stops after the call unless a
                            # tool step lets it run here (``after_calls``), also
                            # if this segment is cancelled or fails first. A go
                            # this segment gave an earlier call's turn is spent:
                            # left there, it would let this turn go on too.
                            with contextlib.suppress(OSError):
                                os.remove(os.path.join(hook_dir, TURN_GO))
                            Path(hook_dir, TURN_GATE).touch()
                            checked = await self._keeps_waiting(
                                message, hook_dir, client, view, calls_of_message, tasks
                            )
                            if checked is not None:
                                waiting = message
                                waiting_cost, waiting_entry = checked
                                break
                            # As without waiting: the turn ends at the deferred call.
                            with contextlib.suppress(OSError):
                                os.remove(os.path.join(hook_dir, TURN_GATE))
                            _decide(hook_dir, message.id, "defer")
                            continue
                        if (
                            isinstance(message, SystemMessage)
                            and message.subtype == "task_started"
                        ):
                            tasks = True
                        if (
                            isinstance(message, AssistantMessage)
                            and message.parent_tool_use_id is None
                        ):
                            # Claude Code sends each block of a message on its own.
                            calls_of_message.setdefault(
                                message.message_id or "", []
                            ).extend(
                                b.id
                                for b in message.content
                                if isinstance(b, ToolUseBlock)
                            )
                        if watch is not None:
                            if _turn_began(message):
                                watch.heard()
                            else:
                                watch.waiting()
                        if job is not None and not joined:
                            joined = True
                            job.joined()
                        if isinstance(message, MirrorErrorMessage):
                            store_error = message.error or "unknown error"
                        elif (
                            isinstance(message, SystemMessage)
                            and message.subtype == "init"
                        ):
                            version = message.data.get("claude_code_version")
                            engine_version = str(version or engine_version)
                        elif (
                            isinstance(message, AssistantMessage)
                            and message.parent_tool_use_id is None  # not a subagent's
                        ):
                            last_assistant = message.uuid or last_assistant
                            for block in message.content:
                                if isinstance(block, TextBlock) and block.text.strip():
                                    emit({"type": "text", "text": block.text})
                        elif isinstance(message, UserMessage):
                            stopped_by_hook = stopped_by_hook or _hook_said_stopped(
                                message
                            )
                        elif isinstance(message, ResultMessage):
                            result = message
                if job is not None and not joined:
                    joined = True
                    job.joined()  # the engine ended without a message
                marker = Path(hook_dir) / "paused_call"  # written when the hook defers
                if marker.exists():
                    paused_by_hook = marker.read_text(encoding="utf-8").strip() or None
                denials = _hook_denials(hook_dir)
            except ResultError as err:
                if watch is not None and watch.fired:
                    version = await self._engine_version() or engine_version
                    raise _not_continued(version) from err
                final = err.subtype in FINAL_RESULT_ERRORS or _final_api_error(err)
                if not final:
                    raise  # other engine errors: let Temporal retry the segment
                # Retrying will not help.
                return SegmentOutput(
                    session_id=session_id, is_error=True, error=str(err)
                )
            except RuntimeError as err:
                if _moved(err):
                    raise _SessionMoved(str(err)) from err
                if job is not None and not joined:
                    job.joined()  # the engine ended before its first message: why
                if watch is not None and watch.fired:
                    version = await self._engine_version() or engine_version
                    raise _not_continued(version) from err
                raise
            except Exception as err:
                if job is not None and not joined:
                    job.joined()
                if (
                    watch is not None
                    and watch.fired
                    and not _moved(err)  # a copy can still continue
                ):
                    version = await self._engine_version() or engine_version
                    raise _not_continued(version) from err
                raise
            finally:
                if watch is not None:
                    watch.heard()  # no later timer
                if stopper is not None:
                    stopper.cancel()
                if job is not None:
                    job.stop()
            if watch is not None and watch.fired:
                # Even with a result: its input had ended, so neither hooks nor
                # in-process tools could answer during the turn.
                raise _not_continued(await self._engine_version() or engine_version)
            if waiting is not None:
                assert client is not None and feed is not None and view is not None
                out = await self._waiting_output(
                    session_id,
                    store,
                    committed,
                    waiting_entry,
                    waiting,
                    max(0.0, waiting_cost - (warm.cost if warm is not None else 0.0)),
                )
                if out.is_error:
                    return out  # the engine ends: its hook denies the call
                assert out.deferred is not None and out.checkpoint is not None
                self._park(
                    _WarmEngine(
                        client=client,
                        feed=feed,
                        session_id=out.session_id,
                        checkpoint=out.checkpoint,
                        paused=waiting.id,
                        shape=self._shape(inp),
                        hook_dir=hook_dir,
                        lock=lock,
                        store=store,
                        cost=waiting_cost,
                        version=engine_version,
                        ran_inside=ran_inside,
                        violations=violations,
                        buffer=(
                            warm.buffer
                            if warm is not None
                            else int(options["max_buffer_size"])
                        ),
                        inbox=inbox,
                        state="call_waits",
                        call_name=out.deferred.name,
                        digest=waiting.digest,
                        view=view,
                        owner=waiting_entry,
                    )
                )
                parked = True
                return out
            if warm is not None and result is None:
                # The engine ended after it was taken (it was alive then): a new
                # engine can do the step.
                raise RuntimeError(
                    "The warm engine ended before its turn's result. Retrying."
                )
            out = await self._segment_output(
                inp,
                injected,
                session_id,
                resume,
                store,
                guard,
                committed,
                warm,
                ran_inside,
                violations,
                hook_dir,
                result,
                engine_version,
                paused_by_hook,
                last_assistant,
                store_error,
                stopped_by_hook,
                denials,
                resumed_at,
            )
            if feed is not None and self._parks(
                out, hook_dir, client, local_copy, denials
            ):
                assert out.deferred is not None and out.checkpoint is not None
                assert result is not None
                self._park(
                    _WarmEngine(
                        client=client,
                        feed=feed,
                        session_id=out.session_id,
                        checkpoint=out.checkpoint,
                        paused=out.deferred.id,
                        shape=self._shape(inp),
                        hook_dir=hook_dir,
                        lock=lock,
                        store=store,
                        cost=float(result.total_cost_usd or 0.0),
                        version=engine_version,
                        ran_inside=ran_inside,
                        violations=violations,
                        buffer=(
                            warm.buffer
                            if warm is not None
                            else int(options["max_buffer_size"])
                        ),
                        inbox=inbox,
                        view=view,
                    )
                )
                parked = True
            return out
        finally:
            if not parked:
                if inbox is not None:
                    # A hook may still wait (a call, or the turn after it): it denies
                    # the call or stops the turn now, instead of when the engine is
                    # made to exit.
                    for name in (TURN_GATE, "stop"):
                        with contextlib.suppress(OSError):
                            Path(hook_dir, name).touch()
                try:
                    if client is not None:
                        await _disconnect(client)
                finally:
                    _release_worker_lock(lock)
                    _remove_hook_folder(hook_dir)  # the hook denies from now on
                    _forget_local_copy(local_copy)
                    if inbox is not None:
                        inbox.stop()

    def _parks(
        self,
        out: SegmentOutput,
        hook_dir: str,
        client: Any,
        local_copy: tuple[Path, ...],
        denials: dict[str, str],
    ) -> bool:
        """Whether an engine that just ran stays warm: it paused at one durable call,
        cleanly.

        Not at a Claude Code tool that runs as its own Activity (a tool step): the
        result of an Edit or a Write goes into the conversation as Claude Code's own
        record of the call (see ``_place``), which only a new engine reads. So every
        tool step's result goes to a new engine, whatever the tool.

        Not an engine that started a conversation the Workflow holds: it writes the
        conversation to its own config folder (``_forget_local_copy``), and a Worker
        that dies while it waits would leave it there. A resumed engine runs in a
        temporary folder, as each segment's engine does today.

        Not after the hook denied a call in the same turn: the denial comes after the
        pause in the session, so the next segment needs a new engine anyway.
        """
        return (
            not local_copy
            and not denials
            and out.deferred is not None
            and out.deferred.kind == "durable"
            and not out.is_error
            and not out.siblings
            and out.checkpoint is not None
            and not os.path.exists(os.path.join(hook_dir, "stop"))
            and _engine_alive(client)
        )

    # ---- Warm tool calls: a call waits in its engine for its tool step ----

    async def _keeps_waiting(
        self,
        call: _WaitingCall,
        hook_dir: str,
        client: Any,
        view: _GuardedStore | None,
        calls_of_message: dict[str, list[str]],
        tasks: bool,
    ) -> tuple[float, dict[str, Any]] | None:
        """Whether the segment pauses with ``call`` waiting in its hook for its tool step.

        Only when this run is not stopping and its hook denied nothing; when no task or
        agent runs (an engine that ends would lose it); and when the call is the only
        tool call of its message, once that message is complete in the session store
        and the store ends with the call's entry (the pause's checkpoint). Claude Code
        starts a call as soon as its block arrives, while the rest of the message may
        still come, and writes the message's entries when it is complete, every 100 ms
        in the middle of a turn: ``ENTRY_SECONDS`` is ample. Only one call of a
        message may pause the run, so a message with more calls pauses as without
        waiting.

        Returns:
            The engine's running cost then (``_engine_cost``) and the call's entry, or
            None: the call is deferred, as without waiting.
        """
        stop = os.path.join(hook_dir, "stop")
        if (
            view is None
            or os.path.exists(stop)
            or _hook_denials(hook_dir)
            or tasks
            or _agents_running(client)
        ):
            return None
        calls = next((c for c in calls_of_message.values() if call.id in c), None)
        if calls is not None and calls != [call.id]:
            return None
        deadline = time.monotonic() + ENTRY_SECONDS
        while True:
            await _mirror_flush(client)
            message_id = next((m for m, c in view.calls.items() if call.id in c), None)
            stored = view.calls.get(message_id) if message_id is not None else None
            if stored is not None and stored != [call.id]:
                return None  # its message holds more calls than this one
            last = view.last
            if (
                stored is not None
                and last is not None
                and _waiting_call_of(last) == call.id
                and last["message"].get("stop_reason") == "tool_use"  # complete
            ):
                break  # the session ends with the call's entry
            if (
                stored is not None
                and isinstance(last, dict)
                and last.get("type") == "assistant"
                and str((last.get("message") or {}).get("id")) == message_id
                and _waiting_call_of(last) != call.id
            ):
                return None  # its message goes on after the call (a text block)
            if time.monotonic() > deadline or os.path.exists(stop):
                return None
            await asyncio.sleep(ENTRY_POLL_SECONDS)
        if calls_of_message.get(str(last["message"].get("id")), [call.id]) != [call.id]:
            return None
        total = await _engine_cost(client)
        if total is None or _hook_denials(hook_dir) or os.path.exists(stop):
            return None
        return total, last

    async def _waiting_output(
        self,
        session_id: str,
        store: Any,
        committed: list[dict[str, Any]] | None,
        entry: dict[str, Any],
        call: _WaitingCall,
        cost: float,
    ) -> SegmentOutput:
        """The pause at a call that waits for its tool step.

        The checkpoint is the call's own entry, which ``_keeps_waiting`` saw last in
        the session. The call's input is the one its hook saw, as a paused call
        reports it (with a relative ``file_path`` made absolute): its tool step must
        bring exactly that input.
        """
        block = entry["message"]["content"][0]
        out = SegmentOutput(
            session_id=session_id,
            deferred=DeferredCall(
                id=call.id,
                name=str(block.get("name")).removeprefix(PREFIX),
                input=dict(call.input),
                kind="engine",
            ),
            checkpoint=entry["uuid"],
            cost_usd=cost,
        )
        out.external_storage = external_storage_on()
        if committed is not None:
            key = {
                "project_key": project_key_for_directory(self._folder()),
                "session_id": session_id,
            }
            entries = cast("list[dict[str, Any]]", await store.load(key) or [])
            keep, covered = _kept(committed, entries)
            out.transcript_keep = keep
            out.transcript_add = _without_cost_state(entries[covered:])
            problem = too_large(out)
            if problem is not None:
                return SegmentOutput(
                    session_id=session_id, is_error=True, error=problem, cost_usd=cost
                )
        return out

    def _take_waiting(self, step: ToolStepInput) -> _WarmEngine | None:
        """The warm engine where this step's call waits, taken from the pool, or None.

        Only for the call it paused at, with the input its hook saw, the same tools, and
        the engine still running. Another warm engine at that checkpoint ends: the
        step runs in a new engine, as without warm engines.
        """
        key = (step.session_id, step.checkpoint)
        warm = self._warm.get(key)
        if warm is None or warm.state != "call_waits":
            return None
        del self._warm[key]
        if warm.timer is not None:
            warm.timer.cancel()
        shape = json.loads(warm.shape)
        if (
            warm.paused == step.call.id
            and warm.call_name == step.call.name
            and warm.digest == input_digest(step.call.input)
            and shape[2]
            == [[t.name, t.description, t.input_schema] for t in step.tools]
            and shape[3] == list(step.builtin_tools)
            and _engine_alive(warm.client)
        ):
            return warm
        self._end_warm(warm)
        return None

    async def _step_in_warm(
        self, step: ToolStepInput, warm: _WarmEngine, allowed: list[bool]
    ) -> ToolOutcome:
        """Run the call in the warm engine where it waits; the turn then waits after it.

        The engine runs the call as it would have in its segment, and writes its own
        result entry. Its ``PostToolBatch`` hook then holds the turn, so Claude is asked
        nothing until the next segment takes the engine; if that does not happen here
        within ``warm_seconds``, the turn stops and the engine ends.
        """
        call = step.call
        hook_dir = warm.hook_dir
        assert warm.inbox is not None
        result: ToolResultBlock | None = None
        stopper = (
            asyncio.ensure_future(_stop_hooks_when_cancelled(hook_dir))
            if activity.in_activity()
            else None
        )
        try:
            with contextlib.suppress(OSError):
                os.remove(os.path.join(hook_dir, TURN_GO))
            Path(hook_dir, TURN_GATE).touch()  # before the call can run
            if warm.view is not None:
                warm.view.ran_here.add(call.id)  # its record goes in without metadata
            _decide(hook_dir, call.id, f"allow {warm.digest}")
            async for message in _inbox_messages(warm.inbox, None):
                if isinstance(message, UserMessage) and not isinstance(
                    message.content, str
                ):
                    for content in message.content:
                        if (
                            isinstance(content, ToolResultBlock)
                            and content.tool_use_id == call.id
                        ):
                            result = content
                    if result is not None:
                        break
                elif isinstance(message, ResultMessage):
                    break  # the turn ended without the call's result
        except BaseException:
            allowed[0] = _close_call(hook_dir)
            self._end_warm(warm)
            raise
        finally:
            if stopper is not None:
                stopper.cancel()
        allowed[0] = _close_call(hook_dir)
        denials = _hook_denials(hook_dir)
        if result is None and not allowed[0] and denial_name(call.id) not in denials:
            # The engine ended before its hook let the call run, and now it never
            # can (``_close_call``): run it in a new engine, as without warm engines.
            self._end_warm(warm)
            return await self._step_in_new_engine(step, allowed)
        if result is None or denial_name(call.id) in denials:
            self._end_warm(warm)
            if result is None and allowed[0]:
                raise ApplicationError(
                    f"Claude Code was let run tool call {call.id} ({call.name}) in its "
                    "step, but the step ended without its result."
                )
            text = _text(result.content) if result is not None else ""
            raise ApplicationError(
                f"Claude Code did not run tool call {call.id} ({call.name}) in its "
                f"step: {text or 'no result'}",
                non_retryable=True,
            )
        # A record of the result, for any tool, as a step in a new engine returns for
        # a call that waited (there is no deferral marker to resume at): Claude Code's
        # own if it already reached the store (it writes every 100 ms), else one of
        # the same shape (``_record``).
        own = None
        if warm.view is not None:
            own = warm.view.results.get(call.id)
            warm.view.results.clear()  # only this call's was needed
        entry: dict[str, Any] | None = None
        if own is not None:
            entry = _result_record(own)
        elif warm.owner is not None:
            entry = _result_record(_record(warm.owner, call.id, ToolOutcome(), []))
        is_error = bool(result.is_error)
        if isinstance(result.content, list):
            outcome = ToolOutcome(
                blocks=list(result.content), is_error=is_error, entry=entry
            )
        else:
            text = _text(result.content)
            saved = _saved_output_tail(
                result.content,
                project_key_for_directory(self._folder()),
                warm.session_id,
            )
            if saved is not None:
                # As in a step of its own: the next segment may run elsewhere.
                text += (
                    "\n\nThe file named above was removed when the step that ran this "
                    f"call ended. The output ends with:\n{saved}"
                )
            outcome = ToolOutcome(content=text, is_error=is_error, entry=entry)
        warm.state = "turn_waits"
        warm.outcome = outcome
        self._park(warm)
        return outcome

    async def _segment_output(
        self,
        inp: SegmentInput,
        injected: dict[str, ToolOutcome],
        session_id: str,
        resume: bool,
        store: Any,
        guard: str | None,
        committed: list[dict[str, Any]] | None,
        warm: _WarmEngine | None,
        ran_inside: list[str],
        violations: list[str],
        hook_dir: str,
        result: ResultMessage | None,
        engine_version: str,
        paused_by_hook: str | None,
        last_assistant: str | None,
        store_error: str | None,
        stopped_by_hook: bool,
        denials: dict[str, str],
        ended_at: str | None = None,
    ) -> SegmentOutput:
        """What one engine run committed: the pause or the answer, or why it failed.

        ``ended_at`` is the session's last entry before the run, when that is not the
        checkpoint (see ``_run_engine``'s ``resumed_at``).

        Raises:
            RuntimeError: If the turn did not reach the session store (retried).
        """
        violation = _hook_violation(violations)
        if violation is not None:
            return SegmentOutput(session_id=session_id, is_error=True, error=violation)
        if stopped_by_hook or REASON_KEYS[STOPPED] in denials.values():
            # Not cancelled, yet the hook denied a call as stopped: it could not see
            # this step's folder. Fail closed rather than commit what Claude said
            # without the tool.
            return SegmentOutput(
                session_id=session_id,
                is_error=True,
                error=(
                    "The pause hook could not see this step's folder "
                    f"({hook_dir}), so it denied Claude's tool calls. Run Claude Code "
                    "where it sees the Worker's temporary folder (for example, not in "
                    "a sandbox with its own /tmp). No tool ran."
                ),
            )
        if store_error is not None:
            raise RuntimeError(
                f"The session store did not save part of this segment's transcript "
                f"({store_error}). Retrying."
            )
        if _too_old(engine_version):
            # The engine reports its version at start. If the check above could not
            # read it, refuse here: the Workflow then never runs the paused tool.
            return _too_old_output(session_id, engine_version)
        if result is None:
            return SegmentOutput(
                session_id=session_id,
                is_error=True,
                error="Claude returned no result message",
            )
        sid = result.session_id or session_id
        cost = float(result.total_cost_usd or 0.0)
        if warm is not None and cost >= warm.cost:
            cost -= warm.cost  # a warm engine reports its running total
        deferred = result.deferred_tool_use
        broken = self._pause_contract_problem(
            ran_inside,
            paused_by_hook,
            deferred,
            set(injected),
            engine_version,
            result.stop_reason,
        )
        if broken:
            return SegmentOutput(
                session_id=sid, is_error=True, error=broken, cost_usd=cost
            )
        if deferred is None and result.is_error:
            return SegmentOutput(
                session_id=sid,
                is_error=True,
                error=str(result.errors or result.subtype),
                cost_usd=cost,
            )
        # A run with nothing new after where it resumed did not reach the store.
        resumed_at = ended_at or (
            guard if committed is None else (inp.checkpoint if resume else None)
        )
        checkpoint, entries = await self._checkpoint(
            store,
            sid,
            last_assistant,
            deferred.id if deferred is not None else None,
            resumed_at,
        )
        if deferred is not None:
            out = SegmentOutput(
                session_id=sid,
                deferred=DeferredCall(
                    id=deferred.id,
                    name=deferred.name.removeprefix(PREFIX),
                    input=dict(deferred.input),
                    # A Claude Code tool that runs as its own Activity (a tool step).
                    kind="durable" if deferred.name.startswith(PREFIX) else "engine",
                ),
                checkpoint=checkpoint,
                cost_usd=cost,
                siblings=_siblings(entries, deferred.id, denials),
            )
        else:
            out = SegmentOutput(
                session_id=sid,
                result=result.result,
                checkpoint=checkpoint,
                cost_usd=cost,
            )
        out.external_storage = external_storage_on()
        if committed is not None:
            # The engine only appends to what it resumed from, which is the committed
            # conversation, cut after the checkpoint when that is the deferral marker.
            # Saved cost totals never join the conversation (``_kept``).
            keep, covered = _kept(committed, entries)
            out.transcript_keep = keep
            out.transcript_add = _without_cost_state(entries[covered:])
            problem = too_large(out)
            if problem is not None:
                return SegmentOutput(
                    session_id=sid, is_error=True, error=problem, cost_usd=cost
                )
        return out

    def _engine_options(
        self,
        inp: SegmentInput,
        injected: dict[str, ToolOutcome],
        session_id: str,
        resume: bool,
        store: Any,
        guard: str | None,
        hook_dir: str,
        durable_server: Any,
        violations: list[str],
    ) -> dict[str, Any]:
        """The ``ClaudeAgentOptions`` fields of one engine run, ``extra_options`` merged in."""
        extra = dict(self._extra)
        default_prompt = extra.pop("system_prompt", None)
        hints = [
            *([ONE_TOOL_HINT] if self._one_tool else []),
            *([SHELL_HINT] if set(_SHELL_TOOLS) & set(inp.builtin_tools) else []),
        ]
        hint = "\n\n".join(hints) or None
        activities = list(inp.tool_activities)

        def decides_alone(name: str) -> bool:  # the plugin decides on these calls
            return name.startswith(PREFIX) or _runs_as_activity(name, activities)

        if "hooks" in extra:
            extra["hooks"] = _guard_hooks(extra["hooks"], decides_alone, violations)
        system_prompt: Any
        if inp.system_prompt is None and isinstance(default_prompt, dict):
            preset = cast("dict[str, Any]", default_prompt)  # e.g. Claude Code's own
            append = "\n\n".join(p for p in (preset.get("append"), hint) if p)
            system_prompt = {**preset, "append": append} if append else dict(preset)
        else:
            base = (
                inp.system_prompt if inp.system_prompt is not None else default_prompt
            )
            system_prompt = "\n\n".join(p for p in (base, hint) if p) or None
        durable_names = [PREFIX + t.name for t in inp.tools]
        payload = sum(len(_text(_result_content(o))) for o in injected.values())
        payload += len(inp.prompt or "")
        options: dict[str, Any] = {
            "system_prompt": system_prompt,
            "model": inp.model or self._model,
            "tools": list(inp.builtin_tools),  # no built-in tools unless asked for
            "max_turns": inp.max_turns,
            "mcp_servers": {
                **(extra.pop("mcp_servers", None) or {}),
                SERVER: durable_server,
            },
            "strict_mcp_config": True,
            "setting_sources": [],
            # Tools that run as Activities are not pre-approved: the hook defers or
            # allows them call by call, so if it ever gives no answer (it failed, or
            # an engine ignores defer), Claude Code refuses a command that changes
            # something instead of running it inside the segment.
            "allowed_tools": list(
                dict.fromkeys(
                    [
                        *durable_names,
                        *(
                            t
                            for t in [
                                *inp.builtin_tools,
                                *(extra.pop("allowed_tools", None) or []),
                            ]
                            if not _runs_as_activity(t, activities)
                        ),
                    ]
                )
            ),
            "settings": str(Path(hook_dir) / "settings.json"),
            "session_store": _guarded(store, session_id, guard),
            "cwd": self._folder(),
            "env": {
                **self._env,
                **(extra.pop("env", None) or {}),
                **ENGINE_ENV,
                "TCA_HOOK_DIR": hook_dir,
                "TCA_ANSWERED_IDS": " ".join(injected),
                "TCA_TOOL_ACTIVITIES": "\n".join(inp.tool_activities),
                **({WORKER_PID: str(os.getpid())} if self._launch else {}),
            },
            # Through the launcher, the engine ends with this Worker process.
            "cli_path": self._launch or self._cli_path,
            "permission_mode": DEFAULT_PERMISSION_MODE,  # extra_options may change it
            # The engine echoes each delivered result as one line: room for any result.
            "max_buffer_size": max(
                MIN_BUFFER_BYTES,
                8 * payload,
                int(extra.pop("max_buffer_size", None) or 0),
            ),
        }
        if self._max_budget is not None:
            options["max_budget_usd"] = self._max_budget  # safety cap per segment
        if resume:
            options["resume"] = session_id
        else:
            options["session_id"] = session_id
        options.update(
            extra
        )  # only options the plugin leaves alone (checked in __init__)
        return options

    @staticmethod
    def _pause_contract_problem(
        ran_inside: list[str],
        paused_by_hook: str | None,
        deferred: Any,
        answered: set[str],
        version: str,
        stop_reason: Any,
    ) -> str | None:
        """Fail closed if the engine did not honor the pause.

        Returns:
            An error message, or None if the engine behaved.
        """
        if ran_inside:
            return (
                f"Claude Code {version} ran durable tool(s) "
                f"{', '.join(sorted(set(ran_inside)))} inside the engine instead of "
                f"pausing. {PAUSE_CONTRACT}"
            )
        if deferred is not None and deferred.id in answered:
            return (
                f"Claude Code {version} paused again at tool call {deferred.id}, whose "
                f"result was just delivered. {PAUSE_CONTRACT}"
            )
        if paused_by_hook and (deferred is None or deferred.id != paused_by_hook):
            got = (
                f"paused at {deferred.id}" if deferred is not None else "did not pause"
            )
            return (
                f"The pause hook deferred tool call {paused_by_hook}, but Claude Code "
                f"{version} {got} (stop_reason={stop_reason}). {PAUSE_CONTRACT}"
            )
        return None

    @staticmethod
    async def _user_message(
        session_id: str, injected: dict[str, ToolOutcome], prompt: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        """One user message: the tool results first, then the prompt of a new task."""
        blocks = [_result_block(tid, o) for tid, o in injected.items()]
        if prompt:
            blocks.append({"type": "text", "text": prompt})
        yield {
            "type": "user",
            "message": {"role": "user", "content": blocks},
            "parent_tool_use_id": None,
            "session_id": session_id,
        }
