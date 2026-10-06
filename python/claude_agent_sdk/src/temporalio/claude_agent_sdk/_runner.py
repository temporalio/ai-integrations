"""The real runner: drives the Claude Agent SDK and its bundled Claude Code engine.

How one segment works:

1. Durable tools are declared to Claude as in-process SDK MCP tools.
2. A settings-file PreToolUse command hook answers "defer" when Claude calls one.
   The run stops and the SDK returns ``ResultMessage.deferred_tool_use``. The
   Workflow runs the tool as a Temporal Activity.
3. The next segment resumes the session and sends the stored result as a normal
   ``tool_result`` message for that ``tool_use_id``. Claude continues and can pause
   again. The hook keeps answering "defer" (never "allow") for durable tools.

Checkpoints make retries clean. After a segment, the runner reads the session back
from the session store and returns where the next segment must continue (the last
transcript entry, or the paused call's deferral marker after parallel calls) as the
segment's checkpoint; the Workflow stores it with the segment's result. Reading it
back also proves the turn reached the store, so a segment only commits what every
Worker can resume. A segment that runs again (a retry after a crash or a timeout,
or the first segment after a failed task) cannot trust what an unfinished attempt
wrote, so it continues in a copy of the session that ends at the checkpoint
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
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import uuid
import warnings
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    MirrorErrorMessage,
    ResultError,
    ResultMessage,
    SessionStore,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    UserMessage,
    create_sdk_mcp_server,
    fork_session_via_store,
    project_key_for_directory,
    tool,
)
from temporalio import activity

from ._defer_hook import STOPPED
from ._events import emit
from ._managed import supervised_query as query
from ._models import DeferredCall, SegmentInput, SegmentOutput, ToolOutcome, ToolSpec

ENV_AUTH = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
"""Logins that survive resumes.

A Claude app login (Keychain OAuth) does not: when the SDK resumes a session from a
session store it copies the login without its refresh token, so once the short-lived
access token expires, resumed segments fail with "OAuth session expired".
"""

SERVER = "durable"
PREFIX = f"mcp__{SERVER}__"
ONE_TOOL_HINT = "Call at most one tool per message, then wait for its result before calling another."
"""With parallel calls the engine keeps only one paused call, so ask for one at a time."""

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

ENGINE_ENV = {"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1"}
"""Set for every engine run. With background tasks (for example a subagent running in
the background), the engine keeps working after it paused at a durable call."""

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
    ]
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
    continues in a copy that ends at it.
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
        elif marker is not None and entry["type"] == "user":
            user_after_marker = True
    return marker if marker is not None and user_after_marker else leaf


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
    """Passes the session store to the SDK, and checks the session when it is resumed.

    Resuming in place is right only while the session still ends at the checkpoint.
    The check rides on the load the SDK does anyway, so it costs nothing extra.
    """

    def __init__(self, inner: Any, session_id: str, checkpoint: str) -> None:
        self._inner = inner
        self._session_id = session_id
        self._checkpoint = checkpoint

    async def append(self, key: Any, entries: Any) -> None:
        await self._inner.append(key, entries)

    async def load(self, key: Any) -> Any:
        entries = await self._inner.load(key)
        if (
            entries
            and key.get("session_id") == self._session_id
            and not key.get("subpath")
            and _last_entry(entries) != self._checkpoint
        ):
            raise _SessionMoved(
                f"Session {self._session_id} continued after checkpoint "
                f"{self._checkpoint}"
            )
        return entries


def _guarded(inner: Any, session_id: str, checkpoint: str) -> Any:
    """``inner`` with the resume check, keeping only the optional methods it has."""
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
    this package would add about a second to every durable tool call.
    """
    hook = Path(__file__).with_name("_defer_hook.py")
    return {"type": "command", "command": sys.executable, "args": [str(hook)]}


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


def _hook_said_stopped(message: UserMessage) -> bool:
    """Whether a tool result in ``message`` is the hook's "step stopped" denial."""
    if isinstance(message.content, str):
        return False
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            text = (
                block.content
                if isinstance(block.content, str)
                else json.dumps(block.content, default=str)
            )
            if STOPPED in text:
                return True
    return False


def _as_outcome(value: Any) -> ToolOutcome:
    if isinstance(value, ToolOutcome):
        return value
    return ToolOutcome(
        content=value.get("content"), is_error=bool(value.get("is_error"))
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


class ClaudeAgentSdkRunner:
    """Runs each segment with the Claude Agent SDK and its bundled Claude Code engine."""

    def __init__(
        self,
        *,
        session_store: Any,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        cli_path: str | None = None,
        extra_options: dict[str, Any] | None = None,
        one_tool_at_a_time: bool = True,
        model: str | None = None,
        max_budget_usd: float | None = None,
    ) -> None:
        """Create the runner.

        Args:
            session_store: A Claude Agent SDK ``SessionStore`` that every Worker can
                reach, so any Worker can resume any session. Each segment's
                checkpoint is read back from it. ``FileSessionStore`` works for one
                machine or a shared disk.
            cwd: Working directory of the engine. The store keys sessions by it, so
                give every Worker the same one.
            env: Extra environment variables for the engine. The engine also
                inherits the Worker's environment.
            cli_path: Path of a Claude Code executable to use instead of the bundled one.
            extra_options: More ``ClaudeAgentOptions`` fields (for example
                ``permission_mode``, ``agents``, ``hooks``, ``setting_sources``,
                ``thinking``). ``env``, ``mcp_servers`` and ``allowed_tools`` are
                merged with the plugin's; ``system_prompt`` (a string or a preset) is
                the default for agents that set none. Options the agent or the plugin
                sets itself, and ``extra_args`` for the same engine flags, are
                refused.
            one_tool_at_a_time: Ask Claude for one tool call per message.
            model: Default model when the agent does not set one.
            max_budget_usd: Cost cap per segment.

        Raises:
            ValueError: If ``extra_options`` sets an option the plugin manages, or
                Claude Code and the SDK would derive different session keys for the
                working directory (for example decomposed Unicode or emoji in its
                path, or a path given in another form than its real one).
        """
        _check_extra_options(extra_options or {})
        directory = cwd or os.getcwd()
        mismatch = _key_mismatch(directory)
        if mismatch is not None:
            raise ValueError(
                f"cwd {directory!r}: Claude Code and the Claude Agent SDK would derive "
                f"different session keys ({mismatch}), so resumed sessions would not "
                "be found. Pass the path exactly as os.path.realpath gives it, or use "
                "a directory with a plain ASCII path."
            )
        self._store = session_store
        self._cwd = cwd
        self._env = env or {}
        self._cli_path = cli_path
        self._extra = extra_options or {}
        self._one_tool = one_tool_at_a_time
        self._model = model
        self._max_budget = max_budget_usd
        self.stub_calls = 0
        """Durable tools the engine ran itself. Stays 0 while the engine honors defer."""
        self._versions: dict[str, str | None] = {}

        def is_set(name: str) -> bool:
            value = self._env.get(name) or os.environ.get(name, "")
            return value.lower() not in ("", "0", "false", "no")

        if not any(is_set(name) for name in ENV_AUTH):
            warnings.warn(
                "temporalio.claude_agent_sdk: no API key, cloud provider, or "
                "CLAUDE_CODE_OAUTH_TOKEN is set. A Claude app login cannot refresh "
                "itself when a session is resumed from a session store, so long-running "
                "agents can fail with 'OAuth session expired'. Set ANTHROPIC_API_KEY, "
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

    async def _start(self, inp: SegmentInput, attempt: int) -> tuple[str, bool]:
        """Where this segment starts: ``(session_id, resume)``.

        A segment that runs again continues in a copy of the session that ends at
        the checkpoint, because an unfinished attempt may have written after it.
        """
        if inp.checkpoint is None:  # nothing committed yet: a new session
            if attempt == 1:
                return inp.session_id, False
            # The failed attempt may have left a partial transcript: start another.
            return str(
                uuid.uuid5(uuid.UUID(inp.session_id), f"attempt-{attempt}")
            ), False
        if attempt == 1 and not inp.fork:
            return inp.session_id, True
        return await self._copy(inp), True

    async def _copy(self, inp: SegmentInput) -> str:
        """A copy of the session that ends at the checkpoint; returns its id."""
        copy = await fork_session_via_store(
            self._store,
            inp.session_id,
            directory=self._cwd,
            up_to_message_id=inp.checkpoint,
        )
        return copy.session_id

    async def _checkpoint(
        self,
        session_id: str,
        assistant_uuid: str | None,
        paused_call: str | None,
        resumed_at: str | None,
    ) -> str:
        """Where the next segment continues, read back from the store (see ``_resume_point``).

        Reading it back also proves that the turn reached the session store.

        Raises:
            RuntimeError: If the store does not have the turn (Temporal retries).
        """
        key = {
            "project_key": project_key_for_directory(self._cwd),
            "session_id": session_id,
        }
        entries = cast("list[dict[str, Any]]", await self._store.load(key) or [])
        stored = assistant_uuid is None or any(
            e.get("uuid") == assistant_uuid for e in entries
        )
        leaf = _resume_point(entries, paused_call)
        if not stored or leaf is None or leaf == resumed_at:  # nothing new stored
            raise RuntimeError(
                f"The session store does not have this segment's turn (session "
                f"{session_id}), so another Worker could not continue it. Retrying."
            )
        return leaf

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """Run one segment until Claude pauses at a durable tool call or finishes.

        Args:
            inp: The segment input.
            attempt: The Activity attempt number, starting at 1.

        Returns:
            The pause or the final answer, with the segment's checkpoint. ``is_error``
            is set when the engine broke the pause contract, or when retrying cannot
            help.

        Raises:
            RuntimeError: If the transcript did not reach the session store (Temporal
                retries the segment).
        """
        reported = await self._engine_version()
        if reported is not None and _too_old(reported):
            return _too_old_output(inp.session_id, reported)  # before the engine starts
        injected = {k: _as_outcome(v) for k, v in inp.injected.items()}
        session_id, resume = await self._start(inp, attempt)
        if resume and not injected and inp.prompt is None:
            return SegmentOutput(
                session_id=session_id,
                is_error=True,
                error="Nothing to send: no tool result and no prompt",
            )
        in_place = resume and session_id == inp.session_id
        try:
            return await self._run_engine(
                inp, injected, session_id, resume, inp.checkpoint if in_place else None
            )
        except _SessionMoved:
            # The session went on after the checkpoint (for example, the Workflow
            # was reset to an earlier point): continue in a copy that ends there.
            return await self._run_engine(inp, injected, await self._copy(inp), True)

    async def _run_engine(
        self,
        inp: SegmentInput,
        injected: dict[str, ToolOutcome],
        session_id: str,
        resume: bool,
        guard: str | None = None,
    ) -> SegmentOutput:
        """Run the engine once. With ``guard``, resuming requires the session to end there.

        Raises:
            _SessionMoved: If the session does not end at ``guard``.
        """
        # Durable tools the engine ran itself (must stay empty).
        ran_inside: list[str] = []

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

        hook_dir = tempfile.mkdtemp(prefix="tca-hook-")
        settings_file = Path(hook_dir) / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {
                                # Every tool: built-in calls after a pause are denied too.
                                "matcher": ".*",
                                "hooks": [_hook_entry()],
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        options = self._engine_options(
            inp,
            injected,
            session_id,
            resume,
            guard,
            hook_dir,
            create_sdk_mcp_server(SERVER, tools=[make_stub(t) for t in inp.tools]),
        )
        prompt: Any
        if not resume:
            prompt = inp.prompt or ""
        elif injected:
            prompt = self._user_message(session_id, injected, inp.prompt)
        else:
            prompt = inp.prompt  # a new task on the session

        result: ResultMessage | None = None
        engine_version = "(unknown version)"
        paused_by_hook: str | None = None
        last_assistant: str | None = None
        store_error: str | None = None
        stopped_by_hook = False
        # When the Activity is cancelled or times out, deny every later tool call
        # while the SDK shuts the engine down.
        stopper = (
            asyncio.ensure_future(_stop_hooks_when_cancelled(hook_dir))
            if activity.in_activity()
            else None
        )
        try:
            async for message in query(
                prompt=prompt, options=ClaudeAgentOptions(**options)
            ):
                if isinstance(message, MirrorErrorMessage):
                    store_error = message.error or "unknown error"
                elif isinstance(message, SystemMessage) and message.subtype == "init":
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
                    stopped_by_hook = stopped_by_hook or _hook_said_stopped(message)
                elif isinstance(message, ResultMessage):
                    result = message
            marker = Path(hook_dir) / "paused_call"  # written when the hook defers
            if marker.exists():
                paused_by_hook = marker.read_text(encoding="utf-8").strip() or None
        except ResultError as err:
            final = err.subtype in FINAL_RESULT_ERRORS or _final_api_error(err)
            if not final:
                raise  # other engine errors: let Temporal retry the segment
            # Retrying will not help.
            return SegmentOutput(session_id=session_id, is_error=True, error=str(err))
        except RuntimeError as err:
            if _moved(err):
                raise _SessionMoved(str(err)) from err
            raise
        finally:
            if stopper is not None:
                stopper.cancel()
            shutil.rmtree(hook_dir, ignore_errors=True)  # the hook denies from now on

        if stopped_by_hook:
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
        checkpoint = await self._checkpoint(
            sid,
            last_assistant,
            deferred.id if deferred is not None else None,
            guard,
        )
        if deferred is not None:
            name = deferred.name.removeprefix(PREFIX)
            return SegmentOutput(
                session_id=sid,
                deferred=DeferredCall(
                    id=deferred.id, name=name, input=dict(deferred.input)
                ),
                checkpoint=checkpoint,
                cost_usd=cost,
            )
        return SegmentOutput(
            session_id=sid, result=result.result, checkpoint=checkpoint, cost_usd=cost
        )

    def _engine_options(
        self,
        inp: SegmentInput,
        injected: dict[str, ToolOutcome],
        session_id: str,
        resume: bool,
        guard: str | None,
        hook_dir: str,
        durable_server: Any,
    ) -> dict[str, Any]:
        """The ``ClaudeAgentOptions`` fields of one engine run, ``extra_options`` merged in."""
        extra = dict(self._extra)
        default_prompt = extra.pop("system_prompt", None)
        hint = ONE_TOOL_HINT if self._one_tool else None
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
        payload = sum(len(_text(o.content)) for o in injected.values())
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
            "allowed_tools": list(
                dict.fromkeys(
                    [
                        *durable_names,
                        *inp.builtin_tools,
                        *(extra.pop("allowed_tools", None) or []),
                    ]
                )
            ),
            "settings": str(Path(hook_dir) / "settings.json"),
            "session_store": (
                _guarded(self._store, session_id, guard)
                if guard is not None
                else self._store
            ),
            "cwd": self._cwd,
            "env": {
                **self._env,
                **(extra.pop("env", None) or {}),
                **ENGINE_ENV,
                "TCA_HOOK_DIR": hook_dir,
                "TCA_ANSWERED_IDS": " ".join(injected),
            },
            "cli_path": self._cli_path,
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
        blocks: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": tid,
                "content": _text(o.content),
                "is_error": o.is_error,
            }
            for tid, o in injected.items()
        ]
        if prompt:
            blocks.append({"type": "text", "text": prompt})
        yield {
            "type": "user",
            "message": {"role": "user", "content": blocks},
            "parent_tool_use_id": None,
            "session_id": session_id,
        }
