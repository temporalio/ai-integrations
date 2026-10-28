"""Native Claude tools (Bash, Read, Edit, ...) for the turn Activity.

A native call's first ``PreToolUse`` defers, so the run stops and the workflow decides it. The
gate is a command hook (``_native_hook.py``) in ``--settings``, not an SDK callback: on resume
the CLI re-presents the deferred call, sometimes before ``initialize`` registers callbacks.

Fail closed wherever the hook can be skipped or fail: never ``bypassPermissions``, native tools
never in ``allowed_tools``, and the hook exits 2 (blocking) on any error. A timed-out hook, or a
``defer`` the CLI ignores when several calls share one message, decides nothing, and 2.1.294
would then run reads and read-only Bash unasked; so every requested tool also has an ``ask``
rule, which outranks both, and ``can_use_tool`` allows only the call the hook just recorded as
started. Bash runs only in the CLI sandbox (``failIfUnavailable``) and without the worker's
credentials, because the SDK hands the CLI the whole environment and Bash output reaches
workflow history.
"""

from __future__ import annotations

import fnmatch
import functools
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

from temporalio.claude_agent_sdk import _native_hook
from temporalio.claude_agent_sdk._cli import ResolvedCLI
from temporalio.claude_agent_sdk._models import (
    NATIVE_TOOL_EFFECTS,
    ClaudeToolResult,
    ClaudeTurnInput,
    cli_tools,
    mcp_tool_name,
)
from temporalio.claude_agent_sdk._models import merge_native_calls as _merge
from temporalio.claude_agent_sdk._sdk import load_sdk

logger = logging.getLogger(__name__)

NON_DURABLE_NATIVE_TOOLS = frozenset({"Agent", "Task", "Skill", "Workflow"})
"""Claude tools that start another agent (``Task`` is 2.1.294's name for ``Agent``), skill, or
workflow. Their nested loop is not in this integration's ledger, so a lost worker could repeat
it. The hook denies the same names; the Activity refuses every tool ``NATIVE_TOOL_EFFECTS``
does not classify, these included."""

NATIVE_TOOLS_VERIFIED: frozenset[tuple[str, str]] = frozenset({("0.2.165", "2.1.294")})
"""``(claude-agent-sdk version, Claude Code CLI version)`` pairs native tools may run on.

The CLI version is what the CLI the turn actually runs reports for ``--version`` (see
``_cli.resolve_cli``), so a ``cli_path`` at another CLI, or an SDK whose bundled CLI changed,
keeps native tools off. amd64 Linux is unverified. Add a pair only after the same checks pass
on it.
"""

NATIVE_STATE_DIR_NAME = ".native-tool-state"
"""Under ``workspace_root``, the parent of each session's native-tool state directory."""

HOOK_SCRIPT = Path(__file__).with_name("_native_hook.py")

HOOK_TIMEOUT_SECONDS = 10
"""Bound on each run of the command hook; a timed-out hook's call is denied by ``can_use_tool``."""

NATIVE_REREQUEST_CAP = 3
"""Native calls a session may make in runs that resume a denial; past it the hook denies them
without asking."""

SANDBOX: dict[str, Any] = {
    "enabled": True,
    # Not in the SDK's SandboxSettings TypedDict; the CLI reads it from --settings and refuses
    # to run commands when no sandbox backend is available instead of running them unconfined.
    "failIfUnavailable": True,
    "autoAllowBashIfSandboxed": False,
    "allowUnsandboxedCommands": False,
    "excludedCommands": [],
    # strictAllowlist also stops a Bash call's own `allowed_domains` from widening the empty
    # allowlist (2.1.294 let one reach the internet without it).
    "network": {
        "allowedDomains": [],
        "strictAllowlist": True,
        "allowLocalBinding": False,
        "allowAllUnixSockets": False,
        "allowUnixSockets": [],
    },
}

DEFAULT_NATIVE_ENV_DENYLIST: tuple[str, ...] = (
    "ANTHROPIC_*",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "TEMPORAL_*KEY*",
    "TEMPORAL_*SECRET*",
    "TEMPORAL_*CERT*",
    "AWS_*",
    "AZURE_*",
    "MODAL_*",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
    "*_API_KEY",
    "*_ACCESS_KEY*",
    "*_SECRET",
    "*_SECRET_KEY",
    "*_PRIVATE_KEY*",
    "*_TOKEN",
    "*_PASSWORD",
    "*DATABASE_URL",
    "*_DSN",
)
"""Worker environment variables native Bash never sees (``fnmatch`` patterns, case-sensitive).
The CLI process keeps all of them, so it still authenticates with ``ANTHROPIC_API_KEY`` (or
``CLAUDE_CODE_OAUTH_TOKEN``, or Bedrock/Vertex credentials); only its sandboxed commands lose
them. See :func:`sandbox_settings`."""

REMOTE_CREDENTIAL_ENV: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)
"""Credentials a remote sandbox's secrets commonly hold. The worker cannot see that
environment, so these names, and the denylist's literal names, are denied to native Bash there
whenever the denylist matches them."""

SUBPROCESS_ENV_SCRUB = "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"
"""Claude Code's documented switch (https://code.claude.com/docs/en/env-vars) that strips
credentials from every subprocess it starts (Bash, hooks, stdio MCP servers) while the CLI keeps
them for its own API calls. Set only on macOS: on Linux, Claude Code 2.1.294 with it runs Bash
with writes outside the cwd allowed. Bubblewrap gives Bash its own PID namespace either way, so a
command cannot read the CLI's environment through ``/proc``."""

ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
"""What ``sandbox.credentials.envVars[].name`` accepts (https://code.claude.com/docs/en/settings)."""
NOT_APPROVED = "Not approved. Call this tool again on its own."

MANAGED_SETTINGS_FILE = Path(
    "/Library/Application Support/ClaudeCode/managed-settings.json"
    if sys.platform == "darwin"
    else "/etc/claude-code/managed-settings.json"
)
"""Claude Code's managed settings on this host. They load whatever ``setting_sources`` says and
outrank the run's ``--settings``, as do the ``managed-settings.d/*.json`` drop-ins beside it and
the cached server-managed ``remote-settings.json`` in the Claude config directory. MDM profiles
are not visible here."""

_MANAGED_OVERRIDES = frozenset(
    {
        "sandbox",
        "permissions",
        "allowManagedHooksOnly",
        "allowManagedPermissionRulesOnly",
        "disableAllHooks",
        "env",
    }
)
"""Managed keys that can replace the run's Bash sandbox, its ask/deny rules, its gate hook, or
the CLI's environment (``BASH_ENV``, ``LD_PRELOAD``, the subprocess scrub). Managed hooks alone
cannot loosen the gate: the most restrictive decision wins, and the ask rules still send every
call through ``can_use_tool``."""


def _managed_files() -> list[Path]:
    """Every managed settings file Claude Code 2.1.294 loads, in its order."""
    drop_ins = MANAGED_SETTINGS_FILE.parent / "managed-settings.d"
    try:
        names = sorted(
            p.name
            for p in drop_ins.iterdir()
            if p.name.endswith(".json") and not p.name.startswith(".")
        )
    except FileNotFoundError:
        names = []
    config = os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude"
    return [
        MANAGED_SETTINGS_FILE,
        *(drop_ins / n for n in names),
        Path(config) / "remote-settings.json",
    ]


def managed_overrides() -> list[str]:
    """The keys of this host's managed settings that would override the native-tool gate."""
    keys: set[str] = set()
    try:
        files = _managed_files()
    except OSError:
        return ["(unreadable)"]
    for path in files:
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            return ["(unreadable)"]
        if not isinstance(data, dict):
            return ["?"]
        keys |= _MANAGED_OVERRIDES & data.keys()
    return sorted(keys)


_MCP_PREFIX = mcp_tool_name("")


def native_tools_refusal(
    input: ClaudeTurnInput,
    *,
    enabled: bool,
    sdk_version: str,
    cli: ResolvedCLI,
    in_sandbox: bool = False,
    require_seccomp_helper: bool = True,
) -> str | None:
    """Why the Activity must refuse ``input``'s native tools, or None when it may run them.

    ``cli`` is the CLI the turn would run. A remote sandbox (``in_sandbox``) checks the CLI
    version when the sandbox starts, and Claude Code's Bash sandbox (``failIfUnavailable``)
    checks the image. The worker's Seatbelt or bubblewrap is not that check.
    ``require_seccomp_helper=False`` skips :func:`seccomp_requirement`.
    """
    if not cli_tools(input):
        return None
    if not enabled:
        return (
            "native Claude tools are disabled on this worker "
            "(ClaudeAgentPlugin(enable_native_tools=True) opts in)"
        )
    if blocked := [t for t in cli_tools(input) if t not in NATIVE_TOOL_EFFECTS]:
        # 2.1.294 also offers tools that act outside the run (Monitor, CronCreate,
        # ScheduleWakeup, SendMessage, EnterWorktree, ...) or start nested orchestration (Task).
        return (
            f"native tools {blocked} cannot be made durable: only "
            f"{sorted(NATIVE_TOOL_EFFECTS)} are recorded by this integration"
        )
    if not in_sandbox and (sdk_version, cli.version) not in NATIVE_TOOLS_VERIFIED:
        return (
            f"native Claude tools are not verified on claude-agent-sdk {sdk_version} with Claude "
            f"Code CLI {cli.version or '(version unknown)'} at {cli.path}: the native-tool release "
            f"gates have passed only on (SDK, CLI) {sorted(NATIVE_TOOLS_VERIFIED)}"
        )
    if not in_sandbox and not sandbox_backend_available():
        return (
            "native Claude tools need a sandbox backend on the worker: Seatbelt on macOS, "
            "bubblewrap and socat on Linux"
        )
    if not in_sandbox and require_seccomp_helper and (problem := seccomp_requirement()):
        return problem
    if not in_sandbox and (keys := managed_overrides()):
        return (
            f"native Claude tools are refused: this host's Claude Code managed settings "
            f"({MANAGED_SETTINGS_FILE}, its managed-settings.d drop-ins, or remote-settings.json) "
            f"set {keys}, which override the run's sandbox, permission rules, gate hook or "
            "environment"
        )
    if input.session_id is None:
        return "native tools need a session_id"
    if bad := [t for t in cli_tools(input) if t.startswith("mcp__")]:
        return f"{bad} are MCP tools, not native tools"
    return None


def sandbox_backend_available() -> bool:
    """Whether Claude Code's Bash sandbox can run on this host: Seatbelt on macOS, bubblewrap
    and socat on Linux. Without one the Activity refuses native tools before starting a CLI;
    ``failIfUnavailable`` stays the second layer (CLI 2.1.294 refuses to start without them).
    """
    if sys.platform == "darwin":
        return Path("/usr/bin/sandbox-exec").exists()
    if sys.platform.startswith("linux"):
        return bool(shutil.which("bwrap") and shutil.which("socat"))
    return False


SANDBOX_RUNTIME_PACKAGE = "@anthropic-ai/sandbox-runtime"
SANDBOX_RUNTIME_PIN = "0.0.79"
"""The ``@anthropic-ai/sandbox-runtime`` version a Linux worker installs globally for its optional
seccomp helper (which blocks Unix sockets in Bash's sandbox), and the only one native tools accept
there. Claude Code starts its sandbox without the helper and its version does not change when the
helper is missing, so ``NATIVE_TOOLS_VERIFIED`` cannot pin it."""


@dataclass(frozen=True)
class SeccompHelper:
    """sandbox-runtime's ``apply-seccomp`` binary in the global npm install."""

    path: Path
    version: str | None


@functools.cache
def find_seccomp_helper() -> SeccompHelper | None:
    """The seccomp helper where Claude Code looks for it (the global npm install of
    ``SANDBOX_RUNTIME_PACKAGE``), for this machine's architecture; looked up once per process.
    Only looks; never runs it.
    """
    if (npm := shutil.which("npm")) is None:
        return None
    try:
        root = subprocess.run(
            [npm, "root", "-g"],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not root:
        return None
    package = Path(root, *SANDBOX_RUNTIME_PACKAGE.split("/"))
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64"}.get(machine, machine)
    binary = next(
        (
            p
            for p in package.rglob("apply-seccomp")
            if p.is_file() and p.parent.name == arch
        ),
        None,
    )
    if binary is None:
        return None
    try:
        version = json.loads((package / "package.json").read_text())["version"]
    except (OSError, ValueError, KeyError, TypeError):
        version = None
    return SeccompHelper(binary, version if isinstance(version, str) else None)


def seccomp_requirement() -> str | None:
    """Why this Linux worker may not run native tools for want of the pinned seccomp helper, or
    None (always off Linux).
    """
    if not sys.platform.startswith("linux"):
        return None
    found = find_seccomp_helper()
    if found is not None and found.version == SANDBOX_RUNTIME_PIN:
        return None
    seen = (
        "none is in the global npm install"
        if found is None
        else f"found version {found.version or '(unknown)'} at {found.path}"
    )
    return (
        f"native Claude tools on Linux need sandbox-runtime's seccomp helper {SANDBOX_RUNTIME_PIN} "
        f"({seen}): install it with `npm install -g {SANDBOX_RUNTIME_PACKAGE}@"
        f"{SANDBOX_RUNTIME_PIN}`, or opt out with ClaudeAgentPlugin(require_seccomp_helper=False)"
    )


def sandbox_settings(
    env: Mapping[str, str], denylist: Sequence[str], *, scrub: bool = True
) -> tuple[dict[str, Any], dict[str, str]]:
    """The run's ``sandbox`` settings and the CLI environment overrides that keep the variables
    of ``env`` (what the CLI process receives) matching ``denylist`` out of native Bash.

    ``CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`` is set only for ``scrub=True`` on macOS: it lifts Bash
    confinement on Claude Code 2.1.294 under Linux, so a remote runtime (Linux) passes
    ``scrub=False``. This function configures Claude Code's Bash sandbox, not a remote
    ``SandboxRuntime``.

    Two documented layers, both leaving the CLI process its own credentials:
    ``sandbox.credentials.envVars`` deny entries, which unset each named variable before every
    sandboxed command (https://code.claude.com/docs/en/sandboxing#protect-credentials; native
    Bash only ever runs sandboxed), and :data:`SUBPROCESS_ENV_SCRUB`, which also covers hooks
    and catches credentials by value. The deny entries take exact names, so the patterns are
    expanded here; a matching name the settings cannot spell is blanked in the CLI's own
    environment instead. An empty ``denylist`` turns both layers off.
    """
    if not denylist:
        return SANDBOX, {}
    denied = sorted(n for n in env if any(fnmatch.fnmatchcase(n, p) for p in denylist))
    entries = [{"name": n, "mode": "deny"} for n in denied if ENV_NAME.match(n)]
    overrides = {n: "" for n in denied if not ENV_NAME.match(n)}
    scrubbed = {SUBPROCESS_ENV_SCRUB: "1"} if scrub and sys.platform == "darwin" else {}
    return {**SANDBOX, "credentials": {"envVars": entries}}, {**scrubbed, **overrides}


# Edit rules also deny Write and NotebookEdit; Write(...) rules are ignored (2.1.294). Claude Code
# merges Read and Edit denies into the Bash sandbox's denyRead and denyWrite.
_FILE_TOOLS = ("Read", "Edit")

_SYSTEM_DENY_DIRS = (PurePosixPath("/proc"), PurePosixPath("/sys"))
"""Kernel views no file tool needs. On Linux ``/proc/<pid>/environ`` holds the CLI's credentials."""

_HOME_DENY_DIRS = ("~/.aws", "~/.ssh", "~/.modal.toml", "~/.config/gcloud", "~/.claude")
"""Credentials, and the default Claude config directory (every session's transcripts), under
the CLI's own home. ``~`` is resolved by the CLI, so the rules hold in a remote sandbox too."""

_GIT_DENIES = ("Edit(//**/.git)", "Edit(//**/.git/**)")
"""Claude Code runs ``git status`` unsandboxed when the CLI starts, and git runs the clean filter
a repository's ``.git/config`` names. Bubblewrap ignores the glob, so the session directory's own
``.git`` is denied by path as well."""


def _permission_denies(*directories: PurePath | str) -> list[str]:
    """Claude Code permission rules that keep the file tools and Bash off these directories.

    ``//`` makes a rule absolute; ``/x`` is relative to a settings file and ``../x`` matches
    nothing (both measured on 2.1.294). The CLI resolves a relative path or a symlink before
    matching, so the resolved directory is enough. A ``PurePosixPath`` names a directory in a
    remote sandbox and a ``~/`` string one under the CLI's home; both are used as given.
    """
    specs = [
        d
        if isinstance(d, str)
        else f"/{(d.resolve() if isinstance(d, Path) else d).as_posix()}"
        for d in directories
    ]
    return [
        f"{tool}({spec}{suffix})"
        for spec in specs
        for tool in _FILE_TOOLS
        for suffix in ("", "/**")
    ]


def hook_command(state_dir: Path) -> str:
    """The shell command Claude Code runs for every ``PreToolUse``: isolated (``-I``) and without
    site packages (``-S``), so nothing but the stdlib and the script itself is loaded.
    """
    return shlex.join([sys.executable, "-I", "-S", str(HOOK_SCRIPT), str(state_dir)])


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def prepare_state(state_dir: Path, input: ClaudeTurnInput) -> None:
    """Write what the hook decides from, for one run, before the CLI starts.

    The workflow's input is the source of truth: everything is rewritten from it on every run, so
    a turn that lands on another worker sees the same decisions and denials. Only the call this run
    resumes gets a decision file, so an earlier approval can never be replayed. The started-call
    ledger is the one file kept: the workflow's copy (``input.native_calls``) is merged into it,
    so a call a lost worker started is refused here too, and an entry only this host recorded
    (the workflow had not heard of it yet) is not dropped.
    """
    if input.native_calls:
        merge_native_calls(state_dir, [c.model_dump() for c in input.native_calls])
    decisions = state_dir / "decisions"
    shutil.rmtree(decisions, ignore_errors=True)
    decisions.mkdir(parents=True)
    (state_dir / "ledger.jsonl").unlink(missing_ok=True)
    decision = input.native_decision
    if decision is not None:
        body = {
            "decision": "approved" if decision.approved else "denied",
            "reason": decision.reason,
            "key": _native_hook.call_key(decision.call.name, decision.call.input),
        }
        _write_atomic(decisions / f"{decision.call.tool_use_id}.json", json.dumps(body))
    _write_atomic(
        state_dir / "denied.jsonl",
        "".join(
            json.dumps({"tool_name": d.name, "tool_input": d.input}) + "\n"
            for d in input.native_denied
        ),
    )
    turn = {
        "resumed": decision.call.tool_use_id if decision else None,
        "after_denial": decision is not None and not decision.approved,
        "rerequest_budget": max(0, NATIVE_REREQUEST_CAP - input.native_rerequests),
        "tools": input.native_tools,
        "auto_allow": input.native_auto_allow or [],
        "run": uuid.uuid4().hex,
    }
    _write_atomic(state_dir / "turn.json", json.dumps(turn))


def ledger(state_dir: Path) -> list[dict[str, Any]]:
    """Every decision the hook made in this run, in order."""
    path = state_dir / "ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def native_calls(state_dir: Path) -> list[dict[str, Any]]:
    """The session's started-call ledger (``calls.json``, see ``_native_hook.py``): one entry per
    call the hook allowed, ``{"tool_use_id", "tool_name", "key", "run", "status"}`` with status
    ``started`` (no PostToolUse yet), ``finished``, ``failed`` or ``reported``. Raises
    ``ValueError`` (or ``json.JSONDecodeError``) on a malformed ledger, as the hook does.
    """
    return _native_hook.load_calls(str(state_dir))


def unfinished_native_calls(state_dir: Path) -> list[dict[str, Any]]:
    """Calls the hook let run whose outcome was never recorded: what a lost attempt may have done."""
    return [c for c in native_calls(state_dir) if c["status"] == "started"]


def record_native_finished(
    state_dir: Path, tool_use_id: str, *, failed: bool = False
) -> None:
    """Mark ``tool_use_id`` finished (or failed) after its revision has committed.

    The command hook does not do this for a non-repeatable tool: that write has to wait until
    the workspace revision containing the effect exists. A no-op for an unknown id.
    """
    _native_hook._mark(
        str(state_dir),
        tool_use_id,
        status="failed" if failed else "finished",
    )


def merge_native_calls(state_dir: Path, calls: list[dict[str, Any]]) -> None:
    """Fold ledger entries carried from elsewhere (the workflow's copy) into this host's ledger
    before the CLI starts: unknown ids are added, and a known id keeps the later of the two
    statuses (started < reported < finished/failed).
    """
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    _native_hook.update_calls(str(state_dir), lambda local: _merge(local, calls))


def settle_answered_deferrals(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The transcript as a CLI that never resumed a deferred call would have written it.

    When the hook defers a call, 2.1.294 writes a ``hook_deferred_tool`` marker after it and
    never removes it. The run that resumes the approved call chains its ``hook_success`` to the
    deferring run's last entry but the call's ``tool_result`` to the ``tool_use``, so the
    transcript forks. A CLI that loads it anchors on that answer and never sees the session's
    next deferred call, so its approved resume runs nothing. This drops the marker of every
    answered call (entries chained to it move to its parent) and chains the answer after the
    resume's ``hook_success``. A marker without an answer stays: the next resume needs it.
    Nothing is added. ``entries`` itself is returned when nothing changes.
    """
    answers: dict[Any, dict[str, Any]] = {}
    for entry in entries:
        if entry.get("type") == "user":
            for block in content_blocks(entry):
                if block.get("type") == "tool_result":
                    answers.setdefault(block.get("tool_use_id"), entry)
    parents = {entry.get("parentUuid") for entry in entries}
    dropped: dict[Any, Any] = {}
    relinked: dict[Any, Any] = {}
    for at, entry in enumerate(entries):
        marker = _attachment(entry, "hook_deferred_tool")
        if marker is None or marker.get("toolUseID") not in answers:
            continue
        dropped[entry.get("uuid")] = entry.get("parentUuid")
        # The resume's hook_success is a branch of its own: nothing is chained to it yet.
        resumed = next(
            (
                later
                for later in entries[at + 1 :]
                if (hook := _attachment(later, "hook_success")) is not None
                and hook.get("toolUseID") == marker["toolUseID"]
                and later.get("uuid") not in parents
            ),
            None,
        )
        answer = answers[marker["toolUseID"]].get("uuid")
        if resumed is not None and answer:
            relinked[answer] = resumed["uuid"]
    if not dropped:
        return entries
    kept = []
    for entry in entries:
        if entry.get("uuid") in dropped:
            continue
        parent = relinked.get(entry.get("uuid"), entry.get("parentUuid"))
        while parent in dropped:
            parent = dropped[parent]
        kept.append(
            entry
            if parent == entry.get("parentUuid")
            else {**entry, "parentUuid": parent}
        )
    return kept


_INHERITED_FIELDS = (
    "isSidechain",
    "userType",
    "entrypoint",
    "cwd",
    "sessionId",
    "version",
    "gitBranch",
)


def result_entry(assistant: dict[str, Any], result: ClaudeToolResult) -> dict[str, Any]:
    """The transcript entry the CLI writes for a tool call's result: a user entry chained to
    the assistant entry holding the call, as it chains a parallel sibling's result.
    """
    content = [{"type": "text", "text": result.content}]
    block: dict[str, Any] = {
        "tool_use_id": result.tool_use_id,
        "type": "tool_result",
        "content": content,
    }
    if result.is_error:
        block["is_error"] = True
    return {
        "parentUuid": assistant["uuid"],
        "type": "user",
        "message": {"role": "user", "content": [block]},
        "uuid": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "toolUseResult": content,
        "sourceToolAssistantUUID": assistant["uuid"],
        **{k: assistant[k] for k in _INHERITED_FIELDS if k in assistant},
    }


def answer_call(
    entries: list[dict[str, Any]], tool_use_id: str, text: str
) -> dict[str, Any] | None:
    """The error result entry answering call ``tool_use_id`` with ``text``; None when the
    transcript does not hold the call or already answers it.
    """
    holder = None
    for entry in entries:
        for block in content_blocks(entry):
            if (
                block.get("type") == "tool_result"
                and block.get("tool_use_id") == tool_use_id
            ):
                return None
            if block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                holder = entry
    if holder is None or "uuid" not in holder:
        return None
    return result_entry(
        holder, ClaudeToolResult(tool_use_id=tool_use_id, content=text, is_error=True)
    )


Answer = tuple[str, str]
"""``(tool_use_id, text)``: a deferred call to answer with an error result before settling."""


def settle_transcript_file(data: bytes, answer: Answer | None = None) -> bytes | None:
    """:func:`settle_answered_deferrals` for a JSONL transcript, after answering ``answer``'s
    call (:func:`answer_call`); None when nothing changes.

    Data that is not a transcript of JSON objects is left alone (None).
    """
    try:
        entries = [
            json.loads(line) for line in data.decode().splitlines() if line.strip()
        ]
    except ValueError:
        return None
    if not all(isinstance(entry, dict) for entry in entries):
        return None
    if answer is not None and (entry := answer_call(entries, *answer)) is not None:
        entries = [*entries, entry]
        kept = settle_answered_deferrals(entries)
    elif (kept := settle_answered_deferrals(entries)) is entries:
        return None
    return "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept).encode()


def settle_local_transcript(path: Path, answer: Answer | None = None) -> None:
    """Rewrite the transcript at ``path`` in place if it has a call to answer or answered
    deferrals to settle.
    """
    try:
        settled = settle_transcript_file(path.read_bytes(), answer)
    except FileNotFoundError:
        return
    if settled is not None:
        _write_atomic(path, settled.decode())


def _attachment(entry: dict[str, Any], kind: str) -> dict[str, Any] | None:
    attachment = entry.get("attachment") if entry.get("type") == "attachment" else None
    if isinstance(attachment, dict) and attachment.get("type") == kind:
        return attachment
    return None


def content_blocks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """The dict content blocks of one transcript entry; empty if its content is a string."""
    content = (entry.get("message") or {}).get("content")
    return (
        [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
    )


class NativeToolHooks:
    """The ``ClaudeAgentOptions`` that put one run's native tool calls behind the command hook.

    Bridged tools (``mcp__temporal__*``) are left to the tool bridge, their one gate.
    ``after_tool(name, tool_use_id, failed)`` persists a non-repeatable effect before the call
    is marked finished.
    """

    def __init__(
        self,
        state_dir: Path,
        after_tool: Callable[[str, str | None, bool], Awaitable[bool]] | None = None,
        *,
        on_settled: Callable[[], Awaitable[None]] | None = None,
        deny_dirs: Sequence[PurePath] = (),
        command: str | None = None,
        pull: Callable[[], Awaitable[None]] | None = None,
        tools: Sequence[str] = (),
    ) -> None:
        """Bind one run's state directory. See the class docstring.

        ``tools`` (the run's built-ins) get ``ask`` rules, so a call the command hook did not
        allow reaches ``can_use_tool`` instead of Claude Code's own read-only auto-approval.

        ``after_tool`` returns False when no committed copy holds the effect yet; the call then
        stays ``started``.
        ``on_settled`` runs only after the ledger marks the call finished, so a progress report
        never carries ``finished`` without the revision that contains the effect.
        ``command`` replaces the local hook command (a remote sandbox runs ``python3`` on a
        script pushed into the sandbox). ``pull`` copies that sandbox's ledger back first.
        """
        self._state_dir = state_dir
        self._after_tool = after_tool
        self._on_settled = on_settled
        self._deny_dirs = tuple(deny_dirs)
        self._command = command
        self._pull = pull
        self._tools = list(tools)

    def options(self) -> dict[str, Any]:
        """Return the option fields that gate native tools."""
        sdk = load_sdk()

        def every_tool(hook: Any) -> list[Any]:
            return [sdk.HookMatcher(matcher=None, hooks=[hook])]

        gate = [
            {
                "matcher": "*",
                "hooks": [
                    {
                        "type": "command",
                        "command": self._command or hook_command(self._state_dir),
                        "timeout": HOOK_TIMEOUT_SECONDS,
                    }
                ],
            }
        ]
        # PostToolUse / PostToolUseFailure run the same script to close the call's ledger entry.
        hooks = {"PreToolUse": gate, "PostToolUse": gate, "PostToolUseFailure": gate}
        settings: dict[str, Any] = {"hooks": hooks}
        # The file tools are not sandboxed. Deny them, and Bash, the ledger, the process-group
        # directory, git metadata, credentials and the kernel's views, even after an approval.
        # Ask rules outrank a hook's allow as well, so can_use_tool admits only what it allowed.
        permissions = {
            "deny": [
                *_permission_denies(
                    *self._deny_dirs, *_SYSTEM_DENY_DIRS, *_HOME_DENY_DIRS
                ),
                *_GIT_DENIES,
            ],
            "ask": self._tools,
        }
        if permissions := {k: v for k, v in permissions.items() if v}:
            settings["permissions"] = permissions
        return {
            "permission_mode": "default",
            # Inline JSON: the SDK silently drops a settings FILE that is missing when it merges
            # in `sandbox`, and the gate would vanish with it.
            "settings": json.dumps(settings),
            "hooks": {
                "PostToolUse": every_tool(self.post_tool_use),
                "PostToolUseFailure": every_tool(self.post_tool_use_failure),
            },
            "can_use_tool": self.can_use_tool,
            "sandbox": SANDBOX,
        }

    async def post_tool_use(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,  # type: ignore
    ) -> dict[str, Any]:
        """Record a successful native call once its workspace revision has committed."""
        await self._after(input_data, tool_use_id, failed=False)
        return {}

    async def post_tool_use_failure(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,  # type: ignore
    ) -> dict[str, Any]:
        """Record a failed native call once its workspace revision has committed."""
        await self._after(input_data, tool_use_id, failed=True)
        return {}

    async def can_use_tool(
        self,
        tool_name: str,  # type: ignore
        tool_input: dict[str, Any],  # type: ignore
        context: Any,  # type: ignore
    ) -> Any:
        """Allow only a call the command hook allowed in this run, which it recorded ``started``
        before answering. The hook is the only approval path; a call it never decided (it timed
        out, or Claude Code skipped it) is denied, as is any other permission request.
        """
        sdk = load_sdk()
        try:
            if self._pull is not None:
                await self._pull()
            run = json.loads((self._state_dir / "turn.json").read_text())["run"]
            allowed = any(
                (c["tool_use_id"], c.get("tool_name"), c.get("run"), c["status"])
                == (context.tool_use_id, tool_name, run, "started")
                for c in native_calls(self._state_dir)
            )
        except Exception as exc:  # noqa: BLE001 - an unreadable ledger approves nothing
            logger.warning(
                "Could not read the native ledger for %s: %r", tool_name, exc
            )
            allowed = False
        if allowed:
            return sdk.PermissionResultAllow()
        return sdk.PermissionResultDeny(message=NOT_APPROVED)

    async def _after(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        *,
        failed: bool,
    ) -> None:
        name = input_data.get("tool_name") or ""
        if name.startswith(_MCP_PREFIX):
            return
        if self._pull is not None:
            await self._pull()
        call_id = input_data.get("tool_use_id") or tool_use_id
        # Persist first. A non-repeatable call stays started when the revision does not commit,
        # so a later worker cannot restore the old tree and treat the effect as already done.
        if (
            self._after_tool is not None
            and await self._after_tool(name, call_id, failed) is False
        ):
            return
        if call_id is not None:
            try:
                record_native_finished(self._state_dir, call_id, failed=failed)
            except Exception as exc:  # noqa: BLE001 - left "started", a retry asks instead of re-running
                logger.warning("Could not record %s as finished: %r", call_id, exc)
        if self._on_settled is not None:
            await self._on_settled()
