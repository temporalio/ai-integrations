"""PreToolUse command hook for the Claude Code engine (settings based, not in process).

The engine runs this file as a plain script (standard library only, so it starts in
tens of milliseconds) for every tool call. It answers "defer" for durable tools and
for the Claude Code tools that run as their own Activities (``$TCA_TOOL_ACTIVITIES``,
name patterns), so the Workflow runs them. It never answers "allow" for them in a
segment: on a resumed paused call, "allow" sends the engine down its auto-resume
path, where a later "defer" is ignored.

Tool steps: with ``$TCA_ALLOW_ID`` set, the engine resumed a session that paused at a
Claude Code tool call, to run exactly that call; the hook allows it and denies
anything else.

Subagents (``agent_id`` in the event) cannot pause the run. The authenticated native
bridge waits for their Activity outcomes and delivers them to the original calls.
Without a bridge, managed child calls are denied. Other tools run in the subagent
as usual.

Parallel calls: the engine keeps only one paused call per run. So the first new call
in a run that runs as an Activity is deferred, and any other call after it in the
same run is denied. (Read-only calls that the engine runs together, as one batch,
reach the hook at the same time: whichever claims ``paused_call`` first is deferred.) The Workflow runs the denied durable calls with the paused one,
and the next segment puts their real results in place of the denials. A denied
Claude Code call keeps the denial, so Claude calls it again. (A call allowed to run
after the pause would have its result cut from the session, and Claude would run it
again.)

Every denial is also written to ``$TCA_HOOK_DIR/denied/<tool_use_id>`` (the reason's
key), so the runner knows which calls the hook denied without reading tool output,
which a tool controls.

Stopped runs: when the segment Activity is cancelled or times out, the runner writes
``$TCA_HOOK_DIR/stop``, so an engine that is still shutting down cannot start another
built-in tool (a durable call still defers: that ends the run, and no one runs it).
The same holds once the Worker process is gone: while the engine runs, the Worker
holds a lock on ``$TCA_HOOK_DIR/worker.lock``, and a hook that can take it knows the
Worker died. If the folder is missing (the run ended, or the engine cannot see the
Worker's temporary folder), every call is denied, and the runner fails a step that
finished that way.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import sys
import urllib.request
from typing import Any

NOT_RUN = (
    "Not an error. This call did not run yet: another tool call in this message "
    "paused the run. Call this tool again after you get that result."
)
"""The reason Claude sees for a built-in call denied after the pause.

Since Claude Code 2.1.281 a denial reaches Claude as ``PreToolUse:<tool> hook error:
<reason>``, and models often retry a call they see fail that way
(anthropics/claude-code#85490). So the reasons Claude reads say first that nothing
failed, then what to do.
"""

STOPPED = "This step was stopped (cancelled or timed out). Do not call tools."
"""The reason Claude sees when the segment is no longer running (the runner looks for it)."""

MAIN_AGENT_ONLY = (
    "Not an error, and calling it again will not help: this tool runs as its own "
    "Temporal Activity, so only the main agent can call it, not a subagent. Finish "
    "and report back; the main agent can call it."
)
"""The reason a subagent sees when it calls a tool that runs as an Activity."""

STEP_ONLY = "This step runs one tool call only."
"""The reason for any other call in a tool step (the stand-in model makes none)."""


DURABLE_PREFIX = "mcp__durable__"
"""Names of durable tools as the engine sees them (the runner's ``PREFIX``)."""

WORKER_LOCK = "worker.lock"
"""The file in ``$TCA_HOOK_DIR`` that the Worker locks while its engine runs."""

REASON_KEYS = {
    NOT_RUN: "not_run",
    STOPPED: "stopped",
    MAIN_AGENT_ONLY: "main_agent_only",
    STEP_ONLY: "step_only",
}
"""What the hook writes in ``denied/<tool_use_id>`` for each reason."""

_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def denial_name(tool_use_id: str) -> str:
    """The file name of a call's denial record (the id, or its SHA-256 if unusual)."""
    if _SAFE_ID.fullmatch(tool_use_id):
        return tool_use_id
    return hashlib.sha256(tool_use_id.encode("utf-8")).hexdigest()


def _deny(reason: str = NOT_RUN) -> dict[str, Any]:
    return {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }


def _record(run_dir: str | None, tool_use_id: str, output: dict[str, Any]) -> None:
    """Write the denial record of a call the hook denies (when the folder exists)."""
    reason = REASON_KEYS.get(str(output.get("permissionDecisionReason")))
    if output.get("permissionDecision") != "deny" or run_dir is None or reason is None:
        return
    try:
        folder = os.path.join(run_dir, "denied")
        os.makedirs(folder, exist_ok=True)
        with open(
            os.path.join(folder, denial_name(tool_use_id)), "w", encoding="utf-8"
        ) as handle:
            handle.write(reason)
    except OSError:
        pass  # the folder is gone: the runner treats the step as stopped


def _worker_gone(run_dir: str) -> bool:
    """Whether the Worker that started this run is gone: nothing holds its lock.

    The operating system releases a dead process's locks, so the hook can take the
    lock only then. No lock file (an older runner, or a folder where locks do not
    work): the Worker is taken to be there. Hooks that test at the same time take a
    shared lock, so they do not take each other for the Worker (Windows has no
    shared lock; there the job object ends the engine with the Worker).
    """
    try:
        fd = os.open(os.path.join(run_dir, WORKER_LOCK), os.O_RDWR)
    except OSError:
        return os.environ.get("TCA_REQUIRE_LOCK") == "1"
    try:
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                return False  # held: the Worker is alive
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return True
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return False  # held: the Worker is alive
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def _runs_as_activity(name: str) -> bool:
    """Whether a tool call pauses the segment.

    Durable tools do, and the Claude Code tools that match ``$TCA_TOOL_ACTIVITIES``
    (one name pattern per line).
    """
    if name.startswith(DURABLE_PREFIX):
        return True
    patterns = os.environ.get("TCA_TOOL_ACTIVITIES", "").splitlines()
    return any(p and fnmatch.fnmatchcase(name, p) for p in patterns)


def decide(event: dict[str, Any]) -> dict[str, Any]:
    """Return the ``hookSpecificOutput`` for one PreToolUse event.

    Also records the paused call in ``$TCA_HOOK_DIR/paused_call``, so the runner can
    check that the engine really paused there.

    Args:
        event: The PreToolUse event the engine sent on stdin.

    Returns:
        The hook decision.
    """
    tool_use_id = str(event.get("tool_use_id") or "")
    name = str(event.get("tool_name") or "")
    answered = set(os.environ.get("TCA_ANSWERED_IDS", "").split())
    run_dir = os.environ.get("TCA_HOOK_DIR")
    marker = os.path.join(run_dir, "paused_call") if run_dir else None
    allow_id = os.environ.get("TCA_ALLOW_ID")
    stopped = run_dir is not None and (
        os.path.exists(os.path.join(run_dir, "stop")) or _worker_gone(run_dir)
    )
    output: dict[str, Any]
    if run_dir is not None and not os.path.isdir(run_dir):
        output = _deny(STOPPED)  # the run ended, or this hook cannot see its folder
    elif allow_id is not None:  # a tool step: exactly this call, nothing else
        if stopped or (
            os.environ.get("TCA_REQUIRE_NATIVE_PROTECTION") == "1"
            and (
                run_dir is None
                or not os.path.isfile(os.path.join(run_dir, "native_protected"))
            )
        ):
            output = _deny(STOPPED)
        elif tool_use_id == allow_id:
            output = {"hookEventName": "PreToolUse", "permissionDecision": "allow"}
        else:
            output = _deny(STEP_ONLY)
    elif tool_use_id in answered and not event.get("agent_id"):
        # On resume the engine re-announces the call whose result was just delivered.
        # It must never run, whatever the tool (the settings may have changed since).
        output = {"hookEventName": "PreToolUse", "permissionDecision": "defer"}
    elif (
        not event.get("agent_id")
        and _runs_as_activity(name)
        and os.environ.get("TCA_CHILD_URL")
    ):
        # Let native children in the same assistant batch finish their durable
        # deliveries before the parent takes the segment's pause boundary.
        output = _child_hook(event)
        if output.get("permissionDecision") != "deny":
            event = {**event, "_children_drained": True}
            url = os.environ.pop("TCA_CHILD_URL")
            try:
                return decide(event)
            finally:
                os.environ["TCA_CHILD_URL"] = url
    elif event.get("agent_id") and _runs_as_activity(name):
        output = (
            _deny(STOPPED) if stopped else _child_hook(event)
        )  # child calls wait on their Workflow-owned Activity outcome
    elif event.get("agent_id") or not _runs_as_activity(name):
        # A tool that runs in the engine runs normally, unless the step was stopped
        # or a call already paused this run.
        if stopped:
            output = _deny(STOPPED)
        elif marker is not None and os.path.exists(marker):
            output = _deny()
        else:
            output = {"hookEventName": "PreToolUse"}
    else:
        output = {"hookEventName": "PreToolUse", "permissionDecision": "defer"}
        # On resume the engine re-announces the call that was just answered. It must
        # not take the "one paused call per run" slot, or Claude's next call is denied.
        if marker is not None and tool_use_id not in answered:
            try:
                # Atomic: exactly one call wins the slot.
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(tool_use_id)
            except FileExistsError:
                with open(marker, encoding="utf-8") as handle:
                    if handle.read().strip() != tool_use_id:
                        output = _deny()
    if run_dir is not None and os.path.isdir(run_dir):
        _record(run_dir, tool_use_id, output)
    log = os.environ.get("TCA_HOOK_LOG")
    if log:  # debugging aid: one line per decision
        with open(log, "a", encoding="utf-8") as handle:
            decision = output.get("permissionDecision", "none")
            handle.write(f"{tool_use_id} {decision}\n")
    return output


def _child_hook(event: dict[str, Any]) -> dict[str, Any]:
    """Ask the live Worker bridge; missing bridges always deny managed child calls."""
    url = os.environ.get("TCA_CHILD_URL")
    key = os.environ.get("TCA_CHILD_KEY")
    if not url or not key:
        return _deny(MAIN_AGENT_ONLY)
    try:
        request = urllib.request.Request(
            url, json.dumps(event).encode(), {"Authorization": key}, method="POST"
        )
        # Never inherit a provider proxy for this authenticated loopback request.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request) as response:
            output = json.load(response)
        if not isinstance(output, dict):
            raise ValueError("Invalid native hook answer")
        return output
    except Exception:
        return _deny(STOPPED)


def main() -> None:
    """Read one event from stdin and print the decision.

    The engine sends UTF-8; read bytes, so a Windows code page cannot garble or
    reject the tool input. The answer is ASCII (``json.dumps`` escapes the rest).
    """
    event = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    kind = event.get("hook_event_name", "PreToolUse")
    output = decide(event) if kind == "PreToolUse" else _child_hook(event)
    print(json.dumps({"hookSpecificOutput": output} if output else {}))


if __name__ == "__main__":
    main()
