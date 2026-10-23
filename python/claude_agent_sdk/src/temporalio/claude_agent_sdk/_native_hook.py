"""Gate every native Claude tool call. Standard library only.

Claude Code runs this file as ``python -I -S`` for PreToolUse, PostToolUse,
and PostToolUseFailure. It must not import this package. Anything it cannot
decide cleanly must not allow the call: an error exits 2, which blocks it, and
``can_use_tool`` admits only a call this script recorded as started.
"""

# Decisions come from files the turn Activity wrote before starting the CLI:
#
#   decisions/<tool_use_id>.json  {"decision": "approved" | "denied", "reason": ..., "key": ...}:
#                                 the workflow's decision for the call this run resumes (only
#                                 that one); an approval holds only for the call with that key
#   denied.jsonl                  {"tool_name", "tool_input"} per call already denied in the session
#   turn.json                     {"resumed": <tool_use_id> | null, "after_denial": bool,
#                                  "rerequest_budget": int, "tools": [the message's native tools],
#                                  "auto_allow": [read-only tools allowed without deferral],
#                                  "run": <id of this CLI run>}
#   ledger.jsonl                  appended here, one line per decision (with its reason); the
#                                 Activity counts its re-requests
#   calls.json                    the started-call ledger, kept across runs (see below)
#
# calls.json is {"version": 1, "calls": [{"tool_use_id", "tool_name", "key",
# "run", "status", "reported_run"}]}. An allowed call is recorded "started" BEFORE the allow is
# printed; PostToolUse marks it "finished", PostToolUseFailure "failed". A call an earlier run left
# "started" may have taken effect, so it never runs again unasked: its tool_use_id is denied, and a
# new identical call is denied once with UNKNOWN_OUTCOME (marked "reported"). Deny rather than
# defer: an auto-approving policy would grant a deferral without anyone learning the call may have
# run. Written with temp file + fsync + rename under an flock; a ledger that does not parse fails
# the hook (closed).

from __future__ import annotations

import json
import os
import sys
import tempfile

MCP_PREFIX = "mcp__temporal__"
REPEAT_DENIED = "This exact call was already denied; do not retry it."
OVER_CAP = "Too many tool requests after a denial. Stop and tell the user."
CALLS_FILE = "calls.json"
UNKNOWN_OUTCOME = (
    "A previous attempt may have already run this exact call: it started, and its result was never "
    "recorded. Check whether it took effect, and ask the user before retrying it."
)
# The same set as ``NON_DURABLE_NATIVE_TOOLS`` in ``_native.py``. This file cannot import it.
NON_DURABLE_TOOLS = frozenset({"Agent", "Task", "Skill", "Workflow"})
ALREADY_RAN = "This call already ran in an earlier attempt; it will not run again."
ONE_AT_A_TIME = (
    "Another tool call from this message is already waiting for approval. Make one tool call "
    "per message: repeat this call by itself after that one is resolved."
)
SAME_RUN_STARTED = "This call already started in this run; it will not run again."
# The activity marks every other native tool finished only after the workspace revision that
# contains the effect has committed. Unknown tools stay started too: a missing row is at-least-once.
REPEATABLE_TOOLS = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "WebFetch",
        "WebSearch",
        "BashOutput",
        "TodoWrite",
    }
)
# The input field naming each auto-allowable file tool's target. Glob and Grep default to the cwd.
PATH_FIELDS = {"Read": "file_path", "Glob": "path", "Grep": "path"}


def call_key(tool_name: str, tool_input: object) -> str:
    """A call's identity for the repeat-deny guard and the started-call ledger. Bash's
    `description` is prose the model rewrites on every retry, so it is not part of the call.
    """
    if isinstance(tool_input, dict):
        tool_input = {k: v for k, v in tool_input.items() if k != "description"}
    return json.dumps([tool_name, tool_input], sort_keys=True, separators=(",", ":"))


def in_cwd(cwd: object, tool_name: str, tool_input: object) -> bool:
    """Whether a file tool's target, symlinks resolved, is inside ``cwd``; True for other tools."""
    field = PATH_FIELDS.get(tool_name)
    if field is None:
        return True
    if not isinstance(cwd, str) or not isinstance(tool_input, dict):
        return False
    target = tool_input.get(field) or (cwd if tool_name != "Read" else None)
    pattern = tool_input.get("pattern") if tool_name == "Glob" else ""
    if not isinstance(target, str) or not isinstance(pattern, str):
        return False
    if os.path.isabs(os.path.expanduser(pattern)) or ".." in pattern.split("/"):
        return False
    root = os.path.realpath(cwd)
    path = os.path.realpath(os.path.join(root, os.path.expanduser(target)))
    return os.path.commonpath([root, path]) == root


def _valid_id(tool_use_id: object) -> bool:
    return (
        isinstance(tool_use_id, str)
        and tool_use_id not in ("", ".", "..")
        and not any(sep in tool_use_id for sep in ("/", "\\", "\0"))
    )


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_calls(state_dir: str) -> list[dict]:
    """The started-call ledger's entries; [] when there is none. Raises on anything malformed."""
    path = os.path.join(state_dir, CALLS_FILE)
    if not os.path.exists(path):
        return []
    data = _read_json(path)
    calls = (
        data["calls"] if isinstance(data, dict) and data.get("version") == 1 else None
    )
    if not isinstance(calls, list) or not all(
        isinstance(c, dict)
        and isinstance(c.get("tool_use_id"), str)
        and isinstance(c.get("key"), str)
        and isinstance(c.get("run"), (str, type(None)))
        and c.get("status") in ("started", "finished", "failed", "reported")
        for c in calls
    ):
        raise ValueError(f"malformed {CALLS_FILE}")
    return calls


def update_calls(state_dir: str, change) -> list[dict]:  # type: ignore
    """Apply ``change(calls)`` (it mutates the list) to the ledger atomically; return the result.
    Hooks of parallel calls serialize on the lock file.
    """
    # Imported here, not at module top: Windows has no fcntl, and the package must still import
    # there (native tools never run on it; sandbox_backend_available() is False off macOS/Linux).
    if sys.platform == "win32":
        raise OSError("native tools need POSIX file locks")
    import fcntl

    with open(os.path.join(state_dir, f".{CALLS_FILE}.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        calls = load_calls(state_dir)
        change(calls)
        path = os.path.join(state_dir, CALLS_FILE)
        fd, tmp = tempfile.mkstemp(
            dir=state_dir, prefix=f".{CALLS_FILE}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "calls": calls}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return calls


def _ledger_refusal(
    state_dir: str, tool_use_id: str, key: str, run: object
) -> str | None:
    """The deny reason when the started-call ledger forbids running (or asking about) this call
    now; marks an earlier run's started entry "reported" when a new call repeats it.
    """
    for call in load_calls(state_dir):
        if call["tool_use_id"] == tool_use_id and call["status"] in (
            "started",
            "finished",
            "failed",
        ):
            if call["status"] == "started":
                return SAME_RUN_STARTED if call.get("run") == run else UNKNOWN_OUTCOME
            return ALREADY_RAN
        if call.get("run") == run and call["status"] != "reported":
            continue
        if call["tool_use_id"] == tool_use_id:
            return (
                UNKNOWN_OUTCOME
                if call["status"] in ("started", "reported")
                else ALREADY_RAN
            )
        if call["key"] == key and call["status"] == "started":
            _mark(state_dir, call["tool_use_id"], status="reported", reported_run=run)
            return UNKNOWN_OUTCOME
        if (
            call["key"] == key
            and call["status"] == "reported"
            and call.get("reported_run") == run
        ):
            return UNKNOWN_OUTCOME
    return None


def _mark(state_dir: str, tool_use_id: str, **fields: object) -> None:
    def change(calls: list[dict]) -> None:
        for call in calls:
            if call["tool_use_id"] == tool_use_id:
                call.update(fields)

    update_calls(state_dir, change)


def decide(state_dir: str, data: dict) -> tuple[str, str, bool] | None:
    """``(permissionDecision, reason, is_rerequest)`` for one call; None makes no decision.

    A re-request is a call, other than the resumed one, made in a run that resumes a denial.
    An allowed call is recorded "started" in the ledger before this returns.
    """
    tool_name = data.get("tool_name") or ""
    if tool_name.startswith(MCP_PREFIX):
        return None  # pre-approved by name; run_tool is its one gate
    if tool_name == "StructuredOutput":
        return (
            None  # the synthetic tool that returns an output_format answer: not native
        )
    tool_use_id = data.get("tool_use_id")
    if not _valid_id(tool_use_id):
        return "deny", "native tool call without a usable tool_use_id", False
    if data.get("agent_id"):
        return "deny", "native tools inside Claude subagents are not supported", False
    if tool_name in NON_DURABLE_TOOLS:
        return "deny", f"{tool_name} cannot be made durable", False
    tool_input = data.get("tool_input")
    if (
        tool_name == "Bash"
        and isinstance(tool_input, dict)
        and tool_input.get("run_in_background")
    ):
        # A background shell outlives the run, so no ledger entry or revision covers its effect.
        return "deny", "background Bash cannot be made durable", False
    if (
        tool_name == "Bash"
        and isinstance(tool_input, dict)
        and tool_input.get("dangerouslyDisableSandbox")
    ):
        return "deny", "Bash runs only in the sandbox", False
    turn = _read_json(os.path.join(state_dir, "turn.json"))
    run = turn.get("run")
    key = call_key(tool_name, tool_input)
    decision_path = os.path.join(state_dir, "decisions", f"{tool_use_id}.json")
    if os.path.exists(decision_path):
        decision = _read_json(decision_path)
        if decision.get("decision") == "approved":
            if decision.get("key") != key:
                return "deny", "this call is not the one that was approved", False
            if refusal := _ledger_refusal(state_dir, tool_use_id, key, run):  # type: ignore
                return "deny", refusal, False
            return _allow(
                state_dir, tool_use_id, tool_name, key, run, "approved", False
            )
        return "deny", decision.get("reason") or "denied", False
    if tool_use_id == turn.get("resumed"):
        return "deny", "no decision was recorded for this call", False
    if tool_name not in turn["tools"]:
        return "deny", f"{tool_name} is not available for this message.", False
    rerequest = bool(turn.get("after_denial"))
    if rerequest:
        earlier = sum(
            1
            for e in _read_jsonl(os.path.join(state_dir, "ledger.jsonl"))
            if e["rerequest"]
        )
        if earlier >= int(turn.get("rerequest_budget", 0)):
            return "deny", OVER_CAP, True
    for denied in _read_jsonl(os.path.join(state_dir, "denied.jsonl")):
        if call_key(denied["tool_name"], denied["tool_input"]) == key:
            return "deny", REPEAT_DENIED, rerequest
    if refusal := _ledger_refusal(state_dir, tool_use_id, key, run):  # type: ignore
        return "deny", refusal, rerequest
    if tool_name in (turn.get("auto_allow") or ()) and in_cwd(
        data.get("cwd"), tool_name, tool_input
    ):
        return _allow(
            state_dir, tool_use_id, tool_name, key, run, "auto-allowed", rerequest
        )
    if any(
        e["decision"] == "defer"
        for e in _read_jsonl(os.path.join(state_dir, "ledger.jsonl"))
    ):
        # The CLI honors one deferral per run. A sibling it did not defer stays an unanswered
        # tool_use, and every resume re-defers a different one without running the approved call.
        return "deny", ONE_AT_A_TIME, False
    return "defer", "awaiting approval", rerequest


def _allow(
    state_dir: str,
    tool_use_id: object,
    tool_name: str,
    key: str,
    run: object,
    reason: str,
    rerequest: bool,
) -> tuple[str, str, bool]:
    """Record the call "started" in the ledger, then allow it."""
    entry = {
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "key": key,
        "run": run,
        "status": "started",
    }
    update_calls(state_dir, lambda calls: calls.append(entry))
    return "allow", reason, rerequest


def finish(state_dir: str, data: dict) -> None:
    """Record a PostToolUse ("finished") or PostToolUseFailure ("failed") for a ledger entry."""
    tool_name = data.get("tool_name") or ""
    tool_use_id = data.get("tool_use_id")
    if tool_name.startswith(MCP_PREFIX) or not _valid_id(tool_use_id):
        return
    if tool_name not in REPEATABLE_TOOLS:
        return
    failed = data.get("hook_event_name") == "PostToolUseFailure"
    _mark(state_dir, tool_use_id, status="failed" if failed else "finished")  # type: ignore


def main() -> int:
    """Read one hook event from stdin and print the verdict."""
    os.umask(0o077)
    state_dir = sys.argv[1]
    data = json.loads(sys.stdin.read())
    if data.get("hook_event_name") in ("PostToolUse", "PostToolUseFailure"):
        finish(state_dir, data)
        return 0
    if sys.platform == "win32":
        raise OSError("native tools need POSIX file locks")
    import fcntl  # see update_calls

    # Parallel calls' hooks decide one at a time, so only the first can defer.
    with open(os.path.join(state_dir, ".ledger.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verdict = decide(state_dir, data)
        if verdict is None:
            return 0
        decision, reason, rerequest = verdict
        entry = {
            "tool_use_id": data.get("tool_use_id"),
            "tool_name": data.get("tool_name"),
            "decision": decision,
            "reason": reason,
            "rerequest": rerequest,
        }
        with open(os.path.join(state_dir, "ledger.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }
    sys.stdout.write(json.dumps(output))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        # Exit 2 blocks the call. Any other failure is a non-blocking error on Claude Code 2.1.294:
        # it falls back to its own permission rules, which run reads in the cwd unasked.
        sys.stderr.write(f"native tool gate failed: {exc!r}\n")
        sys.exit(2)
