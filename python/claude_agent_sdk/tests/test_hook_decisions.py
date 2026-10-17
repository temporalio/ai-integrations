"""Every decision of the PreToolUse hook, against its rules written as a table.

The hook is what keeps a tool call from running twice: it defers the first call that
runs as an Activity, denies the others of the run, lets a tool step run exactly its
own call, and stops everything once the step is stopped or its Worker is gone. These
tests try every combination of what the hook can see (960 of them) and compare
the decision, the denial record and the paused-call file with ``expected``, which
states the rules of the module docstring in their order; ``EXAMPLES`` writes out by
hand the cases where rules meet.
"""

from __future__ import annotations

import builtins
import io
import itertools
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import _defer_hook as hook

CALL_ID = "toolu_this"
OTHER_ID = "toolu_other"
TOOLS = {
    "durable": hook.DURABLE_PREFIX + "refund",
    "activity": "Bash",  # in TCA_TOOL_ACTIVITIES below
    "matched": "mcp__github__create_issue",  # matches a pattern there
    "engine": "Read",  # runs inside the engine
}
ACTIVITY_PATTERNS = "Bash\n\nmcp__github__*\n"  # one per line; empty lines ignored


@dataclass(frozen=True)
class Seen:
    """What the hook can see for one call."""

    folder: str  # "unset" (no TCA_HOOK_DIR), "missing" (set, gone) or "present"
    stop: bool  # the runner wrote <folder>/stop
    worker: str  # "no lock file", "alive" (its lock is held) or "gone" (lock free)
    step: str  # "segment", or a tool step allowed to run "this" call or "another"
    answered: bool  # the call's result was just delivered (TCA_ANSWERED_IDS)
    subagent: bool  # the event carries an agent_id
    tool: str  # a key of TOOLS
    paused: str  # paused_call file: "none", "this" call or "another"


@dataclass(frozen=True)
class Decision:
    decision: str  # allow, defer, deny, or none (the engine's own permissions decide)
    reason: str | None = None  # for deny
    claims: bool = False  # this call takes the run's one paused-call slot


def deny(reason: str) -> Decision:
    return Decision("deny", reason)


def expected(s: Seen) -> Decision:
    """The rules of ``_defer_hook``, one by one."""
    present = s.folder == "present"
    # 1. A folder that is set but gone: the run ended, or the engine cannot see it.
    if s.folder == "missing":
        return deny(hook.STOPPED)
    stopped = present and (s.stop or s.worker == "gone")
    # 2. A tool step runs exactly its own call, unless the step stopped.
    if s.step != "segment":
        if stopped:
            return deny(hook.STOPPED)
        return Decision("allow") if s.step == "this" else deny(hook.STEP_ONLY)
    as_activity = s.tool != "engine"
    # 3. The main agent's call whose result was just delivered: the engine announces
    #    it again on resume. It never runs, and it never takes the slot.
    if s.answered and not s.subagent:
        return Decision("defer")
    # 4. A subagent cannot pause the run.
    if s.subagent and as_activity:
        return deny(hook.MAIN_AGENT_ONLY)
    # 5. A tool that runs in the engine runs, unless the step stopped or a call
    #    already paused this run.
    if s.subagent or not as_activity:
        if stopped:
            return deny(hook.STOPPED)
        if present and s.paused != "none":
            return deny(hook.NOT_RUN)
        return Decision("none")
    # 6. The main agent's call to a tool that runs as an Activity: the first one
    #    pauses the run (even a stopped one: that ends it, and no one runs the call);
    #    any other is denied, to be called again later.
    if present and s.paused == "another":
        return deny(hook.NOT_RUN)
    return Decision("defer", claims=present and s.paused == "none")


def every_case() -> Iterator[Seen]:
    for (
        folder,
        stop,
        worker,
        step,
        answered,
        subagent,
        tool,
        paused,
    ) in itertools.product(
        ("unset", "missing", "present"),
        (False, True),
        ("no lock file", "alive", "gone"),
        ("segment", "this", "another"),
        (False, True),
        (False, True),
        tuple(TOOLS),
        ("none", "this", "another"),
    ):
        if folder != "present" and (
            stop or worker != "no lock file" or paused != "none"
        ):
            continue  # nothing to see without the folder
        yield Seen(folder, stop, worker, step, answered, subagent, tool, paused)


CASES = list(every_case())


def test_every_decision_follows_the_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert len(CASES) == 2 * (3 * 2 * 2 * 4) + 2 * 3 * 3 * 2 * 2 * 4 * 3  # 960
    log = tmp_path / "hook.log"
    monkeypatch.setenv("TCA_TOOL_ACTIVITIES", ACTIVITY_PATTERNS)
    monkeypatch.setenv("TCA_HOOK_LOG", str(log))
    wrong: list[str] = []
    for n, seen in enumerate(CASES):
        run_dir = tmp_path / f"run{n}"
        held = arrange(seen, run_dir, monkeypatch)
        event: dict[str, Any] = {"tool_name": TOOLS[seen.tool], "tool_use_id": CALL_ID}
        if seen.subagent:
            event["agent_id"] = "agent_1"
        try:
            got = hook.decide(event)
        finally:
            if held is not None:
                release(held)
        want = expected(seen)
        problems = check(seen, want, got, run_dir)
        if problems:
            wrong.append(f"{seen}: {problems}")
    assert not wrong, f"{len(wrong)} of {len(CASES)} cases:\n" + "\n".join(wrong[:20])
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(CASES)  # the debugging log: one line per decision
    assert all(line.startswith(CALL_ID + " ") for line in lines)


def arrange(seen: Seen, run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> int | None:
    """Set up what the hook sees; return the Worker's lock to release, if held."""
    if seen.folder == "unset":
        monkeypatch.delenv("TCA_HOOK_DIR", raising=False)
    else:
        monkeypatch.setenv("TCA_HOOK_DIR", str(run_dir))
    if seen.step == "segment":
        monkeypatch.delenv("TCA_ALLOW_ID", raising=False)
    else:
        monkeypatch.setenv("TCA_ALLOW_ID", CALL_ID if seen.step == "this" else OTHER_ID)
    answered = [OTHER_ID, CALL_ID] if seen.answered else [OTHER_ID]
    monkeypatch.setenv("TCA_ANSWERED_IDS", " ".join(answered))
    if seen.folder != "present":
        return None
    run_dir.mkdir()
    if seen.stop:
        (run_dir / "stop").touch()
    if seen.paused != "none":
        (run_dir / "paused_call").write_text(
            CALL_ID if seen.paused == "this" else OTHER_ID, encoding="utf-8"
        )
    if seen.worker == "no lock file":
        return None
    lock = os.open(run_dir / hook.WORKER_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    if seen.worker == "gone":
        os.close(lock)
        return None
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(lock, msvcrt.LK_NBLCK, 1)  # the Worker, alive
    else:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX)  # the Worker, alive
    return lock


def release(lock: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(lock, msvcrt.LK_UNLCK, 1)
    os.close(lock)


def check(seen: Seen, want: Decision, got: dict[str, Any], run_dir: Path) -> list[str]:
    problems: list[str] = []
    decision = got.get("permissionDecision", "none")
    if got.get("hookEventName") != "PreToolUse":
        problems.append(f"event name {got.get('hookEventName')!r}")
    if decision != want.decision:
        problems.append(f"decision {decision}, want {want.decision}")
    if got.get("permissionDecisionReason") != want.reason:
        problems.append(f"reason {got.get('permissionDecisionReason')!r}")
    if seen.folder != "present":
        return problems
    record = run_dir / "denied" / hook.denial_name(CALL_ID)
    if want.decision == "deny":
        key = hook.REASON_KEYS[want.reason or ""]
        if not record.exists() or record.read_text(encoding="utf-8") != key:
            problems.append(f"no denial record {key!r}")
    elif record.exists():
        problems.append("a denial record for a call it did not deny")
    marker = run_dir / "paused_call"
    before = {"none": None, "this": CALL_ID, "another": OTHER_ID}[seen.paused]
    after = marker.read_text(encoding="utf-8") if marker.exists() else None
    if after != (CALL_ID if want.claims else before):
        problems.append(f"paused_call {after!r}")
    return problems


EXAMPLES: list[tuple[Seen, Decision, str]] = [
    (
        Seen(
            "present", True, "no lock file", "segment", False, False, "engine", "none"
        ),
        deny(hook.STOPPED),
        "a stopped step starts no more built-in tools",
    ),
    (
        Seen(
            "present", True, "no lock file", "segment", False, False, "durable", "none"
        ),
        Decision("defer", claims=True),
        "a durable call still defers in a stopped step: that ends the run, and no "
        "one runs it",
    ),
    (
        Seen("present", False, "gone", "segment", False, False, "engine", "none"),
        deny(hook.STOPPED),
        "once the Worker is gone, the step is stopped",
    ),
    (
        Seen(
            "present", True, "no lock file", "segment", False, True, "durable", "none"
        ),
        deny(hook.MAIN_AGENT_ONLY),
        "a subagent's durable call is denied with the hint, stopped or not",
    ),
    (
        Seen(
            "present", True, "no lock file", "segment", True, False, "durable", "none"
        ),
        Decision("defer"),
        "a delivered call, announced again on resume, never runs and takes no slot",
    ),
    (
        Seen("present", False, "gone", "this", False, False, "activity", "none"),
        deny(hook.STOPPED),
        "a tool step whose Worker is gone runs nothing, not even its own call",
    ),
    (
        Seen("present", False, "alive", "this", False, False, "activity", "none"),
        Decision("allow"),
        "a tool step runs exactly its own call",
    ),
    (
        Seen("present", False, "alive", "another", False, False, "activity", "none"),
        deny(hook.STEP_ONLY),
        "and nothing else",
    ),
    (
        Seen(
            "missing", False, "no lock file", "this", False, False, "activity", "none"
        ),
        deny(hook.STOPPED),
        "without its folder the hook denies everything",
    ),
    (
        Seen("present", False, "alive", "segment", False, False, "matched", "another"),
        deny(hook.NOT_RUN),
        "after the pause, another call that runs as an Activity waits its turn",
    ),
    (
        Seen("present", False, "alive", "segment", False, False, "engine", "another"),
        deny(hook.NOT_RUN),
        "after the pause, a built-in call would be cut from the session: denied too",
    ),
    (
        Seen("present", False, "alive", "segment", False, True, "engine", "another"),
        deny(hook.NOT_RUN),
        "a subagent's built-in call after the pause too",
    ),
    (
        Seen("present", False, "alive", "segment", False, False, "durable", "this"),
        Decision("defer"),
        "the paused call itself, announced again, keeps its slot",
    ),
    (
        Seen(
            "unset", False, "no lock file", "segment", False, False, "durable", "none"
        ),
        Decision("defer"),
        "with no folder set (an older runner), durable calls still defer",
    ),
    (
        Seen("present", False, "alive", "segment", False, True, "engine", "none"),
        Decision("none"),
        "a subagent's built-in call in a running step follows Claude Code's rules",
    ),
]
"""Cases where rules meet, each written out by hand from the docs."""


def slug(why: str) -> str:
    """A test id a shell takes as it is (no quotes or commas)."""
    return re.sub(r"[^a-z0-9]+", "-", why.lower()).strip("-")[:60]


@pytest.mark.parametrize(
    ("seen", "want", "why"), EXAMPLES, ids=[slug(e[2]) for e in EXAMPLES]
)
def test_where_rules_meet_the_hook_decides_as_documented(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seen: Seen,
    want: Decision,
    why: str,
) -> None:
    del why
    monkeypatch.setenv("TCA_TOOL_ACTIVITIES", ACTIVITY_PATTERNS)
    monkeypatch.delenv("TCA_HOOK_LOG", raising=False)
    run_dir = tmp_path / "run"
    held = arrange(seen, run_dir, monkeypatch)
    event: dict[str, Any] = {"tool_name": TOOLS[seen.tool], "tool_use_id": CALL_ID}
    if seen.subagent:
        event["agent_id"] = "agent_1"
    try:
        got = hook.decide(event)
    finally:
        if held is not None:
            release(held)
    assert check(seen, want, got, run_dir) == []
    assert expected(seen) == want  # the table above agrees


def test_an_unusual_id_is_recorded_under_a_safe_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "paused_call").parent.mkdir()
    (run_dir / "paused_call").write_text(OTHER_ID, encoding="utf-8")
    monkeypatch.setenv("TCA_HOOK_DIR", str(run_dir))
    for name in ("TCA_ALLOW_ID", "TCA_ANSWERED_IDS", "TCA_HOOK_LOG"):
        monkeypatch.delenv(name, raising=False)
    odd = "../../toolu id"
    got = hook.decide({"tool_name": TOOLS["durable"], "tool_use_id": odd})
    assert got["permissionDecision"] == "deny"
    assert [p.name for p in (run_dir / "denied").iterdir()] == [hook.denial_name(odd)]


def test_a_denial_is_still_answered_when_its_record_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The folder went away between the check and the write: deny anyway."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stop").touch()
    (run_dir / "denied").write_text("not a folder", encoding="utf-8")
    monkeypatch.setenv("TCA_HOOK_DIR", str(run_dir))
    for name in ("TCA_ALLOW_ID", "TCA_ANSWERED_IDS", "TCA_HOOK_LOG"):
        monkeypatch.delenv(name, raising=False)
    got = hook.decide({"tool_name": "Read", "tool_use_id": CALL_ID})
    assert got["permissionDecision"] == "deny"
    assert got["permissionDecisionReason"] == hook.STOPPED


def test_an_input_digest_takes_any_input_and_ignores_the_order_of_keys() -> None:
    first = hook.input_digest({"content": "a\ud83d b \u00e9", "file_path": "/f"})
    again = hook.input_digest({"file_path": "/f", "content": "a\ud83d b \u00e9"})
    assert first == again and len(first) == 64
    assert (
        hook.input_digest({"content": "a\ud83d b \u00e8", "file_path": "/f"}) != first
    )


@pytest.mark.parametrize("given", ["the same input", "other input"])
def test_a_tool_step_runs_its_call_only_with_the_input_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, given: str
) -> None:
    """With ``TCA_ALLOW_INPUT``, the step's call runs only with that input, as Claude
    Code is about to run it (the order of its keys does not matter). With other input
    it is denied, and never takes the ``allowed`` record, so it did not run."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("TCA_HOOK_DIR", str(run_dir))
    monkeypatch.setenv("TCA_ALLOW_ID", CALL_ID)
    approved = {"command": "ls", "description": "list"}
    monkeypatch.setenv("TCA_ALLOW_INPUT", hook.input_digest(approved))
    for name in ("TCA_ANSWERED_IDS", "TCA_HOOK_LOG"):
        monkeypatch.delenv(name, raising=False)
    runs = (
        {"description": "list", "command": "ls"}
        if given == "the same input"
        else {**approved, "command": "rm -rf ."}
    )
    got = hook.decide({"tool_name": "Bash", "tool_use_id": CALL_ID, "tool_input": runs})
    if given == "the same input":
        assert got == {"hookEventName": "PreToolUse", "permissionDecision": "allow"}
        assert (run_dir / hook.ALLOWED).read_text(encoding="utf-8") == CALL_ID
    else:
        assert got["permissionDecision"] == "deny"
        assert got["permissionDecisionReason"] == hook.OTHER_INPUT
        assert not (run_dir / hook.ALLOWED).exists()
        assert (run_dir / "denied" / CALL_ID).read_text(encoding="utf-8") == (
            "other_input"
        )


def hook_env(run_dir: Path, **more: str) -> dict[str, str]:
    """The environment of a hook process for ``run_dir``, as a tool step's own."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in ("TCA_ALLOW_ID", "TCA_ALLOW_INPUT", "TCA_ANSWERED_IDS", "TCA_HOOK_LOG")
    }
    return {**env, "TCA_HOOK_DIR": str(run_dir), **more}


IN_A_THREAD = """\
import runpy, sys, threading
threading.stack_size(2**24)
thread = threading.Thread(
    target=runpy.run_path, args=(sys.argv[1],), kwargs={"run_name": "__main__"}
)
thread.start()
thread.join()
"""
"""Runs the script ``sys.argv[1]`` in a thread with a stack of 16 MiB.

From Python 3.14, how deep ``json`` reads depends on the stack size (in the main
thread of GitHub's Linux and macOS runners it read 100,000 levels), and 1,000,000
levels do not fit in 16 MiB. Before 3.14 the limit is a fixed count, far below that.
"""


@pytest.mark.parametrize("stdin", ["nested too deep", "not JSON"])
def test_a_hook_that_cannot_read_its_event_denies_the_call(
    tmp_path: Path, stdin: str
) -> None:
    """With no answer, Claude Code's own permission check would decide, and a rule
    from a settings file could let the call run. So a hook that fails denies. Here it
    cannot even read the event (as the runner starts it: isolated, no ``site``; the
    deep event in a thread of a known stack size, see ``IN_A_THREAD``)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    approved = {"command": "ls"}
    env = hook_env(
        run_dir, TCA_ALLOW_ID=CALL_ID, TCA_ALLOW_INPUT=hook.input_digest(approved)
    )
    command = [sys.executable, "-I", "-S", str(HOOK)]
    if stdin == "nested too deep":
        deep = "[" * 1_000_000 + "]" * 1_000_000
        data = (
            '{"tool_name": "Bash", "tool_use_id": "%s", "tool_input": '
            '{"command": "ls", "x": %s}}' % (CALL_ID, deep)
        )
        command = [sys.executable, "-I", "-S", "-c", IN_A_THREAD, str(HOOK)]
        read = tmp_path / "read.py"
        read.write_text("import json, sys\njson.load(sys.stdin)\n", encoding="utf-8")
        probe = subprocess.run(
            [sys.executable, "-I", "-S", "-c", IN_A_THREAD, str(read)],
            input=data.encode(),
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert b"RecursionError" in probe.stderr, (
            f"json did not fail with RecursionError here (exit {probe.returncode}): "
            f"{probe.stderr[-300:]!r}"
        )
    else:
        data = "{not json"
    done = subprocess.run(
        command,
        input=data.encode("utf-8"),
        capture_output=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert out["permissionDecisionReason"] == hook.FAILED
    assert not (run_dir / hook.ALLOWED).exists()  # never let run


def test_a_hook_that_fails_while_it_decides_denies_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The event was read, so the denial is recorded under the call's id: the runner
    knows the call did not run without reading the tool output."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("TCA_HOOK_DIR", str(run_dir))
    monkeypatch.setenv("TCA_ALLOW_ID", CALL_ID)
    monkeypatch.setenv("TCA_ALLOW_INPUT", "x")
    for name in ("TCA_ANSWERED_IDS", "TCA_HOOK_LOG"):
        monkeypatch.delenv(name, raising=False)

    def fails(tool_input: Any) -> str:
        del tool_input
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(hook, "input_digest", fails)
    event = {"tool_name": "Bash", "tool_use_id": CALL_ID, "tool_input": {}}
    stdin = io.TextIOWrapper(io.BytesIO(json.dumps(event).encode("utf-8")))
    monkeypatch.setattr(sys, "stdin", stdin)
    hook.main()
    out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert out["permissionDecisionReason"] == hook.FAILED
    assert (run_dir / "denied" / CALL_ID).read_text(encoding="utf-8") == "failed"
    assert not (run_dir / hook.ALLOWED).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="a closed pipe is POSIX-specific")
def test_a_hook_that_cannot_write_its_answer_exits_with_2(tmp_path: Path) -> None:
    """Exit code 2 is Claude Code's other way to deny a call."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    read_end, write_end = os.pipe()
    os.close(read_end)  # nobody reads the answer: writing it fails
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", str(HOOK)],
            stdin=subprocess.PIPE,
            stdout=write_end,
            stderr=subprocess.DEVNULL,
            env=hook_env(run_dir),
        )
    finally:
        os.close(write_end)
    event = {"tool_name": "Read", "tool_use_id": CALL_ID}
    process.communicate(json.dumps(event).encode("utf-8"), timeout=60)
    assert process.returncode == 2


def test_a_debug_log_that_cannot_be_written_changes_no_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("TCA_HOOK_DIR", str(run_dir))
    monkeypatch.setenv("TCA_ALLOW_ID", CALL_ID)
    monkeypatch.setenv("TCA_HOOK_LOG", str(tmp_path))  # a folder: it cannot be opened
    for name in ("TCA_ALLOW_INPUT", "TCA_ANSWERED_IDS"):
        monkeypatch.delenv(name, raising=False)
    got = hook.decide({"tool_name": "Bash", "tool_use_id": CALL_ID, "tool_input": {}})
    assert got == {"hookEventName": "PreToolUse", "permissionDecision": "allow"}


class Interleaved:
    """``os`` and ``open`` for the hook, where another call's hook claims the paused
    call's slot just before this one tries: the moment two hooks can collide."""

    def __init__(self, other: dict[str, Any]) -> None:
        self.other = other
        self.decisions: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def _other_goes_first(self, path: Any) -> None:
        if str(path).endswith("paused_call") and self.other:
            other, self.other = self.other, {}
            self.decisions.append(hook.decide(other))

    def open(self, path: Any, flags: int, *args: Any) -> int:
        if flags & os.O_CREAT:
            self._other_goes_first(path)
        return os.open(path, flags, *args)

    def builtin_open(
        self, path: Any, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if any(c in mode for c in "wxa"):
            self._other_goes_first(path)
        return builtins.open(path, mode, *args, **kwargs)


def test_two_calls_that_reach_the_slot_together_never_both_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The slot is claimed in one step (create-if-absent): the call that comes second
    is denied, even when it checked before the first one wrote."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("TCA_HOOK_DIR", str(run_dir))
    for name in ("TCA_ALLOW_ID", "TCA_ANSWERED_IDS", "TCA_HOOK_LOG"):
        monkeypatch.delenv(name, raising=False)
    second = {"tool_name": TOOLS["durable"], "tool_use_id": OTHER_ID}
    interleaved = Interleaved(second)
    monkeypatch.setattr(hook, "os", interleaved)
    monkeypatch.setattr(hook, "open", interleaved.builtin_open, raising=False)
    got = hook.decide({"tool_name": TOOLS["durable"], "tool_use_id": CALL_ID})
    assert [d["permissionDecision"] for d in interleaved.decisions] == ["defer"]
    assert got["permissionDecision"] == "deny"
    assert got["permissionDecisionReason"] == hook.NOT_RUN
    assert (run_dir / "paused_call").read_text(encoding="utf-8") == OTHER_ID


HOOK = Path(hook.__file__)


def test_exactly_one_of_many_calls_at_once_pauses_the_run(tmp_path: Path) -> None:
    """Engines check read-only calls of one message at the same time, each in its own
    hook process: one call defers and every other one is denied and recorded. (Real
    processes rarely collide; the test above forces the collision.)"""
    for round_ in range(5):
        run_dir = tmp_path / f"run{round_}"
        run_dir.mkdir()
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("TCA_ALLOW_ID", "TCA_ANSWERED_IDS", "TCA_HOOK_LOG")
        }
        env["TCA_HOOK_DIR"] = str(run_dir)
        ids = [f"toolu_{round_}_{i}" for i in range(8)]
        hooks = [
            subprocess.Popen(
                [sys.executable, str(HOOK)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=env,
            )
            for _ in ids
        ]
        events = [{"tool_name": TOOLS["durable"], "tool_use_id": i} for i in ids]
        for process, event in zip(hooks, events):
            assert process.stdin is not None
            process.stdin.write(json.dumps(event).encode("utf-8"))
            process.stdin.close()
        decisions = {}
        for process, call in zip(hooks, ids):
            assert process.stdout is not None
            out = json.loads(process.stdout.read())["hookSpecificOutput"]
            assert process.wait(timeout=30) == 0
            decisions[call] = out["permissionDecision"]
        winners = [c for c, d in decisions.items() if d == "defer"]
        assert len(winners) == 1, decisions
        assert sorted(d for d in decisions.values() if d != "defer") == ["deny"] * 7
        assert (run_dir / "paused_call").read_text(encoding="utf-8") == winners[0]
        denied = sorted(p.name for p in (run_dir / "denied").iterdir())
        assert denied == sorted(c for c in ids if c != winners[0])
