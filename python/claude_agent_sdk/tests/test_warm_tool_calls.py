"""Warm tool calls: a Claude Code tool call that runs as its own Activity waits in its
warm engine for its tool step.

With ``warm_engines`` on, when a segment pauses at a call of a Claude Code tool in
``tool_activities`` (Bash, Edit, Write, an MCP tool), the engine stays on the Worker
with the call waiting in its hook. The call's tool step, still its own Activity, lets
it run there when it comes to the same Worker; the turn then waits right after the
call for the next segment. Anything else (another Worker, a retry, a wait longer than
``warm_seconds``, a stopped or killed Worker) ends the warm engine: the call never
runs there, and the step or the next segment runs as without warm engines.

The tests run the real Claude Code engine with the local fake model, in real Workflows,
and compare each warm run with the same run cold: the same files, each command and
edit once, the same model requests, and the same total cost. Requests are compared
with ids and paths left out, and without the line Claude Code adds when it continues
a turn by itself ("Continue from where you left off."): a new engine that continues
after a tool step's result adds it, an engine that never stopped does not. Claude
Code's own notes (``NOTES``, ``RESULT_NOTES``) and bash's job control warning
(``JOB_CONTROL``) are left out too.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    DeferredCall,
    FileSessionStore,
    SegmentInput,
    ToolSpec,
    ToolStepInput,
    _defer_hook,
    _runner,
)
from temporalio.client import Client
from temporalio.exceptions import ApplicationError
from temporalio.worker import Worker
from tests.endless.activities import ALL
from tests.engine_tools.workflows import ShellOptions, ShellWorkflow
from tests.helpers.fake_messages_api import (
    SLOW_MODEL_SECONDS,
    FakeMessagesAPI,
    engine_env,
    history_of,
)
from tests.helpers.processes import alive, descendants
from tests.helpers.workers import FAIL_FAST
from tests.test_chaos import kill, running_on, start_worker
from tests.test_crash import wait_until
from tests.test_engine_tools import posix
from tests.test_warm_engines import all_ended, engines_of, still_running

pytestmark = [pytest.mark.timeout(300), pytest.mark.usefixtures("shop_dir")]


def activity_calls_of(steps: Any, mode: str) -> int:
    """How many Activity calls of a flow of single calls wait in a warm engine: all but
    the first with the conversation in the Workflow (see ``activity_calls``)."""
    count = sum(1 for name, _ in steps(Path("work")) if name in TOOLS.tool_activities)
    return count if mode == "store" else count - 1


def activity_calls(mode: str) -> int:
    """How many of ``mixed``'s Activity calls wait in a warm engine.

    With the conversation in the Workflow, the engine that starts it never stays warm
    (it writes the conversation to its own folder), so the first call runs cold.
    """
    return 5 if mode == "store" else 4


TOOLS = ShellOptions(
    builtin_tools=["Read", "Edit", "Bash", "Write"],
    tool_activities=["Edit", "Bash", "Write"],
    tool_timeout=60,
)
RESUME_LINE = "Continue from where you left off."
NOTES = (
    "changed on disk since you last read it",
    "hasn't heard from you in a while",
)
"""Claude Code's own notes in a turn: that a file changed on disk since Claude read
it, and (on newer Claude Code versions) that the user has not heard from Claude for
several model calls. A warm engine's turn is one Claude Code run, so it can have notes
that a new engine for each step does not."""
RESULT_NOTES = (
    " (file state is current in your context \u2014 no need to Read it back)",
)
"""Claude Code's notes at the end of a result (on some Claude Code versions): an Edit's
or a Write's file is current in Claude's context (the engine that ran it has read
it)."""
JOB_CONTROL = re.compile(r"(?m)^(?:\S*/)?bash: child setpgid \(\d+ to \d+\): [^\n]*\n?")
"""Bash's warning that job control could not give a command its own process group,
which bash on macOS prints now and then. Claude Code's shell snapshot turns job
control on in its commands whenever ~/.bashrc exists (``set -o monitor``,
anthropics/claude-code#19057), so the line can come with any command, in a warm or in
a new engine. It is not the command's output: runs are compared without it."""
NO_OUTPUT = "(Bash completed with no output)"
"""What Claude Code says for a command that printed nothing."""

Step = tuple[str, dict[str, Any]]


def mixed(work: Path) -> list[Step]:
    """Read, Edit, Bash, a durable call, Write, an Edit of the same file, Bash."""
    notes, log, out = work / "notes.txt", work / "runs.log", work / "out.txt"
    return [
        ("Read", {"file_path": str(notes)}),
        ("Edit", {"file_path": str(notes), "old_string": "old", "new_string": "new"}),
        (
            "Bash",
            {"command": f"cat {posix(notes)} >> {posix(log)}", "description": "log"},
        ),
        ("count", {"n": 1}),
        ("Write", {"file_path": str(out), "content": "x\n"}),
        ("Edit", {"file_path": str(notes), "old_string": "new", "new_string": "newer"}),
        (
            "Bash",
            {
                "command": f"cat {posix(notes)} >> {posix(log)}; exit 3",
                "description": "fail",
            },
        ),
    ]


def commands(work: Path) -> list[Step]:
    """Four commands, one after the other."""
    log = work / "runs.log"
    return [
        ("Bash", {"command": f"echo {n} >> {posix(log)}", "description": f"log {n}"})
        for n in range(1, 5)
    ]


def changed_after_read(work: Path) -> list[Step]:
    """An Edit before a Read, then Edits of a file a command changed after Claude read
    and edited it, and a Write over it."""
    notes = work / "notes.txt"
    return [
        ("Edit", {"file_path": str(notes), "old_string": "old", "new_string": "x"}),
        ("Read", {"file_path": str(notes)}),
        ("Edit", {"file_path": str(notes), "old_string": "old", "new_string": "new"}),
        ("Bash", {"command": f"echo extra >> {posix(notes)}", "description": "change"}),
        ("Edit", {"file_path": str(notes), "old_string": "new", "new_string": "newer"}),
        ("Bash", {"command": f"echo more >> {posix(notes)}", "description": "change"}),
        ("Write", {"file_path": str(notes), "content": "over\n"}),
    ]


def scripted(steps: list[Step] | list[list[Step]]) -> FakeMessagesAPI:
    """Claude makes one message per item (a call, or several calls), then says DONE.

    A ``("text", {"text": ...})`` item is a text block of the message, not a call.
    """
    holder: list[FakeMessagesAPI] = []
    messages = [s if isinstance(s, list) else [s] for s in steps]

    def calls(message: list[Step]) -> int:
        return sum(1 for name, _ in message if name != "text")

    def block(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "text":
            return {"type": "text", "text": args["text"]}
        return holder[0].call(name, args)

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        done, turn = len(history), 0
        for message in messages:
            if done < calls(message):
                break
            done -= calls(message)
            turn += 1
        if turn >= len(messages):
            return [{"type": "text", "text": "DONE"}]
        return [block(name, args) for name, args in messages[turn]]

    api = FakeMessagesAPI(decide)
    holder.append(api)
    return api.start()


class Tracking(ClaudeAgentSdkRunner):
    """Counts new engines and warm work, and can make a warm engine miss its step or
    its next segment (as on another Worker), or make steps and segments come late.

    ``miss_step`` applies to every step that finds its call waiting, or with
    ``miss_only`` to the n-th such step only."""

    def __init__(
        self,
        *,
        miss_step: str = "",
        miss_only: int = 0,
        miss_next: bool = False,
        step_delay: float = 0.0,
        segment_delay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.miss_step = miss_step
        self.miss_only = miss_only
        self.found = 0
        self.miss_next = miss_next
        self.step_delay = step_delay
        self.segment_delay = segment_delay
        self.cold = 0
        self.warm_segments = 0
        self.warm_steps = 0
        self.committed: list[dict[str, Any]] = []
        self.waited: list[int] = []
        """The process ids of engines in which a call waited for its step."""

    async def run(self, inp: SegmentInput, attempt: int) -> Any:
        if inp.injected and self.segment_delay:
            await asyncio.sleep(self.segment_delay)
        out = await super().run(inp, attempt)
        self.committed += out.transcript_add
        return out

    async def run_tool_step(self, step: ToolStepInput, attempt: int) -> Any:
        if self.step_delay:
            await asyncio.sleep(self.step_delay)
        return await super().run_tool_step(step, attempt)

    async def _run_engine(self, *args: Any, warm: Any = None, **kwargs: Any) -> Any:
        if warm is None:
            self.cold += 1
        else:
            self.warm_segments += 1
        return await super()._run_engine(*args, warm=warm, **kwargs)

    def _take_waiting(self, step: ToolStepInput) -> Any:
        warm = super()._take_waiting(step)
        if warm is not None:
            self.found += 1
            if self.miss_only and self.found != self.miss_only:
                return warm
        if warm is not None and self.miss_step == "end":
            self._end_warm(warm)  # the step runs on another Worker
            return None
        if warm is not None and self.miss_step == "alive":
            # The step runs on another Worker while this engine still waits, until
            # the next segment here (or warm_seconds) ends it.
            super()._park(warm)
            return None
        return warm

    async def _step_in_warm(self, step: Any, warm: Any, allowed: list[bool]) -> Any:
        self.warm_steps += 1
        return await super()._step_in_warm(step, warm, allowed)

    def _take_warm(self, inp: SegmentInput, injected: Any, attempt: int) -> Any:
        warm = super()._take_warm(inp, injected, attempt)
        if warm is not None and warm.state == "turn_waits" and self.miss_next:
            self._end_warm(warm)  # the next segment runs on another Worker
            return None
        return warm

    def _park(self, warm: Any) -> None:
        if warm.state == "call_waits":
            self.waited.append(warm.client._transport._process.pid)
        super()._park(warm)


@dataclass
class Run:
    answer: str
    seen: list[Any]
    """What the model saw in the last request (see ``conversation``)."""
    requests: int
    """The model requests that offered the durable tools (Claude's own turns)."""
    files: dict[str, str]
    cost: float
    calls: list[dict[str, Any]]
    runner: Tracking
    notes: list[str]
    """Claude Code's own notes in the last request (``NOTES``)."""
    committed: list[dict[str, Any]]
    """Every entry the segments added to the conversation the Workflow holds."""
    root: Path
    """The run's own folder (left out when runs are compared)."""


def scrub(text: str, root: Path) -> str:
    """``text`` with a run's folder left out, in each form it takes: inside JSON, as
    the system writes it, and with forward slashes (as in Bash commands). On Windows
    these differ: backslashes, doubled in JSON, or forward slashes."""
    for form in (json.dumps(str(root))[1:-1], str(root), root.as_posix()):
        text = text.replace(form, "<ROOT>")
    return text


def conversation(body: dict[str, Any], root: Path) -> tuple[list[Any], list[str]]:
    """The messages of a request without ids, paths, Claude Code's resume line, its
    running token count and its notes; and those notes (``NOTES``)."""
    out: list[Any] = []
    notes: list[str] = []
    for message in body.get("messages", []):
        content = message.get("content")
        blocks = (
            content
            if isinstance(content, list)
            else [{"type": "text", "text": content}]
        )
        kept: list[Any] = []
        for block in blocks:
            kind = block.get("type")
            if kind == "text":
                text = re.sub(
                    r"<total_tokens>\d+ tokens left</total_tokens>", "", block["text"]
                )
                text = scrub(text, root)
                if any(note in text for note in NOTES):
                    notes.append(text.strip())
                    continue
                if text.strip() and text.strip() != RESUME_LINE:
                    kept.append(("text", text.strip()))
            elif kind == "tool_use":
                kept.append(
                    (
                        "call",
                        block["name"],
                        scrub(json.dumps(block["input"], sort_keys=True), root),
                    )
                )
            elif kind == "tool_result":
                raw = block.get("content")
                if isinstance(raw, list):
                    raw = "".join(b.get("text", "") for b in raw if isinstance(b, dict))
                raw = scrub(re.sub(r"toolu_[A-Za-z0-9]+", "<ID>", str(raw)), root)
                quiet = JOB_CONTROL.sub("", raw)
                if quiet != raw:
                    notes.append("bash: child setpgid")
                    raw = quiet.rstrip("\n") or NO_OUTPUT
                for note in RESULT_NOTES:
                    if note in raw:
                        notes.append(note.strip())
                        raw = raw.replace(note, "")
                kept.append(("result", raw, bool(block.get("is_error"))))
        if kept:
            out.append((message.get("role"), kept))
    return out, notes


async def run(
    client: Client,
    tmp_path: Path,
    name: str,
    steps: Any,
    *,
    mode: str,
    warm: bool,
    options: ShellOptions = TOOLS,
    pause: float = 0.0,
    **knobs: Any,
) -> Run:
    """Run one task in a Workflow on one Worker, in a folder of its own.

    ``pause``: seconds the model takes after each call before the rest of its answer.
    """
    root = tmp_path / name
    work = root / "work"
    work.mkdir(parents=True)
    # One line ending on every system (Windows would write "\r\n" in text mode).
    (work / "notes.txt").write_text("old\n", encoding="utf-8", newline="\n")
    api = scripted(steps(work))
    api.pause_after_call = pause
    queue = f"warmcalls-{name}-{uuid.uuid4().hex[:8]}"
    runner = Tracking(
        session_store=FileSessionStore(root / "store") if mode == "store" else None,
        cwd=str(work),
        env=engine_env(api, str(root / "cfg")),
        warm_engines=4 if warm else 0,
        **knobs,
    )
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[ShellWorkflow],
            activities=ALL,
            plugins=[ClaudeAgentPlugin(runner)],
            **FAIL_FAST,
        ):
            handle = await client.start_workflow(
                ShellWorkflow.run, args=["go", options], id=queue, task_queue=queue
            )
            answer = await asyncio.wait_for(handle.result(), 240)
            cost = await handle.query(ShellWorkflow.total_cost_usd)
            calls = await handle.query(ShellWorkflow.tool_calls)
        await all_ended()
    finally:
        api.stop()
    assert api.errors == []
    turns = [
        b
        for b in api.requests
        if any(t["name"].startswith("mcp__durable__") for t in b.get("tools", []))
    ]
    seen, notes = conversation(turns[-1], root)
    files = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(work.iterdir())
        if p.is_file()
    }
    return Run(
        answer,
        seen,
        len(turns),
        files,
        cost,
        calls,
        runner,
        notes,
        runner.committed,
        root,
    )


def same(warm: Run, cold: Run) -> None:
    """The warm run did what the cold one did, and Claude saw the same."""
    assert warm.answer == cold.answer == "DONE"
    assert warm.files == cold.files
    assert warm.seen == cold.seen
    assert warm.requests == cold.requests
    assert warm.cost == pytest.approx(cold.cost, rel=1e-9, abs=1e-12)
    assert [c.get("status") for c in warm.calls] == [
        c.get("status") for c in cold.calls
    ]
    assert inputs(warm) == inputs(cold)
    for each in (warm, cold):  # no record of a step's call carries a copy of a file
        assert not [
            e for e in each.committed if "toolUseResult" in e and answers_step(e, each)
        ]


def inputs(run: Run) -> list[str]:
    """The inputs of a run's calls, as the Workflow got them, its folder left out."""
    return [
        scrub(json.dumps(c.get("input"), sort_keys=True), run.root) for c in run.calls
    ]


def answers_step(entry: dict[str, Any], run: Run) -> bool:
    """Whether a conversation entry is the result of a call that ran in a tool step."""
    steps = {c["id"] for c in run.calls if c.get("name") in ("Edit", "Write", "Bash")}
    return bool(steps.intersection(_runner._result_ids(entry)))  # type: ignore[reportPrivateUsage]


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_calls_run_in_their_warm_engine_with_the_same_results(
    client: Client, tmp_path: Path, mode: str
) -> None:
    """One engine runs the whole task: every Claude Code call waits for its step and
    runs there, each once; files, requests and cost are those of the cold run."""
    cold = await run(client, tmp_path, "cold", mixed, mode=mode, warm=False)
    warm = await run(client, tmp_path, "warm", mixed, mode=mode, warm=True)
    same(warm, cold)
    assert cold.files["runs.log"] == "new\nnewer\n"  # each command ran once
    assert warm.runner.warm_steps == activity_calls(mode)  # Edit, Bash, Write, ...
    assert warm.runner.cold == (1 if mode == "store" else 2)  # the rest ran warm
    changed = [n for n in warm.notes + cold.notes if NOTES[0] in n]
    assert changed == []  # no file changed on disk after a read


async def ended(engines: list[int]) -> None:
    """Wait for these engines to be gone: the runner ends one in the background, and
    ``terminate`` returns before the process is gone (on Windows, a while before)."""
    await wait_until(lambda: not any(alive(pid) for pid in engines), timeout=15)


@pytest.mark.parametrize("miss", ["step", "step-alive", "next", "both"])
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_call_that_misses_its_engine_runs_once(
    client: Client, tmp_path: Path, mode: str, miss: str
) -> None:
    """The step, the next segment, or both run on another Worker: the waiting engine
    ends (its call never runs there), and the task goes on as without warm engines.
    With ``step-alive``, the step runs elsewhere while the engine where its call waits
    still lives (until the next segment here ends it)."""
    cold = await run(client, tmp_path, "cold", mixed, mode=mode, warm=False)
    warm = await run(
        client,
        tmp_path,
        "warm",
        mixed,
        mode=mode,
        warm=True,
        miss_step={"step": "end", "step-alive": "alive", "both": "end"}.get(miss, ""),
        miss_next=miss in ("next", "both"),
    )
    same(warm, cold)
    assert warm.runner.warm_steps == (0 if miss != "next" else activity_calls(mode))
    await ended(warm.runner.waited)


@pytest.mark.parametrize("miss", ["end", "alive"])
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_miss_right_after_a_hit_asks_claude_nothing_more(
    client: Client, tmp_path: Path, mode: str, miss: str
) -> None:
    """A command runs warm and the next segment lets its turn go on there; then the
    next command's step runs on another Worker. The engine's turn stops at that
    command, as after any miss: the go the turn got before is spent, so no model
    request comes from the engine left behind (same requests and cost as cold)."""
    cold = await run(client, tmp_path, "cold", commands, mode=mode, warm=False)
    warm = await run(
        client,
        tmp_path,
        "warm",
        commands,
        mode=mode,
        warm=True,
        miss_step=miss,
        miss_only=2,
    )
    same(warm, cold)
    assert cold.files["runs.log"] == "1\n2\n3\n4\n"
    assert warm.runner.found == activity_calls_of(commands, mode)
    assert warm.runner.warm_steps == warm.runner.found - 1


async def test_claude_code_notes_a_file_changed_after_a_read(
    client: Client, tmp_path: Path
) -> None:
    """Edits and a Write of a file that commands changed after Claude read and edited
    it come out the same warm and cold (Claude Code refuses some of them in both)."""
    cold = await run(
        client, tmp_path, "cold", changed_after_read, mode="held", warm=False
    )
    warm = await run(
        client, tmp_path, "warm", changed_after_read, mode="held", warm=True
    )
    same(warm, cold)
    refused = [
        r for _, blocks in cold.seen for r in blocks if r[0] == "result" and r[2]
    ]
    assert refused, cold.seen  # Edits whose text was gone, a Write after a change
    print("NOTES", json.dumps({"cold": cold.notes, "warm": warm.notes})[:3000])


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_slow_model_answer_costs_the_same(
    client: Client, tmp_path: Path, mode: str
) -> None:
    """The model takes a second after each call before its answer ends, as a real one
    can: Claude Code starts the call, and records the request's cost only at the end.
    Each pause reports its own cost all the same, also when the step misses."""
    cold = await run(client, tmp_path, "cold", mixed, mode=mode, warm=False, pause=1)
    warm = await run(client, tmp_path, "warm", mixed, mode=mode, warm=True, pause=1)
    missed = await run(
        client,
        tmp_path,
        "missed",
        mixed,
        mode=mode,
        warm=True,
        pause=1,
        miss_step="end",
    )
    same(warm, cold)
    same(missed, cold)
    assert warm.runner.warm_steps == activity_calls(mode)


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_relative_path_runs_warm_as_cold(
    client: Client, tmp_path: Path, mode: str
) -> None:
    """Claude Code makes a relative ``file_path`` absolute before hooks see the call:
    the pause reports that input, and the step runs the call warm, as it does cold."""

    def relative(work: Path) -> list[Step]:
        del work
        return [
            ("Read", {"file_path": "notes.txt"}),
            (
                "Edit",
                {"file_path": "notes.txt", "old_string": "old", "new_string": "new"},
            ),
            ("Write", {"file_path": "out.txt", "content": "x\n"}),
        ]

    cold = await run(client, tmp_path, "cold", relative, mode=mode, warm=False)
    warm = await run(client, tmp_path, "warm", relative, mode=mode, warm=True)
    same(warm, cold)
    assert warm.files["notes.txt"] == "new\n" and warm.files["out.txt"] == "x\n"
    assert all(Path(c["input"]["file_path"]).is_absolute() for c in warm.calls)
    assert warm.runner.warm_steps == (2 if mode == "store" else 1)


async def test_calls_do_not_wait_where_claude_code_would_let_them_run(
    client: Client, tmp_path: Path
) -> None:
    """With a permission mode that would let a call run if its hook gave no answer
    (``bypassPermissions`` here), calls never wait in a segment's engine."""
    bypass = {"permission_mode": "bypassPermissions"}
    cold = await run(
        client, tmp_path, "cold", mixed, mode="store", warm=False, extra_options=bypass
    )
    warm = await run(
        client, tmp_path, "warm", mixed, mode="store", warm=True, extra_options=bypass
    )
    same(warm, cold)
    assert warm.runner.waited == [] and warm.runner.warm_steps == 0


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_call_whose_step_comes_too_late_never_runs_there(
    client: Client, tmp_path: Path, mode: str
) -> None:
    """The step comes after ``warm_seconds``: the waiting engine ended, its hook denied
    the call, and its turn stopped (no model request). The step runs the call once,
    in a new engine."""
    cold = await run(client, tmp_path, "cold", mixed, mode=mode, warm=False)
    warm = await run(
        client,
        tmp_path,
        "warm",
        mixed,
        mode=mode,
        warm=True,
        warm_seconds=1.0,
        step_delay=2.0,
    )
    same(warm, cold)
    assert warm.runner.warm_steps == 0
    assert len(warm.runner.waited) == activity_calls(mode)
    await ended(warm.runner.waited)


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_turn_whose_next_segment_comes_too_late_stops(
    client: Client, tmp_path: Path, mode: str
) -> None:
    """The next segment comes after ``warm_seconds``: the turn that waited after the
    call stopped without asking Claude, and a new engine continues from the
    checkpoint. (``warm_seconds`` leaves a busy machine time to bring each step to its
    waiting call.)"""
    cold = await run(client, tmp_path, "cold", mixed, mode=mode, warm=False)
    warm = await run(
        client,
        tmp_path,
        "warm",
        mixed,
        mode=mode,
        warm=True,
        warm_seconds=3.0,
        segment_delay=5.0,
    )
    same(warm, cold)
    assert warm.runner.warm_steps == activity_calls(mode)
    assert warm.runner.warm_segments == 0


@pytest.mark.parametrize("model", ["quick", "slow"])
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_calls_in_one_message_do_not_wait(
    client: Client, tmp_path: Path, mode: str, model: str
) -> None:
    """Only one call of a message may pause the run, so the calls of a message with
    several pause as without warm engines. Each runs once. Also with a model that takes
    its time after each call: Claude Code starts the first call before the second
    arrives (cold, it pauses there; warm, the call waits until the message is
    complete, then pauses as cold), and Claude sees the same."""

    def pair(work: Path) -> list[list[Step]]:
        log = work / "runs.log"
        return [
            [
                ("Bash", {"command": f"echo a >> {posix(log)}", "description": "a"}),
                ("Bash", {"command": f"echo b >> {posix(log)}", "description": "b"}),
            ],
            [("Bash", {"command": f"echo b >> {posix(log)}", "description": "b"})],
        ]

    pause = SLOW_MODEL_SECONDS if model == "slow" else 0.0
    cold = await run(client, tmp_path, "cold", pair, mode=mode, warm=False, pause=pause)
    warm = await run(client, tmp_path, "warm", pair, mode=mode, warm=True, pause=pause)
    same(warm, cold)
    assert sorted(warm.files["runs.log"].split()) == ["a", "b"]
    assert warm.runner.warm_steps == 1  # the second message's single call


async def test_a_call_with_text_after_it_in_its_message_does_not_wait(
    client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow model writes a line after its call, in the same message. The call does not
    wait (the session would not end with it): it pauses as cold, at once, not after
    the time the segment gives the call's entry to arrive (made long here)."""
    monkeypatch.setattr(_runner, "ENTRY_SECONDS", 120.0)

    def flow(work: Path) -> list[list[Step]]:
        log = work / "runs.log"
        return [
            [
                ("Bash", {"command": f"echo a >> {posix(log)}", "description": "a"}),
                ("text", {"text": "Running it now."}),
            ],
            [("Bash", {"command": f"echo b >> {posix(log)}", "description": "b"})],
        ]

    pause = SLOW_MODEL_SECONDS
    cold = await run(
        client, tmp_path, "cold", flow, mode="store", warm=False, pause=pause
    )
    started = time.monotonic()
    warm = await run(
        client, tmp_path, "warm", flow, mode="store", warm=True, pause=pause
    )
    assert time.monotonic() - started < 90
    same(warm, cold)
    assert warm.files["runs.log"] == "a\nb\n"
    assert warm.runner.warm_steps == 1  # the second message's call


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_step_that_times_out_ends_its_warm_engine(
    client: Client, tmp_path: Path, mode: str
) -> None:
    """A command longer than the step's time limit: the warm engine ends with the step,
    and Claude hears what it hears without warm engines. (The limit leaves a busy
    machine time to start a new Claude Code for the next, quick step; the timed-out
    command still writes when it ends, so the lines of the file may come in another
    order.)"""
    options = ShellOptions(
        builtin_tools=["Bash"], tool_activities=["Bash"], tool_timeout=12
    )

    def slow(work: Path) -> list[Step]:
        log = work / "runs.log"
        return [
            (
                "Bash",
                {
                    "command": f"sleep 16; echo late >> {posix(log)}",
                    "description": "slow",
                },
            ),
            ("Bash", {"command": f"echo next >> {posix(log)}", "description": "next"}),
        ]

    async def files(run: Run) -> dict[str, str]:
        """Each file's words, sorted (the commands may write in another order), once
        the timed-out command wrote too (it does when it ends)."""
        work = run.root / "work"
        deadline = time.monotonic() + 60
        found: dict[str, str] = {}
        while True:
            # Windows: a file a command writes at this moment cannot be read; the
            # last full read stays.
            with contextlib.suppress(OSError):
                found = {
                    p.name: " ".join(sorted(p.read_text(encoding="utf-8").split()))
                    for p in work.iterdir()
                    if p.is_file()
                }
            if "late" in found.get("runs.log", "") or time.monotonic() > deadline:
                return found
            await asyncio.sleep(0.2)

    cold = await run(
        client, tmp_path, "cold", slow, mode=mode, warm=False, options=options
    )
    warm = await run(
        client, tmp_path, "warm", slow, mode=mode, warm=True, options=options
    )
    cold.files, warm.files = await files(cold), await files(warm)
    assert warm.files["runs.log"] == "late next"
    same(warm, cold)
    await ended(warm.runner.waited)


async def test_a_worker_that_stops_ends_the_engine_where_a_call_waits(
    client: Client, tmp_path: Path
) -> None:
    """The Worker stops (a deploy) while a call waits for its step: the engine ends
    with it, and the call never runs. (With the conversation in a session store, the
    engine that starts it can stay warm, so its first call waits.)"""
    work = tmp_path / "work"
    work.mkdir()
    log = work / "runs.log"
    api = scripted(
        [("Bash", {"command": f"echo ran >> {posix(log)}", "description": "x"})]
    )
    queue = f"warmstop-{uuid.uuid4().hex[:8]}"
    runner = Tracking(
        session_store=FileSessionStore(tmp_path / "store"),
        cwd=str(work),
        env=engine_env(api, str(tmp_path / "cfg")),
        warm_engines=2,
        warm_seconds=60,
        step_delay=120,  # the step never takes the engine
    )
    options = ShellOptions(builtin_tools=["Bash"], tool_activities=["Bash"])
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[ShellWorkflow],
            activities=ALL,
            plugins=[ClaudeAgentPlugin(runner)],
            **FAIL_FAST,
        ):
            handle = await client.start_workflow(
                ShellWorkflow.run, args=["go", options], id=queue, task_queue=queue
            )
            await wait_until(lambda: bool(runner.waited), timeout=60)
            engine = runner.waited[0]
            hooks = descendants(engine)  # the hook where the call waits
            assert alive(engine) and hooks
        await all_ended()
        await wait_until(lambda: not alive(engine), timeout=15)
        await wait_until(lambda: not any(alive(p) for p in hooks), timeout=15)
        assert not log.exists()  # the call never ran
        await handle.terminate("the test ended")
    finally:
        api.stop()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
@pytest.mark.parametrize("repeatable", [True, False], ids=["repeatable", "once"])
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_killed_worker_takes_the_engine_where_a_call_waits_along(
    client: Client, address: str, tmp_path: Path, mode: str, repeatable: bool
) -> None:
    """The Worker process dies while a call waits for its step (the step is slow to
    start there): the engine and its hook end with it, and the call never runs there.
    In ``repeatable_tools``, Temporal runs the step again on a new Worker, which runs
    the call once; otherwise Claude is told that it may have run, as without warm
    engines."""
    work = tmp_path / "work"
    work.mkdir()
    log = work / "runs.log"
    # A durable call first: with the conversation in the Workflow, the engine that
    # starts it never stays warm, so in both modes the first command is the one that
    # waits in the Worker that dies.
    script: list[Step] = [
        ("count", {"n": 1}),
        ("Bash", {"command": f"echo first >> {posix(log)}", "description": "x"}),
        ("Bash", {"command": f"echo second >> {posix(log)}", "description": "x"}),
    ]
    api = scripted(script)
    queue = f"warmkill-{mode}-{uuid.uuid4().hex[:8]}"
    workers: list[subprocess.Popen[bytes]] = []

    def env(n: int, delay: float) -> dict[str, str]:
        return {
            "CHAOS_ENGINE_ENV": json.dumps(engine_env(api, str(tmp_path / f"cfg{n}"))),
            "ENGINE_CWD": str(work),
            "RUNNER_MODE": mode,
            "SESSION_DIR": str(tmp_path / "sessions"),
            "SHOP_DIR": str(tmp_path / "shop"),
            "WARM_ENGINES": "4",
            "STEP_DELAY": str(delay),  # only for a call that waits in a warm engine
        }

    options = ShellOptions(
        builtin_tools=["Bash"],
        tool_activities=["Bash"],
        tool_timeout=10,
        repeatable_tools=["Bash"] if repeatable else [],
    )
    try:
        workers.append(await start_worker(address, queue, env(0, 120), tmp_path / "w0"))
        handle = await client.start_workflow(
            ShellWorkflow.run, args=["go", options], id=queue, task_queue=queue
        )
        # The step sleeps before it starts: wait until it runs there, with the call
        # waiting in its hook.
        for _ in range(900):
            steps = [
                kind
                for kind, _, _ in await running_on(handle, workers[0].pid)
                if kind == "run_claude_tool_step"
            ]
            waiting = any((tmp_path / "w0" / "tmp").glob("tca-hook-*/waiting"))
            if steps and waiting:
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("the call never waited for its step on the Worker")
        engines = engines_of(workers[0].pid)
        assert engines, "no engine runs on the Worker"
        hooks = [p for pid, _ in engines for p in descendants(pid)]
        assert hooks, "no hook waits"
        kill(workers[0])
        await wait_until(lambda: not still_running(engines), timeout=15)
        await wait_until(lambda: not any(alive(p) for p in hooks), timeout=15)
        assert not log.exists()  # the call never ran in the Worker that died
        workers.append(await start_worker(address, queue, env(1, 0), tmp_path / "w1"))
        answer = await asyncio.wait_for(handle.result(), 200)
        calls = await handle.query(ShellWorkflow.tool_calls)
    finally:
        for worker in workers:
            kill(worker)
        api.stop()
    assert answer == "DONE"
    ran = log.read_text(encoding="utf-8").split()
    assert ran == (["first", "second"] if repeatable else ["second"])
    statuses = [c["status"] for c in calls]
    assert statuses == ["done", "done" if repeatable else "interrupted", "done"]


async def test_a_step_with_other_input_ends_the_waiting_engine(tmp_path: Path) -> None:
    """A step for the waiting call, but with an input other than its hook saw (or for
    another call): the warm engine ends, and neither command runs."""
    work = tmp_path / "work"
    work.mkdir()
    log = work / "runs.log"
    api = scripted(
        [("Bash", {"command": f"echo ran >> {posix(log)}", "description": "x"})]
    )
    runner = Tracking(
        session_store=FileSessionStore(tmp_path / "store"),
        cwd=str(work),
        env=engine_env(api, str(tmp_path / "cfg")),
        warm_engines=2,
    )
    count = ToolSpec("count", "Count.", {"type": "object", "properties": {}})
    try:
        first = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="go",
                tools=[count],
                builtin_tools=["Bash"],
                tool_activities=["Bash"],
            ),
            1,
        )
        assert first.deferred is not None and first.deferred.kind == "engine"
        assert first.checkpoint is not None
        assert len(runner.waited) == 1
        other = DeferredCall(
            id=first.deferred.id,
            name="Bash",
            input={"command": f"echo other >> {posix(log)}", "description": "x"},
            kind="engine",
        )
        with pytest.raises(ApplicationError):
            await runner.run_tool_step(
                ToolStepInput(
                    session_id=first.session_id,
                    checkpoint=first.checkpoint,
                    call=other,
                    tools=[count],
                    builtin_tools=["Bash"],
                ),
                1,
            )
        await all_ended()
        assert runner._warm == {}  # type: ignore[reportPrivateUsage]
        assert runner.warm_steps == 0
        await ended(runner.waited[:1])
        assert not log.exists()
    finally:
        runner._end_all_warm()  # type: ignore[reportPrivateUsage]
        await all_ended()
        api.stop()


def test_runs_are_compared_without_bash_job_control_warnings(tmp_path: Path) -> None:
    """A command's result with bash's job control warning (``JOB_CONTROL``, seen on
    macOS) reads as the same result without it, and the warning goes in the notes."""

    def request(text: str, is_error: bool = False) -> dict[str, Any]:
        result = {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": text,
            "is_error": is_error,
        }
        return {"messages": [{"role": "user", "content": [result]}]}

    warning = "/bin/bash: child setpgid (44121 to 44121): Operation not permitted"
    quiet, none = conversation(request(NO_OUTPUT), tmp_path)
    noisy, notes = conversation(request(warning), tmp_path)
    assert noisy == quiet and none == [] and notes == ["bash: child setpgid"]
    failed, _ = conversation(request(f"Exit code 3\n{warning}", True), tmp_path)
    assert failed == conversation(request("Exit code 3", True), tmp_path)[0]
    output, _ = conversation(request(f"{warning}\nout"), tmp_path)
    assert output == conversation(request("out"), tmp_path)[0]


# ---- the hook's side, without an engine ----


@pytest.fixture
def waiting_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A run folder, and the environment of a hook whose calls may wait."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("TCA_HOOK_DIR", str(run_dir))
    monkeypatch.setenv("TCA_TOOL_ACTIVITIES", "Bash")
    monkeypatch.setenv("TCA_WAIT_SECONDS", "5")
    monkeypatch.delenv("TCA_ALLOW_ID", raising=False)
    monkeypatch.delenv("TCA_ANSWERED_IDS", raising=False)
    return run_dir


EVENT = {"tool_use_id": "toolu_w", "tool_name": "Bash", "tool_input": {"command": "x"}}


def decide_in_thread(event: dict[str, Any]) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []
    thread = threading.Thread(target=lambda: out.append(_defer_hook.decide(event)))
    thread.start()
    return thread, out


def wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, path
        time.sleep(0.002)


@pytest.mark.parametrize(
    ("decision", "expected", "reason"),
    [
        ("defer", "defer", None),
        ("allow {digest}", "allow", None),
        ("allow 0000", "deny", _defer_hook.OTHER_INPUT),
        ("stop", "deny", _defer_hook.STOPPED),
        ("closed", "deny", _defer_hook.STOPPED),
        ("gone", "deny", _defer_hook.STOPPED),
    ],
)
def test_the_hook_lets_a_call_wait_for_the_runner(
    waiting_hook: Path, decision: str, expected: str, reason: str | None
) -> None:
    """The hook names the waiting call and its input, then follows the runner: "defer",
    or "allow" with the input it saw. A step that ended first ("closed": the runner's
    ``allowed`` record), a stopped run or a removed folder deny the call."""
    digest = _defer_hook.input_digest(EVENT["tool_input"])
    thread, out = decide_in_thread(EVENT)
    waiting = waiting_hook / _defer_hook.WAITING
    deadline = time.monotonic() + 5
    while _runner._read_waiting(str(waiting_hook)) is None:  # type: ignore[reportPrivateUsage]
        assert time.monotonic() < deadline
        time.sleep(0.002)
    text = _defer_hook.canonical_input(EVENT["tool_input"])
    assert waiting.read_text(encoding="utf-8") == f"toolu_w\n{digest}\n{text}"
    if decision == "stop":
        (waiting_hook / "stop").touch()
    elif decision == "gone":
        os.rename(waiting_hook, f"{waiting_hook}.gone")
    else:
        if decision == "closed":
            assert not _runner._close_call(str(waiting_hook))  # type: ignore[reportPrivateUsage]
            decision = "allow {digest}"
        _runner._decide(str(waiting_hook), "toolu_w", decision.format(digest=digest))  # type: ignore[reportPrivateUsage]
    thread.join(10)
    assert not thread.is_alive()
    assert out[0].get("permissionDecision") == expected
    assert out[0].get("permissionDecisionReason") == reason
    allowed = waiting_hook / _defer_hook.ALLOWED
    if expected == "allow":
        assert allowed.read_text(encoding="utf-8") == "toolu_w"
    assert not waiting.exists()  # the next call of the engine may wait


def test_a_call_waits_no_longer_than_its_limit(
    waiting_hook: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no decision, the hook denies the call at ``$TCA_WAIT_SECONDS``."""
    monkeypatch.setenv("TCA_WAIT_SECONDS", "0.3")
    started = time.monotonic()
    out = _defer_hook.decide(EVENT)
    assert 0.25 < time.monotonic() - started < 3
    assert not (waiting_hook / _defer_hook.WAITING).exists()
    assert out["permissionDecision"] == "deny"
    assert out["permissionDecisionReason"] == _defer_hook.STOPPED


def test_only_the_call_that_paused_the_run_waits(
    waiting_hook: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second call of the run is denied as before; durable calls never wait; with no
    limit set, calls defer at once."""
    (waiting_hook / "paused_call").write_text("toolu_first", encoding="utf-8")
    assert _defer_hook.decide(EVENT)["permissionDecision"] == "deny"
    (waiting_hook / "paused_call").unlink()
    durable = {**EVENT, "tool_name": _defer_hook.DURABLE_PREFIX + "count"}
    assert _defer_hook.decide(durable)["permissionDecision"] == "defer"
    (waiting_hook / "paused_call").unlink()
    monkeypatch.setenv("TCA_WAIT_SECONDS", "0")
    assert _defer_hook.decide(EVENT)["permissionDecision"] == "defer"
    assert not (waiting_hook / _defer_hook.WAITING).exists()


@pytest.mark.parametrize(
    ("gate", "then", "expected"),
    [
        (False, "nothing", {}),
        (True, "go", {}),
        (True, "stop", {"continue": False, "stopReason": _defer_hook.STOPPED}),
        (True, "gone", {"continue": False, "stopReason": _defer_hook.STOPPED}),
        (True, "go, stop", {"continue": False, "stopReason": _defer_hook.STOPPED}),
    ],
)
def test_the_turn_after_a_call_waits_for_the_next_segment(
    waiting_hook: Path, gate: bool, then: str, expected: dict[str, Any]
) -> None:
    """After a call that ran in its step (``turn_gate``), the ``PostToolBatch`` hook holds
    the turn until the next segment says go; a run that ends stops the turn, also with a
    go there ("go, stop": one an earlier turn got, which the runner removes anyway)."""
    if gate:
        (waiting_hook / _defer_hook.TURN_GATE).touch()
    if then == "go, stop":
        (waiting_hook / _defer_hook.TURN_GO).touch()
        (waiting_hook / "stop").touch()
    out: list[Any] = []
    thread = threading.Thread(target=lambda: out.append(_defer_hook.after_calls()))
    thread.start()
    if gate:
        wait_for_file(waiting_hook / _defer_hook.TURN_WAITS)
        if then == "go":
            (waiting_hook / _defer_hook.TURN_GO).touch()
        elif then == "stop":
            (waiting_hook / "stop").touch()
        elif then == "gone":
            os.rename(waiting_hook, f"{waiting_hook}.gone")
    thread.join(10)
    assert not thread.is_alive()
    assert out[0] == expected


def test_the_turn_waits_no_longer_than_its_limit(
    waiting_hook: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TCA_WAIT_SECONDS", "0.3")
    (waiting_hook / _defer_hook.TURN_GATE).touch()
    assert _defer_hook.after_calls() == {
        "continue": False,
        "stopReason": _defer_hook.STOPPED,
    }


def test_the_after_calls_hook_as_claude_code_runs_it(tmp_path: Path) -> None:
    """As its own process: with no gate it answers at once; with a gate and no time to
    wait (an unreadable limit) it stops the turn."""
    hook = Path(_defer_hook.__file__)
    env = {
        **os.environ,
        "TCA_HOOK_DIR": str(tmp_path / "missing"),
        "TCA_WAIT_SECONDS": "x",
    }
    done = subprocess.run(
        [sys.executable, "-I", "-S", str(hook), "after-calls"],
        input=b"{}",
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert done.returncode == 0
    assert json.loads(done.stdout) == {}  # no gate: nothing to hold
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / _defer_hook.TURN_GATE).touch()
    env["TCA_HOOK_DIR"] = str(tmp_path / "run")
    done = subprocess.run(
        [sys.executable, "-I", "-S", str(hook), "after-calls"],
        input=b"{}",
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert json.loads(done.stdout) == {
        "continue": False,
        "stopReason": _defer_hook.STOPPED,
    }
