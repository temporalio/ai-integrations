"""Rules the plugin's sizing, splicing and quoting keep, checked over many cases.

Each test states rules and checks them over cases from a seeded generator: the same
cases on every run, so a failure is never a flake, and its message names the case.
Limits are lowered where a rule does not depend on their size, so thousands of cases
run in seconds; other tests keep the real limits. Set ``RULE_CASES`` to check more.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import random
import re
import string
import subprocess
import sys
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import (
    AgentState,
    DurableClaudeAgent,
    SegmentInput,
    ToolOutcome,
)
from temporalio.claude_agent_sdk import _conversation as conversation
from temporalio.claude_agent_sdk import _defer_hook as hook
from temporalio.claude_agent_sdk import _events as events
from temporalio.claude_agent_sdk import _runner as runner
from temporalio.claude_agent_sdk import _workflow as wf
from temporalio.contrib.workflow_streams import PublisherState, WorkflowStreamState
from temporalio.contrib.workflow_streams._types import _WorkflowStreamWireItem
from temporalio.converter import DataConverter
from temporalio.exceptions import ApplicationError
from tests.helpers.outside_workflow import CarriedStream, OutsideWorkflow

CONVERTER = DataConverter.default.payload_converter
CASES = int(os.environ.get("RULE_CASES", "300"))
MIB = 1024 * 1024
ALPHABETS = (
    string.ascii_letters + string.digits + " _-",
    "\"'\\$`!*?;|&<>(){}[]#~%^=+,.:/\t\n\r",
    "مرحبا你好Łódźß€",
    "😀🚀‍﻿",
)


def cases(rule: str, count: int = CASES) -> Iterator[tuple[int, random.Random]]:
    """``count`` seeded generators for ``rule``: the same cases on every run."""
    for n in range(count):
        yield n, random.Random(f"{rule}-{n}")


def text(rng: random.Random, length: int) -> str:
    """Text that is plain, full of shell and JSON specials, or in several scripts."""
    pool = rng.choice((ALPHABETS[0], ALPHABETS[0] + ALPHABETS[1], "".join(ALPHABETS)))
    return "".join(rng.choice(pool) for _ in range(length))


def json_value(rng: random.Random, depth: int = 0) -> Any:
    kinds = ["str", "int", "float", "bool", "null"]
    if depth < 3:
        kinds += ["list", "dict", "dict"]
    kind = rng.choice(kinds)
    if kind == "str":
        return text(rng, rng.randint(0, 12))
    if kind == "int":
        return rng.choice((0, -1, 7, 2**40, -(2**63)))
    if kind == "float":
        return rng.choice((0.5, -2.25, 1e-7, 3.14159, 1e300))
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "null":
        return None
    if kind == "list":
        return [json_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {text(rng, rng.randint(0, 6)): json_value(rng, depth + 1) for _ in range(4)}


def payload_data(value: Any) -> bytes:
    return CONVERTER.to_payloads([value])[0].data


# ---- the conversation the Workflow holds ----


def test_pages_put_together_give_the_whole_conversation() -> None:
    for n, rng in cases("pages"):
        lengths = (0, 1, 20, 300)
        entries = [
            {"uuid": str(i), "text": text(rng, rng.choice(lengths))}
            for i in range(rng.randint(0, 30))
        ]
        sizes = [conversation.entry_bytes(e) for e in entries]
        limit = rng.choice((1, 2, 40, 300, 2000, 10**6))
        start, pages = 0, []
        while start < len(entries):
            got = conversation.page(entries, sizes, start, limit)
            case = f"case {n}: limit {limit}, start {start}, sizes {sizes}"
            assert got, case  # at least one entry, so a step always moves on
            data = len(payload_data(got))
            assert data <= limit or len(got) == 1, case  # within the limit, or alone
            end = start + len(got)
            if end < len(entries):  # and as full as it can be
                assert data + 1 + sizes[end] > limit, case
            pages.append(got)
            start = end
        assert [e for p in pages for e in p] == entries, f"case {n}"
        for past in (len(entries), len(entries) + 1):
            assert conversation.page(entries, sizes, past, limit) == []
        first = conversation.page(entries, sizes, 0, limit)
        assert conversation.page(entries, sizes, -3, limit) == first


def test_sizes_are_what_temporals_converter_writes() -> None:
    """For entries (JSON objects) and lists of them. The converter sorts keys, which
    keeps the size. (A top-level None would be an empty binary payload: no entry is.)"""
    for n, rng in cases("sizes"):
        entry = {text(rng, rng.randint(0, 6)): json_value(rng) for _ in range(4)}
        value = rng.choice((entry, [entry, json_value(rng)]))
        data = payload_data(value)
        case = f"case {n}: {value!r}"
        assert conversation.json_bytes(value) == len(data), case
        assert conversation.entry_bytes(value) == len(data), case
        assert json.loads(conversation.entry_text(value)) == json.loads(data), case


def common_prefix(a: list[Any], b: list[Any]) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def with_cost_totals(rng: random.Random, entries: list[Any]) -> list[Any]:
    """``entries`` with saved cost totals anywhere (Claude Code 2.1.277 and newer)."""
    out: list[Any] = []
    for entry in [*entries, None]:
        while rng.random() < 0.25:
            out.append({"type": "cost-state", "totalCostUSD": rng.random()})
        if entry is not None:
            out.append(entry)
    return out


def test_after_a_step_the_workflow_holds_the_steps_session() -> None:
    """``_kept``: what the Workflow keeps, then the rest of the session without cost
    totals (what the runner adds), is the session; whatever cost totals sit on either
    side, and whether the session grew, was rewritten after some point, or ends
    earlier (a seed cut at a deferral marker)."""
    plain = runner._without_cost_state
    for n, rng in cases("kept"):
        shared = [{"type": "user", "uuid": f"u{i}"} for i in range(rng.randint(0, 8))]
        kind = rng.choice(("grows", "rewrites", "ends earlier"))
        old = [
            {"type": "assistant", "uuid": f"old{i}"}
            for i in range(rng.randint(1, 3) if kind != "grows" else 0)
        ]
        new = [
            {"type": "assistant", "uuid": f"new{i}"}
            for i in range(rng.randint(0, 4) if kind != "ends earlier" else 0)
        ]
        committed = with_cost_totals(rng, shared + old)
        session = with_cost_totals(rng, shared + new)
        keep, covered = runner._kept(committed, session)
        same = common_prefix(plain(committed), plain(session))
        case = f"case {n}: {kind}, keep {keep}, covered {covered}"
        assert plain(committed[:keep]) == plain(committed)[:same], case
        assert plain(session[:covered]) == plain(session)[:same], case
        spliced = committed[:keep] + plain(session[covered:])
        assert plain(spliced) == plain(session), case  # nothing lost, nothing twice
        assert (keep == len(committed)) == (same == len(plain(committed))), case


# ---- tool results that do not fit in one payload ----


def segment_input(agent: DurableClaudeAgent) -> Callable[[], SegmentInput]:
    def make() -> SegmentInput:
        return SegmentInput(
            session_id="s",
            prompt=None,
            tools=[],
            checkpoint="c",
            injected=dict(agent._state.pending),
        )

    return make


def input_bytes(value: Any) -> int:
    return sum(p.ByteSize() for p in CONVERTER.to_payloads([value]))


async def fit(
    pending: dict[str, ToolOutcome], external_storage: bool = False
) -> tuple[dict[str, ToolOutcome], int, str | None]:
    """What ``_fit_results`` leaves: the results, the input's size, or why it failed."""
    agent = DurableClaudeAgent(
        state=AgentState(pending=dict(pending), external_storage=external_storage)
    )
    make = segment_input(agent)
    try:
        inp = await agent._fit_results(make)
    except ApplicationError as err:
        assert err.non_retryable
        return dict(agent._state.pending), input_bytes(make()), str(err)
    assert inp.injected == agent._state.pending
    return dict(agent._state.pending), input_bytes(inp), None


async def test_results_that_do_not_fit_become_notes_largest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The input fits whenever it can, with the fewest notes, the largest results
    first (ties by id), the same in any order (a replay must decide the same)."""
    OutsideWorkflow().install(monkeypatch)
    for n, rng in cases("fit", CASES // 2):
        limit = rng.choice((700, 1500, 4000, 9000))
        monkeypatch.setattr(wf, "PAYLOAD_LIMIT_BYTES", limit)
        ids = [f"toolu_{rng.randint(10, 99)}{c}" for c in "abcdef"[: rng.randint(1, 6)]]
        pending = {
            i: ToolOutcome(
                content="r" * rng.choice((0, 10, 300, 1500, 3000)),
                is_error=rng.random() < 0.2,
            )
            for i in ids
        }
        before = input_bytes(
            SegmentInput(
                session_id="s", prompt=None, tools=[], checkpoint="c", injected=pending
            )
        )
        got, size, failed = await fit(pending)
        noted = [i for i in ids if got[i] != pending[i]]
        sizes = {i: input_bytes(o) for i, o in pending.items()}
        order = sorted(ids, key=lambda i: (-sizes[i], i))
        case = f"case {n}: limit {limit}, sizes {sizes}, noted {noted}, failed {failed}"
        if before <= limit:
            assert not noted and failed is None and size == before, case
            continue
        replaced = [i for i in order if i in noted]
        assert replaced == order[: len(noted)], case  # the largest ones
        for i in noted:
            mb = f"{sizes[i] / 1024 / 1024:.1f} MB"
            assert got[i].is_error and mb in str(got[i].content), case
            assert "Do not call the tool again" in str(got[i].content), case
        if failed is None:
            assert size <= limit, case
            fewer = {**pending, **{i: got[i] for i in replaced[:-1]}}
            assert input_bytes(segment_with(fewer)) > limit, case  # one note fewer
        else:
            assert len(noted) == len(ids) and size > limit, case
            assert f"The next step's input is {size / 1024 / 1024:.2f} MB" in failed
        shuffled = dict(rng.sample(list(pending.items()), len(pending)))
        assert await fit(shuffled) == (got, size, failed), case
    # With External Storage the results go to the store: nothing becomes a note.
    big = {"toolu_1": ToolOutcome(content="r" * 5000)}
    got, _, failed = await fit(big, external_storage=True)
    assert got == big and failed is None


async def test_a_tool_steps_record_stays_when_its_result_becomes_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An Edit's or a Write's record of its call (``ToolOutcome.entry``) stays with
    the note, so the note reaches Claude where Claude Code expects the call's result;
    its metadata goes, as it can be what was too large."""
    OutsideWorkflow().install(monkeypatch)
    monkeypatch.setattr(wf, "PAYLOAD_LIMIT_BYTES", 4000)
    record = {
        "type": "user",
        "uuid": "u1",
        "message": {"role": "user", "content": []},
        "toolUseResult": {"originalFile": "o" * 3000},
    }
    got, size, failed = await fit(
        {
            "toolu_1": ToolOutcome(content="r" * 3000, entry=record),
            "toolu_2": ToolOutcome(content="small"),
        }
    )
    assert failed is None and size <= 4000
    assert got["toolu_1"].is_error and "too large" in str(got["toolu_1"].content)
    assert got["toolu_1"].entry == {
        k: v for k, v in record.items() if k != "toolUseResult"
    }
    assert got["toolu_2"] == ToolOutcome(content="small")


def segment_with(pending: dict[str, ToolOutcome]) -> SegmentInput:
    return SegmentInput(
        session_id="s", prompt=None, tools=[], checkpoint="c", injected=pending
    )


# ---- the live output a new run carries ----


def stream_events(rng: random.Random) -> list[_WorkflowStreamWireItem]:
    return [
        _WorkflowStreamWireItem(
            topic=events.TOPIC,
            data=base64.b64encode(rng.randbytes(rng.choice((0, 5, 90, 700)))).decode(),
        )
        for _ in range(rng.randint(0, 120))
    ]


def test_a_new_run_carries_the_newest_events_that_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Also when the first estimate is off (the stream's publishers are not in it) and
    the input is measured again until it fits: many cases are built that way."""
    OutsideWorkflow().install(monkeypatch)
    measured_again = 0
    for n, rng in cases("carry", CASES // 3):
        share = rng.choice((3000, 12000, 60000))
        monkeypatch.setattr(wf, "_HANDOVER_BYTES", share)
        log = stream_events(rng)
        seen = datetime(2026, 10, 3, tzinfo=timezone.utc)
        publishers = {
            f"publisher-{i:02d}": PublisherState(sequence=i, last_seen=seen)
            for i in range(rng.choice((0, 3, 12, 30)))
        }
        if rng.random() < 0.6:  # the share decides: the first estimate can be off
            keep, keep_bytes = 1000, 10**6
        else:
            keep = rng.choice((0, 1, 7, 50, 1000))
            keep_bytes = rng.choice((0, 300, 4000, 10**6))
        filled = int(share * rng.choice((0.0, 0.2, 0.6, 0.85)))  # of the share
        agent = DurableClaudeAgent(
            state=AgentState(conversation=["x" * 900] * (filled // 910)),
            live_output_keep=keep,
            live_output_keep_bytes=keep_bytes,
        )
        stream = WorkflowStreamState(
            log=list(log), base_offset=rng.randint(0, 10**6), publishers=publishers
        )
        agent._stream = CarriedStream(stream)  # type: ignore[assignment]
        state = agent.state()
        external = rng.random() < 0.2
        rest = None if external else agent._input_bytes(state)
        measure = agent._input_bytes
        calls: list[int] = []

        def counted(state: AgentState) -> int:
            calls.append(1)
            return measure(state)

        monkeypatch.setattr(agent, "_input_bytes", counted)
        total = agent._carry_stream(state, rest)
        measured_again += len(calls) > 1
        monkeypatch.setattr(agent, "_input_bytes", measure)
        assert state.stream is not None
        carried = state.stream.log
        k = len(carried)
        case = f"case {n}: share {share}, keep {keep}/{keep_bytes}, kept {k}/{len(log)}"
        assert carried == log[len(log) - k :], case  # the newest events
        assert state.stream.publishers == publishers, case
        assert k <= keep, case
        assert sum(wf._event_bytes(e) for e in carried) <= keep_bytes, case
        if total is None:
            assert external, case
        else:
            assert not external and total == agent._input_bytes(state), case
            assert total <= share or k == 0, case
        if k < len(log):  # one more event would break a limit
            more = log[len(log) - k - 1 :]
            with_more = dataclasses.replace(
                state, stream=dataclasses.replace(state.stream, log=more)
            )
            fits = (
                k + 1 <= keep
                and sum(wf._event_bytes(e) for e in more) <= keep_bytes
                and (external or agent._input_bytes(with_more) <= share)
            )
            assert not fits, case
    assert (
        measured_again >= CASES // 3 // 10
    )  # the second measurement runs (about 1 in 5)


# ---- how close a run's history is to Temporal's limits ----


def test_a_run_is_nearly_full_when_its_next_step_could_pass_the_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The room kept is twice the largest step of a task so far, at least 500 events
    and three payloads; the time between tasks is not a step."""
    outside = OutsideWorkflow().install(monkeypatch)
    for n, rng in cases("nearly-full", CASES // 3):
        after = rng.choice((None, 1000, 60_000))
        agent = DurableClaudeAgent(continue_as_new_after_events=after)
        most = max(51_200, after or 0)
        events_, size = rng.randint(0, 50_000), rng.randint(0, 48 * MIB)
        largest = (0, 0)
        last: tuple[int, int] | None = None
        for step in range(rng.randint(1, 9)):
            if step and rng.random() < 0.2:
                agent._mark = None  # a new task (as the loop does)
                last = None
                events_, size = events_ + 5000, size + 9 * MIB  # idle time between
            elif step:
                events_ += rng.choice((0, 1, 50, 400, 3000))
                size += rng.choice((0, 1000, 2 * MIB, 8 * MIB))
            outside.history.events, outside.history.size = events_, size
            got = agent._history_nearly_full()
            if last is not None:
                largest = (
                    max(largest[0], events_ - last[0]),
                    max(largest[1], size - last[1]),
                )
            last = (events_, size)
            room = (
                max(500, 2 * largest[0]),
                max(3 * conversation.PAYLOAD_LIMIT_BYTES, 2 * largest[1]),
            )
            full = events_ + room[0] >= most or size + room[1] >= 50 * MIB
            case = f"case {n} step {step}: {events_} events, {size} bytes, {largest}"
            assert (got is not None) == full, case
            if got is not None:
                assert f"({events_:,} events, {size / MIB:.1f} MB)" in got, case
                assert f"limits ({most:,} events, 50 MB by default)" in got, case


# ---- live output events ----


def test_a_capped_event_keeps_the_start_of_each_long_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 40
    monkeypatch.setattr(events, "FIELD_LIMIT", limit)
    keys = ("text", "result", "error", "input", "id", "name")
    for n, rng in cases("cap"):
        event: dict[str, Any] = {"type": "tool_call"}
        for key in rng.sample(keys, rng.randint(0, len(keys))):
            event[key] = rng.choice(
                (
                    text(rng, rng.randint(0, 2 * limit)),
                    None,
                    7,
                    {"command": text(rng, rng.randint(0, 2 * limit))},
                    ["a", "b"],
                )
            )
        capped = events.cap_event(event)
        cut = False
        case = f"case {n}: {event!r}"
        for key, value in event.items():
            got = capped[key]
            if key in ("text", "result", "error", "input") and isinstance(value, str):
                assert got == value[:limit], case
                cut = cut or len(value) > limit
            elif key == "input":
                as_text = json.dumps(value, ensure_ascii=False, default=str)
                if len(as_text) > limit:
                    assert got == as_text[:limit], case
                    cut = True
                else:
                    assert got is value, case
            else:
                assert got is value, case
        assert set(capped) == set(event) | ({"truncated"} if cut else set()), case
        assert capped.get("truncated", True) is True, case
        assert events.cap_event(capped) == capped, case  # capping twice changes nothing


# ---- what the hook writes and what the runner reads ----


def test_a_denial_record_is_always_a_safe_file_name() -> None:
    safe = re.compile(r"[A-Za-z0-9_-]{1,128}")
    for n, rng in cases("denial-name"):
        tool_use_id = rng.choice(
            (
                text(rng, rng.randint(0, 140)),
                "toolu_" + "".join(rng.choice(string.ascii_letters) for _ in range(24)),
                "../" * rng.randint(1, 3) + "x",
            )
        )
        name = hook.denial_name(tool_use_id)
        case = f"case {n}: {tool_use_id!r}"
        assert safe.fullmatch(name), case
        if safe.fullmatch(tool_use_id):
            assert name == tool_use_id, case
        else:
            digest = hashlib.sha256(tool_use_id.encode("utf-8")).hexdigest()
            assert name == digest, case


def test_only_the_hooks_own_denial_reads_as_its_denial() -> None:
    """Claude Code shows the reason alone (2.1.273) or after a label (2.1.281 and
    newer). A tool's output that merely contains the reason is never taken for it."""
    reasons = [hook.STOPPED, hook.NOT_RUN, hook.STEP_ONLY, hook.MAIN_AGENT_ONLY]
    for n, rng in cases("says-only"):
        reason = rng.choice([*reasons, text(rng, rng.randint(1, 30)).strip() or "r"])
        tool = rng.choice(("Bash", hook.DURABLE_PREFIX + "refund", "Read", "x y"))
        label = f"PreToolUse:{tool} hook error: "
        case = f"case {n}: {reason!r} {tool!r}"
        for same in (reason, f"  {reason}\n", label + reason, f"\n{label}{reason} "):
            assert runner._says_only(same, reason), case + f" {same!r}"
        for other in (
            f"{reason} and more",
            f"output: {reason}",
            label + reason + ".",
            f"PostToolUse:{tool} hook error: {reason}",
            f"{label}{reason}{label}{reason}",
            [{"type": "text", "text": reason}],
            None,
            "",
        ):
            assert not runner._says_only(other, reason), case + f" {other!r}"


# ---- what a command's shell sees ----

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell semantics (Windows runs Git Bash)"
)


@posix_only
def test_a_quoted_word_reaches_the_shell_unchanged(tmp_path: Path) -> None:
    script = tmp_path / "say.sh"
    for n, rng in cases("quote", CASES // 4):
        value = text(rng, rng.randint(0, 40))
        script.write_text(f"printf %s {runner._sh_quote(value)}\n", encoding="utf-8")
        out = subprocess.run(["/bin/sh", str(script)], capture_output=True, check=True)
        assert out.stdout.decode("utf-8") == value, f"case {n}: {value!r}"


DUMP = "import json, os; print(json.dumps(dict(os.environ)))"
NAMES = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "NO_PROXY", "HTTPS_PROXY")


@posix_only
def test_commands_see_the_workers_own_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Values of any kind come back exactly; no TCA_* variable is left; your own
    CLAUDE_ENV_FILE (found from the working directory) runs last, as Claude Code
    would run it."""
    for n, rng in cases("command-env", CASES // 6):
        worker: dict[str, str] = {}
        for name in (*NAMES, "CLAUDE_ENV_FILE", "TCA_STRAY"):
            monkeypatch.delenv(name, raising=False)
            if name != "CLAUDE_ENV_FILE" and rng.random() < 0.6:
                worker[name] = text(rng, rng.randint(0, 30))
        own_file = rng.random() < 0.4
        mine = rng.choice(NAMES)
        if own_file:
            (tmp_path / f"own{n}.sh").write_text(
                f'export FROM_YOURS="yes"\nexport {mine}="yours"\n', encoding="utf-8"
            )
            worker["CLAUDE_ENV_FILE"] = f"own{n}.sh"  # relative to the working dir
        for name, value in worker.items():
            monkeypatch.setenv(name, value)
        overrides = {
            name: text(rng, 8) for name in rng.sample(NAMES, rng.randint(1, 4))
        }
        hook_dir = tmp_path / f"step{n}"
        hook_dir.mkdir()
        engine = runner._command_env(
            {"TCA_HOOK_DIR": str(hook_dir)},
            str(hook_dir),
            str(tmp_path),
            overrides,
            {"TCA_ALLOW_ID": "toolu_1"},
        )
        seen = subprocess.run(
            [
                "/bin/sh",
                "-c",
                f'. "$CLAUDE_ENV_FILE" && "{sys.executable}" -c "{DUMP}"',
            ],
            env={**os.environ, **engine},
            capture_output=True,
            check=True,
            cwd=tmp_path,
        )
        got = json.loads(seen.stdout)
        case = f"case {n}: worker {worker!r}, overrides {overrides!r}"
        for name in NAMES:
            want = "yours" if own_file and name == mine else worker.get(name)
            assert got.get(name) == want, case
        assert got.get("CLAUDE_ENV_FILE") == worker.get("CLAUDE_ENV_FILE"), case
        assert ("FROM_YOURS" in got) == own_file, case
        assert not [k for k in got if k.startswith("TCA_")], case
