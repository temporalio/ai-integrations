"""The conversation code of tool steps, checked on thousands of random conversations.

The hand-written cases in ``test_tool_step_replay.py`` show each rule once. Here a
seeded generator writes conversations the way Claude Code does (one entry per
content block, hook entries before each result, the deferral marker of the paused
call, denials of the calls after it, earlier pauses whose results came as a message,
earlier tool steps' records, compactions, a subagent's entries, saved cost totals,
links to entries that are gone, entries a later run wrote), and every function is
checked against rules that must hold for all of them. A failure prints its seed, so
it can be run again alone.
"""

from __future__ import annotations

import copy
import json
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from temporalio.claude_agent_sdk import ToolOutcome, _runner
from temporalio.claude_agent_sdk._defer_hook import NOT_RUN, input_digest

SEEDS = range(2000)
ENGINE = ("Read", "Bash", "Edit", "Write", "mcp__github__create_issue")
"""Claude Code tools; Read runs in the segment, the others run as Activities."""
DURABLE = ("mcp__durable__count", "mcp__durable__refund")
RECORDED = set(_runner.RECORDED_TOOLS)


@dataclass
class Conversation:
    """A conversation that paused at a tool call, and what the generator knows of it."""

    entries: list[dict[str, Any]]
    checkpoint: str
    first: int  # where the paused call's hook entries start
    marker: int
    paused: str
    paused_tool: str
    owner: str  # the uuid of the paused call's assistant entry
    message_calls: list[tuple[str, str]]  # (id, tool) of the paused call's message
    denied: list[tuple[str, str]] = field(default_factory=list)  # after the pause


class Writer:
    """Writes entries as Claude Code does, each linked to the one before it."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.entries: list[dict[str, Any]] = []
        self.count = 0
        self.last: str | None = None

    def _uid(self) -> str:
        self.count += 1
        return f"e{self.count}"

    def add(self, kind: str, parent: str | None, **fields: Any) -> str:
        uid = self._uid()
        self.entries.append({"type": kind, "uuid": uid, "parentUuid": parent, **fields})
        self.last = uid
        return uid

    def prompt(self, text: str) -> None:
        self.add("user", self.last, message={"role": "user", "content": text})

    def calls(self, tools: list[str]) -> list[tuple[str, str, str]]:
        """One assistant message with these calls: (call id, tool, entry uuid)."""
        message_id = f"msg_{self._uid()}"
        out = []
        if self.rng.random() < 0.3:  # a text block first, as Claude often writes
            self.add(
                "assistant",
                self.last,
                message={
                    "id": message_id,
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Let me do that."}],
                },
            )
        for tool in tools:
            call_id = f"toolu_{self._uid()}"
            block = {
                "type": "tool_use",
                "id": call_id,
                "name": tool,
                "input": {"n": self.rng.randint(0, 9)},
            }
            uid = self.add(
                "assistant",
                self.last,
                message={"id": message_id, "role": "assistant", "content": [block]},
                cwd="/work",
                sessionId="s1",
                version="2.1.288",
            )
            out.append((call_id, tool, uid))
        return out

    def hook(self, call_id: str, kind: str = "hook_success") -> str:
        return self.add(
            "attachment", self.last, attachment={"type": kind, "toolUseID": call_id}
        )

    def result(self, call_id: str, owner: str, text: str) -> str:
        block = {"type": "tool_result", "tool_use_id": call_id, "content": text}
        return self.add(
            "user",
            owner,
            message={"role": "user", "content": [block]},
            toolUseResult=text,
        )

    def notice(self) -> str:
        return self.add("attachment", self.last, attachment={"type": "prompt_snapshot"})

    def earlier_pause(self, call_id: str, owner: str) -> None:
        """A call that paused an earlier run; the next run got its result as a message."""
        self.last = owner
        for _ in range(self.rng.randint(0, 2)):
            self.hook(call_id)
        self.hook(call_id, "hook_deferred_tool")
        block = {"type": "tool_result", "tool_use_id": call_id, "content": "ran"}
        self.add("user", self.last, message={"role": "user", "content": [block]})

    def earlier_record(self, call_id: str, owner: str) -> None:
        """An Edit or a Write of an earlier tool step: its record, then Claude Code's
        own line for the turn it went on with."""
        block = {"type": "tool_result", "tool_use_id": call_id, "content": "edited"}
        self.add(
            "user",
            owner,
            message={"role": "user", "content": [block]},
            sourceToolAssistantUUID=owner,
        )
        self.add(
            "user",
            self.last,
            isMeta=True,
            message={"role": "user", "content": "Continue from where you left off."},
        )

    def cost_state(self) -> None:
        """The saved cost total a run ends with (Claude Code 2.1.277 and newer)."""
        self.entries.append({"type": "cost-state", "costUSD": 0.01})

    def compaction(self) -> None:
        """A compaction: a boundary with no parent, then a summary."""
        self.add("system", None, subtype="compact_boundary")
        self.add("user", self.last, message={"role": "user", "content": "summary"})

    def sidechain(self) -> None:
        """A subagent's entries in the main transcript: a chain of their own."""
        parent = None
        for kind in ("user", "assistant"):
            uid = self._uid()
            self.entries.append(
                {
                    "type": kind,
                    "uuid": uid,
                    "parentUuid": parent,
                    "isSidechain": True,
                    "message": {"role": kind, "content": "subagent"},
                }
            )
            parent = uid

    def dangling(self) -> None:
        """A notice whose parent is no longer in the session."""
        self.add(
            "attachment", f"gone-{self.count}", attachment={"type": "prompt_snapshot"}
        )


def conversation(seed: int) -> Conversation:
    """A random conversation that paused at a tool call."""
    rng = random.Random(seed)
    w = Writer(rng)
    if rng.random() < 0.2:
        w.compaction()
    w.prompt("Do the task.")
    for _ in range(rng.randint(0, 4)):  # earlier turns, every call answered
        shape = rng.random()
        if shape < 0.2:  # one call that paused a run, as an Activity
            tool = rng.choice([t for t in ENGINE if t != "Read"] + list(DURABLE))
            [(call_id, _, owner)] = w.calls([tool])
            if tool in RECORDED and rng.random() < 0.5:
                w.earlier_record(call_id, owner)  # an earlier tool step's record
            else:
                w.earlier_pause(call_id, owner)
        else:
            made = w.calls(
                [rng.choice(ENGINE + DURABLE) for _ in range(rng.randint(1, 3))]
            )
            for call_id, _, owner in made:
                if rng.random() < 0.8:
                    w.hook(call_id)
                w.result(call_id, owner, f"result of {call_id}")
        if rng.random() < 0.3:
            w.notice()
        if rng.random() < 0.2:
            w.cost_state()
        if rng.random() < 0.1:
            w.sidechain()
        if rng.random() < 0.05:
            w.dangling()
        if rng.random() < 0.05:
            w.compaction()
    # The message that paused: Reads ran, then the first call that runs as an
    # Activity paused the run, and the hook denied every call after it.
    reads = ["Read"] * rng.randint(0, 2)
    paused_tool = rng.choice([t for t in ENGINE if t != "Read"] + list(DURABLE))
    after = [rng.choice(ENGINE + DURABLE) for _ in range(rng.randint(0, 3))]
    made = w.calls([*reads, paused_tool, *after])
    for call_id, _, owner in made[: len(reads)]:
        w.hook(call_id)
        w.result(call_id, owner, f"read {call_id}")
    paused, _, owner = made[len(reads)]
    first = len(w.entries)
    for _ in range(rng.randint(0, 2)):
        w.hook(paused)
    marker = len(w.entries)
    w.hook(paused, "hook_deferred_tool")
    denied = []
    for call_id, tool, call_owner in made[len(reads) + 1 :]:
        w.result(call_id, call_owner, NOT_RUN)
        denied.append((call_id, tool))
    for _ in range(rng.randint(0, 2)):
        w.notice()
    # Where the next segment continues: the marker, or the last notice after it when
    # no denial came (the runner's ``_resume_point``; here sometimes the marker
    # anyway, which the runner also takes).
    checkpoint = (
        w.entries[marker]["uuid"]
        if denied or rng.random() < 0.3
        else w.entries[-1]["uuid"]
    )
    if rng.random() < 0.3:
        w.cost_state()  # the run that paused ends with its cost total
    return Conversation(
        entries=w.entries,
        checkpoint=checkpoint,
        first=first,
        marker=marker,
        paused=paused,
        paused_tool=paused_tool,
        owner=owner,
        message_calls=[(c, t) for c, t, _ in made],
        denied=denied,
    )


def results_ids(entry: dict[str, Any]) -> list[str]:
    return list(_runner._result_ids(entry))  # type: ignore[reportPrivateUsage]


def call_ids(entries: list[dict[str, Any]]) -> list[str]:
    return [
        str(b["id"])
        for e in entries
        if e.get("type") == "assistant"
        for b in (e.get("message") or {}).get("content") or []
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]


def outcome(rng: random.Random, recorded: bool) -> ToolOutcome:
    record = None
    if recorded and rng.random() < 0.7:  # the step's own record (sometimes none)
        record = {
            "type": "user",
            "uuid": f"rec-{rng.randint(0, 10**9)}",
            "parentUuid": "step-copy",
            "message": {"role": "user", "content": []},
            "toolUseResult": {"filePath": "/f"},
        }
    return ToolOutcome(
        content=rng.choice(["done", "This call was interrupted ...", {"n": 1}]),
        is_error=rng.random() < 0.3,
        entry=record,
    )


@pytest.mark.parametrize("seed", SEEDS[:: len(SEEDS) // 20])
def test_the_generator_writes_what_claude_code_writes(seed: int) -> None:
    """Each call has its assistant entry; the marker is the paused call's; the
    denials follow it; uuids are unique."""
    c = conversation(seed)
    uids = [e["uuid"] for e in c.entries if "uuid" in e]
    assert len(uids) == len(set(uids))
    assert c.entries[c.marker]["attachment"]["type"] == "hook_deferred_tool"
    at = [i for i, _ in c.message_calls].index(c.paused)
    assert c.denied == c.message_calls[at + 1 :]
    assert all(t == "Read" for _, t in c.message_calls[:at])


def test_the_pause_is_found_in_every_conversation() -> None:
    """``_hook_span`` finds the paused call's hook entries from the checkpoint, and
    refuses a checkpoint that is not right after a pause."""
    for seed in SEEDS:
        c = conversation(seed)
        span = _runner._hook_span(c.entries, c.checkpoint)  # type: ignore[reportPrivateUsage]
        assert span == (c.first, c.marker, c.paused), seed
        # Any user or assistant entry after the marker, up to the checkpoint: not a pause.
        later = [
            *copy.deepcopy(c.entries),
            {
                "type": "assistant",
                "uuid": "late",
                "parentUuid": c.checkpoint,
                "message": {"content": []},
            },
        ]
        assert _runner._hook_span(later, "late") is None, seed  # type: ignore[reportPrivateUsage]
        assert _runner._hook_span(c.entries, "nowhere") is None, seed  # type: ignore[reportPrivateUsage]
        # A notice after the denials is not right after the pause: a user entry
        # (a denial) comes between them.
        if c.denied:
            notice = {
                "type": "attachment",
                "uuid": "notice-after",
                "parentUuid": c.entries[-1].get("uuid"),
                "attachment": {"type": "prompt_snapshot"},
            }
            assert _runner._hook_span([*c.entries, notice], "notice-after") is None, (
                seed
            )  # type: ignore[reportPrivateUsage]


def test_the_next_segment_continues_where_the_runner_says() -> None:
    """``_resume_point``: at the marker when denials follow it (the next segment
    puts them before the pause), otherwise at the session's last entry."""
    for seed in SEEDS:
        c = conversation(seed)
        transcript = [e for e in c.entries if _runner._is_transcript(e)]  # type: ignore[reportPrivateUsage]
        want = c.entries[c.marker]["uuid"] if c.denied else transcript[-1]["uuid"]
        assert _runner._resume_point(c.entries, c.paused) == want, seed  # type: ignore[reportPrivateUsage]


def test_a_steps_copy_has_only_answered_calls_and_valid_links() -> None:
    for seed in SEEDS:
        c = conversation(seed)
        before = copy.deepcopy(c.entries)
        context = _runner._step_context(c.entries, c.first)  # type: ignore[reportPrivateUsage]
        assert c.entries == before, seed  # the conversation itself is unchanged
        assert all(e.get("type") != "cost-state" for e in context), seed
        kept = [e["uuid"] for e in context]
        # Entries keep their order, and only assistant entries are left out (and
        # the saved cost totals).
        part = [e for e in c.entries[: c.first] if "uuid" in e]
        order = [e["uuid"] for e in part]
        assert kept == [u for u in order if u in set(kept)], seed
        dropped = set(order) - set(kept)
        assert all(e["type"] == "assistant" for e in part if e["uuid"] in dropped), seed
        # Every call left in the copy has its result in it, and every result its call.
        calls = call_ids(context)
        answered = [i for e in context for i in results_ids(e)]
        assert sorted(calls) == sorted(answered), seed
        assert c.paused not in calls, seed
        # Each entry links to its nearest ancestor in the copy; a link to an entry
        # that was already gone (or to nothing) stays as it was.
        parent_of = {e["uuid"]: e.get("parentUuid") for e in part}
        for entry in context:
            parent = parent_of[entry["uuid"]]
            while parent in dropped:
                parent = parent_of[parent]
            assert entry["parentUuid"] == parent, seed
            assert parent is None or parent in set(kept) or parent not in parent_of


def test_a_record_goes_where_the_result_belongs_or_nowhere() -> None:
    placed_count = 0
    for seed in SEEDS:
        c = conversation(seed)
        rng = random.Random(seed + 10**6)
        results = {c.paused: outcome(rng, c.paused_tool in RECORDED)}
        durable_after = [i for i, t in c.denied if t in DURABLE]
        for call_id in durable_after:
            results[call_id] = ToolOutcome(content={"ran": call_id})
        before = copy.deepcopy(c.entries)
        frozen = copy.deepcopy(results)
        placed = _runner._place(c.entries, c.checkpoint, results)  # type: ignore[reportPrivateUsage]
        assert c.entries == before and results == frozen, seed  # inputs unchanged
        if c.paused_tool not in RECORDED:
            assert placed is None, (
                seed
            )  # a command's or an MCP tool's result: a message
            continue
        assert placed is not None, seed
        placed_count += 1
        seed_entries, delivered = placed
        assert delivered == {c.paused, *durable_after}, seed
        # The conversation up to the pause stays as it is.
        assert seed_entries[: c.first] == c.entries[: c.first], seed
        record = seed_entries[c.first]
        assert record["type"] == "user", seed
        assert record["parentUuid"] == record["sourceToolAssistantUUID"] == c.owner, (
            seed
        )
        assert record["message"]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": c.paused,
                "content": _runner._result_content(results[c.paused]),  # type: ignore[reportPrivateUsage]
                "is_error": results[c.paused].is_error,
            }
        ], seed
        # Then the denials of the calls after it, durable ones with their results.
        after = seed_entries[c.first + 1 :]
        assert [results_ids(e) for e in after] == [[i] for i, _ in c.denied], seed
        for entry, (call_id, tool) in zip(after, c.denied, strict=True):
            text = entry["message"]["content"][0]["content"]
            if tool in DURABLE:
                assert text == json.dumps({"ran": call_id}), seed
            else:
                assert text == NOT_RUN, seed  # Claude calls it again
        # Every call of the conversation has exactly one result, and uuids are unique.
        answered = [i for e in seed_entries for i in results_ids(e)]
        assert sorted(answered) == sorted(call_ids(seed_entries)), seed
        uids = [e["uuid"] for e in seed_entries if "uuid" in e]
        assert len(uids) == len(set(uids)), seed
        # A retry (a session store's copy) after attempt 1 put the record into the
        # session and an unfinished attempt went on: the same conversation again.
        went_on = [
            *copy.deepcopy(c.entries),
            {**copy.deepcopy(record), "parentUuid": c.owner},
            {
                "type": "assistant",
                "uuid": "late-1",
                "parentUuid": record["uuid"],
                "message": {
                    "id": "msg_late",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_late",
                            "name": "Bash",
                            "input": {},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "uuid": "late-2",
                "parentUuid": "late-1",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_late",
                            "content": "x",
                        }
                    ],
                },
            },
        ]
        again = _runner._place(went_on, c.checkpoint, results)  # type: ignore[reportPrivateUsage]
        assert again is not None, seed
        assert [e["uuid"] for e in again[0] if "uuid" in e] == uids, seed
    assert placed_count > 300  # the generator reaches the Edit and Write case often


def test_a_record_without_the_steps_own_is_the_same_in_every_attempt() -> None:
    for seed in SEEDS:
        c = conversation(seed)
        if c.paused_tool not in RECORDED:
            continue
        results = {c.paused: ToolOutcome(content="interrupted", is_error=True)}
        results.update({i: ToolOutcome(content=1) for i, t in c.denied if t in DURABLE})
        one = _runner._place(c.entries, c.checkpoint, results)  # type: ignore[reportPrivateUsage]
        two = _runner._place(copy.deepcopy(c.entries), c.checkpoint, results)  # type: ignore[reportPrivateUsage]
        assert one is not None and two is not None, seed
        r1, r2 = one[0][c.first], two[0][c.first]
        assert (
            r1["uuid"]
            == r2["uuid"]
            == str(uuid.uuid5(_runner._RECORD_NAMESPACE, c.paused))
        ), seed  # type: ignore[reportPrivateUsage]
        assert {k: r1[k] for k in ("cwd", "sessionId", "version")} == {
            "cwd": "/work",
            "sessionId": "s1",
            "version": "2.1.288",
        }, seed


def test_an_input_digest_depends_on_the_input_only() -> None:
    rng = random.Random(7)
    alphabet = 'aé😀\ud83d\n\t"\\ /._-0'
    for _ in range(3000):
        value = {
            "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6))): rng.choice(
                [
                    rng.randint(-5, 5),
                    None,
                    True,
                    "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 9))),
                    [1, "x"],
                    {"k": 0.5},
                ]
            )
            for _ in range(rng.randint(0, 5))
        }
        shuffled = dict(rng.sample(list(value.items()), len(value)))
        assert input_digest(value) == input_digest(shuffled)
        if value:
            key = next(iter(value))
            changed = {**value, key: ["something else"]}
            assert input_digest(changed) != input_digest(value)
