"""How a tool step runs a Claude Code call, and how the next segment continues.

Brian Strauch's bounded native call replay (his hybrid prototype): the step resumes a
copy of the conversation that ends before the call, a stand-in model answers with the
call itself, and Claude Code runs it in an ordinary turn that ends after it. For an
Edit or a Write, the step returns Claude Code's own record of the result, and the next
segment puts it where the call's result belongs, so Claude Code does not check the
call again (anthropics/claude-code#99041), and continues the turn by itself. Other
results go as a message, as before.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    InMemorySessionStore,
    PermissionResultAllow,
    ToolPermissionContext,
    project_key_for_directory,
)

from temporalio.claude_agent_sdk import (
    AgentState,
    ClaudeAgentSdkRunner,
    FileSessionStore,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolSpec,
    ToolStepInput,
    _runner,
    _stand_in,
)
from temporalio.claude_agent_sdk._models import TOOL_CALL_NOT_RUN
from temporalio.converter import DataConverter
from temporalio.exceptions import ApplicationError
from tests.helpers.fake_messages_api import (
    RESUME_LINE,
    SLOW_MODEL_SECONDS,
    FakeMessagesAPI,
    engine_env,
    history_of,
    late_blocks,
    user_words,
)

pytestmark = pytest.mark.timeout(240)
TOOLS = [ToolSpec("count", "Count one step.", {"type": "object"})]
NEXT_TASK = "Now the next task."
INTERRUPTED = (
    "This call was interrupted (its step timed out), so it may have run, in full or "
    "in part. Check its effects before you run it again."
)
"""What the Workflow gives Claude for a step that broke after its call may have run."""
RESUME_SWITCHES = {
    "CLAUDE_CODE_RESUME_INTERRUPTED_TURN": "1",
    "CLAUDE_CODE_RESUME_PROMPT": "Go on, the Worker says.",
}
"""Claude Code's switches for continuing an interrupted turn, as a Worker could set."""


class Turns:
    """A scripted model on the fake API: with ``n`` messages of calls so far, it makes
    the calls of ``turns[n]``; then a final text that lists the results Claude saw.
    After ``NEXT_TASK`` it answers that task."""

    def __init__(self, turns: list[list[tuple[str, dict[str, Any]]]]) -> None:
        self.turns = turns
        self.api = FakeMessagesAPI(self.decide).start()

    def decide(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        _, texts, history = history_of(body)
        if user_words(texts)[-1:] == [NEXT_TASK]:
            return [{"type": "text", "text": "next task done"}]
        made = sum(
            1
            for m in body.get("messages", [])
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        )
        if made < len(self.turns):
            return [self.api.call(name, args) for name, args in self.turns[made]]
        seen = " | ".join(
            f"{h.name}:{'error ' if h.is_error else ''}{str(h.content)[:60]}"
            for h in history
        )
        return [{"type": "text", "text": f"FINAL {seen}"}]


def make_runner(
    tmp_path: Path,
    api: FakeMessagesAPI,
    store: bool = False,
    extra_env: dict[str, str] | None = None,
    **options: Any,
) -> ClaudeAgentSdkRunner:
    (tmp_path / "work").mkdir(exist_ok=True)
    return ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "sessions") if store else None,
        cwd=str(tmp_path / "work"),
        env={**engine_env(api, str(tmp_path / "cfg")), **(extra_env or {})},
        **options,
    )


class Session:
    """Drives segments and tool steps the way the Workflow does, in order."""

    def __init__(
        self,
        runner: ClaudeAgentSdkRunner,
        builtin: list[str],
        activities: list[str],
        store: bool,
    ) -> None:
        self.runner = runner
        self.builtin = builtin
        self.activities = activities
        self.store = store
        self.session_id = str(uuid.uuid4())
        self.checkpoint: str | None = None
        self.conversation: list[dict[str, Any]] = []

    def _input(
        self, prompt: str | None, injected: dict[str, ToolOutcome]
    ) -> SegmentInput:
        return SegmentInput(
            session_id=self.session_id,
            prompt=prompt,
            tools=TOOLS,
            builtin_tools=self.builtin,
            tool_activities=self.activities,
            checkpoint=self.checkpoint,
            injected=injected,
            transcript=None if self.store else self.conversation,
        )

    async def segment(
        self,
        prompt: str | None = None,
        injected: dict[str, ToolOutcome] | None = None,
        attempt: int = 1,
    ) -> SegmentOutput:
        out = await self.runner.run(self._input(prompt, injected or {}), attempt)
        assert not out.is_error, out.error
        self.session_id = out.session_id
        self.checkpoint = out.checkpoint
        if not self.store:
            keep = out.transcript_keep
            assert keep is not None
            self.conversation = [*self.conversation[:keep], *out.transcript_add]
        return out

    async def step(self, out: SegmentOutput) -> ToolOutcome:
        assert out.deferred is not None and out.deferred.kind == "engine", out
        return await self.runner.run_tool_step(
            ToolStepInput(
                session_id=out.session_id,
                checkpoint=out.checkpoint or "",
                call=out.deferred,
                tools=TOOLS,
                builtin_tools=self.builtin,
                transcript=None if self.store else self.conversation,
            ),
            1,
        )


def answered_with(body: dict[str, Any], call_id: str) -> list[dict[str, Any]]:
    """The blocks of the user message that answers ``call_id`` in a model request."""
    for message in body.get("messages", []):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            if any(b.get("tool_use_id") == call_id for b in content):
                return content
    raise AssertionError(f"no result for {call_id}")


def seen_after(body: dict[str, Any], call_id: str) -> list[tuple[str, Any]]:
    """What Claude sees for ``call_id``: the result's error flag, then the lines that
    follow it in the same message (Claude Code's own line, when it goes on by itself),
    without the engine's reminders."""
    blocks = answered_with(body, call_id)
    return [
        ("error" if b.get("is_error") else "result", None)
        if b.get("type") == "tool_result"
        else (str(b.get("type")), b.get("text"))
        for b in blocks
        if not str(b.get("text", "")).lstrip().startswith("<system-reminder>")
    ]


def from_result_on(body: dict[str, Any], call_id: str) -> list[tuple[str, str]]:
    """Every block Claude sees from the call's result to the end, as (role, text),
    without the engine's reminders."""
    messages = body["messages"]
    at = next(
        i
        for i, m in enumerate(messages)
        if m["role"] == "user" and call_id in json.dumps(m)
    )
    seen: list[tuple[str, str]] = []
    for message in messages[at:]:
        if message["role"] not in ("user", "assistant"):
            continue
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for block in content:
            text = str(block.get("text", ""))
            if block.get("type") == "tool_result":
                seen.append((message["role"], "result"))
            elif not text.lstrip().startswith("<system-reminder>"):
                seen.append((message["role"], text))
    return seen


async def stored(runner: ClaudeAgentSdkRunner, session_id: str) -> list[Any]:
    """A session as the runner's session store holds it."""
    store = runner._store  # type: ignore[reportPrivateUsage]
    assert store is not None
    key = {
        "project_key": project_key_for_directory(runner._cwd),  # type: ignore[reportPrivateUsage]
        "session_id": session_id,
    }
    return list(await store.load(key) or [])  # type: ignore[arg-type]


# ---- the real engine ----


async def test_real_engine_a_tool_step_returns_claude_codes_own_record_of_its_call(
    tmp_path: Path,
) -> None:
    """Claude reads a file and writes over it. The step runs the Write once, asks no
    real model, and returns Claude Code's own record of the result without the result
    itself. The next segment puts the record where the call's result belongs: Claude
    sees the result, then only Claude Code's own line for a turn it continues."""
    notes = tmp_path / "work" / "notes.txt"
    model = Turns(
        [
            [("Read", {"file_path": str(notes)})],
            [("Write", {"file_path": str(notes), "content": "new\n"})],
        ]
    )
    runner = make_runner(tmp_path, model.api)
    notes.write_text("old\n")
    session = Session(runner, ["Read", "Write"], ["Write"], store=False)
    try:
        paused = await session.segment("Write the notes.")
        asked = len(model.api.requests)
        outcome = await session.step(paused)
        assert len(model.api.requests) == asked  # the step asked no real model
        assert runner._stand_in.requests >= 1  # type: ignore[reportPrivateUsage]
        committed = list(session.conversation)
        final = await session.segment(injected={paused.deferred.id: outcome})  # type: ignore[union-attr]
    finally:
        model.api.stop()
    call = paused.deferred
    assert call is not None
    assert notes.read_text() == "new\n" and not outcome.is_error
    assert str(outcome.content).startswith("The file ")
    record = outcome.entry
    assert record is not None and record["type"] == "user"
    assert record["message"]["content"] == []  # the result is the outcome's
    assert "toolUseResult" not in record  # no copy of the file in the history
    # The next segment: the record replaces the pause's hook entries.
    owner = next(
        e
        for e in committed
        if e.get("type") == "assistant"
        and any(b.get("id") == call.id for b in e["message"]["content"])
    )
    keep = final.transcript_keep
    assert keep is not None and committed[keep - 1]["uuid"] == owner["uuid"]
    placed = final.transcript_add[0]
    assert placed["uuid"] == record["uuid"] and placed["parentUuid"] == owner["uuid"]
    assert placed["message"]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": outcome.content,
            "is_error": False,
        }
    ]
    assert "toolUseResult" not in placed
    # What Claude saw: the result, then Claude Code's own line, and nothing else.
    blocks = answered_with(model.api.requests[-1], call.id)
    assert [b["type"] for b in blocks] == ["tool_result", "text"]
    assert blocks[1]["text"] == RESUME_LINE and not blocks[0].get("is_error")
    assert final.result is not None and final.result.startswith("FINAL Read:")
    assert "modified since read" not in json.dumps(model.api.requests)
    assert model.api.errors == []

    # Claude Code writes these in each entry; a record the segment writes takes them.
    fields = set(_runner._RECORD_FIELDS) - {"gitBranch"}  # type: ignore[reportPrivateUsage]
    assert fields <= set(owner) and fields <= set(record)


def edit_turns(notes: Path, *edits: tuple[str, str]) -> list[Any]:
    """Claude reads the notes, then makes each edit, one per message."""
    return [
        [("Read", {"file_path": str(notes)})],
        *(
            [("Edit", {"file_path": str(notes), "old_string": a, "new_string": b})]
            for a, b in edits
        ),
    ]


async def test_real_engine_a_commands_result_goes_as_a_message(
    tmp_path: Path,
) -> None:
    """Claude Code does not check a command again when its result arrives, so the
    result goes as a message, as before: no record, and no line Claude did not
    write."""
    model = Turns([[("Bash", {"command": "echo hi", "description": "run"})]])
    runner = make_runner(tmp_path, model.api)
    session = Session(runner, ["Bash"], ["Bash"], store=False)
    try:
        paused = await session.segment("Run it.")
        outcome = await session.step(paused)
        final = await session.segment(injected={paused.deferred.id: outcome})  # type: ignore[union-attr]
    finally:
        model.api.stop()
    assert outcome.entry is None and outcome.content == "hi"
    assert seen_after(model.api.requests[-1], paused.deferred.id) == [("result", None)]  # type: ignore[union-attr]
    assert RESUME_LINE not in json.dumps(model.api.requests)
    assert final.result == "FINAL Bash:hi"
    assert model.api.errors == []


async def test_real_engine_a_command_that_fails_is_its_calls_result(
    tmp_path: Path,
) -> None:
    """A command that exits with an error ran: the step returns its output as an
    error result (Claude sees what it printed), not a failure of the step."""
    effects = (tmp_path / "work" / "effects.log").as_posix()
    command = f"echo ran >> {effects}; echo out; echo err >&2; exit 3"
    model = Turns([[("Bash", {"command": command, "description": "fail"})]])
    runner = make_runner(tmp_path, model.api)
    session = Session(runner, ["Bash"], ["Bash"], store=False)
    try:
        paused = await session.segment("Run it.")
        outcome = await session.step(paused)
        final = await session.segment(injected={paused.deferred.id: outcome})  # type: ignore[union-attr]
    finally:
        model.api.stop()
    assert outcome.is_error and outcome.entry is None
    text = str(outcome.content)
    assert "out" in text and "err" in text and "3" in text, text
    assert Path(effects).read_text() == "ran\n"  # once
    assert final.result is not None and final.result.startswith("FINAL Bash:error ")
    assert model.api.errors == []


@pytest.mark.parametrize("together", [False, True], ids=["alone", "with count"])
async def test_real_engine_with_a_session_store_the_record_goes_into_a_copy(
    tmp_path: Path, together: bool
) -> None:
    """With a session store the record goes into a copy of the session, in place of
    the pause's hook entries (as in the conversation the Workflow holds), also when
    calls of the same message were denied after the pause. The session itself stays
    as it was."""
    out = tmp_path / "work" / "out.txt"
    calls: list[tuple[str, dict[str, Any]]] = [
        ("Write", {"file_path": str(out), "content": "x\n"})
    ]
    if together:
        calls.append(("count", {"n": 1}))
    model = Turns([calls])
    runner = make_runner(tmp_path, model.api, store=True)
    session = Session(runner, ["Write"], ["Write"], store=True)
    try:
        paused = await session.segment("Write it.")
        outcome = await session.step(paused)
        call = paused.deferred
        assert call is not None and outcome.entry is not None
        injected = {call.id: outcome}
        if together:
            assert [s.name for s in paused.siblings] == ["count"]
            injected[paused.siblings[0].id] = ToolOutcome(content={"n": 1})
        final = await session.segment(injected=injected)
        entries = await stored(runner, final.session_id)
        original = await stored(runner, paused.session_id)
    finally:
        model.api.stop()
    assert final.session_id != paused.session_id  # a copy: a new id
    assert [e.get("uuid") for e in entries].count(outcome.entry["uuid"]) == 1
    markers = [_runner._marker_of(e) for e in entries]  # type: ignore[reportPrivateUsage]
    assert call.id not in markers  # the record took the pause's place
    assert outcome.entry["uuid"] not in [e.get("uuid") for e in original]
    assert final.result is not None and final.result.startswith("FINAL Write:")
    assert ("| count:{" in final.result) is together
    assert seen_after(model.api.requests[-1], call.id)[-1] == ("text", RESUME_LINE)
    assert model.api.errors == []


@pytest.mark.parametrize(
    "case",
    ["interrupted", "interrupted, store", "kept without a record"],
)
async def test_real_engine_an_edit_with_no_record_of_its_own_gets_one(
    tmp_path: Path, case: str
) -> None:
    """The Edit ran, but its outcome has no record: the step broke after the edit
    (the Workflow then tells Claude the call was interrupted), or a Worker of an
    older version ran it. The segment writes a record of the same shape, so Claude
    Code does not check the edit again: Claude sees the outcome, not "File has been
    modified since read"."""
    store = "store" in case
    notes = tmp_path / "work" / "notes.txt"
    model = Turns(edit_turns(notes, ("old", "new")))
    runner = make_runner(tmp_path, model.api, store=store)
    notes.write_text("old\n")
    session = Session(runner, ["Read", "Edit"], ["Edit"], store=store)
    try:
        paused = await session.segment("Edit the notes.")
        call = paused.deferred
        assert call is not None
        outcome = await session.step(paused)
        assert notes.read_text() == "new\n" and outcome.entry is not None
        if case.startswith("interrupted"):
            outcome = ToolOutcome(content=INTERRUPTED, is_error=True)
        else:
            outcome = ToolOutcome(content=outcome.content)
        final = await session.segment(injected={call.id: outcome})
    finally:
        model.api.stop()
    interrupted = case.startswith("interrupted")
    assert seen_after(model.api.requests[-1], call.id) == [
        ("error" if interrupted else "result", None),
        ("text", RESUME_LINE),
    ]
    assert "modified since read" not in json.dumps(model.api.requests)
    expected = (
        "Edit:error This call was interrupted" if interrupted else "Edit:The file"
    )
    assert final.result is not None and expected in final.result
    assert model.api.errors == []


async def test_real_engine_an_edit_claude_code_refuses_in_its_step(
    tmp_path: Path,
) -> None:
    """The file changed after the segment paused: in its step, Claude Code refuses
    the Edit before it runs. That refusal is the call's result, with Claude Code's own
    record of it, and Claude sees it."""
    notes = tmp_path / "work" / "notes.txt"
    model = Turns(edit_turns(notes, ("old", "new")))
    runner = make_runner(tmp_path, model.api)
    notes.write_text("old\n")
    session = Session(runner, ["Read", "Edit"], ["Edit"], store=False)
    try:
        paused = await session.segment("Edit the notes.")
        call = paused.deferred
        assert call is not None
        time.sleep(0.05)  # a newer time on the file than the Read's
        notes.write_text("changed by someone else\n")
        outcome = await session.step(paused)
        final = await session.segment(injected={call.id: outcome})
    finally:
        model.api.stop()
    assert outcome.is_error and outcome.entry is not None
    assert notes.read_text() == "changed by someone else\n"
    assert seen_after(model.api.requests[-1], call.id) == [
        ("error", None),
        ("text", RESUME_LINE),
    ]
    assert final.result is not None and "Edit:error " in final.result
    assert model.api.errors == []


async def test_real_engine_edits_after_records_without_their_metadata(
    tmp_path: Path,
) -> None:
    """A record goes without Claude Code's metadata of the call (it can hold the whole
    file). Claude Code needs it neither to go on, nor for the next Edit of the same
    file."""
    notes = tmp_path / "work" / "notes.txt"
    model = Turns(edit_turns(notes, ("one", "ONE"), ("two", "TWO")))
    runner = make_runner(tmp_path, model.api)
    notes.write_text("one\ntwo\n")
    session = Session(runner, ["Read", "Edit"], ["Edit"], store=False)
    try:
        out = await session.segment("Edit the notes.")
        for _ in range(2):
            call = out.deferred
            assert call is not None
            outcome = await session.step(out)
            assert outcome.entry is not None and "toolUseResult" not in outcome.entry
            out = await session.segment(injected={call.id: outcome})
    finally:
        model.api.stop()
    assert notes.read_text() == "ONE\nTWO\n"
    assert out.result is not None and out.result.startswith("FINAL Read:")
    assert out.result.count("Edit:The file") == 2
    text = json.dumps(model.api.requests)
    assert "modified since read" not in text and "has not been read" not in text
    assert model.api.errors == []


@pytest.mark.parametrize("store", [False, True], ids=["held", "store"])
async def test_real_engine_a_read_and_an_edit_in_one_message(
    tmp_path: Path, store: bool
) -> None:
    """The Read runs in the segment and the Edit pauses it. The step's copy keeps the
    Read with its result, so Claude Code runs the Edit; Claude then sees both."""
    notes = tmp_path / "work" / "notes.txt"
    read = ("Read", {"file_path": str(notes)})
    edit = ("Edit", {"file_path": str(notes), "old_string": "old", "new_string": "new"})
    model = Turns([[read, edit]])
    runner = make_runner(tmp_path, model.api, store=store)
    notes.write_text("old\n")
    session = Session(runner, ["Read", "Edit"], ["Edit"], store=store)
    try:
        paused = await session.segment("Edit the notes.")
        call = paused.deferred
        assert call is not None and call.name == "Edit"
        outcome = await session.step(paused)
        final = await session.segment(injected={call.id: outcome})
    finally:
        model.api.stop()
    assert notes.read_text() == "new\n" and not outcome.is_error
    assert final.result is not None and final.result.startswith("FINAL Read:")
    assert "| Edit:The file" in final.result
    assert "modified since read" not in json.dumps(model.api.requests)
    assert model.api.errors == []


@pytest.mark.parametrize("first", ["Edit", "Bash"])
@pytest.mark.parametrize("store", [False, True], ids=["held", "store"])
async def test_real_engine_a_slow_model_keeps_every_call_of_its_message(
    tmp_path: Path, store: bool, first: str
) -> None:
    """The model takes its time after each call of a message, so Claude Code runs the
    first call's hooks and pauses there before the second call arrives: the second
    call's entries come after the pause's deferral marker. The next segment still has
    both calls, the paused one with its result (an Edit's as Claude Code's record) and
    the other with its denial. Before, the copy that ends at the pause lost the second
    call, and Claude never saw it."""
    work = tmp_path / "work"
    notes, effects = work / "notes.txt", work / "effects.log"
    edit: tuple[str, dict[str, Any]] = (
        "Edit",
        {"file_path": str(notes), "old_string": "old", "new_string": "new"},
    )
    bash: tuple[str, dict[str, Any]] = (
        "Bash",
        {"command": f"echo ran >> {effects.as_posix()}", "description": "x"},
    )
    pair: list[tuple[str, dict[str, Any]]] = (
        [edit, bash] if first == "Edit" else [bash, edit]
    )
    model = Turns([[("Read", {"file_path": str(notes)})], pair])
    model.api.pause_after_call = SLOW_MODEL_SECONDS
    runner = make_runner(tmp_path, model.api, store=store)
    notes.write_text("old\n")
    session = Session(runner, ["Read", "Edit", "Bash"], ["Edit", "Bash"], store=store)
    try:
        paused = await session.segment("Go.")
        call = paused.deferred
        assert call is not None and call.name == first
        entries = (
            await stored(runner, paused.session_id) if store else session.conversation
        )
        assert late_blocks(entries)  # the second call came after the pause
        outcome = await session.step(paused)
        final = await session.segment(injected={call.id: outcome})
    finally:
        model.api.stop()
    assert final.result is not None and final.result.startswith("FINAL ")
    seen = final.result.removeprefix("FINAL ").split(" | ")
    assert sorted(s.split(":")[0] for s in seen) == ["Bash", "Edit", "Read"]
    denied = [s.split(":")[0] for s in seen if s.split(":", 1)[1].startswith("error")]
    assert denied == ["Bash" if first == "Edit" else "Edit"]
    assert notes.read_text() == ("new\n" if first == "Edit" else "old\n")
    assert effects.exists() == (first == "Bash")
    assert "modified since read" not in json.dumps(model.api.requests)
    assert model.api.errors == []


@pytest.mark.parametrize("store", [False, True], ids=["held", "store"])
async def test_real_engine_a_slow_model_with_text_after_its_call(
    tmp_path: Path, store: bool
) -> None:
    """A slow model writes a line after its call, in the same message: the line comes
    after the pause. The step runs the call, and Claude sees its whole message and the
    result (before, the line was the checkpoint, and the step failed: the session did
    not pause at the call)."""
    effects = tmp_path / "work" / "effects.log"
    ref: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if history:
            said = "Running it now." in json.dumps(
                [m for m in body.get("messages", []) if m.get("role") == "assistant"]
            )
            return [{"type": "text", "text": f"FINAL {len(history)} {said}"}]
        command = {"command": f"echo ran >> {effects.as_posix()}", "description": "x"}
        return [
            ref[0].call("Bash", command),
            {"type": "text", "text": "Running it now."},
        ]

    api = FakeMessagesAPI(decide).start()
    ref.append(api)
    api.pause_after_call = SLOW_MODEL_SECONDS
    runner = make_runner(tmp_path, api, store=store)
    session = Session(runner, ["Bash"], ["Bash"], store=store)
    try:
        paused = await session.segment("Go.")
        call = paused.deferred
        assert call is not None and call.name == "Bash"
        entries = (
            await stored(runner, paused.session_id) if store else session.conversation
        )
        assert late_blocks(entries)  # the line came after the pause
        outcome = await session.step(paused)
        final = await session.segment(injected={call.id: outcome})
    finally:
        api.stop()
    assert effects.read_text() == "ran\n" and not outcome.is_error
    assert final.result == "FINAL 1 True"
    assert api.errors == []


@pytest.mark.parametrize("attempt", [1, 3])
async def test_real_engine_a_segment_whose_engine_does_not_continue_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempt: int
) -> None:
    """A Claude Code that does not continue the turn by itself waits for a message
    forever. The segment ends its input instead, and fails so that Temporal runs it
    again, with a message that says why; at the last attempt, the task stops."""
    out = tmp_path / "work" / "out.txt"
    model = Turns([[("Write", {"file_path": str(out), "content": "x\n"})]])
    runner = make_runner(tmp_path, model.api)
    session = Session(runner, ["Write"], ["Write"], store=False)
    try:
        paused = await session.segment("Write it.")
        outcome = await session.step(paused)
        monkeypatch.setattr(_runner, "CONTINUE_ENV", {})  # such an engine
        monkeypatch.setattr(_runner, "CONTINUE_START_SECONDS", 3.0)
        inp = session._input(None, {paused.deferred.id: outcome})  # type: ignore[union-attr, reportPrivateUsage]
        started = time.monotonic()
        if attempt < _runner.CONTINUE_ATTEMPTS:
            with pytest.raises(RuntimeError, match="did not continue the turn") as err:
                await runner.run(inp, attempt)
            assert "Retrying" in str(err.value)
        else:
            failed = await runner.run(inp, attempt)
            assert failed.is_error and failed.error is not None
            assert "did not continue the turn" in failed.error
            assert f"did not in {attempt} attempts" in failed.error
    finally:
        model.api.stop()
    assert time.monotonic() - started < 60


async def test_real_engine_a_turn_that_starts_after_its_time_is_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the time to start runs out, the engine's input is ended: hooks and
    in-process tools can no longer answer, so even a turn that then starts and ends
    is not kept. The segment runs again."""
    out = tmp_path / "work" / "out.txt"
    model = Turns([[("Write", {"file_path": str(out), "content": "x\n"})]])
    runner = make_runner(tmp_path, model.api)
    session = Session(runner, ["Write"], ["Write"], store=False)
    try:
        paused = await session.segment("Write it.")
        outcome = await session.step(paused)
        injected = {paused.deferred.id: outcome}  # type: ignore[union-attr]
        with monkeypatch.context() as patch:
            patch.setattr(_runner, "CONTINUE_START_SECONDS", 0.001)
            with pytest.raises(RuntimeError, match="did not continue the turn"):
                await runner.run(session._input(None, injected), 1)  # type: ignore[reportPrivateUsage]
        final = await session.segment(injected=injected)
    finally:
        model.api.stop()
    assert final.result is not None and final.result.startswith("FINAL Write:")


async def test_real_engine_a_continued_turn_may_take_longer_than_its_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the start of the turn is timed: once Claude Code has said something, the
    turn takes as long as it takes."""
    out = tmp_path / "work" / "out.txt"
    model = Turns([[("Write", {"file_path": str(out), "content": "x\n"})]])
    answer = model.decide

    def slow(body: dict[str, Any]) -> list[dict[str, Any]]:
        if history_of(body)[2]:  # the turn that goes on after the result
            time.sleep(10)
        return answer(body)

    model.api.decide = slow
    runner = make_runner(tmp_path, model.api)
    session = Session(runner, ["Write"], ["Write"], store=False)
    try:
        paused = await session.segment("Write it.")
        outcome = await session.step(paused)
        # Long enough for the engine to start on a busy machine, shorter than the
        # model's answer.
        monkeypatch.setattr(_runner, "CONTINUE_START_SECONDS", 6.0)
        final = await session.segment(injected={paused.deferred.id: outcome})  # type: ignore[union-attr]
    finally:
        model.api.stop()
    assert final.result is not None and final.result.startswith("FINAL Write:")


@pytest.mark.parametrize(
    "where", ["held", "store", "store, attempt 2", "held, resume switches on"]
)
async def test_real_engine_a_new_task_after_a_tool_steps_result(
    tmp_path: Path, where: str
) -> None:
    """A task stopped right after its tool steps: the next task's prompt comes with
    the step's record. Claude Code first closes the interrupted turn: Claude sees the
    call's result, Claude Code's answer in Claude's place, then the new task, in one
    turn (also when the Worker's environment asks Claude Code to go on with an
    interrupted turn by itself). Claude Code 2.1.280 and older add their own line for
    an interrupted turn before that answer. In a session store, every attempt puts the
    record into a copy."""
    store = where.startswith("store")
    out = tmp_path / "work" / "out.txt"
    model = Turns([[("Write", {"file_path": str(out), "content": "x\n"})]])
    runner = make_runner(
        tmp_path,
        model.api,
        store=store,
        extra_env=RESUME_SWITCHES if "switches" in where else None,
    )
    session = Session(runner, ["Write"], ["Write"], store=store)
    try:
        paused = await session.segment("Write it.")
        outcome = await session.step(paused)
        asked = len(model.api.requests)
        final = await session.segment(
            NEXT_TASK,
            injected={paused.deferred.id: outcome},  # type: ignore[union-attr]
            attempt=2 if "attempt 2" in where else 1,
        )
    finally:
        model.api.stop()
    assert final.result == "next task done"
    assert len(model.api.requests) == asked + 1  # one turn: the new task's
    _, _, history = history_of(model.api.requests[-1])
    assert [(h.name, h.is_error) for h in history] == [("Write", False)]
    # The result, Claude Code's own line for an interrupted turn on 2.1.280 and older
    # (whatever the Worker's environment says), its answer in Claude's place, then the
    # new task.
    version = _runner._version(await runner._engine_version() or "")  # type: ignore[reportPrivateUsage]
    assert version is not None
    expected = [("user", "result")]
    if version <= (2, 1, 280):
        expected.append(("user", RESUME_LINE))
    expected += [("assistant", "No response requested."), ("user", NEXT_TASK)]
    call_id = paused.deferred.id  # type: ignore[union-attr]
    assert from_result_on(model.api.requests[-1], call_id) == expected
    if store:
        assert final.session_id != paused.session_id  # a copy with the record
    assert model.api.errors == []


@pytest.mark.parametrize("store", [False, True], ids=["held", "store"])
async def test_real_engine_a_pause_with_a_denied_call_after_records(
    tmp_path: Path, store: bool
) -> None:
    """Two Writes as tool steps, then a message whose Bash pauses and whose second
    Write is denied, so the next segment goes on in a copy that puts that denial
    before the pause. Before, with a session store, the records stayed in the
    session beside their pauses' deferral markers, and in that copy Claude Code
    took the paused Bash for interrupted ("[Tool result missing due to internal
    error]") and ran the turn on past the next pause."""
    work = tmp_path / "work"
    effects = (work / "effects.log").as_posix()

    def write(name: str) -> tuple[str, dict[str, Any]]:
        return ("Write", {"file_path": str(work / name), "content": f"{name}\n"})

    bash = ("Bash", {"command": f"echo ran >> {effects}", "description": "run"})
    model = Turns([[write("a.txt")], [write("b.txt")], [bash, write("c.txt")]])
    runner = make_runner(tmp_path, model.api, store=store)
    session = Session(runner, ["Bash", "Write"], ["Bash", "Write"], store=store)
    ran: list[str] = []
    try:
        out = await session.segment("Go.")
        while out.deferred is not None:
            call = out.deferred
            ran.append(call.name)
            outcome = await session.step(out)
            out = await session.segment(injected={call.id: outcome})
    finally:
        model.api.stop()
    assert ran == ["Write", "Write", "Bash"]
    assert out.result is not None and out.result.startswith("FINAL ")
    assert "Tool result missing" not in json.dumps(model.api.requests)
    assert Path(effects).read_text() == "ran\n"
    assert model.api.errors == []


async def test_real_engine_a_store_segment_that_runs_again_after_its_record_went_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attempt 1 went on in a copy with the record, then failed. The retry makes a
    new copy from the session, which never holds the record, so the record is there
    once."""
    out = tmp_path / "work" / "out.txt"
    model = Turns([[("Write", {"file_path": str(out), "content": "x\n"})]])
    runner = make_runner(tmp_path, model.api, store=True)
    session = Session(runner, ["Write"], ["Write"], store=True)
    try:
        paused = await session.segment("Write it.")
        call = paused.deferred
        assert call is not None
        outcome = await session.step(paused)
        assert outcome.entry is not None
        inp = session._input(None, {call.id: outcome})  # type: ignore[reportPrivateUsage]
        with monkeypatch.context() as patch:
            patch.setattr(_runner, "CONTINUE_ENV", {})  # attempt 1 does not go on
            patch.setattr(_runner, "CONTINUE_START_SECONDS", 3.0)
            with pytest.raises(RuntimeError, match="did not continue the turn"):
                await runner.run(inp, 1)
        first = await stored(runner, paused.session_id)
        final = await runner.run(inp, 2)
        copy_entries = await stored(runner, final.session_id)
    finally:
        model.api.stop()
    record = outcome.entry["uuid"]
    assert [e.get("uuid") for e in first].count(record) == 0  # never in place
    assert not final.is_error and final.session_id != paused.session_id
    assert [e.get("uuid") for e in copy_entries].count(record) == 1
    assert final.result is not None and final.result.startswith("FINAL Write:")
    assert (
        len([h for h in history_of(model.api.requests[-1])[2] if h.name == "Write"])
        == 1
    )
    assert model.api.errors == []


async def test_real_engine_warm_engines_and_a_turn_that_goes_on(tmp_path: Path) -> None:
    """With warm engines on, a new engine continues the turn after a Write's record
    (only a new engine reads it); it then pauses at a durable call, stays warm, and
    takes that call's result."""
    out = tmp_path / "work" / "out.txt"
    model = Turns(
        [
            [("Write", {"file_path": str(out), "content": "x\n"})],
            [("count", {"n": 1})],
        ]
    )
    runner = make_runner(tmp_path, model.api, warm_engines=2)
    session = Session(runner, ["Write"], ["Write"], store=False)
    try:
        paused = await session.segment("Write it, then count.")
        assert not runner._warm  # type: ignore[reportPrivateUsage]
        outcome = await session.step(paused)
        counting = await session.segment(injected={paused.deferred.id: outcome})  # type: ignore[union-attr]
        call = counting.deferred
        assert call is not None and call.name == "count"
        assert len(runner._warm) == 1  # type: ignore[reportPrivateUsage]
        final = await session.segment(injected={call.id: ToolOutcome(content={"n": 1})})
        assert not runner._warm  # type: ignore[reportPrivateUsage]
    finally:
        runner._end_all_warm()  # type: ignore[reportPrivateUsage]
        model.api.stop()
    assert final.result is not None and final.result.startswith("FINAL Write:")
    assert "| count:{" in final.result
    assert model.api.errors == []


@pytest.mark.parametrize("mode", ["default", "bypassPermissions"])
@pytest.mark.parametrize("tool", ["Bash", "Edit"])
async def test_real_engine_in_a_tool_step_only_the_hook_lets_the_call_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, mode: str
) -> None:
    """In a tool step, no rule, permission mode or callback from ``extra_options``
    approves the call: with a hook that gives no answer (it crashed), Claude Code
    refuses it, so a step never runs its call without knowing. The step fails as one
    whose call did not run, so the Workflow can try it again."""

    async def allow_all(
        name: str, args: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResultAllow:
        del name, args, context
        return PermissionResultAllow()

    effects = tmp_path / "work" / "effects.log"
    notes = tmp_path / "work" / "notes.txt"
    extra: dict[str, Any] = {"can_use_tool": allow_all}
    if tool == "Bash":
        command = f"echo ran >> {effects.as_posix()}"
        turns = [[("Bash", {"command": command, "description": "run"})]]
        builtin = ["Bash"]
        extra["allowed_tools"] = ["Bash"]
    else:
        turns = edit_turns(notes, ("old", "new"))
        builtin = ["Read", "Edit"]
        extra["permission_mode"] = "acceptEdits"
    if mode == "bypassPermissions":  # the segment runs in it; the step does not
        extra["permission_mode"] = mode
    model = Turns(turns)
    runner = make_runner(tmp_path, model.api, extra_options=extra)
    notes.write_text("old\n")
    session = Session(runner, builtin, [tool], store=False)
    try:
        paused = await session.segment("Go.")
        assert paused.deferred is not None and paused.deferred.name == tool
        monkeypatch.setattr(
            _runner,
            "_hook_entry",
            lambda: {
                "type": "command",
                "command": sys.executable,
                "args": ["-c", "import sys; sys.exit(1)"],
            },
        )
        with pytest.raises(ApplicationError) as failed:
            await session.step(paused)
    finally:
        model.api.stop()
    # It did not run, and a new attempt may run it.
    assert failed.value.type == TOOL_CALL_NOT_RUN and not failed.value.non_retryable
    assert "its hook gave no answer" in failed.value.message
    assert not effects.exists() and notes.read_text() == "old\n"


async def test_real_engine_a_step_whose_record_did_not_reach_its_copy_still_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claude Code's own record of the Write never reached the step's copy of the
    conversation: the step returns one of the same shape, so the next segment (here
    with a session store, where it reads the session only for such results) still
    puts the result where it belongs."""

    class Forgetful(InMemorySessionStore):
        """Keeps the copy the step starts from, and no result written after it."""

        started = False

        async def append(self, key: Any, entries: Any) -> None:
            if self.started:
                entries = [e for e in entries if not _runner._result_ids(e)]  # type: ignore[reportPrivateUsage]
            self.started = True
            await super().append(key, entries)

    out = tmp_path / "work" / "out.txt"
    model = Turns(
        [
            [("Read", {"file_path": str(out)})],
            [("Write", {"file_path": str(out), "content": "new\n"})],
        ]
    )
    runner = make_runner(tmp_path, model.api, store=True)
    out.write_text("old\n")
    session = Session(runner, ["Read", "Write"], ["Write"], store=True)
    try:
        paused = await session.segment("Write it.")
        call = paused.deferred
        assert call is not None
        with monkeypatch.context() as patch:
            patch.setattr(_runner, "InMemorySessionStore", Forgetful)
            outcome = await session.step(paused)
        final = await session.segment(injected={call.id: outcome})
    finally:
        model.api.stop()
    assert out.read_text() == "new\n" and not outcome.is_error
    assert outcome.entry is not None
    assert outcome.entry["uuid"] == str(uuid.uuid5(_runner._RECORD_NAMESPACE, call.id))  # type: ignore[reportPrivateUsage]
    assert seen_after(model.api.requests[-1], call.id) == [
        ("result", None),
        ("text", RESUME_LINE),
    ]
    assert "modified since read" not in json.dumps(model.api.requests)
    assert final.result is not None and "| Write:The file" in final.result
    assert model.api.errors == []


@pytest.mark.parametrize("tool", ["Bash", "Edit", "Write"])
async def test_real_engine_the_workers_own_resume_settings_change_nothing(
    tmp_path: Path, tool: str
) -> None:
    """Claude Code's switches for continuing an interrupted turn, set in the Worker's
    environment, reach neither a segment that sends a message, nor a tool step (its
    copy ends with the Read's result, a turn Claude Code could go on with by itself),
    nor the line Claude sees when a turn goes on. Each step asks the stand-in once."""
    notes = tmp_path / "work" / "notes.txt"
    out = tmp_path / "work" / "out.txt"
    call = {
        "Bash": ("Bash", {"command": "echo hi", "description": "run"}),
        "Edit": (
            "Edit",
            {"file_path": str(notes), "old_string": "notes", "new_string": "edited"},
        ),
        "Write": ("Write", {"file_path": str(out), "content": "x\n"}),
    }[tool]
    model = Turns([[("Read", {"file_path": str(notes)})], [call]])
    runner = make_runner(tmp_path, model.api, extra_env=RESUME_SWITCHES)
    notes.write_text("notes\n")
    session = Session(runner, ["Read", tool], [tool], store=False)
    try:
        paused = await session.segment("Do it.")
        assert paused.deferred is not None
        asked = runner._stand_in.requests  # type: ignore[reportPrivateUsage]
        outcome = await session.step(paused)
        assert runner._stand_in.requests == asked + 1  # type: ignore[reportPrivateUsage]
        final = await session.segment(injected={paused.deferred.id: outcome})
    finally:
        model.api.stop()
    assert final.result is not None and final.result.startswith("FINAL Read:")
    assert f"| {tool}:" in final.result
    expected: list[tuple[str, Any]] = [("result", None)]
    if tool != "Bash":  # Claude Code's own record: the turn goes on by itself
        expected.append(("text", RESUME_LINE))
    assert seen_after(model.api.requests[-1], paused.deferred.id) == expected
    assert "the Worker says" not in json.dumps(model.api.requests)
    assert model.api.errors == []


@pytest.mark.parametrize("own", [False, True], ids=["inherited", "its own"])
async def test_real_engine_an_mcp_server_in_a_tool_step_sees_the_stand_in(
    tmp_path: Path, own: bool
) -> None:
    """An MCP server that Claude Code starts gets the engine's environment: in a tool
    step, ``ANTHROPIC_BASE_URL`` points at the stand-in (unlike a command, which gets
    the Worker's own values back), unless the server's own ``env`` sets it. It also
    sees the plugin's own variables (``TCA_*``)."""
    name = "mcp__env__show"
    server = Path(__file__).parent / "helpers" / "env_mcp_server.py"
    config: dict[str, Any] = {
        "type": "stdio",
        "command": sys.executable,
        "args": [str(server)],
    }
    if own:
        config["env"] = {"ANTHROPIC_BASE_URL": "http://own.example"}
    model = Turns([[(name, {})]])
    runner = make_runner(
        tmp_path,
        model.api,
        extra_options={"mcp_servers": {"env": config}, "allowed_tools": [name]},
    )
    session = Session(runner, [], [name], store=False)
    try:
        paused = await session.segment("Show it.")
        assert paused.deferred is not None and paused.deferred.name == name
        outcome = await session.step(paused)
    finally:
        model.api.stop()
    assert not outcome.is_error and outcome.blocks is not None
    seen = json.loads(outcome.blocks[0]["text"])
    stand_in = runner._stand_in.base_url  # type: ignore[reportPrivateUsage]
    assert seen["ANTHROPIC_BASE_URL"] == ("http://own.example" if own else stand_in)
    assert seen["TCA_HOOK_DIR"]  # the step's folder
    assert model.api.errors == []


@pytest.mark.parametrize("rule", ["ask", "deny", "no rule", "deny the tool"])
async def test_real_engine_deny_and_ask_rules_still_apply_in_a_tool_step(
    tmp_path: Path, rule: str
) -> None:
    """Claude Code checks deny and ask rules after the hook's "allow". A tool step
    never asks a permission callback (only the hook decides there), so an ``ask``
    rule for the tool refuses its call in the step, whatever the callback would say:
    decide on such calls with ``tool_approvals``. Without a rule, the call runs. A
    deny rule for a whole tool takes the tool away from Claude: nothing pauses."""
    asked: list[str] = []

    async def callback(
        name: str, args: dict[str, Any], context: ToolPermissionContext
    ) -> Any:
        del args, context
        asked.append(name)
        return PermissionResultAllow()

    work = tmp_path / "work"
    (work / ".claude").mkdir(parents=True)
    rules: dict[str, list[str]] = {
        "ask": {"ask": ["Bash"]},
        "deny": {"deny": ["Bash(echo:*)"]},
        "no rule": {},
        "deny the tool": {"deny": ["Bash"]},
    }[rule]
    (work / ".claude" / "settings.json").write_text(json.dumps({"permissions": rules}))
    effects = (work / "effects.log").as_posix()
    model = Turns(
        [[("Bash", {"command": f"echo ran >> {effects}", "description": "run"})]]
    )
    runner = make_runner(
        tmp_path,
        model.api,
        extra_options={"can_use_tool": callback, "setting_sources": ["project"]},
    )
    session = Session(runner, ["Bash"], ["Bash"], store=False)
    try:
        paused = await session.segment("Run it.")
        outcome = None if paused.deferred is None else await session.step(paused)
    finally:
        model.api.stop()
    assert asked == []  # neither in the segment (the hook paused it) nor in the step
    assert model.api.errors == []
    if rule == "deny the tool":
        offered = [
            str(t.get("name")) for r in model.api.requests for t in r.get("tools", [])
        ]
        assert "Bash" not in offered and outcome is None
        assert not Path(effects).exists()
        return
    assert outcome is not None
    if rule == "no rule":
        assert not outcome.is_error and Path(effects).read_text() == "ran\n"
    else:
        # Refused (with Claude Code's own reason, which differs by version), as the
        # same step without a rule shows.
        assert outcome.is_error and not Path(effects).exists()


@pytest.mark.parametrize(
    "where", ["in the working directory", "outside it", ".git", ".claude"]
)
async def test_real_engine_an_edit_step_keeps_claude_codes_guard_of_its_own_files(
    tmp_path: Path, where: str
) -> None:
    """The hook's "allow" does not lift Claude Code's own guard of sensitive files: an
    Edit of ``.git/config`` or of ``.claude/settings.json`` is refused in the step. It
    does lift the working directory check: an Edit outside it runs, as a command
    could. To decide on each edit, use ``tool_approvals``."""
    work = tmp_path / "work"
    target = {
        "in the working directory": work / "notes.txt",
        "outside it": tmp_path / "outside.txt",
        ".git": work / ".git" / "config",
        ".claude": work / ".claude" / "settings.json",
    }[where]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"old": 1}\n')
    model = Turns(edit_turns(target, ("old", "new")))
    runner = make_runner(tmp_path, model.api)
    session = Session(runner, ["Read", "Edit"], ["Edit"], store=False)
    try:
        paused = await session.segment("Edit it.")
        outcome = await session.step(paused)
    finally:
        model.api.stop()
    guarded = where in (".git", ".claude")
    assert outcome.is_error is guarded, outcome.content
    assert target.read_text() == ('{"old": 1}\n' if guarded else '{"new": 1}\n')
    assert model.api.errors == []


async def test_real_engine_tool_steps_run_in_plan_mode(tmp_path: Path) -> None:
    """The permission mode does not decide on calls in ``tool_activities``: the hook
    pauses them in the segment before Claude Code's permission check, and the step
    runs them in the default mode. So even ``plan`` mode runs a command and an edit."""
    notes = tmp_path / "work" / "notes.txt"
    effects = (tmp_path / "work" / "effects.log").as_posix()
    model = Turns(
        [
            [("Bash", {"command": f"echo ran >> {effects}", "description": "run"})],
            *edit_turns(notes, ("old", "new")),
        ]
    )
    runner = make_runner(tmp_path, model.api, extra_options={"permission_mode": "plan"})
    notes.write_text("old\n")
    session = Session(runner, ["Bash", "Read", "Edit"], ["Bash", "Edit"], store=False)
    ran: list[str] = []
    try:
        out = await session.segment("Go.")
        while out.deferred is not None:
            call = out.deferred
            ran.append(call.name)
            outcome = await session.step(out)
            assert not outcome.is_error, outcome.content
            out = await session.segment(injected={call.id: outcome})
    finally:
        model.api.stop()
    assert ran == ["Bash", "Edit"]
    assert Path(effects).read_text() == "ran\n" and notes.read_text() == "new\n"
    assert out.result is not None and out.result.startswith("FINAL ")
    assert model.api.errors == []


@pytest.mark.parametrize("change", ["name", "input"])
async def test_a_tool_step_runs_only_the_call_the_segment_reported(
    tmp_path: Path, change: str
) -> None:
    """The step answers with the call as the conversation recorded it. A call the
    conversation does not hold as the segment reported it does not start, and the
    hook lets the call run only with the input the segment reported (what an
    approval saw), as Claude Code is about to run it."""
    effects = (tmp_path / "work" / "effects.log").as_posix()
    command = {"command": f"echo ran >> {effects}", "description": "run"}
    model = Turns([[("Bash", command)]])
    runner = make_runner(tmp_path, model.api)
    session = Session(runner, ["Bash"], ["Bash"], store=False)
    try:
        paused = await session.segment("Run it.")
        assert paused.deferred is not None
        if change == "name":
            paused.deferred.name = "PowerShell"  # not the recorded call
        else:
            paused.deferred.input = {**command, "command": "echo other"}
        with pytest.raises(ApplicationError) as failed:
            await session.step(paused)
    finally:
        model.api.stop()
    assert failed.value.type == TOOL_CALL_NOT_RUN and failed.value.non_retryable
    assert not Path(effects).exists()
    if change == "name":
        assert "did not pause at tool call" in failed.value.message
        assert runner._stand_in.requests == 0  # type: ignore[reportPrivateUsage]
    else:
        assert "only with the input it was given" in failed.value.message


async def test_real_engine_edits_with_relative_paths(tmp_path: Path) -> None:
    """Claude Code makes a relative ``file_path`` absolute before its hook sees the
    call, in the segment (so the segment reports it that way) and in the step alike:
    the step runs the call it was given."""
    work = tmp_path / "work"
    model = Turns(
        [
            [("Read", {"file_path": "notes.txt"})],
            [
                (
                    "Edit",
                    {
                        "file_path": "notes.txt",
                        "old_string": "old",
                        "new_string": "new",
                    },
                )
            ],
            [("Write", {"file_path": "./sub/../made.txt", "content": "made\n"})],
        ]
    )
    runner = make_runner(tmp_path, model.api)
    (work / "notes.txt").write_text("old\n")
    session = Session(runner, ["Read", "Edit", "Write"], ["Edit", "Write"], store=False)
    try:
        out = await session.segment("Edit the notes, then write.")
        for name in ("Edit", "Write"):
            call = out.deferred
            assert call is not None and call.name == name
            assert Path(call.input["file_path"]).is_absolute()  # as reported
            outcome = await session.step(out)
            assert not outcome.is_error, outcome.content
            out = await session.segment(injected={call.id: outcome})
    finally:
        model.api.stop()
    assert (work / "notes.txt").read_text() == "new\n"
    assert (work / "made.txt").read_text() == "made\n"
    assert out.result is not None and out.result.startswith("FINAL Read:")
    assert "modified since read" not in json.dumps(model.api.requests)
    assert model.api.errors == []


async def test_real_engine_notices_before_a_turn_are_not_its_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SessionStart hook in your settings makes Claude Code report its events while
    it waits for a message, before any turn. They do not count as the start of the
    turn, so an engine that waits instead of going on still fails in time."""
    work = tmp_path / "work"
    (work / ".claude").mkdir(parents=True)
    hook = {"type": "command", "command": "echo started"}
    (work / ".claude" / "settings.json").write_text(
        json.dumps({"hooks": {"SessionStart": [{"hooks": [hook]}]}})
    )
    out = work / "out.txt"
    model = Turns([[("Write", {"file_path": str(out), "content": "x\n"})]])
    runner = make_runner(
        tmp_path, model.api, extra_options={"setting_sources": ["project"]}
    )
    session = Session(runner, ["Write"], ["Write"], store=False)
    try:
        paused = await session.segment("Write it.")
        outcome = await session.step(paused)
        monkeypatch.setattr(_runner, "CONTINUE_ENV", {})  # an engine that waits
        monkeypatch.setattr(_runner, "CONTINUE_START_SECONDS", 5.0)
        inp = session._input(None, {paused.deferred.id: outcome})  # type: ignore[union-attr, reportPrivateUsage]
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="did not continue the turn"):
            await asyncio.wait_for(runner.run(inp, 1), 90)
    finally:
        model.api.stop()
    assert time.monotonic() - started < 60


# ---- the conversation, without an engine ----


def entry(kind: str, uid: str, parent: str | None, **fields: Any) -> dict[str, Any]:
    return {"type": kind, "uuid": uid, "parentUuid": parent, **fields}


def call_entry(uid: str, parent: str, call_id: str, name: str = "Edit") -> Any:
    block = {"type": "tool_use", "id": call_id, "name": name, "input": {}}
    return entry(
        "assistant",
        uid,
        parent,
        message={"id": "msg_1", "role": "assistant", "content": [block]},
        cwd="/work",
        sessionId="s1",
        version="2.1.288",
        requestId="req_1",
    )


def result_entry(uid: str, parent: str, call_id: str, text: str = "ok") -> Any:
    block = {"type": "tool_result", "tool_use_id": call_id, "content": text}
    return entry(
        "user",
        uid,
        parent,
        message={"role": "user", "content": [block]},
        toolUseResult=text,
    )


def hook(uid: str, parent: str, kind: str, call_id: str) -> Any:
    return entry(
        "attachment", uid, parent, attachment={"type": kind, "toolUseID": call_id}
    )


def snapshot(uid: str, parent: str) -> Any:
    return entry("attachment", uid, parent, attachment={"type": "prompt_snapshot"})


PROMPT = entry("user", "p", None, message={"role": "user", "content": "go"})


def one_call(name: str = "Edit") -> list[dict[str, Any]]:
    """[Read r, Edit e] in one message: the Read ran, the Edit paused."""
    return [
        PROMPT,
        call_entry("a1", "p", "r", "Read"),
        call_entry("a2", "a1", "e", name),
        hook("h1", "a2", "hook_success", "r"),
        result_entry("u1", "a1", "r", "read it"),
        hook("h2", "u1", "hook_success", "e"),
        hook("m", "h2", "hook_deferred_tool", "e"),
        snapshot("s", "m"),
    ]


def with_sibling() -> list[dict[str, Any]]:
    """[Edit e, count c] in one message: the Edit paused, the count was denied."""
    return [
        PROMPT,
        call_entry("a1", "p", "e"),
        call_entry("a2", "a1", "c", "mcp__durable__count"),
        hook("h1", "a2", "hook_success", "e"),
        hook("m", "h1", "hook_deferred_tool", "e"),
        result_entry("d", "a2", "c", "Not an error. This call did not run yet"),
        snapshot("s", "d"),
    ]


def record(uid: str = "rec") -> dict[str, Any]:
    return {
        "type": "user",
        "uuid": uid,
        "parentUuid": "step-copy",
        "sourceToolAssistantUUID": "step-copy",
        "message": {"role": "user", "content": []},
        "toolUseResult": {"filePath": "/f"},
    }


def test_a_tool_steps_copy_of_the_conversation_ends_before_its_call() -> None:
    """The calls that ran before keep their results; the paused call and the calls
    after it go. An entry whose parent went links to that entry's parent; the other
    links stay, also an entry with none (where a compaction starts)."""
    entries = one_call()
    first = _runner._hook_span(entries, "s")  # type: ignore[reportPrivateUsage]
    assert first == (5, 6, "e")
    context = _runner._step_context(entries, 5)  # type: ignore[reportPrivateUsage]
    assert [e["uuid"] for e in context] == ["p", "a1", "h1", "u1"]
    assert [e["parentUuid"] for e in context] == [None, "p", "a1", "a1"]
    assert entries == one_call()  # the conversation itself is unchanged
    context = _runner._step_context(with_sibling(), 3)  # type: ignore[reportPrivateUsage]
    assert [e["uuid"] for e in context] == ["p"]
    compacted = [
        entry("system", "b", None, subtype="compact_boundary"),
        entry("user", "sum", "b", message={"role": "user", "content": "summary"}),
        *one_call()[1:],
    ]
    compacted[2] = {**compacted[2], "parentUuid": "sum"}
    context = _runner._step_context(compacted, 6)  # type: ignore[reportPrivateUsage]
    assert [(e["uuid"], e["parentUuid"]) for e in context] == [
        ("b", None),
        ("sum", "b"),
        ("a1", "sum"),
        ("h1", "a1"),
        ("u1", "a1"),
    ]


def test_a_steps_copy_takes_data_nested_as_deep_as_temporal_carries_it() -> None:
    """``copy.deepcopy`` takes two Python frames per level: on a call input nested
    700 levels deep it failed, so every later tool step failed. Temporal's own
    converter carries such data, and so do the step's copy and its call."""
    deep: Any = "x"
    for _ in range(700):
        deep = [deep]
    assert DataConverter.default.payload_converter.to_payloads([deep])
    entries = one_call()
    entries[1] = json.loads(json.dumps(entries[1]))
    entries[1]["message"]["content"][0]["input"] = {"deep": deep}  # the Read
    entries[2] = json.loads(json.dumps(entries[2]))
    entries[2]["message"]["content"][0]["input"] = {"deep": deep}  # the Edit
    context = _runner._step_context(entries, 5)  # type: ignore[reportPrivateUsage]
    assert context[1]["message"]["content"][0]["input"] == {"deep": deep}
    block = _runner._recorded_call(entries[:5], "e")  # type: ignore[reportPrivateUsage]
    assert block is not None and block["input"] == {"deep": deep}
    stand_in = _stand_in.StandInModel()
    call = stand_in.serve(block)
    assert call.block == block and stand_in.done(call) is False


def test_the_record_goes_where_the_calls_result_belongs() -> None:
    entries = one_call()
    outcome = ToolOutcome(content="edited", entry=record())
    placed = _runner._place(entries, "s", {"e": outcome})  # type: ignore[reportPrivateUsage]
    assert placed is not None
    seed, delivered = placed
    assert [e["uuid"] for e in seed] == ["p", "a1", "a2", "h1", "u1", "rec"]
    assert delivered == {"e"}
    rec = seed[-1]
    assert rec["parentUuid"] == rec["sourceToolAssistantUUID"] == "a2"
    assert rec["message"]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "e",
            "content": "edited",
            "is_error": False,
        }
    ]
    assert rec["toolUseResult"] == {"filePath": "/f"}
    assert outcome.entry == record()  # the outcome itself is unchanged
    # With a call of the same message denied after the pause: its real result follows.
    results = {"e": outcome, "c": ToolOutcome(content={"n": 1})}
    placed = _runner._place(with_sibling(), "m", results)  # type: ignore[reportPrivateUsage]
    assert placed is not None
    seed, delivered = placed
    assert [e["uuid"] for e in seed] == ["p", "a1", "a2", "rec", "d"]
    assert delivered == {"e", "c"}
    assert seed[-1]["message"]["content"][0]["content"] == '{"n": 1}'


@pytest.mark.parametrize("case", ["no record", "uuid taken", "not a user entry"])
def test_without_a_record_of_its_own_the_segment_writes_one(case: str) -> None:
    """A step that failed, a rejected call, or an older Worker's step: no record of
    Claude Code's own, so one of the same shape takes its place, with the fields
    Claude Code writes in each entry from the call's own entry, and the same uuid
    in every attempt."""
    outcome = ToolOutcome(content=INTERRUPTED, is_error=True, entry=record())
    if case == "no record":
        outcome.entry = None
    elif case == "uuid taken":
        outcome.entry = record("u1")
    else:
        outcome.entry = {**record(), "type": "assistant"}
    placed = _runner._place(one_call(), "s", {"e": outcome})  # type: ignore[reportPrivateUsage]
    assert placed is not None
    seed, delivered = placed
    assert delivered == {"e"} and [e["uuid"] for e in seed[:-1]] == [
        "p",
        "a1",
        "a2",
        "h1",
        "u1",
    ]
    rec = seed[-1]
    stamp = rec.pop("timestamp")
    assert stamp.endswith("Z") and "T" in stamp
    assert rec == {
        "cwd": "/work",
        "sessionId": "s1",
        "version": "2.1.288",
        "type": "user",
        "uuid": str(uuid.uuid5(_runner._RECORD_NAMESPACE, "e")),  # type: ignore[reportPrivateUsage]
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "e",
                    "content": INTERRUPTED,
                    "is_error": True,
                }
            ],
        },
        "parentUuid": "a2",
        "sourceToolAssistantUUID": "a2",
    }
    again = _runner._place(one_call(), "s", {"e": outcome})  # type: ignore[reportPrivateUsage]
    assert again is not None and again[0][-1]["uuid"] == rec["uuid"]


def test_entries_a_later_run_wrote_are_not_moved() -> None:
    """In a session store, an unfinished attempt may have gone on after the record
    (in place): a copy that ends at the checkpoint takes none of that."""
    entries = [
        *one_call(),
        {**record(), "parentUuid": "a2"},
        call_entry("a3", "rec", "n", "Bash"),
        result_entry("u3", "a3", "n"),
    ]
    outcome = ToolOutcome(content="edited", entry=record())
    placed = _runner._place(entries, "s", {"e": outcome})  # type: ignore[reportPrivateUsage]
    assert placed is not None
    assert [e["uuid"] for e in placed[0]] == ["p", "a1", "a2", "h1", "u1", "rec"]


@pytest.mark.parametrize(
    "case",
    [
        "a command",
        "an MCP tool",
        "a result for another call too",
        "no result for the paused call",
        "not a pause",
        "unknown checkpoint",
    ],
)
def test_the_record_is_only_placed_where_it_belongs(case: str) -> None:
    """Otherwise the results go as a message, as before (``_deliver``)."""
    entries = one_call()
    checkpoint = "s"
    results = {"e": ToolOutcome(content="edited", entry=record())}
    if case == "a command":
        entries = one_call("Bash")
    elif case == "an MCP tool":
        entries = one_call("mcp__github__create_issue")
    elif case == "a result for another call too":
        results["x"] = ToolOutcome(content="other")
    elif case == "no result for the paused call":
        results = {"r": ToolOutcome(content="edited", entry=record())}
    elif case == "not a pause":
        entries.append(entry("assistant", "f", "s", message={"content": []}))
        checkpoint = "f"
    else:
        checkpoint = "nowhere"
    assert _runner._place(entries, checkpoint, results) is None  # type: ignore[reportPrivateUsage]


def test_a_record_never_carries_the_calls_metadata() -> None:
    """An Edit's metadata holds the whole file as it was; Claude Code does not need it
    to go on, so the record the step returns leaves it out, and the result too (the
    outcome carries that)."""
    given = {
        **record(),
        "message": {"role": "user", "content": ["the result"]},
        "toolUseResult": {"filePath": "/f", "originalFile": "secret=1\n"},
    }
    kept = _runner._result_record(given)  # type: ignore[reportPrivateUsage]
    assert kept["message"] == {"role": "user", "content": []}
    assert "toolUseResult" not in kept
    assert {k: v for k, v in kept.items() if k != "message"} == {
        k: v for k, v in given.items() if k not in ("message", "toolUseResult")
    }
    # Not changed in place.
    assert given["message"]["content"] == ["the result"]
    assert given["toolUseResult"]["originalFile"] == "secret=1\n"


def test_a_segment_that_never_continues_stops_after_its_attempts() -> None:
    """Temporal runs the segment again until ``CONTINUE_ATTEMPTS``; then the task
    stops with an error that says what to change, instead of trying forever."""
    err = _runner._not_continued("9.9.9")  # type: ignore[reportPrivateUsage]
    assert "Claude Code 9.9.9 did not continue the turn" in str(err)
    assert _runner.RESUME in str(err) and "from 2.1.273 to 2.1.295" in str(err)
    for attempt in range(1, _runner.CONTINUE_ATTEMPTS):
        with pytest.raises(RuntimeError, match="Retrying"):
            _runner._not_continued_output("s1", err, attempt)  # type: ignore[reportPrivateUsage]
    out = _runner._not_continued_output("s1", err, _runner.CONTINUE_ATTEMPTS)  # type: ignore[reportPrivateUsage]
    assert out.is_error and out.session_id == "s1" and out.error is not None
    assert f"did not in {_runner.CONTINUE_ATTEMPTS} attempts" in out.error
    assert "tool_activities" in out.error


async def test_the_time_to_start_a_turn_starts_again_with_each_notice() -> None:
    """A notice before the turn (a slow SessionStart hook's events, for example)
    shows the engine is still getting ready; once the turn starts, nothing more is
    timed. Silence for the whole time ends the engine's input."""
    feed = _runner._Feed([])  # type: ignore[reportPrivateUsage]
    watch = _runner._StartWatch(feed, 0.3)  # type: ignore[reportPrivateUsage]
    for _ in range(3):
        await asyncio.sleep(0.2)
        watch.waiting()
    assert not watch.fired  # 0.6 s in all, never 0.3 s without a notice
    await asyncio.sleep(0.5)
    assert watch.fired  # silent too long: the input ended
    watch.waiting()
    assert watch.fired
    started = _runner._StartWatch(_runner._Feed([]), 0.2)  # type: ignore[reportPrivateUsage]
    started.heard()
    started.waiting()  # a notice after the start changes nothing
    await asyncio.sleep(0.4)
    assert not started.fired


def test_a_pending_record_moves_to_the_next_run() -> None:
    """Continue-As-New: a result that waits for the next segment keeps its record."""
    state = AgentState(
        session_id="s1",
        pending={"e": ToolOutcome(content="edited", entry=record())},
    )
    converter = DataConverter.default.payload_converter
    back = converter.from_payloads(converter.to_payloads([state]), [AgentState])[0]
    assert isinstance(back, AgentState)
    assert back.pending["e"].entry == record()
    assert back.pending["e"].content == "edited"
