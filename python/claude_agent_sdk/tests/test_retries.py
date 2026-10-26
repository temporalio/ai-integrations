"""Retries on the real engine: a segment that runs again starts clean from its checkpoint.

A segment attempt can write to the session and then never commit: the Worker dies,
the attempt times out, or the Activity result is lost. The next attempt must not
build on those writes. These tests drive ClaudeAgentSdkRunner segment by segment,
the way the Workflow does, and throw an attempt's result away at the worst moments,
with the conversation in the Workflow ("held", the default) and in a session store.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ResultError

from temporalio.claude_agent_sdk import (
    ClaudeAgentSdkRunner,
    FileSessionStore,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolSpec,
)
from tests.endless.policy import count_policy
from tests.helpers.fake_messages_api import (
    FakeMessagesAPI,
    engine_env,
    history_of,
    policy_decider,
    user_words,
)

TOOLS = [ToolSpec("count", "Count one step.", {"type": "object"})]
pytestmark = pytest.mark.timeout(240)
MODES = ["held", "store"]


class Session:
    """Drives one Claude session segment by segment, like the Workflow.

    When the runner has no session store, it also holds the conversation and splices
    in what each committed segment changed, like the Workflow.
    """

    def __init__(self, runner: ClaudeAgentSdkRunner) -> None:
        self.runner = runner
        self.held = runner._store is None  # type: ignore[reportPrivateUsage]
        self.session_id = str(uuid.uuid4())
        self.checkpoint: str | None = None
        self.transcript: list[dict[str, Any]] = []
        self.index = 0

    def input(
        self,
        prompt: str | None = None,
        injected: dict[str, ToolOutcome] | None = None,
        fork: bool = False,
    ) -> SegmentInput:
        return SegmentInput(
            session_id=self.session_id,
            prompt=prompt,
            tools=TOOLS,
            checkpoint=self.checkpoint,
            injected=injected or {},
            segment_index=self.index,
            fork=fork,
            transcript=list(self.transcript) if self.held else None,
        )

    def commit(self, out: SegmentOutput) -> SegmentOutput:
        assert not out.is_error, out.error
        assert out.checkpoint is not None
        self.session_id, self.checkpoint = out.session_id, out.checkpoint
        if self.held:
            assert out.transcript_keep is not None
            self.transcript = self.transcript[: out.transcript_keep] + list(
                out.transcript_add
            )
        else:
            assert out.transcript_keep is None and out.transcript_add == []
        self.index += 1
        return out

    async def run(self, inp: SegmentInput, attempt: int = 1) -> SegmentOutput:
        return self.commit(await self.runner.run(inp, attempt))

    async def finish(self, out: SegmentOutput) -> SegmentOutput:
        """Run the tool calls (``count`` returns its step) until the final answer."""
        while out.deferred is not None:
            out = await self.run(self.input(injected=results(out)))
        return out


def results(out: SegmentOutput) -> dict[str, ToolOutcome]:
    """``count`` results for every call of a paused message, as the Workflow runs them."""
    calls = [out.deferred, *out.siblings] if out.deferred is not None else []
    return {c.id: ToolOutcome({"n": c.input["n"]}) for c in calls}


def start_api(api_holder: list[FakeMessagesAPI]) -> FakeMessagesAPI:
    api = FakeMessagesAPI(policy_decider(api_holder, count_policy))
    api_holder.append(api)
    return api.start()


def make_runner(
    api: FakeMessagesAPI, tmp_path: Path, mode: str = "held"
) -> ClaudeAgentSdkRunner:
    (tmp_path / "work").mkdir(exist_ok=True)
    return ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store") if mode == "store" else None,
        cwd=str(tmp_path / "work"),
        env=engine_env(api, str(tmp_path / "cfg")),
    )


def conversation(api: FakeMessagesAPI) -> tuple[list[str], list[str]]:
    """The tool calls and the user's words Claude saw in the last request."""
    _, texts, history = history_of(api.requests[-1])
    return [h.id for h in history], user_words(texts)


def check_clean(api: FakeMessagesAPI, runner: ClaudeAgentSdkRunner) -> None:
    assert api.errors == []  # every request followed the real API's tool rules
    assert runner.stub_calls == 0  # the engine never ran a durable tool itself


@pytest.mark.parametrize("mode", MODES)
async def test_retry_after_the_attempt_reached_the_next_pause(
    tmp_path: Path, mode: str
) -> None:
    """The lost attempt paused at the next call; the retry decides again from the checkpoint."""
    holder: list[FakeMessagesAPI] = []
    api = start_api(holder)
    try:
        runner = make_runner(api, tmp_path, mode)
        s = Session(runner)
        first = await s.run(s.input("count to 3"))
        assert first.deferred is not None
        c1 = first.deferred
        retry = s.input(injected={c1.id: ToolOutcome({"n": 1})})
        lost = await runner.run(retry, 1)  # finished, but its result never arrived
        assert lost.deferred is not None
        second = await s.run(retry, 2)
        assert second.deferred is not None
        assert second.deferred.id != lost.deferred.id  # Claude decided again
        assert second.deferred.input == {"n": 2}
        final = await s.finish(second)
    finally:
        api.stop()
    assert final.result == "counted to 3"
    calls, said = conversation(api)
    assert lost.deferred.id not in calls  # the lost attempt left no trace
    assert calls[:2] == [c1.id, second.deferred.id] and len(calls) == 3
    assert said == ["count to 3"]
    check_clean(api, runner)


@pytest.mark.parametrize("mode", MODES)
async def test_retry_after_a_crash_during_the_model_call(
    tmp_path: Path, mode: str
) -> None:
    """The attempt delivered the tool result, then died waiting for Claude."""
    holder: list[FakeMessagesAPI] = []
    api = start_api(holder)
    policy = api.decide
    arrived, release = threading.Event(), threading.Event()
    requests = 0

    def hang_once(body: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal requests
        requests += 1
        if requests == 2:  # the first model call after the tool result
            arrived.set()
            release.wait(30)
        return policy(body)

    api.decide = hang_once
    try:
        runner = make_runner(api, tmp_path, mode)
        s = Session(runner)
        first = await s.run(s.input("count to 3"))
        assert first.deferred is not None
        c1 = first.deferred
        retry = s.input(injected={c1.id: ToolOutcome({"n": 1})})
        crashed = asyncio.ensure_future(runner.run(retry, 1))
        assert await asyncio.to_thread(arrived.wait, 60)
        crashed.cancel()  # the Worker dies in the middle of the model call
        with pytest.raises(asyncio.CancelledError):
            await crashed
        release.set()
        second = await s.run(retry, 2)
        assert second.deferred is not None and second.deferred.input == {"n": 2}
        final = await s.finish(second)
    finally:
        release.set()
        api.stop()
    assert final.result == "counted to 3"
    calls, said = conversation(api)
    assert calls[0] == c1.id and len(calls) == 3
    assert said == ["count to 3"]
    check_clean(api, runner)


@pytest.mark.parametrize("mode", MODES)
async def test_retry_of_a_new_task_on_the_same_session(
    tmp_path: Path, mode: str
) -> None:
    """The second task's first segment is lost after it paused; the retry sends the prompt once."""
    holder: list[FakeMessagesAPI] = []
    api = start_api(holder)
    try:
        runner = make_runner(api, tmp_path, mode)
        s = Session(runner)
        done = await s.finish(await s.run(s.input("count to 2")))
        assert done.result == "counted to 2"
        task = s.input("count to 4")
        lost = await runner.run(task, 1)
        assert lost.deferred is not None
        again = await s.run(task, 2)
        assert again.deferred is not None and again.deferred.input == {"n": 3}
        final = await s.finish(again)
    finally:
        api.stop()
    assert final.result == "counted to 4"
    calls, said = conversation(api)
    assert lost.deferred.id not in calls and len(calls) == 4
    assert said == ["count to 2", "count to 4"]  # the prompt reached Claude once
    check_clean(api, runner)


@pytest.mark.parametrize("mode", MODES)
async def test_next_task_after_a_failed_one_answers_the_waiting_call(
    tmp_path: Path, mode: str
) -> None:
    """A task stopped while Claude waited for a tool result: the next task answers it."""
    holder: list[FakeMessagesAPI] = []
    api = start_api(holder)
    try:
        runner = make_runner(api, tmp_path, mode)
        s = Session(runner)
        first = await s.run(s.input("count to 5"))
        assert first.deferred is not None
        stopped = first.deferred.id
        # The Workflow stopped the task here (for example at max_segments). The next
        # task delivers an error result for the waiting call, with its own prompt.
        out = await s.run(
            s.input(
                "count to 1",
                injected={stopped: ToolOutcome("This tool call did not run.", True)},
                fork=True,
            )
        )
        final = await s.finish(out)
    finally:
        api.stop()
    assert final.result == "counted to 1"
    calls, said = conversation(api)
    assert calls[0] == stopped and len(calls) == 2
    assert said[-1] == "count to 1"
    check_clean(api, runner)


class LosingStore(FileSessionStore):
    """A session store whose writes fail, like a store that is down."""

    async def append(self, key: Any, entries: Any) -> None:
        raise ConnectionError("the store is down")


async def test_a_segment_whose_transcript_was_not_stored_does_not_commit(
    tmp_path: Path,
) -> None:
    holder: list[FakeMessagesAPI] = []
    api = start_api(holder)
    try:
        runner = make_runner(api, tmp_path, "store")
        runner._store = LosingStore(tmp_path / "down")  # type: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError, match="session store"):
            await runner.run(
                SegmentInput(
                    session_id=str(uuid.uuid4()), prompt="count to 1", tools=TOOLS
                ),
                1,
            )
    finally:
        api.stop()


@pytest.mark.parametrize("mode", MODES)
async def test_a_stale_checkpoint_continues_in_a_copy(
    tmp_path: Path, mode: str
) -> None:
    """After a Workflow reset, the session holds later turns than the checkpoint.

    A conversation the Workflow holds goes back with the reset, so it simply continues.
    """
    holder: list[FakeMessagesAPI] = []
    api = start_api(holder)
    try:
        runner = make_runner(api, tmp_path, mode)
        s = Session(runner)
        first = await s.run(s.input("count to 4"))
        assert first.deferred is not None
        c1 = first.deferred
        second = await s.run(s.input(injected={c1.id: ToolOutcome({"n": 1})}))
        assert second.deferred is not None
        c2 = second.deferred
        at_reset = (s.session_id, s.checkpoint, s.index, s.transcript)
        later = await s.run(s.input(injected={c2.id: ToolOutcome({"n": 2})}))
        assert later.deferred is not None  # the run that the reset throws away
        s.session_id, s.checkpoint, s.index, s.transcript = at_reset  # a reset
        before = len(api.requests)
        again = await s.run(s.input(injected={c2.id: ToolOutcome({"n": 2})}))
        assert len(api.requests) - before == 1  # one model call, no loop
        assert again.deferred is not None and again.deferred.input == {"n": 3}
        if mode == "store":
            assert again.session_id != at_reset[0]  # it continued in a copy
        final = await s.finish(again)
    finally:
        api.stop()
    assert final.result == "counted to 4"
    calls, said = conversation(api)
    assert later.deferred.id not in calls and calls[:2] == [c1.id, c2.id]
    assert said == ["count to 4"]
    check_clean(api, runner)


@pytest.mark.parametrize("mode", MODES)
async def test_an_empty_answer_is_a_final_answer(tmp_path: Path, mode: str) -> None:
    """A model reply with no content ends the segment instead of retrying it forever."""
    api = FakeMessagesAPI(lambda body: []).start()
    try:
        runner = make_runner(api, tmp_path, mode)
        out = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()), prompt="Say nothing.", tools=TOOLS
            ),
            1,
        )
    finally:
        api.stop()
    assert not out.is_error and out.deferred is None
    assert out.result == "" and out.checkpoint is not None


@pytest.mark.parametrize(
    ("status", "message", "final"),
    [
        (413, "Request too large.", True),
        (400, "prompt is too long: 250000 tokens > 200000 maximum", True),
        (400, "tools.0.custom.input_schema: JSON schema is invalid", True),
        (404, "model: claude-nonexistent", True),
        (400, "Your credit balance is too low to access the Anthropic API.", False),
        (401, "invalid x-api-key", False),
    ],
    ids=[
        "too-large",
        "prompt-too-long",
        "invalid-request",
        "unknown-model",
        "credit-balance",
        "bad-api-key",
    ],
)
async def test_only_requests_refused_for_good_are_not_retried(
    tmp_path: Path, status: int, message: str, final: bool
) -> None:
    """A request the API will always refuse fails the segment; others let Temporal retry."""
    holder: list[FakeMessagesAPI] = []
    api = start_api(holder)
    api.fail_status, api.fail_message = status, message
    inp = SegmentInput(session_id=str(uuid.uuid4()), prompt="count to 1", tools=TOOLS)
    try:
        runner = make_runner(api, tmp_path)
        if final:
            out = await runner.run(inp, 1)
            assert out.is_error and out.error is not None
        else:
            with pytest.raises(ResultError):  # Temporal retries the segment
                await runner.run(inp, 1)
    finally:
        api.stop()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("retry_at", [None, 2], ids=["in-place", "retried"])
async def test_parallel_calls_in_a_resumed_session_keep_every_result(
    tmp_path: Path, retry_at: int | None, mode: str
) -> None:
    """Claude keeps sending two calls per message; both run, and both results arrive,
    also when the step that delivers them runs again."""
    ref: list[FakeMessagesAPI] = []

    def two_at_a_time(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        done = [h for h in history if h.name == "count" and not h.is_error]
        if len(done) >= 6:
            return [{"type": "text", "text": f"FINAL {len(done)}"}]
        step = len(done) // 2 + 1
        return [
            ref[0].tool_use("count", {"n": step}),
            ref[0].tool_use("count", {"n": step, "again": True}),
        ]

    api = FakeMessagesAPI(two_at_a_time)
    ref.append(api)
    api.start()
    handed: list[int] = []
    try:
        runner = make_runner(api, tmp_path, mode)
        s = Session(runner)
        out = await s.run(s.input("count to 3"))
        while out.deferred is not None and len(handed) < 12:
            handed += [c.input["n"] for c in [out.deferred, *out.siblings]]
            inp = s.input(injected=results(out))
            if s.index == retry_at:  # the first attempt ran, but its result was lost
                lost = await runner.run(inp, 1)
                assert lost.deferred is not None and len(lost.siblings) == 1
                out = await s.run(inp, 2)
            else:
                out = await s.run(inp)
    finally:
        api.stop()
    assert out.result == "FINAL 6"
    assert handed == [1, 1, 2, 2, 3, 3]  # each call was handed to the Workflow once
    missing = [
        r for r in api.requests if "Tool result missing" in str(r.get("messages"))
    ]
    assert missing == []
    check_clean(api, runner)


@pytest.mark.parametrize("mode", MODES)
async def test_a_builtin_call_after_a_pause_waits_for_its_result(
    tmp_path: Path, mode: str
) -> None:
    """[durable, built-in] in one message: the built-in must not run before the paused call's result."""
    ref: list[FakeMessagesAPI] = []

    def count_and_write(body: dict[str, Any]) -> list[dict[str, Any]]:
        uses, _, history = history_of(body)
        counted = any(h.name == "count" and not h.is_error for h in history)
        wrote = any(h.name == "Write" and not h.is_error for h in history)
        write = {
            "type": "tool_use",
            "id": ref[0].next_id("toolu_write"),
            "name": "Write",
            "input": {
                "file_path": str(tmp_path / "work" / f"note-{len(uses)}.txt"),
                "content": "written",
            },
        }
        if not counted:
            return [ref[0].tool_use("count", {"n": 1}), write]
        if not wrote:
            return [write]
        return [{"type": "text", "text": "FINAL"}]

    api = FakeMessagesAPI(count_and_write)
    ref.append(api)
    api.start()
    try:
        runner = make_runner(api, tmp_path, mode)
        s = Session(runner)
        first = await s.run(
            SegmentInput(
                session_id=s.session_id,
                prompt="count, then write a note",
                tools=TOOLS,
                builtin_tools=["Write"],
                transcript=[] if s.held else None,
            )
        )
        assert first.deferred is not None
        assert list((tmp_path / "work").glob("note-*.txt")) == []  # denied, not run
        call = first.deferred
        final = await s.run(
            SegmentInput(
                session_id=s.session_id,
                prompt=None,
                tools=TOOLS,
                builtin_tools=["Write"],
                checkpoint=s.checkpoint,
                injected={call.id: ToolOutcome({"n": 1})},
                segment_index=s.index,
                transcript=list(s.transcript) if s.held else None,
            )
        )
    finally:
        api.stop()
    assert final.result == "FINAL"
    assert len(list((tmp_path / "work").glob("note-*.txt"))) == 1  # written once
    check_clean(api, runner)
