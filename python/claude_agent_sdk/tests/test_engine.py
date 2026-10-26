"""The real Claude Code engine (bundled in claude-agent-sdk) against a local fake Messages API.

These tests drive ClaudeAgentSdkRunner directly, segment by segment, the way the
Workflow does (holding the conversation, by default, or with a session store), so they
pin down the engine behavior the plugin relies on.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import (
    DeferredToolUse,
    HookMatcher,
    InMemorySessionStore,
    ResultMessage,
    SystemMessage,
)

from temporalio.claude_agent_sdk import (
    ClaudeAgentSdkRunner,
    FileSessionStore,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolSpec,
    _runner,
)
from tests.helpers.fake_messages_api import (
    SLOW_MODEL_SECONDS,
    FakeMessagesAPI,
    engine_env,
    history_of,
    start_with_policy,
)
from tests.refund import shop
from tests.refund.policy import refund_policy

ORDER_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}},
    "required": ["order_id"],
}
TOOLS = [
    ToolSpec("look_up_order", "Look up an order.", ORDER_SCHEMA),
    ToolSpec("issue_refund", "Refund money.", {"type": "object"}),
    ToolSpec("email_customer", "Email the customer.", {"type": "object"}),
]


def _make_runner(
    api: FakeMessagesAPI, tmp_path: Path, mode: str = "held"
) -> ClaudeAgentSdkRunner:
    return ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store") if mode == "store" else None,
        cwd=str(tmp_path),
        env=engine_env(api, str(tmp_path / "cfg")),
    )


def _fake_tool(name: str, args: dict[str, Any]) -> ToolOutcome:
    if name == "look_up_order":
        return ToolOutcome(shop.ORDERS[args["order_id"]])
    if name == "issue_refund":
        return ToolOutcome({"refund_id": "R-000001", "amount": args["amount"]})
    return ToolOutcome({"sent": True})


async def _drive(
    runner: ClaudeAgentSdkRunner,
    prompt: str,
    tools: list[ToolSpec],
    limit: int = 8,
    added: list[int] | None = None,
    builtin: list[str] | None = None,
    outputs: list[SegmentOutput] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], SegmentOutput]:
    """Run segments like the Workflow does: pause, run the tool, resume with its result.

    Without a session store, it holds the conversation like the Workflow, and records
    how many entries each segment added in ``added``. Every segment's output goes in
    ``outputs``.
    """
    held = runner._store is None  # type: ignore[reportPrivateUsage]
    transcript: list[dict[str, Any]] = []

    def hold(out: SegmentOutput) -> SegmentOutput:
        nonlocal transcript
        if outputs is not None:
            outputs.append(out)
        if held and not out.is_error:
            assert out.transcript_keep is not None
            transcript = transcript[: out.transcript_keep] + out.transcript_add
            if added is not None:
                added.append(len(out.transcript_add))
        return out

    seg = hold(
        await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt=prompt,
                tools=tools,
                builtin_tools=builtin or [],
                transcript=[] if held else None,
            ),
            1,
        )
    )
    paused: list[tuple[str, dict[str, Any]]] = []
    steps = 0
    while seg.deferred is not None and len(paused) < limit:
        calls = [seg.deferred, *seg.siblings]  # the Workflow runs them all
        paused += [(call.name, call.input) for call in calls]
        steps += 1
        seg = hold(
            await runner.run(
                SegmentInput(
                    session_id=seg.session_id,
                    prompt=None,
                    tools=tools,
                    checkpoint=seg.checkpoint,
                    injected={c.id: _fake_tool(c.name, c.input) for c in calls},
                    builtin_tools=builtin or [],
                    segment_index=steps,
                    transcript=list(transcript) if held else None,
                ),
                1,
            )
        )
    return paused, seg


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_engine_pauses_at_every_durable_call_and_resumes(
    tmp_path: Path, mode: str
) -> None:
    api = start_with_policy(refund_policy)
    try:
        runner = _make_runner(api, tmp_path, mode)
        paused, final = await _drive(
            runner, "Order A-1001 arrived broken, I want my money back.", TOOLS
        )
    finally:
        api.stop()
    assert [name for name, _ in paused] == [
        "look_up_order",
        "issue_refund",
        "email_customer",
    ]
    assert not final.is_error, final.error
    assert final.result is not None and final.result.startswith(
        "Done. Refunded 49.99 EUR"
    )
    assert api.errors == []  # every request followed the real API's tool rules
    assert runner.stub_calls == 0  # the engine never ran a durable tool itself


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_engine_reports_each_steps_own_cost(tmp_path: Path, mode: str) -> None:
    """Each step reports what it spent, so the Workflow's sum is the task's cost.

    Claude Code 2.1.277 and newer save the session's running total in the transcript
    (a ``cost-state`` entry) and start a resumed run from it, so a resumed run would
    report everything spent so far. Here every step makes one model call with the
    same usage: every step costs the same.
    """
    api = start_with_policy(refund_policy)
    outputs: list[SegmentOutput] = []
    try:
        runner = _make_runner(api, tmp_path, mode)
        await _drive(runner, "Order A-1001 arrived broken.", TOOLS, outputs=outputs)
    finally:
        api.stop()
    costs = [out.cost_usd for out in outputs]
    assert len(costs) == 4 and costs[0] > 0, costs
    assert costs == [pytest.approx(costs[0])] * 4, costs
    for out in outputs:  # the Workflow never holds a saved total
        assert all(e.get("type") != "cost-state" for e in out.transcript_add)
    assert api.errors == []


async def test_saved_cost_totals_never_reach_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversation that holds saved totals (as an older version left them, one after
    each step, from Claude Code 2.1.277 and newer) resumes without them, on any
    engine version: the seed leaves them out, the step costs what it spent, not 5
    dollars more, and sends nothing of the conversation again."""
    firsts: list[list[Any]] = []  # each run's first append (a resumed run: its seed)

    class Recording(InMemorySessionStore):
        appended = False

        async def append(self, key: Any, entries: Any) -> None:
            if not self.appended:
                self.appended = True
                firsts.append(list(entries))
            await super().append(key, entries)

    monkeypatch.setattr(_runner, "InMemorySessionStore", Recording)
    api = start_with_policy(refund_policy)
    try:
        runner = _make_runner(api, tmp_path)
        first = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="Order A-1001 arrived broken.",
                tools=TOOLS,
                transcript=[],
            ),
            1,
        )
        assert first.deferred is not None and first.transcript_keep == 0
        saved = {
            "type": "cost-state",
            "sessionId": first.session_id,
            "totalCostUSD": 5.0,
            "modelUsage": {},
        }
        middle = len(first.transcript_add) // 2
        before = first.transcript_add
        transcript = [*before[:middle], saved, *before[middle:], saved]
        second = await runner.run(
            SegmentInput(
                session_id=first.session_id,
                prompt=None,
                tools=TOOLS,
                checkpoint=first.checkpoint,
                injected={
                    first.deferred.id: _fake_tool(
                        first.deferred.name, first.deferred.input
                    )
                },
                transcript=transcript,
            ),
            1,
        )
    finally:
        api.stop()
    assert not second.is_error, second.error
    seed = firsts[1]
    assert [e.get("uuid") for e in seed] == [
        e.get("uuid") for e in first.transcript_add
    ]  # the whole conversation, without the saved total
    assert second.cost_usd == pytest.approx(first.cost_usd)
    assert second.transcript_keep == len(transcript)  # the old totals stay, harmless
    old = {e["uuid"] for e in first.transcript_add if "uuid" in e}
    new = [e["uuid"] for e in second.transcript_add if "uuid" in e]
    assert new and not old.intersection(new)  # nothing of the conversation again
    assert all(e.get("type") != "cost-state" for e in second.transcript_add)


async def test_the_store_loads_sessions_without_saved_cost_totals() -> None:
    """With a session store the engine loads through the plugin's view of it, which
    drops saved totals; the store itself keeps what the engine wrote."""
    store = InMemorySessionStore()
    key = {"project_key": "-work", "session_id": "s1"}
    entries = [
        {"type": "user", "uuid": "u1", "message": {"role": "user", "content": "hi"}},
        {"type": "cost-state", "sessionId": "s1", "totalCostUSD": 1.5},
    ]
    await store.append(key, entries)  # type: ignore[arg-type]
    view = _runner._guarded(store, "s1", None)  # type: ignore[reportPrivateUsage]
    assert await view.load(key) == entries[:1]
    assert await store.load(key) == entries  # type: ignore[arg-type]
    assert _runner._without_cost_state([]) == []  # type: ignore[reportPrivateUsage]


@pytest.mark.parametrize("model", ["quick", "slow"])
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_parallel_calls_all_run_and_every_result_reaches_claude(
    tmp_path: Path, mode: str, model: str
) -> None:
    """Claude calls three tools in one message: all three run, and the next step shows
    Claude every call with its own result. Also when the model takes its time after
    each call, so the engine pauses at the first before the others arrive."""
    ref: list[FakeMessagesAPI] = []

    def all_at_once(body: dict[str, Any]) -> list[dict[str, Any]]:
        uses, _, history = history_of(body)
        if not uses:
            return [
                ref[0].tool_use("look_up_order", {"order_id": "A-1001"}),
                ref[0].tool_use("look_up_order", {"order_id": "A-1002"}),
                ref[0].tool_use("email_customer", {"to": "maria@example.com"}),
            ]
        got = [
            h.content.get("order_id", "sent") if isinstance(h.content, dict) else "?"
            for h in history
            if not h.is_error
        ]
        return [{"type": "text", "text": "FINAL: " + ", ".join(sorted(got))}]

    api = FakeMessagesAPI(all_at_once)
    ref.append(api)
    api.start()
    if model == "slow":
        api.pause_after_call = SLOW_MODEL_SECONDS
    try:
        runner = _make_runner(api, tmp_path, mode)
        paused, final = await _drive(
            runner, "Check orders A-1001 and A-1002, and email Maria.", TOOLS
        )
    finally:
        api.stop()
    assert [name for name, _ in paused] == [
        "look_up_order",
        "look_up_order",
        "email_customer",
    ]
    assert final.result == "FINAL: A-1001, A-1002, sent"
    assert len(api.requests) == 2  # the calls, then the answer: nothing was retried
    assert api.errors == []  # each call had its own result
    assert runner.stub_calls == 0


@pytest.mark.parametrize("model", ["quick", "slow"])
@pytest.mark.parametrize("mode", ["held", "store"])
async def test_builtin_and_durable_calls_mixed_in_one_message(
    tmp_path: Path, mode: str, model: str
) -> None:
    """[Glob, durable, durable, Glob]: the first Glob runs in the engine, both durable
    calls run, and the Glob after them is denied, so Claude calls it again. Also with
    a model that takes its time after each call."""
    (tmp_path / "found.txt").write_text("here")
    ref: list[FakeMessagesAPI] = []

    def glob() -> dict[str, Any]:
        return {
            "type": "tool_use",
            "id": ref[0].next_id("toolu_glob"),
            "name": "Glob",
            "input": {"pattern": "*.txt"},
        }

    def mixed(body: dict[str, Any]) -> list[dict[str, Any]]:
        uses, _, history = history_of(body)
        globs = [h for h in history if h.name == "Glob" and not h.is_error]
        if not uses:
            return [
                glob(),
                ref[0].tool_use("look_up_order", {"order_id": "A-1001"}),
                ref[0].tool_use("look_up_order", {"order_id": "A-1002"}),
                glob(),
            ]
        if len(globs) < 2:
            return [glob()]  # the one that did not run
        orders = [h for h in history if h.name == "look_up_order" and not h.is_error]
        denied = [h for h in history if h.name == "Glob" and h.is_error]
        return [
            {
                "type": "text",
                "text": f"FINAL orders={len(orders)} globs={len(globs)} "
                f"denied={len(denied)}",
            }
        ]

    api = FakeMessagesAPI(mixed)
    ref.append(api)
    api.start()
    if model == "slow":
        api.pause_after_call = SLOW_MODEL_SECONDS
    try:
        runner = _make_runner(api, tmp_path, mode)
        paused, final = await _drive(
            runner, "Look around and check both orders.", TOOLS, builtin=["Glob"]
        )
    finally:
        api.stop()
    assert [args["order_id"] for _, args in paused] == ["A-1001", "A-1002"]
    assert final.result == "FINAL orders=2 globs=2 denied=1"
    assert api.errors == [] and runner.stub_calls == 0


@pytest.mark.parametrize(
    ("reported", "parsed"),
    [
        ("2.1.259 (Claude Code)", (2, 1, 259)),
        ("2.1.273", (2, 1, 273)),
        ("(unknown version)", None),
        ("", None),
    ],
)
def test_engine_versions_are_parsed(
    reported: str, parsed: tuple[int, ...] | None
) -> None:
    assert _runner._version(reported) == parsed


async def test_engine_older_than_the_minimum_is_refused_before_it_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Claude Code 2.1.259 drops a tool result when Claude calls two tools at once (tested)."""

    async def old_version(self: ClaudeAgentSdkRunner) -> str:
        del self
        return "2.1.259 (Claude Code)"

    def must_not_start(*args: Any) -> Any:
        del args
        raise AssertionError("the engine must not start")

    monkeypatch.setattr(ClaudeAgentSdkRunner, "_engine_version", old_version)
    monkeypatch.setattr(_runner, "_engine_messages", must_not_start)
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path),
        env={"ANTHROPIC_API_KEY": "sk-ant-fake-not-real"},
    )
    out = await runner.run(
        SegmentInput(
            session_id=str(uuid.uuid4()), prompt="Refund A-1001.", tools=TOOLS
        ),
        1,
    )
    assert out.is_error and out.deferred is None and out.result is None
    assert out.error is not None and "older than 2.1.273" in out.error


async def test_old_engine_found_at_start_never_hands_back_a_tool_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the version could not be read up front, the engine's own report still stops it."""

    async def unknown_version(self: ClaudeAgentSdkRunner) -> None:
        del self

    async def old_engine(*args: Any) -> AsyncIterator[Any]:
        del args
        yield SystemMessage(subtype="init", data={"claude_code_version": "2.1.259"})
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            stop_reason="tool_deferred",
            deferred_tool_use=DeferredToolUse(
                id="toolu_1", name="mcp__durable__issue_refund", input={}
            ),
        )

    monkeypatch.setattr(ClaudeAgentSdkRunner, "_engine_version", unknown_version)
    monkeypatch.setattr(_runner, "_engine_messages", old_engine)
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path),
        env={"ANTHROPIC_API_KEY": "sk-ant-fake-not-real"},
    )
    out = await runner.run(
        SegmentInput(
            session_id=str(uuid.uuid4()), prompt="Refund A-1001.", tools=TOOLS
        ),
        1,
    )
    assert out.is_error and out.deferred is None  # the refund is never handed back
    assert out.error is not None and "older than 2.1.273" in out.error


@pytest.mark.parametrize(
    ("max_turns", "budget", "stop"),
    [(None, None, None), (1, None, "turns"), (None, 0.0001, "budget")],
    ids=["runs-inside", "max-turns", "max-budget"],
)
async def test_builtin_tools_run_inside_the_segment(
    tmp_path: Path, max_turns: int | None, budget: float | None, stop: str | None
) -> None:
    """A built-in tool runs in the engine; a turn or budget cap ends the segment for good."""
    ref: list[FakeMessagesAPI] = []

    def glob_then_answer(body: dict[str, Any]) -> list[dict[str, Any]]:
        answered = any(
            block.get("type") == "tool_result"
            for message in body.get("messages", [])
            if isinstance(message.get("content"), list)
            for block in message["content"]
        )
        if answered:
            return [{"type": "text", "text": "FINAL: looked around"}]
        return [
            {
                "type": "tool_use",
                "id": ref[0].next_id("toolu_glob"),
                "name": "Glob",
                "input": {"pattern": "*.txt"},
            }
        ]

    api = FakeMessagesAPI(glob_then_answer)
    ref.append(api)
    api.start()
    try:
        runner = _make_runner(api, tmp_path)
        runner._max_budget = budget  # type: ignore[reportPrivateUsage]
        out = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="Look around.",
                tools=TOOLS,
                builtin_tools=["Glob"],
                max_turns=max_turns,
            ),
            1,
        )
    finally:
        api.stop()
    assert out.deferred is None
    if stop is None:
        assert out.result == "FINAL: looked around" and out.checkpoint is not None
    else:
        assert out.is_error and out.error is not None and stop in out.error
    assert api.errors == [] and runner.stub_calls == 0


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_your_hooks_are_asked_in_every_segment(tmp_path: Path, mode: str) -> None:
    """A resumed engine first repeats the result its session paused with. The engine's
    input used to end there, before the resumed turn: a hook from ``extra_options``
    was never asked, and Claude Code refused the call it was asked about."""
    asked: list[str] = []

    async def record(event: Any, tool_use_id: Any, context: Any) -> Any:
        del tool_use_id, context
        asked.append(event.get("tool_name", "?"))
        return {}

    ref: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if not history:
            return [ref[0].tool_use("look_up_order", {"order_id": "A-1001"})]
        if len(history) == 1:
            return [ref[0].call("Glob", {"pattern": "*.nothing"})]
        glob = history[-1]
        return [{"type": "text", "text": f"FINAL {glob.is_error} {glob.content}"}]

    api = FakeMessagesAPI(decide).start()
    ref.append(api)
    held = mode == "held"
    runner = ClaudeAgentSdkRunner(
        session_store=None if held else FileSessionStore(tmp_path / "store"),
        cwd=str(tmp_path),
        env=engine_env(api, str(tmp_path / "cfg")),
        extra_options={
            "hooks": {"PreToolUse": [HookMatcher(matcher=None, hooks=[record])]}
        },
    )
    try:
        first = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="Look up A-1001.",
                tools=TOOLS,
                builtin_tools=["Glob"],
                transcript=[] if held else None,
            ),
            1,
        )
        assert first.deferred is not None
        second = await runner.run(
            SegmentInput(
                session_id=first.session_id,
                prompt=None,
                tools=TOOLS,
                builtin_tools=["Glob"],
                checkpoint=first.checkpoint,
                injected={first.deferred.id: ToolOutcome({"found": True})},
                transcript=first.transcript_add if held else None,
                segment_index=1,
            ),
            1,
        )
    finally:
        api.stop()
    assert "Glob" in asked, asked
    assert second.result == "FINAL False No files found", second.result
    assert api.errors == []


class _Scripted:
    """A streaming client that replays an engine's messages, with the delegated agents
    running after each one, and notes when the runner ended the engine's input."""

    def __init__(self, script: list[tuple[Any, set[str]]], feed: Any) -> None:
        self.script = script
        self.feed = feed
        self.ended_after: int | None = None
        self._query = SimpleNamespace(_inflight_tasks=set[str]())

    async def receive_messages(self) -> AsyncIterator[Any]:
        for i, (message, running) in enumerate(self.script):
            self._query._inflight_tasks = running
            yield message
            if self.ended_after is None and not self.feed._queue.empty():
                self.ended_after = i


def _result(call: str | None = None, *, error: bool = False) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=error,
        num_turns=1,
        session_id="s",
        deferred_tool_use=(
            DeferredToolUse(id=call, name="mcp__durable__x", input={}) if call else None
        ),
    )


_INIT = SystemMessage(subtype="init", data={})
_PAUSE = _result("toolu_1")  # the result a resumed engine repeats before init


@pytest.mark.parametrize(
    ("resumed", "script", "ended_after"),
    [
        (False, [(_INIT, set()), (_result(), set())], 1),
        (True, [(_PAUSE, set()), (_INIT, set()), (_result(), set())], 2),
        (True, [(_PAUSE, set()), (_PAUSE, set())], 1),
        (True, [(_result(error=True), set())], 0),
        (
            False,
            [(_INIT, set()), (_result(), {"a1"}), (_result(), set())],
            2,
        ),
        (False, [(_INIT, set()), (_result(error=True), set())], 1),
    ],
    ids=[
        "new",
        "resumed",
        "two-before-init",
        "error-before-init",
        "agent-running",
        "error",
    ],
)
async def test_the_input_ends_after_the_turns_own_result(
    resumed: bool, script: list[tuple[Any, set[str]]], ended_after: int
) -> None:
    """Only a pause a resumed engine repeats before ``init`` is skipped: an engine
    that fails before it starts its turn must not wait for input forever."""
    feed = _runner._Feed([])  # type: ignore[reportPrivateUsage]
    client = _Scripted(script, feed)
    seen = [
        m
        async for m in _runner._turn_messages(client, feed, resumed)  # type: ignore[reportPrivateUsage]
    ]
    assert len(seen) == len(script)  # it reads until the engine is done
    assert client.ended_after == ended_after


async def test_an_engine_stopped_while_it_exits_leaves_no_resume_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine whose ending was cut short gets SIGTERM, and its resume folder goes,
    also the lines it writes while it exits."""
    monkeypatch.setattr(_runner, "EXIT_GRACE_SECONDS", 0.05)
    folder = tmp_path / "claude-resume-x"
    (folder / "projects").mkdir(parents=True)
    signals: list[str] = []
    process = SimpleNamespace(returncode=None, terminate=lambda: signals.append("TERM"))
    _runner._stop_now(process, SimpleNamespace(config_dir=folder))  # type: ignore[reportPrivateUsage]
    assert signals == ["TERM"] and not folder.exists()
    (folder / "projects").mkdir(parents=True)  # written while it exits
    await asyncio.sleep(0.2)
    assert not folder.exists()
