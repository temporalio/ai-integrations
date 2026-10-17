"""Warm engines (``warm_engines``): an engine that paused takes its session's next segment.

The durable boundary stays the pause: the next segment continues in the running engine
only on the same Worker, at the same checkpoint, with the paused call's result, within
``warm_seconds``. In any other case it starts a new engine from the checkpoint, as
without warm engines. With the conversation in the Workflow, the engine that starts it
never stays warm (it writes the conversation to its own config folder), so the second
step starts a new engine too. The engine is the real Claude Code with the local fake model;
segments run like the Workflow runs them (``_drive``), or in real Workflows on one or
two Workers, and on Worker processes that are killed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import sys
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import (
    DeferredToolUse,
    ResultError,
    ResultMessage,
    SessionKey,
    SystemMessage,
    project_key_for_directory,
)

from temporalio.claude_agent_sdk import (
    ClaudeAgentPlugin,
    ClaudeAgentSdkRunner,
    FileSessionStore,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    _defer_hook,
    _runner,
)
from temporalio.client import Client
from temporalio.worker import Worker
from tests.conftest import wait_for_approval
from tests.helpers.fake_messages_api import (
    FakeMessagesAPI,
    engine_env,
    history_of,
    start_with_policy,
)
from tests.helpers.workers import FAIL_FAST
from tests.refund import shop
from tests.refund.activities import ALL as SHOP_TOOLS
from tests.refund.policy import refund_policy
from tests.refund.workflows import MANAGER, RefundAgentWorkflow
from tests.test_crash import kill, start_worker, wait_until
from tests.test_engine import TOOLS, _drive, _fake_tool
from tests.test_workflow_engine import shared_settings
from tests.warm.activities import RAN, echo
from tests.warm.workflows import EchoRoundsWorkflow

pytestmark = pytest.mark.timeout(240)

PROMPT = "Order A-1001 arrived broken, I want my money back."
KEY = {"ANTHROPIC_API_KEY": "sk-ant-fake-not-real"}


class Counting(ClaudeAgentSdkRunner):
    """Counts engine runs: new engines, and segments a warm engine took."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cold = 0
        self.warm = 0
        self.wait = 0.0
        """Seconds to wait before each segment that brings a result (a slow tool)."""

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        if inp.injected and self.wait:
            await asyncio.sleep(self.wait)
        return await super().run(inp, attempt)

    async def _run_engine(self, *args: Any, warm: Any = None, **kwargs: Any) -> Any:
        if warm is None:
            self.cold += 1
        else:
            self.warm += 1
        return await super()._run_engine(*args, warm=warm, **kwargs)


def counting(
    api: FakeMessagesAPI,
    tmp_path: Path,
    mode: str,
    *,
    config: str = "cfg",
    **options: Any,
) -> Counting:
    """A runner in ``tmp_path``; runners with another ``config`` are other machines."""
    return Counting(
        session_store=FileSessionStore(tmp_path / "store") if mode == "store" else None,
        cwd=str(tmp_path),
        env=engine_env(api, str(tmp_path / config)),
        **options,
    )


def echo_rounds(count: int) -> FakeMessagesAPI:
    """Claude calls ``echo`` once per turn, ``count`` times, then says DONE."""
    holder: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if len(history) >= count:
            return [{"type": "text", "text": f"DONE {len(history)}"}]
        return [holder[0].tool_use("echo", {"n": len(history)})]

    api = FakeMessagesAPI(decide)
    holder.append(api)
    return api.start()


def _stat(pid: int) -> list[str] | None:
    """The fields of ``/proc/<pid>/stat`` after the name: state, parent, ..."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None


def engines_of(worker: int) -> set[tuple[int, str]] | None:
    """A Worker's running child processes (its engines), with their start times.

    Linux only (it reads ``/proc``); None elsewhere.
    """
    if not sys.platform.startswith("linux"):
        return None
    found: set[tuple[int, str]] = set()
    for path in Path("/proc").iterdir():
        fields = _stat(int(path.name)) if path.name.isdigit() else None
        if fields and fields[1] == str(worker) and fields[0] not in "ZX":
            found.add((int(path.name), fields[19]))
    return found


def still_running(processes: set[tuple[int, str]]) -> set[tuple[int, str]]:
    """Those of ``processes`` that still run (same start time, not a zombie)."""
    alive: set[tuple[int, str]] = set()
    for pid, started in processes:
        fields = _stat(pid)
        if fields and fields[19] == started and fields[0] not in "ZX":
            alive.add((pid, started))
    return alive


async def all_ended() -> None:
    """Warm engines end in the background: wait until they did."""
    while _runner._ending:  # type: ignore[reportPrivateUsage]
        await asyncio.gather(*list(_runner._ending))  # type: ignore[reportPrivateUsage]


def shapes(api: FakeMessagesAPI) -> list[list[tuple[str, list[str]]]]:
    """What each model request showed Claude: roles and block types, ids left out."""
    out = []
    for body in api.requests:
        if not any(t["name"].startswith("mcp__durable__") for t in body["tools"]):
            continue
        out.append(
            [
                (
                    m["role"],
                    [
                        b.get("type", "text") if isinstance(b, dict) else "text"
                        for b in (
                            m["content"]
                            if isinstance(m["content"], list)
                            else [m["content"]]
                        )
                    ],
                )
                for m in body["messages"]
            ]
        )
    return out


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_paused_engine_takes_the_next_segment(
    tmp_path: Path, mode: str
) -> None:
    """One warm engine for the task; Claude sees what a new engine per step shows;
    each step still reports its own cost; nothing stays warm after the answer."""
    outputs: list[SegmentOutput] = []
    cold_api = start_with_policy(refund_policy)
    warm_api = start_with_policy(refund_policy)
    try:
        (tmp_path / "cold").mkdir()
        (tmp_path / "warm").mkdir()
        cold = counting(cold_api, tmp_path / "cold", mode, warm_engines=0)
        _, cold_final = await _drive(cold, PROMPT, TOOLS)
        runner = counting(warm_api, tmp_path / "warm", mode, warm_engines=2)
        paused, final = await _drive(runner, PROMPT, TOOLS, outputs=outputs)
        await all_ended()
    finally:
        cold_api.stop()
        warm_api.stop()
    assert [name for name, _ in paused] == [
        "look_up_order",
        "issue_refund",
        "email_customer",
    ]
    assert final.result == cold_final.result and final.result is not None
    assert (cold.cold, cold.warm) == (4, 0)
    assert (runner.cold, runner.warm) == ((2, 2) if mode == "held" else (1, 3))
    assert shapes(warm_api) == shapes(cold_api)
    costs = [out.cost_usd for out in outputs]
    assert costs[0] > 0 and costs == [pytest.approx(costs[0])] * 4, costs
    assert runner._warm == {}  # type: ignore[reportPrivateUsage]
    assert warm_api.errors == [] and runner.stub_calls == 0


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_wait_longer_than_warm_seconds_starts_a_new_engine(
    tmp_path: Path, mode: str
) -> None:
    api = start_with_policy(refund_policy)
    try:
        runner = counting(api, tmp_path, mode, warm_engines=2, warm_seconds=0.05)
        runner.wait = 0.5
        paused, final = await _drive(runner, PROMPT, TOOLS)
        await all_ended()
    finally:
        api.stop()
    assert len(paused) == 3 and final.result and final.result.startswith("Done.")
    assert (runner.cold, runner.warm) == (4, 0)
    assert runner._warm == {}  # type: ignore[reportPrivateUsage]
    assert api.errors == []


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_retried_segment_starts_a_new_engine(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second step fails after its warm turn, like a Worker that loses the reply:
    the retry resumes from the checkpoint in a new engine, and the engine that took
    the failed attempt is gone (it went past the checkpoint)."""
    api = start_with_policy(refund_policy)
    original = ClaudeAgentSdkRunner._segment_output  # type: ignore[reportPrivateUsage]
    failed: list[Any] = []

    async def fail_once(self: Any, *args: Any) -> SegmentOutput:
        out = await original(self, *args)
        warm = args[7]
        if warm is not None and not failed:
            failed.append(warm.client._transport._process)
            raise RuntimeError("injected: the reply was lost")
        return out

    monkeypatch.setattr(ClaudeAgentSdkRunner, "_segment_output", fail_once)

    class Retrying(Counting):
        async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
            try:
                return await super().run(inp, attempt)
            except RuntimeError as err:
                if "injected" not in str(err):
                    raise
                return await super().run(inp, attempt + 1)  # as Temporal would

    try:
        runner = Retrying(
            session_store=(
                FileSessionStore(tmp_path / "store") if mode == "store" else None
            ),
            cwd=str(tmp_path),
            env=engine_env(api, str(tmp_path / "cfg")),
            warm_engines=2,
        )
        paused, final = await _drive(runner, PROMPT, TOOLS)
        await all_ended()
    finally:
        api.stop()
    assert [name for name, _ in paused] == [
        "look_up_order",
        "issue_refund",
        "email_customer",
    ]
    assert final.result and final.result.startswith("Done. Refunded 49.99 EUR")
    assert len(failed) == 1 and failed[0].returncode is not None
    # The failed warm attempt, and the retry in a new engine:
    assert (runner.cold, runner.warm) == ((3, 2) if mode == "held" else (2, 3))
    assert api.errors == []


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_calls_in_one_message_do_not_stay_warm(tmp_path: Path, mode: str) -> None:
    """The results of the calls denied after the pause go into the conversation, which
    only a new engine reads."""
    ref: list[FakeMessagesAPI] = []

    def all_at_once(body: dict[str, Any]) -> list[dict[str, Any]]:
        uses, _, history = history_of(body)
        if not uses:
            return [
                ref[0].tool_use("look_up_order", {"order_id": "A-1001"}),
                ref[0].tool_use("look_up_order", {"order_id": "A-1002"}),
            ]
        got = sorted(h.content.get("order_id", "?") for h in history if not h.is_error)
        return [{"type": "text", "text": "FINAL: " + ", ".join(got)}]

    api = FakeMessagesAPI(all_at_once)
    ref.append(api)
    api.start()
    try:
        runner = counting(api, tmp_path, mode, warm_engines=2)
        paused, final = await _drive(runner, "Check A-1001 and A-1002.", TOOLS)
        await all_ended()
    finally:
        api.stop()
    assert [name for name, _ in paused] == ["look_up_order", "look_up_order"]
    assert final.result == "FINAL: A-1001, A-1002"
    assert runner.warm == 0 and runner._warm == {}  # type: ignore[reportPrivateUsage]
    assert api.errors == []


class StubProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = 0

    def terminate(self) -> None:
        self.terminated += 1


class StubClient:
    """Stands in for the SDK's streaming client of a warm engine."""

    def __init__(
        self, tmp_path: Path, name: str, messages: list[Any] | None = None
    ) -> None:
        self.name = name
        self.messages = messages or []
        self.disconnected = 0
        self.wait = 0.0
        """How long the engine takes to exit once its input ends."""
        self.process = StubProcess()
        self.resume_folder = tmp_path / f"resume-{name}"
        self.resume_folder.mkdir()
        self._transport: Any = SimpleNamespace(_process=self.process)
        self._materialized: Any = SimpleNamespace(config_dir=self.resume_folder)
        self._query = SimpleNamespace(_inflight_tasks=set[str]())

    async def receive_messages(self) -> AsyncIterator[Any]:
        for message in self.messages:
            yield message

    async def disconnect(self) -> None:  # as the SDK does it
        self.disconnected += 1
        await asyncio.sleep(self.wait)
        self.process.returncode = 0
        shutil.rmtree(self.resume_folder)
        self._transport = self._materialized = None


def stub_engine(
    runner: ClaudeAgentSdkRunner, tmp_path: Path, name: str, client: StubClient
) -> Any:
    """A warm engine of session "s", paused at call "t1" at checkpoint "c"."""
    hook_dir = tmp_path / f"hook-{name}"
    hook_dir.mkdir()
    return _runner._WarmEngine(  # type: ignore[reportPrivateUsage]
        client=client,
        feed=_runner._Feed([]),  # type: ignore[reportPrivateUsage]
        session_id="s",
        checkpoint="c",
        paused="t1",
        shape=runner._shape(SegmentInput("s", None, TOOLS, checkpoint="c")),  # type: ignore[reportPrivateUsage]
        hook_dir=str(hook_dir),
        lock=None,
        store=None,
        cost=0.0,
        version="2.1.288 (Claude Code)",
        ran_inside=[],
        violations=[],
        buffer=1000,
    )


async def test_only_the_same_agent_and_the_paused_calls_result_take_a_warm_engine(
    tmp_path: Path,
) -> None:
    """Another prompt, another result, other tools, a fork, a result too big for the
    engine's buffer, or a retry: a new engine. The warm one ends."""
    runner = ClaudeAgentSdkRunner(cwd=str(tmp_path), env=KEY, warm_engines=8)
    base = SegmentInput(session_id="s", prompt=None, tools=TOOLS, checkpoint="c")
    result = {"t1": ToolOutcome("ok")}
    clients: dict[str, StubClient] = {}

    def park(name: str) -> None:
        clients[name] = StubClient(tmp_path, name)
        runner._park(stub_engine(runner, tmp_path, name, clients[name]))  # type: ignore[reportPrivateUsage]

    cases: dict[str, tuple[SegmentInput, dict[str, ToolOutcome], int]] = {
        "prompt": (SegmentInput("s", "go on", TOOLS, checkpoint="c"), result, 1),
        "result": (base, {"t2": ToolOutcome("ok")}, 1),
        "tools": (SegmentInput("s", None, TOOLS[:1], checkpoint="c"), result, 1),
        "fork": (SegmentInput("s", None, TOOLS, checkpoint="c", fork=True), result, 1),
        "big": (base, {"t1": ToolOutcome("x" * 200)}, 1),
        "retry": (base, result, 2),
    }
    for name, (inp, injected, attempt) in cases.items():
        park(name)
        assert runner._take_warm(inp, injected, attempt) is None, name  # type: ignore[reportPrivateUsage]
    park("same")
    taken = runner._take_warm(base, result, 1)  # type: ignore[reportPrivateUsage]
    assert taken is not None and taken.client is clients["same"]
    assert runner._take_warm(base, result, 1) is None  # type: ignore[reportPrivateUsage]
    await all_ended()
    assert sorted(n for n, c in clients.items() if c.disconnected) == sorted(cases)
    for name in cases:
        assert not (tmp_path / f"hook-{name}").exists()


def test_hooks_callbacks_and_in_process_servers_keep_engines_cold(
    tmp_path: Path,
) -> None:
    """The SDK runs them (and a stderr callback) in tasks of the segment that started
    the engine: in a later segment they would see that segment's Activity."""
    from claude_agent_sdk import HookMatcher, create_sdk_mcp_server

    async def hook(event: Any, tool_use_id: Any, context: Any) -> Any:
        del event, tool_use_id, context
        return {}

    async def allow(name: Any, args: Any, context: Any) -> Any:
        del name, args, context

    inp = SegmentInput("s", None, TOOLS, checkpoint="c")
    plain = ClaudeAgentSdkRunner(cwd=str(tmp_path), env=KEY, warm_engines=2)
    assert plain._may_park(inp)  # type: ignore[reportPrivateUsage]
    for extra in (
        {"hooks": {"PreToolUse": [HookMatcher(matcher=None, hooks=[hook])]}},
        {"can_use_tool": allow},
        {"stderr": print},
        {"mcp_servers": {"notes": create_sdk_mcp_server("notes", tools=[])}},
    ):
        runner = ClaudeAgentSdkRunner(
            cwd=str(tmp_path), env=KEY, warm_engines=2, extra_options=extra
        )
        assert not runner._may_park(inp), extra  # type: ignore[reportPrivateUsage]


@pytest.mark.parametrize("when", ["before-it-ran", "while-it-waits"])
async def test_ending_a_warm_engine_cleans_up_also_when_cut_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    """The task that ends a warm engine can be cancelled (an event loop that closes,
    a second cancel): the engine still gets SIGTERM, and its folders and the Worker
    lock go, the lock exactly once."""
    released: list[int | None] = []
    monkeypatch.setattr(_runner, "_release_worker_lock", released.append)
    runner = ClaudeAgentSdkRunner(cwd=str(tmp_path), env=KEY, warm_engines=2)
    client = StubClient(tmp_path, "slow")
    client.wait = 30
    engine = stub_engine(runner, tmp_path, "slow", client)
    engine.lock = 7
    ClaudeAgentSdkRunner._end_warm(engine)  # type: ignore[reportPrivateUsage]
    (task,) = _runner._ending  # type: ignore[reportPrivateUsage]
    if when == "while-it-waits":
        await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)  # the task's done callbacks
    assert client.disconnected == (1 if when == "while-it-waits" else 0)
    assert client.process.terminated >= 1
    assert not client.resume_folder.exists()
    assert not Path(engine.hook_dir).exists()
    assert released == [7]
    assert not _runner._ending  # type: ignore[reportPrivateUsage]


@pytest.mark.parametrize("failure", ["store-down", "no-result"])
async def test_a_taken_warm_engine_never_leaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Once taken from the pool, a warm engine ends on every way out: the session
    store cannot be read, or the engine ends before its turn's result (then the step
    is retried, in a new engine)."""

    async def version(self: Any) -> str:
        del self
        return "2.1.288 (Claude Code)"

    async def nothing(self: Any) -> None:
        del self

    async def check(self: Any, inp: Any, warm: Any) -> bool:
        del self, inp, warm
        if failure == "store-down":
            raise OSError("the store is down")
        return True

    monkeypatch.setattr(ClaudeAgentSdkRunner, "_engine_version", version)
    monkeypatch.setattr(ClaudeAgentSdkRunner, "_prepare_engine", nothing)
    monkeypatch.setattr(ClaudeAgentSdkRunner, "_ends_at_checkpoint", check)
    runner = ClaudeAgentSdkRunner(
        session_store=FileSessionStore(tmp_path / "store"),
        cwd=str(tmp_path),
        env=KEY,
        warm_engines=2,
    )
    client = StubClient(tmp_path, "taken")
    engine = stub_engine(runner, tmp_path, "taken", client)
    runner._park(engine)  # type: ignore[reportPrivateUsage]
    step = SegmentInput(
        "s", None, TOOLS, checkpoint="c", injected={"t1": ToolOutcome("ok")}
    )
    match = "store is down" if failure == "store-down" else "ended before"
    with pytest.raises((OSError, RuntimeError), match=match):
        await runner.run(step, 1)
    await all_ended()
    assert client.disconnected == 1 and client.process.returncode == 0
    assert not Path(engine.hook_dir).exists()
    assert runner._warm == {}  # type: ignore[reportPrivateUsage]


def test_the_hook_reads_calls_answered_after_the_engine_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warm engine's environment is fixed when it starts; a call answered since then
    must never run when the engine announces it again."""
    for name in ("TCA_ALLOW_ID", "TCA_ANSWERED_IDS", "TCA_TOOL_ACTIVITIES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TCA_HOOK_DIR", str(tmp_path))
    call = {"tool_name": "Bash", "tool_use_id": "t9", "tool_input": {}}
    assert _defer_hook.decide(call).get("permissionDecision") is None  # it would run
    (tmp_path / _defer_hook.ANSWERED).write_text("t1\nt9\n", encoding="utf-8")
    assert _defer_hook.decide(call)["permissionDecision"] == "defer"


def test_warm_options_are_checked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="warm_engines"):
        ClaudeAgentSdkRunner(cwd=str(tmp_path), env=KEY, warm_engines=-1)
    with pytest.raises(ValueError, match="warm_engines"):
        ClaudeAgentSdkRunner(cwd=str(tmp_path), env=KEY, warm_seconds=0)
    json.dumps(ClaudeAgentSdkRunner._shape(SegmentInput("s", None, TOOLS)))  # type: ignore[reportPrivateUsage]


async def test_a_warm_engine_does_not_continue_a_session_that_went_on_elsewhere(
    tmp_path: Path,
) -> None:
    """Store mode, two Workers: A paused the session and keeps its engine warm; B ran
    the next segment from that checkpoint. When that segment runs again on A (a
    reset: attempt 1 again), A's engine must not continue, as it does not know B's
    turn: a new engine continues a copy that ends at the checkpoint, and the shared
    session gets no second branch."""
    api = start_with_policy(refund_policy)
    try:
        a = counting(api, tmp_path, "store", config="cfg-a", warm_engines=2)
        b = counting(api, tmp_path, "store", config="cfg-b", warm_engines=2)
        first = await a.run(
            SegmentInput(session_id=str(uuid.uuid4()), prompt=PROMPT, tools=TOOLS), 1
        )
        assert first.deferred is not None and first.checkpoint is not None
        assert (first.session_id, first.checkpoint) in a._warm  # type: ignore[reportPrivateUsage]
        second = SegmentInput(
            session_id=first.session_id,
            prompt=None,
            tools=TOOLS,
            checkpoint=first.checkpoint,
            injected={
                first.deferred.id: _fake_tool(first.deferred.name, first.deferred.input)
            },
            segment_index=1,
        )
        on_b = await b.run(second, 1)
        retried_on_a = await a.run(second, 1)
        a._end_all_warm()  # type: ignore[reportPrivateUsage]
        b._end_all_warm()  # type: ignore[reportPrivateUsage]
        await all_ended()
        store = FileSessionStore(tmp_path / "store")
        key = SessionKey(
            project_key=project_key_for_directory(str(tmp_path)),
            session_id=first.session_id,
        )
        entries = await store.load(key) or []
    finally:
        api.stop()
    for out in (on_b, retried_on_a):
        assert out.deferred is not None and out.deferred.name == "issue_refund"
    assert retried_on_a.session_id != first.session_id  # the copy
    assert a.warm == 0 and b.warm == 0
    parents = Counter(
        e.get("parentUuid")
        for e in entries
        if _runner._is_transcript(e) and e.get("parentUuid")  # type: ignore[reportPrivateUsage]
    )
    assert max(parents.values()) == 1, parents  # one line of turns, no branch
    assert api.errors == []


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_workflow_on_one_worker_runs_in_one_warm_engine(
    client: Client, tmp_path: Path, mode: str
) -> None:
    """A real Workflow and Worker: one engine takes every step after the first (held:
    after the second)."""
    api = echo_rounds(5)
    queue = f"warm-{uuid.uuid4().hex[:8]}"
    runner = counting(api, tmp_path, mode, warm_engines=4)
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[EchoRoundsWorkflow],
            activities=[echo],
            plugins=[ClaudeAgentPlugin(runner)],
            **FAIL_FAST,
        ):
            answer = await client.execute_workflow(
                EchoRoundsWorkflow.run, "go", id=queue, task_queue=queue
            )
    finally:
        api.stop()
    assert answer == "DONE 5"
    assert RAN[queue] == [0, 1, 2, 3, 4]  # each call ran once
    assert (runner.cold, runner.warm) == ((2, 4) if mode == "held" else (1, 5))
    if mode == "held":  # nothing of the conversation stayed on the Worker's disk
        assert not list((tmp_path / "cfg").glob("projects/*/*"))
    assert runner._warm == {} and not _runner._ending  # type: ignore[reportPrivateUsage]
    assert api.errors == [] and runner.stub_calls == 0


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_two_workers_on_one_queue(
    client: Client, tmp_path: Path, mode: str
) -> None:
    """Two Workers (two machines, each with its own connection) poll one queue.
    Whichever runs each segment, the task finishes, each call runs once, and Claude
    always sees a valid conversation."""
    rounds = 8
    api = echo_rounds(rounds)
    queue = f"warm2-{uuid.uuid4().hex[:8]}"
    runners = [
        counting(api, tmp_path, mode, config=f"cfg{n}", warm_engines=4) for n in (1, 2)
    ]
    other = await Client.connect(
        client.service_client.config.target_host, namespace=client.namespace
    )

    def worker(connection: Client, runner: Counting) -> Worker:
        return Worker(
            connection,
            task_queue=queue,
            workflows=[EchoRoundsWorkflow],
            activities=[echo],
            plugins=[ClaudeAgentPlugin(runner)],
            **FAIL_FAST,
        )

    try:
        async with worker(client, runners[0]), worker(other, runners[1]):
            answer = await client.execute_workflow(
                EchoRoundsWorkflow.run, "go", id=queue, task_queue=queue
            )
    finally:
        api.stop()
    warm = sum(r.warm for r in runners)
    print(f"{mode}: {warm} of {rounds} segments after a tool continued warm")
    assert answer == f"DONE {rounds}"
    assert RAN[queue] == list(range(rounds))
    assert sum(r.cold for r in runners) + warm >= rounds + 1
    assert all(r._warm == {} for r in runners)  # type: ignore[reportPrivateUsage]
    assert api.errors == []


async def test_a_worker_that_stops_ends_its_warm_engines(
    client: Client, shop_dir: Path, tmp_path: Path
) -> None:
    """The engine waits warm while the manager decides; the Worker stops (a deploy):
    the engine ends with it, not ``warm_seconds`` later."""
    del shop_dir
    api = start_with_policy(refund_policy)
    queue = f"warmstop-{uuid.uuid4().hex[:8]}"
    runner = counting(api, tmp_path, "held", warm_engines=4)
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[RefundAgentWorkflow],
            activities=SHOP_TOOLS,
            plugins=[ClaudeAgentPlugin(runner)],
            **FAIL_FAST,
        ):
            handle = await client.start_workflow(
                RefundAgentWorkflow.run, PROMPT, id=queue, task_queue=queue
            )
            pending = await wait_for_approval(handle, timeout=120)
            assert pending is not None and pending["name"] == "issue_refund"
            processes = [
                w.client._transport._process
                for w in runner._warm.values()  # type: ignore[reportPrivateUsage]
            ]
            assert len(processes) == 1 and processes[0].returncode is None
        await all_ended()
        assert runner._warm == {}  # type: ignore[reportPrivateUsage]
        assert processes[0].returncode is not None
        await handle.terminate()
    finally:
        api.stop()


@pytest.mark.parametrize("mode", ["held", "store"])
async def test_a_killed_worker_takes_its_warm_engine_along(
    client: Client, address: str, shop_dir: Path, tmp_path: Path, mode: str
) -> None:
    """The Worker process dies while its engine waits warm for the manager's approval.
    The engine ends with it; a new Worker on another machine resumes from the
    checkpoint, and every tool runs once."""
    api = start_with_policy(refund_policy)
    queue = f"warmkill-{uuid.uuid4().hex[:8]}"
    shared = {**shared_settings(tmp_path, shop_dir, mode), "WARM_ENGINES": "4"}
    (tmp_path / "work").mkdir()

    def machine(n: int) -> dict[str, str]:
        return {**shared, **engine_env(api, str(tmp_path / f"claude-config-{n}"))}

    workers = [
        await start_worker(address, queue, machine(1), tmp_path / "w1.log", real=True)
    ]
    try:
        handle = await client.start_workflow(
            RefundAgentWorkflow.run, PROMPT, id=queue, task_queue=queue
        )
        pending = await wait_for_approval(handle, timeout=120)
        assert pending is not None and pending["name"] == "issue_refund"
        waiting = engines_of(workers[0].pid)
        kill(workers[0])
        if waiting is not None:
            assert len(waiting) == 1, waiting  # the engine that paused waits warm
            await wait_until(lambda: not still_running(waiting), timeout=15)
        workers.append(
            await start_worker(
                address, queue, machine(2), tmp_path / "w2.log", real=True
            )
        )
        await handle.execute_update(
            RefundAgentWorkflow.review, args=[pending["id"], True, MANAGER]
        )
        result = await asyncio.wait_for(handle.result(), 180)
    finally:
        for worker in workers:
            if worker.poll() is None:
                kill(worker)
        api.stop()
    kinds: Counter[str] = Counter()
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            kinds[
                event.activity_task_scheduled_event_attributes.activity_type.name
            ] += 1
    assert result.startswith("Done. Refunded 49.99 EUR for order A-1001")
    assert dict(kinds) == {
        "run_claude_segment": 4,
        "look_up_order": 1,
        "issue_refund": 1,
        "email_customer": 1,
    }
    for tool in ("look_up_order", "issue_refund", "email_customer"):
        assert len(shop.executions(tool)) == 1, tool
    assert api.errors == []


@pytest.mark.parametrize("at", ["start", "warm"])
@pytest.mark.parametrize(
    ("status", "message", "final"),
    [
        (413, "Request too large.", True),
        (400, "Your credit balance is too low to access the Anthropic API.", False),
    ],
    ids=["refused-for-good", "retried"],
)
async def test_api_errors_end_or_retry_a_step_as_without_warm_engines(
    tmp_path: Path, at: str, status: int, message: str, final: bool
) -> None:
    """An engine that stays open does not exit after an error result, so the SDK does
    not raise it: the runner ends the engine's input and reads on, as the one-shot
    query does. The step then fails for good, or Temporal retries it in a new engine."""
    api = start_with_policy(refund_policy)
    runner = counting(api, tmp_path, "store", warm_engines=2)
    first = SegmentInput(session_id=str(uuid.uuid4()), prompt=PROMPT, tools=TOOLS)
    try:
        step = first
        if at == "warm":
            out = await runner.run(first, 1)
            assert out.deferred is not None and out.checkpoint is not None
            step = SegmentInput(
                session_id=out.session_id,
                prompt=None,
                tools=TOOLS,
                checkpoint=out.checkpoint,
                injected={
                    out.deferred.id: _fake_tool(
                        out.deferred.name, {"order_id": "A-1001"}
                    )
                },
                segment_index=1,
            )
        api.fail_status, api.fail_message = status, message
        if final:
            failed = await runner.run(step, 1)
            assert failed.is_error and failed.error is not None
        else:
            with pytest.raises(ResultError):  # Temporal retries the segment
                await runner.run(step, 1)
            api.fail_status = None
            again = await runner.run(step, 2)
            assert again.deferred is not None and not again.is_error
        runner._end_all_warm()  # type: ignore[reportPrivateUsage]
        await all_ended()
    finally:
        api.stop()
    assert runner.warm == (1 if at == "warm" else 0)  # the retry: a new engine
    assert api.errors == []


class Scripted:
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


def result(call: str | None = "t1", *, error: bool = False) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=error,
        num_turns=1,
        session_id="s",
        deferred_tool_use=(
            DeferredToolUse(id=call, name="mcp__durable__echo", input={})
            if call
            else None
        ),
    )


INIT = SystemMessage(subtype="init", data={})
TASK = SystemMessage(subtype="task_started", data={"task_id": "a1"})
DONE = SystemMessage(subtype="task_notification", data={"task_id": "a1"})
NONE: set[str] = set()


@pytest.mark.parametrize(
    ("resumed", "script", "read", "ended_after"),
    [
        (False, [(INIT, NONE), (result(), NONE)], 2, None),
        (True, [(result("t0"), NONE), (INIT, NONE), (result(), NONE)], 3, None),
        (True, [(INIT, NONE), (result("t0"), NONE), (result(), NONE)], 3, 1),
        (False, [(INIT, NONE), (result(error=True), NONE)], 2, 1),
        (False, [(INIT, NONE), (result(None), NONE)], 2, 1),
        (
            False,
            [(INIT, NONE), (TASK, {"a1"}), (result(), {"a1"}), (DONE, NONE)]
            + [(result(None), NONE)],
            5,
            4,
        ),
    ],
    ids=[
        "paused",
        "resumed-repeats-its-pause",
        "repeat-after-init",
        "error",
        "answer",
        "task-started",
    ],
)
async def test_an_engine_stays_open_only_when_it_paused_cleanly(
    resumed: bool,
    script: list[tuple[Any, set[str]]],
    read: int,
    ended_after: int | None,
) -> None:
    """A clean pause leaves the engine running (it can stay warm). An error, an
    answer, a task it started, or a result at a call this run answered (a resumed
    engine repeats it) finish like the one-shot query: the input ends once no agent
    runs, and the messages until the engine exits are read too."""
    feed = _runner._Feed([])  # type: ignore[reportPrivateUsage]
    client = Scripted(script, feed)
    seen = [
        m
        async for m in _runner._turn_messages(  # type: ignore[reportPrivateUsage]
            client, feed, resumed, stay=True, answered={"t0"}
        )
    ]
    assert len(seen) == read
    assert client.ended_after == ended_after
