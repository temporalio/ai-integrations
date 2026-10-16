"""Workflow-side checks, and the session lease (two attempts never run a CLI on one session id),
none of which need a Temporal server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

import pytest
from pydantic import BaseModel, ValidationError

from temporalio import workflow
from temporalio.claude_agent_sdk._lease import (
    FileLeaseStore,
    Lease,
    LeaseRecord,
    SessionLeased,
    SessionLeaseLost,
    fenced_session_store,
    session_lease,
)
from temporalio.claude_agent_sdk._models import (
    CLAUDE_STRUCTURED_OUTPUT_ERROR_TYPE,
    CLAUDE_TURN_CONFIG_ERROR_TYPE,
    ClaudeDeferredTool,
    ClaudeTurnInput,
    ClaudeTurnResult,
)
from temporalio.claude_agent_sdk._session import (
    ClaudeSession,
    ClaudeSessionState,
    validate_structured_output,
)
from temporalio.claude_agent_sdk.workflow import run_turn
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.workflow import ActivityCancellationType, ActivityConfig


class Forecast(BaseModel):
    city: str
    temp_f: int


def test_structured_output_is_validated_into_the_model():
    result = ClaudeTurnResult(
        text="ignored", structured_output={"city": "Paris", "temp_f": 72}
    )

    assert validate_structured_output(Forecast, result) == Forecast(
        city="Paris", temp_f=72
    )


@pytest.mark.parametrize(
    "result",
    [
        ClaudeTurnResult(
            text='{"city": "Paris", "temp_f": 72}', structured_output=None
        ),
        ClaudeTurnResult(structured_output={"city": "Paris", "temp_f": "warm"}),
    ],
    ids=["missing-even-if-text-is-json", "wrong-shape"],
)
def test_missing_or_invalid_structured_output_fails_the_message(
    result: ClaudeTurnResult,
):
    with pytest.raises(ApplicationError) as info:
        validate_structured_output(Forecast, result)

    assert info.value.type == CLAUDE_STRUCTURED_OUTPUT_ERROR_TYPE
    assert info.value.non_retryable


def test_workflow_facing_modules_do_not_import_the_sdk():
    code = (
        "import sys\n"
        "import temporalio.claude_agent_sdk\n"
        "import temporalio.claude_agent_sdk._session\n"
        "assert 'claude_agent_sdk' not in sys.modules, sorted(\n"
        "    m for m in sys.modules if m.startswith('claude_agent_sdk'))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_package_imports_where_fcntl_does_not_exist():
    """Windows has no ``fcntl``; the package must import there (native tools fail closed)."""
    code = (
        "import sys\n"
        "sys.modules['fcntl'] = None  # what a missing module looks like to an import\n"
        "import temporalio.claude_agent_sdk\n"
        "import temporalio.claude_agent_sdk._native_hook\n"
        "import temporalio.claude_agent_sdk._session\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_activity_input_carries_only_plain_json():
    dumped = ClaudeTurnInput(prompt="hi").model_dump(mode="json")

    assert ClaudeTurnInput.model_validate(dumped) == ClaudeTurnInput(prompt="hi")
    assert set(dumped) == {
        "prompt",
        "options",
        "session_id",
        "resume",
        "prompt_uuid",
        "tools",
        "bridge_token",
        "native_tools",
        "native_decision",
        "native_denied",
        "native_rerequests",
        "resumes_lost_attempt",
        "attempt",
        "resumes_api_error",
        "native_calls",
        "workspace_revision",
        "report_progress",
    }


def test_a_session_snapshot_is_a_deep_copy_of_its_whole_state():
    """Every field lives in one model, so a snapshot cannot leave one out, and it is a copy: a
    later turn mutating the session never rewrites a snapshot already handed to a new run."""
    session = ClaudeSession("s", started=True)
    session.state.native_denied.append(ClaudeDeferredTool(tool_use_id="t", name="Bash"))
    session.state.cost_usd = 0.5

    snapshot = session.snapshot()
    session.state.native_denied.append(ClaudeDeferredTool(tool_use_id="u", name="Bash"))
    session.state.degraded = "lost a batch"

    assert snapshot.model_dump() == {
        **ClaudeSessionState().model_dump(),
        "session_id": "s",
        "started": True,
        "native_denied": [{"tool_use_id": "t", "name": "Bash", "input": {}}],
        "cost_usd": 0.5,
    }
    restored = ClaudeSession.restore(snapshot)
    assert restored.state == snapshot and restored.state is not snapshot


def test_a_session_state_from_a_newer_version_is_refused():
    with pytest.raises(ValidationError):
        ClaudeSessionState.model_validate({"version": 3, "session_id": "s"})


def _activity_error(cause: BaseException) -> ActivityError:
    error = ActivityError(
        "Activity task failed",
        scheduled_event_id=1,
        started_event_id=2,
        identity="worker",
        activity_type="claude_agent.run_turn",
        activity_id="1",
        retry_state=None,
    )
    error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    "config",
    [
        {"retry_policy": RetryPolicy(maximum_attempts=5)},
        {"cancellation_type": ActivityCancellationType.TRY_CANCEL},
        {
            "retry_policy": RetryPolicy(maximum_attempts=5),
            "cancellation_type": ActivityCancellationType.ABANDON,
        },
    ],
    ids=["retry_policy", "cancellation_type", "both"],
)
async def test_run_turn_rejects_fixed_activity_options(
    monkeypatch: pytest.MonkeyPatch, config: ActivityConfig
) -> None:
    """``retry_policy`` and ``cancellation_type`` stay fixed. Setting either fails the message
    with ``ClaudeTurnConfigError`` and does not schedule the activity.
    """

    async def execute_activity(*_args: object, **_kwargs: object) -> ClaudeTurnResult:
        raise AssertionError("the activity was scheduled")

    monkeypatch.setattr(workflow, "execute_activity", execute_activity)
    with pytest.raises(ApplicationError) as info:
        await run_turn(ClaudeTurnInput(prompt="hi"), activity_config=config)

    assert info.value.type == CLAUDE_TURN_CONFIG_ERROR_TYPE
    assert info.value.non_retryable
    for key in config:
        assert key in info.value.message


async def test_run_turn_keeps_one_attempt_and_waits_out_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller timeouts and task queue merge over the defaults. The fixed options do not."""
    seen: dict[str, object] = {}

    async def execute_activity(*_args: object, **kwargs: object) -> ClaudeTurnResult:
        seen.update(kwargs)
        return ClaudeTurnResult(text="ok")

    monkeypatch.setattr(workflow, "execute_activity", execute_activity)
    result = await run_turn(
        ClaudeTurnInput(prompt="hi"),
        activity_config={
            "task_queue": "claude",
            "start_to_close_timeout": timedelta(minutes=1),
        },
    )

    assert result.text == "ok"
    retry = seen["retry_policy"]
    assert isinstance(retry, RetryPolicy) and retry.maximum_attempts == 1
    assert seen["cancellation_type"] == (
        ActivityCancellationType.WAIT_CANCELLATION_COMPLETED
    )
    assert seen["task_queue"] == "claude"
    assert seen["start_to_close_timeout"] == timedelta(minutes=1)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
async def test_run_turn_surfaces_the_activity_application_error(
    monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    """An activity failure's own text is only "Activity task failed". The Claude error under it,
    including one wrapped in another activity failure and itself caused by a runtime error, is
    what the workflow sees.
    """
    claude = ApplicationError(
        "Claude run failed (api_error): overloaded",
        {"api_error_status": 529},
        type="ClaudeResultError",
        non_retryable=True,
    )
    claude.__cause__ = RuntimeError("socket closed")
    failure: BaseException = _activity_error(claude)
    if nested:
        failure = _activity_error(failure)

    async def execute_activity(*_args: object, **_kwargs: object) -> ClaudeTurnResult:
        raise failure

    monkeypatch.setattr(workflow, "execute_activity", execute_activity)
    with pytest.raises(ApplicationError) as info:
        await run_turn(ClaudeTurnInput(prompt="hi"))

    assert info.value.type == "ClaudeResultError"
    assert info.value.message == "Claude run failed (api_error): overloaded"
    assert info.value.non_retryable
    assert info.value.details == ({"api_error_status": 529},)


class _Waiter(Protocol):
    def wait(self) -> object: ...


class _Results(Protocol):
    def put(self, item: tuple[int, str]) -> None: ...


SESSION = "11111111-2222-3333-4444-555555555555"
LONG = timedelta(seconds=30)
# Filesystem leases need room for Windows I/O and event-loop scheduling under CI load.
SHORT = timedelta(seconds=2)


@pytest.mark.parametrize("operation", ["beat", "latest"])
async def test_stalled_renewal_still_enforces_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A hung storage call must not leave the lease holder alive past its deadline."""
    store = FileLeaseStore(tmp_path)
    entered = asyncio.Event()
    lost = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()
    started = loop.time()
    stopped_after: list[float] = []
    original = getattr(store, operation)

    def on_lost() -> None:
        stopped_after.append(loop.time() - started)
        lost.set()

    def blocked(*args: Any, **kwargs: Any) -> Any:
        loop.call_soon_threadsafe(entered.set)
        try:
            if not release.wait(5):
                raise AssertionError("test did not release stalled storage call")
            return original(*args, **kwargs)
        finally:
            finished.set()

    async def holder() -> None:
        nonlocal started
        async with session_lease(store, SESSION, SHORT, on_lost=on_lost):
            started = loop.time()
            monkeypatch.setattr(store, operation, blocked)
            await asyncio.Event().wait()

    task = asyncio.create_task(holder())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        try:
            await asyncio.wait_for(lost.wait(), SHORT.total_seconds() * 2)
            stopped = True
        except TimeoutError:
            stopped = False
        assert stopped, f"{operation} stalled: holder still alive after two lease TTLs"
        assert stopped_after[0] < SHORT.total_seconds(), stopped_after
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def store(tmp_path: Path) -> FileLeaseStore:
    return FileLeaseStore(tmp_path)


def _held(lease: Lease | None) -> Lease:
    assert lease is not None
    return lease


def _latest(store: FileLeaseStore) -> LeaseRecord:
    record = store.latest(SESSION)
    assert record is not None
    return record


@pytest.mark.parametrize(
    "failure", ["permission", "invalid-json", "non-object", "invalid-fields"]
)
def test_an_unreadable_beat_is_not_a_missing_beat(
    store: FileLeaseStore, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    assert store.claim(SESSION, 1, "holder")
    store.beat(SESSION, 1, "holder", 7)
    read_text = Path.read_text

    def unreadable(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.suffix == ".beat":
            if failure == "permission":
                raise PermissionError("heartbeat file is temporarily inaccessible")
            if failure == "invalid-fields":
                return json.dumps(
                    {"owner": "holder", "beat": "unknown", "released": False}
                )
            return "{" if failure == "invalid-json" else "[]"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(OSError):
        store.latest(SESSION)


async def test_an_acquiring_read_error_cannot_take_over_a_live_holder(
    store: FileLeaseStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    renewed = asyncio.Event()
    ttl = timedelta(seconds=2)
    contender = FileLeaseStore(store.root)
    latest = contender.latest
    reads = 0

    def temporarily_unreadable(session_id: str) -> LeaseRecord | None:
        nonlocal reads
        reads += 1
        if reads == 1:
            raise PermissionError("heartbeat read is temporarily denied")
        return latest(session_id)

    async with session_lease(store, SESSION, ttl, on_beat=renewed.set) as holder:
        reported = (_held(holder).generation, _held(holder).beat)
        renewed.clear()
        await asyncio.wait_for(renewed.wait(), 2)
        monkeypatch.setattr(contender, "latest", temporarily_unreadable)
        with pytest.raises(SessionLeased):
            async with session_lease(
                contender, SESSION, ttl, wait=timedelta(seconds=0.2), silent=reported
            ):
                pytest.fail("an unreadable beat authorized a takeover")
        assert reads >= 2
        assert _latest(store).generation == _held(holder).generation
        assert _held(holder).lost is None


async def test_an_unreadable_lease_acquisition_stops_at_its_wait_deadline(
    store: FileLeaseStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert store.claim(SESSION, 1, "holder")

    def unreadable(_session_id: str) -> LeaseRecord | None:
        raise PermissionError("heartbeat read is persistently denied")

    monkeypatch.setattr(store, "latest", unreadable)
    started = time.monotonic()
    with pytest.raises(SessionLeased, match="could not read"):
        async with session_lease(store, SESSION, LONG, wait=timedelta(seconds=0.1)):
            pytest.fail("unreadable ownership authorized a new holder")
    assert time.monotonic() - started < 1
    assert FileLeaseStore(store.root).latest(SESSION) == LeaseRecord(
        1, "holder", -1, False
    )


async def test_transient_renewal_error_retries_before_the_deadline(
    store: FileLeaseStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    renewed = asyncio.Event()
    attempts = 0
    original = store.beat

    def transient(*args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        if not kwargs.get("released"):
            attempts += 1
            if attempts == 1:
                raise PermissionError("a reader briefly holds the beat file")
        original(*args, **kwargs)

    async with session_lease(
        store, SESSION, timedelta(seconds=1.2), on_beat=renewed.set
    ):
        renewed.clear()
        monkeypatch.setattr(store, "beat", transient)
        await asyncio.wait_for(renewed.wait(), 2)
        assert attempts == 2


async def test_a_second_attempt_waits_then_is_refused_while_the_first_renews(
    store: FileLeaseStore,
) -> None:
    async with session_lease(store, SESSION, SHORT):
        started = time.monotonic()
        with pytest.raises(SessionLeased, match=SESSION):
            async with session_lease(store, SESSION, SHORT):
                pytest.fail("a second CLI took a leased session")
        # It waited out a whole TTL of renewals rather than failing the message at once.
        assert time.monotonic() - started >= SHORT.total_seconds()


async def test_a_released_lease_is_taken_at_once(store: FileLeaseStore) -> None:
    async with session_lease(store, SESSION, LONG) as first:
        pass
    started = time.monotonic()
    async with session_lease(store, SESSION, LONG, wait=timedelta(0)) as second:
        assert _held(second).generation == _held(first).generation + 1
    assert time.monotonic() - started < 1


async def test_a_waiting_attempt_takes_the_lease_when_the_holder_releases(
    store: FileLeaseStore,
) -> None:
    async def hold() -> None:
        async with session_lease(store, SESSION, LONG):
            await asyncio.sleep(0.2)

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    async with session_lease(store, SESSION, LONG, wait=timedelta(seconds=5)) as lease:
        assert _held(lease).generation == 2
    await holder


async def test_a_dead_holder_goes_stale_after_one_ttl_and_the_takeover_is_logged(
    store: FileLeaseStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A worker that died never renews; nobody's wall clock decides that, only silence."""
    assert store.claim(SESSION, 1, "a-worker-that-died")
    started = time.monotonic()
    with caplog.at_level("WARNING"):
        async with session_lease(store, SESSION, SHORT) as lease:
            assert _held(lease).generation == 2
    assert time.monotonic() - started >= SHORT.total_seconds()
    assert any("did not release it" in r.message for r in caplog.records), caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        async with session_lease(store, SESSION, LONG):
            pass
    assert not caplog.records, f"a clean handover logged: {caplog.text}"


async def test_a_live_holder_keeps_its_lease_past_the_ttl(
    store: FileLeaseStore,
) -> None:
    async with session_lease(store, SESSION, SHORT) as lease:
        await asyncio.sleep(SHORT.total_seconds() * 2)
        assert _latest(store).beat > 0, "the lease was not renewed"
        with pytest.raises(SessionLeased):
            async with session_lease(store, SESSION, SHORT):
                pytest.fail("a live holder's lease was stolen after its TTL")
        assert _held(lease).lost is None


async def test_different_sessions_do_not_block_each_other(
    store: FileLeaseStore,
) -> None:
    other = "99999999-8888-7777-6666-555555555555"
    async with session_lease(store, SESSION, LONG):
        async with session_lease(store, other, LONG, wait=timedelta(0)):
            pass


async def test_a_turn_without_a_session_needs_no_lease(
    store: FileLeaseStore, tmp_path: Path
) -> None:
    async with session_lease(store, None, LONG) as lease:
        assert lease is None
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("owner", ["holder", "contender"])
def test_an_unreadable_existing_claim_is_not_reported_as_owned(
    store: FileLeaseStore, monkeypatch: pytest.MonkeyPatch, owner: str
) -> None:
    assert store.claim(SESSION, 1, "holder")
    target = store.root / SESSION / "000000000001"
    read_text = Path.read_text

    def unreadable(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == target:
            raise PermissionError("existing claim is temporarily unreadable")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    assert not store.claim(SESSION, 1, owner)
    assert json.loads(read_text(target))["owner"] == "holder"


def _claim_generations(
    root: str,
    owner: str,
    generations: int,
    start: _Waiter,
    wins: _Results,
) -> None:
    store = FileLeaseStore(root)
    start.wait()
    try:
        for generation in range(1, generations + 1):
            if store.claim(SESSION, generation, owner):
                wins.put((generation, owner))
    except Exception:
        # xdist does not forward a spawned child's traceback to this test's report.
        (Path(root) / f"{owner}.error").write_text(traceback.format_exc())
        raise


def test_racing_claims_from_processes_yield_exactly_one_holder_per_generation(
    tmp_path: Path,
) -> None:
    """The takeover race: every contender saw the same stale generation and claims the next.
    Delete-then-create let a second contender delete the first one's fresh claim; an exclusive
    link of a fully written record lets exactly one win."""
    ctx = multiprocessing.get_context("spawn")
    start, wins = ctx.Event(), ctx.Queue()
    generations = 200
    procs = [
        ctx.Process(
            target=_claim_generations,
            args=(str(tmp_path), f"p{n}", generations, start, wins),
        )
        for n in range(6)
    ]
    for p in procs:
        p.start()
    start.set()
    try:
        for p in procs:
            p.join(timeout=60)
        errors = "\n".join(path.read_text() for path in tmp_path.glob("*.error"))
        assert all(p.exitcode == 0 for p in procs), errors
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
    won: dict[int, list[str]] = {}
    while not wins.empty():
        generation, owner = wins.get()
        won.setdefault(generation, []).append(owner)
    # A contender that falls behind may find a generation already passed and win none.
    assert won.get(generations) and len(won[generations]) == 1
    assert all(len(owners) == 1 for owners in won.values()), {
        g: o for g, o in won.items() if len(o) > 1
    }


def test_a_generation_already_passed_cannot_be_claimed_after_it_was_pruned(
    store: FileLeaseStore,
) -> None:
    """A slow contender that read generation 1 tries to claim 2 after 2 and 3 were claimed and
    3 pruned 2; re-creating 2 must not make it a holder below the real one."""
    assert store.claim(SESSION, 1, "a")
    assert store.claim(SESSION, 2, "b")
    assert store.claim(SESSION, 3, "c")
    assert not store.claim(SESSION, 2, "slow")
    assert _latest(store).owner == "c"


async def test_contenders_over_a_stale_lease_never_hold_it_together(
    store: FileLeaseStore,
) -> None:
    """The full protocol under contention: one holder at a time, each in its own generation."""
    assert store.claim(SESSION, 1, "a-worker-that-died")
    holding: list[int] = []
    generations: list[int] = []

    async def take() -> None:
        with contextlib.suppress(SessionLeased):
            async with session_lease(
                store, SESSION, SHORT, wait=timedelta(seconds=3)
            ) as lease:
                holding.append(_held(lease).generation)
                assert len(holding) == 1, f"held together: {holding}"
                generations.append(_held(lease).generation)
                await asyncio.sleep(0.02)
                holding.remove(_held(lease).generation)

    await asyncio.gather(*(take() for _ in range(8)))
    assert generations == sorted(set(generations)), generations


async def test_a_taken_over_holder_kills_its_cli_and_never_overwrites_the_new_holder(
    store: FileLeaseStore,
) -> None:
    killed: list[str] = []
    lease: Lease | None = None
    with pytest.raises(SessionLeaseLost, match="taken over"):
        async with session_lease(
            # Leave room for Windows to commit the takeover before the renewal deadline.
            store,
            SESSION,
            timedelta(seconds=2),
            on_lost=lambda: killed.append("cli"),
        ) as lease:
            # Another worker judged us dead and claimed the next generation.
            assert store.claim(SESSION, _held(lease).generation + 1, "the-new-holder")
            store.beat(SESSION, _held(lease).generation + 1, "the-new-holder", 7)
            await asyncio.wait_for(asyncio.Event().wait(), 5)
            pytest.fail("the block kept running after its lease was lost")
    assert killed == ["cli"]
    latest = _latest(store)
    assert (latest.generation, latest.owner, latest.beat, latest.released) == (
        _held(lease).generation + 1,
        "the-new-holder",
        7,
        False,
    ), "the old holder's renewal or release touched the new holder's lease"


async def test_a_holder_that_cannot_confirm_renewal_gives_up_before_it_could_go_stale(
    store: FileLeaseStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contender claims after one TTL of silence; the holder must be gone before that."""
    killed: list[float] = []
    started = time.monotonic()

    def unreachable(*_args: object, **_kwargs: Any) -> None:
        raise OSError("the lease volume went away")

    with pytest.raises(SessionLeaseLost, match="no renewal confirmed"):
        async with session_lease(
            store,
            SESSION,
            SHORT,
            on_lost=lambda: killed.append(time.monotonic() - started),
        ):
            monkeypatch.setattr(store, "beat", unreachable)
            await asyncio.sleep(5)
    assert killed and killed[0] < SHORT.total_seconds(), killed


async def test_losing_the_lease_leaves_the_task_uncancelled(
    store: FileLeaseStore,
) -> None:
    """The loss is delivered as a cancellation; the caller must see only SessionLeaseLost."""
    with pytest.raises(SessionLeaseLost):
        async with session_lease(store, SESSION, SHORT) as lease:
            store.claim(SESSION, _held(lease).generation + 1, "intruder")
            await asyncio.sleep(5)
    task = asyncio.current_task()
    assert task is not None
    assert task.cancelling() == 0
    await asyncio.sleep(0)  # and no cancellation is still pending


class _Store:
    def __init__(self) -> None:
        self.appended: list[list[dict]] = []

    async def append(self, _key: object, entries: list[Any]) -> None:
        self.appended.append(entries)

    async def load(self, _key: object) -> None:
        return None

    async def list_subkeys(self, _key: object) -> list[str]:
        return ["sub"]


async def test_a_superseded_generation_cannot_append_to_the_session_store(
    store: FileLeaseStore,
) -> None:
    inner = _Store()
    key = {"project_key": "p", "session_id": SESSION}
    async with session_lease(store, SESSION, LONG) as lease:
        fenced = fenced_session_store(inner, lease)
        await fenced.append(key, [{"type": "user", "uuid": "1"}])
        assert store.claim(SESSION, _held(lease).generation + 1, "intruder")
        with pytest.raises(SessionLeaseLost, match="superseded"):
            await fenced.append(key, [{"type": "user", "uuid": "2"}])
        _held(
            lease
        ).lost = None  # let the block end normally; the renewal has not noticed yet
    assert inner.appended == [[{"type": "user", "uuid": "1"}]]


async def test_a_lease_still_at_a_timed_out_attempts_last_beat_is_taken_at_once(
    store: FileLeaseStore,
) -> None:
    """Temporal timed the holder out a whole TTL after the heartbeat that reported its beat, so a
    record still at that beat is claimed without watching it for another TTL, and the holder it
    superseded can no longer append. A holder that renewed since is watched as before."""
    inner = _Store()
    key = {"project_key": "p", "session_id": SESSION}
    assert store.claim(SESSION, 1, "timed-out-holder")
    zombie = fenced_session_store(inner, Lease(store, SESSION, 1, "timed-out-holder"))
    started = time.monotonic()
    async with session_lease(store, SESSION, LONG, silent=(1, -1)) as lease:
        assert _held(lease).generation == 2
        assert time.monotonic() - started < SHORT.total_seconds()
        with pytest.raises(SessionLeaseLost, match="superseded"):
            await zombie.append(key, [{"type": "user", "uuid": "zombie"}])
    assert inner.appended == []

    renewed = asyncio.Event()
    ttl = timedelta(seconds=2)
    async with session_lease(store, SESSION, ttl, on_beat=renewed.set) as renewing:
        reported = (_held(renewing).generation, _held(renewing).beat)
        renewed.clear()
        await asyncio.wait_for(renewed.wait(), 2)
        assert _latest(store).beat != reported[1]
        with pytest.raises(SessionLeased):
            async with session_lease(
                store, SESSION, ttl, wait=timedelta(seconds=0.4), silent=reported
            ):
                pytest.fail(
                    "a holder that renewed after its last heartbeat was taken over"
                )
        assert _held(renewing).lost is None


class _InFlight:
    """An append that pauses after the fenced store's generation check has returned."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.batches: list[list[Any]] = []

    async def append(self, _key: object, entries: list[Any]) -> None:
        self.entered.set()
        await self.release.wait()
        self.batches.append(entries)

    async def load(self, _key: object) -> list[Any] | None:
        loaded = [entry for batch in self.batches for entry in batch]
        return loaded or None


async def test_an_append_past_its_generation_check_lands_after_takeover(
    store: FileLeaseStore,
) -> None:
    """SessionStore has no conditional write. A batch that already passed the generation check
    still commits after a newer generation is claimed, and a later load returns it."""
    inner = _InFlight()
    key = {"project_key": "p", "session_id": SESSION}
    stale = [{"type": "assistant", "uuid": "in-flight"}]
    async with session_lease(store, SESSION, LONG) as lease:
        fenced = fenced_session_store(inner, lease)
        writing = asyncio.create_task(fenced.append(key, stale))
        await inner.entered.wait()
        assert store.claim(SESSION, _held(lease).generation + 1, "intruder")
        inner.release.set()
        await writing
        assert await fenced.load(key) == stale


async def test_the_fenced_store_exposes_only_what_the_real_store_implements(
    store: FileLeaseStore,
) -> None:
    """The SDK probes optional methods on the store's class; a blanket delegate would claim
    list_sessions exists and then fail when the SDK called it."""
    async with session_lease(store, SESSION, LONG) as lease:
        fenced = fenced_session_store(_Store(), lease)
        assert await fenced.list_subkeys({}) == ["sub"]
        assert not hasattr(type(fenced), "list_sessions")
        assert fenced_session_store(None, lease) is None


def test_a_claim_whose_reply_was_lost_still_counts_as_won(
    store: FileLeaseStore,
) -> None:
    """On NFS a retransmitted link reports EEXIST for the record it just created."""
    owner = uuid.uuid4().hex
    assert store.claim(SESSION, 1, owner)
    assert store.claim(SESSION, 1, owner)
    assert not store.claim(SESSION, 1, "someone-else")
