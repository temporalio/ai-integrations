"""Retry of a lost Claude turn."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from temporalio import workflow
from temporalio.claude_agent_sdk import _session
from temporalio.claude_agent_sdk import workflow as claude_workflow
from temporalio.claude_agent_sdk._activity import _Progress
from temporalio.claude_agent_sdk._models import (
    CLAUDE_NATIVE_OUTCOME_UNKNOWN_ERROR_TYPE,
    CLAUDE_SANDBOX_LOST_ERROR_TYPE,
    CLAUDE_SESSION_DEGRADED_ERROR_TYPE,
    CLAUDE_SESSION_LEASE_LOST_ERROR_TYPE,
    CLAUDE_SESSION_MIRROR_ERROR_TYPE,
    RESUMABLE_DETAIL,
    ClaudeDeferredTool,
    ClaudeNativeCall,
    ClaudeNativeDecision,
    ClaudeTurnInput,
    ClaudeTurnOptions,
    ClaudeTurnProgress,
    ClaudeTurnResult,
)
from temporalio.claude_agent_sdk._native_hook import UNKNOWN_OUTCOME, call_key
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    CancelledError,
    TimeoutError,
    TimeoutType,
)
from temporalio.workflow import ActivityConfig

SESSION = "11111111-2222-3333-4444-555555555555"
Attempts = Callable[..., list[ClaudeTurnInput]]


@pytest.fixture(autouse=True)
def _outside_a_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_dispatch` only ever runs in workflow code; give it a plain logger."""
    monkeypatch.setattr(workflow, "logger", logging.getLogger(__name__))


def _turn(**overrides: Any) -> ClaudeTurnInput:
    return ClaudeTurnInput(
        prompt="hello",
        prompt_uuid="99999999-8888-7777-6666-555555555555",
        options=ClaudeTurnOptions(),
        session_id=SESSION,
        **overrides,
    )


def _result() -> ClaudeTurnResult:
    return ClaudeTurnResult(text="done", session_id=SESSION)


def _lost(
    timeout: TimeoutType = TimeoutType.HEARTBEAT, **details: Any
) -> ActivityError:
    """What a worker that died (or timed out) looks like to the workflow."""
    error = ActivityError(
        "Activity task failed",
        scheduled_event_id=1,
        started_event_id=2,
        identity="w",
        activity_type="t",
        activity_id="a",
        retry_state=None,
    )
    error.__cause__ = TimeoutError(
        "activity timeout",
        type=timeout,
        last_heartbeat_details=[details] if details else [],
    )
    return error


def _resumable() -> ActivityError:
    return _lost(resumable=True)


@pytest.fixture
def attempts(monkeypatch: pytest.MonkeyPatch) -> Attempts:
    """Record each attempt's input, replaying `outcomes` in order."""
    seen: list[ClaudeTurnInput] = []

    def arm(*outcomes: ClaudeTurnResult | BaseException) -> list[ClaudeTurnInput]:
        pending = list(outcomes)

        async def run_turn(
            turn: ClaudeTurnInput, *, activity_config: ActivityConfig | None = None
        ) -> ClaudeTurnResult:
            _ = activity_config
            seen.append(turn)
            outcome = pending.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(claude_workflow, "run_turn", run_turn)
        return seen

    return arm


async def _dispatch(
    session: _session.ClaudeSession,
    turn: ClaudeTurnInput,
    config: ActivityConfig | None = None,
) -> ClaudeTurnResult:
    activity_config: ActivityConfig = {} if config is None else config
    return await _session._dispatch(session, turn, activity_config)


async def test_a_lost_resumable_turn_is_retried_under_the_same_prompt_uuid(
    attempts: Attempts,
) -> None:
    """The Activity finds the prompt in the store by its uuid, so every attempt must carry the
    same one, and each retry is a new attempt number."""
    seen = attempts(_resumable(), _result())
    assert (await _dispatch(_session.ClaudeSession(SESSION), _turn())).text == "done"
    assert [t.prompt_uuid for t in seen] == [_turn().prompt_uuid] * 2
    assert [t.attempt for t in seen] == [1, 2]


def _lease_lost(*, non_retryable: bool, **details: Any) -> ApplicationError:
    return ApplicationError(
        "lease lost",
        {"session_id": SESSION, "resumable": True, **details},
        type=CLAUDE_SESSION_LEASE_LOST_ERROR_TYPE,
        non_retryable=non_retryable,
    )


async def test_a_lease_lost_before_the_cli_started_is_retried(
    attempts: Attempts,
) -> None:
    seen = attempts(_lease_lost(non_retryable=False), _result())
    assert (await _dispatch(_session.ClaudeSession(SESSION), _turn())).text == "done"
    assert len(seen) == 2


async def test_a_lease_lost_after_the_cli_started_is_not_retried_but_keeps_its_ledger(
    attempts: Attempts,
) -> None:
    """Another attempt owns the session and this one may have run tools: nothing runs it again,
    yet what the lost attempt recorded still reaches the session."""
    call = {
        "tool_use_id": "toolu_x",
        "tool_name": "Read",
        "key": "k",
        "status": "finished",
    }
    seen = attempts(
        _lease_lost(non_retryable=True, native_calls=[call], workspace_revision=7),
        _result(),
    )
    session = _session.ClaudeSession(SESSION)
    with pytest.raises(ApplicationError):
        await _dispatch(session, _turn())
    assert len(seen) == 1
    assert [c.tool_use_id for c in session.state.native_calls] == ["toolu_x"]
    assert session.state.workspace_revision == 7


def _hub_with(session: _session.ClaudeSession) -> _session._BridgeHub:
    hub = _session._BridgeHub()
    hub.sessions[SESSION] = session
    return hub


async def test_each_attempt_is_the_one_the_session_listens_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listening: list[str | None] = []

    async def run_turn(
        _turn: ClaudeTurnInput, *, activity_config: ActivityConfig | None = None
    ) -> ClaudeTurnResult:
        _ = activity_config
        listening.append(session.attempt_id)
        if len(listening) == 1:
            raise _resumable()
        return _result()

    monkeypatch.setattr(claude_workflow, "run_turn", run_turn)
    session = _session.ClaudeSession(SESSION)
    await _dispatch(session, _turn())
    prompt = _turn().prompt_uuid
    assert listening == [f"{prompt}.1", f"{prompt}.2"]


def test_progress_from_a_superseded_attempt_is_dropped():
    """A zombie attempt must not rewind the revision or the ledger the live one runs on."""
    session = _session.ClaudeSession(SESSION)
    session.attempt_id = "p.2"
    hub = _hub_with(session)
    call = ClaudeNativeCall(
        tool_use_id="toolu_x", tool_name="Read", key="k", status="started"
    )

    hub.on_progress(
        ClaudeTurnProgress(
            session_id=SESSION,
            attempt_id="p.1",
            workspace_revision=9,
            native_calls=[call],
            mirror_errors=["dropped"],
        )
    )
    assert session.state.workspace_revision is None
    assert not session.state.native_calls and session.state.degraded is None

    hub.on_progress(
        ClaudeTurnProgress(
            session_id=SESSION,
            attempt_id="p.2",
            workspace_revision=3,
            native_calls=[call],
        )
    )
    assert session.state.workspace_revision == 3
    assert [c.tool_use_id for c in session.state.native_calls] == ["toolu_x"]


@pytest.mark.parametrize(
    "lost",
    [
        # Not resumable: another attempt would re-send the prompt and repeat the turn.
        _lost(resumable=False),
        _lost(),
        # The turn's own time ran out; a retry would just spend it again.
        _lost(TimeoutType.START_TO_CLOSE, resumable=True),
    ],
    ids=["worker-without-a-store", "heartbeat-without-details", "start-to-close"],
)
async def test_a_lost_turn_that_is_not_resumable_is_not_retried(
    attempts: Attempts, lost: ActivityError
) -> None:
    seen = attempts(lost, _result())
    with pytest.raises(ActivityError):
        await _dispatch(_session.ClaudeSession(SESSION), _turn())
    assert len(seen) == 1


async def test_only_the_resume_of_a_lost_attempt_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Activity refuses that resume when it cannot see the lost attempt's lease."""
    flags: list[bool] = []
    pending: list[BaseException | ClaudeTurnResult] = [_resumable(), _result()]

    async def run_turn(
        turn: ClaudeTurnInput, *, activity_config: ActivityConfig | None = None
    ) -> ClaudeTurnResult:
        _ = activity_config
        flags.append(turn.resumes_lost_attempt)
        outcome = pending.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(claude_workflow, "run_turn", run_turn)
    await _dispatch(_session.ClaudeSession(SESSION), _turn())
    assert flags == [False, True]


async def test_a_lost_lease_is_an_unknown_outcome_not_a_resume(
    attempts: Attempts,
) -> None:
    """The CLI was killed mid-run with another attempt on the session: never run it again."""
    lost = ApplicationError(
        "lease lost", type="ClaudeSessionLeaseLost", non_retryable=True
    )
    session = _session.ClaudeSession(SESSION, started=True)
    seen = attempts(lost, _result())
    with pytest.raises(ApplicationError, match="lease lost"):
        await _dispatch(session, _turn())
    assert len(seen) == 1


async def test_a_lost_turn_gives_up_after_the_attempt_cap(attempts: Attempts) -> None:
    seen = attempts(*[_resumable()] * _session._RESUME_ATTEMPTS)
    with pytest.raises(ActivityError):
        await _dispatch(_session.ClaudeSession(SESSION), _turn())
    assert len(seen) == _session._RESUME_ATTEMPTS


async def test_a_cancelled_turn_is_not_resumed(attempts: Attempts) -> None:
    """Cancellation is deliberate; resuming it would undo the caller's decision."""
    cancelled = _resumable()
    cancelled.__cause__ = CancelledError("Cancelled")
    seen = attempts(cancelled, _result())
    with pytest.raises(ActivityError):
        await _dispatch(_session.ClaudeSession(SESSION), _turn())
    assert len(seen) == 1


BASH = ClaudeDeferredTool(
    tool_use_id="toolu_x", name="Bash", input={"command": "echo hi >> f"}
)


def _approved(call: ClaudeDeferredTool = BASH) -> ClaudeTurnInput:
    """The run that resumes an approved native call: the one native call a lost attempt of it
    may have run."""
    decision = ClaudeNativeDecision(call=call, approved=True)
    return _turn(native_tools=[call.name], native_decision=decision)


def _entry(call: ClaudeDeferredTool, status: str) -> dict:
    return {
        "tool_use_id": call.tool_use_id,
        "tool_name": call.name,
        "key": "k",
        "run": "r",
        "status": status,
    }


async def test_a_native_turn_that_ran_no_native_call_is_resumed(
    attempts: Attempts,
) -> None:
    """Every native call defers before it runs, so a run resuming no approved call had none."""
    seen = attempts(_resumable(), _result())
    assert (
        await _dispatch(_session.ClaudeSession(SESSION), _turn(native_tools=["Bash"]))
    ).text == "done"
    assert len(seen) == 2


async def test_an_unsettled_approved_bash_call_fails_the_message_as_outcome_unknown(
    attempts: Attempts,
) -> None:
    """Nothing says whether the lost attempt ran it, and a repeat could duplicate it: the message
    fails, the call stays `started` in the ledger, and the next run settles it as unknown."""
    seen = attempts(_resumable(), _result())
    session = _session.ClaudeSession(SESSION, started=True)
    with pytest.raises(ApplicationError) as raised:
        await _dispatch(session, _approved())
    assert (
        raised.value.type == CLAUDE_NATIVE_OUTCOME_UNKNOWN_ERROR_TYPE
        and raised.value.non_retryable
    )
    assert len(seen) == 1
    [entry] = session.state.native_calls
    assert (entry.tool_use_id, entry.status) == ("toolu_x", "started")
    assert entry.key == call_key("Bash", BASH.input), (
        "the hook denies a repeat by this key"
    )
    assert session.state.native_unresolved == BASH
    settled = _session._settle(session)
    assert settled is not None and settled.reason == UNKNOWN_OUTCOME


@pytest.mark.parametrize("where", ["heartbeat", "progress signal"])
async def test_an_approved_call_recorded_finished_is_resumed_and_carried(
    attempts: Attempts,
    where: str,
) -> None:
    """The ledger proves the call finished, so the retry runs, and it carries the entry, which
    makes the hook on whatever worker it lands refuse the call instead of running it again."""
    session = _session.ClaudeSession(SESSION, started=True)
    if where == "heartbeat":
        lost = _lost(resumable=True, native_calls=[_entry(BASH, "finished")])
    else:
        _session._record_progress(session, [_entry(BASH, "finished")], None)
        lost = _resumable()
    seen = attempts(lost, _result())
    assert (await _dispatch(session, _approved())).text == "done"
    assert [(c.tool_use_id, c.status) for c in seen[1].native_calls] == [
        ("toolu_x", "finished")
    ]
    assert seen[1].resumes_lost_attempt


async def test_a_heartbeat_never_reopens_a_call_the_session_has_settled(
    attempts: Attempts,
) -> None:
    session = _session.ClaudeSession(SESSION, started=True)
    _session._record_progress(session, [_entry(BASH, "finished")], None)
    attempts(_lost(resumable=True, native_calls=[_entry(BASH, "started")]), _result())
    await _dispatch(session, _approved())
    assert [c.status for c in session.state.native_calls] == ["finished"]


def _sandbox_lost(**details: Any) -> ApplicationError:
    """A sandbox loss whose final ledger pull also failed."""
    payload: dict[str, Any] = {
        "session_id": SESSION,
        "num_turns": 0,
        "sandbox_snapshot": None,
        RESUMABLE_DETAIL: True,
        "ledger_unreadable": True,
        "native_calls": [],
    }
    payload.update(details)
    return ApplicationError(
        "Claude turn lost its sandbox",
        payload,
        type=CLAUDE_SANDBOX_LOST_ERROR_TYPE,
    )


READ = ClaudeDeferredTool(tool_use_id="toolu_r", name="Read", input={"file_path": "f"})


@pytest.mark.parametrize(
    ("native_calls", "turn"),
    [
        ([], _turn(native_tools=["Bash"])),
        ([_entry(READ, "started")], _approved(READ)),
    ],
    ids=["no-native-effect", "repeatable-effect"],
)
async def test_a_lost_sandbox_is_resumed(
    attempts: Attempts, native_calls: list[dict], turn: ClaudeTurnInput
) -> None:
    seen = attempts(_sandbox_lost(native_calls=native_calls), _result())
    assert (
        await _dispatch(_session.ClaudeSession(SESSION, started=True), turn)
    ).text == "done"
    assert len(seen) == 2


async def test_only_a_heartbeat_timeout_hands_the_lost_attempts_lease_to_the_retry(
    attempts: Attempts,
) -> None:
    """The timeout is what proves the lease went a whole TTL unrenewed; a failure the attempt
    raised itself proves nothing about it."""
    seen = attempts(
        _lost(resumable=True, lease=[3, 7]), _sandbox_lost(lease=[4, 1]), _result()
    )
    await _dispatch(_session.ClaudeSession(SESSION, started=True), _turn())
    assert [t.lost_lease for t in seen] == [None, (3, 7), None]


async def test_a_lost_sandbox_with_an_unsettled_effect_is_not_resumed(
    attempts: Attempts,
) -> None:
    seen = attempts(_sandbox_lost(native_calls=[_entry(BASH, "started")]), _result())
    session = _session.ClaudeSession(SESSION, started=True)
    with pytest.raises(ApplicationError) as raised:
        await _dispatch(session, _approved())
    assert raised.value.type == CLAUDE_NATIVE_OUTCOME_UNKNOWN_ERROR_TYPE
    assert raised.value.non_retryable
    assert len(seen) == 1
    assert [(call.tool_use_id, call.status) for call in session.state.native_calls] == [
        ("toolu_x", "started")
    ]


async def test_an_unsettled_repeatable_call_is_resumed_and_keeps_its_start(
    attempts: Attempts,
) -> None:
    """A Read may run again, so the retry runs it."""
    seen = attempts(_resumable(), _result())
    await _session._dispatch(
        _session.ClaudeSession(SESSION, started=True),
        _approved(READ),
        {},
    )
    assert len(seen) == 2


async def test_a_lost_attempts_rounds_are_charged_to_the_budget(
    attempts: Attempts,
) -> None:
    """The retry gets what the lost attempt left of max_turns."""
    budget = _session._Budget(ClaudeTurnOptions(max_turns=5))
    seen = attempts(_lost(resumable=True, num_turns=2), _result())
    await _session._dispatch(
        _session.ClaudeSession(SESSION),
        _turn(),
        {},
        None,
        budget,
    )
    assert budget.turns == 2
    assert [t.options.max_turns for t in seen] == [None, 3]


async def test_a_lost_attempt_that_used_up_the_budget_is_not_resumed(
    attempts: Attempts,
) -> None:
    budget = _session._Budget(ClaudeTurnOptions(max_turns=4))
    seen = attempts(_lost(resumable=True, num_turns=4), _result())
    with pytest.raises(ApplicationError) as raised:
        await _session._dispatch(
            _session.ClaudeSession(SESSION),
            _turn(),
            {},
            None,
            budget,
        )
    assert raised.value.type == "ClaudeTurnBudgetExhausted"
    assert len(seen) == 1


async def test_a_lost_attempts_cost_is_charged_to_the_budget(
    attempts: Attempts,
) -> None:
    """The retry gets what the lost attempt left of max_budget_usd, and that spend stays charged.
    A heartbeat from before cost was recorded charges nothing. Spending the cap is not resumed.
    """
    options = ClaudeTurnOptions(max_budget_usd=1.0)
    turn = _turn().model_copy(update={"options": options})
    budget = _session._Budget(options)
    seen = attempts(
        _lost(resumable=True, num_turns=1, cost_usd=0.8),
        ClaudeTurnResult(
            text="done", session_id=SESSION, num_turns=1, total_cost_usd=0.1
        ),
    )
    session = _session.ClaudeSession(SESSION)
    result = await _session._dispatch(session, turn, {}, None, budget)
    budget.spend(session, result)
    assert result.text == "done"
    assert seen[0].options.max_budget_usd == 1.0
    assert seen[1].options.max_budget_usd == pytest.approx(0.2)
    assert budget.cost_usd == pytest.approx(0.9)
    assert session.state.cost_usd == pytest.approx(0.9)

    progress = _Progress(_turn(), None, None, None, True, [])
    assert progress.details()["cost_usd"] == 0

    bare = _session._Budget(options)
    before = len(seen)
    attempts(_lost(resumable=True, num_turns=1), _result())
    await _session._dispatch(_session.ClaudeSession(SESSION), turn, {}, None, bare)
    assert bare.cost_usd == 0
    assert seen[before + 1].options.max_budget_usd == 1.0

    spent = _session._Budget(options)
    before = len(seen)
    attempts(_lost(resumable=True, num_turns=1, cost_usd=1.0), _result())
    with pytest.raises(ApplicationError) as raised:
        await _session._dispatch(_session.ClaudeSession(SESSION), turn, {}, None, spent)
    assert raised.value.type == "ClaudeTurnBudgetExhausted"
    assert len(seen) == before + 1
    assert spent.cost_usd == pytest.approx(1.0)


def test_each_run_is_charged_its_own_cost_whichever_total_it_reports() -> None:
    """A SessionStore resume reports the run's own cost. A resume of Claude Code's own transcript
    reports the session total it restored, which is the last completed run's total: a lost
    attempt's spend is not in it, because a sandbox resumes from the snapshot before it."""
    store = _session.ClaudeSession(SESSION)
    budget = _session._Budget(ClaudeTurnOptions())
    budget.spend(store, ClaudeTurnResult(total_cost_usd=0.5))
    budget.spend(store, ClaudeTurnResult(total_cost_usd=0.6))
    assert store.state.cost_usd == pytest.approx(1.1)

    snapshot = _session.ClaudeSession(SESSION)
    budget.spend(snapshot, ClaudeTurnResult(total_cost_usd=0.5))
    lost = ClaudeTurnResult(total_cost_usd=0.9, cumulative_cost=True)
    budget.spend(snapshot, lost, completed=False)
    budget.spend(snapshot, ClaudeTurnResult(total_cost_usd=0.7, cumulative_cost=True))
    assert snapshot.state.cost_usd == pytest.approx(0.5 + 0.4 + 0.2)


async def test_a_lost_attempts_reported_cost_counts_against_the_session_cap(
    attempts: Attempts,
) -> None:
    """An attempt lost after its result (say, while its sandbox was snapshotted) heartbeated its
    own cost; all of it is charged, however much the session had spent before."""
    session = _session.ClaudeSession(SESSION, started=True, max_cost_usd=1.0)
    session.state.cost_usd = session.state.restored_cost_usd = 0.5
    budget = _session._Budget(ClaudeTurnOptions())
    attempts(_lost(resumable=True, num_turns=1, cost_usd=0.8), _result())
    await _session._dispatch(session, _turn(), {}, None, budget)
    assert session.state.cost_usd == pytest.approx(1.3)


async def test_a_failed_runs_cost_is_charged(attempts: Attempts) -> None:
    failed = ApplicationError(
        "Claude run failed (error_max_turns)",
        {"subtype": "error_max_turns", "num_turns": 3, "total_cost_usd": 0.4},
        type="ClaudeResultError",
        non_retryable=True,
    )
    attempts(failed)
    session = _session.ClaudeSession(SESSION)
    budget = _session._Budget(ClaudeTurnOptions())
    with pytest.raises(ApplicationError):
        await _session._dispatch(session, _turn(), {}, None, budget)
    assert (budget.turns, session.state.cost_usd) == (3, pytest.approx(0.4))


async def test_a_completed_turn_whose_store_dropped_a_batch_is_not_a_success(
    attempts: Attempts,
) -> None:
    """A finished result fails as ``ClaudeSessionMirrorError`` when the store dropped a batch."""
    session = _session.ClaudeSession(SESSION)
    seen = attempts(_result().model_copy(update={"mirror_errors": ["store is down"]}))
    with pytest.raises(ApplicationError) as raised:
        await _dispatch(session, _turn())
    assert raised.value.type == CLAUDE_SESSION_MIRROR_ERROR_TYPE
    assert raised.value.non_retryable
    assert "store is down" in raised.value.message
    assert session.state.degraded == "store is down"
    assert session.state.started
    assert len(seen) == 1


async def test_a_completed_turn_whose_store_dropped_a_batch_is_still_charged(
    attempts: Attempts,
) -> None:
    attempts(
        _result().model_copy(
            update={
                "mirror_errors": ["store is down"],
                "num_turns": 2,
                "total_cost_usd": 0.3,
            }
        )
    )
    session = _session.ClaudeSession(SESSION)
    budget = _session._Budget(ClaudeTurnOptions())
    with pytest.raises(ApplicationError):
        await _session._dispatch(session, _turn(), {}, None, budget)
    assert (budget.turns, session.state.cost_usd) == (2, pytest.approx(0.3))


_NOW = datetime.fromtimestamp(1_800_000_000, timezone.utc)


@pytest.mark.parametrize(
    ("resets_at", "retries", "expected"),
    [
        (1_800_000_300, 0, timedelta(minutes=5)),
        (1_800_000_301, 0, None),
        (None, 3, timedelta(minutes=4)),
        (None, 4, None),
    ],
    ids=[
        "reset-at-five-minutes",
        "reset-past-five-minutes",
        "backoff-under-the-cap",
        "backoff-past-the-cap",
    ],
)
def test_an_api_error_past_the_retry_cap_is_not_resumed(
    monkeypatch: pytest.MonkeyPatch,
    resets_at: int | None,
    retries: int,
    expected: timedelta | None,
) -> None:
    """The wait is the doubled backoff, or the rate-limit reset when that is later.

    Five minutes is the longest wait that still resumes. A reset or a backoff one second past
    that does not resume; the delay is not shortened to the cap and then resumed.
    """
    monkeypatch.setattr(workflow, "now", lambda: _NOW)
    details: dict[str, Any] = {"session_store": True, "api_error_status": 429}
    if resets_at is not None:
        details["rate_limit_resets_at"] = resets_at
    error = ApplicationError(
        "rate limited",
        details,
        type="ClaudeResultError",
        non_retryable=True,
    )
    assert (
        _session._api_error_delay(
            _session.ClaudeSession(SESSION), _turn(), error, retries
        )
        == expected
    )


@pytest.mark.parametrize(
    ("details", "resumed"),
    [
        ({"error_kind": "rate_limit"}, True),
        ({"error_kind": "server_error"}, True),
        ({"error_kind": "unknown"}, False),
        (
            {"error_kind": "authentication_failed", "terminal_reason": "api_error"},
            False,
        ),
        ({"error_kind": "billing_error", "api_error_status": 529}, False),
        ({"error_kind": "invalid_request", "terminal_reason": "api_error"}, False),
        # An OAuth refresh refused: Claude Code 2.1.294 reports it with no HTTP status.
        ({"error_kind": "account_on_hold", "terminal_reason": "api_error"}, False),
    ],
)
def test_the_sdk_error_kind_decides_an_api_error_resume(
    monkeypatch: pytest.MonkeyPatch, details: dict[str, Any], resumed: bool
) -> None:
    monkeypatch.setattr(workflow, "now", lambda: _NOW)
    error = ApplicationError(
        "failed",
        {"session_store": True, **details},
        type="ClaudeResultError",
        non_retryable=True,
    )
    delay = _session._api_error_delay(
        _session.ClaudeSession(SESSION), _turn(), error, 0
    )
    assert (delay is not None) is resumed


async def test_a_lost_attempt_whose_store_dropped_a_batch_fails_the_message_as_degraded(
    attempts: Attempts,
) -> None:
    """Its last heartbeat carried mirror errors, so the store copy it would resume from is
    incomplete: the session is marked degraded and nothing is resumed."""
    session = _session.ClaudeSession(SESSION)
    seen = attempts(_lost(resumable=True, mirror_errors=["store is down"]), _result())
    with pytest.raises(ApplicationError) as raised:
        await _dispatch(session, _turn())
    assert raised.value.type == CLAUDE_SESSION_DEGRADED_ERROR_TYPE
    assert session.state.degraded == "store is down"
    assert len(seen) == 1


async def test_a_dropped_batch_reported_mid_run_stops_the_resume_of_a_stale_heartbeat(
    attempts: Attempts,
) -> None:
    """The progress signal marked the session degraded while the attempt ran; its last heartbeat,
    sent before the mirror error, still said resumable."""
    session = _session.ClaudeSession(SESSION)
    seen = attempts(_lost(resumable=True), _result())
    session.state.degraded = "store is down"
    with pytest.raises(ApplicationError) as raised:
        await _dispatch(session, _turn())
    assert raised.value.type == CLAUDE_SESSION_DEGRADED_ERROR_TYPE
    assert len(seen) == 1


def test_an_attempt_whose_store_dropped_a_batch_says_it_is_not_resumable():
    mirror_errors: list[str] = []
    progress = _Progress(_turn(), None, None, None, True, mirror_errors)
    assert progress.details()[RESUMABLE_DETAIL] is True

    mirror_errors.append("store is down")

    details = progress.details()
    assert details[RESUMABLE_DETAIL] is False
    assert details["mirror_errors"] == ["store is down"]


async def test_a_lost_attempts_workspace_revision_is_recorded_and_restored_by_the_retry(
    attempts: Attempts,
) -> None:
    session = _session.ClaudeSession(SESSION, started=True)
    seen = attempts(_lost(resumable=True, workspace_revision=4), _result())
    await _dispatch(session, _turn())
    assert [t.workspace_revision for t in seen] == [None, 4]
    assert session.state.workspace_revision == 4


async def test_claudes_own_failure_is_not_resumed(attempts: Attempts) -> None:
    """A non-retryable failure is Claude's answer, not a lost turn."""
    seen = attempts(
        ApplicationError("bad prompt", type="ClaudeResultError", non_retryable=True)
    )
    with pytest.raises(ApplicationError):
        await _dispatch(_session.ClaudeSession(SESSION), _turn())
    assert len(seen) == 1


async def test_a_retryable_failure_after_the_cli_ran_is_not_resumed(
    attempts: Attempts,
) -> None:
    """An Activity that raised reported its outcome; only rejections before the CLI repeat."""
    seen = attempts(ApplicationError("bug in the activity", type="KeyError"), _result())
    with pytest.raises(ApplicationError):
        await _dispatch(_session.ClaudeSession(SESSION), _turn())
    assert len(seen) == 1


async def test_a_retryable_rejection_is_resumed(attempts: Attempts) -> None:
    """A session still leased by the dying attempt frees up once that lease expires."""
    leased = ApplicationError("still leased", type="ClaudeSessionLeased")
    seen = attempts(leased, _result())
    session = _session.ClaudeSession(SESSION, started=True)
    assert (await _dispatch(session, _turn())).text == "done"
    assert [t.resume for t in seen] == [True, True]


async def test_a_resume_leaves_the_lease_wait_to_the_activity(
    attempts: Attempts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry's Activity watches the lost attempt's lease for a whole TTL whenever it starts,
    so a workflow timer before it would only add latency."""

    async def sleep(duration: timedelta) -> None:
        raise AssertionError(f"the workflow slept {duration} before resuming")

    monkeypatch.setattr(workflow, "sleep", sleep)
    attempts(_resumable(), _result())
    config: ActivityConfig = {"heartbeat_timeout": timedelta(seconds=30)}
    assert (
        await _dispatch(_session.ClaudeSession(SESSION), _turn(), config)
    ).text == "done"


async def test_only_a_completed_run_marks_the_session_started(
    attempts: Attempts,
) -> None:
    """``started`` tells the Activity a transcript must exist; a lost or rejected attempt proves
    nothing, and the Activity looks for the transcript itself."""
    session = _session.ClaudeSession(SESSION)
    seen = attempts(_resumable(), _result())
    await _dispatch(session, _turn())
    assert [t.resume for t in seen] == [False, False]
    assert session.state.started


# Pytest discovers fixtures by name.
_ = _outside_a_workflow
