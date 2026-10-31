"""Which trials come back as results, and which Temporal runs again."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest
from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import JsonValue

from temporalio.api.failure.v1 import Failure
from temporalio.api.workflow.v1 import PendingActivityInfo
from temporalio.client import (
    Client,
    WorkflowExecutionStatus,
    WorkflowFailureError,
    WorkflowHandle,
)
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.harbor import (
    HarborPlugin,
    TrialContext,
    TrialHooks,
    TrialOutcome,
)
from temporalio.harbor._activity import _OUTCOME_MAX
from temporalio.worker import Worker
from tests.harbor_fixtures import agents, history, tasks
from tests.harbor_fixtures.workflows import (
    RetryJob,
    RunJob,
    RunOneTrial,
    TrialArgs,
    new_worker,
    run_one,
)

on_posix = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)


def _trial(task: Path, trials_dir: Path, name: str) -> TrialConfig:
    return TrialConfig(
        task=TaskConfig(path=task),
        trial_name=name,
        trials_dir=trials_dir,
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )


def _with(client: Client, hooks: TrialHooks) -> Client:
    config = client.config()
    config["plugins"] = [HarborPlugin(trial_hooks=hooks)]
    return Client(**config)


@on_posix
async def test_recorded_failure_is_returned_not_retried(
    harbor_client: Client, tmp_path: Path
) -> None:
    # The verifier writes no reward the first time; a retry would succeed.
    config = _trial(
        tasks.flaky(tmp_path / "tasks", failures=1), tmp_path / "trials", "flaky__0"
    )
    outcome, attempts = await run_one(harbor_client, TrialArgs(config=config))

    assert outcome.result.exception_info is not None
    assert outcome.result.exception_info.exception_type == "RewardFileNotFoundError"
    assert outcome.attempt == 1
    assert attempts == [1]


def _failing(error: str) -> AgentConfig:
    return AgentConfig(import_path=agents.FAILING_AGENT, kwargs={"error": error})


async def _with_two_attempts(
    client: Client, config: TrialConfig
) -> tuple[TrialOutcome, list[int], list[Failure]]:
    """Run the trial allowing two attempts: its outcome, final attempt and failures."""
    async with new_worker(client, RunOneTrial) as worker:
        handle = await client.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=2),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()
    return (
        outcome,
        await history.final_attempts(handle),
        await history.activity_failures(handle),
    )


@on_posix
@pytest.mark.parametrize(
    ("make", "agent", "environment", "error"),
    [
        pytest.param(
            partial(tasks.slow, seconds=30),
            AgentConfig(name="oracle", override_timeout_sec=1),
            tasks.LOCAL_ENV,
            "AgentTimeoutError",
            id="agent-timeout",
        ),
        pytest.param(
            tasks.passing,
            _failing("AgentAuthenticationError"),
            tasks.LOCAL_ENV,
            "AgentAuthenticationError",
            id="auth-error",
        ),
        # Harbor records each step's agent run on that step, not on the trial.
        pytest.param(
            tasks.two_steps,
            AgentConfig(name="oracle"),
            tasks.UNSTOPPABLE_ENV,
            "RuntimeError",
            id="steps-ran-then-environment-stop",
        ),
    ],
)
async def test_failure_after_the_agent_started_is_returned(
    harbor_client: Client,
    tmp_path: Path,
    make: Callable[[Path], Path],
    agent: AgentConfig,
    environment: str,
    error: str,
) -> None:
    config = _trial(make(tmp_path / "tasks"), tmp_path / "trials", "trial__0")
    config.agent = agent
    config.environment = EnvironmentConfig(import_path=environment)
    outcome, attempts, failures = await _with_two_attempts(harbor_client, config)

    assert outcome.result.exception_info is not None
    assert outcome.result.exception_info.exception_type == error
    assert outcome.attempt == 1
    assert attempts == [1]
    assert failures == []


@on_posix
@pytest.mark.parametrize(
    ("make", "environment", "error"),
    [
        pytest.param(
            tasks.passing,
            tasks.UNSTARTABLE_ENV,
            "RuntimeError",
            id="environment-start",
        ),
        pytest.param(
            partial(tasks.two_steps, ready=False),
            tasks.LOCAL_ENV,
            "HealthcheckError",
            id="first-step-healthcheck",
        ),
    ],
)
async def test_failure_before_the_agent_started_is_retried(
    harbor_client: Client,
    tmp_path: Path,
    make: Callable[[Path], Path],
    environment: str,
    error: str,
) -> None:
    config = _trial(make(tmp_path / "tasks"), tmp_path / "trials", "trial__0")
    config.environment = EnvironmentConfig(import_path=environment)
    outcome, attempts, failures = await _with_two_attempts(harbor_client, config)

    # Temporal ran out of attempts; the last attempt's outcome comes back.
    assert outcome.attempt == 2
    assert attempts == [2]
    [failure] = failures
    assert failure.application_failure_info.type == error
    assert not failure.application_failure_info.non_retryable


@on_posix
@pytest.mark.parametrize(
    ("make", "error"),
    [
        *(
            pytest.param(tasks.passing, error, id=error)
            for error in (
                "ApiOverloadedError",
                "ApiInternalServerError",
                "ApiConnectionClosedError",
                "ApiResponseStalledError",
                "NetworkConnectionError",
            )
        ),
        pytest.param(tasks.two_steps, "ApiOverloadedError", id="steps"),
    ],
)
async def test_transient_provider_error_is_retried(
    harbor_client: Client, tmp_path: Path, make: Callable[[Path], Path], error: str
) -> None:
    config = _trial(make(tmp_path / "tasks"), tmp_path / "trials", "trial__0")
    config.agent = _failing(error)
    outcome, attempts, failures = await _with_two_attempts(harbor_client, config)

    assert outcome.attempt == 2
    assert attempts == [2]
    [failure] = failures
    info = failure.application_failure_info
    assert info.type == error
    assert not info.non_retryable
    assert not info.HasField("next_retry_delay")


async def _awaiting_retry(handle: WorkflowHandle[Any, Any]) -> PendingActivityInfo:
    async with asyncio.timeout(60):
        while True:
            description = await handle.describe()
            assert description.status == WorkflowExecutionStatus.RUNNING, (
                "the trial was not retried"
            )
            for pending in description.raw_description.pending_activities:
                if pending.HasField("last_failure"):
                    return pending
            await asyncio.sleep(0.1)


@on_posix
async def test_rate_limited_trial_is_retried_after_two_minutes(
    harbor_client: Client, tmp_path: Path
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    config.agent = _failing("ApiRateLimitError")
    async with new_worker(harbor_client, RunOneTrial) as worker:
        handle = await harbor_client.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=2),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        try:
            pending = await _awaiting_retry(handle)
        finally:
            with contextlib.suppress(Exception):
                await handle.terminate()

    assert pending.attempt == 2
    assert pending.current_retry_interval.seconds == 120
    failure = pending.last_failure
    # The server replaces an oversized failure with its own, the original as cause.
    if failure.HasField("server_failure_info"):
        failure = failure.cause
    assert failure.application_failure_info.type == "ApiRateLimitError"
    assert not failure.application_failure_info.non_retryable


@on_posix
async def test_job_retry_config_is_ignored(
    harbor_client: Client, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = JobConfig(
        job_name="retrying",
        jobs_dir=tmp_path / "jobs",
        tasks=[TaskConfig(path=tasks.flaky(tmp_path / "tasks", failures=1))],
        retry=RetryConfig(max_retries=3, exclude_exceptions=set()),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    async with new_worker(harbor_client, RunJob) as worker:
        handle = await harbor_client.start_workflow(
            RunJob.run, config, id=f"job-{uuid.uuid4()}", task_queue=worker.task_queue
        )
        result = await handle.result()

    assert result.stats.n_errored_trials == 1
    assert result.stats.n_retries == 0
    assert set(await history.final_attempts(handle)) == {1}
    assert any("JobConfig.retry" in r.getMessage() for r in caplog.records)


class RetryFirstAttempt(TrialHooks):
    """Send every trial's first attempt back to Temporal."""

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        if context.attempt == 1:
            raise ApplicationError("try once more", type="TryAgain")
        return None


@on_posix
async def test_retries_are_counted_from_outcomes(
    client: Client, tmp_path: Path
) -> None:
    config = JobConfig(
        job_name="counted",
        jobs_dir=tmp_path / "jobs",
        tasks=[
            TaskConfig(path=tasks.passing(tmp_path / "tasks")),
            TaskConfig(path=tasks.no_reward(tmp_path / "tasks")),
        ],
        n_attempts=2,
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    hooked = _with(client, RetryFirstAttempt())
    async with new_worker(hooked, RetryJob) as worker:
        result = await hooked.execute_workflow(
            RetryJob.run, config, id=f"job-{uuid.uuid4()}", task_queue=worker.task_queue
        )

    assert result.stats.n_completed_trials == 4
    assert result.stats.n_retries == 4


class AlwaysRetry(TrialHooks):
    def __init__(self, error: Callable[[], ApplicationError]) -> None:
        self.error = error

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        raise self.error()


@on_posix
@pytest.mark.parametrize(
    ("error", "args", "attempt"),
    [
        (lambda: ApplicationError("again"), {"maximum_attempts": 2}, 2),
        (
            lambda: ApplicationError("never again", non_retryable=True),
            {"maximum_attempts": 3},
            1,
        ),
        (
            lambda: ApplicationError("later", next_retry_delay=timedelta(minutes=10)),
            {"schedule_to_close_timeout": timedelta(minutes=1)},
            1,
        ),
    ],
    ids=["attempts-exhausted", "non-retryable", "schedule-to-close"],
)
async def test_last_outcome_returned_when_temporal_stops_retrying(
    client: Client,
    tmp_path: Path,
    error: Callable[[], ApplicationError],
    args: dict[str, object],
    attempt: int,
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    outcome, attempts = await run_one(
        _with(client, AlwaysRetry(error)),
        TrialArgs.model_validate({"config": config, **args}),
    )

    assert outcome.attempt == attempt
    assert attempts == [attempt]
    assert outcome.result.verifier_result is not None
    assert outcome.result.verifier_result.rewards == {"reward": 1.0}


class BrokenOutput(TrialHooks):
    """Fails to read the result, as a hook with a bug does."""

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        raise KeyError("score")


@on_posix
async def test_output_that_raises_otherwise_is_not_retried(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    hooked = _with(client, BrokenOutput())
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=3),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()

    assert outcome.attempt == 1
    assert outcome.result.verifier_result is not None
    assert outcome.result.verifier_result.rewards == {"reward": 1.0}
    assert await history.final_attempts(handle) == [1]
    [failure] = await history.activity_failures(handle)
    assert failure.application_failure_info.type == "HarborOutputFailed"
    assert failure.application_failure_info.non_retryable
    assert failure.cause.application_failure_info.type == "KeyError"


class _Opaque:
    """A value with no JSON form."""


class UnsendableOutput(TrialHooks):
    """Returns a value Temporal cannot carry, as a hook with a bug does."""

    def __init__(self, value: object) -> None:
        self.value = value

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        return cast(JsonValue, self.value)


@on_posix
@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"score": _Opaque()}, id="not-json"),
        pytest.param("x" * 3_000_000, id="too-large"),
    ],
)
async def test_output_that_cannot_be_sent_is_not_retried(
    client: Client, tmp_path: Path, value: object
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    hooked = _with(client, UnsendableOutput(value))
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=3),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()

    assert outcome.attempt == 1
    assert outcome.output is None
    assert outcome.result.verifier_result is not None
    assert outcome.result.verifier_result.rewards == {"reward": 1.0}
    assert await history.final_attempts(handle) == [1]
    [failure] = await history.activity_failures(handle)
    assert failure.application_failure_info.type == "HarborOutputFailed"
    assert failure.application_failure_info.non_retryable


class RejectWithUnsendableDetails(TrialHooks):
    """Rejects the result with details Temporal cannot carry."""

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        raise ApplicationError(
            "rejected", _Opaque(), type="Rejected", non_retryable=True
        )


@on_posix
async def test_output_error_that_cannot_be_converted_keeps_the_outcome(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    hooked = _with(client, RejectWithUnsendableDetails())
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=3),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()

    assert outcome.attempt == 1
    assert outcome.output is None
    assert outcome.result.verifier_result is not None
    assert outcome.result.verifier_result.rewards == {"reward": 1.0}
    assert await history.final_attempts(handle) == [1]
    [failure] = await history.activity_failures(handle)
    assert failure.application_failure_info.type == "HarborFailureNotConverted"
    assert failure.application_failure_info.non_retryable


class FailAfterLargeOutput(TrialHooks):
    """Returns an output near the limit, then fails the attempt with a long error."""

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        return "x" * (_OUTCOME_MAX - 32 * 1024)

    @contextlib.asynccontextmanager
    async def scope(self, context: TrialContext) -> AsyncIterator[None]:
        yield
        raise ApplicationError(
            "e" * (256 * 1024), type="UploadFailed", non_retryable=True
        )


@on_posix
async def test_outcome_without_output_when_the_error_cannot_carry_both(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    hooked = _with(client, FailAfterLargeOutput())
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=3),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()

    assert outcome.attempt == 1
    assert outcome.output is None
    assert outcome.result.verifier_result is not None
    assert outcome.result.verifier_result.rewards == {"reward": 1.0}
    assert await history.final_attempts(handle) == [1]
    [failure] = await history.activity_failures(handle)
    info = failure.application_failure_info
    assert info.type == "UploadFailed"
    [carried] = hooked.data_converter.payload_converter.from_payloads(
        list(info.details.payloads)
    )
    attached = TrialOutcome.model_validate(carried)
    assert attached.output is None
    assert attached.result.id == outcome.result.id


class RejectWithItsOwnError(TrialHooks):
    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        raise ApplicationError(
            "the verifier's answer was rejected",
            type="AnswerRejected",
            non_retryable=True,
            next_retry_delay=timedelta(minutes=3),
        )


@on_posix
async def test_output_error_reaches_temporal_as_raised(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    hooked = _with(client, RejectWithItsOwnError())
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=3),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()

    assert outcome.attempt == 1
    [failure] = await history.activity_failures(handle)
    info = failure.application_failure_info
    assert failure.message == "the verifier's answer was rejected"
    assert info.type == "AnswerRejected"
    assert info.non_retryable
    assert info.next_retry_delay.ToTimedelta() == timedelta(minutes=3)


class RetryThenLoseCredentials(TrialHooks):
    """Send the first attempt back to Temporal; every later one cannot start."""

    @contextlib.asynccontextmanager
    async def scope(self, context: TrialContext) -> AsyncIterator[None]:
        if context.attempt > 1:
            raise RuntimeError("the worker lost its credentials")
        yield

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        raise ApplicationError("try once more", type="TryAgain")


class HangAfterResult(TrialHooks):
    """The worker stops answering after harbor produced a result, as a lost worker does."""

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        await asyncio.sleep(600)
        return None


@on_posix
@pytest.mark.parametrize(
    ("hooks", "args", "attempts", "failure"),
    [
        (
            RetryThenLoseCredentials,
            {"maximum_attempts": 2},
            [2],
            "application_failure_info",
        ),
        (
            HangAfterResult,
            {
                "maximum_attempts": 1,
                "start_to_close_timeout": timedelta(seconds=40),
                "extra_time": timedelta(minutes=30),
            },
            [1],
            "timeout_failure_info",
        ),
    ],
    ids=["worker-raised", "server-timed-out"],
)
async def test_outcome_returned_when_the_last_attempt_had_none(
    client: Client,
    tmp_path: Path,
    hooks: type[TrialHooks],
    args: dict[str, object],
    attempts: list[int],
    failure: str,
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    hooked = _with(client, hooks())
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs.model_validate({"config": config, **args}),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()

    # The last attempt ended without a result: it raised before running the
    # trial, or the server timed it out. The first attempt's outcome comes back.
    assert outcome.attempt == 1
    assert outcome.result.verifier_result is not None
    assert outcome.result.verifier_result.rewards == {"reward": 1.0}
    assert await history.final_attempts(handle) == attempts
    [recorded] = await history.activity_failures(handle)
    assert recorded.WhichOneof("failure_info") == failure


class ShutDownOnRetry(TrialHooks):
    """Sends the first attempt back to Temporal, then holds the next one open."""

    def __init__(self) -> None:
        self.retrying = asyncio.Event()

    async def trial_created(self, trial: Trial, context: TrialContext) -> None:
        if context.attempt > 1:
            self.retrying.set()
            await asyncio.sleep(600)

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        raise ApplicationError("try once more", type="TryAgain")


@on_posix
async def test_outcome_returned_when_the_worker_shut_down_in_the_last_attempt(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    hooks = ShutDownOnRetry()
    hooked = _with(client, hooks)
    worker = Worker(
        hooked,
        task_queue=str(uuid.uuid4()),
        workflows=[RunOneTrial],
        graceful_shutdown_timeout=timedelta(0),
    )
    running = asyncio.create_task(worker.run())
    try:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=2),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        await asyncio.wait_for(hooks.retrying.wait(), timeout=60)
    finally:
        await worker.shutdown()
        await running
    # Another worker finishes the workflow once Temporal stops retrying.
    async with Worker(
        _with(client, TrialHooks()),
        task_queue=worker.task_queue,
        workflows=[RunOneTrial],
    ):
        outcome = await handle.result()

    assert outcome.attempt == 1
    assert outcome.result.verifier_result is not None
    assert outcome.result.verifier_result.rewards == {"reward": 1.0}
    assert await history.final_attempts(handle) == [2]
    [failure] = await history.activity_failures(handle)
    assert failure.application_failure_info.type == "WorkerShutdown"
    assert failure.application_failure_info.details.payloads


async def test_trial_that_could_not_run_raises(
    harbor_client: Client, tmp_path: Path
) -> None:
    # A task that does not exist fails before harbor can record a result.
    config = _trial(tmp_path / "missing", tmp_path / "trials", "missing__0")
    async with new_worker(harbor_client, RunOneTrial) as worker:
        handle = await harbor_client.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=2),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError) as failure:
            await handle.result()

    activity_error = failure.value.cause
    assert isinstance(activity_error, ActivityError)
    cause = activity_error.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "FileNotFoundError"
    assert await history.final_attempts(handle) == [2]


@on_posix
async def test_failed_environment_check_is_retried(
    harbor_client: Client, tmp_path: Path
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    config.environment = EnvironmentConfig(import_path=tasks.NO_CREDENTIALS_ENV)
    async with new_worker(harbor_client, RunOneTrial) as worker:
        handle = await harbor_client.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=2),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError) as failure:
            await handle.result()

    activity_error = failure.value.cause
    assert isinstance(activity_error, ActivityError)
    cause = activity_error.cause
    assert isinstance(cause, ApplicationError)
    assert "LOCAL_API_KEY is not set" in cause.message
    assert not cause.non_retryable
    assert await history.final_attempts(handle) == [2]


@on_posix
async def test_misconfigured_trial_is_not_retried(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HARBOR_TEST_UNSET_VAR", raising=False)
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    config.agent = AgentConfig(name="oracle", env={"X": "${HARBOR_TEST_UNSET_VAR}"})
    async with new_worker(harbor_client, RunOneTrial) as worker:
        handle = await harbor_client.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=2),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError) as failure:
            await handle.result()

    activity_error = failure.value.cause
    assert isinstance(activity_error, ActivityError)
    cause = activity_error.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "ValueError"
    assert cause.non_retryable
    assert "HARBOR_TEST_UNSET_VAR" in cause.message
    assert await history.final_attempts(handle) == [1]


@on_posix
async def test_trial_named_outside_its_directory_is_rejected(
    harbor_client: Client, tmp_path: Path
) -> None:
    kept = tmp_path / "trials" / "victim" / "keep.txt"
    kept.parent.mkdir(parents=True)
    kept.write_text("still here")
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials" / "job", "../victim"
    )
    async with new_worker(harbor_client, RunOneTrial) as worker:
        handle = await harbor_client.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=2),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError):
            await handle.result()

    [failure] = await history.activity_failures(handle)
    assert failure.application_failure_info.type == "HarborInvalidTrialName"
    assert failure.application_failure_info.non_retryable
    assert await history.final_attempts(handle) == [1]
    assert kept.read_text() == "still here"


@on_posix
async def test_default_trial_retry_policy(
    harbor_client: Client, tmp_path: Path
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    async with new_worker(harbor_client, RunOneTrial) as worker:
        handle = await harbor_client.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        await handle.result()

    [event] = await history.scheduled(handle)
    policy = event.activity_task_scheduled_event_attributes.retry_policy
    assert policy.maximum_attempts == 3
    assert policy.initial_interval.ToTimedelta() == timedelta(seconds=15)
    assert policy.backoff_coefficient == 2.0
    assert policy.maximum_interval.ToTimedelta() == timedelta(minutes=2)
