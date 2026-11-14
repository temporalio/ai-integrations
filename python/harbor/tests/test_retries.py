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
from harbor.models.task.config import MCPServerConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.result import TrialResult
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


@on_posix
@pytest.mark.parametrize("make", [tasks.passing, tasks.two_steps])
async def test_reward_that_is_not_a_number_is_a_verifier_failure(
    harbor_client: Client, tmp_path: Path, make: Callable[[Path], Path]
) -> None:
    config = _trial(make(tmp_path / "tasks"), tmp_path / "trials", "trial__0")
    config.verifier = VerifierConfig(import_path=agents.NAN_VERIFIER)
    async with new_worker(harbor_client, RunOneTrial) as worker:
        handle = await harbor_client.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=2),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        outcome = await handle.result()

    result = outcome.result
    for recorded in [result, *(result.step_results or [])]:
        assert recorded.verifier_result is None
        assert recorded.exception_info is not None
        assert recorded.exception_info.exception_type == "VerifierOutputParseError"
    assert outcome.attempt == 1
    assert await history.activity_failures(handle) == []


def _failing(error: str, **kwargs: str) -> AgentConfig:
    return AgentConfig(
        import_path=agents.FAILING_AGENT, kwargs={"error": error, **kwargs}
    )


async def _with_two_attempts(
    client: Client, config: TrialConfig
) -> tuple[TrialOutcome, list[int], list[Failure], list[Failure]]:
    """Run the trial allowing two attempts.

    Returns its outcome, its final attempt, the failure history recorded for
    it, and the failure of the attempt Temporal retried, if it did.
    """
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
        await history.retried_failures(handle),
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
        # Harbor's policy retries it, but cannot say why the agent exited.
        pytest.param(
            tasks.passing,
            _failing("NonZeroAgentExitCodeError"),
            tasks.LOCAL_ENV,
            "NonZeroAgentExitCodeError",
            id="unclassified",
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
    outcome, attempts, failures, _ = await _with_two_attempts(harbor_client, config)

    assert outcome.result.exception_info is not None
    assert outcome.result.exception_info.exception_type == error
    assert outcome.attempt == 1
    assert attempts == [1]
    assert failures == []


def _recorded_types(outcome: TrialOutcome) -> set[str]:
    # A multi-step trial records each step's exception on that step.
    result = outcome.result
    recorded = [result, *(result.step_results or [])]
    return {r.exception_info.exception_type for r in recorded if r.exception_info}


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
    outcome, attempts, failures, retried = await _with_two_attempts(
        harbor_client, config
    )

    # The last attempt Temporal allows returns its result, as harbor does.
    assert outcome.attempt == 2
    assert _recorded_types(outcome) == {error}
    assert attempts == [2]
    assert failures == []
    [failure] = retried
    assert failure.application_failure_info.type == error
    assert not failure.application_failure_info.non_retryable


@on_posix
@pytest.mark.parametrize(
    ("make", "error"),
    [
        pytest.param(tasks.passing, "ApiOverloadedError", id="ApiOverloadedError"),
        pytest.param(tasks.two_steps, "ApiOverloadedError", id="steps"),
    ],
)
async def test_error_harbor_retries_is_retried(
    harbor_client: Client, tmp_path: Path, make: Callable[[Path], Path], error: str
) -> None:
    config = _trial(make(tmp_path / "tasks"), tmp_path / "trials", "trial__0")
    config.agent = _failing(error)
    outcome, attempts, failures, retried = await _with_two_attempts(
        harbor_client, config
    )

    assert outcome.attempt == 2
    assert _recorded_types(outcome) == {error}
    assert attempts == [2]
    assert failures == []
    [failure] = retried
    info = failure.application_failure_info
    assert info.type == error
    assert not info.non_retryable
    assert not info.HasField("next_retry_delay")
    assert not failure.application_failure_info.details.payloads


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
@pytest.mark.parametrize("error", ["ApiRateLimitError", "RateLimitError"])
async def test_rate_limited_trial_is_retried_after_two_minutes(
    harbor_client: Client, tmp_path: Path, error: str
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    config.agent = _failing(error)
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
    assert failure.application_failure_info.type == error
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


async def _trial_failure(
    client: Client, args: TrialArgs
) -> tuple[ActivityError, WorkflowHandle[Any, Any]]:
    """Run :class:`RunOneTrial`, whose trial fails; return its error and handle."""
    async with new_worker(client, RunOneTrial) as worker:
        handle = await client.start_workflow(
            RunOneTrial.run,
            args,
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError) as failure:
            await handle.result()
    assert isinstance(failure.value.cause, ActivityError)
    return failure.value.cause, handle


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
async def test_output_error_fails_the_trial_when_temporal_stops_retrying(
    client: Client,
    tmp_path: Path,
    error: Callable[[], ApplicationError],
    args: dict[str, object],
    attempt: int,
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    raised, handle = await _trial_failure(
        _with(client, AlwaysRetry(error)),
        TrialArgs.model_validate({"config": config, **args}),
    )

    assert isinstance(raised.cause, ApplicationError)
    assert raised.cause.message == error().message
    assert await history.final_attempts(handle) == [attempt]


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
    _, handle = await _trial_failure(
        _with(client, BrokenOutput()), TrialArgs(config=config, maximum_attempts=3)
    )

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
    _, handle = await _trial_failure(
        _with(client, UnsendableOutput(value)),
        TrialArgs(config=config, maximum_attempts=3),
    )

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
async def test_output_error_that_cannot_be_converted_is_not_retried(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    _, handle = await _trial_failure(
        _with(client, RejectWithUnsendableDetails()),
        TrialArgs(config=config, maximum_attempts=3),
    )

    assert await history.final_attempts(handle) == [1]
    [failure] = await history.activity_failures(handle)
    assert failure.application_failure_info.type == "HarborFailureNotConverted"
    assert failure.application_failure_info.non_retryable


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
    _, handle = await _trial_failure(
        _with(client, RejectWithItsOwnError()),
        TrialArgs(config=config, maximum_attempts=3),
    )

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


class RetryThenHang(TrialHooks):
    """Send the first attempt back to Temporal; the worker then stops answering."""

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        if context.attempt == 1:
            raise ApplicationError("try once more", type="TryAgain")
        await asyncio.sleep(600)
        return None


@on_posix
@pytest.mark.parametrize(
    ("hooks", "args", "failure"),
    [
        (RetryThenLoseCredentials, {}, "application_failure_info"),
        (
            RetryThenHang,
            {
                "start_to_close_timeout": timedelta(seconds=15),
                "extra_time": timedelta(minutes=30),
            },
            "timeout_failure_info",
        ),
    ],
    ids=["worker-raised", "lost"],
)
async def test_last_attempt_without_a_result_fails_the_trial(
    client: Client,
    tmp_path: Path,
    hooks: type[TrialHooks],
    args: dict[str, object],
    failure: str,
) -> None:
    # The first attempt's result does not come back in its place.
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    _, handle = await _trial_failure(
        _with(client, hooks()),
        TrialArgs.model_validate({"config": config, "maximum_attempts": 2, **args}),
    )

    assert await history.final_attempts(handle) == [2]
    [recorded] = await history.activity_failures(handle)
    assert recorded.WhichOneof("failure_info") == failure


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


def _with_skills(tmp_path: Path) -> AgentConfig:
    skill = tmp_path / "skills" / "greeting"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: greeting\ndescription: Hi.\n---\n")
    return AgentConfig(name="oracle", skills=[str(skill)])


def _with_mcp_server(_: Path) -> AgentConfig:
    server = MCPServerConfig(name="docs", url="http://localhost:9/sse")
    return AgentConfig(name="oracle", mcp_servers=[server])


@on_posix
@pytest.mark.parametrize(
    "agent",
    [
        pytest.param(_with_skills, id="skills"),
        pytest.param(_with_mcp_server, id="mcp-servers"),
    ],
)
async def test_unsupported_agent_capability_is_not_retried(
    harbor_client: Client, tmp_path: Path, agent: Callable[[Path], AgentConfig]
) -> None:
    # Harbor checks what the agent supports as the trial starts, before its
    # environment, and records the error in the trial's result.
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", "answers__0"
    )
    config.agent = agent(tmp_path)
    outcome, attempts, failures, retried = await _with_two_attempts(
        harbor_client, config
    )

    recorded = outcome.result.exception_info
    assert recorded is not None
    assert recorded.exception_type == "ValueError"
    assert "does not support" in recorded.exception_message
    assert outcome.result.environment_setup is None
    assert outcome.attempt == 1
    assert attempts == [1]
    assert failures == []
    assert retried == []


@on_posix
async def test_trial_named_outside_its_directory_is_rejected(
    harbor_client: Client, tmp_path: Path
) -> None:
    kept = tmp_path / "trials" / "victim" / "keep.txt"
    kept.parent.mkdir(parents=True)
    kept.write_text("still here")
    # Deleting job/../victim reaches the victim only once job exists.
    (tmp_path / "trials" / "job").mkdir()
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
