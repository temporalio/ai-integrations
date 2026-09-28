"""Trial failures are retried the way harbor retries them; lost work by Temporal."""

from __future__ import annotations

import itertools
import sys
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.models.trial.result import TrialResult
from harbor.trial.queue import TrialQueue

from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.harbor import retry_policy_from_harbor
from temporalio.harbor._activity import should_retry
from tests.harbor_fixtures import history, tasks
from tests.harbor_fixtures.workflows import RunOneTrial, TrialArgs, new_worker

on_posix = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)


def _fast(
    max_retries: int = 0, exclude_exceptions: set[str] | None = None
) -> RetryConfig:
    # Retry quickly: these tests are about which attempts happen, not their spacing.
    return RetryConfig(
        max_retries=max_retries,
        exclude_exceptions=exclude_exceptions,
        min_wait_sec=0.1,
        max_wait_sec=0.1,
    )


def _trial(task: Path, trials_dir: Path, name: str) -> TrialConfig:
    return TrialConfig(
        task=TaskConfig(path=task),
        trial_name=name,
        trials_dir=trials_dir,
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )


async def _run(client: Client, args: TrialArgs) -> tuple[TrialResult, list[int]]:
    async with new_worker(client, RunOneTrial) as worker:
        handle = await client.start_workflow(
            RunOneTrial.run,
            args,
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    return result, await history.final_attempts(handle)


@on_posix
async def test_excluded_error_is_recorded_not_retried(
    harbor_client: Client, tmp_path: Path
) -> None:
    # Harbor's defaults exclude RewardFileNotFoundError, and allow no retries.
    config = _trial(
        tasks.no_reward(tmp_path / "tasks"), tmp_path / "trials", "no-reward__0"
    )
    result, attempts = await _run(harbor_client, TrialArgs(config=config))

    assert result.exception_info is not None
    assert result.exception_info.exception_type == "RewardFileNotFoundError"
    assert result.verifier_result is None
    assert attempts == [1]


@on_posix
async def test_retryable_error_reruns_slot_in_place(
    harbor_client: Client, tmp_path: Path
) -> None:
    trials_dir = tmp_path / "trials"
    config = _trial(tasks.flaky(tmp_path / "tasks", failures=1), trials_dir, "flaky__0")
    retry = _fast(max_retries=2, exclude_exceptions=set())
    result, attempts = await _run(harbor_client, TrialArgs(config=config, retry=retry))

    assert result.exception_info is None
    assert result.verifier_result is not None
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert attempts == [2]
    # The retry ran in the same slot, with the failed attempt kept beside it.
    assert (trials_dir / "flaky__0" / "result.json").is_file()
    assert (trials_dir / "flaky__0.attempt-1" / "result.json").is_file()


@on_posix
async def test_exhausted_trial_retries_return_errored_result(
    harbor_client: Client, tmp_path: Path
) -> None:
    config = _trial(
        tasks.no_reward(tmp_path / "tasks"), tmp_path / "trials", "no-reward__0"
    )
    retry = _fast(max_retries=1, exclude_exceptions=set())
    result, attempts = await _run(harbor_client, TrialArgs(config=config, retry=retry))

    # Out of retries, the trial is a recorded failure, as harbor reports it,
    # not a failed workflow.
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "RewardFileNotFoundError"
    assert attempts == [2]


async def test_infra_failure_fails_workflow_after_retries(
    harbor_client: Client, tmp_path: Path
) -> None:
    # A task that does not exist fails before harbor can record a result.
    config = _trial(tmp_path / "missing", tmp_path / "trials", "missing__0")
    args = TrialArgs(config=config, retry=_fast(), infrastructure_retries=1)
    async with new_worker(harbor_client, RunOneTrial) as worker:
        handle = await harbor_client.start_workflow(
            RunOneTrial.run,
            args,
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


def test_retry_policy_mapping() -> None:
    policy = retry_policy_from_harbor(
        RetryConfig(max_retries=2, min_wait_sec=2, wait_multiplier=3, max_wait_sec=30),
        infrastructure_retries=4,
    )
    assert policy.initial_interval == timedelta(seconds=2)
    assert policy.backoff_coefficient == 3
    assert policy.maximum_interval == timedelta(seconds=30)
    assert policy.maximum_attempts == 2 + 1 + 4

    # Harbor clamps even the first delay to max_wait_sec.
    clamped = retry_policy_from_harbor(RetryConfig(min_wait_sec=90, max_wait_sec=60))
    assert clamped.initial_interval == timedelta(seconds=60)

    with pytest.raises(ValueError):
        retry_policy_from_harbor(RetryConfig(wait_multiplier=0.5))
    with pytest.raises(ValueError):
        retry_policy_from_harbor(RetryConfig(), infrastructure_retries=-1)


@pytest.mark.parametrize(
    ("include", "exclude"),
    list(itertools.product([None, set(), {"A"}, {"A", "B"}], [None, set(), {"B"}])),
)
def test_retry_decision_matches_harbor(
    include: set[str] | None, exclude: set[str] | None
) -> None:
    config = RetryConfig(
        max_retries=1, include_exceptions=include, exclude_exceptions=exclude
    )
    queue = TrialQueue(n_concurrent=1, retry_config=config)
    for exception_type in ("A", "B", "C"):
        expected = queue._should_retry_exception(exception_type)
        assert should_retry(config, exception_type) == expected, exception_type
