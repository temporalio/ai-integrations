"""Values that cross the Activity boundary, and the helpers both sides share.

Imports none of harbor's trial or environment code, so workflow code can import
this module.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID

from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from pydantic import BaseModel, JsonValue

from temporalio.exceptions import ApplicationError

PLAN_JOB = "harbor.plan_job"
RUN_TRIAL = "harbor.run_trial"
AGGREGATE_JOB = "harbor.aggregate_job"

Rewards = dict[str, float | int]


class PlanJobInput(BaseModel):
    """Input to the planning Activity."""

    config: JobConfig
    untimed_agent_timeout: timedelta


class PlanJobResult(BaseModel):
    """Output of the planning Activity.

    ``task_names`` and ``timeouts`` follow ``task_configs``; each task's
    ``timeouts`` follow the job's agents. Each git task in ``task_configs`` is
    pinned to the commit it was resolved to, and each package task given no
    ``ref`` to the content hash it was resolved to. ``datasets`` are the job's
    datasets with each package dataset's ``ref`` pinned to the content hash
    its tasks were resolved from.
    """

    task_configs: list[TaskConfig]
    task_names: list[str]
    timeouts: list[list[timedelta]]
    datasets: list[DatasetConfig]


class TrialPlan(BaseModel):
    """One trial of a planned job, as :func:`temporalio.harbor.execute_trial` takes it.

    ``timeout`` is the sum of the phase timeouts harbor enforces for this
    trial: environment build, agent setup, agent, verifier and the rest, as
    the trial's task and configuration set them, with an agent that has no
    timeout counted as :func:`temporalio.harbor.plan_job`'s
    ``untimed_agent_timeout``. :func:`temporalio.harbor.execute_trial` derives
    the Activity's default ``start_to_close_timeout`` from it.
    """

    config: TrialConfig
    timeout: timedelta


def check_trial_name(name: str) -> None:
    """Fail without a retry unless ``name`` is a single path component.

    A trial's name is its directory under ``trials_dir``, which each attempt
    deletes before it runs. ``:`` is rejected because on Windows it names a
    drive.
    """
    if name in ("", ".", "..") or any(c in name for c in "/\\:"):
        raise ApplicationError(
            f"trial name {name!r} is not a single path component",
            type="HarborInvalidTrialName",
            non_retryable=True,
        )


class TemporalJobPlan(BaseModel):
    """A harbor job resolved into the trials it will run.

    Produced by :func:`temporalio.harbor.plan_job`. ``trials`` is what the
    workflow fans out; the rest is what :func:`temporalio.harbor.aggregate_job`
    needs. ``id`` is the job's id, as harbor records it in each trial's
    ``job_id`` and in the job's result. Each git task is pinned to the commit
    it was resolved to, and each package task given no ``ref`` to the content
    hash it was resolved to, as the job's ``lock.json`` records them. In
    ``config``, each package dataset's ``ref`` is pinned to the content hash
    its tasks were resolved from, and its ``version`` is cleared.
    """

    id: UUID
    config: JobConfig
    task_configs: list[TaskConfig]
    trials: list[TrialPlan]


class RunTrialInput(BaseModel):
    """Input to the trial Activity."""

    config: TrialConfig
    extra_time: timedelta
    data: JsonValue = None


class TrialOutcome(BaseModel):
    """What one trial produced, as :func:`temporalio.harbor.execute_trial` returns it.

    ``result`` is harbor's result for the trial, passed through
    :meth:`temporalio.harbor.TrialHooks.redact`, less rollout details and
    agent metadata, which harbor's aggregation does not read, and with long
    exception messages and tracebacks shortened. It is read from the
    ``result.json`` harbor writes to the trial directory on the worker that
    ran the attempt, after harbor removed the secrets it knows of from that
    directory; the file keeps what the outcome leaves out, and a later
    attempt of the trial that uses the same directory starts by deleting it.
    ``attempt`` is the Activity attempt that produced the outcome, counting
    from 1. ``output`` is what the worker's
    :meth:`temporalio.harbor.TrialHooks.output` returned.
    """

    result: TrialResult
    attempt: int
    output: JsonValue = None


class AggregateJobInput(BaseModel):
    """Input to the aggregation Activity.

    ``rewards`` holds each eval's rewards, one entry per trial. ``result`` is
    the job's result so far, without metrics or trial results.
    """

    config: JobConfig
    task_configs: list[TaskConfig]
    rewards: dict[str, list[Rewards | None]]
    result: JobResult
