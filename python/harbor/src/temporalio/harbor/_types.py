"""Values that cross the Activity boundary, and the helpers both sides share.

Imports none of harbor's trial or environment code, so workflow code can import
this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Any
from uuid import UUID

from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from pydantic import BaseModel, JsonValue, ValidationError

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
    ``timeouts`` follow the job's agents. ``datasets`` are the job's datasets
    with each package dataset's ``ref`` pinned to the content hash its tasks
    were resolved from.
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
    deletes before it runs.
    """
    if name in ("", ".", "..") or "/" in name or "\\" in name:
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
    ``job_id`` and in the job's result. In ``config``, each package dataset's
    ``ref`` is pinned to the content hash its tasks were resolved from, and
    its ``version`` is cleared.
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
    exception messages and tracebacks shortened. Harbor writes the complete
    result to the trial directory on the worker that ran the attempt, and a
    later attempt of the trial that uses the same directory starts by
    deleting it. ``attempt`` is the Activity attempt that produced the
    outcome, counting from 1, which can be earlier than the last attempt
    Temporal ran. ``output`` is what the worker's
    :meth:`temporalio.harbor.TrialHooks.output` returned, or ``None`` when
    the attempt could not carry it, for example because the hook raised.
    """

    result: TrialResult
    attempt: int
    output: JsonValue = None


def as_outcome(value: Any) -> TrialOutcome | None:
    """``value`` as a :class:`TrialOutcome`, or ``None`` if it is not one."""
    try:
        return TrialOutcome.model_validate(value)
    except ValidationError:
        return None


def heartbeat_outcome(details: Sequence[Any]) -> TrialOutcome | None:
    """The outcome a trial Activity's last heartbeat carried, if any."""
    last = details[-1] if details else None
    return as_outcome(last.get("outcome")) if isinstance(last, dict) else None


class AggregateJobInput(BaseModel):
    """Input to the aggregation Activity.

    ``rewards`` holds each eval's rewards, one entry per trial. ``result`` is
    the job's result so far, without metrics or trial results.
    """

    config: JobConfig
    task_configs: list[TaskConfig]
    rewards: dict[str, list[Rewards | None]]
    result: JobResult
