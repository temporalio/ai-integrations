"""Values that cross the Activity boundary.

Kept free of harbor's trial and environment machinery so that workflow code can
import them: everything here is a harbor pydantic model or a container of them.
"""

from __future__ import annotations

from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.trial.config import TaskConfig, TrialConfig
from pydantic import BaseModel

RESOLVE_JOB = "harbor.resolve_job"
RUN_TRIAL = "harbor.run_trial"
COMPUTE_METRICS = "harbor.compute_metrics"

Rewards = dict[str, float | int]


class JobPlan(BaseModel):
    """A harbor job resolved into the trials it will run.

    Produced by :func:`temporalio.harbor.plan_job`. ``trials`` is what the
    workflow fans out; ``config`` and ``task_configs`` are what
    :func:`temporalio.harbor.aggregate_job` needs to reproduce harbor's metrics.
    """

    config: JobConfig
    task_configs: list[TaskConfig]
    trials: list[TrialConfig]


class RunTrialInput(BaseModel):
    """Input to the trial Activity."""

    config: TrialConfig
    retry: RetryConfig


class ComputeMetricsInput(BaseModel):
    """Input to the metrics Activity: each eval's rewards, one entry per trial."""

    config: JobConfig
    task_configs: list[TaskConfig]
    rewards: dict[str, list[Rewards | None]]
