"""Values that cross the Activity boundary.

Kept free of harbor's trial and environment machinery so that workflow code can
import them: everything here is a harbor pydantic model or a container of them.
"""

from __future__ import annotations

from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from pydantic import BaseModel, JsonValue

RESOLVE_JOB = "harbor.resolve_job"
RUN_TRIAL = "harbor.run_trial"
COMPUTE_METRICS = "harbor.compute_metrics"

Rewards = dict[str, float | int]


class ResolveJobResult(BaseModel):
    """Output of the resolve Activity: the tasks, and each dataset's ref.

    Only a package dataset's ref is pinned, to the content hash it resolved to.
    """

    task_configs: list[TaskConfig]
    dataset_refs: list[str | None]


class JobPlan(BaseModel):
    """A harbor job resolved into the trials it will run.

    Produced by :func:`temporalio.harbor.plan_job`. ``trials`` is what the
    workflow fans out; ``config`` and ``task_configs`` are what
    :func:`temporalio.harbor.aggregate_job` needs to reproduce harbor's metrics.
    In ``config``, each package dataset's ``ref`` is pinned to the content
    hash its tasks were resolved from, and its ``version`` is cleared.
    """

    config: JobConfig
    task_configs: list[TaskConfig]
    trials: list[TrialConfig]


class RunTrialInput(BaseModel):
    """Input to the trial Activity."""

    config: TrialConfig
    retry: RetryConfig
    data: JsonValue = None


class TrialOutcome(BaseModel):
    """What one trial produced, as :func:`temporalio.harbor.execute_trial` returns it.

    ``result`` is harbor's result for the trial, less rollout details and agent
    metadata, which harbor's aggregation does not read; the complete result
    stays in the trial directory. ``output`` is what the worker's
    :meth:`temporalio.harbor.TrialHooks.output` returned, ``None`` by default.
    """

    result: TrialResult
    output: JsonValue = None


class ComputeMetricsInput(BaseModel):
    """Input to the metrics Activity: each eval's rewards, one entry per trial."""

    config: JobConfig
    task_configs: list[TaskConfig]
    rewards: dict[str, list[Rewards | None]]
