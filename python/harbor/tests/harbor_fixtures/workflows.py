"""Workflows the suite runs, written the way a user of the plugin would."""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from typing import Any

from pydantic import BaseModel

from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.job.result import JobStats
    from harbor.models.trial.config import TrialConfig
    from harbor.models.trial.result import TrialResult

    from temporalio.harbor import JobPlan, aggregate_job, execute_trial, plan_job

TRIAL_TIMEOUT = timedelta(minutes=2)


@workflow.defn
class RunJob:
    """Plan a harbor job, run every trial, aggregate."""

    @workflow.run
    async def run(self, config: JobConfig) -> JobStats:
        plan = await plan_job(config)
        results = await asyncio.gather(
            *(
                execute_trial(
                    trial, retry=config.retry, start_to_close_timeout=TRIAL_TIMEOUT
                )
                for trial in plan.trials
            )
        )
        return await aggregate_job(plan, results)


@workflow.defn
class PlanJob:
    """Only plan the job."""

    @workflow.run
    async def run(self, config: JobConfig) -> JobPlan:
        return await plan_job(config)


@workflow.defn
class TrialNames:
    """Run a job's trials and report the names it planned them under."""

    @workflow.run
    async def run(self, config: JobConfig) -> list[str]:
        plan = await plan_job(config)
        await asyncio.gather(
            *(
                execute_trial(t, start_to_close_timeout=TRIAL_TIMEOUT)
                for t in plan.trials
            )
        )
        return [t.trial_name for t in plan.trials]


class TrialArgs(BaseModel):
    """How :class:`RunOneTrial` should run its trial."""

    config: TrialConfig
    retry: RetryConfig | None = None
    infrastructure_retries: int = 3
    summary: str | None = None


@workflow.defn
class RunOneTrial:
    """Run a single trial with explicit options."""

    @workflow.run
    async def run(self, args: TrialArgs) -> TrialResult:
        return await execute_trial(
            args.config,
            retry=args.retry,
            start_to_close_timeout=TRIAL_TIMEOUT,
            infrastructure_retries=args.infrastructure_retries,
            summary=args.summary,
        )


def new_worker(client: Client, *workflows: type, **kwargs: Any) -> Worker:
    """A worker on a fresh task queue; the client's plugins supply the activities."""
    return Worker(client, task_queue=str(uuid.uuid4()), workflows=workflows, **kwargs)
