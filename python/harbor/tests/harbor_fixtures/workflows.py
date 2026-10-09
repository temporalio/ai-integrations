"""Workflows the suite runs, written the way a user of the plugin would."""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from typing import Any

from pydantic import BaseModel, JsonValue

from temporalio import workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker
from tests.harbor_fixtures import history

with workflow.unsafe.imports_passed_through():
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.job.result import JobStats
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig

    from temporalio.harbor import (
        JobPlan,
        TrialOutcome,
        aggregate_job,
        execute_trial,
        plan_job,
    )

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


def _descriptive_name(task: TaskConfig, agent: AgentConfig, attempt: int) -> str:
    return f"{task.get_task_id().get_name()}__{agent.name}__{attempt}"


@workflow.defn
class PlanNamedJob:
    """Plan the job with trial names derived from what each trial is."""

    @workflow.run
    async def run(self, config: JobConfig) -> JobPlan:
        return await plan_job(config, trial_name=_descriptive_name)


@workflow.defn
class PlanCollidingJob:
    """Plan the job with a naming function that gives every trial one name."""

    @workflow.run
    async def run(self, config: JobConfig) -> JobPlan:
        return await plan_job(config, trial_name=lambda *_: "same")


class TrialArgs(BaseModel):
    """How :class:`RunOneTrial` should run its trial."""

    config: TrialConfig
    retry: RetryConfig | None = None
    infrastructure_retries: int = 3
    maximum_attempts: int | None = None
    data: JsonValue = None
    summary: str | None = None


@workflow.defn
class RunOneTrial:
    """Run a single trial with explicit options."""

    @workflow.run
    async def run(self, args: TrialArgs) -> TrialOutcome:
        policy = None
        if args.maximum_attempts is not None:
            policy = RetryPolicy(
                initial_interval=timedelta(milliseconds=100),
                maximum_attempts=args.maximum_attempts,
            )
        return await execute_trial(
            args.config,
            retry=args.retry,
            start_to_close_timeout=TRIAL_TIMEOUT,
            infrastructure_retries=args.infrastructure_retries,
            retry_policy=policy,
            data=args.data,
            summary=args.summary,
        )


def new_worker(client: Client, *workflows: type, **kwargs: Any) -> Worker:
    """A worker on a fresh task queue; the client's plugins supply the activities."""
    return Worker(client, task_queue=str(uuid.uuid4()), workflows=workflows, **kwargs)


async def run_one(client: Client, args: TrialArgs) -> tuple[TrialOutcome, list[int]]:
    """Run :class:`RunOneTrial`; return its outcome and each activity's final attempt."""
    async with new_worker(client, RunOneTrial) as worker:
        handle = await client.start_workflow(
            RunOneTrial.run,
            args,
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()
    return outcome, await history.final_attempts(handle)
