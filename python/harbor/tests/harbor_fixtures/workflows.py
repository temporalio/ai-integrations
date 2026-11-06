"""The workflows the suite runs."""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, JsonValue

from temporalio import workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker
from tests.harbor_fixtures import history

with workflow.unsafe.imports_passed_through():
    from harbor.models.job.config import JobConfig
    from harbor.models.job.result import JobResult
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig

    from temporalio.harbor import (
        TemporalJobPlan,
        TrialOutcome,
        TrialPlan,
        aggregate_job,
        execute_trial,
        plan_job,
    )

TRIAL_TIMEOUT = timedelta(minutes=2)


@workflow.defn
class RunJob:
    """Plan a harbor job, run every trial, aggregate what came back."""

    @workflow.run
    async def run(self, config: JobConfig) -> JobResult:
        plan = await plan_job(config)
        outcomes = await asyncio.gather(
            *(execute_trial(trial) for trial in plan.trials), return_exceptions=True
        )
        return await aggregate_job(
            plan, [o for o in outcomes if isinstance(o, TrialOutcome)]
        )


@workflow.defn
class RetryJob:
    """Run a job whose trials Temporal may run twice, quickly."""

    @workflow.run
    async def run(self, config: JobConfig) -> JobResult:
        plan = await plan_job(config)
        policy = RetryPolicy(
            initial_interval=timedelta(milliseconds=100), maximum_attempts=2
        )
        outcomes = await asyncio.gather(
            *(execute_trial(trial, retry_policy=policy) for trial in plan.trials)
        )
        return await aggregate_job(plan, outcomes)


class QueuedJob(BaseModel):
    """A job, and the task queue its Activities run on."""

    config: JobConfig
    task_queue: str


@workflow.defn
class RunJobOnQueue:
    """Run a job with every Activity on a given task queue."""

    @workflow.run
    async def run(self, job: QueuedJob) -> JobResult:
        plan = await plan_job(job.config, task_queue=job.task_queue)
        outcomes = await asyncio.gather(
            *(execute_trial(t, task_queue=job.task_queue) for t in plan.trials)
        )
        return await aggregate_job(plan, outcomes, task_queue=job.task_queue)


@workflow.defn
class PlanJob:
    """Only plan the job."""

    @workflow.run
    async def run(self, config: JobConfig) -> TemporalJobPlan:
        return await plan_job(config)


@workflow.defn
class PlanAndAggregate:
    """Plan the job and aggregate it without running any trial."""

    @workflow.run
    async def run(self, config: JobConfig) -> JobResult:
        return await aggregate_job(await plan_job(config), [])


@workflow.defn
class TrialNames:
    """Run a job's trials and report the names it planned them under."""

    @workflow.run
    async def run(self, config: JobConfig) -> list[str]:
        plan = await plan_job(config)
        await asyncio.gather(*(execute_trial(t) for t in plan.trials))
        return [t.config.trial_name for t in plan.trials]


def _descriptive_name(task: TaskConfig, agent: AgentConfig, attempt: int) -> str:
    return f"{cast(Path, task.path).name}__{agent.name}__{attempt}"


def _escaping_name(task: TaskConfig, agent: AgentConfig, attempt: int) -> str:
    return f"../{_descriptive_name(task, agent, attempt)}"


@workflow.defn
class PlanNamedJob:
    """Plan the job with trial names derived from what each trial is."""

    @workflow.run
    async def run(self, config: JobConfig) -> TemporalJobPlan:
        return await plan_job(config, trial_name=_descriptive_name)


@workflow.defn
class PlanCollidingJob:
    """Plan the job with a naming function that gives every trial one name."""

    @workflow.run
    async def run(self, config: JobConfig) -> TemporalJobPlan:
        return await plan_job(config, trial_name=lambda *_: "same")


def _no_name(*_: object) -> Any:
    return None


@workflow.defn
class PlanEscapingJob:
    """Plan the job with a naming function that puts each trial outside the job."""

    @workflow.run
    async def run(self, config: JobConfig) -> TemporalJobPlan:
        return await plan_job(config, trial_name=_escaping_name)


@workflow.defn
class PlanWithInvalidTimeout:
    """Plan the job with an ``untimed_agent_timeout`` that is not a duration."""

    @workflow.run
    async def run(self, config: JobConfig) -> TemporalJobPlan:
        return await plan_job(config, untimed_agent_timeout=cast(Any, "soon"))


@workflow.defn
class PlanAndAggregateOnce:
    """Plan the job and aggregate it in one attempt, without running any trial."""

    @workflow.run
    async def run(self, config: JobConfig) -> JobResult:
        plan = await plan_job(config)
        return await aggregate_job(
            plan, [], retry_policy=RetryPolicy(maximum_attempts=1)
        )


@workflow.defn
class PlanInvalidNameJob:
    """Plan the job with a naming function that returns no name at all."""

    @workflow.run
    async def run(self, config: JobConfig) -> TemporalJobPlan:
        return await plan_job(config, trial_name=_no_name)


@workflow.defn
class PlanEmptyNameJob:
    """Plan the job with a naming function that returns an empty name."""

    @workflow.run
    async def run(self, config: JobConfig) -> TemporalJobPlan:
        return await plan_job(config, trial_name=lambda *_: "")


class TrialArgs(BaseModel):
    """How :class:`RunOneTrial` should run its trial."""

    config: TrialConfig
    timeout: timedelta = TRIAL_TIMEOUT
    extra_time: timedelta = timedelta(minutes=15)
    start_to_close_timeout: timedelta | None = None
    schedule_to_close_timeout: timedelta | None = None
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
            TrialPlan(config=args.config, timeout=args.timeout),
            extra_time=args.extra_time,
            start_to_close_timeout=args.start_to_close_timeout,
            schedule_to_close_timeout=args.schedule_to_close_timeout,
            retry_policy=policy,
            data=args.data,
            summary=args.summary,
        )


@workflow.defn
class RunTrialWithInvalidData:
    """Run a trial with ``data`` that is not JSON."""

    @workflow.run
    async def run(self, config: TrialConfig) -> TrialOutcome:
        return await execute_trial(
            TrialPlan(config=config, timeout=TRIAL_TIMEOUT),
            data=cast(Any, {"run": workflow.uuid4()}),
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
