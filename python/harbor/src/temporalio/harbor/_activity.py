"""The Activities a harbor job is made of."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from datetime import timedelta
from pathlib import Path
from typing import Any

from harbor.environments.factory import EnvironmentFactory
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.trial import Trial

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.harbor import _compat
from temporalio.harbor._hooks import TrialContext, TrialHooks
from temporalio.harbor._types import (
    COMPUTE_METRICS,
    RESOLVE_JOB,
    RUN_TRIAL,
    ComputeMetricsInput,
    ResolveJobResult,
    Rewards,
    RunTrialInput,
    TrialOutcome,
)

# A recorded traceback is kept for debugging but never read by harbor's
# aggregation. The tail holds the frames that matter; the head of a deep
# harbor traceback is the same event-loop plumbing every time.
_TRACEBACK_MAX = 8_000


def slim(result: TrialResult) -> TrialResult:
    """Drop the parts of a result that harbor's aggregation never reads.

    Rollout details carry token IDs and log-probabilities for every trajectory,
    and agent metadata is free-form; together they are most of a result's size.
    What remains is exactly what ``JobStats`` and pass@k read. The complete
    result is still on disk in the trial directory, where harbor wrote it.
    """

    def strip(context: AgentContext | None) -> AgentContext | None:
        if context is None:
            return None
        return context.model_copy(update={"rollout_details": None, "metadata": None})

    update: dict[str, Any] = {"agent_result": strip(result.agent_result)}
    if result.step_results is not None:
        update["step_results"] = [
            step.model_copy(update={"agent_result": strip(step.agent_result)})
            for step in result.step_results
        ]
    exc = result.exception_info
    if exc is not None and len(exc.exception_traceback) > _TRACEBACK_MAX:
        update["exception_info"] = exc.model_copy(
            update={"exception_traceback": exc.exception_traceback[-_TRACEBACK_MAX:]}
        )
    return result.model_copy(update=update)


def _set_aside(trial_dir: Path, attempt: int) -> None:
    """Keep an earlier attempt's directory instead of letting this one reuse it.

    The trial name is stable across retries so the slot is re-run in place, but
    harbor writes a lock and a result into the directory; moving it aside keeps
    the evidence of why the earlier attempt failed.
    """
    if not trial_dir.exists():
        return
    n = attempt - 1
    while (aside := trial_dir.with_name(f"{trial_dir.name}.attempt-{n}")).exists():
        n += 1
    trial_dir.rename(aside)


def _last_attempt(info: activity.Info) -> bool:
    policy = info.retry_policy
    return (
        policy is not None
        and policy.maximum_attempts > 0
        and info.attempt >= policy.maximum_attempts
    )


class HarborActivities:
    """The Activities registered by :class:`temporalio.harbor.HarborPlugin`."""

    def __init__(self, *, heartbeat_interval: timedelta, hooks: TrialHooks) -> None:
        """Configure how often a running trial heartbeats, and how it runs."""
        if heartbeat_interval <= timedelta(0):
            raise ValueError("heartbeat_interval must be positive")
        self._heartbeat_interval = heartbeat_interval.total_seconds()
        self._hooks = hooks

    @activity.defn(name=RESOLVE_JOB)
    async def resolve_job(self, config: JobConfig) -> ResolveJobResult:
        """Resolve tasks and dataset refs, reaching registries as harbor does."""
        EnvironmentFactory.validate_resource_policies(config.environment)
        task_configs = await _compat.resolve_task_configs(config)
        return ResolveJobResult(
            task_configs=task_configs,
            dataset_refs=[dataset.ref for dataset in config.datasets],
        )

    @activity.defn(name=RUN_TRIAL)
    async def run_trial(self, input: RunTrialInput) -> TrialOutcome:
        """Run one harbor trial.

        A trial that runs and fails is a result, as it is in ``harbor run``:
        it is returned, unless the hooks' retry decision says to try it again,
        in which case it raises so that Temporal re-runs the slot.
        """
        info = activity.info()
        context = TrialContext(
            config=input.config,
            retry=input.retry,
            attempt=info.attempt,
            data=input.data,
        )
        _set_aside(context.trial_dir, info.attempt)

        # Harbor's lifecycle events name the phase; "load" covers the hooks'
        # setup and fetching the task, before the trial can emit anything.
        phase = "load"
        started = time.monotonic()

        async def heartbeat() -> None:
            # A ticker rather than a beat per event: the agent phase can run for
            # hours with no event, and the heartbeat timeout must still hold.
            while True:
                activity.heartbeat(
                    {"phase": phase, "elapsed_sec": round(time.monotonic() - started)}
                )
                await asyncio.sleep(self._heartbeat_interval)

        def track(
            event: TrialEvent,
        ) -> Callable[[TrialHookEvent], Coroutine[Any, Any, None]]:
            async def hook(_: TrialHookEvent) -> None:
                nonlocal phase
                phase = event.value

            return hook

        outcome: TrialOutcome | None = None
        beat = asyncio.create_task(heartbeat())
        try:
            async with self._hooks.scope(context):
                trial = await Trial.create(context.config)
                for event in TrialEvent:
                    trial.add_hook(event, track(event))
                await self._hooks.trial_created(trial, context)
                result = await trial.run()
                exc = result.exception_info
                if exc is not None and not _last_attempt(info):
                    retry = self._hooks.retry(result, context)
                    if retry is not None:
                        activity.logger.warning(
                            f"trial {context.config.trial_name} attempt "
                            f"{info.attempt} raised {exc.exception_type}; "
                            f"retrying in {retry.delay}"
                        )
                        raise ApplicationError(
                            f"{exc.exception_type}: {exc.exception_message}",
                            type=exc.exception_type,
                            next_retry_delay=retry.delay,
                        )
                output = await self._hooks.output(result, context)
                outcome = TrialOutcome(result=slim(result), output=output)
        finally:
            beat.cancel()
        # Type checkers take the scope at its word that it never suppresses
        # an exception; one that does leaves no result to return.
        if outcome is None:  # type: ignore[reportUnnecessaryComparison]
            raise RuntimeError(  # type: ignore[reportUnreachable]
                f"{type(self._hooks).__name__}.scope suppressed an exception, "
                f"so trial {context.config.trial_name} produced no result"
            )
        return outcome

    @activity.defn(name=COMPUTE_METRICS)
    async def compute_metrics(
        self, input: ComputeMetricsInput
    ) -> dict[str, list[dict[str, float | int]]]:
        """Score each eval's rewards with the metrics harbor assigns its dataset.

        Metrics are resolved and computed together because a dataset's metric
        can be a script harbor downloads to this machine and runs with ``uv``.
        """
        metrics = await _compat.resolve_metrics(input.config, input.task_configs)
        scored: dict[str, list[dict[str, float | int]]] = {}
        for evals_key, rewards in input.rewards.items():
            # Harbor keys evals "agent__model__dataset" (or "agent__dataset")
            # and reads the dataset back off the end the same way.
            dataset = evals_key.split("__")[-1]
            scored[evals_key] = [
                await asyncio.to_thread(_compute, metric, rewards)
                for metric in metrics[dataset]
            ]
        return scored


def _compute(metric: Any, rewards: list[Rewards | None]) -> dict[str, float | int]:
    return metric.compute(rewards)
