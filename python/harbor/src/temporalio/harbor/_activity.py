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
from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.trial.config import TaskConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.trial import Trial

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.harbor import _compat
from temporalio.harbor._types import (
    COMPUTE_METRICS,
    RESOLVE_JOB,
    RUN_TRIAL,
    ComputeMetricsInput,
    Rewards,
    RunTrialInput,
)

# A recorded traceback is kept for debugging but never read by harbor's
# aggregation. The tail holds the frames that matter; the head of a deep
# harbor traceback is the same event-loop plumbing every time.
_TRACEBACK_MAX = 8_000


def should_retry(config: RetryConfig, exception_type: str) -> bool:
    """Whether harbor would retry a trial that recorded ``exception_type``.

    ``exclude_exceptions`` takes precedence over ``include_exceptions``, and an
    unset ``include_exceptions`` admits everything, as in harbor's own queue.
    """
    if config.exclude_exceptions and exception_type in config.exclude_exceptions:
        return False
    if config.include_exceptions and exception_type not in config.include_exceptions:
        return False
    return True


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


class HarborActivities:
    """The Activities registered by :class:`temporalio.harbor.HarborPlugin`."""

    def __init__(self, *, heartbeat_interval: timedelta) -> None:
        """Configure how often a running trial heartbeats."""
        if heartbeat_interval <= timedelta(0):
            raise ValueError("heartbeat_interval must be positive")
        self._heartbeat_interval = heartbeat_interval.total_seconds()

    @activity.defn(name=RESOLVE_JOB)
    async def resolve_job(self, config: JobConfig) -> list[TaskConfig]:
        """Resolve a job's tasks, reaching dataset registries as harbor does."""
        EnvironmentFactory.validate_resource_policies(config.environment)
        return await _compat.resolve_task_configs(config)

    @activity.defn(name=RUN_TRIAL)
    async def run_trial(self, input: RunTrialInput) -> TrialResult:
        """Run one harbor trial.

        A trial that runs and fails is a result, as it is in ``harbor run``:
        it is returned, unless harbor's retry configuration says to try it
        again, in which case it raises so that Temporal re-runs the slot.
        """
        info = activity.info()
        config = input.config
        _set_aside(Path(config.trials_dir) / config.trial_name, info.attempt)

        # Harbor's lifecycle events name the phase; "load" covers fetching the
        # task, which happens before the trial exists to emit anything.
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

        beat = asyncio.create_task(heartbeat())
        try:
            trial = await Trial.create(config)
            for event in TrialEvent:
                trial.add_hook(event, track(event))
            result = await trial.run()
        finally:
            beat.cancel()

        exc = result.exception_info
        if (
            exc is not None
            and should_retry(input.retry, exc.exception_type)
            and info.attempt <= input.retry.max_retries
        ):
            activity.logger.warning(
                f"trial {config.trial_name} attempt {info.attempt} raised "
                f"{exc.exception_type}; retrying"
            )
            raise ApplicationError(
                f"{exc.exception_type}: {exc.exception_message}",
                type=exc.exception_type,
            )
        return slim(result)

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
