"""Plan a harbor job, run its trials and aggregate them, from workflow code."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.job.result import JobStats
    from harbor.models.trial.config import TaskConfig, TrialConfig
    from harbor.models.trial.result import TrialResult
    from harbor.utils.pass_at_k import compute_pass_at_k_by_evals

    from temporalio.harbor._retry import retry_policy_from_harbor
    from temporalio.harbor._types import (
        COMPUTE_METRICS,
        RESOLVE_JOB,
        RUN_TRIAL,
        ComputeMetricsInput,
        JobPlan,
        Rewards,
        RunTrialInput,
    )

# Resolving a job and scoring it reach dataset registries and may run a
# dataset's metric script, so they retry transient failures; a ValueError is
# harbor rejecting the job's configuration, which no retry fixes.
_LIFECYCLE_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=5,
    non_retryable_error_types=["ValueError"],
)


def _task_name(task: TaskConfig) -> str:
    return task.get_task_id().get_name().split("/")[-1]


def _trial_name(task: TaskConfig) -> str:
    # Harbor's own format, with the random suffix drawn from the workflow so
    # that replay produces the same name and a retry re-runs the same slot.
    return f"{_task_name(task)[:32].rstrip('_-')}__{workflow.uuid4().hex[:7]}"


async def plan_job(
    config: JobConfig, *, start_to_close_timeout: timedelta = timedelta(minutes=10)
) -> JobPlan:
    """Resolve a harbor job into the trials ``harbor run`` would run.

    Datasets are resolved in an Activity, since that can reach a registry. The
    trials are then expanded here exactly as harbor's ``Job`` expands them:
    every attempt of every task for every agent. Each gets a name that is
    stable across replay, so a retried trial re-runs the same slot.

    Build ``config`` outside the workflow and pass it in: ``JobConfig``
    defaults ``job_name`` from the clock.

    Args:
        config: The job, as it would be given to ``harbor run``.
        start_to_close_timeout: Bound on resolving the job's datasets.

    Raises:
        ApplicationError: If ``config`` is a regrade job, which is not supported.
    """
    if config.is_regrade:
        raise ApplicationError(
            "regrade jobs are not supported by temporalio.harbor",
            type="HarborRegradeUnsupported",
            non_retryable=True,
        )
    task_configs = await workflow.execute_activity(
        RESOLVE_JOB,
        config,
        result_type=list[TaskConfig],
        summary=f"resolve {config.job_name}",
        start_to_close_timeout=start_to_close_timeout,
        retry_policy=_LIFECYCLE_RETRY,
    )
    job_id = workflow.uuid4()
    job_dir = config.jobs_dir / config.job_name
    trials = [
        TrialConfig(
            task=task_config,
            trial_name=_trial_name(task_config),
            trials_dir=job_dir,
            install_only=config.install_only,
            agent=agent_config,
            user_agent=config.user_agent,
            timeout_multiplier=config.timeout_multiplier,
            agent_timeout_multiplier=config.agent_timeout_multiplier,
            verifier_timeout_multiplier=config.verifier_timeout_multiplier,
            agent_setup_timeout_multiplier=config.agent_setup_timeout_multiplier,
            environment_build_timeout_multiplier=config.environment_build_timeout_multiplier,
            environment=config.environment,
            verifier=config.verifier,
            artifacts=config.artifacts,
            extra_instruction_paths=config.extra_instruction_paths,
            extra_instructions=config.extra_instructions,
            job_id=job_id,
        )
        for _ in range(config.n_attempts)
        for task_config in task_configs
        # Agents innermost, as harbor orders them, so consecutive trials spread
        # across model providers.
        for agent_config in config.agents
    ]
    return JobPlan(config=config, task_configs=task_configs, trials=trials)


def _summary(config: TrialConfig) -> str:
    agent = config.agent.name or config.agent.import_path or "agent"
    if config.agent.model_name:
        agent = f"{agent}/{config.agent.model_name}"
    return f"{_task_name(config.task)} · {agent}"


async def execute_trial(
    config: TrialConfig,
    *,
    start_to_close_timeout: timedelta,
    retry: RetryConfig | None = None,
    heartbeat_timeout: timedelta = timedelta(minutes=2),
    infrastructure_retries: int = 3,
    summary: str | None = None,
) -> TrialResult:
    """Run one harbor trial as one Activity.

    A trial that runs and fails comes back as a result with ``exception_info``
    set, exactly as ``harbor run`` records it, after any retries ``retry``
    allows. Only a trial that could not run at all, after its infrastructure
    retries, raises.

    The result omits rollout details and agent metadata, which harbor's
    aggregation does not read; the full result stays in the trial directory.

    Args:
        config: The trial, usually one of :attr:`JobPlan.trials`.
        start_to_close_timeout: Bound on one attempt: environment start, agent
            and verifier. Required because trials range from seconds to hours.
        retry: Harbor's retry configuration. Defaults to harbor's, which does
            not retry.
        heartbeat_timeout: How long the trial may go without a heartbeat before
            its worker is presumed lost.
        infrastructure_retries: Extra attempts for failures harbor never
            records, such as a lost worker.
        summary: Shown for the Activity in the Temporal UI. Defaults to the
            task and agent.
    """
    retry = retry if retry is not None else RetryConfig()
    return await workflow.execute_activity(
        RUN_TRIAL,
        RunTrialInput(config=config, retry=retry),
        result_type=TrialResult,
        summary=summary if summary is not None else _summary(config),
        start_to_close_timeout=start_to_close_timeout,
        heartbeat_timeout=heartbeat_timeout,
        retry_policy=retry_policy_from_harbor(
            retry, infrastructure_retries=infrastructure_retries
        ),
    )


async def aggregate_job(
    plan: JobPlan,
    results: Sequence[TrialResult],
    *,
    start_to_close_timeout: timedelta = timedelta(minutes=10),
) -> JobStats:
    """Compute a job's statistics the way ``harbor run`` reports them.

    Reward and error statistics, token and cost totals, and pass@k come from
    harbor's own functions, run here over results already in the workflow.
    Each dataset's metrics are computed in an Activity from rewards alone,
    because a dataset's metric may be a script harbor runs with ``uv``.

    Pass every result of the job at once. Pass@k averages over tasks, so it
    cannot be combined from separately aggregated shards.

    Args:
        plan: The job's plan, from :func:`plan_job`.
        results: The job's trial results, from :func:`execute_trial`.
        start_to_close_timeout: Bound on computing the metrics.
    """
    results = list(results)
    stats = JobStats.from_trial_results(results, n_total_trials=len(plan.trials))

    rewards: defaultdict[str, list[Rewards | None]] = defaultdict(list)
    for result in results:
        model = result.agent_info.model_info
        evals_key = JobStats.format_agent_evals_key(
            result.agent_info.name,
            model.name if model else None,
            result.source or "adhoc",
        )
        verifier = result.verifier_result
        rewards[evals_key].append(verifier.rewards if verifier is not None else None)

    metrics = await workflow.execute_activity(
        COMPUTE_METRICS,
        ComputeMetricsInput(
            config=plan.config, task_configs=plan.task_configs, rewards=dict(rewards)
        ),
        result_type=dict[str, list[dict[str, float | int]]],
        summary=f"metrics · {plan.config.job_name}",
        start_to_close_timeout=start_to_close_timeout,
        retry_policy=_LIFECYCLE_RETRY,
    )
    for evals_key, computed in metrics.items():
        stats.evals[evals_key].metrics.extend(computed)
    for evals_key, pass_at_k in compute_pass_at_k_by_evals(results).items():
        stats.evals[evals_key].pass_at_k = pass_at_k
    return stats
