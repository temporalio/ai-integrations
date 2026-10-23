"""Plan a harbor job, run its trials and aggregate them, from workflow code."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import PurePath

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.exceptions import TimeoutError as ActivityTimeoutError

with workflow.unsafe.imports_passed_through():
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
    from harbor.utils.pass_at_k import compute_pass_at_k_by_evals
    from pydantic import JsonValue, ValidationError

    from temporalio.harbor._types import (
        AGGREGATE_JOB,
        PLAN_JOB,
        RUN_TRIAL,
        AggregateJobInput,
        PlanJobInput,
        PlanJobResult,
        Rewards,
        RunTrialInput,
        TemporalJobPlan,
        TrialOutcome,
        TrialPlan,
        as_outcome,
        check_trial_name,
        heartbeat_outcome,
    )

# Planning reaches dataset registries, so it retries transient failures, but
# only a few times, since no trial has run yet. The planning Activity marks
# harbor rejecting the job's configuration as non-retryable.
_LIFECYCLE_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=5,
)

_TRIAL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=15),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=2),
    maximum_attempts=3,
)

# Harbor's trial-name suffix: 7 characters of ShortUUID's base57 alphabet.
_SUFFIX_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_SUFFIX_LENGTH = 7


def _invalid(function: str, exc: ValidationError) -> ApplicationError:
    # A ValidationError raised in workflow code fails the workflow task, which
    # Temporal retries indefinitely, instead of failing the workflow.
    return ApplicationError(
        f"{function}'s arguments are not valid: {exc}",
        type="ValidationError",
        non_retryable=True,
    )


async def plan_job(
    config: JobConfig,
    *,
    untimed_agent_timeout: timedelta = timedelta(hours=2),
    trial_name: Callable[[TaskConfig, AgentConfig, int], str] | None = None,
    start_to_close_timeout: timedelta = timedelta(minutes=10),
    task_queue: str | None = None,
) -> TemporalJobPlan:
    """Resolve a harbor job into the trials it will run.

    An Activity resolves the job's datasets into tasks as harbor's ``Job``
    does and checks what does not depend on the worker: each agent's options
    against the schema the agent declares, and each dataset's metrics. It
    writes the job's ``config.json`` and ``lock.json`` under
    ``jobs_dir/job_name``, and fails, as harbor does, if that directory
    already has a ``config.json`` that differs from this job's. The trials are
    then expanded here as harbor expands them: every task, for every agent and
    each of the job's ``n_attempts``. Each trial gets a name that is stable
    across replay, and a timeout that adds up the phase timeouts harbor will
    enforce for it.

    A ``config`` whose ``job_name`` was never set, such as one built in
    workflow code or received without one, is named here from the
    workflow's clock, in harbor's format but in UTC, so the name holds
    across replay; read it from the returned plan's ``config``.

    ``config.retry`` and each agent's ``n_concurrent`` and
    ``concurrency_group`` have no effect; a warning is logged when they are
    set. Temporal retries trials as :func:`temporalio.harbor.execute_trial`
    describes, and the worker's ``max_concurrent_activities`` bounds how many
    trials run at once.

    Args:
        config: The job, as it would be given to ``harbor run``.
        untimed_agent_timeout: Counted as the agent's timeout in a trial's
            timeout when neither the task nor the agent's configuration sets
            one. Harbor still runs such an agent without a timeout.
        trial_name: Names each trial from its task, its agent and which of the
            job's ``n_attempts`` it is, counting from 0. A trial's name is its
            directory. Must be deterministic and unique within the job.
            Defaults to harbor's format, with a random suffix drawn from the
            workflow's random source.
        start_to_close_timeout: Bound on the planning Activity.
        task_queue: Task queue for the planning Activity. Defaults to the
            workflow's.

    Raises:
        ApplicationError: Non-retryable, if ``config`` is a regrade job, which
            is not supported, if an argument or a trial's configuration is
            invalid, if a trial's name is not a single path component, or if
            two trials are given the same name.
        ActivityError: If planning failed, including when harbor rejected the
            job's configuration or ``jobs_dir/job_name`` holds a different one.
    """
    if "job_name" not in config.model_fields_set:
        # JobConfig took the default job_name from the clock when workflow
        # code built it, and takes another on every replay.
        config = config.model_copy(
            update={"job_name": workflow.now().strftime("%Y-%m-%d__%H-%M-%S")}
        )
    if config.is_regrade:
        raise ApplicationError(
            "regrade jobs are not supported by temporalio.harbor",
            type="HarborRegradeUnsupported",
            non_retryable=True,
        )
    if config.retry != RetryConfig():
        workflow.logger.warning(
            "JobConfig.retry is ignored: Temporal retries trials under "
            "execute_trial's retry_policy"
        )
    capped = [
        agent.name or agent.import_path
        for agent in config.agents
        if agent.n_concurrent is not None or agent.concurrency_group is not None
    ]
    if capped:
        workflow.logger.warning(
            f"n_concurrent and concurrency_group are ignored for agents {capped}: "
            "the worker's max_concurrent_activities bounds concurrent trials"
        )
    try:
        plan_input = PlanJobInput(
            config=config, untimed_agent_timeout=untimed_agent_timeout
        )
    except ValidationError as exc:
        raise _invalid("plan_job", exc) from None
    planned = await workflow.execute_activity(
        PLAN_JOB,
        plan_input,
        result_type=PlanJobResult,
        summary=f"plan {config.job_name}",
        task_queue=task_queue,
        start_to_close_timeout=start_to_close_timeout,
        retry_policy=_LIFECYCLE_RETRY,
    )
    job_id = workflow.uuid4()
    random = workflow.random()
    trials: list[TrialPlan] = []
    seen: set[str] = set()
    for attempt in range(config.n_attempts):
        for task, task_name, timeouts in zip(
            planned.task_configs, planned.task_names, planned.timeouts, strict=True
        ):
            # Agents innermost, as harbor orders them.
            for agent, timeout in zip(config.agents, timeouts, strict=True):
                if trial_name is not None:
                    name = trial_name(task, agent, attempt)
                else:
                    suffix = "".join(
                        random.choice(_SUFFIX_ALPHABET) for _ in range(_SUFFIX_LENGTH)
                    )
                    name = f"{task_name[:32].rstrip('_-')}__{suffix}"
                # Checked before TrialConfig, which replaces an empty name with
                # a random one. A name that is not a string fails validation.
                if isinstance(name, str):
                    check_trial_name(name)
                try:
                    trial = TrialConfig(
                        task=task,
                        trial_name=name,
                        trials_dir=config.jobs_dir / config.job_name,
                        install_only=config.install_only,
                        agent=agent,
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
                except ValidationError as exc:
                    raise ApplicationError(
                        f"trial {name!r} is not a valid harbor trial: {exc}",
                        type="ValidationError",
                        non_retryable=True,
                    ) from None
                if trial.trial_name in seen:
                    raise ApplicationError(
                        f"two trials are named {trial.trial_name!r}; each trial "
                        "needs its own directory",
                        type="HarborDuplicateTrialName",
                        non_retryable=True,
                    )
                seen.add(trial.trial_name)
                trials.append(TrialPlan(config=trial, timeout=timeout))
    return TemporalJobPlan(
        id=job_id,
        config=config.model_copy(update={"datasets": planned.datasets}),
        task_configs=planned.task_configs,
        trials=trials,
    )


def _summary(config: TrialConfig) -> str:
    task = PurePath(config.task.name or config.task.path or "").name
    agent = config.agent.name or config.agent.import_path or "agent"
    if config.agent.model_name:
        agent = f"{agent}/{config.agent.model_name}"
    return f"{task} · {agent}"


def _recorded(error: ActivityError) -> TrialOutcome | None:
    # A failure the worker raised carries the outcome in its details; a
    # timeout the server recorded, in the attempt's last heartbeat.
    cause = error.__cause__
    while cause is not None:
        outcome = None
        if isinstance(cause, ActivityTimeoutError):
            outcome = heartbeat_outcome(cause.last_heartbeat_details)
        elif isinstance(cause, ApplicationError) and cause.details:
            outcome = as_outcome(cause.details[-1])
        if outcome is not None:
            return outcome
        cause = cause.__cause__
    return None


async def execute_trial(
    trial: TrialPlan,
    *,
    extra_time: timedelta = timedelta(minutes=15),
    start_to_close_timeout: timedelta | None = None,
    heartbeat_timeout: timedelta = timedelta(minutes=2),
    retry_policy: RetryPolicy | None = None,
    schedule_to_close_timeout: timedelta | None = None,
    task_queue: str | None = None,
    data: JsonValue = None,
    summary: str | None = None,
) -> TrialOutcome:
    """Run one harbor trial as one Activity.

    A trial harbor ran comes back as its result, including a result that
    records an exception, such as an agent that ran out of time or exited
    with an error: harbor scored it. Temporal retries the trial under
    ``retry_policy`` when it could not run, for example because its task
    could not be fetched or its environment check failed, when harbor's
    result shows it failed before its agent started, or when an agent run
    ended in one of harbor's transient provider errors:
    ``ApiRateLimitError``, ``ApiOverloadedError``, ``ApiInternalServerError``,
    ``ApiConnectionClosedError``, ``ApiResponseStalledError`` or
    ``NetworkConnectionError``. A trial whose name is not a single path
    component, or that harbor rejects as misconfigured while creating it,
    for example because an agent option is invalid or a ``${VAR}`` its
    config names is unset on the worker, fails without a retry. For a trial
    harbor ran, the error is an ``ApplicationError`` whose type is harbor's
    exception class name, with the outcome attached; after
    ``ApiRateLimitError``, the next attempt waits 2 minutes. A worker's
    :meth:`temporalio.harbor.TrialHooks.output` can also have Temporal run a
    trial again. When Temporal stops retrying, the most recent outcome any
    attempt produced is returned, however the last attempt ended.

    The result omits rollout details and agent metadata, which harbor's
    aggregation does not read. Harbor writes the complete result to the trial
    directory on the worker that ran the attempt, and a later attempt of the
    trial that uses the same directory starts by deleting it.

    Args:
        trial: The trial, one of
            :attr:`temporalio.harbor.TemporalJobPlan.trials`.
        extra_time: Added to the default ``start_to_close_timeout`` for what
            harbor does not time: the worker's hooks, fetching the task,
            copying files, and tearing the environment down. It also bounds
            the total time an attempt spends in the worker's async hooks; a
            hook that runs past it fails the attempt for Temporal to retry.
        start_to_close_timeout: Bound on one attempt. Defaults to 1.5 times
            the trial's timeout, plus ``extra_time``.
        heartbeat_timeout: How long an attempt may go without a heartbeat
            before Temporal presumes its worker lost and retries the trial.
        retry_policy: How Temporal retries an attempt that raised. Defaults
            to 3 attempts, 15 seconds apart at first, doubling up to 2
            minutes.
        schedule_to_close_timeout: Bound on the trial across all its attempts.
        task_queue: Task queue for the Activity. Defaults to the workflow's.
        data: Handed to the worker's :class:`temporalio.harbor.TrialHooks` as
            :attr:`temporalio.harbor.TrialContext.data`.
        summary: Shown for the Activity in the Temporal UI. Defaults to the
            task and agent.

    Raises:
        ApplicationError: Non-retryable, if an argument is invalid, for
            example ``data`` that is not JSON.
        ActivityError: If Temporal stopped retrying and no attempt produced a
            result.
    """
    try:
        run_input = RunTrialInput(config=trial.config, extra_time=extra_time, data=data)
    except ValidationError as exc:
        raise _invalid("execute_trial", exc) from None
    try:
        return await workflow.execute_activity(
            RUN_TRIAL,
            run_input,
            result_type=TrialOutcome,
            summary=summary if summary is not None else _summary(trial.config),
            task_queue=task_queue,
            start_to_close_timeout=(
                start_to_close_timeout
                if start_to_close_timeout is not None
                else trial.timeout * 1.5 + extra_time
            ),
            schedule_to_close_timeout=schedule_to_close_timeout,
            heartbeat_timeout=heartbeat_timeout,
            retry_policy=retry_policy if retry_policy is not None else _TRIAL_RETRY,
        )
    except ActivityError as error:
        outcome = _recorded(error)
        if outcome is None:
            raise
        return outcome


async def aggregate_job(
    plan: TemporalJobPlan,
    outcomes: Sequence[TrialOutcome],
    *,
    start_to_close_timeout: timedelta = timedelta(minutes=10),
    retry_policy: RetryPolicy | None = None,
    task_queue: str | None = None,
) -> JobResult:
    """Aggregate a job's trial outcomes into harbor's ``JobResult``.

    Reward and error statistics, token and cost totals, and pass@k come from
    harbor's own functions, run here over the results in the workflow.
    ``n_retries`` counts the attempts Temporal made before each outcome. Each
    dataset's metrics are computed in an Activity from rewards alone, because
    a dataset's metric may be a script harbor runs with ``uv``; the Activity
    looks the metrics up again and writes the job's ``result.json``.

    Pass every outcome of the job at once: pass@k averages over tasks, so it
    cannot be combined from separately aggregated parts. A planned trial with
    no outcome counts toward ``n_total_trials`` and ``n_pending_trials``, as a
    trial harbor has not finished.

    The returned ``JobResult`` carries every trial's result, so it can exceed
    Temporal's payload size limit if a workflow returns it whole; return its
    ``stats`` instead.

    Args:
        plan: The job's plan, from :func:`temporalio.harbor.plan_job`.
        outcomes: The job's trial outcomes, from
            :func:`temporalio.harbor.execute_trial`.
        start_to_close_timeout: Bound on the aggregation Activity.
        retry_policy: How Temporal retries the aggregation Activity. Defaults
            to Temporal's default policy, which retries until the Activity
            succeeds; harbor rejecting the job's configuration is not retried.
        task_queue: Task queue for the aggregation Activity. Defaults to the
            workflow's.

    Raises:
        ApplicationError: Non-retryable, if an argument is invalid.
        ActivityError: If the aggregation Activity failed, including when
            harbor rejected the job's configuration.
    """
    results = [outcome.result for outcome in outcomes]
    stats = JobStats.from_trial_results(
        results,
        n_total_trials=len(plan.trials),
        n_retries=sum(outcome.attempt - 1 for outcome in outcomes),
    )
    for evals_key, pass_at_k in compute_pass_at_k_by_evals(results).items():
        stats.evals[evals_key].pass_at_k = pass_at_k

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

    now = workflow.now()
    started = [result.started_at for result in results if result.started_at]
    try:
        aggregate_input = AggregateJobInput(
            config=plan.config,
            task_configs=plan.task_configs,
            rewards=dict(rewards),
            result=JobResult(
                id=plan.id,
                started_at=min(started, default=now),
                updated_at=now,
                finished_at=now,
                n_total_trials=len(plan.trials),
                stats=stats,
            ),
        )
    except ValidationError as exc:
        raise _invalid("aggregate_job", exc) from None
    aggregated = await workflow.execute_activity(
        AGGREGATE_JOB,
        aggregate_input,
        result_type=JobResult,
        summary=f"aggregate {plan.config.job_name}",
        task_queue=task_queue,
        start_to_close_timeout=start_to_close_timeout,
        retry_policy=retry_policy,
    )
    aggregated.trial_results = results
    return aggregated
