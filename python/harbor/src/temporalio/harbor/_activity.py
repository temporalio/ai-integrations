"""The Activities a harbor job is made of."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import time
import weakref
from collections.abc import Awaitable, Callable, Collection, Coroutine, Iterator
from datetime import timedelta
from typing import Any, TypeVar
from uuid import uuid4

import harbor.skills
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import (
    ApiConnectionClosedError,
    ApiInternalServerError,
    ApiOverloadedError,
    ApiRateLimitError,
    ApiResponseStalledError,
    NetworkConnectionError,
)
from harbor.environments.factory import EnvironmentFactory
from harbor.job_plan import JobPlan as HarborJobPlan
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.job.lock import LOCK_FILENAME, build_job_lock
from harbor.models.job.result import JobResult
from harbor.models.task.config import (
    HealthcheckConfig,
    VerifierCollectConfig,
    VerifierEnvironmentMode,
)
from harbor.models.task.config import TaskConfig as TaskDefinition
from harbor.models.task.paths import TaskPaths
from harbor.models.task.verifier_mode import (
    resolve_step_verifier_mode,
    resolve_task_verifier_mode,
)
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.models.trial.result import ExceptionInfo, TrialResult
from harbor.registry.client.base import EmptyDatasetError, UnavailableDatasetTasksError
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.trial import Trial
from pydantic import ValidationError

from temporalio import activity
from temporalio.api.failure.v1 import Failure
from temporalio.converter import ActivitySerializationContext
from temporalio.exceptions import ApplicationError, FailureError
from temporalio.harbor._hooks import TrialContext, TrialHooks
from temporalio.harbor._types import (
    AGGREGATE_JOB,
    PLAN_JOB,
    RUN_TRIAL,
    AggregateJobInput,
    PlanJobInput,
    PlanJobResult,
    RunTrialInput,
    TrialOutcome,
    check_trial_name,
    heartbeat_outcome,
)

_T = TypeVar("_T")

# Harbor's aggregation reads an exception's type, never its text.
_TEXT_MAX = 8_000

# Temporal's default limit on a payload is 2 MiB. An outcome also travels in
# heartbeat details, in error details and through a payload codec, so leave
# room for what each adds.
_OUTCOME_MAX = 2 * 1024 * 1024 - 64 * 1024

_TRANSIENT = frozenset(
    error.__name__
    for error in (
        ApiRateLimitError,
        ApiOverloadedError,
        ApiInternalServerError,
        ApiConnectionClosedError,
        ApiResponseStalledError,
        NetworkConnectionError,
    )
)

# The text aggregate_job reads from a trial's result, as paths into its JSON
# form. Redaction leaves it as harbor recorded it.
_AGGREGATED = (
    ("trial_name",),
    ("task_name",),
    ("source",),
    ("started_at",),
    ("agent_info", "name"),
    ("agent_info", "model_info", "name"),
    ("exception_info", "exception_type"),
)


def _strip(result: TrialResult) -> TrialResult:
    """Drop rollout details and agent metadata, which aggregation never reads."""

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
    return result.model_copy(update=update)


def _cap(result: TrialResult) -> TrialResult:
    """Cap each exception's message (head) and traceback (tail) at 8,000 characters."""

    def cap(exc: ExceptionInfo | None) -> ExceptionInfo | None:
        if exc is None:
            return None
        return exc.model_copy(
            update={
                "exception_message": exc.exception_message[:_TEXT_MAX],
                "exception_traceback": exc.exception_traceback[-_TEXT_MAX:],
            }
        )

    update: dict[str, Any] = {"exception_info": cap(result.exception_info)}
    if result.step_results is not None:
        update["step_results"] = [
            step.model_copy(update={"exception_info": cap(step.exception_info)})
            for step in result.step_results
        ]
    return result.model_copy(update=update)


def _retried(result: TrialResult) -> ExceptionInfo | None:
    """The recorded exception for which Temporal runs the trial again, if any.

    That is the first exception when no agent started, else the first
    transient provider error. Harbor sets ``agent_execution`` as an agent
    starts: on the result, or on each step's result for a multi-step trial.
    """
    steps = result.step_results or []
    recorded = sorted(
        filter(None, [result.exception_info, *(s.exception_info for s in steps)]),
        key=lambda exc: exc.occurred_at,
    )
    if all(r.agent_execution is None for r in [result, *steps]):
        return next(iter(recorded), None)
    return next((exc for exc in recorded if exc.exception_type in _TRANSIENT), None)


def _healthcheck_sec(check: HealthcheckConfig | None) -> float:
    # Each check runs for int(timeout_sec): every start_interval_sec during the
    # start period, then until `retries` consecutive failures, interval_sec apart.
    if check is None:
        return 0
    run = int(check.timeout_sec)
    start = (
        check.start_period_sec + run + check.start_interval_sec
        if check.start_period_sec > 0
        else 0
    )
    return start + max(check.retries, 1) * (run + check.interval_sec)


def _collect_sec(hooks: list[VerifierCollectConfig]) -> float:
    return sum(int(hook.timeout_sec) for hook in hooks)


def trial_timeout(
    trial: TrialConfig, task: TaskDefinition, untimed_agent_timeout: timedelta
) -> timedelta:
    """The phase timeouts harbor 0.23's ``Trial`` enforces for a trial, added up.

    Each is resolved exactly as ``Trial`` resolves it. An agent harbor runs
    with no timeout counts as ``untimed_agent_timeout``.
    """

    def resolve(
        base: float, multiplier: float | None, cap: float | None = None
    ) -> float:
        if multiplier is None:
            multiplier = trial.timeout_multiplier
        return min(base, cap or float("inf")) * multiplier

    def agent(task_sec: float | None) -> float:
        base = trial.agent.override_timeout_sec or task_sec
        if base is None:
            return untimed_agent_timeout.total_seconds()
        return resolve(
            base, trial.agent_timeout_multiplier, trial.agent.max_timeout_sec
        )

    def verifier(task_sec: float) -> float:
        return resolve(
            trial.verifier.override_timeout_sec or task_sec,
            trial.verifier_timeout_multiplier,
            trial.verifier.max_timeout_sec,
        )

    build = resolve(
        task.environment.build_timeout_sec, trial.environment_build_timeout_multiplier
    )
    setup = resolve(
        trial.agent.override_setup_timeout_sec
        if trial.agent.override_setup_timeout_sec is not None
        else Trial._AGENT_SETUP_TIMEOUT_SEC,
        trial.agent_setup_timeout_multiplier,
    )
    # A user agent adds its own setup, and the bridge's setup, trajectory
    # export and teardown, each bounded by the agent setup timeout.
    total = build + setup * (5 if trial.user_agent is not None else 1)
    total += _healthcheck_sec(task.environment.healthcheck)
    collect = _collect_sec(task.verifier.collect)
    if not task.steps:
        total += agent(task.agent.timeout_sec) + collect
        total += verifier(task.verifier.timeout_sec)
        if resolve_task_verifier_mode(task) == VerifierEnvironmentMode.SEPARATE:
            total += build
    for step in task.steps or []:
        total += _healthcheck_sec(step.healthcheck)
        total += agent(
            step.agent.timeout_sec
            if step.agent.timeout_sec is not None
            else task.agent.timeout_sec
        )
        total += collect + _collect_sec(step.verifier.collect)
        total += verifier(step.verifier.timeout_sec)
        if resolve_step_verifier_mode(task, step) == VerifierEnvironmentMode.SEPARATE:
            total += build
    return timedelta(seconds=total)


@contextlib.contextmanager
def _config_errors() -> Iterator[None]:
    # Harbor rejecting a job's configuration, which no retry fixes.
    try:
        yield
    except (ValueError, EmptyDatasetError, UnavailableDatasetTasksError) as exc:
        raise ApplicationError(
            str(exc), type=type(exc).__name__, non_retryable=True
        ) from exc


def _check_options(agent: AgentConfig) -> None:
    model = AgentFactory.get_agent_class_from_config(agent).options_model
    if model is None:
        return
    try:
        model.model_validate(agent.kwargs)
    except ValidationError as exc:
        # An option missing here may come from the worker's environment.
        problems = [
            "unknown option '{}'".format(".".join(map(str, e["loc"])))
            if e["type"] == "extra_forbidden"
            else "{}: {}".format(".".join(map(str, e["loc"])), e["msg"])
            for e in exc.errors(include_url=False, include_input=False)
            if e["type"] != "missing"
        ]
        if problems:
            raise ValueError(
                f"invalid options for agent {agent.name or agent.import_path!r}: "
                + "; ".join(problems)
            ) from None


async def _check_environment(config: TrialConfig) -> None:
    try:
        await asyncio.to_thread(
            EnvironmentFactory.run_preflight,
            type=config.environment.type,
            import_path=config.environment.import_path,
        )
    except SystemExit as exc:
        raise RuntimeError(f"harbor's environment check failed: {exc}") from None


def _resolve_skills(config: TrialConfig) -> None:
    # Trial.create would otherwise run git under a file lock on the event loop.
    agent = config.agent
    sources = [skill for skill in agent.skills if isinstance(skill, str)]
    if config.source_trial is None and sources:
        resolved = harbor.skills.resolve_skill_sources(sources)
        agent.skills = [str(path) for path in resolved]


class _HookBudget:
    """The time one attempt's hooks may take between them."""

    def __init__(self, hooks: TrialHooks, extra_time: timedelta) -> None:
        self._hooks = type(hooks).__name__
        self._extra_time = extra_time
        self._remaining = extra_time.total_seconds()

    async def run(self, hook: str, call: Awaitable[_T]) -> _T:
        loop = asyncio.get_running_loop()
        started = loop.time()
        timeout = asyncio.timeout(self._remaining)
        try:
            async with timeout:
                return await call
        except TimeoutError:
            if not timeout.expired():
                raise
            raise ApplicationError(
                f"{self._hooks}.{hook} did not finish within the trial's extra "
                f"time of {self._extra_time}",
                type="HarborHookTimeout",
            ) from None
        finally:
            self._remaining -= loop.time() - started


class _RedactFailed(Exception):
    """:meth:`TrialHooks.redact` could not redact a text; ``failure`` says why."""

    def __init__(self, failure: ApplicationError) -> None:
        super().__init__()
        self.failure = failure


def _redact_failed(
    hooks: TrialHooks, context: TrialContext, exc: Exception
) -> ApplicationError:
    return ApplicationError(
        f"{type(hooks).__name__}.redact raised {type(exc).__name__} for trial "
        f"{context.config.trial_name}",
        type="HarborRedactFailed",
        non_retryable=True,
    )


def _redact_returned(
    hooks: TrialHooks, context: TrialContext, value: object
) -> ApplicationError:
    return ApplicationError(
        f"{type(hooks).__name__}.redact returned {type(value).__name__}, not str, "
        f"for trial {context.config.trial_name}",
        type="HarborRedactFailed",
        non_retryable=True,
    )


def _not_converted(context: TrialContext, exc: Exception) -> ApplicationError:
    return ApplicationError(
        f"the payload converter raised {type(exc).__name__} while converting the "
        f"error of trial {context.config.trial_name}",
        type="HarborFailureNotConverted",
        non_retryable=True,
    )


def _interrupted(
    details: activity.ActivityCancellationDetails | None, context: TrialContext
) -> ApplicationError:
    trial = context.config.trial_name
    if details is not None and details.worker_shutdown:
        return ApplicationError(
            f"the worker shut down before trial {trial} finished",
            type="WorkerShutdown",
        )
    if details is not None and details.paused:
        return ApplicationError(
            f"the Activity was paused while trial {trial} ran", type="ActivityPause"
        )
    if details is not None and details.reset:
        return ApplicationError(
            f"the Activity was reset while trial {trial} ran", type="ActivityReset"
        )
    return ApplicationError(
        f"trial {trial} was cancelled, but not by its workflow", type="CancelledError"
    )


def _redact_strings(
    value: Any,
    hooks: TrialHooks,
    context: TrialContext,
    keep: Collection[tuple[str, ...]] = (),
    path: tuple[str, ...] = (),
) -> Any:
    if path in keep:
        return value
    if isinstance(value, str):
        try:
            redacted = hooks.redact(value, context)
        except Exception as exc:
            raise _RedactFailed(_redact_failed(hooks, context, exc)) from None
        # Type checkers take redact at its word that it returns a str.
        if not isinstance(redacted, str):
            raise _RedactFailed(  # type: ignore[reportUnreachable]
                _redact_returned(hooks, context, redacted)
            )
        return redacted
    if isinstance(value, list):
        return [_redact_strings(item, hooks, context) for item in value]
    if isinstance(value, dict):
        return {
            key: _redact_strings(item, hooks, context, keep, (*path, key))
            for key, item in value.items()
        }
    return value


def _rewrites(
    before: Any, after: Any, path: tuple[Any, ...] = ()
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """The values ``after`` changed from ``before``, each with its path."""
    if isinstance(before, dict):
        for key, item in before.items():
            yield from _rewrites(item, after[key], (*path, key))
    elif isinstance(before, list):
        for index, item in enumerate(before):
            yield from _rewrites(item, after[index], (*path, index))
    elif before != after:
        yield path, after


def _replaced(data: Any, path: tuple[Any, ...], value: Any) -> Any:
    """``data`` with the value at ``path`` replaced, copying only along the path."""
    if not path:
        return value
    copied: Any = dict(data) if isinstance(data, dict) else list(data)
    copied[path[0]] = _replaced(data[path[0]], path[1:], value)
    return copied


def _redacted(
    result: TrialResult, hooks: TrialHooks, context: TrialContext
) -> TrialResult:
    recorded = result.model_dump(mode="json")
    try:
        redacted = _redact_strings(recorded, hooks, context, _AGGREGATED)
    except _RedactFailed as failed:
        raise failed.failure from None
    try:
        return TrialResult.model_validate(redacted)
    except ValidationError:
        pass
    # A rewrite left a value that is not free text invalid, such as a UUID, a
    # timestamp or a host the agent is allowed to reach. Apply the rewrites
    # one at a time, and leave out each one the result does not validate with.
    kept = recorded
    for path, text in _rewrites(recorded, redacted):
        candidate = _replaced(kept, path, text)
        try:
            TrialResult.model_validate(candidate)
        except ValidationError:
            continue
        kept = candidate
    return TrialResult.model_validate(kept)


def _failure(
    exc: Exception,
    outcome: TrialOutcome | None,
    hooks: TrialHooks,
    context: TrialContext,
) -> Exception:
    """``exc`` converted and redacted, carrying as much of ``outcome`` as it can."""
    info = activity.info()
    converter = activity.payload_converter()
    failure_converter = (
        activity.client()
        .data_converter.with_context(
            ActivitySerializationContext(
                namespace=info.namespace,
                activity_id=info.activity_id,
                activity_type=info.activity_type,
                activity_task_queue=info.task_queue,
                workflow_id=info.workflow_id,
                workflow_type=info.workflow_type,
                is_local=info.is_local,
            )
        )
        .failure_converter
    )

    def replaced(error: Exception) -> Failure:
        replacement = Failure()
        failure_converter.to_failure(
            error.failure
            if isinstance(error, _RedactFailed)
            else _not_converted(context, error),
            converter,
            replacement,
        )
        return replacement

    def attach(failure: Failure) -> None:
        if outcome is None or not failure.HasField("application_failure_info"):
            return
        payloads = failure.application_failure_info.details.payloads
        kept = len(payloads)
        for carried in (outcome, outcome.model_copy(update={"output": None})):
            try:
                converted = converter.to_payloads([carried])
            except Exception:
                continue
            payloads.extend(converted)
            if carried is not outcome or failure.ByteSize() <= _OUTCOME_MAX:
                return
            del payloads[kept:]

    def plain(failure: Failure) -> None:
        if failure.HasField("encoded_attributes"):
            [attributes] = converter.from_payloads([failure.encoded_attributes])
            if isinstance(attributes, dict):
                failure.message = str(attributes.get("message", ""))
                failure.stack_trace = str(attributes.get("stack_trace", ""))
                failure.ClearField("encoded_attributes")

    failure = Failure()
    try:
        failure_converter.to_failure(exc, converter, failure)
    except Exception as error:
        failure = replaced(error)
    else:
        node = failure
        try:
            while True:
                node.message = _redact_strings(node.message, hooks, context)
                node.stack_trace = _redact_strings(node.stack_trace, hooks, context)
                if node.HasField("encoded_attributes"):
                    [attributes] = converter.from_payloads([node.encoded_attributes])
                    [encoded] = converter.to_payloads(
                        [_redact_strings(attributes, hooks, context)]
                    )
                    node.encoded_attributes.CopyFrom(encoded)
                if not node.HasField("cause"):
                    break
                node = node.cause
        except Exception as error:
            # Send none of the attempt's error: its text could not be redacted.
            failure = replaced(error)
    attach(failure)
    # The worker passes this through its failure converter once more, which
    # encodes the top level's message and stack trace again if it encodes
    # them at all, so the top level goes back as plain text.
    try:
        plain(failure)
    except Exception as error:
        failure = replaced(error)
        attach(failure)
        with contextlib.suppress(Exception):
            plain(failure)
    return FailureError(failure.message, failure=failure)


class HarborActivities:
    """The Activities registered by :class:`temporalio.harbor.HarborPlugin`."""

    def __init__(self, *, hooks: TrialHooks) -> None:
        """Run trials with ``hooks`` taking part."""
        self._hooks = hooks
        self._attempts: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    @activity.defn(name=PLAN_JOB)
    async def plan_job(self, input: PlanJobInput) -> PlanJobResult:
        """Resolve a job's tasks and check its configuration, as harbor's ``Job`` does.

        Writes the job's ``config.json`` and ``lock.json`` under
        ``jobs_dir/job_name``, and works out each trial's timeout from its
        task's ``task.toml``. Fails without retrying if that directory already
        has a ``config.json`` that differs from this job's, as harbor does.
        """
        config = input.config.model_copy(deep=True)
        with _config_errors():
            for agent in [*config.agents, *filter(None, [config.user_agent])]:
                _check_options(agent)
            await asyncio.to_thread(HarborJobPlan.resolve_agent_skills, config)
            task_configs = await HarborJobPlan.resolve_task_configs(config)
            EnvironmentFactory.validate_resource_policies(config.environment)
            await HarborJobPlan.resolve_metrics(config, task_configs)
            downloads = await HarborJobPlan.cache_tasks(task_configs)
            trials = HarborJobPlan.build_trial_configs(
                config, task_configs, job_id=uuid4()
            )

            def read_tasks() -> list[TaskDefinition]:
                return [
                    TaskDefinition.model_validate_toml(
                        TaskPaths(
                            downloads[task.get_task_id()].path
                        ).config_path.read_text()
                    )
                    for task in task_configs
                ]

            definitions = await asyncio.to_thread(read_tasks)
            n_agents = len(config.agents)
            timeouts = [
                [
                    trial_timeout(
                        trials[i * n_agents + j], task, input.untimed_agent_timeout
                    )
                    for j in range(n_agents)
                ]
                for i, task in enumerate(definitions)
            ]
        # Harbor resolves a package dataset by its ref alone and rejects one
        # that sets both ref and version.
        datasets = [
            dataset.model_copy(update={"version": None})
            if dataset.is_package()
            else dataset
            for dataset in config.datasets
        ]
        lock = await asyncio.to_thread(
            build_job_lock,
            config=config,
            trial_configs=trials,
            task_download_results=downloads,
        )
        recorded = config.model_copy(update={"datasets": datasets})

        def write() -> None:
            job_dir = config.jobs_dir / config.job_name
            job_dir.mkdir(parents=True, exist_ok=True)
            path = job_dir / "config.json"
            if path.exists():
                try:
                    existing = JobConfig.model_validate_json(path.read_text())
                except ValueError:
                    existing = None
                # As harbor does. An equal one may be an earlier attempt's.
                if existing is None or existing != recorded:
                    raise ApplicationError(
                        f"Job directory {job_dir} already exists with a "
                        "different config.",
                        type="FileExistsError",
                        non_retryable=True,
                    )
            path.write_text(recorded.model_dump_json(indent=4, exclude_defaults=True))
            (job_dir / LOCK_FILENAME).write_text(
                lock.model_dump_json(indent=4, exclude_none=True)
            )

        await asyncio.to_thread(write)
        return PlanJobResult(
            task_configs=task_configs,
            task_names=[
                task.get_task_id().get_name().split("/")[-1] for task in task_configs
            ],
            timeouts=timeouts,
            datasets=datasets,
        )

    @activity.defn(name=RUN_TRIAL)
    async def run_trial(self, input: RunTrialInput) -> TrialOutcome:
        """Run one attempt of a harbor trial.

        Returns the trial's result whenever harbor produced one, even one that
        records an exception, unless the trial failed before its agent started
        or its agent's run ended in a transient provider error. Then, as when
        the trial could not run or :meth:`TrialHooks.scope` or
        :meth:`TrialHooks.trial_created` raises, it raises for Temporal to run
        the trial again. It raises without a retry when the trial's name is not
        a single path component, when harbor rejects the trial's configuration
        while creating it, and when :meth:`TrialHooks.redact` raises. An
        ``ApplicationError`` from :meth:`TrialHooks.output` reaches Temporal as
        the hook raised it; anything else it raises fails the attempt without
        a retry.
        """
        info = activity.info()
        hooks = self._hooks
        context = TrialContext(
            config=input.config, attempt=info.attempt, data=input.data
        )

        # "waiting" for an earlier attempt of this trial on this worker; "load"
        # for the hooks' setup and fetching the task; then harbor's lifecycle
        # events name the phase.
        phase = "waiting"
        started = time.monotonic()
        # The most recent outcome of any attempt. Every heartbeat carries it,
        # so a later attempt, or a timeout the server records, can return it.
        latest = heartbeat_outcome(info.heartbeat_details)

        def beat() -> None:
            details: dict[str, Any] = {
                "phase": phase,
                "elapsed_sec": round(time.monotonic() - started),
            }
            if latest is not None:
                details["outcome"] = latest.model_dump(mode="json")
            activity.heartbeat(details)

        async def heartbeat(interval: float) -> None:
            # The agent phase can go hours without a harbor event, so beat on a timer.
            while True:
                beat()
                await asyncio.sleep(interval)

        def track(
            event: TrialEvent,
        ) -> Callable[[TrialHookEvent], Coroutine[Any, Any, None]]:
            async def hook(_: TrialHookEvent) -> None:
                nonlocal phase
                phase = event.value

            return hook

        ticker = None
        if info.heartbeat_timeout:
            ticker = asyncio.create_task(
                heartbeat(info.heartbeat_timeout.total_seconds() / 4)
            )
        outcome: TrialOutcome | None = None
        try:
            check_trial_name(context.config.trial_name)
            # Held through teardown, so an overlapping attempt of the same
            # trial waits and then starts from an empty directory.
            lock = self._attempts.setdefault(context.config.trial_name, asyncio.Lock())
            async with lock:
                phase = "load"
                await asyncio.to_thread(
                    shutil.rmtree, context.trial_dir, ignore_errors=True
                )
                budget = _HookBudget(hooks, input.extra_time)
                scope = hooks.scope(context)
                await budget.run("scope", scope.__aenter__())
                try:
                    await _check_environment(context.config)
                    with _config_errors():
                        await asyncio.to_thread(_resolve_skills, context.config)
                        trial = await Trial.create(context.config)
                    for event in TrialEvent:
                        trial.add_hook(event, track(event))
                    await budget.run(
                        "trial_created", hooks.trial_created(trial, context)
                    )
                    result = await trial.run()
                    # Redacted in full, then shortened: shortening first could
                    # cut a secret in two and leave a part redact cannot match.
                    outcome = TrialOutcome(
                        result=_cap(_redacted(_strip(result), hooks, context)),
                        attempt=info.attempt,
                    )
                    latest = outcome
                    beat()
                    try:
                        output = await budget.run(
                            "output", hooks.output(result, context)
                        )
                    except ApplicationError:
                        raise
                    except Exception as hook_exc:
                        raise ApplicationError(
                            f"{type(hooks).__name__}.output raised "
                            f"{type(hook_exc).__name__} for trial "
                            f"{context.config.trial_name}",
                            type="HarborOutputFailed",
                            non_retryable=True,
                        ) from hook_exc
                    # What output returns joins the outcome only if Temporal can
                    # carry it; otherwise the attempt would fail with no outcome
                    # to return.
                    if output is not None:
                        try:
                            payloads = activity.payload_converter().to_payloads(
                                [outcome.model_copy(update={"output": output})]
                            )
                        except Exception as convert_exc:
                            raise ApplicationError(
                                f"{type(hooks).__name__}.output returned a value "
                                "that cannot be converted for trial "
                                f"{context.config.trial_name}",
                                type="HarborOutputFailed",
                                non_retryable=True,
                            ) from convert_exc
                        size = sum(payload.ByteSize() for payload in payloads)
                        if size > _OUTCOME_MAX:
                            raise ApplicationError(
                                f"{type(hooks).__name__}.output returned a value "
                                "that makes the outcome of trial "
                                f"{context.config.trial_name} {size} bytes, more "
                                f"than {_OUTCOME_MAX}",
                                type="HarborOutputFailed",
                                non_retryable=True,
                            )
                        outcome.output = output
                    retried = _retried(result)
                    if retried is not None:
                        try:
                            message = _redact_strings(
                                retried.exception_message, hooks, context
                            )
                        except _RedactFailed as failed:
                            raise failed.failure from None
                        raise ApplicationError(
                            message[:_TEXT_MAX],
                            type=retried.exception_type,
                            next_retry_delay=(
                                timedelta(minutes=2)
                                if retried.exception_type == ApiRateLimitError.__name__
                                else None
                            ),
                        )
                except BaseException as exc:
                    leave = scope.__aexit__(type(exc), exc, exc.__traceback__)
                    if not await budget.run("scope", leave):
                        raise
                else:
                    await budget.run("scope", scope.__aexit__(None, None, None))
            # Type checkers take the scope at its word that it never suppresses
            # an exception; one that does leaves no result to return.
            if outcome is None:  # type: ignore[reportUnnecessaryComparison]
                raise RuntimeError(  # type: ignore[reportUnreachable]
                    f"{type(hooks).__name__}.scope suppressed an exception, "
                    f"so trial {context.config.trial_name} produced no result"
                )
        except asyncio.CancelledError as exc:
            # A cancellation the workflow did not ask for keeps the latest outcome.
            details = activity.cancellation_details()
            if latest is None or (details is not None and details.cancel_requested):
                raise
            raise _failure(
                _interrupted(details, context), latest, hooks, context
            ) from exc
        except Exception as exc:
            raise _failure(exc, latest, hooks, context) from exc
        finally:
            if ticker is not None:
                ticker.cancel()
        return outcome

    @activity.defn(name=AGGREGATE_JOB)
    async def aggregate_job(self, input: AggregateJobInput) -> JobResult:
        """Score each eval's rewards with its dataset's metrics; write ``result.json``.

        Metrics are resolved and computed together because a dataset's metric
        can be a script harbor downloads to this machine and runs with ``uv``.
        """
        with _config_errors():
            metrics = await HarborJobPlan.resolve_metrics(
                input.config, input.task_configs
            )
        result = input.result.model_copy(deep=True)
        for evals_key, rewards in input.rewards.items():
            # Harbor keys evals "agent__model__dataset" (or "agent__dataset")
            # and reads the dataset back off the end the same way.
            dataset = evals_key.split("__")[-1]
            result.stats.evals[evals_key].metrics = [
                await asyncio.to_thread(metric.compute, rewards)
                for metric in metrics[dataset]
            ]

        def write() -> None:
            job_dir = input.config.jobs_dir / input.config.job_name
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "result.json").write_text(
                result.model_dump_json(indent=4, exclude={"trial_results"})
            )

        await asyncio.to_thread(write)
        return result
