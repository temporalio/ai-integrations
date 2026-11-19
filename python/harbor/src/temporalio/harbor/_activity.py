"""The Activities a harbor job is made of."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import json
import math
import shutil
import threading
import time
import weakref
from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Generator,
    Iterator,
)
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import harbor.skills
from harbor.agents.factory import AgentFactory, agent_config_kwargs
from harbor.agents.installed.base import ApiRateLimitError
from harbor.environments.factory import EnvironmentFactory
from harbor.job_plan import JobPlan as HarborJobPlan
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.job.lock import LOCK_FILENAME, JobLock, build_job_lock
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
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.error_policy import FailureCause
from harbor.models.trial.result import ExceptionInfo, StepResult, TrialResult
from harbor.registry.client.base import EmptyDatasetError, UnavailableDatasetTasksError
from harbor.tasks.client import TaskDownloadResult, TaskIdType
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.trial import Trial
from harbor.verifier.verifier import VerifierOutputParseError
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
)

_T = TypeVar("_T")

# Harbor's aggregation reads an exception's type, never its text.
_TEXT_MAX = 8_000

# Temporal's default limit on a payload is 2 MiB; leave room for what a
# payload codec adds.
_OUTCOME_MAX = 2 * 1024 * 1024 - 64 * 1024

# Harbor's, and litellm's.
_RATE_LIMITED = frozenset({ApiRateLimitError.__name__, "RateLimitError"})

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


def _raised(result: TrialResult) -> ExceptionInfo | None:
    """The trial's exception, if the attempt raises it.

    As harbor retries a trial, only the trial's own exception counts, not a
    step's. The attempt raises it when no agent started, or when harbor's
    error policy retries it, unless harbor cannot classify it. Harbor sets
    ``agent_execution`` as an agent starts: on the result, or on each step's
    result for a multi-step trial. Temporal runs the trial again unless
    harbor rejected the trial's configuration as it started
    (:func:`_rejected`).
    """
    exc = result.exception_info
    if exc is None:
        return None
    steps = result.step_results or []
    if all(r.agent_execution is None for r in [result, *steps]):
        return exc
    if exc.policy.retry and exc.policy.cause != FailureCause.UNCLASSIFIED:
        return exc
    return None


def _rejected(result: TrialResult) -> bool:
    """Whether harbor rejected the trial's configuration as the trial started.

    Before it starts the environment, harbor checks that the agent can honor
    what the trial configures for it, such as skills, MCP servers and a
    trajectory to load, and raises ``ValueError`` when it cannot, as
    ``Trial.create`` does for a misconfigured trial.
    """
    exc = result.exception_info
    return (
        exc is not None
        and exc.exception_type == ValueError.__name__
        and result.environment_setup is None
    )


def _finite(data: dict[str, Any], result: TrialResult, occurred_at: datetime) -> None:
    """Record in ``data``, a dump of ``result``, each reward that is not finite.

    Harbor writes such a reward as ``null``. Wherever ``result`` has one, the
    verifier's result is dropped and, as harbor's own verifier records such a
    reward, a ``VerifierOutputParseError`` recorded instead.
    """
    pairs: list[tuple[dict[str, Any], TrialResult | StepResult]] = [
        (data, result),
        *zip(data.get("step_results") or [], result.step_results or []),
    ]
    for item, source in pairs:
        verifier = source.verifier_result
        rewards = verifier.rewards if verifier is not None else None
        if rewards and not all(math.isfinite(r) for r in rewards.values()):
            item["verifier_result"] = None
            item["exception_info"] = item.get("exception_info") or ExceptionInfo(
                exception_type=VerifierOutputParseError.__name__,
                exception_message=f"Non-finite reward in {rewards}",
                exception_traceback="",
                occurred_at=occurred_at,
            ).model_dump(mode="json")


def _strings(
    value: Any, path: tuple[Any, ...] = ()
) -> Iterator[tuple[tuple[Any, ...], str]]:
    """Each string in ``value``, with its path."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, (*path, index))
    elif isinstance(value, str):
        yield path, value


def _at(data: Any, path: tuple[Any, ...]) -> Any:
    """The value at ``path`` in ``data``, or ``None`` if there is none."""
    for key in path:
        if isinstance(data, dict):
            data = data.get(key)
        elif isinstance(data, list) and isinstance(key, int) and key < len(data):
            data = data[key]
        else:
            return None
    return data


def _recorded(path: Path, result: TrialResult) -> TrialResult:
    """The result harbor wrote to ``path`` once it removed the trial's secrets.

    A reward that is not finite is recorded as :func:`_finite` describes. The
    text aggregation reads keeps its value in ``result``, as :func:`_redacted`
    keeps it, since harbor removes a secret wherever its text appears in the
    file, names included. That can also leave the file invalid, for example
    where the secret is part of a UUID, a timestamp or a key. Then the result
    is ``result``, less what :func:`_strip` drops, with each other string
    taken from the file where the file holds one in its place, and otherwise
    replaced with ``[REDACTED]``. As :func:`_redacted` does, each change is
    kept only where the result still validates with it.
    """
    occurred_at = datetime.now(timezone.utc)
    try:
        recorded: Any = json.loads(path.read_text())
    except ValueError:
        recorded = None
    kept: Any = _strip(result).model_dump(mode="json")
    _finite(kept, result, occurred_at)
    if isinstance(recorded, dict):
        _finite(recorded, result, occurred_at)
        for where in _AGGREGATED:
            parent = _at(recorded, where[:-1])
            if isinstance(parent, dict) and where[-1] in parent:
                parent[where[-1]] = _at(kept, where)
        try:
            return TrialResult.model_validate(recorded)
        except ValidationError:
            pass
    for where, text in list(_strings(kept)):
        if where in _AGGREGATED:
            continue
        found = _at(recorded, where)
        scrubbed = found if isinstance(found, str) else "[REDACTED]"
        if scrubbed == text:
            continue
        candidate = _replaced(kept, where, scrubbed)
        try:
            TrialResult.model_validate(candidate)
        except ValidationError:
            continue
        kept = candidate
    return TrialResult.model_validate(kept)


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
    """The phase timeouts harbor 0.25's ``Trial`` enforces for a trial, added up.

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
    setup_sec = trial.agent.override_setup_timeout_sec
    if setup_sec is None:
        agent_class = AgentFactory.get_agent_class_from_config(trial.agent)
        setup_sec = agent_class.DEFAULT_SETUP_TIMEOUT_SEC
    setup = resolve(setup_sec, trial.agent_setup_timeout_multiplier)
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


def _not_finite(value: Any, path: tuple[str, ...] = ()) -> Iterator[str]:
    """The paths of the times in ``value`` that are infinite or NaN."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _not_finite(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _not_finite(item, (*path, str(index)))
    elif (
        isinstance(value, float)
        and not math.isfinite(value)
        and path[-1].endswith("_sec")
    ):
        yield ".".join(path)


async def _on_own_loop(make: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """Run harbor's async code on a fresh event loop in a daemon thread.

    Cancelling the caller cancels that code at its next ``await``.
    """
    done: concurrent.futures.Future[_T] = concurrent.futures.Future()
    context = contextvars.copy_context()  # keeps activity.logger and tracing
    cancelled = threading.Event()
    inner: list[asyncio.Future[_T]] = []

    async def main() -> _T:
        task = asyncio.ensure_future(make())
        inner.append(task)
        if cancelled.is_set():
            task.cancel()
        return await task

    def run() -> None:
        if not done.set_running_or_notify_cancel():
            return
        try:
            done.set_result(context.run(asyncio.run, main()))
        except BaseException as exc:
            done.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    try:
        return await asyncio.wrap_future(done)
    except asyncio.CancelledError:
        cancelled.set()
        for task in inner:
            # The loop is closed if harbor's code finished meanwhile.
            with contextlib.suppress(RuntimeError):
                task.get_loop().call_soon_threadsafe(task.cancel)
        raise


def _leaves(error: Exception) -> Iterator[Exception]:
    if isinstance(error, ExceptionGroup):
        for member in error.exceptions:
            yield from _leaves(member)
    else:
        yield error


@contextlib.contextmanager
def _config_errors(*, lookups: bool = False) -> Generator[None, None, None]:
    """Fail without a retry when harbor rejects the job's configuration.

    With ``lookups``, a ``ValueError`` raised while handling another error is
    retried: harbor's registry clients raise one when a lookup fails, and a
    bare one for a dataset or task that does not exist. Harbor downloads tasks
    in a ``TaskGroup``: a group fails without a retry if any error in it,
    nested groups included, would.
    """
    try:
        yield
    except Exception as exc:
        rejected = [
            error
            for error in _leaves(exc)
            if isinstance(error, (EmptyDatasetError, UnavailableDatasetTasksError))
            or (
                isinstance(error, ValueError)
                and not (lookups and error.__context__ is not None)
            )
        ]
        if rejected:
            raise ApplicationError(
                "; ".join(map(str, rejected)),
                type=type(rejected[0]).__name__,
                non_retryable=True,
            ) from exc
        raise


def _check_options(agent: AgentConfig) -> None:
    # As harbor's AgentFactory.run_preflight and BaseAgent.parse_options check
    # them, without the worker's environment.
    agent_class = AgentFactory.get_agent_class_from_config(agent)
    kwargs = agent_config_kwargs(agent)
    for name in ("model_api_key_name", "model_api_format"):
        kwargs.pop(name, None)
    problems: list[str] = []
    if agent_class.options_model is not None:
        try:
            agent_class.options_model.model_validate(kwargs)
        except ValidationError as exc:
            # An option missing here may come from the worker's environment.
            problems = [
                "unknown option '{}'".format(".".join(map(str, e["loc"])))
                if e["type"] == "extra_forbidden"
                else "{}: {}".format(".".join(map(str, e["loc"])), e["msg"])
                for e in exc.errors(include_url=False, include_input=False)
                if e["type"] != "missing"
            ]
    problems += [
        f"{name} is not supported"
        for name in ("base_url", "reasoning_effort")
        if kwargs.get(name) is not None and not getattr(agent_class.capabilities, name)
    ]
    if problems:
        raise ValueError(
            f"invalid options for agent {agent.name or agent.import_path!r}: "
            + "; ".join(problems)
        )


def _pinned(
    task: TaskConfig, downloads: dict[TaskIdType, TaskDownloadResult]
) -> TaskConfig:
    download = downloads[task.get_task_id()]
    commit = download.resolved_git_commit_id
    if task.is_git_task() and commit is not None:
        return task.model_copy(update={"git_commit_id": commit})
    digest = download.content_hash
    if task.is_package_task() and task.ref is None and digest is not None:
        ref = digest if digest.startswith("sha256:") else f"sha256:{digest}"
        return task.model_copy(update={"ref": ref})
    return task


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


class _HookTimeout(ApplicationError):
    """A hook ran past what was left of the attempt's ``extra_time``."""


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
            raise _HookTimeout(
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


def _failure(exc: Exception, hooks: TrialHooks, context: TrialContext) -> Exception:
    """``exc`` converted and redacted."""
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
    # The worker passes this through its failure converter once more, which
    # encodes the top level's message and stack trace again if it encodes
    # them at all, so the top level goes back as plain text.
    try:
        plain(failure)
    except Exception as error:
        failure = replaced(error)
        with contextlib.suppress(Exception):
            plain(failure)
    return FailureError(failure.message, failure=failure)


class HarborActivities:
    """The Activities registered by :class:`temporalio.harbor.HarborPlugin`."""

    def __init__(self, *, hooks: TrialHooks) -> None:
        """Run trials with ``hooks`` taking part."""
        self._hooks = hooks
        self._attempts: weakref.WeakValueDictionary[Path, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._tasks: weakref.WeakValueDictionary[TaskIdType, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    @activity.defn(name=PLAN_JOB)
    async def plan_job(self, input: PlanJobInput) -> PlanJobResult:
        """Resolve a job's tasks and check its configuration, as harbor's ``Job`` does.

        Writes the job's ``config.json`` and ``lock.json`` under
        ``jobs_dir/job_name``, and works out each trial's timeout from its
        task's ``task.toml``. Fails without retrying if that directory already
        has a ``config.json`` or ``lock.json`` that differs from this job's,
        as harbor does.
        """
        timeout = activity.info().heartbeat_timeout

        async def heartbeat(interval: float) -> None:
            while True:
                activity.heartbeat()
                await asyncio.sleep(interval)

        ticker = None
        if timeout:
            ticker = asyncio.create_task(heartbeat(timeout.total_seconds() / 4))
        try:
            return await self._plan(input)
        finally:
            if ticker is not None:
                ticker.cancel()

    async def _plan(self, input: PlanJobInput) -> PlanJobResult:
        config = input.config.model_copy(deep=True)
        with _config_errors():
            for agent in [*config.agents, *filter(None, [config.user_agent])]:
                _check_options(agent)
            await asyncio.to_thread(HarborJobPlan.resolve_agent_skills, config)

        async def resolve() -> tuple[
            list[TaskConfig], dict[TaskIdType, TaskDownloadResult]
        ]:
            task_configs = await HarborJobPlan.resolve_task_configs(config)
            # Configuration, not a lookup: never retried.
            with _config_errors():
                EnvironmentFactory.validate_resource_policies(config.environment)
            await HarborJobPlan.resolve_metrics(config, task_configs)
            return task_configs, await HarborJobPlan.cache_tasks(task_configs)

        with _config_errors(lookups=True):
            task_configs, downloads = await _on_own_loop(resolve)
        # config.json records each git task's commit and each package task's
        # content hash, as it records a package dataset's, so harbor job resume
        # matches the trials.
        config.tasks = [_pinned(task, downloads) for task in config.tasks]
        trials = HarborJobPlan.build_trial_configs(config, task_configs, job_id=uuid4())

        def read_tasks() -> list[TaskDefinition]:
            return [
                TaskDefinition.model_validate_toml(
                    TaskPaths(
                        downloads[task.get_task_id()].path
                    ).config_path.read_text()
                )
                for task in task_configs
            ]

        with _config_errors():
            definitions = await asyncio.to_thread(read_tasks)
        for task, definition in zip(task_configs, definitions):
            fields = list(_not_finite(definition.model_dump()))
            if fields:
                raise ApplicationError(
                    f"task {task.get_task_id().get_name()} sets "
                    f"{', '.join(fields)} to a value that is not finite",
                    type="HarborInvalidTimeout",
                    non_retryable=True,
                )
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
        with _config_errors():
            lock = await asyncio.to_thread(
                build_job_lock,
                config=config,
                trial_configs=trials,
                task_download_results=downloads,
            )

        def write() -> None:
            job_dir = config.jobs_dir / config.job_name
            job_dir.mkdir(parents=True, exist_ok=True)
            config_path = job_dir / "config.json"
            lock_path = job_dir / LOCK_FILENAME
            # As harbor does. An equal one may be an earlier attempt's.
            if config_path.exists():
                try:
                    existing = JobConfig.model_validate_json(config_path.read_text())
                except ValueError:
                    existing = None
                if (
                    existing is None
                    or existing.model_copy(update={"jobs_dir": config.jobs_dir})
                    != config
                ):
                    raise ApplicationError(
                        f"Job directory {job_dir} already exists with a "
                        "different config.",
                        type="FileExistsError",
                        non_retryable=True,
                    )
            if lock_path.exists():
                try:
                    existing_lock = JobLock.model_validate_json(lock_path.read_text())
                except ValueError:
                    existing_lock = None
                if existing_lock is None or existing_lock != lock:
                    raise ApplicationError(
                        f"Job directory {job_dir} already has a lock.json that "
                        "does not match the resolved job lock.",
                        type="FileExistsError",
                        non_retryable=True,
                    )
                lock.created_at = existing_lock.created_at
                lock.harbor = existing_lock.harbor
            config_path.write_text(
                config.model_dump_json(indent=4, exclude_defaults=True)
            )
            lock_path.write_text(lock.model_dump_json(indent=4, exclude_none=True))

        await asyncio.to_thread(write)
        return PlanJobResult(
            task_configs=[_pinned(task, downloads) for task in task_configs],
            task_names=[
                task.get_task_id().get_name().split("/")[-1] for task in task_configs
            ],
            timeouts=timeouts,
            datasets=config.datasets,
        )

    @activity.defn(name=RUN_TRIAL)
    async def run_trial(self, input: RunTrialInput) -> TrialOutcome:
        """Run one attempt of a harbor trial.

        :func:`temporalio.harbor.execute_trial` describes when it returns the
        trial's result and when it raises.
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

        def beat() -> None:
            activity.heartbeat(
                {"phase": phase, "elapsed_sec": round(time.monotonic() - started)}
            )

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
            lock = self._attempts.setdefault(context.trial_dir, asyncio.Lock())
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
                    # Harbor fetches a git or package task into a directory
                    # every trial of it on this worker reads.
                    fetching = self._tasks.setdefault(
                        context.config.task.get_task_id(), asyncio.Lock()
                    )
                    with _config_errors():
                        await asyncio.to_thread(_resolve_skills, context.config)
                        async with fetching:
                            trial = await Trial.create(context.config)
                    for event in TrialEvent:
                        trial.add_hook(event, track(event))
                    try:
                        await budget.run(
                            "trial_created", hooks.trial_created(trial, context)
                        )
                    except BaseException:
                        # Only trial.run() closes the trial.log Trial.create opened.
                        trial._close_logger_handler()
                        raise
                    result = await trial.run()
                    recorded = await asyncio.to_thread(
                        _recorded, trial.paths.result_path, result
                    )
                    # Redacted in full, then shortened: shortening first could
                    # cut a secret in two and leave a part redact cannot match.
                    outcome = TrialOutcome(
                        result=_cap(_redacted(_strip(recorded), hooks, context)),
                        attempt=info.attempt,
                    )
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
                    # carry it.
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
                    raised = _raised(recorded)
                    # A trial harbor rejected as it started gets no retry.
                    if raised is not None and not _rejected(recorded):
                        try:
                            message = _redact_strings(
                                raised.exception_message, hooks, context
                            )
                        except _RedactFailed as failed:
                            raise failed.failure from None
                        # execute_trial returns the outcome if Temporal stops retrying.
                        raise ApplicationError(
                            message[:_TEXT_MAX],
                            outcome,
                            type=raised.exception_type,
                            next_retry_delay=(
                                timedelta(minutes=2)
                                if raised.exception_type in _RATE_LIMITED
                                else None
                            ),
                        )
                except BaseException as exc:
                    leave = scope.__aexit__(type(exc), exc, exc.__traceback__)
                    try:
                        suppressed = await budget.run("scope", leave)
                    except _HookTimeout:
                        # The error already leaving the scope is the attempt's.
                        suppressed = False
                    if not suppressed:
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
        except Exception as exc:
            raise _failure(exc, hooks, context) from exc
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
        with _config_errors(lookups=True):
            metrics = await _on_own_loop(
                partial(HarborJobPlan.resolve_metrics, input.config, input.task_configs)
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
