"""A worker's TrialHooks take part in each attempt, in a fixed order."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import re
import sys
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import harbor.skills
import harbor.trial.trial
import pytest
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.models.trial.result import ExceptionInfo, StepResult, TrialResult
from harbor.trial.trial import Trial
from pydantic import JsonValue

from temporalio.activity import ActivityCancellationDetails
from temporalio.api.common.v1 import Payload
from temporalio.client import Client, WorkflowFailureError
from temporalio.contrib.pydantic import (
    PydanticPayloadConverter,
    pydantic_data_converter,
)
from temporalio.converter import DefaultFailureConverterWithEncodedAttributes
from temporalio.exceptions import ActivityError, ApplicationError, FailureError
from temporalio.harbor import HarborPlugin, TrialContext, TrialHooks, TrialOutcome
from temporalio.harbor._activity import HarborActivities
from temporalio.harbor._types import RunTrialInput
from temporalio.testing import ActivityEnvironment
from tests.harbor_fixtures import agents, history, tasks
from tests.harbor_fixtures.workflows import (
    RunJob,
    RunOneTrial,
    TrialArgs,
    new_worker,
    run_one,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)


def _with(client: Client, hooks: TrialHooks) -> Client:
    config = client.config()
    config["plugins"] = [HarborPlugin(trial_hooks=hooks)]
    return Client(**config)


def _trial(
    task: Path, trials_dir: Path, agent: AgentConfig | None = None
) -> TrialConfig:
    return TrialConfig(
        task=TaskConfig(path=task),
        trial_name=f"{task.name}__0",
        trials_dir=trials_dir,
        agent=agent or AgentConfig(name="oracle"),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )


class Recording(TrialHooks):
    """Records each hook call, and sends a recorded failure back to Temporal once."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.redacted: set[int] = set()

    @contextlib.asynccontextmanager
    async def scope(self, context: TrialContext) -> AsyncIterator[None]:
        self.events.append(f"enter {context.attempt}")
        try:
            yield
        except BaseException as e:
            self.events.append(f"exit {context.attempt} {type(e).__name__}")
            raise
        self.events.append(f"exit {context.attempt}")

    async def trial_created(self, trial: Trial, context: TrialContext) -> None:
        assert isinstance(trial, Trial)
        self.events.append(f"created {context.attempt}")

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        self.events.append(f"output {context.attempt}")
        if result.exception_info is not None and context.attempt == 1:
            raise ApplicationError("the verifier was flaky", type="TryAgain")
        rollouts = result.agent_result.rollout_details if result.agent_result else None
        return {
            "data": context.data,
            "trial_dir": str(context.trial_dir),
            "rollouts": len(rollouts or []),
        }

    def redact(self, text: str, context: TrialContext) -> str:
        self.redacted.add(context.attempt)
        return text


async def test_hooks_run_in_order_for_every_attempt(
    client: Client, tmp_path: Path
) -> None:
    hooks = Recording()
    config = _trial(tasks.flaky(tmp_path / "tasks", failures=1), tmp_path / "trials")
    outcome, attempts = await run_one(
        _with(client, hooks),
        TrialArgs(config=config, maximum_attempts=2, data={"cell": "c-1"}),
    )

    assert attempts == [2]
    assert outcome.result.exception_info is None
    assert hooks.events == [
        "enter 1",
        "created 1",
        # The first attempt produced a result; output sent it back to Temporal.
        "output 1",
        "exit 1 ApplicationError",
        "enter 2",
        "created 2",
        "output 2",
        "exit 2",
    ]
    assert hooks.redacted == {1, 2}
    assert outcome.output == {
        "data": {"cell": "c-1"},
        "trial_dir": str(tmp_path / "trials" / "flaky__0"),
        "rollouts": 0,
    }


async def test_output_sees_the_complete_result(client: Client, tmp_path: Path) -> None:
    config = _trial(
        tasks.passing(tmp_path / "tasks"),
        tmp_path / "trials",
        AgentConfig(import_path=agents.TRANSCRIPT_AGENT),
    )
    outcome, _ = await run_one(_with(client, Recording()), TrialArgs(config=config))

    # The hook read the rollouts; the workflow got the result without them.
    assert isinstance(outcome.output, dict)
    rollouts = outcome.output["rollouts"]
    assert isinstance(rollouts, int) and rollouts > 0
    assert outcome.result.agent_result is not None
    assert outcome.result.agent_result.rollout_details is None


class Redacting(TrialHooks):
    """Removes the test's paths from everything leaving the worker."""

    def __init__(self, *secrets: str) -> None:
        self.secrets = secrets

    def redact(self, text: str, context: TrialContext) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return text


class RejectMissingReward(Redacting):
    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        exc = result.exception_info
        assert exc is not None
        raise ApplicationError(f"rejected: {exc.exception_message}", non_retryable=True)


@pytest.mark.parametrize("hooks", [Redacting, RejectMissingReward])
async def test_redact_reaches_results_errors_and_details(
    client: Client, tmp_path: Path, hooks: type[Redacting]
) -> None:
    # The recorded failure names the reward file, which is under tmp_path.
    secrets = (str(tmp_path), __file__)
    config = _trial(tasks.no_reward(tmp_path / "tasks"), tmp_path / "trials")
    hooked = _with(client, hooks(*secrets))
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()

    assert outcome.result.exception_info is not None
    assert "[REDACTED]" in outcome.result.exception_info.exception_message
    recorded = await history.recorded_output(handle)
    assert "[REDACTED]" in recorded
    assert all(secret not in recorded for secret in secrets)


async def test_redact_leaves_what_aggregation_reads(
    client: Client, tmp_path: Path
) -> None:
    root = tmp_path / "tasks"
    config = JobConfig(
        job_name="redacted",
        jobs_dir=tmp_path / "jobs",
        tasks=[
            TaskConfig(path=tasks.passing(root), source="private-ds"),
            TaskConfig(path=tasks.no_reward(root), source="private-ds"),
        ],
        agents=[AgentConfig(name="oracle")],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    hooks = Redacting(
        "oracle", "answers", "no-reward", "private-ds", "RewardFileNotFoundError"
    )
    hooked = _with(client, hooks)
    async with new_worker(hooked, RunJob) as worker:
        result = await hooked.execute_workflow(
            RunJob.run,
            config,
            id=f"job-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=90),
        )

    [(evals_key, evals)] = result.stats.evals.items()
    assert evals_key == "oracle__private-ds"
    assert set(evals.exception_stats) == {"RewardFileNotFoundError"}
    assert sorted(t.task_name for t in result.trial_results) == [
        "answers",
        "no-reward",
    ]
    for trial in result.trial_results:
        assert trial.trial_name.startswith(f"{trial.task_name}__")
        assert trial.config.agent.name == "[REDACTED]"


def _cannot_start(trial_dir: Path) -> None:
    try:
        raise KeyError(f"no credentials for {trial_dir}")
    except KeyError as e:
        raise RuntimeError(f"cannot start {trial_dir}") from e


class FailInScope(Redacting):
    @contextlib.asynccontextmanager
    async def scope(self, context: TrialContext) -> AsyncIterator[None]:
        _cannot_start(context.trial_dir)
        yield


@pytest.mark.parametrize(
    ("hooks", "missing", "types"),
    [
        (FailInScope, False, ["RuntimeError", "KeyError"]),
        (Redacting, True, ["FileNotFoundError"]),
    ],
    ids=["scope", "trial-create"],
)
async def test_redact_reaches_errors_without_a_result(
    client: Client,
    tmp_path: Path,
    hooks: type[Redacting],
    missing: bool,
    types: list[str],
) -> None:
    secrets = (str(tmp_path), __file__)
    task = tmp_path / "missing" if missing else tasks.passing(tmp_path / "tasks")
    hooked = _with(client, hooks(*secrets))
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=_trial(task, tmp_path / "trials"), maximum_attempts=1),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError) as failure:
            await handle.result()

    error = failure.value.cause
    assert isinstance(error, ActivityError)
    chain = []
    cause = error.cause
    while isinstance(cause, ApplicationError):
        chain.append(cause.type)
        assert "[REDACTED]" in cause.message
        cause = cause.cause
    assert chain == types
    recorded = await history.recorded_output(handle)
    assert all(secret not in recorded for secret in secrets)


def _encoding(client: Client, hooks: TrialHooks) -> Client:
    config = client.config()
    config["data_converter"] = dataclasses.replace(
        pydantic_data_converter,
        failure_converter_class=DefaultFailureConverterWithEncodedAttributes,
    )
    config["plugins"] = [HarborPlugin(trial_hooks=hooks)]
    return Client(**config)


@pytest.mark.parametrize(
    ("hooks", "levels"),
    [(FailInScope, 2), (RejectMissingReward, 1)],
    ids=["no-result", "result"],
)
async def test_worker_failure_converter_encodes_every_level(
    client: Client, tmp_path: Path, hooks: type[Redacting], levels: int
) -> None:
    secrets = (str(tmp_path), __file__)
    make = tasks.passing if hooks is FailInScope else tasks.no_reward
    config = _trial(make(tmp_path / "tasks"), tmp_path / "trials")
    encoding = _encoding(client, hooks(*secrets))
    async with new_worker(encoding, RunOneTrial) as worker:
        handle = await encoding.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=1),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        if hooks is FailInScope:
            with pytest.raises(WorkflowFailureError):
                await handle.result()
        else:
            # The hook's non-retryable error carried the outcome back.
            outcome = await handle.result()
            assert outcome.result.exception_info is not None
            assert "[REDACTED]" in outcome.result.exception_info.exception_message

    [failure] = await history.activity_failures(handle)
    converter = encoding.data_converter.payload_converter
    nodes = []
    node = failure
    while True:
        nodes.append(node)
        if not node.HasField("cause"):
            break
        node = node.cause
    assert len(nodes) == levels
    for node in nodes:
        # Encoded once, by the worker's converter, at every level.
        assert (node.message, node.stack_trace) == ("Encoded failure", "")
        attributes = converter.from_payloads([node.encoded_attributes])[0]
        assert attributes["message"] != "Encoded failure"
        assert "[REDACTED]" in attributes["message"] + attributes["stack_trace"]
        assert all(
            secret not in attributes["message"] + attributes["stack_trace"]
            for secret in secrets
        )
    recorded = await history.recorded_output(handle)
    assert all(secret not in recorded for secret in secrets)


class RedactRaises(TrialHooks):
    """Cannot redact anything."""

    def redact(self, text: str, context: TrialContext) -> str:
        raise KeyError("no redaction rules loaded")


async def test_failed_redact_sends_none_of_the_error(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(tmp_path / "missing", tmp_path / "trials")
    hooked = _with(client, RedactRaises())
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=1),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError):
            await handle.result()

    [failure] = await history.activity_failures(handle)
    assert failure.application_failure_info.type == "HarborRedactFailed"
    assert failure.application_failure_info.non_retryable
    assert not failure.HasField("cause")
    assert str(tmp_path) not in await history.recorded_output(handle)


class RedactFailsOnRetry(Redacting):
    """Sends each attempt back to Temporal, and cannot redact after the first."""

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        raise ApplicationError("try once more", type="TryAgain")

    def redact(self, text: str, context: TrialContext) -> str:
        if context.attempt > 1:
            raise KeyError("no redaction rules loaded")
        return super().redact(text, context)


async def test_failed_redact_keeps_the_last_outcome(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(tasks.passing(tmp_path / "tasks"), tmp_path / "trials")
    hooked = _with(client, RedactFailsOnRetry(str(tmp_path)))
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(config=config, maximum_attempts=2),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()

    assert outcome.attempt == 1
    assert await history.final_attempts(handle) == [2]
    [failure] = await history.activity_failures(handle)
    assert failure.application_failure_info.type == "HarborRedactFailed"
    assert failure.application_failure_info.non_retryable
    assert not failure.HasField("cause")
    assert str(tmp_path) not in await history.recorded_output(handle)


class SlowToCreate(TrialHooks):
    async def trial_created(self, trial: Trial, context: TrialContext) -> None:
        await asyncio.sleep(10)


async def test_hooks_are_bounded_by_extra_time(client: Client, tmp_path: Path) -> None:
    config = _trial(tasks.passing(tmp_path / "tasks"), tmp_path / "trials")
    hooked = _with(client, SlowToCreate())
    async with new_worker(hooked, RunOneTrial) as worker:
        with pytest.raises(WorkflowFailureError) as failure:
            await hooked.execute_workflow(
                RunOneTrial.run,
                TrialArgs(
                    config=config,
                    extra_time=timedelta(seconds=1),
                    maximum_attempts=1,
                ),
                id=f"trial-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )

    error = failure.value.cause
    assert isinstance(error, ActivityError)
    assert isinstance(error.cause, ApplicationError)
    assert error.cause.type == "HarborHookTimeout"
    assert "SlowToCreate.trial_created" in error.cause.message


class QuickOutput(TrialHooks):
    """Yields to the event loop once in output, and takes no other time."""

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        await asyncio.sleep(0)
        return "on time"


async def test_extra_time_does_not_bound_the_trial(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(tasks.slow(tmp_path / "tasks", 3), tmp_path / "trials")
    hooked = _with(client, QuickOutput())
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(
                config=config, extra_time=timedelta(seconds=2), maximum_attempts=1
            ),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()

    assert outcome.output == "on time"
    assert outcome.result.exception_info is None
    assert await history.activity_failures(handle) == []


class SlowTwice(TrialHooks):
    """Spends 0.7 seconds in each of two hooks."""

    async def trial_created(self, trial: Trial, context: TrialContext) -> None:
        await asyncio.sleep(0.7)

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        await asyncio.sleep(0.7)
        return None


async def test_hooks_share_one_extra_time(client: Client, tmp_path: Path) -> None:
    config = _trial(tasks.passing(tmp_path / "tasks"), tmp_path / "trials")
    hooked = _with(client, SlowTwice())
    async with new_worker(hooked, RunOneTrial) as worker:
        handle = await hooked.start_workflow(
            RunOneTrial.run,
            TrialArgs(
                config=config, extra_time=timedelta(seconds=1), maximum_attempts=1
            ),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        outcome = await handle.result()

    assert outcome.output is None
    [failure] = await history.activity_failures(handle)
    assert failure.application_failure_info.type == "HarborHookTimeout"
    assert "SlowTwice.output" in failure.message


def _environment(client: Client) -> ActivityEnvironment:
    env = ActivityEnvironment(client=client)
    env.info = dataclasses.replace(
        env.info, heartbeat_timeout=timedelta(milliseconds=800)
    )
    env.payload_converter = pydantic_data_converter.payload_converter
    return env


def _input(config: TrialConfig) -> RunTrialInput:
    return RunTrialInput(config=config, extra_time=timedelta(minutes=1))


async def test_scope_sees_cancellation(client: Client, tmp_path: Path) -> None:
    hooks = Recording()
    env = _environment(client)
    config = _trial(tasks.slow(tmp_path / "tasks", 30), tmp_path / "trials")
    running = asyncio.create_task(
        env.run(HarborActivities(hooks=hooks).run_trial, _input(config))
    )
    for _ in range(200):
        if "created 1" in hooks.events:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)
    env.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert hooks.events == ["enter 1", "created 1", "exit 1 CancelledError"]


class Swallow(TrialHooks):
    @contextlib.asynccontextmanager
    async def scope(self, context: TrialContext) -> AsyncIterator[None]:
        with contextlib.suppress(Exception):
            yield


async def test_scope_that_suppresses_fails_the_attempt(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(tmp_path / "missing", tmp_path / "trials")
    with pytest.raises(FailureError, match="Swallow.scope suppressed") as raised:
        await _environment(client).run(
            HarborActivities(hooks=Swallow()).run_trial, _input(config)
        )
    assert raised.value.failure is not None
    assert raised.value.failure.application_failure_info.type == "RuntimeError"


async def test_skills_resolve_off_the_event_loop(
    client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "skills" / "greeting"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: greeting\ndescription: Hi.\n---\n")
    on_event_loop: list[bool] = []

    def resolve(values: list[str]) -> list[Path]:
        resolved: list[Path] = []
        for value in values:
            if value != "org/skills@v1":
                resolved.append(Path(value))
                continue
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                on_event_loop.append(False)
            else:
                on_event_loop.append(True)
            resolved.append(skill)
        return resolved

    monkeypatch.setattr(harbor.skills, "resolve_skill_sources", resolve)
    monkeypatch.setattr(harbor.trial.trial, "resolve_skill_sources", resolve)
    agent = AgentConfig(name="oracle", skills=["org/skills@v1"])
    config = _trial(tasks.passing(tmp_path / "tasks"), tmp_path / "trials", agent)
    outcome = await _environment(client).run(
        HarborActivities(hooks=TrialHooks()).run_trial, _input(config)
    )

    assert on_event_loop == [False]
    assert outcome.result.exception_info is None


_SECRET = "sk-test-0123456789abcdef"


class RedactSecret(TrialHooks):
    def redact(self, text: str, context: TrialContext) -> str:
        return text.replace(_SECRET, "[REDACTED]")


@pytest.mark.parametrize("agent_started", [True, False], ids=["returned", "retried"])
async def test_redact_sees_text_before_it_is_shortened(
    client: Client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_started: bool,
) -> None:
    # The secret runs through each cut: 8,000 characters into the message,
    # and 8,000 from the end of the stack trace.
    message = "m" * 7_995 + _SECRET + "m" * 50
    stack = "t" * 50 + _SECRET + "t" * 7_995
    recorded = ExceptionInfo(
        exception_type="RuntimeError",
        exception_message=message,
        exception_traceback=stack,
        occurred_at=datetime.now(timezone.utc),
    )
    run = Trial.run

    async def run_and_record(trial: Trial) -> TrialResult:
        result = await run(trial)
        return result.model_copy(
            update={
                "exception_info": recorded,
                "step_results": [StepResult(step_name="one", exception_info=recorded)],
                "agent_execution": result.agent_execution if agent_started else None,
            }
        )

    monkeypatch.setattr(Trial, "run", run_and_record)
    config = _trial(tasks.passing(tmp_path / "tasks"), tmp_path / "trials")
    run_trial = HarborActivities(hooks=RedactSecret()).run_trial
    expected_message = message.replace(_SECRET, "[REDACTED]")[:8_000]
    expected_stack = stack.replace(_SECRET, "[REDACTED]")[-8_000:]

    if agent_started:
        outcome = await _environment(client).run(run_trial, _input(config))
        assert outcome.result.step_results is not None
        for exc in (
            outcome.result.exception_info,
            *(step.exception_info for step in outcome.result.step_results),
        ):
            assert exc is not None
            assert exc.exception_message == expected_message
            assert exc.exception_traceback == expected_stack
    else:
        with pytest.raises(FailureError) as raised:
            await _environment(client).run(run_trial, _input(config))
        assert raised.value.failure is not None
        assert raised.value.failure.message == expected_message


class RedactPattern(TrialHooks):
    """Replaces every match of a pattern, wherever it occurs."""

    def __init__(self, pattern: str) -> None:
        self.pattern = re.compile(pattern)

    def redact(self, text: str, context: TrialContext) -> str:
        return self.pattern.sub("[REDACTED]", text)


_HOST = "proxy.corp.internal"


@pytest.mark.parametrize(
    ("pattern", "agent", "free_text", "redacted"),
    [
        pytest.param(
            r"[A-Za-z0-9-]{30,}",
            AgentConfig(name="oracle"),
            ("task_checksum",),
            "[REDACTED]",
            id="long-token",
        ),
        pytest.param(
            r"\d{6,}",
            AgentConfig(name="oracle"),
            ("config", "trial_name"),
            "answers__[REDACTED]",
            id="numeric-id",
        ),
        pytest.param(
            re.escape(_HOST),
            AgentConfig(
                name="oracle",
                extra_allowed_hosts=[_HOST],
                env={"PROXY_URL": f"https://{_HOST}"},
            ),
            ("config", "agent", "env", "PROXY_URL"),
            "https://[REDACTED]",
            id="allowed-host",
        ),
    ],
)
async def test_redact_leaves_values_that_are_not_free_text(
    client: Client,
    tmp_path: Path,
    pattern: str,
    agent: AgentConfig,
    free_text: tuple[str, ...],
    redacted: str,
) -> None:
    # Each pattern also matches a value its field checks: the result's id,
    # its timestamps, or a host the agent is allowed to reach.
    config = _trial(
        tasks.passing(tmp_path / "tasks"), tmp_path / "trials", agent
    ).model_copy(update={"trial_name": "answers__1234567"})
    outcome = await _environment(client).run(
        HarborActivities(hooks=RedactPattern(pattern)).run_trial, _input(config)
    )

    recorded = TrialResult.model_validate_json(
        (tmp_path / "trials" / "answers__1234567" / "result.json").read_text()
    )
    result = outcome.result
    assert result.verifier_result is not None
    assert result.verifier_result.rewards == {"reward": 1.0}
    value: Any = result.model_dump()
    for key in free_text:
        value = value[key]
    assert value == redacted
    assert result.id == recorded.id
    assert result.finished_at == recorded.finished_at
    assert result.config.agent.extra_allowed_hosts == agent.extra_allowed_hosts


class DecodeFailsOnce(PydanticPayloadConverter):
    """Fails the first payload it decodes, as a converter with a fault does."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def from_payloads(
        self, payloads: Sequence[Payload], type_hints: list[type] | None = None
    ) -> list[Any]:
        if not self.failed:
            self.failed = True
            raise RuntimeError("the payload could not be decoded")
        return super().from_payloads(payloads, type_hints)


async def test_converter_error_is_not_blamed_on_redact(
    client: Client, tmp_path: Path
) -> None:
    env = ActivityEnvironment(client=_encoding(client, TrialHooks()))
    env.payload_converter = DecodeFailsOnce()
    config = _trial(tmp_path / "missing", tmp_path / "trials")
    with pytest.raises(FailureError) as raised:
        await env.run(HarborActivities(hooks=TrialHooks()).run_trial, _input(config))

    failure = raised.value.failure
    assert failure is not None
    assert failure.application_failure_info.type == "HarborFailureNotConverted"
    assert failure.application_failure_info.non_retryable
    assert "the payload converter raised RuntimeError" in failure.message
    assert "redact" not in failure.message


class DecodeAlwaysFails(PydanticPayloadConverter):
    """Fails every payload it decodes, as a converter with a fault does."""

    def from_payloads(
        self, payloads: Sequence[Payload], type_hints: list[type] | None = None
    ) -> list[Any]:
        raise RuntimeError("the payload could not be decoded")


async def test_error_the_converter_cannot_decode_fails_with_the_outcome(
    client: Client, tmp_path: Path
) -> None:
    earlier = await _environment(client).run(
        HarborActivities(hooks=TrialHooks()).run_trial,
        _input(_trial(tasks.passing(tmp_path / "tasks"), tmp_path / "trials")),
    )
    env = ActivityEnvironment(client=_encoding(client, TrialHooks()))
    env.info = dataclasses.replace(
        env.info, heartbeat_details=[{"outcome": earlier.model_dump(mode="json")}]
    )
    env.payload_converter = DecodeAlwaysFails()
    config = _trial(tmp_path / "missing", tmp_path / "trials")
    with pytest.raises(FailureError) as raised:
        await env.run(HarborActivities(hooks=TrialHooks()).run_trial, _input(config))

    failure = raised.value.failure
    assert failure is not None
    info = failure.application_failure_info
    assert info.type == "HarborFailureNotConverted"
    assert info.non_retryable
    carried = pydantic_data_converter.payload_converter.from_payloads(
        list(info.details.payloads)
    )
    assert TrialOutcome.model_validate(carried[-1]).result.id == earlier.result.id


class RefusesOutcomes(PydanticPayloadConverter):
    """Cannot convert a trial's outcome, as a converter with a fault does."""

    def to_payloads(self, values: Sequence[Any]) -> list[Payload]:
        if any(isinstance(value, TrialOutcome) for value in values):
            raise RuntimeError("the outcome could not be converted")
        return super().to_payloads(values)


async def test_outcome_that_cannot_be_converted_is_left_off_the_error(
    client: Client, tmp_path: Path
) -> None:
    outcome = await _environment(client).run(
        HarborActivities(hooks=TrialHooks()).run_trial,
        _input(_trial(tasks.passing(tmp_path / "tasks"), tmp_path / "trials")),
    )
    env = _environment(client)
    env.info = dataclasses.replace(
        env.info, heartbeat_details=[{"outcome": outcome.model_dump(mode="json")}]
    )
    env.payload_converter = RefusesOutcomes()
    config = _trial(tmp_path / "missing", tmp_path / "trials")
    with pytest.raises(FailureError) as raised:
        await env.run(HarborActivities(hooks=TrialHooks()).run_trial, _input(config))

    failure = raised.value.failure
    assert failure is not None
    assert failure.application_failure_info.type == "FileNotFoundError"
    assert not failure.application_failure_info.details.payloads


class RedactReturnsNone(TrialHooks):
    """Returns no text, against the type redact declares."""

    def redact(self, text: str, context: TrialContext) -> str:
        return cast(Any, None)


async def test_redact_that_returns_no_text_fails_the_trial(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(tasks.passing(tmp_path / "tasks"), tmp_path / "trials")
    with pytest.raises(FailureError) as raised:
        await _environment(client).run(
            HarborActivities(hooks=RedactReturnsNone()).run_trial, _input(config)
        )

    failure = raised.value.failure
    assert failure is not None
    assert failure.application_failure_info.type == "HarborRedactFailed"
    assert failure.application_failure_info.non_retryable
    assert failure.message == (
        f"RedactReturnsNone.redact returned NoneType, not str, for trial "
        f"{config.trial_name}"
    )


class RedactFailsOnSecondSight(TrialHooks):
    """Passes text through, and fails on ``message`` the second time it sees it."""

    def __init__(self, message: str, raises: bool) -> None:
        self.message = message
        self.raises = raises
        self.seen = 0

    def redact(self, text: str, context: TrialContext) -> str:
        if text == self.message:
            self.seen += 1
            if self.seen > 1 and self.raises:
                raise KeyError("no redaction rules loaded")
            if self.seen > 1:
                return cast(Any, None)
        return text


@pytest.mark.parametrize(
    ("raises", "failed"),
    [
        pytest.param(True, "raised KeyError", id="raises"),
        pytest.param(False, "returned NoneType, not str,", id="returns-no-text"),
    ],
)
async def test_retried_error_fails_as_redact_does(
    client: Client, tmp_path: Path, raises: bool, failed: str
) -> None:
    # redact sees the retried error's message twice: in the result, then as
    # the message of the error Temporal retries the trial for.
    message = "the agent's run ended in ApiOverloadedError"
    agent = AgentConfig(
        import_path=agents.FAILING_AGENT, kwargs={"error": "ApiOverloadedError"}
    )
    config = _trial(tasks.passing(tmp_path / "tasks"), tmp_path / "trials", agent)
    hooks = RedactFailsOnSecondSight(message, raises)
    with pytest.raises(FailureError) as raised:
        await _environment(client).run(
            HarborActivities(hooks=hooks).run_trial, _input(config)
        )

    assert hooks.seen == 2
    failure = raised.value.failure
    assert failure is not None
    assert failure.application_failure_info.type == "HarborRedactFailed"
    assert failure.application_failure_info.non_retryable
    assert failure.message == (
        f"RedactFailsOnSecondSight.redact {failed} for trial {config.trial_name}"
    )


@pytest.mark.parametrize(
    ("details", "error"),
    [
        pytest.param(
            ActivityCancellationDetails(cancel_requested=True), None, id="requested"
        ),
        pytest.param(
            ActivityCancellationDetails(paused=True), "ActivityPause", id="paused"
        ),
        pytest.param(
            ActivityCancellationDetails(reset=True), "ActivityReset", id="reset"
        ),
        pytest.param(ActivityCancellationDetails(), "CancelledError", id="unrequested"),
    ],
)
async def test_cancellation_keeps_the_outcome_unless_the_workflow_asked(
    client: Client,
    tmp_path: Path,
    details: ActivityCancellationDetails,
    error: str | None,
) -> None:
    earlier = await _environment(client).run(
        HarborActivities(hooks=TrialHooks()).run_trial,
        _input(_trial(tasks.passing(tmp_path / "tasks"), tmp_path / "trials")),
    )
    env = _environment(client)
    env.info = dataclasses.replace(
        env.info, heartbeat_details=[{"outcome": earlier.model_dump(mode="json")}]
    )
    hooks = Recording()
    config = _trial(tasks.slow(tmp_path / "tasks", 30), tmp_path / "trials")
    running = asyncio.create_task(
        env.run(HarborActivities(hooks=hooks).run_trial, _input(config))
    )
    for _ in range(200):
        if "created 1" in hooks.events:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)
    env.cancel(details)

    if error is None:
        with pytest.raises(asyncio.CancelledError):
            await running
    else:
        with pytest.raises(FailureError) as raised:
            await running
        failure = raised.value.failure
        assert failure is not None
        info = failure.application_failure_info
        assert info.type == error
        assert not info.non_retryable
        carried = pydantic_data_converter.payload_converter.from_payloads(
            list(info.details.payloads)
        )
        assert TrialOutcome.model_validate(carried[-1]).result.id == earlier.result.id
    assert hooks.events == ["enter 1", "created 1", "exit 1 CancelledError"]
