"""A trial's Activity timeout follows from the phase timeouts harbor enforces."""

from __future__ import annotations

import sys
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.bridge import BridgeConfig, BridgeKind
from harbor.models.task.config import TaskConfig as TaskDefinition
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    UserAgentConfig,
    VerifierConfig,
)
from harbor.trial.multi_step import MultiStepTrial
from harbor.trial.trial import Trial

from temporalio.client import Client
from temporalio.harbor._activity import trial_timeout
from tests.harbor_fixtures import history, tasks
from tests.harbor_fixtures.workflows import RunOneTrial, TrialArgs, new_worker

_UNTIMED = timedelta(hours=2)


def _harbor(trial: TrialConfig, task: TaskDefinition) -> MultiStepTrial:
    # Harbor's own resolution, without loading a task or starting anything.
    resolved = object.__new__(MultiStepTrial)
    resolved.config = trial
    resolved.task = SimpleNamespace(config=task)  # type: ignore[assignment]
    resolved._init_timeouts()
    return resolved


def _trial(**update: object) -> TrialConfig:
    return TrialConfig.model_validate(
        {"task": TaskConfig(path=Path("/tasks/t")), "trial_name": "t__0", **update}
    )


def _task(toml: str) -> TaskDefinition:
    return TaskDefinition.model_validate_toml('version = "1.0"\n' + toml)


_TIMED = """
[agent]
timeout_sec = 120.0

[verifier]
timeout_sec = 90.0

[environment]
build_timeout_sec = 300.0
"""


@pytest.mark.parametrize(
    "trial",
    [
        _trial(),
        _trial(
            agent=AgentConfig(
                override_timeout_sec=100,
                max_timeout_sec=80,
                override_setup_timeout_sec=30,
            ),
            verifier=VerifierConfig(override_timeout_sec=50, max_timeout_sec=40),
            timeout_multiplier=2.0,
            agent_timeout_multiplier=3.0,
            environment_build_timeout_multiplier=0.5,
        ),
        _trial(agent=AgentConfig(max_timeout_sec=1000), verifier_timeout_multiplier=4),
    ],
    ids=["task", "overrides-caps-multipliers", "cap-above-task"],
)
def test_single_step_sum_matches_harbor(trial: TrialConfig) -> None:
    task = _task(_TIMED)
    harbor = _harbor(trial, task)
    agent = harbor._agent_timeout_sec
    assert agent is not None

    expected = (
        harbor._environment_build_timeout_sec
        + harbor._agent_setup_timeout_sec
        + agent
        + harbor._verifier_timeout_sec
    )
    assert trial_timeout(trial, task, _UNTIMED) == timedelta(seconds=expected)


def test_agent_setup_defaults_to_harbors() -> None:
    trial = _trial()
    task = _task(_TIMED)
    assert (
        _harbor(trial, task)._agent_setup_timeout_sec == Trial._AGENT_SETUP_TIMEOUT_SEC
    )
    assert trial_timeout(trial, task, _UNTIMED) == timedelta(
        seconds=300 + 360 + 120 + 90
    )


def test_untimed_agent_uses_untimed_agent_timeout() -> None:
    trial = _trial(timeout_multiplier=2.0)
    task = _task("[verifier]\ntimeout_sec = 90.0\n")
    assert _harbor(trial, task)._agent_timeout_sec is None

    # It is wall-clock time for the agent, so no multiplier applies to it.
    expected = (600 + 360 + 90) * 2 + _UNTIMED.total_seconds()
    assert trial_timeout(trial, task, _UNTIMED) == timedelta(seconds=expected)
    assert trial_timeout(trial, task, timedelta(minutes=5)) == timedelta(
        seconds=(600 + 360 + 90) * 2 + 300
    )


def test_healthcheck_and_collect_hooks_count() -> None:
    task = _task(
        _TIMED
        + """
[environment.healthcheck]
command = "true"
timeout_sec = 10.5
interval_sec = 2.0
retries = 3
start_period_sec = 5.0
start_interval_sec = 1.0

[[verifier.collect]]
command = "dump-db"
timeout_sec = 20.7

[[verifier.collect]]
command = "dump-cache"
service = "cache"
"""
    )
    # Harbor runs each check for int(timeout_sec): checks every
    # start_interval_sec during the start period, then up to `retries`
    # failures interval_sec apart. Each collect hook runs for int(timeout_sec).
    healthcheck = (5 + 10 + 1) + 3 * (10 + 2)
    collect = 20 + 60
    assert trial_timeout(_trial(), task, _UNTIMED) == timedelta(
        seconds=300 + 360 + 120 + 90 + healthcheck + collect
    )


def test_separate_verifier_environment_builds_again() -> None:
    task = _task(
        _TIMED.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
    )
    trial = _trial(environment_build_timeout_multiplier=2.0)
    harbor = _harbor(trial, task)
    assert trial_timeout(trial, task, _UNTIMED) == timedelta(
        seconds=2 * harbor._environment_build_timeout_sec + 360 + 120 + 90
    )


def test_user_agent_setup_bridge_and_teardown_count() -> None:
    trial = _trial(
        user_agent=UserAgentConfig(
            name="oracle", bridge=BridgeConfig(kind=BridgeKind.ACP)
        ),
        agent=AgentConfig(override_setup_timeout_sec=10),
    )
    # Agent setup, user agent setup, bridge setup, trajectory export and
    # bridge teardown are each bounded by the setup timeout.
    assert trial_timeout(trial, _task(_TIMED), _UNTIMED) == timedelta(
        seconds=300 + 5 * 10 + 120 + 90
    )


_STEPS = """
[agent]
timeout_sec = 120.0

[verifier]
timeout_sec = 90.0

[[verifier.collect]]
command = "dump-db"
timeout_sec = 15.0

[[steps]]
name = "one"
agent.timeout_sec = 30.0
verifier.timeout_sec = 20.0

[[steps.verifier.collect]]
command = "dump-step"
timeout_sec = 5.0

[steps.healthcheck]
command = "true"
timeout_sec = 4.0
interval_sec = 1.0
retries = 2

[[steps]]
name = "two"
verifier.environment_mode = "separate"
"""


@pytest.mark.parametrize(
    "trial",
    [
        _trial(),
        _trial(
            agent=AgentConfig(override_timeout_sec=45),
            verifier=VerifierConfig(override_timeout_sec=25),
            timeout_multiplier=1.5,
        ),
    ],
    ids=["steps", "overrides"],
)
def test_multi_step_sums_each_step(trial: TrialConfig) -> None:
    task = _task(_STEPS)
    assert task.steps is not None
    one, two = task.steps
    harbor = _harbor(trial, task)
    agent_one = harbor._step_agent_timeout_sec(one)
    agent_two = harbor._step_agent_timeout_sec(two)
    verifier_one = harbor._step_verifier_timeout_sec(one)
    verifier_two = harbor._step_verifier_timeout_sec(two)
    assert None not in (agent_one, agent_two, verifier_one, verifier_two)

    build = harbor._environment_build_timeout_sec
    expected = (
        build
        + harbor._agent_setup_timeout_sec
        # Step one: its healthcheck, agent, the task's and its own collect
        # hooks, and its verifier.
        + 2 * (4 + 1)
        + agent_one  # type: ignore[operator]
        + 15
        + 5
        + verifier_one  # type: ignore[operator]
        # Step two: agent, the task's collect hooks, and a verifier in an
        # environment of its own.
        + agent_two  # type: ignore[operator]
        + 15
        + build
        + verifier_two  # type: ignore[operator]
    )
    assert trial_timeout(trial, task, _UNTIMED) == timedelta(seconds=expected)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)
async def test_activity_timeout_defaults_to_the_trial_timeout(
    harbor_client: Client, tmp_path: Path
) -> None:
    config = TrialConfig(
        task=TaskConfig(path=tasks.passing(tmp_path / "tasks")),
        trial_name="answers__0",
        trials_dir=tmp_path / "trials",
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    timeouts = []
    async with new_worker(harbor_client, RunOneTrial) as worker:
        for args in (
            TrialArgs(config=config, timeout=timedelta(minutes=20)),
            TrialArgs(
                config=config,
                timeout=timedelta(minutes=20),
                extra_time=timedelta(minutes=1),
            ),
            TrialArgs(
                config=config,
                timeout=timedelta(minutes=20),
                start_to_close_timeout=timedelta(minutes=7),
            ),
        ):
            handle = await harbor_client.start_workflow(
                RunOneTrial.run,
                args,
                id=f"trial-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            await handle.result()
            [event] = await history.scheduled(handle)
            attributes = event.activity_task_scheduled_event_attributes
            timeouts.append(
                (
                    attributes.start_to_close_timeout.ToTimedelta(),
                    attributes.heartbeat_timeout.ToTimedelta(),
                )
            )

    assert timeouts == [
        (timedelta(minutes=30 + 15), timedelta(minutes=2)),
        (timedelta(minutes=30 + 1), timedelta(minutes=2)),
        (timedelta(minutes=7), timedelta(minutes=2)),
    ]
