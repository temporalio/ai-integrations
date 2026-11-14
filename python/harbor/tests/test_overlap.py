"""On one worker, attempts of one trial take turns in its directory, and trials
of one task take turns fetching it."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import sys
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial

from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.harbor import TrialContext, TrialHooks
from temporalio.harbor._activity import HarborActivities
from temporalio.harbor._types import RunTrialInput
from temporalio.testing import ActivityEnvironment
from tests.harbor_fixtures import tasks

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)


class Recording(TrialHooks):
    def __init__(self) -> None:
        self.events: list[str] = []

    @contextlib.asynccontextmanager
    async def scope(self, context: TrialContext) -> AsyncIterator[None]:
        self.events.append(f"enter {context.attempt}")
        try:
            yield
        finally:
            self.events.append(f"exit {context.attempt}")

    async def trial_created(self, trial: Trial, context: TrialContext) -> None:
        self.events.append(f"created {context.attempt}")


def _environment(attempt: int) -> ActivityEnvironment:
    env = ActivityEnvironment()
    env.info = dataclasses.replace(
        env.info, attempt=attempt, heartbeat_timeout=timedelta(milliseconds=800)
    )
    env.payload_converter = pydantic_data_converter.payload_converter
    return env


async def test_later_attempt_waits_then_starts_clean(tmp_path: Path) -> None:
    hooks = Recording()
    activities = HarborActivities(hooks=hooks)
    trials_dir = tmp_path / "trials"
    config = TrialConfig(
        task=TaskConfig(path=tasks.slow(tmp_path / "tasks", 2)),
        trial_name="slow__0",
        trials_dir=trials_dir,
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    input = RunTrialInput(config=config, extra_time=timedelta(minutes=1))

    first = asyncio.create_task(_environment(1).run(activities.run_trial, input))
    for _ in range(200):
        if "created 1" in hooks.events:
            break
        await asyncio.sleep(0.05)
    # Something the first attempt left behind.
    (trials_dir / "slow__0" / "left-behind.txt").write_text("first attempt\n")

    # Temporal started a second attempt, say after the first one's heartbeat
    # timed out, while the first is still running here.
    second_env = _environment(2)
    waiting: list[dict[str, Any]] = []
    second_env.on_heartbeat = lambda *details: waiting.append(details[0])
    second = asyncio.create_task(second_env.run(activities.run_trial, input))

    first_outcome = await first
    second_outcome = await second

    assert hooks.events == [
        "enter 1",
        "created 1",
        "exit 1",
        "enter 2",
        "created 2",
        "exit 2",
    ]
    assert any(detail["phase"] == "waiting" for detail in waiting)
    for outcome in (first_outcome, second_outcome):
        assert outcome.result.verifier_result is not None
        assert outcome.result.verifier_result.rewards == {"reward": 1.0}
    assert [p.name for p in trials_dir.iterdir()] == ["slow__0"]
    assert not (trials_dir / "slow__0" / "left-behind.txt").exists()
    assert (trials_dir / "slow__0" / "result.json").exists()


def _held(tmp_path: Path, trials_dir: Path, name: str) -> TrialConfig:
    task = tmp_path / "tasks" / "held"
    if not task.exists():
        tasks.held(task.parent, tmp_path / "waiting", tmp_path / "release")
    return TrialConfig(
        task=TaskConfig(path=task),
        trial_name=name,
        trials_dir=trials_dir,
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )


async def test_same_named_trials_of_two_jobs_run_at_once(tmp_path: Path) -> None:
    hooks = Recording()
    activities = HarborActivities(hooks=hooks)
    running = [
        asyncio.create_task(
            _environment(1).run(
                activities.run_trial,
                RunTrialInput(
                    config=_held(tmp_path, tmp_path / job, "held__0"),
                    extra_time=timedelta(minutes=1),
                ),
            )
        )
        for job in ("one", "two")
    ]
    for _ in range(200):
        if hooks.events.count("created 1") == 2:
            break
        await asyncio.sleep(0.05)
    together = hooks.events.count("created 1") == 2
    (tmp_path / "release").touch()
    for outcome in await asyncio.gather(*running):
        assert outcome.result.verifier_result is not None
        assert outcome.result.verifier_result.rewards == {"reward": 1.0}

    assert together


async def test_trials_of_one_task_take_turns_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create = Trial.create.__func__
    creating: list[str] = []
    most = 0

    async def one_at_a_time(cls: type[Trial], config: TrialConfig) -> Trial:
        nonlocal most
        creating.append(config.trial_name)
        most = max(most, len(creating))
        try:
            await asyncio.sleep(0.2)
            return await create(cls, config)
        finally:
            creating.remove(config.trial_name)

    monkeypatch.setattr(Trial, "create", classmethod(one_at_a_time))
    (tmp_path / "release").touch()
    activities = HarborActivities(hooks=TrialHooks())
    await asyncio.gather(
        *(
            _environment(1).run(
                activities.run_trial,
                RunTrialInput(
                    config=_held(tmp_path, tmp_path / "trials", f"held__{i}"),
                    extra_time=timedelta(minutes=1),
                ),
            )
            for i in range(2)
        )
    )

    assert most == 1
