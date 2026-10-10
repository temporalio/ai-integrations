"""A worker's TrialHooks take part in each attempt, in a fixed order."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from datetime import timedelta
from pathlib import Path

import pytest
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import JsonValue

from temporalio.client import Client
from temporalio.harbor import HarborPlugin, TrialContext, TrialHooks, TrialRetry
from temporalio.harbor._activity import HarborActivities
from temporalio.harbor._types import RunTrialInput
from temporalio.testing import ActivityEnvironment
from tests.harbor_fixtures import agents, tasks
from tests.harbor_fixtures.workflows import TrialArgs, run_one

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)

_INTERVAL = timedelta(milliseconds=200)
_FAST = RetryConfig(
    max_retries=1, exclude_exceptions=set(), min_wait_sec=0.1, max_wait_sec=0.1
)


def _with(client: Client, hooks: TrialHooks) -> Client:
    config = client.config()
    config["plugins"] = [HarborPlugin(heartbeat_interval=_INTERVAL, trial_hooks=hooks)]
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
    """Records each hook call, and hands back what it saw."""

    def __init__(self) -> None:
        self.events: list[str] = []

    @contextlib.asynccontextmanager
    async def _scope(self, context: TrialContext) -> AsyncIterator[None]:
        self.events.append(f"enter {context.attempt}")
        try:
            yield
        except BaseException as e:
            self.events.append(f"exit {context.attempt} {type(e).__name__}")
            raise
        self.events.append(f"exit {context.attempt}")

    def scope(self, context: TrialContext) -> AbstractAsyncContextManager[None]:
        return self._scope(context)

    async def trial_created(self, trial: Trial, context: TrialContext) -> None:
        assert isinstance(trial, Trial)
        self.events.append(f"created {context.attempt}")

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        self.events.append(f"output {context.attempt}")
        rollouts = result.agent_result.rollout_details if result.agent_result else None
        return {
            "data": context.data,
            "trial_dir": str(context.trial_dir),
            "rollouts": len(rollouts or []),
        }


async def test_hooks_run_in_order_and_scope_sees_retry(
    client: Client, tmp_path: Path
) -> None:
    hooks = Recording()
    config = _trial(tasks.flaky(tmp_path / "tasks", failures=1), tmp_path / "trials")
    outcome, attempts = await run_one(
        _with(client, hooks),
        TrialArgs(config=config, retry=_FAST, data={"cell": "c-1"}),
    )

    assert attempts == [2]
    assert outcome.result.exception_info is None
    assert hooks.events == [
        "enter 1",
        "created 1",
        # The retry leaves the scope as an exception, so cleanup sees it.
        "exit 1 ApplicationError",
        "enter 2",
        "created 2",
        "output 2",
        "exit 2",
    ]
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


class RetryMissingReward(TrialHooks):
    """Retry one failure harbor excludes; defer to harbor for the rest."""

    def __init__(self) -> None:
        self.deferred: list[str] = []

    def retry(self, result: TrialResult, context: TrialContext) -> TrialRetry | None:
        exc = result.exception_info
        assert exc is not None
        if exc.exception_type == "RewardFileNotFoundError" and context.attempt < 2:
            return TrialRetry(delay=timedelta(milliseconds=100))
        self.deferred.append(exc.exception_type)
        return super().retry(result, context)


async def test_retry_override_composes_with_harbors_decision(
    client: Client, tmp_path: Path
) -> None:
    hooks = RetryMissingReward()
    config = _trial(tasks.no_reward(tmp_path / "tasks"), tmp_path / "trials")
    # Harbor's default configuration excludes RewardFileNotFoundError.
    outcome, attempts = await run_one(_with(client, hooks), TrialArgs(config=config))

    assert attempts == [2]
    assert outcome.result.exception_info is not None
    assert outcome.result.exception_info.exception_type == "RewardFileNotFoundError"
    # The second attempt fell through to harbor, which declined.
    assert hooks.deferred == ["RewardFileNotFoundError"]
    assert (tmp_path / "trials" / "no-reward__0.attempt-1").is_dir()


class AlwaysRetry(TrialHooks):
    def retry(self, result: TrialResult, context: TrialContext) -> TrialRetry | None:
        return TrialRetry(delay=timedelta(milliseconds=100))


async def test_last_attempt_returns_result_instead_of_retrying(
    client: Client, tmp_path: Path
) -> None:
    config = _trial(tasks.no_reward(tmp_path / "tasks"), tmp_path / "trials")
    outcome, attempts = await run_one(
        _with(client, AlwaysRetry()), TrialArgs(config=config, maximum_attempts=2)
    )

    # Out of attempts, the recorded failure comes back rather than being lost.
    assert attempts == [2]
    assert outcome.result.exception_info is not None


def _activities(hooks: TrialHooks) -> HarborActivities:
    return HarborActivities(heartbeat_interval=_INTERVAL, hooks=hooks)


async def test_scope_sees_cancellation(tmp_path: Path) -> None:
    hooks = Recording()
    env = ActivityEnvironment()
    config = _trial(tasks.slow(tmp_path / "tasks", 30), tmp_path / "trials")
    running = asyncio.create_task(
        env.run(
            _activities(hooks).run_trial,
            RunTrialInput(config=config, retry=RetryConfig()),
        )
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
    async def _scope(self) -> AsyncIterator[None]:
        with contextlib.suppress(Exception):
            yield

    def scope(self, context: TrialContext) -> AbstractAsyncContextManager[None]:
        return self._scope()


async def test_scope_that_suppresses_fails_the_attempt(tmp_path: Path) -> None:
    config = _trial(tmp_path / "missing", tmp_path / "trials")
    with pytest.raises(RuntimeError, match="Swallow.scope suppressed"):
        await ActivityEnvironment().run(
            _activities(Swallow()).run_trial,
            RunTrialInput(config=config, retry=RetryConfig()),
        )
