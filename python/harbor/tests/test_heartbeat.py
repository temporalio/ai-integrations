"""A running trial heartbeats on a fixed cadence and reports its phase."""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)

from temporalio.harbor._activity import HarborActivities
from temporalio.harbor._types import RunTrialInput
from temporalio.testing import ActivityEnvironment
from tests.harbor_fixtures import tasks

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)

_INTERVAL = timedelta(milliseconds=200)
_LIFECYCLE = [
    "load",
    "start",
    "environment-start",
    "agent-start",
    "agent-end",
    "verification-start",
    "end",
]


async def _beats(
    tmp_path: Path, agent_seconds: float
) -> list[tuple[float, dict[str, Any]]]:
    beats: list[tuple[float, dict[str, Any]]] = []
    env = ActivityEnvironment()
    env.on_heartbeat = lambda *details: beats.append((time.monotonic(), details[0]))
    config = TrialConfig(
        task=TaskConfig(path=tasks.slow(tmp_path / "tasks", agent_seconds)),
        trial_name="slow__0",
        trials_dir=tmp_path / "trials",
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    activities = HarborActivities(heartbeat_interval=_INTERVAL)
    result = await env.run(
        activities.run_trial, RunTrialInput(config=config, retry=RetryConfig())
    )
    assert result.verifier_result is not None
    assert result.verifier_result.rewards == {"reward": 1.0}
    return beats


async def test_heartbeats_on_interval_through_silent_agent(tmp_path: Path) -> None:
    # The agent prints nothing and emits no harbor event for three seconds.
    beats = await _beats(tmp_path, agent_seconds=3)

    gaps = [later - earlier for (earlier, _), (later, _) in zip(beats, beats[1:])]
    assert len(beats) >= 3 / _INTERVAL.total_seconds() * 0.8
    assert max(gaps) < 4 * _INTERVAL.total_seconds()


async def test_heartbeat_detail_tracks_phase(tmp_path: Path) -> None:
    beats = await _beats(tmp_path, agent_seconds=1)

    phases = [detail["phase"] for _, detail in beats]
    seen = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    assert "agent-start" in seen
    # Phases only ever move forward through harbor's lifecycle.
    assert [_LIFECYCLE.index(p) for p in seen] == sorted(
        _LIFECYCLE.index(p) for p in seen
    )

    elapsed = [detail["elapsed_sec"] for _, detail in beats]
    assert elapsed == sorted(elapsed)
