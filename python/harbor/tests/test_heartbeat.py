"""A running trial heartbeats at a cadence set by its heartbeat timeout."""

from __future__ import annotations

import dataclasses
import sys
import time
from collections.abc import Callable
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

from temporalio.harbor import TrialHooks
from temporalio.harbor._activity import HarborActivities
from temporalio.harbor._types import RunTrialInput
from temporalio.testing import ActivityEnvironment
from tests.harbor_fixtures import tasks

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)

_HEARTBEAT_TIMEOUT = timedelta(milliseconds=800)
# Two heartbeat timeouts at four beats each, counting the first beat.
_SILENT_BEATS = 9
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
    tmp_path: Path,
    task: Path,
    on_beat: Callable[[float, dict[str, Any]], None] = lambda *_: None,
) -> list[tuple[float, dict[str, Any]]]:
    beats: list[tuple[float, dict[str, Any]]] = []
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, heartbeat_timeout=_HEARTBEAT_TIMEOUT)

    def record(*details: Any) -> None:
        beats.append((time.monotonic(), details[0]))
        on_beat(*beats[-1])

    env.on_heartbeat = record
    config = TrialConfig(
        task=TaskConfig(path=task),
        trial_name=f"{task.name}__0",
        trials_dir=tmp_path / "trials",
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    outcome = await env.run(
        HarborActivities(hooks=TrialHooks()).run_trial,
        RunTrialInput(config=config, extra_time=timedelta(minutes=1)),
    )
    result = outcome.result
    assert result.verifier_result is not None
    assert result.verifier_result.rewards == {"reward": 1.0}
    return beats


async def test_heartbeats_through_silent_agent(tmp_path: Path) -> None:
    # The agent waits, printing nothing and emitting no harbor event, until
    # _SILENT_BEATS beats arrive. Only those are measured: harbor's setup before
    # it blocks the event loop and can delay a beat on a loaded machine.
    waiting, release = tmp_path / "waiting", tmp_path / "release"
    silent: list[tuple[float, str]] = []

    def on_beat(at: float, detail: dict[str, Any]) -> None:
        if waiting.exists() and not release.exists():
            silent.append((at, detail["phase"]))
            if len(silent) == _SILENT_BEATS:
                release.touch()

    await _beats(tmp_path, tasks.held(tmp_path / "tasks", waiting, release), on_beat)

    # Several beats fit in each heartbeat timeout, however long the agent runs.
    timeout = _HEARTBEAT_TIMEOUT.total_seconds()
    times = [at for at, _ in silent]
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    assert len(silent) == _SILENT_BEATS
    assert {phase for _, phase in silent} == {"agent-start"}
    assert times[-1] - times[0] > timeout
    assert max(gaps) < timeout / 2


async def test_heartbeat_detail_tracks_phase(tmp_path: Path) -> None:
    beats = await _beats(tmp_path, tasks.slow(tmp_path / "tasks", 1))

    phases = [detail["phase"] for _, detail in beats]
    seen = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    assert "agent-start" in seen
    # Phases only ever move forward through harbor's lifecycle.
    assert [_LIFECYCLE.index(p) for p in seen] == sorted(
        _LIFECYCLE.index(p) for p in seen
    )

    elapsed = [detail["elapsed_sec"] for _, detail in beats]
    assert elapsed == sorted(elapsed)
