"""Replaying a job from history schedules nothing twice and renames nothing."""

from __future__ import annotations

import sys
import uuid
from collections import Counter
from pathlib import Path

import pytest
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

from temporalio.client import Client
from temporalio.harbor._types import RunTrialInput
from tests.harbor_fixtures import history, tasks
from tests.harbor_fixtures.workflows import RunJob, TrialNames, new_worker

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)


def _config(tmp_path: Path) -> JobConfig:
    root = tmp_path / "tasks"
    return JobConfig(
        job_name="replayed",
        jobs_dir=tmp_path / "jobs",
        tasks=[
            TaskConfig(path=tasks.passing(root)),
            TaskConfig(path=tasks.failing(root)),
        ],
        n_attempts=2,
        agents=[AgentConfig(name="oracle")],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )


async def test_one_schedule_per_trial(harbor_client: Client, tmp_path: Path) -> None:
    # No workflow cache: every workflow task replays the whole history so far.
    async with new_worker(harbor_client, RunJob, max_cached_workflows=0) as worker:
        handle = await harbor_client.start_workflow(
            RunJob.run,
            _config(tmp_path),
            id=f"job-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        stats = await handle.result()

    assert stats.n_completed_trials == 4
    events = await history.scheduled(handle)
    by_type = Counter(
        e.activity_task_scheduled_event_attributes.activity_type.name for e in events
    )
    assert by_type == {
        "harbor.resolve_job": 1,
        "harbor.run_trial": 4,
        "harbor.compute_metrics": 1,
    }
    ids = {e.activity_task_scheduled_event_attributes.activity_id for e in events}
    assert len(ids) == len(events)


async def test_trial_names_are_stable_across_replay(
    harbor_client: Client, tmp_path: Path
) -> None:
    async with new_worker(harbor_client, TrialNames, max_cached_workflows=0) as worker:
        handle = await harbor_client.start_workflow(
            TrialNames.run,
            _config(tmp_path),
            id=f"names-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        # Computed by the last replay, long after the trials were scheduled.
        returned = await handle.result()

    converter = harbor_client.data_converter.payload_converter
    scheduled = [
        converter.from_payload(
            e.activity_task_scheduled_event_attributes.input.payloads[0], RunTrialInput
        ).config.trial_name
        for e in await history.scheduled(handle)
        if e.activity_task_scheduled_event_attributes.activity_type.name
        == "harbor.run_trial"
    ]
    assert returned == scheduled
