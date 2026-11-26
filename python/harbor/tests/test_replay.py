"""A recorded harbor job replays cleanly with the plugin installed."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

from temporalio.client import Client
from temporalio.harbor import HarborPlugin
from temporalio.worker import Replayer
from tests.harbor_fixtures import tasks
from tests.harbor_fixtures.workflows import RunJob, new_worker

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)


async def test_replay_with_plugin(harbor_client: Client, tmp_path: Path) -> None:
    config = JobConfig(
        job_name="recorded",
        jobs_dir=tmp_path / "jobs",
        tasks=[TaskConfig(path=tasks.passing(tmp_path / "tasks"), source="ds")],
        n_attempts=2,
        agents=[AgentConfig(name="oracle")],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    async with new_worker(harbor_client, RunJob) as worker:
        handle = await harbor_client.start_workflow(
            RunJob.run,
            config,
            id=f"job-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        await handle.result()

    recorded = await handle.fetch_history()
    await Replayer(workflows=[RunJob], plugins=[HarborPlugin()]).replay_workflow(
        recorded
    )
