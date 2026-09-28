"""plan_job expands a JobConfig exactly as harbor's Job does."""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import pytest
from harbor.job import Job
from harbor.models.job.config import JobConfig, SourceJobConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ApplicationError
from tests.harbor_fixtures import tasks
from tests.harbor_fixtures.workflows import PlanJob, new_worker

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)


def _config(
    tmp_path: Path, source_jobs: list[SourceJobConfig] | None = None
) -> JobConfig:
    root = tmp_path / "tasks"
    return JobConfig(
        job_name="expansion",
        jobs_dir=tmp_path / "jobs",
        tasks=[
            TaskConfig(path=tasks.passing(root), source="ds"),
            TaskConfig(path=tasks.failing(root), source="ds"),
        ],
        n_attempts=2,
        agents=[AgentConfig(name="oracle"), AgentConfig(name="nop")],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
        timeout_multiplier=1.5,
        source_jobs=source_jobs or [],
    )


async def test_expansion_matches_harbor(harbor_client: Client, tmp_path: Path) -> None:
    config = _config(tmp_path)
    async with new_worker(harbor_client, PlanJob) as worker:
        plan = await harbor_client.execute_workflow(
            PlanJob.run,
            config,
            id=f"plan-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )

    job = await Job.create(config)
    reference = job._trial_configs

    # Harbor's own equality for a trial ignores its name and job id, which are
    # generated; every other field, in order, must match.
    assert plan.trials == reference
    assert len(plan.trials) == 2 * 2 * 2

    names = [t.trial_name for t in plan.trials]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"(answers|stays-silent)__[0-9a-f]{7}", n) for n in names)
    assert {t.job_id for t in plan.trials} != {None}
    assert len({t.job_id for t in plan.trials}) == 1
    assert {t.trials_dir for t in plan.trials} == {tmp_path / "jobs" / "expansion"}


async def test_regrade_rejected(harbor_client: Client, tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        source_jobs=[
            SourceJobConfig(action="regrade", type="local", path=tmp_path / "old")
        ],
    )
    async with new_worker(harbor_client, PlanJob) as worker:
        with pytest.raises(WorkflowFailureError) as failure:
            await harbor_client.execute_workflow(
                PlanJob.run,
                config,
                id=f"regrade-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
    cause = failure.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "HarborRegradeUnsupported"
