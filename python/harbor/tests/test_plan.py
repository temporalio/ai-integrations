"""plan_job expands a JobConfig exactly as harbor's Job does."""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import pytest
from harbor.job import Job
from harbor.models.job.config import DatasetConfig, JobConfig, SourceJobConfig
from harbor.models.registry import DatasetMetadata
from harbor.models.task.id import PackageTaskId
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig
from harbor.registry.client import package

from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ApplicationError
from tests.harbor_fixtures import tasks
from tests.harbor_fixtures.workflows import (
    PlanAndAggregate,
    PlanCollidingJob,
    PlanJob,
    PlanNamedJob,
    new_worker,
)

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


class _PackageRegistry:
    def __init__(self) -> None:
        self.lookups: list[str] = []

    async def get_dataset_metadata(self, name: str) -> DatasetMetadata:
        self.lookups.append(name)
        return DatasetMetadata(
            name="org/ds",
            version="sha256:v1",
            task_ids=[PackageTaskId(org="org", name="task", ref="sha256:t1")],
        )

    async def download_dataset_files(self, _: DatasetMetadata) -> dict[str, Path]:
        return {}


@pytest.mark.parametrize(
    "dataset",
    [DatasetConfig(name="org/ds"), DatasetConfig(name="org/ds", version="1.0")],
    ids=["unversioned", "versioned"],
)
async def test_metrics_come_from_the_package_version_resolved(
    harbor_client: Client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: DatasetConfig,
) -> None:
    registry = _PackageRegistry()
    monkeypatch.setattr(package, "PackageDatasetClient", lambda: registry)
    config = JobConfig(
        job_name="pinned",
        jobs_dir=tmp_path / "jobs",
        datasets=[dataset],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    async with new_worker(harbor_client, PlanAndAggregate) as worker:
        await harbor_client.execute_workflow(
            PlanAndAggregate.run,
            config,
            id=f"pinned-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
    assert registry.lookups == ["org/ds@latest", "org/ds@sha256:v1"]


async def test_plan_pins_only_package_datasets(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package, "PackageDatasetClient", _PackageRegistry)
    local = tmp_path / "local"
    tasks.passing(local)
    config = JobConfig(
        job_name="mixed",
        jobs_dir=tmp_path / "jobs",
        datasets=[
            DatasetConfig(path=local, version="1.0"),
            DatasetConfig(name="org/ds", version="1.0"),
        ],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    async with new_worker(harbor_client, PlanJob) as worker:
        plan = await harbor_client.execute_workflow(
            PlanJob.run,
            config,
            id=f"mixed-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
    assert plan.config.datasets == [
        DatasetConfig(path=local, version="1.0"),
        DatasetConfig(name="org/ds", ref="sha256:v1"),
    ]


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


async def test_trial_names_can_be_derived(
    harbor_client: Client, tmp_path: Path
) -> None:
    async with new_worker(harbor_client, PlanNamedJob) as worker:
        plan = await harbor_client.execute_workflow(
            PlanNamedJob.run,
            _config(tmp_path),
            id=f"plan-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
    assert [t.trial_name for t in plan.trials] == [
        f"{task}__{agent}__{attempt}"
        for attempt in (0, 1)
        for task in ("answers", "stays-silent")
        for agent in ("oracle", "nop")
    ]


async def test_colliding_trial_names_rejected(
    harbor_client: Client, tmp_path: Path
) -> None:
    async with new_worker(harbor_client, PlanCollidingJob) as worker:
        with pytest.raises(WorkflowFailureError) as failure:
            await harbor_client.execute_workflow(
                PlanCollidingJob.run,
                _config(tmp_path),
                id=f"plan-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
    cause = failure.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "HarborDuplicateTrialName"
