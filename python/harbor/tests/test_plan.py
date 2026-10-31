"""plan_job expands a JobConfig as harbor's Job does and rejects a misconfigured job."""

from __future__ import annotations

import re
import sys
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from harbor.job import Job
from harbor.models.bridge import BridgeConfig, BridgeKind
from harbor.models.job.config import DatasetConfig, JobConfig, SourceJobConfig
from harbor.models.job.lock import JobLock
from harbor.models.registry import DatasetMetadata
from harbor.models.task.id import LocalTaskId, PackageTaskId
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    UserAgentConfig,
)
from harbor.registry.client import package
from harbor.registry.client.base import UnavailableDatasetTasksError
from harbor.tasks.client import BatchDownloadResult, TaskClient, TaskDownloadResult
from shortuuid import ShortUUID

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError, WorkflowHandle
from temporalio.exceptions import ActivityError, ApplicationError
from tests.harbor_fixtures import history, tasks
from tests.harbor_fixtures.workflows import (
    PlanAndAggregate,
    PlanAndAggregateOnce,
    PlanCollidingJob,
    PlanEmptyNameJob,
    PlanEscapingJob,
    PlanInvalidNameJob,
    PlanJob,
    PlanNamedJob,
    PlanWithInvalidTimeout,
    QueuedJob,
    RunJob,
    RunJobOnQueue,
    RunTrialWithInvalidData,
    new_worker,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)

# A planning failure that wrongly retried would outlast this.
_PLANNED_WITHIN = timedelta(seconds=30)


def _config(
    tmp_path: Path, source_jobs: list[SourceJobConfig] | None = None, **update: Any
) -> JobConfig:
    root = tmp_path / "tasks"
    return JobConfig.model_validate(
        {
            "job_name": "expansion",
            "jobs_dir": tmp_path / "jobs",
            "tasks": [
                TaskConfig(path=tasks.passing(root), source="ds"),
                TaskConfig(path=tasks.failing(root), source="ds"),
            ],
            "n_attempts": 2,
            "agents": [AgentConfig(name="oracle"), AgentConfig(name="nop")],
            "environment": EnvironmentConfig(import_path=tasks.LOCAL_ENV),
            "timeout_multiplier": 1.5,
            "source_jobs": source_jobs or [],
            **update,
        }
    )


async def _run(client: Client, workflow_class: type[Any], arg: Any) -> Any:
    async with new_worker(client, workflow_class) as worker:
        return await client.execute_workflow(
            workflow_class.run,
            arg,
            id=f"plan-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=_PLANNED_WITHIN,
        )


async def test_expansion_matches_harbor(harbor_client: Client, tmp_path: Path) -> None:
    config = _config(tmp_path)
    job = await Job.create(config.model_copy(deep=True))
    plan = await _run(harbor_client, PlanJob, config)

    # Harbor's own equality for a trial ignores its name and job id, which are
    # generated; every other field, in order, must match.
    assert [t.config for t in plan.trials] == job._trial_configs
    assert len(plan.trials) == 2 * 2 * 2

    names = [t.config.trial_name for t in plan.trials]
    assert len(set(names)) == len(names)
    suffix = f"[{re.escape(ShortUUID().get_alphabet())}]{{7}}"
    assert all(re.fullmatch(rf"(answers|stays-silent)__{suffix}", n) for n in names)
    assert {t.config.job_id for t in plan.trials} == {plan.id}
    assert {t.config.trials_dir for t in plan.trials} == {
        tmp_path / "jobs" / "expansion"
    }

    # Agent setup at harbor's default, the tasks' agent and verifier, and the
    # default build, all under the job's multiplier.
    assert {t.timeout for t in plan.trials} == {
        timedelta(seconds=(600 + 360 + 60 + 60) * 1.5)
    }

    # The job's config and lock, in harbor's formats, where harbor keeps them.
    job_dir = tmp_path / "jobs" / "expansion"
    assert JobConfig.model_validate_json((job_dir / "config.json").read_text()) == (
        plan.config
    )
    job._init_job_lock()
    assert JobLock.model_validate_json((job_dir / "lock.json").read_text()) == (
        job._job_lock
    )


@pytest.mark.parametrize(
    "existing",
    [
        pytest.param({"timeout_multiplier": 2.0}, id="different-config"),
        pytest.param(None, id="not-a-config"),
    ],
)
async def test_job_directory_with_another_config_fails_planning(
    harbor_client: Client, tmp_path: Path, existing: dict[str, Any] | None
) -> None:
    config_json = tmp_path / "jobs" / "expansion" / "config.json"
    if existing is None:
        config_json.parent.mkdir(parents=True)
        config_json.write_text("{")
    else:
        await _run(harbor_client, PlanJob, _config(tmp_path, **existing))
    before = config_json.read_text()

    cause, handle = await _planning_failure(harbor_client, PlanJob, _config(tmp_path))
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.non_retryable
    assert cause.cause.type == "FileExistsError"
    assert await history.final_attempts(handle) == [1]
    assert config_json.read_text() == before


async def test_planning_retry_finds_the_files_it_wrote(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = JobLock.model_dump_json
    failed: list[bool] = []

    def fail_once(self: JobLock, **kwargs: Any) -> str:
        if not failed:
            failed.append(True)
            raise OSError("no space left on device")
        return dump(self, **kwargs)

    # The first attempt fails after writing config.json.
    monkeypatch.setattr(JobLock, "model_dump_json", fail_once)
    async with new_worker(harbor_client, PlanJob) as worker:
        handle = await harbor_client.start_workflow(
            PlanJob.run,
            _config(tmp_path),
            id=f"plan-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=_PLANNED_WITHIN,
        )
        await handle.result()
    assert failed
    assert await history.final_attempts(handle) == [2]
    assert (tmp_path / "jobs" / "expansion" / "lock.json").exists()


async def test_job_directory_configs_compare_as_harbor_compares_them(
    harbor_client: Client, tmp_path: Path
) -> None:
    # Harbor's equality ignores debug.
    await _run(harbor_client, PlanJob, _config(tmp_path))
    plan = await _run(harbor_client, PlanJob, _config(tmp_path, debug=True))
    assert plan.config.debug


async def test_untimed_agent_counts_untimed_agent_timeout(
    harbor_client: Client, tmp_path: Path
) -> None:
    task = tmp_path / "untimed"
    tasks.passing(task.parent, task.name)
    (task / "task.toml").write_text('version = "1.0"\n')
    config = _config(tmp_path, tasks=[TaskConfig(path=task)], timeout_multiplier=1.0)
    plan = await _run(harbor_client, PlanJob, config)
    assert {t.timeout for t in plan.trials} == {
        timedelta(seconds=600 + 360 + 600) + timedelta(hours=2)
    }


def _no_file_system_in_workflows(monkeypatch: pytest.MonkeyPatch) -> None:
    def guard(name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(*args: Any, **kwargs: Any) -> Any:
            if workflow.in_workflow():
                raise ApplicationError(
                    f"workflow code called {name}",
                    type="WorkflowTouchedFileSystem",
                    non_retryable=True,
                )
            return real(*args, **kwargs)

        return guarded

    monkeypatch.setattr(
        LocalTaskId, "get_name", guard("get_name", LocalTaskId.get_name)
    )
    monkeypatch.setattr(Path, "resolve", guard("Path.resolve", Path.resolve))


async def test_workflow_code_leaves_the_file_system_alone(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_file_system_in_workflows(monkeypatch)
    config = _config(tmp_path, n_attempts=1, agents=[AgentConfig(name="oracle")])
    result = await _run(harbor_client, RunJob, config)
    assert result.stats.n_completed_trials == 2


async def _planning_failure(
    client: Client, workflow_class: type[Any], arg: Any
) -> tuple[BaseException | None, WorkflowHandle[Any, Any]]:
    async with new_worker(client, workflow_class) as worker:
        handle = await client.start_workflow(
            workflow_class.run,
            arg,
            id=f"plan-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=_PLANNED_WITHIN,
        )
        with pytest.raises(WorkflowFailureError) as failure:
            await handle.result()
    return failure.value.cause, handle


async def test_unknown_agent_option_fails_planning(
    harbor_client: Client, tmp_path: Path
) -> None:
    config = _config(
        tmp_path,
        agents=[AgentConfig(name="claude-code", kwargs={"no_such_option": 1})],
    )
    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.non_retryable
    assert "no_such_option" in cause.cause.message
    assert await history.final_attempts(handle) == [1]


async def test_agent_options_are_checked_without_the_workers_environment(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Credentials are the worker's business; planning must not need them.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = _config(
        tmp_path,
        agents=[
            AgentConfig(
                name="claude-code",
                env={"ANTHROPIC_API_KEY": "${SET_ON_THE_WORKER}"},
                kwargs={"max_turns": 3},
            )
        ],
    )
    plan = await _run(harbor_client, PlanJob, config)
    assert len(plan.trials) == 2 * 2


class _PackageRegistry:
    def __init__(self, *, empty: bool = False, unavailable: int = 0) -> None:
        self.lookups: list[str] = []
        self.empty = empty
        self.unavailable = unavailable

    async def get_dataset_metadata(self, name: str) -> DatasetMetadata:
        self.lookups.append(name)
        if self.unavailable:
            raise UnavailableDatasetTasksError(name, self.unavailable)
        return DatasetMetadata(
            name="org/ds",
            version="sha256:v1",
            task_ids=[]
            if self.empty
            else [PackageTaskId(org="org", name="task", ref="sha256:t1")],
        )

    async def download_dataset_files(self, _: DatasetMetadata) -> dict[str, Path]:
        return {}


def _local_packages(monkeypatch: pytest.MonkeyPatch, task: Path) -> None:
    async def download_tasks(
        _client: TaskClient, task_ids: list[Any], **_: Any
    ) -> BatchDownloadResult:
        return BatchDownloadResult(
            results=[
                TaskDownloadResult(
                    path=task, download_time_sec=0, cached=True, content_hash="0" * 64
                )
                for _ in task_ids
            ],
            total_time_sec=0,
        )

    monkeypatch.setattr(TaskClient, "download_tasks", download_tasks)


async def test_empty_dataset_fails_planning(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        package, "PackageDatasetClient", lambda: _PackageRegistry(empty=True)
    )
    config = JobConfig(
        job_name="empty",
        jobs_dir=tmp_path / "jobs",
        datasets=[DatasetConfig(name="org/ds")],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.type == "EmptyDatasetError"
    assert cause.cause.non_retryable
    assert await history.final_attempts(handle) == [1]


async def test_invalid_trial_config_fails_the_workflow(
    harbor_client: Client, tmp_path: Path
) -> None:
    cause, _ = await _planning_failure(
        harbor_client, PlanInvalidNameJob, _config(tmp_path)
    )
    assert isinstance(cause, ApplicationError)
    assert cause.type == "ValidationError"
    assert cause.non_retryable


def _one_trial(tmp_path: Path) -> TrialConfig:
    return TrialConfig(
        task=TaskConfig(path=tasks.passing(tmp_path / "tasks")),
        trial_name="answers__0",
        trials_dir=tmp_path / "trials",
        agent=AgentConfig(name="oracle"),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )


@pytest.mark.parametrize(
    ("workflow_class", "make_arg", "function"),
    [
        (PlanWithInvalidTimeout, _config, "plan_job"),
        (RunTrialWithInvalidData, _one_trial, "execute_trial"),
    ],
    ids=["plan_job", "execute_trial"],
)
async def test_invalid_arguments_fail_the_workflow(
    harbor_client: Client,
    tmp_path: Path,
    workflow_class: type[Any],
    make_arg: Callable[[Path], Any],
    function: str,
) -> None:
    cause, handle = await _planning_failure(
        harbor_client, workflow_class, make_arg(tmp_path)
    )
    assert isinstance(cause, ApplicationError)
    assert cause.type == "ValidationError"
    assert cause.non_retryable
    assert f"{function}'s arguments are not valid" in cause.message
    assert await history.scheduled(handle) == []


async def test_concurrency_caps_are_reported_unused(
    harbor_client: Client, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config(
        tmp_path,
        agents=[AgentConfig(name="oracle", n_concurrent=1, concurrency_group="g")],
    )
    await _run(harbor_client, PlanJob, config)
    assert any("n_concurrent" in r.getMessage() for r in caplog.records)


async def test_every_activity_runs_on_the_given_task_queue(
    harbor_client: Client, tmp_path: Path
) -> None:
    config = _config(tmp_path, n_attempts=1, agents=[AgentConfig(name="oracle")])
    async with (
        new_worker(harbor_client, RunJobOnQueue) as workflows,
        new_worker(harbor_client) as activities,
    ):
        handle = await harbor_client.start_workflow(
            RunJobOnQueue.run,
            QueuedJob(config=config, task_queue=activities.task_queue),
            id=f"queued-{uuid.uuid4()}",
            task_queue=workflows.task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        await handle.result()

    queues = {
        e.activity_task_scheduled_event_attributes.task_queue.name
        for e in await history.scheduled(handle)
    }
    assert queues == {activities.task_queue}


@pytest.mark.parametrize(
    ("workflow_class", "maximum_attempts"),
    [(PlanAndAggregate, 0), (PlanAndAggregateOnce, 1)],
    ids=["default", "given"],
)
async def test_aggregation_retry_policy(
    harbor_client: Client,
    tmp_path: Path,
    workflow_class: type[Any],
    maximum_attempts: int,
) -> None:
    async with new_worker(harbor_client, workflow_class) as worker:
        handle = await harbor_client.start_workflow(
            workflow_class.run,
            _config(tmp_path),
            id=f"aggregate-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=_PLANNED_WITHIN,
        )
        await handle.result()

    [aggregation] = [
        event.activity_task_scheduled_event_attributes
        for event in await history.scheduled(handle)
        if event.activity_task_scheduled_event_attributes.activity_type.name
        == "harbor.aggregate_job"
    ]
    assert aggregation.retry_policy.maximum_attempts == maximum_attempts


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
    _local_packages(monkeypatch, tasks.passing(tmp_path / "tasks"))
    config = JobConfig(
        job_name="pinned",
        jobs_dir=tmp_path / "jobs",
        datasets=[dataset],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    await _run(harbor_client, PlanAndAggregate, config)
    # Planning resolves the dataset and looks up its metric; aggregation looks
    # the metric up again, by the content hash planning pinned.
    assert registry.lookups == [
        "org/ds@latest",
        "org/ds@sha256:v1",
        "org/ds@sha256:v1",
    ]


async def test_plan_pins_only_package_datasets(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package, "PackageDatasetClient", _PackageRegistry)
    local = tmp_path / "local"
    _local_packages(monkeypatch, tasks.passing(local))
    config = JobConfig(
        job_name="mixed",
        jobs_dir=tmp_path / "jobs",
        datasets=[
            DatasetConfig(path=local, version="1.0"),
            DatasetConfig(name="org/ds", version="1.0"),
        ],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    plan = await _run(harbor_client, PlanJob, config)
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
    cause, _ = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ApplicationError)
    assert cause.type == "HarborRegradeUnsupported"


async def test_trial_names_can_be_derived(
    harbor_client: Client, tmp_path: Path
) -> None:
    plan = await _run(harbor_client, PlanNamedJob, _config(tmp_path))
    assert [t.config.trial_name for t in plan.trials] == [
        f"{task}__{agent}__{attempt}"
        for attempt in (0, 1)
        for task in ("answers", "stays-silent")
        for agent in ("oracle", "nop")
    ]


async def test_colliding_trial_names_rejected(
    harbor_client: Client, tmp_path: Path
) -> None:
    cause, _ = await _planning_failure(
        harbor_client, PlanCollidingJob, _config(tmp_path)
    )
    assert isinstance(cause, ApplicationError)
    assert cause.type == "HarborDuplicateTrialName"


async def test_trial_names_outside_the_job_rejected(
    harbor_client: Client, tmp_path: Path
) -> None:
    cause, _ = await _planning_failure(
        harbor_client, PlanEscapingJob, _config(tmp_path)
    )
    assert isinstance(cause, ApplicationError)
    assert cause.type == "HarborInvalidTrialName"
    assert cause.non_retryable


async def test_empty_trial_name_rejected(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_file_system_in_workflows(monkeypatch)
    cause, _ = await _planning_failure(
        harbor_client, PlanEmptyNameJob, _config(tmp_path)
    )
    assert isinstance(cause, ApplicationError)
    assert cause.type == "HarborInvalidTrialName"
    assert cause.non_retryable


async def test_unknown_user_agent_option_fails_planning(
    harbor_client: Client, tmp_path: Path
) -> None:
    config = _config(
        tmp_path,
        user_agent=UserAgentConfig(
            name="claude-code",
            kwargs={"no_such_option": 1},
            bridge=BridgeConfig(kind=BridgeKind.ACP),
        ),
    )
    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.non_retryable
    assert "no_such_option" in cause.cause.message
    assert await history.final_attempts(handle) == [1]


async def test_unavailable_dataset_tasks_fail_planning(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        package, "PackageDatasetClient", lambda: _PackageRegistry(unavailable=2)
    )
    config = JobConfig(
        job_name="unavailable",
        jobs_dir=tmp_path / "jobs",
        datasets=[DatasetConfig(name="org/ds")],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.type == "UnavailableDatasetTasksError"
    assert cause.cause.non_retryable
    assert await history.final_attempts(handle) == [1]


@pytest.mark.parametrize(
    "toml",
    ['version = "1.0"\n[agent\n', 'version = "1.0"\n[agent]\ntimeout_sec = "soon"\n'],
    ids=["syntax", "schema"],
)
async def test_malformed_task_definition_fails_planning(
    harbor_client: Client, tmp_path: Path, toml: str
) -> None:
    task = tmp_path / "malformed"
    tasks.passing(task.parent, task.name)
    (task / "task.toml").write_text(toml)
    config = _config(tmp_path, tasks=[TaskConfig(path=task)])
    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.non_retryable
    assert await history.final_attempts(handle) == [1]
