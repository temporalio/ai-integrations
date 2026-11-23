"""plan_job expands a JobConfig as harbor's Job does and rejects a misconfigured job."""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import pytest
from harbor.auth import tokens as harbor_tokens
from harbor.job import Job
from harbor.models.bridge import BridgeConfig, BridgeKind
from harbor.models.job.config import DatasetConfig, JobConfig, SourceJobConfig
from harbor.models.job.lock import JobLock
from harbor.models.job.result import JobResult
from harbor.models.registry import DatasetMetadata
from harbor.models.task.id import LocalTaskId, PackageTaskId
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    UserAgentConfig,
    VerifierConfig,
)
from harbor.registry.client import package
from harbor.registry.client.base import UnavailableDatasetTasksError
from harbor.tasks.client import BatchDownloadResult, TaskClient, TaskDownloadResult
from shortuuid import ShortUUID

from temporalio import activity, workflow
from temporalio.client import Client, WorkflowFailureError, WorkflowHandle
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.harbor._activity import _on_own_loop
from tests.harbor_fixtures import history, tasks
from tests.harbor_fixtures.workflows import (
    AggregateExceptions,
    AggregateWithSecondsOfTimeout,
    PlanAndAggregate,
    PlanAndAggregateOnce,
    PlanCaseCollidingJob,
    PlanCollidingJob,
    PlanDriveNameJob,
    PlanEmptyNameJob,
    PlanEscapingJob,
    PlanInvalidNameJob,
    PlanJob,
    PlanNamedJob,
    PlanTwoUnnamedJobs,
    PlanWithInvalidTimeout,
    PlanWithSecondsOfTimeout,
    PlanWithZeroTimeout,
    QueuedJob,
    RunJob,
    RunJobOnQueue,
    RunTrialWithInvalidData,
    RunTrialWithSecondsOfHeartbeat,
    new_worker,
)

on_posix = pytest.mark.skipif(
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
    extra = tmp_path / "extra.md"
    extra.write_text("Answer in English.\n")
    # Every field harbor maps from the job onto its trials, set.
    config = _config(
        tmp_path,
        install_only=True,
        user_agent=UserAgentConfig(
            name="oracle", bridge=BridgeConfig(kind=BridgeKind.ACP)
        ),
        agent_timeout_multiplier=2.0,
        verifier_timeout_multiplier=3.0,
        agent_setup_timeout_multiplier=0.5,
        environment_build_timeout_multiplier=0.25,
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV, env={"MODE": "x"}),
        verifier=VerifierConfig(override_timeout_sec=45.0),
        artifacts=["/logs/answer.txt"],
        extra_instruction_paths=[extra],
        extra_instructions=["Be brief."],
    )
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

    # The default build, five agent setups at harbor's default for a user
    # agent, the tasks' agent and the job's verifier, each under its multiplier.
    assert {t.timeout for t in plan.trials} == {
        timedelta(seconds=600 * 0.25 + 5 * 360 * 0.5 + 60 * 2.0 + 45 * 3.0)
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
    # Harbor's equality ignores debug, and the path a job directory is reached by.
    await _run(harbor_client, PlanJob, _config(tmp_path))
    plan = await _run(
        harbor_client,
        PlanJob,
        _config(tmp_path, debug=True, jobs_dir=tmp_path / "jobs" / ".." / "jobs"),
    )
    assert plan.config.debug


async def test_job_directory_with_another_lock_fails_planning(
    harbor_client: Client, tmp_path: Path
) -> None:
    await _run(harbor_client, PlanJob, _config(tmp_path))
    lock_json = tmp_path / "jobs" / "expansion" / "lock.json"
    before = lock_json.read_text()
    config = _config(tmp_path)
    # The same config, but a task's content has changed since.
    (tmp_path / "tasks" / "answers" / "instruction.md").write_text("Say hello.\n")

    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.non_retryable
    assert cause.cause.type == "FileExistsError"
    assert "lock.json" in cause.cause.message
    assert await history.final_attempts(handle) == [1]
    assert lock_json.read_text() == before


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


@on_posix
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


@pytest.mark.parametrize(
    ("agent", "accepted"),
    [
        pytest.param(
            AgentConfig(
                name="claude-code", kwargs={"model_api_format": "anthropic_messages"}
            ),
            True,
            id="model-api-format",
        ),
        pytest.param(
            AgentConfig(name="hermes", kwargs={"reasoning_effort": "high"}),
            False,
            id="unsupported-capability",
        ),
    ],
)
async def test_agent_options_are_checked_as_harbor_checks_them(
    harbor_client: Client, tmp_path: Path, agent: AgentConfig, accepted: bool
) -> None:
    config = _config(tmp_path, agents=[agent])
    if accepted:
        plan = await _run(harbor_client, PlanJob, config)
        assert len(plan.trials) == 2 * 2
        return
    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.non_retryable
    assert "reasoning_effort" in cause.cause.message
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
    def __init__(
        self,
        *,
        empty: bool = False,
        unavailable: int = 0,
        missing: bool = False,
        unreachable: frozenset[int] = frozenset(),
    ) -> None:
        self.lookups: list[str] = []
        self.loops: list[asyncio.AbstractEventLoop] = []
        self.activities: list[str] = []
        self.empty = empty
        self.unavailable = unavailable
        self.missing = missing
        self.unreachable = unreachable

    async def get_dataset_metadata(self, name: str) -> DatasetMetadata:
        self.lookups.append(name)
        self.loops.append(asyncio.get_running_loop())
        self.activities.append(activity.info().activity_type)
        if len(self.lookups) - 1 in self.unreachable:
            # As harbor's registry clients report a lookup that failed.
            try:
                raise ConnectionError("the registry did not answer")
            except ConnectionError:
                raise ValueError(f"Error getting dataset {name}")  # noqa: B904
        if self.missing:
            raise ValueError(f"Dataset {name} not found")
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


def _package_job(tmp_path: Path, name: str = "packaged") -> JobConfig:
    return JobConfig(
        job_name=name,
        jobs_dir=tmp_path / "jobs",
        datasets=[DatasetConfig(name="org/ds")],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )


@pytest.mark.parametrize(
    ("registry", "attempts"),
    [
        pytest.param({"unreachable": frozenset({0})}, [2], id="lookup-failed"),
        pytest.param({"missing": True}, [1], id="not-found"),
    ],
)
async def test_registry_lookup_is_retried_unless_not_found(
    harbor_client: Client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: dict[str, Any],
    attempts: list[int],
) -> None:
    shared = _PackageRegistry(**registry)
    monkeypatch.setattr(package, "PackageDatasetClient", lambda: shared)
    _local_packages(monkeypatch, tasks.passing(tmp_path / "tasks"))
    async with new_worker(harbor_client, PlanJob) as worker:
        handle = await harbor_client.start_workflow(
            PlanJob.run,
            _package_job(tmp_path),
            id=f"plan-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=_PLANNED_WITHIN,
        )
        try:
            await handle.result()
        except WorkflowFailureError as failure:
            assert isinstance(failure.cause, ActivityError)
            assert isinstance(failure.cause.cause, ApplicationError)
            assert failure.cause.cause.non_retryable
    assert await history.final_attempts(handle) == attempts


async def test_registry_lookup_at_aggregation_is_retried(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Planning looks the dataset up twice; aggregation's lookup fails once.
    registry = _PackageRegistry(unreachable=frozenset({2}))
    monkeypatch.setattr(package, "PackageDatasetClient", lambda: registry)
    _local_packages(monkeypatch, tasks.passing(tmp_path / "tasks"))
    async with new_worker(harbor_client, PlanAndAggregate) as worker:
        handle = await harbor_client.start_workflow(
            PlanAndAggregate.run,
            _package_job(tmp_path),
            id=f"aggregate-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=_PLANNED_WITHIN,
        )
        await handle.result()
    assert await history.final_attempts(handle) == [1, 2]


async def test_registry_lookups_run_off_the_event_loop(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A registry client can block, and every trial on the worker heartbeats
    # from its event loop.
    registry = _PackageRegistry()
    monkeypatch.setattr(package, "PackageDatasetClient", lambda: registry)
    _local_packages(monkeypatch, tasks.passing(tmp_path / "tasks"))
    await _run(harbor_client, PlanAndAggregate, _package_job(tmp_path))
    assert asyncio.get_running_loop() not in registry.loops
    assert registry.activities == [
        "harbor.plan_job",
        "harbor.plan_job",
        "harbor.aggregate_job",
    ]


async def test_harbor_token_lock_works_on_every_loop(
    harbor_client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Planning runs harbor on loops of its own, beside the worker's.
    async def exchange(api_key: str) -> tuple[str, float]:
        await asyncio.sleep(0.05)
        return f"token for {api_key}", 3600.0

    monkeypatch.setattr(harbor_tokens, "exchange_api_key", exchange)

    async def refresh_twice_at_once() -> list[str]:
        harbor_tokens.invalidate_token()
        return list(
            await asyncio.gather(
                harbor_tokens._cache.get_access_token("key"),
                harbor_tokens._cache.get_access_token("key"),
            )
        )

    async with new_worker(harbor_client):
        expected = ["token for key", "token for key"]
        assert await _on_own_loop(refresh_twice_at_once) == expected
        assert await refresh_twice_at_once() == expected
        assert await _on_own_loop(refresh_twice_at_once) == expected


async def test_cancelled_planning_cancels_harbor_on_its_loop() -> None:
    started = threading.Event()
    stopped = threading.Event()

    async def download() -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            stopped.set()
            raise

    waiting = asyncio.create_task(_on_own_loop(download))
    assert await asyncio.to_thread(started.wait, 10)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert await asyncio.to_thread(stopped.wait, 10)


async def test_missing_package_task_fails_planning(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package, "PackageDatasetClient", _PackageRegistry)

    async def download_tasks(*_args: Any, **_kwargs: Any) -> BatchDownloadResult:
        # As harbor downloads package tasks: in a task group.
        async def missing() -> None:
            raise ValueError("Task version not found: org/task@sha256:t1")

        async with asyncio.TaskGroup() as group:
            group.create_task(missing())
        raise AssertionError("unreachable")

    monkeypatch.setattr(TaskClient, "download_tasks", download_tasks)
    cause, handle = await _planning_failure(
        harbor_client, PlanJob, _package_job(tmp_path)
    )
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.type == "ValueError"
    assert cause.cause.non_retryable
    assert await history.final_attempts(handle) == [1]


async def test_group_with_a_config_error_fails_planning(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package, "PackageDatasetClient", _PackageRegistry)

    async def download_tasks(*_args: Any, **_kwargs: Any) -> BatchDownloadResult:
        raise ExceptionGroup(
            "downloads",
            [
                ConnectionError("the registry did not answer"),
                ValueError("Task version not found: org/task@sha256:t1"),
                ExceptionGroup(
                    "nested", [ValueError("Task version not found: org/next@sha256:t2")]
                ),
            ],
        )

    monkeypatch.setattr(TaskClient, "download_tasks", download_tasks)
    cause, handle = await _planning_failure(
        harbor_client, PlanJob, _package_job(tmp_path)
    )
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.type == "ValueError"
    assert cause.cause.non_retryable
    assert cause.cause.message == (
        "Task version not found: org/task@sha256:t1; "
        "Task version not found: org/next@sha256:t2"
    )
    assert isinstance(cause.cause.cause, ApplicationError)
    assert cause.cause.cause.type == "ExceptionGroup"
    assert await history.final_attempts(handle) == [1]


async def test_skill_without_skill_md_fails_planning(
    harbor_client: Client, tmp_path: Path
) -> None:
    skill = tmp_path / "skills" / "greeting"
    skill.mkdir(parents=True)
    config = _config(tmp_path, agents=[AgentConfig(name="oracle", skills=[str(skill)])])
    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.non_retryable
    assert await history.final_attempts(handle) == [1]


async def test_unknown_environment_import_path_fails_planning(
    harbor_client: Client, tmp_path: Path
) -> None:
    config = _config(
        tmp_path,
        environment=EnvironmentConfig(
            import_path="tests.harbor_fixtures.local_env:NoSuchEnvironment"
        ),
    )
    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.non_retryable
    assert "NoSuchEnvironment" in cause.cause.message
    assert await history.final_attempts(handle) == [1]


async def test_infinite_task_timeout_fails_planning(
    harbor_client: Client, tmp_path: Path
) -> None:
    task = tasks.passing(tmp_path, "forever")
    toml = task / "task.toml"
    toml.write_text(
        toml.read_text().replace(
            "[agent]\ntimeout_sec = 60.0", "[agent]\ntimeout_sec = inf"
        )
    )
    config = _config(tmp_path, tasks=[TaskConfig(path=task)])
    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ActivityError)
    assert isinstance(cause.cause, ApplicationError)
    assert cause.cause.non_retryable
    assert "forever" in cause.cause.message
    assert "agent.timeout_sec" in cause.cause.message
    assert await history.final_attempts(handle) == [1]


def _git_task(tmp_path: Path) -> tuple[TaskConfig, str]:
    """A task in a new git repository, and the repository's commit."""
    repo = tmp_path / "repo"
    tasks.passing(repo)
    for command in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "task"],
    ):
        subprocess.run(["git", *command], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    task = TaskConfig(
        git_url=repo.as_uri(), path=Path("answers"), download_dir=tmp_path / "cache"
    )
    return task, commit


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_git_task_is_pinned_to_the_commit_planning_resolved(
    harbor_client: Client, tmp_path: Path
) -> None:
    task, commit = _git_task(tmp_path)
    plan = await _run(harbor_client, PlanJob, _config(tmp_path, tasks=[task]))

    assert [t.git_commit_id for t in plan.task_configs] == [commit]
    assert {t.config.task.git_commit_id for t in plan.trials} == {commit}
    lock = JobLock.model_validate_json(
        (tmp_path / "jobs" / "expansion" / "lock.json").read_text()
    )
    assert {t.task.git_commit_id for t in lock.trials} == {commit}


@on_posix
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_harbor_resumes_a_finished_job_with_a_git_task(
    harbor_client: Client, tmp_path: Path
) -> None:
    task, _ = _git_task(tmp_path)
    await _run(harbor_client, RunJob, _config(tmp_path, tasks=[task]))

    # As harbor job resume does: expand config.json again and match each
    # trial already in the job directory to one of its trials.
    config_json = tmp_path / "jobs" / "expansion" / "config.json"
    job = await Job.create(JobConfig.model_validate_json(config_json.read_text()))
    try:
        assert job._remaining_trial_configs == []
    finally:
        job._close_logger_handlers()


async def test_unnamed_jobs_are_named_apart(
    harbor_client: Client, tmp_path: Path
) -> None:
    config = _config(tmp_path).model_dump(mode="json", exclude={"job_name"})
    async with new_worker(harbor_client, PlanTwoUnnamedJobs) as worker:
        names = await harbor_client.execute_workflow(
            "PlanTwoUnnamedJobs",
            config,
            id=f"plan-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=_PLANNED_WITHIN,
            result_type=list[str],
        )
    assert len(set(names)) == 2
    assert all(
        re.fullmatch(r"\d{4}-\d\d-\d\d__\d\d-\d\d-\d\d__[0-9a-f]{8}", name)
        for name in names
    )


async def test_aggregating_an_exception_fails_the_workflow(
    harbor_client: Client, tmp_path: Path
) -> None:
    cause, _ = await _planning_failure(
        harbor_client, AggregateExceptions, _config(tmp_path)
    )
    assert isinstance(cause, ApplicationError)
    assert cause.type == "ValidationError"
    assert cause.non_retryable
    assert "aggregate_job's arguments are not valid" in cause.message


async def test_unfinished_job_has_no_finished_at(
    harbor_client: Client, tmp_path: Path
) -> None:
    result = await _run(harbor_client, PlanAndAggregate, _config(tmp_path))
    assert result.stats.n_pending_trials == 8
    assert result.finished_at is None
    written = JobResult.model_validate_json(
        (tmp_path / "jobs" / "expansion" / "result.json").read_text()
    )
    assert written.finished_at is None


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


@pytest.mark.parametrize(
    ("workflow_class", "make_arg", "function", "argument"),
    [
        (PlanWithSecondsOfTimeout, _config, "plan_job", "start_to_close_timeout"),
        (
            RunTrialWithSecondsOfHeartbeat,
            _one_trial,
            "execute_trial",
            "heartbeat_timeout",
        ),
        (
            AggregateWithSecondsOfTimeout,
            _config,
            "aggregate_job",
            "start_to_close_timeout",
        ),
        (PlanWithZeroTimeout, _config, "plan_job", "start_to_close_timeout"),
    ],
    ids=["plan_job", "execute_trial", "aggregate_job", "zero"],
)
async def test_timeout_that_is_not_a_positive_timedelta_fails_the_workflow(
    harbor_client: Client,
    tmp_path: Path,
    workflow_class: type[Any],
    make_arg: Callable[[Path], Any],
    function: str,
    argument: str,
) -> None:
    cause, _ = await _planning_failure(
        harbor_client, workflow_class, make_arg(tmp_path)
    )
    assert isinstance(cause, ApplicationError)
    assert cause.type == "ValidationError"
    assert cause.non_retryable
    assert f"{function}'s arguments are not valid" in cause.message
    assert argument in cause.message


async def test_concurrency_caps_are_reported_unused(
    harbor_client: Client, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config(
        tmp_path,
        agents=[AgentConfig(name="oracle", n_concurrent=1, concurrency_group="g")],
    )
    await _run(harbor_client, PlanJob, config)
    assert any("n_concurrent" in r.getMessage() for r in caplog.records)


@on_posix
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
        f"org/ds@{dataset.version or 'latest'}",
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


@pytest.mark.parametrize("action", ["regrade", "diff"])
async def test_jobs_from_source_jobs_rejected(
    harbor_client: Client, tmp_path: Path, action: Literal["regrade", "diff"]
) -> None:
    config = _config(
        tmp_path,
        source_jobs=[
            SourceJobConfig(action=action, type="local", path=tmp_path / "old")
        ],
    )
    cause, handle = await _planning_failure(harbor_client, PlanJob, config)
    assert isinstance(cause, ApplicationError)
    assert cause.type == "HarborSourceJobsUnsupported"
    assert cause.non_retryable
    assert await history.scheduled(handle) == []


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


@pytest.mark.parametrize(
    "workflow_class",
    [PlanCollidingJob, PlanCaseCollidingJob],
    ids=["same", "differ-in-case"],
)
async def test_colliding_trial_names_rejected(
    harbor_client: Client, tmp_path: Path, workflow_class: type[Any]
) -> None:
    cause, _ = await _planning_failure(harbor_client, workflow_class, _config(tmp_path))
    assert isinstance(cause, ApplicationError)
    assert cause.type == "HarborDuplicateTrialName"


@pytest.mark.parametrize(
    "workflow_class", [PlanEscapingJob, PlanDriveNameJob], ids=["parent", "drive"]
)
async def test_trial_names_outside_the_job_rejected(
    harbor_client: Client, tmp_path: Path, workflow_class: type[Any]
) -> None:
    cause, _ = await _planning_failure(harbor_client, workflow_class, _config(tmp_path))
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
