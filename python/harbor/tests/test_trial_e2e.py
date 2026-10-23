"""Real harbor trials through the plugin, compared with harbor's own Job."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from harbor.job import Job
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.job.lock import JobLock
from harbor.models.job.result import JobStats
from harbor.models.metric.config import MetricConfig
from harbor.models.metric.type import MetricType
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.viewer.scanner import JobScanner

from temporalio.client import Client
from tests.harbor_fixtures import agents, history, tasks
from tests.harbor_fixtures.workflows import RunJob, RunOneTrial, TrialArgs, new_worker

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)

_ATTEMPTS = 3


def _job(jobs_dir: Path, root: Path, metrics: list[MetricConfig]) -> JobConfig:
    return JobConfig(
        job_name="fidelity",
        jobs_dir=jobs_dir,
        quiet=True,
        datasets=[DatasetConfig(path=root / "ds-a")],
        tasks=[TaskConfig(path=root / "no-reward", source="ds-b")],
        n_attempts=_ATTEMPTS,
        agents=[
            AgentConfig(name="oracle"),
            AgentConfig(import_path=agents.TRANSCRIPT_AGENT, model_name="test/model"),
        ],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
        metrics=metrics,
    )


def _by_task(names: list[str]) -> list[str]:
    # Trial names end in a random suffix that differs between any two runs.
    return sorted(name.split("__")[0] for name in names)


def _comparable(stats: JobStats) -> dict[str, Any]:
    data = stats.model_dump(mode="json")
    for evals in data["evals"].values():
        evals["reward_stats"] = {
            reward: {value: _by_task(names) for value, names in by_value.items()}
            for reward, by_value in evals["reward_stats"].items()
        }
        evals["exception_stats"] = {
            kind: _by_task(names) for kind, names in evals["exception_stats"].items()
        }
    return data


@pytest.mark.parametrize(
    "metrics",
    [[], [MetricConfig(type=MetricType.MAX)]],
    ids=["dataset-default", "job-metric"],
)
async def test_plugin_job_matches_harbor_job(
    harbor_client: Client, tmp_path: Path, metrics: list[MetricConfig]
) -> None:
    root = tmp_path / "tasks"
    tasks.passing(root / "ds-a")
    tasks.failing(root / "ds-a")
    tasks.no_reward(root)

    reference = await (await Job.create(_job(tmp_path / "harbor", root, metrics))).run()

    async with new_worker(harbor_client, RunJob) as worker:
        result = await harbor_client.execute_workflow(
            RunJob.run,
            _job(tmp_path / "plugin", root, metrics),
            id=f"job-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )

    assert _comparable(result.stats) == _comparable(reference.stats)
    stats = result.stats
    n_trials = 3 * 2 * _ATTEMPTS
    assert (stats.n_completed_trials, stats.n_errored_trials, stats.n_retries) == (
        n_trials,
        2 * _ATTEMPTS,
        0,
    )
    # Only the transcript agent reports usage, once per trial it ran.
    assert stats.n_input_tokens == 3 * _ATTEMPTS * agents.TOKENS
    assert stats.cost_usd == pytest.approx(3 * _ATTEMPTS * 0.25)
    assert len(stats.evals) == 4
    assert all(evals.metrics for evals in stats.evals.values())
    assert result.n_total_trials == n_trials
    assert len(result.trial_results) == n_trials

    # Harbor's own tools read the plugin's job directory as they read harbor's.
    plugin = JobScanner(tmp_path / "plugin")
    harbor = JobScanner(tmp_path / "harbor")
    plugin_config = plugin.get_job_config("fidelity")
    harbor_config = harbor.get_job_config("fidelity")
    assert plugin_config is not None and harbor_config is not None
    assert plugin_config.model_copy(update={"jobs_dir": Path()}) == (
        harbor_config.model_copy(update={"jobs_dir": Path()})
    )
    plugin_result = plugin.get_job_result("fidelity")
    harbor_result = harbor.get_job_result("fidelity")
    assert plugin_result is not None and harbor_result is not None
    assert plugin_result.id == result.id
    assert plugin_result.n_total_trials == harbor_result.n_total_trials
    assert plugin_result.finished_at is not None
    assert _comparable(plugin_result.stats) == _comparable(harbor_result.stats)

    job_dir = tmp_path / "plugin" / "fidelity"
    harbor_dir = tmp_path / "harbor" / "fidelity"
    assert JobLock.model_validate_json((job_dir / "lock.json").read_text()) == (
        JobLock.model_validate_json((harbor_dir / "lock.json").read_text())
    )
    names = sorted(r.trial_name for r in result.trial_results)
    assert plugin.list_trials("fidelity") == names
    assert sorted(p.name for p in job_dir.iterdir()) == sorted(
        ["config.json", "lock.json", "result.json", *names]
    )


async def test_activity_summaries(harbor_client: Client, tmp_path: Path) -> None:
    config = TrialConfig(
        task=TaskConfig(path=tasks.passing(tmp_path / "tasks")),
        trial_name="answers__0000000",
        trials_dir=tmp_path / "trials",
        agent=AgentConfig(name="oracle", model_name="none"),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    async with new_worker(harbor_client, RunOneTrial) as worker:
        found = []
        for summary in (None, "grading the answer"):
            handle = await harbor_client.start_workflow(
                RunOneTrial.run,
                TrialArgs(
                    config=config.model_copy(
                        update={"trial_name": f"answers__{uuid.uuid4().hex[:7]}"}
                    ),
                    summary=summary,
                ),
                id=f"summary-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
            await handle.result()
            found += await history.summaries(handle, harbor_client.data_converter)
    assert found == ["answers · oracle/none", "grading the answer"]
