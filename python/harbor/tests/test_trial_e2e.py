"""Real harbor trials through the plugin, compared with harbor's own Job."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobStats
from harbor.models.metric.config import MetricConfig
from harbor.models.metric.type import MetricType
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)

from temporalio.client import Client
from tests.harbor_fixtures import history, tasks
from tests.harbor_fixtures.workflows import RunJob, RunOneTrial, TrialArgs, new_worker

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)


def _job(
    jobs_dir: Path, task_list: list[TaskConfig], metrics: list[MetricConfig]
) -> JobConfig:
    return JobConfig(
        job_name="fidelity",
        jobs_dir=jobs_dir,
        quiet=True,
        tasks=task_list,
        n_attempts=2,
        agents=[AgentConfig(name="oracle")],
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


def _expected(metric: str, values: tuple[float, float]) -> dict[str, Any]:
    ds_a, ds_b = values
    return {
        "n_completed_trials": 6,
        "n_running_trials": 0,
        "n_pending_trials": 0,
        "n_cancelled_trials": 0,
        "n_errored_trials": 2,
        "n_retries": 0,
        "n_input_tokens": None,
        "n_cache_tokens": None,
        "n_output_tokens": None,
        "cost_usd": None,
        "evals": {
            "oracle__ds-a": {
                "n_trials": 4,
                "n_errors": 0,
                "metrics": [{metric: ds_a}],
                "pass_at_k": {"2": 0.5},
                "reward_stats": {
                    "reward": {
                        "0.0": ["stays-silent", "stays-silent"],
                        "1.0": ["answers", "answers"],
                    }
                },
                "exception_stats": {},
            },
            "oracle__ds-b": {
                "n_trials": 0,
                "n_errors": 2,
                "metrics": [{metric: ds_b}],
                "pass_at_k": {"2": 0.0},
                "reward_stats": {},
                "exception_stats": {
                    "RewardFileNotFoundError": ["no-reward", "no-reward"]
                },
            },
        },
    }


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ([], _expected("mean", (0.5, 0.0))),
        ([MetricConfig(type=MetricType.MAX)], _expected("max", (1.0, 0.0))),
    ],
    ids=["dataset-default", "job-metric"],
)
async def test_plugin_job_matches_harbor_job(
    harbor_client: Client,
    tmp_path: Path,
    metrics: list[MetricConfig],
    expected: dict[str, Any],
) -> None:
    root = tmp_path / "tasks"
    task_list = [
        TaskConfig(path=tasks.passing(root), source="ds-a"),
        TaskConfig(path=tasks.failing(root), source="ds-a"),
        TaskConfig(path=tasks.no_reward(root), source="ds-b"),
    ]

    job = await Job.create(_job(tmp_path / "harbor", task_list, metrics))
    reference = await job.run()

    async with new_worker(harbor_client, RunJob) as worker:
        stats = await harbor_client.execute_workflow(
            RunJob.run,
            _job(tmp_path / "plugin", task_list, metrics),
            id=f"job-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )

    assert _comparable(stats) == _comparable(reference.stats)
    assert _comparable(stats) == expected


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
