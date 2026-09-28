"""What a trial Activity returns is small, and enough for harbor's aggregation."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.job.result import JobStats
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.models.trial.result import AgentInfo, ExceptionInfo, StepResult, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.utils.pass_at_k import compute_pass_at_k_by_evals

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.harbor._activity import slim
from tests.harbor_fixtures import agents, tasks
from tests.harbor_fixtures.workflows import RunOneTrial, TrialArgs, new_worker


def _heavy_context(tokens: int) -> AgentContext:
    return AgentContext(
        n_input_tokens=tokens,
        n_cache_tokens=tokens // 2,
        n_output_tokens=tokens // 10,
        cost_usd=tokens / 1000,
        rollout_details=[
            {
                "prompt_token_ids": [list(range(4000))],
                "completion_token_ids": [list(range(4000))],
                "logprobs": [[-0.25] * 4000],
            }
        ],
        metadata={"transcript": "x" * 20_000},
    )


def _result(
    task: str, trial: str, reward: float | None, *, steps: bool = False
) -> TrialResult:
    config = TrialConfig(task=TaskConfig(path=Path("/tasks") / task), trial_name=trial)
    exception = None
    if reward is None:
        exception = ExceptionInfo(
            exception_type="RewardFileNotFoundError",
            exception_message="no reward",
            exception_traceback="frame\n" * 5_000 + "the frame that failed\n",
            occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
    verifier = (
        VerifierResult(rewards={"reward": reward}) if reward is not None else None
    )
    agent_result = None if steps else _heavy_context(1000)
    step_results = (
        [
            StepResult(step_name=f"step-{i}", agent_result=_heavy_context(100))
            for i in range(2)
        ]
        if steps
        else None
    )
    return TrialResult(
        task_name=task,
        trial_name=trial,
        trial_uri=f"file:///trials/{trial}",
        task_id=config.task.get_task_id(),
        source="ds",
        task_checksum="0" * 64,
        config=config,
        agent_info=AgentInfo(name=AgentName.ORACLE.value, version="1"),
        agent_result=agent_result,
        verifier_result=verifier,
        exception_info=exception,
        step_results=step_results,
    )


def _size(result: TrialResult) -> int:
    return len(pydantic_data_converter.payload_converter.to_payloads([result])[0].data)


def _stats(results: list[TrialResult]) -> dict[str, Any]:
    return JobStats.from_trial_results(
        results, n_total_trials=len(results)
    ).model_dump()


def test_slim_results_aggregate_identically() -> None:
    full = [
        _result("a", "a__1", 1.0),
        _result("a", "a__2", 0.0),
        _result("b", "b__1", None),
        _result("b", "b__2", 1.0, steps=True),
    ]
    slimmed = [slim(r) for r in full]

    assert _stats(slimmed) == _stats(full)
    assert compute_pass_at_k_by_evals(slimmed) == compute_pass_at_k_by_evals(full)

    for before, after in zip(full, slimmed):
        assert _size(after) < _size(before) / 10
        contexts = [after.agent_result] + [
            s.agent_result for s in after.step_results or []
        ]
        assert all(
            c is None or (c.rollout_details is None and c.metadata is None)
            for c in contexts
        )


def test_long_traceback_keeps_the_failing_frames() -> None:
    result = slim(_result("b", "b__1", None))
    assert result.exception_info is not None
    assert result.exception_info.exception_traceback.endswith("the frame that failed\n")
    assert len(result.exception_info.exception_traceback) <= 8_000


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)
async def test_trial_activity_returns_slim_result(
    harbor_client: Client, tmp_path: Path
) -> None:
    trials_dir = tmp_path / "trials"
    config = TrialConfig(
        task=TaskConfig(path=tasks.passing(tmp_path / "tasks")),
        trial_name="answers__0",
        trials_dir=trials_dir,
        agent=AgentConfig(import_path=agents.TRANSCRIPT_AGENT),
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    async with new_worker(harbor_client, RunOneTrial) as worker:
        returned = await harbor_client.execute_workflow(
            RunOneTrial.run,
            TrialArgs(config=config),
            id=f"trial-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )

    assert returned.verifier_result is not None
    assert returned.verifier_result.rewards == {"reward": 1.0}
    assert returned.agent_result is not None
    assert returned.agent_result.rollout_details is None
    assert returned.agent_result.metadata is None
    assert returned.agent_result.n_input_tokens == agents.TOKENS
    assert returned.agent_result.cost_usd == 0.25

    # The complete result is still where harbor wrote it.
    on_disk = TrialResult.model_validate_json(
        (trials_dir / "answers__0" / "result.json").read_text()
    )
    assert on_disk.agent_result is not None
    assert on_disk.agent_result.rollout_details is not None
    assert _size(returned) < _size(on_disk) / 10
