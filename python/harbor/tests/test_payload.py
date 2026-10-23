"""What a trial Activity returns is small, and enough for harbor's aggregation."""

from __future__ import annotations

import dataclasses
import sys
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from harbor.metrics.mean import Mean
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.job.config import JobConfig
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

from temporalio.api.common.v1 import Payload
from temporalio.client import Client
from temporalio.contrib.pydantic import (
    PydanticPayloadConverter,
    pydantic_data_converter,
)
from temporalio.converter import (
    DataConverter,
    DefaultFailureConverterWithEncodedAttributes,
    PayloadCodec,
)
from temporalio.harbor import HarborPlugin
from temporalio.harbor._activity import _cap, _strip
from tests.harbor_fixtures import agents, tasks
from tests.harbor_fixtures.workflows import (
    PlanJob,
    RunJob,
    RunOneTrial,
    TrialArgs,
    new_worker,
)

on_posix = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the local test environment runs task scripts with bash",
)


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


def _exception(message: str = "no reward") -> ExceptionInfo:
    return ExceptionInfo(
        exception_type="RewardFileNotFoundError",
        exception_message=message,
        exception_traceback="frame\n" * 5_000 + "the frame that failed\n",
        occurred_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def _result(
    task: str, trial: str, reward: float | None, *, steps: bool = False
) -> TrialResult:
    config = TrialConfig(task=TaskConfig(path=Path("/tasks") / task), trial_name=trial)
    exception = _exception() if reward is None else None
    verifier = (
        VerifierResult(rewards={"reward": reward}) if reward is not None else None
    )
    agent_result = None if steps else _heavy_context(1000)
    step_results = (
        [
            StepResult(
                step_name=f"step-{i}",
                agent_result=_heavy_context(100),
                exception_info=_exception() if reward is None else None,
            )
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
    slimmed = [_cap(_strip(r)) for r in full]

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


def test_slim_caps_every_recorded_exception() -> None:
    full = _result("b", "b__1", None, steps=True)
    assert full.step_results is not None
    long = _exception("no reward in the file the verifier wrote\n" + "y" * 20_000)
    result = _cap(
        _strip(
            full.model_copy(
                update={
                    "exception_info": long,
                    "step_results": [
                        step.model_copy(update={"exception_info": long})
                        for step in full.step_results
                    ],
                }
            )
        )
    )
    assert result.step_results is not None
    recorded = [result.exception_info] + [s.exception_info for s in result.step_results]
    assert len(recorded) == 3
    for exc in recorded:
        assert exc is not None
        # A message leads with what went wrong; a traceback ends with where.
        assert exc.exception_message.startswith("no reward in the file")
        assert len(exc.exception_message) <= 8_000
        assert exc.exception_traceback.endswith("the frame that failed\n")
        assert len(exc.exception_traceback) <= 8_000


@on_posix
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

    assert returned.output is None
    returned = returned.result
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


class _Counting(PayloadCodec):
    """Marks every payload it encodes, and refuses any it did not."""

    def __init__(self) -> None:
        self.encoded = 0

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        self.encoded += len(payloads)
        return [
            Payload(metadata={**p.metadata, "counted": b"1"}, data=p.data)
            for p in payloads
        ]

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        assert all(p.metadata.get("counted") == b"1" for p in payloads)
        return [
            Payload(
                metadata={k: v for k, v in p.metadata.items() if k != "counted"},
                data=p.data,
            )
            for p in payloads
        ]


@on_posix
async def test_plugin_keeps_a_payload_codec(client: Client, tmp_path: Path) -> None:
    codec = _Counting()
    config = client.config()
    config["data_converter"] = dataclasses.replace(
        DataConverter.default, payload_codec=codec
    )
    config["plugins"] = [HarborPlugin()]
    coded = Client(**config)
    assert coded.data_converter.payload_codec is codec
    assert isinstance(coded.data_converter.payload_converter, PydanticPayloadConverter)

    job = JobConfig(
        job_name="coded",
        jobs_dir=tmp_path / "jobs",
        tasks=[TaskConfig(path=tasks.passing(tmp_path / "tasks"))],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    async with new_worker(coded, PlanJob) as worker:
        plan = await coded.execute_workflow(
            PlanJob.run,
            job,
            id=f"plan-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=60),
        )
    assert len(plan.trials) == 1
    assert codec.encoded > 0


def test_plugin_leaves_a_custom_payload_converter() -> None:
    converter = dataclasses.replace(pydantic_data_converter, payload_codec=_Counting())
    configured = HarborPlugin().configure_client(
        {"data_converter": converter}  # type: ignore[typeddict-item]
    )
    assert configured["data_converter"] is converter


def test_plugin_keeps_a_failure_converter() -> None:
    converter = dataclasses.replace(
        DataConverter.default,
        failure_converter_class=DefaultFailureConverterWithEncodedAttributes,
    )
    configured = HarborPlugin().configure_client(
        {"data_converter": converter}  # type: ignore[typeddict-item]
    )["data_converter"]
    assert configured.payload_converter_class is PydanticPayloadConverter
    assert (
        configured.failure_converter_class
        is DefaultFailureConverterWithEncodedAttributes
    )


@on_posix
async def test_metric_that_returns_non_numbers(
    harbor_client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def compute(_mean: Mean, rewards: list[Any]) -> dict[str, Any]:
        return {"mean": "not a number", "rewards": [r for r in rewards if r]}

    monkeypatch.setattr(Mean, "compute", compute)
    job = JobConfig(
        job_name="labelled",
        jobs_dir=tmp_path / "jobs",
        tasks=[TaskConfig(path=tasks.passing(tmp_path / "tasks"))],
        environment=EnvironmentConfig(import_path=tasks.LOCAL_ENV),
    )
    async with new_worker(harbor_client, RunJob) as worker:
        result = await harbor_client.execute_workflow(
            RunJob.run,
            job,
            id=f"job-{uuid.uuid4()}",
            task_queue=worker.task_queue,
            execution_timeout=timedelta(seconds=60),
        )
    [evals] = result.stats.evals.values()
    assert evals.metrics == [{"mean": "not a number", "rewards": [{"reward": 1.0}]}]
