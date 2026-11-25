"""Locus's clock reads and ids are the workflow's, so a replayed run agrees with History."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from locus.core.messages import ToolCall
from locus.core.state import AgentState
from locus.core.termination import TimeLimit
from locus.models.base import ModelProtocol

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.locus import LocusPlugin
from temporalio.worker import Worker


@activity.defn
async def echo(values: list[str]) -> list[str]:
    return values


def _no_model() -> ModelProtocol:
    raise AssertionError("no model call is expected")


def _read() -> list[str]:
    state = AgentState()
    return [
        state.run_id,
        state.started_at.isoformat(),
        ToolCall(name="t").id,
    ]


@workflow.defn
class ReadsWorkflow:
    @workflow.run
    async def run(self) -> dict[str, object]:
        first = _read()
        # With no workflow cache this activity's completion replays the code
        # above: the values recorded as its input must be read again unchanged.
        recorded = await workflow.execute_activity(
            echo, first, start_to_close_timeout=timedelta(seconds=10)
        )
        limit = TimeLimit(seconds=1)
        before_timer = limit.check(AgentState())[0]
        await workflow.sleep(timedelta(seconds=2))
        return {
            "replayed_agree": recorded == first,
            "started_at": first[1],
            "workflow_now": workflow.now().isoformat(),
            "limit_before_timer": before_timer,
            "limit_after_timer": limit.check(AgentState())[0],
        }


async def test_locus_reads_are_workflow_values_and_survive_replay(
    client: Client,
) -> None:
    config = client.config()
    config["plugins"] = [LocusPlugin(models={"unused": _no_model})]
    client = Client(**config)
    task_queue = f"locus-{uuid.uuid4()}"
    before = datetime.now(UTC)
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ReadsWorkflow],
        activities=[echo],
        max_cached_workflows=0,
    ):
        result = await client.execute_workflow(
            ReadsWorkflow.run,
            id=f"reads-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
    assert result["replayed_agree"] is True
    # TimeLimit runs on the workflow clock, which the 2 s timer moved past it.
    assert result["limit_before_timer"] is False
    assert result["limit_after_timer"] is True
    assert datetime.fromisoformat(result["started_at"]) >= before - timedelta(seconds=5)  # type: ignore[arg-type]

    # Outside a workflow, Locus keeps the real clock and random ids.
    assert AgentState().run_id != AgentState().run_id
    assert abs((AgentState().started_at - datetime.now(UTC)).total_seconds()) < 5
