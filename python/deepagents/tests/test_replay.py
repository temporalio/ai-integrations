"""Recorded histories replay cleanly through the plugin.

The Deep Agents control loop runs in the workflow, so replay determinism is the
core safety property. This records the history of a real run (a plain fake agent
driven by ``run_deep_agent``, no LangChain needed) and feeds it back through a
``Replayer`` configured with the plugin. A nondeterministic seam would raise on
replay; a clean pass proves the in-workflow dispatch is deterministic.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.testing import WorkflowEnvironment

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)
from temporalio import workflow
from temporalio.client import WorkflowHistory
from temporalio.deepagents import DeepAgentsPlugin, TemporalModel, run_deep_agent
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from langchain_core.messages import HumanMessage


class FakeAgent:
    async def ainvoke(self, input: Any) -> dict:
        messages = list(input.get("messages", [])) if isinstance(input, dict) else []
        return {"messages": [*messages, "answered"], "todos": []}


@workflow.defn
class ReplayWorkflow:
    @workflow.run
    async def run(self, input: dict) -> dict:
        return await run_deep_agent(FakeAgent(), input)


@pytest.mark.asyncio
async def test_replay_with_plugin(env: WorkflowEnvironment) -> None:
    plugin = DeepAgentsPlugin()
    async with Worker(
        env.client,
        task_queue="da-replay",
        workflows=[ReplayWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ReplayWorkflow.run,
            {"messages": ["question"]},
            id=f"da-replay-{uuid.uuid4()}",
            task_queue="da-replay",
        )
        await handle.result()
        history = await handle.fetch_history()

    # A fresh replayer (new worker identity) must replay the recorded history
    # without a nondeterminism error.
    replayer = Replayer(
        workflows=[ReplayWorkflow],
        plugins=[DeepAgentsPlugin()],
    )
    await replayer.replay_workflow(history)


# A history recorded on upstream main (63d7aec), before the default heartbeat
# timeout existed: deepagents.invoke_model scheduled with
# start_to_close=300s and heartbeat_timeout=0. To regenerate it, check out a
# pre-default commit, run DefaultHeartbeatModelWorkflow through a worker built
# from it (DeepAgentsPlugin(model_provider=mock_model_provider(["ok"]))), dump
# the finished run's history with handle.fetch_history(), and scrub the
# run-identifying identity / request IDs (as this copy has been).
_PRE_HEARTBEAT_HISTORY = (
    Path(__file__).parent / "histories" / "model_call_without_heartbeat.json"
)


@workflow.defn(name="DefaultHeartbeatModelWorkflow")
class DefaultHeartbeatModelWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        model = TemporalModel(model="fake:model")
        message = await model.ainvoke([HumanMessage(content=prompt)])
        return str(message.content)


@pytest.mark.asyncio
async def test_pre_heartbeat_history_replays() -> None:
    """A history recorded without the default heartbeat timeout replays
    unchanged: activity options such as ``heartbeat_timeout`` are not part of
    command matching, so adding the default is replay-safe for workflows still
    in flight when a worker is upgraded."""
    plugin = DeepAgentsPlugin()
    replayer = Replayer(workflows=[DefaultHeartbeatModelWorkflow], plugins=[plugin])
    history = WorkflowHistory.from_json(
        "pre-heartbeat-replay", _PRE_HEARTBEAT_HISTORY.read_text()
    )
    await replayer.replay_workflow(history)
