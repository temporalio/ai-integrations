"""Recorded histories replay cleanly through the plugin.

The Deep Agents control loop runs in the workflow, so replay determinism is the
core safety property. This records the history of a real run (a plain fake agent
driven by ``run_deep_agent``, no LangChain needed) and feeds it back through a
``Replayer`` configured with the plugin. A nondeterministic seam would raise on
replay; a clean pass proves the in-workflow dispatch is deterministic.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

import pytest

from temporalio import workflow
from temporalio.deepagents import DeepAgentsPlugin, TemporalModel, run_deep_agent
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from tests.helpers.deepagents import scheduled_attributes

with workflow.unsafe.imports_passed_through():
    from langchain_core.messages import HumanMessage

    from temporalio.deepagents.testing import mock_model_provider


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


@workflow.defn(name="DefaultHeartbeatModelWorkflow")
class DefaultHeartbeatModelWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        model = TemporalModel(model="fake:model")
        message = await model.ainvoke([HumanMessage(content=prompt)])
        return str(message.content)


@pytest.mark.asyncio
async def test_history_without_heartbeat_timeout_replays(
    env: WorkflowEnvironment,
) -> None:
    """A model call scheduled without a heartbeat timeout, as before the default
    existed (recorded here through the ``heartbeat_timeout=None`` opt-out),
    replays through the default configuration: activity options such as
    ``heartbeat_timeout`` are not part of command matching, so the default is
    replay-safe for runs in flight when a worker is upgraded."""
    async with Worker(
        env.client,
        task_queue="da-replay-no-hb",
        workflows=[DefaultHeartbeatModelWorkflow],
        plugins=[
            DeepAgentsPlugin(
                model_provider=mock_model_provider(["ok"]),
                model_activity_options={"heartbeat_timeout": None},
            )
        ],
    ):
        handle = await env.client.start_workflow(
            DefaultHeartbeatModelWorkflow.run,
            "hi",
            id=f"da-replay-no-hb-{uuid.uuid4()}",
            task_queue="da-replay-no-hb",
        )
        assert await handle.result() == "ok"
        history = await handle.fetch_history()
    attrs = await scheduled_attributes(handle, "deepagents.invoke_model")
    assert attrs.heartbeat_timeout.ToTimedelta() == timedelta(0)
    await Replayer(
        workflows=[DefaultHeartbeatModelWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(history)
