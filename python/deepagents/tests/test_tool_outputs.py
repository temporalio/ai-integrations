"""Wrapped tools preserve native result metadata without changing old histories."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError, WorkflowHistory
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from tests.helpers.deepagents import count_scheduled_activities

with workflow.unsafe.imports_passed_through():
    from langchain_core.documents import Document
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import ToolException, tool

    from temporalio.deepagents import DeepAgentsPlugin, tool_as_activity


@tool(response_format="content_and_artifact")
async def artifact_tool(query: str) -> tuple[str, dict]:
    """Return a retrieval result and its source record."""
    return "matching source", {"source": query, "record": 42}


@tool(response_format="content_and_artifact")
async def document_tool(query: str) -> tuple[str, Document]:
    """Return a retrieval result with a native Document artifact."""
    return "matching source", Document(page_content=query)


@tool(return_direct=True)
async def direct_tool(query: str) -> str:
    """Return a result without another model call."""
    return query


@tool
async def error_tool(query: str) -> str:
    """Return a handled tool failure."""
    raise ToolException(f"lookup unavailable: {query}")


error_tool.handle_tool_error = True


@workflow.defn
class ToolOutputWorkflow:
    @workflow.run
    async def run(self, which: str) -> dict:
        original = (
            artifact_tool
            if which in {"artifact", "bare", "empty-id"}
            else document_tool
            if which == "document"
            else direct_tool
            if which == "direct"
            else error_tool
        )
        call: dict[str, Any] = {
            "name": original.name,
            "args": {"query": "manual"},
            "id": "" if which == "empty-id" else "model-call",
            "type": "tool_call",
        }
        argument = call["args"] if which == "bare" else call
        native = await original.ainvoke(argument)
        wrapped = tool_as_activity(
            original, start_to_close_timeout=timedelta(seconds=10)
        )
        durable = await wrapped.ainvoke(argument)
        if which == "bare":
            return {"native": native, "durable": durable}

        def describe(message: ToolMessage) -> dict[str, Any]:
            artifact = message.artifact
            return {
                "content": message.content,
                "artifact": artifact.model_dump()
                if isinstance(artifact, Document)
                else artifact,
                "artifact_class": type(artifact).__name__,
                "status": message.status,
                "call_id": message.tool_call_id,
            }

        return {
            "native": describe(native),
            "durable": describe(durable),
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "which", ["artifact", "document", "status", "bare", "empty-id"]
)
async def test_wrapped_output_matches_native(
    env: WorkflowEnvironment, which: str
) -> None:
    task_queue = f"ga-output-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[ToolOutputWorkflow],
        plugins=[DeepAgentsPlugin()],
    ):
        handle = await env.client.start_workflow(
            ToolOutputWorkflow.run, which, id=str(uuid.uuid4()), task_queue=task_queue
        )
        out = await handle.result()
    await Replayer(
        workflows=[ToolOutputWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(await handle.fetch_history())
    assert out["durable"] == out["native"], out
    for event in (await handle.fetch_history()).events:
        if event.HasField("activity_task_scheduled_event_attributes"):
            payload = event.activity_task_scheduled_event_attributes.input.payloads[0]
            actual = json.loads(payload.data)
            if which != "bare":
                assert actual["tool_call_id"] == (
                    "" if which == "empty-id" else "model-call"
                )


async def test_return_direct_is_refused_before_a_tool_runs(
    env: WorkflowEnvironment,
) -> None:
    queue = f"direct-tool-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=queue,
        workflows=[ToolOutputWorkflow],
        plugins=[DeepAgentsPlugin()],
    ):
        handle = await env.client.start_workflow(
            ToolOutputWorkflow.run, "direct", id=str(uuid.uuid4()), task_queue=queue
        )
        with pytest.raises(
            WorkflowFailureError, match="Workflow execution failed"
        ) as failure:
            await handle.result()
    assert "return_direct=True" in str(failure.value.cause)
    assert not (await count_scheduled_activities(handle))["deepagents.invoke_tool"]


@activity.defn(name="legacy_tool_observation")
async def legacy_tool_observation() -> str:
    return "observed"


@activity.defn(name="complete_tool_observation")
async def complete_tool_observation() -> str:
    return "observed"


@workflow.defn
class LegacyToolCallWorkflow:
    @workflow.run
    async def run(self) -> str:
        wrapped = tool_as_activity(
            artifact_tool, start_to_close_timeout=timedelta(seconds=10)
        )
        message = await wrapped.ainvoke(
            {
                "name": artifact_tool.name,
                "args": {"query": "legacy"},
                "id": "recorded-model-call",
                "type": "tool_call",
            }
        )
        observer = (
            "legacy_tool_observation"
            if message.artifact is None
            else "complete_tool_observation"
        )
        return await workflow.execute_activity(
            observer,
            start_to_close_timeout=timedelta(seconds=10),
            activity_id=workflow.uuid4().hex,
        )


async def test_new_tool_results_replay(env: WorkflowEnvironment) -> None:
    queue = f"complete-tool-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=queue,
        workflows=[LegacyToolCallWorkflow],
        activities=[legacy_tool_observation, complete_tool_observation],
        plugins=[DeepAgentsPlugin()],
    ):
        handle = await env.client.start_workflow(
            LegacyToolCallWorkflow.run, id=str(uuid.uuid4()), task_queue=queue
        )
        assert await handle.result() == "observed"
    counts = await count_scheduled_activities(handle)
    assert counts["complete_tool_observation"] == 1
    await Replayer(
        workflows=[LegacyToolCallWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(await handle.fetch_history())


async def test_pre_context_history_keeps_tool_results_and_uuid_sequence() -> None:
    # Recorded with the unchanged tool adapter at e9ade41b plus #605-#608,
    # on Temporal Server 1.32.2, using LegacyToolCallWorkflow above. The observer's
    # Activity type checks the old result, and its id checks the next RNG draw.
    history = (
        Path(__file__).parent
        / "histories"
        / "0.x"
        / "wrapped_tool_before_call_context.json"
    )
    await Replayer(
        workflows=[LegacyToolCallWorkflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(
        WorkflowHistory.from_json(
            "wrapped-tool-before-call-context", history.read_text()
        )
    )
