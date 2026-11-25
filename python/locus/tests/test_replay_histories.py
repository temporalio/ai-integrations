"""Histories recorded by earlier builds of the plugin still replay.

Replay is promised within a major (see AGENTS.md), so the histories in
``histories/0.x`` have to replay until 1.0 ships. Each was recorded by running
the workflow below on a local dev server with the scripted model and
locus-sdk 5.4.0. A change to the plugin, or a Locus release, that makes the
agent loop schedule different work fails here.

To record a history for a new workflow, run it as the tests in
``test_behaviors.py`` do and write ``(await handle.fetch_history()).to_json()``
to ``histories/<major>.x/<name>.json``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from locus.agent import Agent
from locus.core.events import ModelChunkEvent
from locus.core.messages import Message, Role, ToolCall
from locus.models.base import ModelResponse
from locus.tools.decorator import tool

from temporalio import activity, workflow
from temporalio.client import WorkflowHistory
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.locus import LocusPlugin, TemporalModel, activity_as_tool
from temporalio.worker import Replayer

# The plugin's current major, the major of its next release.
REPLAY_MAJOR = 0
HISTORIES = Path(__file__).parent / "histories" / f"{REPLAY_MAJOR}.x"
TIMEOUT = timedelta(seconds=30)


class ReplayModel:
    """Looks an order up, refunds it, then answers. Not called during replay."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        del kwargs
        names = [t["function"]["name"] for t in tools or []]
        results = [m for m in messages if m.role == Role.TOOL]
        if len(results) < len(names):
            name = names[len(results)]
            arguments: dict[str, Any] = {"order_id": "ORD-7842"}
            return ModelResponse(
                message=Message.assistant(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=f"call_{len(results) + 1}",
                            name=name,
                            arguments=arguments,
                        )
                    ],
                ),
                stop_reason="tool_calls",
            )
        answer = f"Done: {results[-1].content}" if results else "Hello there."
        return ModelResponse(
            message=Message.assistant(content=answer), stop_reason="stop"
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        response = await self.complete(messages, tools, **kwargs)
        for word in (response.message.content or "").split(" "):
            if word:
                yield ModelChunkEvent(content=word + " ")
        yield ModelChunkEvent(tool_calls=response.message.tool_calls or None, done=True)


@activity.defn(name="replay_lookup_order")
async def lookup_order(order_id: str) -> str:
    """Return an order's status."""
    return f"{order_id} delivered damaged"


@activity.defn(name="replay_issue_refund")
async def issue_refund(order_id: str) -> str:
    """Refund an order."""
    return f"refunded {order_id}"


def _model(**options: Any) -> TemporalModel:
    return TemporalModel("replay", start_to_close_timeout=TIMEOUT, **options)


@workflow.defn(name="LocusReplayToolCalls")
class ToolCallsWorkflow:
    """Two Activity tools in turn, with unstreamed model calls."""

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=_model(),
            tools=[
                activity_as_tool(lookup_order, start_to_close_timeout=TIMEOUT),
                activity_as_tool(issue_refund, start_to_close_timeout=TIMEOUT),
            ],
        )
        return (await agent.arun(prompt)).message or ""


@workflow.defn(name="LocusReplayApproval")
class ApprovalWorkflow:
    """A Workflow-code tool that waits on a Signal, then runs an Activity."""

    def __init__(self) -> None:
        self.approved: bool | None = None

    @workflow.signal
    def decide(self, approved: bool) -> None:
        self.approved = approved

    @workflow.run
    async def run(self, prompt: str) -> str:
        @tool
        async def refund_with_approval(order_id: str) -> str:
            """Refund an order once a person approves."""
            await workflow.wait_condition(lambda: self.approved is not None)
            return await workflow.execute_activity(
                issue_refund, order_id, start_to_close_timeout=TIMEOUT
            )

        agent = Agent(model=_model(), tools=[refund_with_approval])
        return (await agent.arun(prompt)).message or ""


@workflow.defn(name="LocusReplayStreamed")
class StreamedWorkflow:
    """A streamed run that publishes its chunks to a Workflow Stream."""

    def __init__(self) -> None:
        self.stream = WorkflowStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=_model(
                streaming_topic="tokens",
                streaming_batch_interval=timedelta(milliseconds=50),
            ),
            tools=[activity_as_tool(lookup_order, start_to_close_timeout=TIMEOUT)],
        )
        text = ""
        async for event in agent.run(prompt, stream_tokens=True):
            if isinstance(event, ModelChunkEvent) and event.content:
                text += event.content
        return text.strip()


WORKFLOWS: dict[str, type] = {
    "tool_calls": ToolCallsWorkflow,
    "approval_signal": ApprovalWorkflow,
    "streamed_with_topic": StreamedWorkflow,
}


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
async def test_recorded_history_replays(name: str) -> None:
    history = WorkflowHistory.from_json(
        f"replay-{name}", (HISTORIES / f"{name}.json").read_text(encoding="utf-8")
    )
    replayer = Replayer(
        workflows=[WORKFLOWS[name]],
        plugins=[LocusPlugin(models={"replay": ReplayModel})],
    )
    await replayer.replay_workflow(history)


def test_every_history_has_a_workflow() -> None:
    assert {p.stem for p in HISTORIES.glob("*.json")} == set(WORKFLOWS)
