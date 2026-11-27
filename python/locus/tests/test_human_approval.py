"""Locus's own pauses for a person, answered by a Signal: ``ask_user``, ``interrupt()`` and ``admit()``."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

from locus.agent import Agent
from locus.control import Action, AdmissionError, AuditTrail, ControlPolicy, admit
from locus.core.command import Command
from locus.core.events import InterruptEvent, ModelChunkEvent, ToolCompleteEvent
from locus.core.interrupt import interrupt
from locus.core.messages import Message, Role, ToolCall
from locus.models.base import ModelResponse
from locus.multiagent.graph import END, START, StateGraph

from temporalio import activity, workflow
from temporalio.client import Client, WorkflowHandle
from temporalio.locus import LocusPlugin, TemporalModel, activity_as_tool
from temporalio.worker import Replayer, Worker

REFUNDS: list[str] = []


@activity.defn
async def issue_refund(order_id: str) -> str:
    """Refund an order."""
    REFUNDS.append(order_id)
    return f"refunded {order_id}"


class AskingModel:
    """Asks the user, refunds on a yes, then completes the task."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        del tools, kwargs
        results = [m.content or "" for m in messages if m.role == Role.TOOL]
        if not results:
            call = ToolCall(
                id="call_ask",
                name="ask_user",
                arguments={"question": "Refund ORD-7842?", "options": "yes,no"},
            )
        elif len(results) == 1 and "yes" in results[0]:
            call = ToolCall(
                id="call_refund",
                name="issue_refund",
                arguments={"order_id": "ORD-7842"},
            )
        else:
            call = ToolCall(
                id="call_done",
                name="task_complete",
                arguments={"summary": f"user said {results[0]}; {results[-1]}"},
            )
        return ModelResponse(
            message=Message.assistant(content=None, tool_calls=[call]),
            stop_reason="tool_calls",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        response = await self.complete(messages, tools, **kwargs)
        yield ModelChunkEvent(tool_calls=response.message.tool_calls, done=True)


class _Asks:
    """A Workflow that shows what it is waiting on and takes the answer by Signal."""

    def __init__(self) -> None:
        self.question: str | None = None
        self.answer: str | None = None

    @workflow.query
    def pending(self) -> str | None:
        return self.question

    @workflow.signal
    def respond(self, answer: str) -> None:
        self.answer = answer

    async def ask(self, question: str) -> str:
        self.question = question
        await workflow.wait_condition(lambda: self.answer is not None)
        answer, self.answer, self.question = self.answer, None, None
        assert answer is not None
        return answer


@workflow.defn
class AskUserWorkflow(_Asks):
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=TemporalModel("m", start_to_close_timeout=timedelta(seconds=30)),
            tools=[
                activity_as_tool(
                    issue_refund, start_to_close_timeout=timedelta(seconds=30)
                )
            ],
            completion_mode="explicit",
        )
        events = agent.run(prompt)
        while True:
            question = None
            final = ""
            async for event in events:
                if isinstance(event, InterruptEvent):
                    question = event.question
                elif isinstance(event, ToolCompleteEvent):
                    final = event.result or ""
            if question is None:
                return final
            events = agent.resume(await self.ask(question))


@workflow.defn
class GraphInterruptWorkflow(_Asks):
    @workflow.run
    async def run(self, order_id: str) -> dict[str, Any]:
        async def approve(state: dict[str, Any]) -> dict[str, Any]:
            return {"decision": interrupt(f"Refund {state['order_id']}?")}

        async def act(state: dict[str, Any]) -> dict[str, Any]:
            if state["decision"] != "yes":
                return {"outcome": "not refunded"}
            return {
                "outcome": await workflow.execute_activity(
                    issue_refund,
                    state["order_id"],
                    start_to_close_timeout=timedelta(seconds=30),
                )
            }

        graph = StateGraph()
        graph.add_node("approve", approve)
        graph.add_node("act", act)
        graph.add_edge(START, "approve")
        graph.add_edge("approve", "act")
        graph.add_edge("act", END)

        paused = await graph.execute({"order_id": order_id})
        assert paused.interrupt is not None
        asked = paused.interrupt.interrupt
        answer = await self.ask(str(asked.payload))
        result = await graph.execute(Command(resume=answer, update=paused.final_state))
        return {
            "success": result.success,
            "outcome": result.final_state["outcome"],
            "interrupt_id": asked.interrupt_id,
            "asked_at": asked.created_at.isoformat(),
            "asked_at_is_workflow_time": asked.created_at <= workflow.now(),
        }


@workflow.defn
class AdmitWorkflow(_Asks):
    @workflow.run
    async def run(self, order_id: str) -> dict[str, Any]:
        trail = AuditTrail(clock=lambda: workflow.now().isoformat())
        action = Action(name="issue_refund", asset=order_id, environment="production")

        async def perform() -> str:
            return await workflow.execute_activity(
                issue_refund, order_id, start_to_close_timeout=timedelta(seconds=30)
            )

        try:
            outcome = await admit(action, perform, policy=ControlPolicy(), trail=trail)
        except AdmissionError as held:
            approver = await self.ask(f"{held.decision.outcome}: refund {order_id}?")
            outcome = await admit(
                action,
                perform,
                policy=ControlPolicy(),
                trail=trail,
                approved_by=approver,
            )
        return {
            "outcome": outcome,
            "trail_verifies": trail.verify(),
            "trail": trail.export_jsonl(),
        }


async def _run(
    client: Client, workflow_class: Any, argument: str, answer: str
) -> tuple[Any, str, list[str]]:
    """Run to the pause, restart the Worker, answer, and replay the History."""
    plugin = LocusPlugin(models={"m": AskingModel})
    config = client.config()
    config["plugins"] = [plugin]
    client = Client(**config)
    task_queue = f"locus-{uuid.uuid4()}"

    def worker() -> Worker:
        return Worker(
            client,
            task_queue=task_queue,
            workflows=[workflow_class],
            activities=[issue_refund],
            max_cached_workflows=0,
        )

    async def pending(handle: WorkflowHandle[Any, Any]) -> str:
        while True:
            question = await handle.query(workflow_class.pending)
            if question:
                return question
            await asyncio.sleep(0.05)

    REFUNDS.clear()
    async with worker():
        handle = await client.start_workflow(
            workflow_class.run,
            argument,
            id=f"approval-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        question = await pending(handle)
    # Nothing was refunded before the answer, and no Worker is running now.
    assert REFUNDS == []
    await handle.signal(workflow_class.respond, answer)
    async with worker():
        result = await handle.result()
    history = await handle.fetch_history()
    await Replayer(workflows=[workflow_class], plugins=[plugin]).replay_workflow(
        history
    )
    return result, question, list(REFUNDS)


async def test_ask_user_is_answered_by_a_signal(client: Client) -> None:
    result, question, refunds = await _run(client, AskUserWorkflow, "refund it", "yes")
    assert question == "Refund ORD-7842?"
    assert refunds == ["ORD-7842"]
    assert "user said yes; refunded ORD-7842" in result


async def test_ask_user_declined(client: Client) -> None:
    result, _, refunds = await _run(client, AskUserWorkflow, "refund it", "no")
    assert refunds == []
    assert "user said no" in result


async def test_graph_interrupt_is_resumed_by_a_signal(client: Client) -> None:
    result, question, refunds = await _run(
        client, GraphInterruptWorkflow, "ORD-7842", "yes"
    )
    assert question == "Refund ORD-7842?"
    assert refunds == ["ORD-7842"]
    assert result["success"] is True
    assert result["outcome"] == "refunded ORD-7842"
    assert result["asked_at_is_workflow_time"] is True


async def test_admit_holds_the_action_until_a_person_approves(client: Client) -> None:
    result, question, refunds = await _run(client, AdmitWorkflow, "ORD-7842", "dana")
    assert question == "require_human: refund ORD-7842?"
    assert refunds == ["ORD-7842"]
    assert result["outcome"] == "refunded ORD-7842"
    assert result["trail_verifies"] is True
    held, approved = result["trail"].splitlines()
    assert '"outcome": "require_human"' in held and "approved_by" not in held
    assert '"approved_by": "dana"' in approved
