"""Locus's multi-agent shapes run in a Workflow, each agent's model call an Activity."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

from locus.agent import Agent
from locus.agent.composition import ParallelPipeline, SequentialPipeline
from locus.core.events import ModelChunkEvent
from locus.core.messages import Message, Role
from locus.models.base import ModelResponse
from locus.multiagent.graph import END, START, StateGraph
from locus.multiagent.handoff import create_handoff_agent, create_handoff_manager
from locus.multiagent.orchestrator import create_orchestrator
from locus.multiagent.specialist import Specialist
from locus.multiagent.swarm import create_swarm, create_swarm_agent

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.locus import LocusPlugin, TemporalModel
from temporalio.worker import Replayer, Worker


class RoleModel:
    """Answers with the agent's system prompt and the end of its prompt."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        del tools, kwargs
        system = next((m.content for m in messages if m.role == Role.SYSTEM), "")
        user = [m.content for m in messages if m.role == Role.USER][-1] or ""
        if "## Available Specialists" in user:
            # An Orchestrator asking which specialists to route to.
            return ModelResponse(
                message=Message.assistant(
                    content='{"specialists": ["logs", "metrics"], "reasoning": "both",'
                    ' "subtasks": {"logs": "read the logs", "metrics": "read the metrics"}}'
                ),
                stop_reason="stop",
            )
        return ModelResponse(
            message=Message.assistant(content=f"[{system}] {user[-18:]}"),
            stop_reason="stop",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        response = await self.complete(messages, tools, **kwargs)
        yield ModelChunkEvent(content=response.message.content, done=True)


def _agent(role: str) -> Agent:
    return Agent(
        model=TemporalModel("m", start_to_close_timeout=timedelta(seconds=30)),
        system_prompt=role,
    )


@activity.defn
async def echo_values(values: list[str]) -> list[str]:
    return values


@workflow.defn
class SequentialWorkflow:
    @workflow.run
    async def run(self, task: str) -> list[str]:
        pipeline = SequentialPipeline(agents=[_agent("research"), _agent("write")])
        result = await pipeline.run(task)
        return [str(result.success), *result.outputs]


@workflow.defn
class ParallelWorkflow:
    @workflow.run
    async def run(self, task: str) -> list[str]:
        pipeline = ParallelPipeline(agents=[_agent("a"), _agent("b"), _agent("c")])
        result = await pipeline.run(task)
        return [str(result.success), *sorted(result.outputs)]


@workflow.defn
class GraphWorkflow:
    @workflow.run
    async def run(self, task: str) -> dict[str, Any]:
        async def draft(state: dict[str, Any]) -> dict[str, Any]:
            return {"draft": (await _agent("draft").arun(state["task"])).message}

        async def review(state: dict[str, Any]) -> dict[str, Any]:
            return {"review": (await _agent("review").arun(state["draft"])).message}

        graph = StateGraph()
        graph.add_node("draft", draft)
        graph.add_node("review", review)
        graph.add_edge(START, "draft")
        graph.add_edge("draft", "review")
        graph.add_edge("review", END)
        result = await graph.execute({"task": task})
        started = result.node_results["draft"].started_at
        read = [result.graph_id, started.isoformat() if started else ""]
        # With no workflow cache this Activity's completion replays the graph:
        # its id and timestamps must be read again unchanged.
        recorded = await workflow.execute_activity(
            echo_values, read, start_to_close_timeout=timedelta(seconds=10)
        )
        return {
            "success": result.success,
            "review": result.node_results["review"].output,
            "replayed_agree": recorded == read,
        }


def _model() -> TemporalModel:
    return TemporalModel("m", start_to_close_timeout=timedelta(seconds=30))


@workflow.defn
class OrchestratorWorkflow:
    @workflow.run
    async def run(self, task: str) -> dict[str, Any]:
        specialists = [
            Specialist(
                id=name,
                name=name,
                specialist_type=name,
                description=f"reads {name}",
                system_prompt=name,
                model=_model(),
            )
            for name in ("logs", "metrics")
        ]
        orchestrator = create_orchestrator(specialists=specialists, model=_model())
        result = await orchestrator.execute(task)
        return {
            "success": result.success,
            "error": result.error,
            "specialists": sorted(result.specialist_results),
            "has_summary": bool(result.summary),
        }


@workflow.defn
class SwarmWorkflow:
    @workflow.run
    async def run(self, task: str) -> dict[str, Any]:
        swarm = create_swarm(
            agents=[create_swarm_agent("one"), create_swarm_agent("two")],
            model=_model(),
        )
        swarm.add_task(f"{task}: part one")
        swarm.add_task(f"{task}: part two")
        result = await swarm.execute(decompose_tasks=False)
        return {
            "success": result.success,
            "error": result.error,
            "completed": len(result.completed_tasks),
            "failed": len(result.failed_tasks),
        }


@workflow.defn
class HandoffWorkflow:
    @workflow.run
    async def run(self, task: str) -> dict[str, Any]:
        agents = [
            create_handoff_agent(name, system_prompt=name, model=_model())
            for name in ("triage", "billing")
        ]
        manager = create_handoff_manager(agents)
        results = await manager.chain_handoff([agent.id for agent in agents], task)
        return {
            "success": [result.success for result in results],
            "errors": [result.error for result in results],
            "outputs": [bool(result.output) for result in results],
        }


async def _run(client: Client, workflow_class: Any) -> tuple[Any, list[str]]:
    plugin = LocusPlugin(models={"m": RoleModel})
    config = client.config()
    config["plugins"] = [plugin]
    client = Client(**config)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[workflow_class],
        activities=[echo_values],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            workflow_class.run,
            "ship the report",
            id=f"shape-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        result = await handle.result()
    history = await handle.fetch_history()
    await Replayer(workflows=[workflow_class], plugins=[plugin]).replay_workflow(
        history
    )
    scheduled = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField("activity_task_scheduled_event_attributes")
    ]
    return result, scheduled


async def test_sequential_pipeline(client: Client) -> None:
    result, scheduled = await _run(client, SequentialWorkflow)
    assert result[0] == "True"
    assert result[1] == "[research] ship the report"
    assert result[2].startswith("[write] ")
    assert scheduled == ["locus_model_complete", "locus_model_complete"]


async def test_parallel_pipeline(client: Client) -> None:
    result, scheduled = await _run(client, ParallelWorkflow)
    assert result == [
        "True",
        "[a] ship the report",
        "[b] ship the report",
        "[c] ship the report",
    ]
    assert scheduled == ["locus_model_complete"] * 3


async def test_state_graph_with_agent_nodes(client: Client) -> None:
    result, scheduled = await _run(client, GraphWorkflow)
    assert result["success"] is True
    assert result["review"] == {"review": "[review] t] ship the report"}
    assert result["replayed_agree"] is True
    assert scheduled == ["locus_model_complete", "locus_model_complete", "echo_values"]


async def test_orchestrator_routes_to_specialists(client: Client) -> None:
    result, scheduled = await _run(client, OrchestratorWorkflow)
    assert result == {
        "success": True,
        "error": None,
        "specialists": ["logs", "metrics"],
        "has_summary": True,
    }
    # Routing, the two specialists, correlation and the summary.
    assert scheduled == ["locus_model_complete"] * 5


async def test_swarm_works_its_task_queue(client: Client) -> None:
    result, scheduled = await _run(client, SwarmWorkflow)
    assert result == {"success": True, "error": None, "completed": 2, "failed": 0}
    # The two tasks and the swarm's summary.
    assert scheduled == ["locus_model_complete"] * 3


async def test_handoff_chain(client: Client) -> None:
    result, scheduled = await _run(client, HandoffWorkflow)
    # One handoff, triage to billing, answered by billing's model.
    assert result == {"success": [True], "errors": [None], "outputs": [True]}
    assert scheduled == ["locus_model_complete"]
