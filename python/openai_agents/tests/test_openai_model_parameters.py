"""Explicit agent/model configuration exercised through real Workflow histories."""

import asyncio
import os
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, Model, RunConfig, Runner, function_tool, handoff
from agents.models import get_default_model_settings

from temporalio import workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowHistory
from temporalio.common import RetryPolicy
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.openai_agents import (
    AgentsWorkflowError,
    ModelSummaryProvider,
    OpenAIAgentsPlugin,
    TemporalAgent,
    TemporalModel,
)
from temporalio.openai_agents._invoke_model_activity import ActivityModelInput
from temporalio.openai_agents.testing import (
    AgentEnvironment,
    ResponseBuilders,
    TestModel,
)
from temporalio.worker import Replayer
from temporalio.workflow import ActivityConfig
from tests.helpers import new_worker


def test_public_types_defaults_and_clone():
    class Context:
        pass

    config = ActivityConfig(start_to_close_timeout=timedelta(minutes=20))
    agent = TemporalAgent[Context]("Researcher", model="gpt-5", activity_config=config)
    assert isinstance(agent, Agent)
    assert agent.model == "gpt-5"
    assert agent.model_settings == Agent(name="Native", model="gpt-5").model_settings
    clone = agent.clone(name="Clone", model="gpt-4o")
    assert isinstance(clone, TemporalAgent)
    assert clone.activity_config == config
    assert clone.activity_config is not agent.activity_config
    assert clone.model_settings == Agent(name="Native", model="gpt-4o").model_settings
    config["start_to_close_timeout"] = timedelta(seconds=1)
    assert agent.activity_config == ActivityConfig(
        start_to_close_timeout=timedelta(minutes=20)
    )
    model = TemporalModel("gpt-5")
    assert isinstance(model, Model)
    assert model.get_default_model_settings() == get_default_model_settings("gpt-5")


class SummaryProvider(ModelSummaryProvider):
    def provide(self, agent: Agent[Any] | None, instructions: Any, input: Any) -> str:
        assert agent is not None
        return f"Calling {agent.name}"


def test_summary_conflict():
    for factory in (TemporalModel, TemporalAgent, OpenAIAgentsPlugin, AgentEnvironment):
        kwargs: dict[str, Any] = {"name": "Agent"} if factory is TemporalAgent else {}
        with pytest.raises(ValueError, match="summary_provider"):
            factory(
                activity_config=ActivityConfig(summary="static"),
                summary_provider=SummaryProvider(),
                **kwargs,
            )


@workflow.defn
class ExplicitModelConfigWorkflow:
    @workflow.run
    async def run(self) -> None:
        base = TemporalModel(
            "gpt-4o",
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=20),
                heartbeat_timeout=timedelta(seconds=3),
            ),
        )
        agent = TemporalAgent[None](
            name="Agent",
            model=base,
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(minutes=20),
                heartbeat_timeout=None,
            ),
        )
        for config in (
            RunConfig(),
            RunConfig(model="gpt-5"),
            RunConfig(
                model=TemporalModel(
                    "gpt-5.1",
                    activity_config=ActivityConfig(
                        start_to_close_timeout=timedelta(seconds=40),
                        retry_policy=None,
                    ),
                )
            ),
        ):
            result = await Runner.run(agent, "Hello", run_config=config)
            assert result.final_output == "done"
            assert result.last_agent is agent
            assert agent.model is base
        await Runner.run(Agent(name="Native", model=TemporalModel("gpt-4o")), "Hello")
        await Runner.run(Agent(name="Configured model", model=base), "Hello")
        await Runner.run(TemporalAgent(name="Implicit"), "Hello")
        await Runner.run(TemporalAgent(name="String", model="gpt-5"), "Hello")


async def run_history(
    client: Client, workflow_type: Any, *, options: dict[str, Any] | None = None
) -> tuple[WorkflowHistory, Client]:
    async with AgentEnvironment(
        model=TestModel(lambda: ResponseBuilders.output_message("done")),
        **(options or {}),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(client, workflow_type) as worker:
            handle = await client.start_workflow(
                workflow_type.run,
                id=str(uuid.uuid4()),
                task_queue=worker.task_queue,
                execution_timeout=timedelta(hours=1),
            )
            await handle.result()
            history = await handle.fetch_history()
        await Replayer(
            workflows=[workflow_type], plugins=[env.openai_agents_plugin]
        ).replay_workflow(history)
        return history, client


def scheduled(history: WorkflowHistory):
    return [
        e.activity_task_scheduled_event_attributes
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]


async def test_model_config_precedence_and_run_overrides(client: Client):
    history, client = await run_history(
        client,
        ExplicitModelConfigWorkflow,
        options={
            "activity_config": ActivityConfig(
                start_to_close_timeout=timedelta(seconds=300),
                schedule_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(seconds=7),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        },
    )
    attrs = scheduled(history)
    assert [a.start_to_close_timeout.seconds for a in attrs] == [
        1200,
        1200,
        40,
        300,
        20,
        300,
        300,
    ]
    assert [a.heartbeat_timeout.seconds for a in attrs] == [0, 0, 0, 7, 3, 7, 7]
    assert all(a.schedule_to_close_timeout.seconds == 1800 for a in attrs)
    assert [a.retry_policy.maximum_attempts for a in attrs] == [2, 2, 0, 2, 2, 2, 2]
    inputs = [
        (await client.data_converter.decode(a.input.payloads, [ActivityModelInput]))[0]
        for a in attrs
    ]
    assert [i["model_name"] for i in inputs] == [
        "gpt-4o",
        "gpt-5",
        "gpt-5.1",
        "gpt-4o",
        "gpt-4o",
        None,
        "gpt-5",
    ]


@workflow.defn
class ConcurrentModelConfigWorkflow:
    @workflow.run
    async def run(self) -> None:
        model = TemporalModel(
            "gpt-4o", activity_config=ActivityConfig(heartbeat_timeout=None)
        )
        await asyncio.gather(
            *(
                Runner.run(
                    TemporalAgent(
                        name=f"Agent {seconds}",
                        model=model,
                        activity_config=ActivityConfig(
                            start_to_close_timeout=timedelta(seconds=seconds)
                        ),
                    ),
                    "Hello",
                )
                for seconds in (11, 22, 33)
            )
        )


async def test_concurrent_agent_configuration(client: Client):
    history, _ = await run_history(client, ConcurrentModelConfigWorkflow)
    assert sorted(a.start_to_close_timeout.seconds for a in scheduled(history)) == [
        11,
        22,
        33,
    ]
    assert all(a.heartbeat_timeout.seconds == 0 for a in scheduled(history))


@workflow.defn
class NestedModelConfigWorkflow:
    @workflow.run
    async def run(self, kind: str) -> str:
        child = TemporalAgent[None](
            name="Child",
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=33)
            ),
        )
        parent = TemporalAgent[None](
            name="Parent",
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=120)
            ),
        )

        @function_tool(failure_error_function=None)
        async def delegate() -> str:
            result = await Runner.run(child, "Hello")
            return str(result.final_output)

        config = RunConfig()
        if kind in ("run", "fresh"):
            activity_config = ActivityConfig(
                start_to_close_timeout=timedelta(seconds=55)
            )
            if kind == "fresh":
                child.summary_provider = SummaryProvider()
                activity_config["summary"] = "outer run"
            config.model = TemporalModel("gpt-5", activity_config=activity_config)
        if kind == "handoff":
            parent.handoffs = [handoff(child)]
        else:
            parent.tools = [
                delegate
                if kind == "fresh"
                else child.as_tool(tool_name="delegate", tool_description="Delegate")
            ]
        result = await Runner.run(parent, "Delegate to the child", run_config=config)
        return str(result.final_output)


@pytest.mark.parametrize("kind", ["handoff", "tool", "run", "fresh"])
async def test_nested_agent_configuration(client: Client, kind: str):
    responses = [
        ResponseBuilders.tool_call(
            '{"input":"Hello"}' if kind in ("tool", "run") else "{}",
            "transfer_to_child" if kind == "handoff" else "delegate",
        ),
        ResponseBuilders.output_message("done"),
    ]
    if kind != "handoff":
        responses.append(ResponseBuilders.output_message("done"))
    async with AgentEnvironment(model=TestModel.returning_responses(responses)) as env:
        client = env.applied_on_client(client)
        async with new_worker(client, NestedModelConfigWorkflow) as worker:
            handle = await client.start_workflow(
                NestedModelConfigWorkflow.run,
                kind,
                id=str(uuid.uuid4()),
                task_queue=worker.task_queue,
            )
            assert await handle.result() == "done"
            history = await handle.fetch_history()
    assert [a.start_to_close_timeout.seconds for a in scheduled(history)] == {
        "handoff": [120, 33],
        "tool": [120, 33, 120],
        "run": [55, 55, 55],
        "fresh": [55, 33, 55],
    }[kind]


@workflow.defn
class ValidateModelConfigWorkflow:
    @workflow.run
    async def run(self) -> None:
        cases: list[tuple[dict[str, Any], str]] = [
            (
                {"activity_config": ActivityConfig(start_to_close_timeout=None)},
                "requires",
            ),
            (
                {
                    "use_local_activity": True,
                    "activity_config": ActivityConfig(task_queue="other"),
                },
                "task_queue",
            ),
            (
                {
                    "use_local_activity": True,
                    "activity_config": ActivityConfig(heartbeat_timeout=None),
                },
                "heartbeat_timeout",
            ),
        ]
        for options, match in cases:
            try:
                await Runner.run(TemporalAgent(name="Invalid", **options), "Hello")
            except AgentsWorkflowError as e:
                assert match in str(e)
            else:
                raise AssertionError("Expected invalid configuration to fail")
        for model in (None, "gpt-5"):
            try:
                await Runner.run(Agent(name="Native", model=model), "Hello")
            except ValueError as e:
                assert "TemporalAgent" in str(e)
            else:
                raise AssertionError("Expected explicit model API validation")
        streaming_cases: list[tuple[dict[str, Any], str]] = [
            ({}, "streaming_topic"),
            (
                {"use_local_activity": True, "streaming_topic": "events"},
                "use_local_activity",
            ),
        ]
        for options, match in streaming_cases:
            try:
                Runner.run_streamed(TemporalAgent(name="Invalid", **options), "Hello")
            except AgentsWorkflowError as e:
                assert match in str(e)
            else:
                raise AssertionError("Expected streaming validation")


async def test_invalid_effective_configuration_before_scheduling(client: Client):
    history, _ = await run_history(client, ValidateModelConfigWorkflow)
    assert not scheduled(history)


@workflow.defn
class SummaryModelConfigWorkflow:
    @workflow.run
    async def run(self) -> None:
        for config in (
            ActivityConfig(summary="static"),
            ActivityConfig(summary=None),
            ActivityConfig(),
        ):
            await Runner.run(
                TemporalAgent(name="Agent", activity_config=config), "Hello"
            )
        await Runner.run(
            TemporalAgent(name="Dynamic", summary_provider=SummaryProvider()), "Hello"
        )


async def test_static_and_dynamic_summaries(client: Client):
    history, _ = await run_history(client, SummaryModelConfigWorkflow)
    assert [
        e.user_metadata.summary.data
        for e in history.events
        if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ] == [
        b'"static"',
        b"",
        b'"Agent"',
        b'"Calling Dynamic"',
    ]


@workflow.defn
class LocalModelConfigWorkflow:
    @workflow.run
    async def run(self) -> None:
        await Runner.run(TemporalAgent(name="Local", use_local_activity=True), "Hello")
        await Runner.run(
            Agent(name="Local model", model=TemporalModel(use_local_activity=True)),
            "Hello",
        )


async def test_local_model_configuration(client: Client):
    history, _ = await run_history(client, LocalModelConfigWorkflow)
    assert not scheduled(history)
    assert (
        sum(
            e.marker_recorded_event_attributes.marker_name == "core_local_activity"
            for e in history.events
        )
        == 2
    )


async def test_record_explicit_api_history(client: Client):
    history, _ = await run_history(client, ExplicitModelConfigWorkflow)
    assert [a.start_to_close_timeout.seconds for a in scheduled(history)][-2:] == [
        60,
        60,
    ]
    assert [a.heartbeat_timeout.seconds for a in scheduled(history)][-2:] == [10, 10]
    destination = (
        Path(__file__).with_name("histories")
        / "2.x"
        / "explicit-model-config-workflow-history.json"
    )
    if os.environ.get("UPDATE_OPENAI_EXPLICIT_HISTORY") == "1":
        destination.write_text(history.to_json())
    assert destination.exists()


@workflow.defn
class StreamingModelConfigWorkflow:
    @workflow.init
    def __init__(self, kind: str) -> None:
        self.stream = WorkflowStream()

    @workflow.run
    async def run(self, kind: str) -> str:
        agent: Agent[None]
        config = RunConfig()
        if kind == "agent":
            agent = TemporalAgent(
                name="Streaming",
                streaming_topic="agent-events",
                use_local_activity=False,
                activity_config=ActivityConfig(
                    start_to_close_timeout=timedelta(seconds=21)
                ),
            )
        elif kind == "model":
            agent = Agent(
                name="Streaming",
                model=TemporalModel(
                    streaming_topic="model-events",
                    use_local_activity=False,
                    activity_config=ActivityConfig(
                        start_to_close_timeout=timedelta(seconds=22)
                    ),
                ),
            )
        else:
            agent = TemporalAgent(name="Streaming", use_local_activity=True)
            config.model = TemporalModel(
                streaming_topic="run-events",
                use_local_activity=False,
                activity_config=ActivityConfig(
                    start_to_close_timeout=timedelta(seconds=23)
                ),
            )
        result = Runner.run_streamed(agent, "Hello", run_config=config)
        async for _ in result.stream_events():
            pass
        return str(result.final_output)


@pytest.mark.parametrize(
    ("kind", "seconds"), [("agent", 21), ("model", 22), ("run", 23)]
)
async def test_streaming_model_configuration(client: Client, kind: str, seconds: int):
    from tests.test_openai_streaming import StreamingTestModel

    async with AgentEnvironment(
        model=StreamingTestModel(), use_local_activity=True
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(client, StreamingModelConfigWorkflow) as worker:
            # The agent and model cases must explicitly override the Worker's local default.
            # The run case supplies that override on RunConfig.model.
            handle = await client.start_workflow(
                StreamingModelConfigWorkflow.run,
                kind,
                id=str(uuid.uuid4()),
                task_queue=worker.task_queue,
                execution_timeout=timedelta(seconds=30),
            )
            assert await handle.result() == "Hello world!"
            history = await handle.fetch_history()
        await Replayer(
            workflows=[StreamingModelConfigWorkflow], plugins=[env.openai_agents_plugin]
        ).replay_workflow(history)
    attrs = scheduled(history)
    assert len(attrs) == 1
    assert attrs[0].activity_type.name == "invoke_model_activity_streaming"
    assert attrs[0].start_to_close_timeout.seconds == seconds
