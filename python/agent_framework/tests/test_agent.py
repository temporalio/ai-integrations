"""A vanilla ``agent_framework.Agent`` runs its model calls as Activities."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from agent_framework import Agent, tool
from pydantic import BaseModel

from temporalio import workflow
from temporalio.agent_framework import AgentFrameworkPlugin, TemporalChatClient
from temporalio.agent_framework.testing import (
    FakeChatClient,
    fake_chat_client,
    function_call_response,
    text_response,
)
from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.worker import Worker
from tests.helpers.agent_framework import (
    INVOKE_MODEL,
    INVOKE_MODEL_STREAMING,
    SlowFakeChatClient,
    contents_summary,
    count_scheduled_activities,
    new_worker,
    role_of,
    scheduled_activities,
    scheduled_activity_events,
    started_activities,
    summary_of,
    workflow_id,
)


@dataclass
class RunInput:
    prompt: str = "weather in Paris?"
    model: str = "model"
    summary: str | None = None
    task_queue: str | None = None
    start_to_close_seconds: float | None = None
    heartbeat_seconds: float | None = None


def _client(input: RunInput) -> TemporalChatClient:
    options: dict[str, Any] = {}
    if input.start_to_close_seconds is not None:
        options["start_to_close_timeout"] = timedelta(
            seconds=input.start_to_close_seconds
        )
    if input.heartbeat_seconds is not None:
        options["heartbeat_timeout"] = timedelta(seconds=input.heartbeat_seconds)
    return TemporalChatClient(
        input.model, summary=input.summary, task_queue=input.task_queue, **options
    )


@workflow.defn
class ToolLoopWorkflow:
    @workflow.run
    async def run(self, input: RunInput) -> dict[str, Any]:
        log: list[str] = []

        @tool
        async def get_weather(city: str) -> str:
            """Get the weather for a city."""
            log.append(city)
            return f"rainy in {city}"

        agent = Agent(
            client=_client(input),
            instructions="You are a weather bot.",
            tools=[get_weather],
        )
        response = await agent.run(input.prompt)
        return {
            "text": response.text,
            "contents": contents_summary(response),
            "log": log,
        }


@workflow.defn
class AsAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = TemporalChatClient("model").as_agent(instructions="Answer briefly.")
        return (await agent.run(prompt)).text


@workflow.defn
class MultiModelWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> dict[str, str]:
        fast = Agent(client=TemporalChatClient("fast"), name="fast")
        smart = Agent(client=TemporalChatClient("smart"), name="smart")
        return {
            "fast": (await fast.run(prompt)).text,
            "smart": (await smart.run(prompt)).text,
            "fast_again": (await fast.run(prompt)).text,
        }


class Person(BaseModel):
    name: str
    age: int


@workflow.defn
class StructuredOutputWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> Person:
        agent = Agent(client=TemporalChatClient("model"))
        response = await agent.run(prompt, options={"response_format": Person})
        value = response.value
        if not isinstance(value, Person):
            raise ApplicationError(
                f"expected a Person, got {value!r}", non_retryable=True
            )
        return value


@workflow.defn
class AgentAsToolWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> dict[str, Any]:
        researcher = Agent(
            client=TemporalChatClient("sub", summary="researcher"),
            name="researcher",
            description="Finds facts.",
        )
        main = Agent(
            client=TemporalChatClient("main", summary="main"),
            tools=[researcher.as_tool()],
        )
        response = await main.run(prompt)
        return {"text": response.text, "contents": contents_summary(response)}


async def test_vanilla_agent_runs_through_activities(client: Client) -> None:
    factory = fake_chat_client(
        [
            function_call_response("get_weather", {"city": "Paris"}, call_id="call-1"),
            text_response("Rainy in Paris."),
        ]
    )
    plugin = AgentFrameworkPlugin(chat_clients={"model": factory})
    async with new_worker(client, ToolLoopWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ToolLoopWorkflow.run,
            RunInput(),
            id=workflow_id("tool-loop"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()

    assert result["text"] == "Rainy in Paris."
    assert result["log"] == ["Paris"]
    assert result["contents"] == [
        "call:get_weather",
        "result:rainy in Paris",
        "text:Rainy in Paris.",
    ]
    assert await count_scheduled_activities(handle) == {INVOKE_MODEL: 2}

    first, second = factory().calls
    assert [role_of(m) for m in first["messages"]] == ["user"]
    assert first["messages"][0].text == "weather in Paris?"
    assert first["options"]["instructions"] == "You are a weather bot."
    (declared,) = first["options"]["tools"]
    assert declared.name == "get_weather" and declared.declaration_only
    assert declared.parameters()["properties"]["city"]["type"] == "string"
    assert declared.parameters()["required"] == ["city"]
    assert [role_of(m) for m in second["messages"]] == ["user", "assistant", "tool"]


async def test_as_agent(client: Client) -> None:
    plugin = AgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(["Hi there."])}
    )
    async with new_worker(client, AsAgentWorkflow, plugin=plugin) as worker:
        result = await client.execute_workflow(
            AsAgentWorkflow.run,
            "hello",
            id=workflow_id("as-agent"),
            task_queue=worker.task_queue,
        )
    assert result == "Hi there."


async def test_multi_model_dispatch(client: Client) -> None:
    factory_calls = {"fast": 0, "smart": 0}

    def factory(name: str, reply: str) -> Any:
        def build() -> FakeChatClient:
            factory_calls[name] += 1
            return FakeChatClient([reply])

        return build

    plugin = AgentFrameworkPlugin(
        chat_clients={
            "fast": factory("fast", "fast reply"),
            "smart": factory("smart", "smart reply"),
        }
    )
    async with new_worker(client, MultiModelWorkflow, plugin=plugin) as worker:
        result = await client.execute_workflow(
            MultiModelWorkflow.run,
            "hello",
            id=workflow_id("multi-model"),
            task_queue=worker.task_queue,
        )
    assert result == {
        "fast": "fast reply",
        "smart": "smart reply",
        "fast_again": "fast reply",
    }
    # Each factory runs once per worker, however many calls it serves.
    assert factory_calls == {"fast": 1, "smart": 1}


async def test_unknown_model_name(client: Client) -> None:
    plugin = AgentFrameworkPlugin(chat_clients={"model": fake_chat_client(["unused"])})
    async with new_worker(client, ToolLoopWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ToolLoopWorkflow.run,
            RunInput(model="missing"),
            id=workflow_id("unknown-model"),
            task_queue=worker.task_queue,
        )
        with pytest.raises(WorkflowFailureError) as excinfo:
            await handle.result()
    cause = excinfo.value.cause
    assert isinstance(cause, ActivityError)
    failure = cause.__cause__
    assert isinstance(failure, ApplicationError)
    assert failure.type == "AgentFrameworkUnknownChatClient"
    assert failure.non_retryable
    assert "'missing'" in failure.message and "'model'" in failure.message
    (started,) = await started_activities(handle)
    assert started.attempt == 1


async def test_response_format(client: Client) -> None:
    factory = fake_chat_client(['{"name": "Ada", "age": 36}'])
    plugin = AgentFrameworkPlugin(chat_clients={"model": factory})
    async with new_worker(client, StructuredOutputWorkflow, plugin=plugin) as worker:
        result = await client.execute_workflow(
            StructuredOutputWorkflow.run,
            "Who was the first programmer?",
            id=workflow_id("structured"),
            task_queue=worker.task_queue,
        )
    assert result == Person(name="Ada", age=36)
    (call,) = factory().calls
    assert call["options"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Person", "schema": Person.model_json_schema()},
    }


async def test_agent_as_tool(client: Client) -> None:
    main = fake_chat_client(
        [
            function_call_response(
                "researcher", {"task": "find the fact"}, call_id="call-1"
            ),
            text_response("The fact is 42."),
        ]
    )
    sub = fake_chat_client(["42"])
    plugin = AgentFrameworkPlugin(chat_clients={"main": main, "sub": sub})
    async with new_worker(client, AgentAsToolWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            AgentAsToolWorkflow.run,
            "what is the fact?",
            id=workflow_id("agent-as-tool"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    assert result["text"] == "The fact is 42."
    assert result["contents"][:2] == ["call:researcher", "result:42"]
    # as_tool() streams the sub-agent's run, so its one call is a streaming Activity.
    assert await count_scheduled_activities(handle) == {
        INVOKE_MODEL: 2,
        INVOKE_MODEL_STREAMING: 1,
    }
    assert [summary_of(e) for e in await scheduled_activity_events(handle)] == [
        "main",
        "researcher",
        "main",
    ]
    assert sub().calls[0]["messages"][-1].text == "find the fact"


async def test_side_effect_counts(client: Client) -> None:
    plugin = AgentFrameworkPlugin(
        chat_clients={
            "model": fake_chat_client(
                [
                    function_call_response(
                        "get_weather", {"city": "Paris"}, call_id="call-1"
                    ),
                    function_call_response(
                        "get_weather", {"city": "Oslo"}, call_id="call-2"
                    ),
                    text_response("Rainy in both."),
                ]
            )
        }
    )
    async with new_worker(client, ToolLoopWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ToolLoopWorkflow.run,
            RunInput(prompt="Paris and Oslo?"),
            id=workflow_id("side-effects"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    assert result["log"] == ["Paris", "Oslo"]
    # The worker replays every workflow task from history, so a tool that ran
    # again on replay would show in the log, and an extra model call here.
    assert await count_scheduled_activities(handle) == {INVOKE_MODEL: 3}
    ids = [a.activity_id for a in await scheduled_activities(handle)]
    assert len(set(ids)) == 3


@pytest.mark.parametrize(
    ("summary", "expected"), [(None, "model"), ("weather bot", "weather bot")]
)
async def test_activity_summary(
    client: Client, summary: str | None, expected: str
) -> None:
    plugin = AgentFrameworkPlugin(chat_clients={"model": fake_chat_client(["Sunny."])})
    async with new_worker(client, ToolLoopWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ToolLoopWorkflow.run,
            RunInput(summary=summary),
            id=workflow_id("summary"),
            task_queue=worker.task_queue,
        )
        await handle.result()
    assert [summary_of(e) for e in await scheduled_activity_events(handle)] == [
        expected
    ]


async def test_activity_options_applied(client: Client) -> None:
    slow = SlowFakeChatClient(["Slow answer."], delay=1.6)
    plugin = AgentFrameworkPlugin(chat_clients={"model": lambda: slow})
    async with new_worker(client, ToolLoopWorkflow, plugin=plugin) as worker:
        handle = await client.start_workflow(
            ToolLoopWorkflow.run,
            RunInput(start_to_close_seconds=42, heartbeat_seconds=1),
            id=workflow_id("options"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    assert result["text"] == "Slow answer."
    (scheduled,) = await scheduled_activities(handle)
    assert scheduled.start_to_close_timeout.ToTimedelta() == timedelta(seconds=42)
    assert scheduled.heartbeat_timeout.ToTimedelta() == timedelta(seconds=1)
    # A 1.6 second call under a 1 second heartbeat timeout only completes on
    # its first attempt when the Activity heartbeats on its own.
    (started,) = await started_activities(handle)
    assert started.attempt == 1


async def test_model_task_queue(client: Client) -> None:
    plugin = AgentFrameworkPlugin(
        chat_clients={"model": fake_chat_client(["From afar."])}
    )
    model_queue = f"agent-framework-models-{uuid.uuid4()}"
    async with (
        Worker(client, task_queue=model_queue, plugins=[plugin]),
        new_worker(client, ToolLoopWorkflow, plugin=plugin) as worker,
    ):
        handle = await client.start_workflow(
            ToolLoopWorkflow.run,
            RunInput(task_queue=model_queue),
            id=workflow_id("task-queue"),
            task_queue=worker.task_queue,
        )
        result = await handle.result()
    assert result["text"] == "From afar."
    (scheduled,) = await scheduled_activities(handle)
    assert scheduled.task_queue.name == model_queue
