"""An agent's loop runs in the workflow; model and tool calls run as activities."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

from locus.agent import Agent
from locus.core.events import ModelChunkEvent
from locus.core.messages import Message, Role, ToolCall
from locus.models.base import ModelResponse

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.locus import LocusPlugin, TemporalModel, activity_as_tool
from temporalio.worker import Worker


class WeatherModel:
    """Asks for the weather once, then answers with what the tool returned."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        del tools, kwargs
        self.calls += 1
        results = [m for m in messages if m.role == Role.TOOL]
        if not results:
            call = ToolCall(
                id="call_1", name="get_weather", arguments={"city": "Tokyo"}
            )
            return ModelResponse(
                message=Message.assistant(content=None, tool_calls=[call]),
                usage={"prompt_tokens": 5, "completion_tokens": 3},
                stop_reason="tool_calls",
            )
        return ModelResponse(
            message=Message.assistant(content=f"Forecast: {results[-1].content}"),
            usage={"prompt_tokens": 9, "completion_tokens": 4},
            stop_reason="stop",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        response = await self.complete(messages, tools, **kwargs)
        yield ModelChunkEvent(
            content=response.message.content,
            tool_calls=response.message.tool_calls or None,
            done=True,
        )


TOOL_RUNS: list[str] = []


@activity.defn
async def get_weather(city: str) -> str:
    """Return the weather in a city."""
    TOOL_RUNS.append(city)
    return f"sunny in {city}"


@workflow.defn
class WeatherWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = Agent(
            model=TemporalModel(
                "weather", start_to_close_timeout=timedelta(seconds=30)
            ),
            tools=[
                activity_as_tool(
                    get_weather, start_to_close_timeout=timedelta(seconds=30)
                )
            ],
            system_prompt="Answer with the forecast.",
        )
        result = await agent.arun(prompt)
        return result.message or ""


async def test_agent_runs_model_and_tool_calls_as_activities(client: Client) -> None:
    model = WeatherModel()
    TOOL_RUNS.clear()
    config = client.config()
    config["plugins"] = [LocusPlugin(models={"weather": lambda: model})]
    client = Client(**config)
    task_queue = f"locus-{uuid.uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[WeatherWorkflow],
        activities=[get_weather],
    ):
        handle = await client.start_workflow(
            WeatherWorkflow.run,
            "What's the weather in Tokyo?",
            id=f"weather-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        assert await handle.result() == "Forecast: sunny in Tokyo"

    assert TOOL_RUNS == ["Tokyo"]
    assert model.calls == 2
    scheduled = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        async for event in handle.fetch_history_events()
        if event.HasField("activity_task_scheduled_event_attributes")
    ]
    assert scheduled == ["locus_model_complete", "get_weather", "locus_model_complete"]
