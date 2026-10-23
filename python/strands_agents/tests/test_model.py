import asyncio
from collections.abc import AsyncIterable
from datetime import timedelta
from typing import Any
from uuid import uuid4

from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.types.streaming import StreamEvent

from temporalio import workflow
from temporalio.client import Client
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.worker import Replayer, Worker
from tests.common import get_activities
from tests.mock_model import MockModel


@workflow.defn
class ModelWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=15),
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await self.agent.invoke_async(prompt)
        return str(result)


async def test_model(client: Client):
    task_queue = f"test_model-{uuid4()}"
    plugin = StrandsPlugin(models={"mock": lambda: MockModel(["Done!"])})

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ModelWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            ModelWorkflow.run,
            "Hello",
            id=f"test_model_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == "Done!\n"

    history = await handle.fetch_history()
    assert get_activities(history) == ["strands_get_model_config", "invoke_model"]

    await Replayer(
        workflows=[ModelWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


class ConfiguredMockModel(MockModel):
    def get_config(self) -> dict[str, Any]:
        return {"model_id": "mock-1", "context_window_limit": 1000, "api_key": "k"}


@workflow.defn
class ModelConfigWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> dict[str, Any]:
        agent = TemporalAgent(
            model="configured", start_to_close_timeout=timedelta(seconds=15)
        )
        # The config loads inside Strands' invocation guard, so the guard
        # rejects an overlapping first invocation, as it does natively.
        first, second = await asyncio.gather(
            agent.invoke_async(prompt),
            agent.invoke_async(prompt),
            return_exceptions=True,
        )
        completed = await asyncio.gather(
            agent.invoke_async(prompt, idempotency_token="completed"),
            agent.invoke_async(prompt, idempotency_token="completed"),
        )

        # A duplicate of a first invocation that is cancelled while the config
        # loads is aborted, not run; the next invocation reuses the read.
        other = TemporalAgent(
            model="configured", start_to_close_timeout=timedelta(seconds=15)
        )
        primary = asyncio.create_task(
            other.invoke_async("cancelled", idempotency_token="t")
        )
        duplicate = asyncio.create_task(
            other.invoke_async("duplicate", idempotency_token="t")
        )
        await asyncio.sleep(0)
        primary.cancel()
        cancelled = await asyncio.gather(primary, duplicate, return_exceptions=True)
        await other.invoke_async(prompt)
        return {
            "invocations": [str(first), type(second).__name__],
            "deduplicated": [str(result) for result in completed],
            "cancelled": [type(r).__name__ for r in cancelled],
            "prompts": [
                block["text"]
                for message in other.messages
                if message["role"] == "user"
                for block in message["content"]
                if "text" in block
            ],
            "context_window_limit": agent.model.context_window_limit,
            "config": agent.model.get_config(),
        }


async def test_model_config_reaches_workflow(client: Client):
    """Workflow code sees the worker model's id and context window limit."""
    task_queue = f"test_model_config_reaches_workflow-{uuid4()}"
    # One spare response, so a duplicate that wrongly runs fails the asserts.
    plugin = StrandsPlugin(
        models={
            "configured": lambda: ConfiguredMockModel(
                ["Done!", "Again!", "Later!", "Spare!"]
            )
        }
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ModelConfigWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            ModelConfigWorkflow.run,
            "Hello",
            id=f"test_model_config_reaches_workflow_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == {
            "invocations": ["Done!\n", "ConcurrencyException"],
            "deduplicated": ["Again!\n", "Again!\n"],
            "cancelled": ["CancelledError", "IdempotencyAbortedError"],
            "prompts": ["Hello"],
            "context_window_limit": 1000,
            "config": {"model_id": "mock-1", "context_window_limit": 1000},
        }

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "strands_get_model_config",
        "invoke_model",
        "invoke_model",
        "strands_get_model_config",
        "invoke_model",
    ]

    await Replayer(
        workflows=[ModelConfigWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


class StatefulMockModel(MockModel):
    """Keeps the conversation "server-side", as OpenAIResponsesModel(stateful=True) does."""

    calls: list[dict[str, Any]] = []

    @property
    def stateful(self) -> bool:
        return True

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[StreamEvent]:
        model_state = kwargs.get("model_state")
        StatefulMockModel.calls.append(
            {
                "previous_response_id": (model_state or {}).get("response_id"),
                "messages": len(args[0]),
            }
        )
        if model_state is not None:
            model_state["response_id"] = f"resp_{len(StatefulMockModel.calls)}"
        async for event in super().stream(*args, **kwargs):
            yield event


@workflow.defn
class StatefulModelWorkflow:
    @workflow.run
    async def run(self, give_manager: bool) -> list[int] | str:
        kwargs: dict[str, Any] = {}
        if give_manager:
            kwargs["conversation_manager"] = SlidingWindowConversationManager()
        agent = TemporalAgent(
            model="stateful", start_to_close_timeout=timedelta(seconds=15), **kwargs
        )
        messages_after_turn = []
        try:
            for turn in ("turn one", "turn two"):
                await agent.invoke_async(turn)
                messages_after_turn.append(len(agent.messages))
        except ValueError as err:
            return f"refused: {err}"
        return messages_after_turn


async def test_stateful_model_keeps_state_server_side(client: Client):
    """A stateful worker model gets its response id back and the agent clears its messages."""
    task_queue = f"test_stateful_model_keeps_state_server_side-{uuid4()}"
    StatefulMockModel.calls = []
    plugin = StrandsPlugin(
        models={"stateful": lambda: StatefulMockModel(["one", "two", "three"])}
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[StatefulModelWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            StatefulModelWorkflow.run,
            False,
            id=f"test_stateful_model_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == [0, 0]
        assert StatefulMockModel.calls == [
            {"previous_response_id": None, "messages": 1},
            {"previous_response_id": "resp_1", "messages": 1},
        ]

        refused = await client.execute_workflow(
            StatefulModelWorkflow.run,
            True,
            id=f"test_stateful_model_manager_{uuid4()}",
            task_queue=task_queue,
        )
        assert refused == (
            "refused: context_manager and conversation_manager cannot be used with "
            "a stateful model. The model manages conversation state server-side."
        )

    await Replayer(
        workflows=[StatefulModelWorkflow],
        plugins=[plugin],
    ).replay_workflow(await handle.fetch_history())
