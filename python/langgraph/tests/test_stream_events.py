"""Event streams complete in Workflows and replay without repeating Activities."""

import asyncio
from datetime import timedelta
from uuid import uuid4

from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.client import Client
from temporalio.langgraph import LangGraphPlugin, graph
from temporalio.worker import Replayer, Worker


class State(TypedDict):
    text: str


def shout(state: State) -> dict[str, str]:
    return {"text": state["text"].upper()}


@workflow.defn
class StreamEventsWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        app = graph("shout").compile()
        return [
            event["event"]
            async for event in app.astream_events({"text": "hi"}, version="v2")
        ]


async def test_stream_events_completes_and_replays(client: Client) -> None:
    g = StateGraph(State)
    g.add_node("shout", shout, metadata={"execute_in": "activity"})
    g.add_edge(START, "shout")
    g.add_edge("shout", END)
    expected = [
        event["event"]
        async for event in g.compile().astream_events({"text": "hi"}, version="v2")
    ]
    plugin = LangGraphPlugin(
        graphs={"shout": g},
        default_activity_options={"start_to_close_timeout": timedelta(seconds=10)},
    )
    queue = f"stream-events-{uuid4()}"
    async with Worker(
        client, task_queue=queue, workflows=[StreamEventsWorkflow], plugins=[plugin]
    ):
        handle = await client.start_workflow(
            StreamEventsWorkflow.run, id=str(uuid4()), task_queue=queue
        )
        try:
            result = await asyncio.wait_for(handle.result(), 10)
            history = await handle.fetch_history()
        except BaseException:
            await handle.terminate()
            raise
    assert result == expected
    await Replayer(workflows=[StreamEventsWorkflow], plugins=[plugin]).replay_workflow(
        history
    )
