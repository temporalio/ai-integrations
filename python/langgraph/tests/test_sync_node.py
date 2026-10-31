from datetime import timedelta
from typing import Any
from uuid import uuid4

from langgraph.graph import START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.client import Client
from temporalio.langgraph import LangGraphPlugin, graph
from temporalio.worker import Replayer, Worker


class State(TypedDict):
    value: str


def sync_node(state: State) -> dict[str, str]:
    return {"value": state["value"] + "!"}


def make_appender(suffix: str):
    """Return a closure node appending ``suffix`` (a stand-in node factory)."""

    def append(state: State) -> dict[str, str]:
        return {"value": state["value"] + suffix}

    return append


@workflow.defn
class SyncNodeWorkflow:
    def __init__(self) -> None:
        self.app = graph("my-graph").compile()

    @workflow.run
    async def run(self, input: str) -> Any:
        return await self.app.ainvoke({"value": input})


async def test_sync_node(client: Client):
    g = StateGraph(State)
    g.add_node("sync_node", sync_node, metadata={"execute_in": "activity"})
    g.add_edge(START, "sync_node")

    task_queue = f"sync-node-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[SyncNodeWorkflow],
        plugins=[
            LangGraphPlugin(
                graphs={"my-graph": g},
                default_activity_options={
                    "start_to_close_timeout": timedelta(seconds=10)
                },
            )
        ],
    ):
        result = await client.execute_workflow(
            SyncNodeWorkflow.run,
            "hello",
            id=f"test-sync-node-{uuid4()}",
            task_queue=task_queue,
        )

    assert result == {"value": "hello!"}


@workflow.defn
class FactoryNodesWorkflow:
    def __init__(self) -> None:
        self.app = graph("factory-graph").compile()

    @workflow.run
    async def run(self, input: str) -> Any:
        return await self.app.ainvoke({"value": input})


async def test_factory_nodes_run_as_activities(client: Client):
    """Node-factory closures run as activities and their history replays.

    The closures have no module-level qualname (they share
    ``make_appender.<locals>.append``), so the plugin keys each node by
    its ``<graph>.<node>`` activity name instead of raising at
    construction.
    """
    g = StateGraph(State)
    g.add_node("shout", make_appender("!"), metadata={"execute_in": "activity"})
    g.add_node("full_stop", make_appender("."), metadata={"execute_in": "activity"})
    g.add_edge(START, "shout")
    g.add_edge("shout", "full_stop")

    task_queue = f"factory-graph-{uuid4()}"
    plugin = LangGraphPlugin(
        graphs={"factory-graph": g},
        default_activity_options={"start_to_close_timeout": timedelta(seconds=10)},
    )
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[FactoryNodesWorkflow],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            FactoryNodesWorkflow.run,
            "hello",
            id=f"test-factory-graph-{uuid4()}",
            task_queue=task_queue,
        )
        result = await handle.result()

    assert result == {"value": "hello!."}

    await Replayer(workflows=[FactoryNodesWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )
