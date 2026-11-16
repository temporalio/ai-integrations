"""A Worker or a Replayer takes one LangGraphPlugin instance."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from langgraph.func import (  # pyright: ignore[reportMissingTypeStubs]
    entrypoint as lg_entrypoint,
)
from langgraph.func import task  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph import START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.client import Client
from temporalio.langgraph import LangGraphPlugin, entrypoint, graph
from temporalio.worker import Interceptor, Replayer, Worker


class State(TypedDict):
    value: str


async def first_node(state: State) -> dict[str, str]:  # pyright: ignore[reportUnusedParameter]
    return {"value": "first"}


async def second_node(state: State) -> dict[str, str]:  # pyright: ignore[reportUnusedParameter]
    return {"value": "second"}


@task
def shout(value: str) -> str:
    return value.upper()


@lg_entrypoint()
async def shout_entrypoint(value: str) -> dict[str, str]:
    return {"value": await shout(value)}


@workflow.defn
class NamedGraphWorkflow:
    @workflow.run
    async def run(self, name: str) -> Any:
        return await graph(name).compile().ainvoke({"value": ""})


@workflow.defn
class ShoutWorkflow:
    @workflow.run
    async def run(self, value: str) -> Any:
        return await entrypoint("shout").ainvoke(value)


_DEFAULT_ACTIVITY_OPTIONS = {"start_to_close_timeout": timedelta(seconds=10)}


def one_node_graph(node: Callable[..., Any]) -> StateGraph[Any, Any, Any, Any]:
    g = StateGraph(State)
    g.add_node("node", node, metadata={"execute_in": "activity"})
    g.add_edge(START, "node")
    return g


def graph_plugin(name: str, node: Callable[..., Any] = first_node) -> LangGraphPlugin:
    return LangGraphPlugin(
        graphs={name: one_node_graph(node)},
        default_activity_options=_DEFAULT_ACTIVITY_OPTIONS,
    )


def client_with(client: Client, plugins: Sequence[LangGraphPlugin]) -> Client:
    config = client.config()
    config["plugins"] = list(plugins)
    return Client(**config)


@pytest.mark.filterwarnings("ignore:The same plugin type")
@pytest.mark.parametrize("second", ["graph-api", "functional-api"])
@pytest.mark.parametrize(
    "on_client",
    [0, 1, 2],
    ids=["two-on-worker", "one-on-client-one-on-worker", "two-on-client"],
)
async def test_worker_refuses_second_instance(
    client: Client, on_client: int, second: str
) -> None:
    plugins = [
        graph_plugin("first"),
        graph_plugin("second", second_node)
        if second == "graph-api"
        else LangGraphPlugin(entrypoints={"shout": shout_entrypoint}),
    ]
    with pytest.raises(
        ValueError, match="This Worker has more than one LangGraphPlugin instance"
    ):
        Worker(
            client_with(client, plugins[:on_client]),
            task_queue=f"plugin-instances-{uuid4()}",
            workflows=[NamedGraphWorkflow, ShoutWorkflow],
            plugins=plugins[on_client:],
        )


def test_replayer_refuses_second_instance() -> None:
    with pytest.raises(
        ValueError, match="This Replayer has more than one LangGraphPlugin instance"
    ):
        Replayer(
            workflows=[NamedGraphWorkflow],
            plugins=[graph_plugin("first"), graph_plugin("second", second_node)],
        )


@pytest.mark.filterwarnings("ignore:The same plugin type")
@pytest.mark.parametrize(
    ("on_client", "on_worker"),
    [(1, 0), (1, 1), (0, 2)],
    ids=["on-client", "on-client-and-worker", "twice-on-worker"],
)
async def test_worker_applies_one_instance_once(
    client: Client, on_client: int, on_worker: int
) -> None:
    plugin = graph_plugin("my-graph")
    client = client_with(client, [plugin] * on_client)
    task_queue = f"plugin-instances-{uuid4()}"

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[NamedGraphWorkflow],
        plugins=[plugin] * on_worker,
    )

    config = worker.config(active_config=True)
    assert list(config.get("activities", [])) == plugin.activities
    assert list(config.get("interceptors", [])) == list(plugin.interceptors or [])
    async with worker:
        result = await client.execute_workflow(
            NamedGraphWorkflow.run,
            "my-graph",
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
        )
    assert result == {"value": "first"}


async def test_worker_built_from_another_workers_interceptors_is_configured(
    client: Client,
) -> None:
    """The active interceptors of a Worker include the plugin's own.

    A Worker that gets them, and lists the instance once, still gets the
    instance's Activities.
    """
    plugin = graph_plugin("my-graph")
    other = Worker(
        client,
        task_queue=f"plugin-instances-{uuid4()}",
        workflows=[NamedGraphWorkflow],
        plugins=[plugin],
    )

    worker = Worker(
        client,
        task_queue=f"plugin-instances-{uuid4()}",
        workflows=[NamedGraphWorkflow],
        interceptors=other.config(active_config=True).get("interceptors", []),
        plugins=[plugin],
    )

    config = worker.config(active_config=True)
    assert list(config.get("activities", [])) == plugin.activities


async def test_workers_in_one_process_use_their_own_instance(client: Client) -> None:
    """Each Worker's Workflows get the graph of that Worker's instance."""
    task_queues = [f"plugin-instances-{uuid4()}" for _ in range(2)]

    async with (
        Worker(
            client,
            task_queue=task_queues[0],
            workflows=[NamedGraphWorkflow],
            plugins=[graph_plugin("my-graph", first_node)],
        ),
        Worker(
            client,
            task_queue=task_queues[1],
            workflows=[NamedGraphWorkflow],
            plugins=[graph_plugin("my-graph", second_node)],
        ),
    ):
        results = await asyncio.gather(
            *(
                client.execute_workflow(
                    NamedGraphWorkflow.run,
                    "my-graph",
                    id=f"test-workflow-{uuid4()}",
                    task_queue=task_queue,
                )
                for task_queue in task_queues
            )
        )

    assert results == [{"value": "first"}, {"value": "second"}]


@pytest.mark.parametrize(
    "replayer_plugins", ["the-workers", "the-workers-twice", "its-own"]
)
async def test_replayer_next_to_a_running_worker(
    client: Client, replayer_plugins: str
) -> None:
    plugin = graph_plugin("my-graph")
    on_replayer = {
        "the-workers": [plugin],
        "the-workers-twice": [plugin, plugin],
        "its-own": [graph_plugin("my-graph")],
    }[replayer_plugins]
    task_queue = f"plugin-instances-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[NamedGraphWorkflow],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            NamedGraphWorkflow.run,
            "my-graph",
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
        )
        await handle.result()

        await Replayer(
            workflows=[NamedGraphWorkflow], plugins=on_replayer
        ).replay_workflow(await handle.fetch_history())

        result = await client.execute_workflow(
            NamedGraphWorkflow.run,
            "my-graph",
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
        )
    assert result == {"value": "first"}


async def test_one_instance_serves_graphs_and_entrypoints(client: Client) -> None:
    task_queue = f"plugin-instances-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[NamedGraphWorkflow, ShoutWorkflow],
        plugins=[
            LangGraphPlugin(
                graphs={"my-graph": one_node_graph(first_node)},
                entrypoints={"shout": shout_entrypoint},
                tasks=[shout],
                activity_options={"shout": {"execute_in": "activity"}},
                default_activity_options=_DEFAULT_ACTIVITY_OPTIONS,
            )
        ],
    ):
        graph_result = await client.execute_workflow(
            NamedGraphWorkflow.run,
            "my-graph",
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
        )
        entrypoint_result = await client.execute_workflow(
            ShoutWorkflow.run,
            "second",
            id=f"test-workflow-{uuid4()}",
            task_queue=task_queue,
        )

    assert graph_result == {"value": "first"}
    assert entrypoint_result == {"value": "SECOND"}


async def test_interceptors_given_as_an_iterator_are_kept(client: Client) -> None:
    class CallerInterceptor(Interceptor):
        pass

    caller_interceptor = CallerInterceptor()
    # A one-shot iterable, outside the Sequence type the Worker declares.
    one_shot: Any = iter([caller_interceptor])

    worker = Worker(
        client,
        task_queue=f"plugin-instances-{uuid4()}",
        workflows=[NamedGraphWorkflow],
        interceptors=one_shot,
        plugins=[graph_plugin("my-graph")],
    )

    assert caller_interceptor in worker.config(active_config=True).get(
        "interceptors", []
    )
