"""Test that the task result cache belongs to one Workflow run.

A call repeated in a run executes again, a Workflow never gets another
Workflow's result, and a cache carried through continue-as-new is served once,
wherever the Workflow fetches its graphs and entrypoints.
"""

import asyncio
import gc
import operator
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Annotated, Any
from uuid import uuid4

import pytest
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import (  # pyright: ignore[reportMissingTypeStubs]
    entrypoint as lg_entrypoint,
)
from langgraph.func import task  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
    add_messages,
)
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Send, interrupt
from typing_extensions import TypedDict

from temporalio import activity, workflow
from temporalio.client import Client, WorkflowHistory
from temporalio.langgraph import LangGraphPlugin, cache, entrypoint, graph
from temporalio.langgraph._task_cache import (
    _RUN_CACHE,
    _run_cache,
    _RunCache,
    cache_key,
    cache_lookup,
    cache_put,
    get_task_cache,
    set_task_cache,
)
from temporalio.worker import Replayer, Worker

_ACTIVITY: dict[str, Any] = {
    "execute_in": "activity",
    "start_to_close_timeout": timedelta(seconds=10),
}

# How often each counted body ran, by Workflow ID or by name.
_calls: dict[str, int] = {}


def _called(name: str) -> int:
    _calls[name] = _calls.get(name, 0) + 1
    return _calls[name]


class Count(TypedDict):
    count: int
    owner: str


_INPUT = Count(count=0, owner="")


async def count_calls(state: Count) -> dict[str, Any]:  # pyright: ignore[reportUnusedParameter]
    """Return how often this body ran for the calling Workflow, and its ID.

    Every call gets the same input, so a result that came from the cache shows
    as a count seen before, and another Workflow's result as another owner.
    """
    owner = activity.info().workflow_id or ""
    return {"count": _called(owner), "owner": owner}


def _one_node_graph(node: Any, state: Any = Count) -> StateGraph[Any, Any, Any, Any]:
    g: StateGraph[Any, Any, Any, Any] = StateGraph(state)
    g.add_node("node", node, metadata=_ACTIVITY)
    g.add_edge(START, "node")
    return g


def _counter_plugin() -> LangGraphPlugin:
    return LangGraphPlugin(graphs={"counter": _one_node_graph(count_calls)})


@dataclass
class Chain:
    """Input and result of a Workflow that continues as new with its task cache."""

    graph: str = "counter"
    entrypoint: str = ""
    input: Any = field(default_factory=lambda: _INPUT)
    calls: int = 1
    fetch_for_each_call: bool = False
    runs_left: int = 0
    cache: dict[str, Any] | None = None
    workflow_id: str = ""
    run_ids: list[str] = field(default_factory=list)
    results: list[list[Any]] = field(default_factory=list)


def _end_run(chain: Chain, results: list[Any]) -> Chain:
    """Record a run's results, and continue as new with the cache if runs are left."""
    info = workflow.info()
    chain = replace(
        chain,
        workflow_id=info.workflow_id,
        run_ids=[*chain.run_ids, info.run_id],
        results=[*chain.results, results],
    )
    if chain.runs_left:
        workflow.continue_as_new(
            replace(chain, runs_left=chain.runs_left - 1, cache=cache())
        )
    return chain


def _fetch(chain: Chain) -> Any:
    """Fetch the chain's graph or entrypoint, passing its carried cache."""
    if chain.entrypoint:
        return entrypoint(chain.entrypoint, cache=chain.cache)
    return graph(chain.graph, cache=chain.cache).compile()


@workflow.defn
class ChainWorkflow:
    """Makes the same call ``calls`` times in each run of the chain."""

    @workflow.run
    async def run(self, chain: Chain) -> Chain:
        app = _fetch(chain)
        results: list[Any] = []
        for _ in range(chain.calls):
            if chain.fetch_for_each_call:
                app = _fetch(chain)
            results.append(await app.ainvoke(chain.input))
        return _end_run(chain, results)


@workflow.defn
class ChainInInitWorkflow:
    """A chain that fetches its graph or entrypoint in ``@workflow.init``."""

    @workflow.init
    def __init__(self, chain: Chain) -> None:
        self.app = _fetch(chain)

    @workflow.run
    async def run(self, chain: Chain) -> Chain:
        return _end_run(
            chain, [await self.app.ainvoke(chain.input) for _ in range(chain.calls)]
        )


async def _run_chains(
    client: Client,
    plugin: LangGraphPlugin,
    *chains: Chain,
    workflow_cls: Any = ChainWorkflow,
    one_thread: bool = False,
    max_cached_workflows: int = 1000,
) -> list[tuple[Chain, list[WorkflowHistory]]]:
    """Run chains one after another on a new Worker.

    Return each chain's result and the history of each of its runs. With
    ``one_thread``, each Workflow is created on the thread the last one used.
    """
    done: list[tuple[Chain, list[WorkflowHistory]]] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        async with Worker(
            client,
            task_queue=f"task-cache-{uuid4()}",
            workflows=[workflow_cls],
            plugins=[plugin],
            workflow_task_executor=executor if one_thread else None,
            max_cached_workflows=max_cached_workflows,
        ) as worker:
            for chain in chains:
                handle = await client.start_workflow(
                    workflow_cls.run,
                    chain,
                    id=f"task-cache-{uuid4()}",
                    task_queue=worker.task_queue,
                    execution_timeout=timedelta(seconds=30),
                )
                result: Chain = await handle.result()
                histories = [
                    await client.get_workflow_handle(
                        handle.id, run_id=run_id
                    ).fetch_history()
                    for run_id in result.run_ids
                ]
                done.append((result, histories))
    return done


def _scheduled(histories: list[WorkflowHistory]) -> list[int]:
    """Count the Activities each run scheduled."""
    return [
        sum(
            event.HasField("activity_task_scheduled_event_attributes")
            for event in history.events
        )
        for history in histories
    ]


async def _replay(
    plugin: LangGraphPlugin, workflow_cls: Any, histories: list[WorkflowHistory]
) -> None:
    """Replay as a restarted Worker could: on a new thread, last run first."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        replayer = Replayer(
            workflows=[workflow_cls], plugins=[plugin], workflow_task_executor=executor
        )
        for history in reversed(histories):
            await replayer.replay_workflow(history)


@pytest.mark.parametrize(
    "workflow_cls", [ChainWorkflow, ChainInInitWorkflow], ids=["run", "init"]
)
async def test_repeated_call_executes_in_every_workflow(
    client: Client, workflow_cls: Any
) -> None:
    """Each of three Workflows on one Worker thread runs its three calls."""
    plugin = _counter_plugin()
    chains = await _run_chains(
        client,
        plugin,
        *(Chain(calls=3) for _ in range(3)),
        workflow_cls=workflow_cls,
        one_thread=True,
    )

    for chain, histories in chains:
        assert chain.results == [
            [Count(count=n, owner=chain.workflow_id) for n in (1, 2, 3)]
        ]
        assert _scheduled(histories) == [3]
    await _replay(plugin, workflow_cls, [history for _, (history,) in chains])


@pytest.mark.parametrize(
    "max_cached_workflows,fetch_for_each_call",
    [(1000, False), (0, False), (1000, True)],
    ids=["cached", "evicted", "cache-passed-again"],
)
async def test_carried_result_is_served_once(
    client: Client, max_cached_workflows: int, fetch_for_each_call: bool
) -> None:
    """A call made twice per run gets both of the last run's results, in order.

    With no cached Workflows, every Workflow Task replays the run from its
    history first. Passing the carried cache again before the second call
    does not change that.
    """
    plugin = _counter_plugin()
    [(chain, histories)] = await _run_chains(
        client,
        plugin,
        Chain(calls=2, fetch_for_each_call=fetch_for_each_call, runs_left=2),
        max_cached_workflows=max_cached_workflows,
    )

    assert [[result["count"] for result in run] for run in chain.results] == [
        [1, 2],
        [1, 2],
        [1, 2],
    ]
    assert _scheduled(histories) == [2, 0, 0]
    await _replay(plugin, ChainWorkflow, histories)


async def test_cache_given_in_init_stays_with_its_workflow(client: Client) -> None:
    """The run that carried a cache uses it, and the next Workflow does not."""
    plugin = _counter_plugin()
    (first, first_histories), (second, second_histories) = await _run_chains(
        client,
        plugin,
        Chain(runs_left=1),
        Chain(),
        workflow_cls=ChainInInitWorkflow,
        one_thread=True,
    )

    assert first.results == [[Count(count=1, owner=first.workflow_id)]] * 2
    assert second.results == [[Count(count=1, owner=second.workflow_id)]]
    assert _scheduled(first_histories) == [1, 0]
    assert _scheduled(second_histories) == [1]
    await _replay(plugin, ChainInInitWorkflow, second_histories)
    await _replay(plugin, ChainInInitWorkflow, first_histories)


@workflow.defn
class TurnsWorkflow:
    """Runs a turn per Update and Signal, with the graph fetched in ``__init__``."""

    @workflow.init
    def __init__(self, chain: Chain) -> None:
        self.app = _fetch(chain)
        self.turns: list[Any] = []
        self.told = False

    @workflow.run
    async def run(self, chain: Chain) -> list[Any]:  # pyright: ignore[reportUnusedParameter]
        await workflow.wait_condition(lambda: self.told)
        return self.turns

    @workflow.update
    async def ask(self) -> None:
        self.turns.append(await self.app.ainvoke(_INPUT))

    @workflow.signal
    async def tell(self) -> None:
        self.turns.append(await self.app.ainvoke(_INPUT))
        self.told = True

    @workflow.query
    def turn_count(self) -> int:
        return len(self.turns)


async def test_handler_turns_use_their_own_workflows_cache(client: Client) -> None:
    """Two Workflows taking turns on one Worker thread each run every turn."""
    plugin = _counter_plugin()
    task_queue = f"turns-{uuid4()}"
    with ThreadPoolExecutor(max_workers=1) as executor:
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[TurnsWorkflow],
            plugins=[plugin],
            workflow_task_executor=executor,
        ):
            handles = [
                await client.start_workflow(
                    TurnsWorkflow.run,
                    Chain(),
                    id=f"turns-{uuid4()}",
                    task_queue=task_queue,
                    execution_timeout=timedelta(seconds=30),
                )
                for _ in range(2)
            ]
            for _ in range(2):
                for handle in handles:
                    await handle.execute_update(TurnsWorkflow.ask)
            for handle in handles:
                await handle.signal(TurnsWorkflow.tell)
            for handle in handles:
                assert await handle.result() == [
                    Count(count=n, owner=handle.id) for n in (1, 2, 3)
                ]
            histories = [await handle.fetch_history() for handle in handles]

    await _replay(plugin, TurnsWorkflow, histories)


async def test_replayer_beside_the_worker_leaves_an_open_run_alone(
    client: Client,
) -> None:
    """A Replayer in the Worker's process keeps out of the cache of the run it replays."""
    plugin = _counter_plugin()
    [(earlier, _)] = await _run_chains(client, plugin, Chain(runs_left=1))
    task_queue = f"open-run-{uuid4()}"
    async with Worker(
        client, task_queue=task_queue, workflows=[TurnsWorkflow], plugins=[plugin]
    ):
        handle = await client.start_workflow(
            TurnsWorkflow.run,
            Chain(cache=earlier.cache),
            id=f"open-run-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        # The run has started and holds the carried result, not served yet.
        assert await handle.query(TurnsWorkflow.turn_count) == 0
        await Replayer(workflows=[TurnsWorkflow], plugins=[plugin]).replay_workflow(
            await handle.fetch_history()
        )
        await handle.signal(TurnsWorkflow.tell)
        assert await handle.result() == earlier.results[0]
        history = await handle.fetch_history()

    assert _scheduled([history]) == [0]
    await _replay(plugin, TurnsWorkflow, [history])


class Value(TypedDict):
    value: int


def _counted(name: str) -> Any:
    """Return a node that counts its calls under ``name``."""

    async def node(state: Value) -> dict[str, int]:  # pyright: ignore[reportUnusedParameter]
        return {"value": _called(name)}

    return node


async def run_child(state: Value) -> dict[str, Any]:
    return await graph("child").compile().ainvoke(state)


@workflow.defn
class ManyGraphsWorkflow:
    """Passes its carried cache to the first of its graph() calls only."""

    @workflow.run
    async def run(self, chain: Chain) -> Chain:
        first = graph("first", cache=chain.cache).compile()
        second = graph("second").compile()
        parent = graph("parent").compile()
        return _end_run(
            chain,
            [await app.ainvoke(Value(value=0)) for app in (first, second, parent)],
        )


async def test_every_graph_of_a_run_uses_its_cache(client: Client) -> None:
    """A later graph() call and a child graph fetched in a node keep the cache."""
    _calls.clear()
    parent: StateGraph[Any, Any, Any, Any] = StateGraph(Value)
    parent.add_node("node", run_child, metadata={"execute_in": "workflow"})
    parent.add_edge(START, "node")
    plugin = LangGraphPlugin(
        graphs={
            "first": _one_node_graph(_counted("first"), Value),
            "second": _one_node_graph(_counted("second"), Value),
            "parent": parent,
            "child": _one_node_graph(_counted("child"), Value),
        }
    )
    [(chain, histories)] = await _run_chains(
        client, plugin, Chain(runs_left=1), workflow_cls=ManyGraphsWorkflow
    )

    assert chain.results == [[{"value": 1}] * 3] * 2
    assert _calls == {"first": 1, "second": 1, "child": 1}
    assert _scheduled(histories) == [3, 0]
    await _replay(plugin, ManyGraphsWorkflow, histories)


async def count_with_config(state: Value, config: RunnableConfig) -> dict[str, int]:  # pyright: ignore[reportUnusedParameter]
    return {"value": _called("config")}


class Messages(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def send_email(to: str) -> str:
    """Send an email."""
    _called(to)
    return f"sent to {to}"


async def test_node_that_gets_config_is_cached(client: Client) -> None:
    """The config LangGraph injects, as into a ToolNode, is not part of the key."""
    _calls.clear()
    plugin = LangGraphPlugin(
        graphs={
            "config": _one_node_graph(count_with_config, Value),
            "tools": _one_node_graph(ToolNode([send_email]), Messages),
        }
    )
    request = {
        "role": "assistant",
        "content": "",
        "id": "request-1",
        "tool_calls": [
            {"name": "send_email", "args": {"to": "ada@example.com"}, "id": "call-1"}
        ],
    }
    (_, config_histories), (_, tools_histories) = await _run_chains(
        client,
        plugin,
        Chain(graph="config", input=Value(value=0), runs_left=1),
        Chain(graph="tools", input={"messages": [request]}, runs_left=1),
    )

    assert _calls == {"config": 1, "ada@example.com": 1}
    assert _scheduled(config_histories) == [1, 0]
    assert _scheduled(tools_histories) == [1, 0]


class Question(TypedDict):
    question: str


class Answers(TypedDict):
    question: str
    answers: Annotated[list[str], operator.add]


class Speaker:
    def __init__(self, role: str) -> None:
        self.role = role

    async def answer(self, state: Question) -> dict[str, list[str]]:
        _called(self.role)
        return {"answers": [f"{self.role}: {state['question']}"]}


async def test_nodes_built_from_one_method_keep_their_own_results(
    client: Client,
) -> None:
    """Two nodes that share a function and get the same input both run.

    After continue-as-new each gets its own result back.
    """
    _calls.clear()
    g: StateGraph[Any, Any, Any, Any] = StateGraph(Answers)
    for role in ("researcher", "critic"):
        g.add_node(
            role, Speaker(role).answer, input_schema=Question, metadata=_ACTIVITY
        )
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "critic")
    question = {"question": "why", "answers": []}
    [(chain, histories)] = await _run_chains(
        client,
        LangGraphPlugin(graphs={"speakers": g}),
        Chain(graph="speakers", input=question, runs_left=1),
    )

    answers = {"question": "why", "answers": ["researcher: why", "critic: why"]}
    assert chain.results == [[answers], [answers]]
    assert _calls == {"researcher": 1, "critic": 1}
    assert _scheduled(histories) == [2, 0]


class Counts(TypedDict):
    counts: Annotated[list[int], operator.add]


class Item(TypedDict):
    item: str


async def count_item(state: Item) -> dict[str, list[int]]:
    return {"counts": [_called(state["item"])]}


def send_twice(state: Counts) -> list[Send]:  # pyright: ignore[reportUnusedParameter]
    return [Send("count_item", Item(item="sent")) for _ in range(2)]


@task
async def count_job(job: str) -> int:
    return _called(job)


@lg_entrypoint()
async def count_twice_at_once(job: str) -> list[int]:
    first, second = count_job(job), count_job(job)
    return [await first, await second]


async def test_parallel_calls_with_equal_input_all_run(client: Client) -> None:
    """Two identical calls started together both run.

    The next run gets both results back, each in its own place.
    """
    _calls.clear()
    fan_out: StateGraph[Any, Any, Any, Any] = StateGraph(Counts)
    fan_out.add_node("count_item", count_item, metadata=_ACTIVITY)
    fan_out.add_conditional_edges(START, send_twice, ["count_item"])
    plugin = LangGraphPlugin(
        graphs={"fan-out": fan_out},
        entrypoints={"count_twice_at_once": count_twice_at_once},
        tasks=[count_job],
        activity_options={"count_job": _ACTIVITY},
    )
    (sent, sent_histories), (jobs, job_histories) = await _run_chains(
        client,
        plugin,
        Chain(graph="fan-out", input={"counts": []}, runs_left=1),
        Chain(entrypoint="count_twice_at_once", input="job", runs_left=1),
    )

    for first, second in (
        [run[0]["counts"] for run in sent.results],
        [run[0] for run in jobs.results],
    ):
        assert sorted(first) == [1, 2]
        assert second == first
    for histories in (sent_histories, job_histories):
        assert _scheduled(histories) == [2, 0]
        await _replay(plugin, ChainWorkflow, histories)


async def command_then_result(state: Value) -> Any:  # pyright: ignore[reportUnusedParameter]
    """Return a Command for the first call and a plain result after that."""
    count = _called("kinds")
    return Command(update={"value": count}) if count == 1 else {"value": count}


async def test_repeated_calls_get_back_a_command_and_a_result(client: Client) -> None:
    """The next run gets back the first call's Command and the second call's result."""
    _calls.clear()
    [(chain, histories)] = await _run_chains(
        client,
        LangGraphPlugin(graphs={"kinds": _one_node_graph(command_then_result, Value)}),
        Chain(graph="kinds", input=Value(value=0), calls=2, runs_left=1),
    )

    assert chain.results == [[{"value": 1}, {"value": 2}]] * 2
    assert _scheduled(histories) == [2, 0]


@task
async def check_job(job: str) -> str:
    return "done" if _called(job) >= 3 else "running"


@lg_entrypoint()
async def wait_for_job(job: str) -> list[str]:
    seen: list[str] = []
    while not seen or seen[-1] != "done":
        seen.append(await check_job(job))
    return seen


class Job(TypedDict):
    job: str
    status: str


async def poll_job(state: Job) -> dict[str, str]:
    return {"status": "done" if _called(state["job"]) >= 3 else "running"}


def poll_again(state: Job) -> str:
    return END if state["status"] == "done" else "poll_job"


async def test_poll_loop_sees_the_job_finish(client: Client) -> None:
    """A task or node polled with the same input runs until its result changes."""
    _calls.clear()
    g: StateGraph[Any, Any, Any, Any] = StateGraph(Job)
    g.add_node("poll_job", poll_job, metadata=_ACTIVITY)
    g.add_edge(START, "poll_job")
    g.add_conditional_edges("poll_job", poll_again, ["poll_job", END])
    plugin = LangGraphPlugin(
        graphs={"poll": g},
        entrypoints={"wait_for_job": wait_for_job},
        tasks=[check_job],
        activity_options={"check_job": _ACTIVITY},
    )
    (polled, task_histories), (routed, node_histories) = await _run_chains(
        client,
        plugin,
        Chain(entrypoint="wait_for_job", input="task-job"),
        Chain(graph="poll", input=Job(job="node-job", status="")),
    )

    assert polled.results == [[["running", "running", "done"]]]
    assert routed.results == [[Job(job="node-job", status="done")]]
    assert _calls == {"task-job": 3, "node-job": 3}
    for histories in (task_histories, node_histories):
        assert _scheduled(histories) == [3]
        await _replay(plugin, ChainWorkflow, histories)


@task
async def add_one(number: int) -> int:
    return number + 1


@lg_entrypoint()
async def add_one_once(number: int) -> int:
    return await add_one(number)


async def test_task_key_is_its_module_and_qualname(client: Client) -> None:
    """A task's cache entries are keyed by its activity name, arguments and context.

    The key is written out, so a change to it fails here: a run that carries
    a cache through continue-as-new needs the same keys to replay. The seeded
    result differs from a real one, so a cache miss shows.
    """
    key = "8dbd7ebad4548ee623f6ba3e8d83ae19"
    assert key == cache_key(f"{__name__}.add_one", (1,), {}, None)
    plugin = LangGraphPlugin(
        entrypoints={"add_one_once": add_one_once},
        tasks=[add_one],
        activity_options={"add_one": _ACTIVITY},
    )
    [(chain, histories)] = await _run_chains(
        client, plugin, Chain(entrypoint="add_one_once", input=1, cache={key: 41})
    )

    assert chain.results == [[41]]
    assert _scheduled(histories) == [0]


class Request(TypedDict):
    action: str
    approved: bool


async def approve(state: Request) -> dict[str, bool]:
    return {"approved": interrupt(state["action"]) == "yes"}


@workflow.defn
class ApprovalWorkflow:
    """Makes the same request twice. A human approves the first one only."""

    @workflow.run
    async def run(self) -> list[bool]:
        app = graph("approval").compile(checkpointer=InMemorySaver())
        request = Request(action="wire $10,000", approved=False)
        first = RunnableConfig({"configurable": {"thread_id": "1"}})
        second = RunnableConfig({"configurable": {"thread_id": "2"}})
        asked = ["__interrupt__" in await app.ainvoke(request, first)]
        await app.ainvoke(Command(resume="yes"), first)
        asked.append("__interrupt__" in await app.ainvoke(request, second))
        return asked


async def test_second_identical_request_asks_for_approval_again(
    client: Client,
) -> None:
    """An approval is not reused for a later request with the same input."""
    plugin = LangGraphPlugin(graphs={"approval": _one_node_graph(approve, Request)})
    task_queue = f"approval-{uuid4()}"
    async with Worker(
        client, task_queue=task_queue, workflows=[ApprovalWorkflow], plugins=[plugin]
    ):
        asked = await client.execute_workflow(
            ApprovalWorkflow.run,
            id=f"approval-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )

    assert asked == [True, True]


async def test_finished_runs_leave_no_cache_behind(client: Client) -> None:
    """A run's cache is freed with its Workflow while the Worker keeps running."""

    def run_caches() -> int:
        gc.collect()
        return sum(type(item) is _RunCache for item in gc.get_objects())

    before = run_caches()
    task_queue = f"freed-{uuid4()}"
    with ThreadPoolExecutor(max_workers=1) as executor:
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ChainWorkflow],
            plugins=[_counter_plugin()],
            workflow_task_executor=executor,
        ):
            for _ in range(4):
                await client.execute_workflow(
                    ChainWorkflow.run,
                    Chain(runs_left=1),
                    id=f"freed-{uuid4()}",
                    task_queue=task_queue,
                    execution_timeout=timedelta(seconds=30),
                )
            # LangGraph leaves the Workflow thread's event loop set to the last
            # Workflow that ran a graph there, which keeps that one Workflow. The
            # Worker evicts a run a moment after it reports the run's result.
            for _ in range(100):
                if run_caches() <= before + 1:
                    break
                await asyncio.sleep(0.1)
            assert run_caches() <= before + 1


def test_carried_entries_are_served_once_and_replaced_by_new_results() -> None:
    async def main() -> dict[str, Any]:
        set_task_cache({"a": 1, "a:types": [], "b:command": {}, "c": 3})
        assert cache_lookup("a") == {"a": 1, "a:types": []}
        # Served once, also when the same cache is passed again.
        set_task_cache({"a": 1, "a:types": [], "d": 4})
        assert cache_lookup("a") is None
        # A result of this run replaces every carried entry of the call.
        cache_put("b", {"b": 2})
        assert cache_lookup("b") is None
        return get_task_cache()

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(main())
    finally:
        loop.close()
    assert result == {"c": 3, "d": 4, "a": 1, "a:types": [], "b": 2}


def test_cache_is_empty_outside_a_workflow() -> None:
    """With no event loop, and on a running loop that is not a Workflow's."""

    async def main() -> dict[str, Any] | None:
        return cache()

    assert cache() is None
    loop = asyncio.new_event_loop()
    try:
        assert loop.run_until_complete(main()) is None
        assert not hasattr(loop, _RUN_CACHE)
    finally:
        loop.close()


def test_run_cache_is_freed_with_its_event_loop() -> None:
    async def main() -> weakref.ref[_RunCache]:
        set_task_cache({"a": 1})
        return weakref.ref(_run_cache())

    loop = asyncio.new_event_loop()
    run_cache = loop.run_until_complete(main())
    assert run_cache() is not None
    loop.close()
    del loop
    gc.collect()
    assert run_cache() is None
