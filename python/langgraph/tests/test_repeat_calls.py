"""A node or task called again with the same input in one run executes again.

Native LangGraph runs every call. A poll loop that calls the same task, or a
node that routes back to itself, with the same input until a job is done must
see the job finish under the plugin too.
"""

import asyncio
import operator
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import cache as once
from typing import Annotated, Any
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import (  # pyright: ignore[reportMissingTypeStubs]
    entrypoint as lg_entrypoint,
)
from langgraph.func import task  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from langgraph.types import Command, interrupt
from typing_extensions import NotRequired, TypedDict

from temporalio import workflow
from temporalio.client import Client, WorkflowHistory
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.langgraph import LangGraphPlugin, cache, entrypoint, graph
from temporalio.worker import Replayer, Worker

_needs_functional_api = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="LangGraph Functional API requires Python >= 3.11 for async context propagation",
)

_ACTIVITY_OPTIONS = {"start_to_close_timeout": timedelta(seconds=10)}

# Polls per job ID, counted where the Activities run.
_polls: dict[str, int] = {}


def check_job(job_id: str) -> str:
    """Report the job as done on its third poll."""
    _polls[job_id] = _polls.get(job_id, 0) + 1
    return "done" if _polls[job_id] >= 3 else "running"


def _wait_for_job_plugin() -> LangGraphPlugin:
    """An entrypoint that polls ``check_job`` until done, at most ``max_polls`` times."""
    check = task(check_job)

    @lg_entrypoint()
    async def wait_for_job(request: dict[str, Any]) -> list[str]:
        seen: list[str] = []
        for _ in range(request["max_polls"]):
            seen.append(await check(request["job_id"]))
            if seen[-1] == "done":
                break
        return seen

    return LangGraphPlugin(
        entrypoints={"wait-for-job": wait_for_job},
        tasks=[check],
        activity_options={"check_job": {"execute_in": "activity"}},
        default_activity_options=_ACTIVITY_OPTIONS,
    )


@workflow.defn
class WaitForJobWorkflow:
    @workflow.run
    async def run(self, job_id: str) -> Any:
        return await entrypoint("wait-for-job").ainvoke(
            {"job_id": job_id, "max_polls": 5}
        )


@dataclass
class WaitAcrossContinueAsNewInput:
    job_id: str
    cache: dict[str, Any] | None = None
    phase: int = 1


@workflow.defn
class WaitAcrossContinueAsNewWorkflow:
    """Polls once, continues-as-new with the cache, then polls until done."""

    @workflow.run
    async def run(self, input: WaitAcrossContinueAsNewInput) -> Any:
        if input.phase == 1:
            await entrypoint("wait-for-job").ainvoke(
                {"job_id": input.job_id, "max_polls": 1}
            )
            workflow.continue_as_new(
                WaitAcrossContinueAsNewInput(
                    job_id=input.job_id, cache=cache(), phase=2
                )
            )
        return await entrypoint("wait-for-job", cache=input.cache).ainvoke(
            {"job_id": input.job_id, "max_polls": 5}
        )


# Calls per job ID, counted where the Activities run.
_calls: dict[str, int] = {}


def next_value(job_id: str) -> str:
    """Return a new value on every call: ``<job_id>:1``, ``<job_id>:2``, ..."""
    _calls[job_id] = _calls.get(job_id, 0) + 1
    return f"{job_id}:{_calls[job_id]}"


def _repeat_plugin() -> LangGraphPlugin:
    """Entrypoints that call ``next_value`` with the same job ID ``calls`` times.

    ``repeat-in-sequence`` awaits each call before making the next one;
    ``repeat-in-parallel`` makes all the calls first, then awaits them.
    """
    fetch = task(next_value)

    @lg_entrypoint()
    async def repeat_in_sequence(request: dict[str, Any]) -> list[str]:
        return [await fetch(request["job_id"]) for _ in range(request["calls"])]

    @lg_entrypoint()
    async def repeat_in_parallel(request: dict[str, Any]) -> list[str]:
        futures = [fetch(request["job_id"]) for _ in range(request["calls"])]
        return [await future for future in futures]

    return LangGraphPlugin(
        entrypoints={
            "repeat-in-sequence": repeat_in_sequence,
            "repeat-in-parallel": repeat_in_parallel,
        },
        tasks=[fetch],
        activity_options={"next_value": {"execute_in": "activity"}},
        default_activity_options=_ACTIVITY_OPTIONS,
    )


def wait_for_value(job_id: str, wait: bool) -> bool:
    """If ``wait``, return once ``next_value`` has run for ``job_id``."""
    deadline = time.monotonic() + 10
    while wait and job_id not in _calls and time.monotonic() < deadline:
        time.sleep(0.01)
    return wait


def fail_first_call(job_id: str) -> str:
    """Fail the first call for ``job_id``; later calls return a new value."""
    value = next_value(job_id)
    if value.endswith(":1"):
        raise ApplicationError("first call fails", non_retryable=True)
    return value


def fail_second_call(job_id: str) -> str:
    """Fail the second call for ``job_id``; other calls return a new value."""
    value = next_value(job_id)
    if value.endswith(":2"):
        raise ApplicationError("second call fails", non_retryable=True)
    return value


def fail_twice(job_id: str) -> str:
    """Fail the first two calls for ``job_id``; later calls return a new value."""
    value = next_value(job_id)
    if value.endswith((":1", ":2")):
        raise ApplicationError("call fails", non_retryable=True)
    return value


# Calls of slow_first_call per job ID, counted where the Activities run.
_slow_calls: dict[str, int] = {}


def slow_first_call(job_id: str) -> str:
    """Take 2 seconds the first time for ``job_id``, return at once after."""
    _slow_calls[job_id] = _slow_calls.get(job_id, 0) + 1
    if _slow_calls[job_id] == 1:
        time.sleep(2)
        return "slow"
    return "fast"


_wait = task(wait_for_value)
_next = task(next_value)
_fail_first = task(fail_first_call)
_fail_second = task(fail_second_call)
_slow_first = task(slow_first_call)
_fail_twice = task(fail_twice)


@task
async def wait_then_call(job_id: str, wait: bool) -> str:
    """Await an Activity of its own, then make the call the other branch makes."""
    await _wait(job_id, wait)
    return await _next(job_id)


class ValueState(TypedDict):
    job_id: str
    value: NotRequired[str]


def fetch_value(state: ValueState) -> dict[str, str]:
    return {"value": next_value(state["job_id"])}


_value_graph = StateGraph(ValueState)
_value_graph.add_node("fetch", fetch_value, metadata={"execute_in": "activity"})
_value_graph.add_edge(START, "fetch")


@task
async def take_then_fail(job_id: str) -> list[str]:
    """Take a new value, then make a call that fails the first time."""
    return [await _next(job_id), await _fail_first(f"{job_id}-f")]


@task
async def take_unless(job_id: str, fail: bool) -> str:
    """Take a new value, then raise if ``fail``."""
    value = await _next(job_id)
    if fail:
        raise ValueError(f"failed after {value}")
    return value


@task
async def take_unless_bad(job_id: str, config: str) -> str:
    """Take a new value, then raise if ``config`` is "bad"."""
    value = await _next(job_id)
    if config == "bad":
        raise ValueError(f"failed after {value}")
    return value


async def take_then_fail_node(state: ValueState) -> dict[str, str]:
    """Like ``take_then_fail``, as a node."""
    first = await _next(state["job_id"])
    return {"value": f"{first} {await _fail_first(state['job_id'] + '-f')}"}


_recover_graph = StateGraph(ValueState)
_recover_graph.add_node(
    "take", take_then_fail_node, metadata={"execute_in": "workflow"}
)
_recover_graph.add_edge(START, "take")


async def take_twice_node(state: ValueState) -> dict[str, str]:
    """Make the same call twice; the second fails the first time it runs."""
    first = await _fail_second(state["job_id"])
    return {"value": f"{first} {await _fail_second(state['job_id'])}"}


_recover_twice_graph = StateGraph(ValueState)
_recover_twice_graph.add_node(
    "take", take_twice_node, metadata={"execute_in": "workflow"}
)
_recover_twice_graph.add_edge(START, "take")


async def take_three_node(state: ValueState) -> dict[str, str]:
    """Take a new value, then make two calls that each fail the first time."""
    job_id = state["job_id"]
    first = await _next(job_id)
    second = await _fail_first(f"{job_id}-b")
    return {"value": f"{first} {second} {await _fail_first(f'{job_id}-c')}"}


_recover_thrice_graph = StateGraph(ValueState)
_recover_thrice_graph.add_node(
    "take", take_three_node, metadata={"execute_in": "workflow"}
)
_recover_thrice_graph.add_edge(START, "take")


async def take_then_slow_node(state: ValueState) -> dict[str, str]:
    """Take a new value, then make a call that is slow the first time."""
    first = await _next(state["job_id"])
    return {"value": f"{first} {await _slow_first(state['job_id'])}"}


_cancel_graph = StateGraph(ValueState)
_cancel_graph.add_node("take", take_then_slow_node, metadata={"execute_in": "workflow"})
_cancel_graph.add_edge(START, "take")


class RouteState(TypedDict):
    job_id: str
    routes: Annotated[list[str], operator.add]


def wait_for_router(state: RouteState) -> dict[str, list[str]]:
    """Return once a router has made its call for the job."""
    wait_for_value(state["job_id"], True)
    return {"routes": []}


def no_wait(state: RouteState) -> dict[str, list[str]]:  # pyright: ignore[reportUnusedParameter]
    return {"routes": []}


def _route_by_value(source: str):
    async def route(state: RouteState) -> str:
        """Route to ``<source><n>``, where ``<n>`` is the call's count for the job."""
        value = await _next(state["job_id"])
        return f"{source}{value.rsplit(':', 1)[1]}"

    return route


def _record(name: str):
    def record(state: RouteState) -> dict[str, list[str]]:  # pyright: ignore[reportUnusedParameter]
        return {"routes": [name]}

    return record


_routes_graph = StateGraph(RouteState)
_routes_graph.add_node("a", wait_for_router, metadata={"execute_in": "activity"})
_routes_graph.add_node("b", no_wait, metadata={"execute_in": "activity"})
for _source in ("a", "b"):
    _routes_graph.add_edge(START, _source)
    _routes_graph.add_conditional_edges(
        _source, _route_by_value(_source), [f"{_source}1", f"{_source}2"]
    )
    for _target in (f"{_source}1", f"{_source}2"):
        _routes_graph.add_node(
            _target, _record(_target), metadata={"execute_in": "workflow"}
        )


class ValueClient:
    """A helper object made in the workflow; its ``str()`` shows its address."""

    def __init__(self) -> None:
        # Keep each client alive so that no later one gets its address, as
        # on another Worker.
        _clients.append(self)

    async def fetch(self, job_id: str) -> str:
        return await _next(job_id)


_clients: list[ValueClient] = []


@task
async def fetch_with(job_id: str, client: ValueClient) -> str:
    """Fetch a value through ``client``."""
    return await client.fetch(job_id)


@task
async def wait_then_subgraph(job_id: str, wait: bool) -> str:
    """Like ``wait_then_call``, but make the call through a subgraph."""
    await _wait(job_id, wait)
    result = await _value_graph.compile().ainvoke({"job_id": job_id})
    return result["value"]


@once
def _callers_plugin() -> LangGraphPlugin:
    """Entrypoints whose identical calls come from parallel tasks or follow a failure.

    Built once: the plugin wraps the module-level tasks above in place.
    """

    @lg_entrypoint()
    async def two_branches(request: dict[str, Any]) -> dict[str, str]:
        a = wait_then_call(request["job_id"], True)
        b = wait_then_call(request["job_id"], False)
        return {"a": await a, "b": await b}

    @lg_entrypoint()
    async def retry_on_failure(request: dict[str, Any]) -> list[str]:
        try:
            return [await _fail_first(request["job_id"])]
        except ActivityError:
            return ["failed", await _fail_first(request["job_id"])]

    @lg_entrypoint()
    async def retry_then_task(request: dict[str, Any]) -> list[str]:
        try:
            first = await _fail_first(request["job_id"])
        except ActivityError:
            first = await _fail_first(request["job_id"])
        return [first, await wait_then_call(request["job_id"], False)]

    @lg_entrypoint()
    async def retry_task(request: dict[str, Any]) -> list[str]:
        try:
            return await take_then_fail(request["job_id"])
        except ActivityError:
            return await take_then_fail(request["job_id"])

    @lg_entrypoint()
    async def other_input_after_failure(request: dict[str, Any]) -> str:
        try:
            return await take_unless(request["job_id"], True)
        except ValueError:
            return await take_unless(request["job_id"], False)

    @lg_entrypoint()
    async def two_subgraphs(request: dict[str, Any]) -> dict[str, str]:
        a = wait_then_subgraph(request["job_id"], True)
        b = wait_then_subgraph(request["job_id"], False)
        return {"a": await a, "b": await b}

    @lg_entrypoint()
    async def fetch_with_client(request: dict[str, Any]) -> str:
        return await fetch_with(request["job_id"], ValueClient())

    @lg_entrypoint()
    async def take_then_fail_entry(request: dict[str, Any]) -> list[str]:
        job_id = request["job_id"]
        return [await _next(job_id), await _fail_first(f"{job_id}-f")]

    @lg_entrypoint(checkpointer=InMemorySaver())
    async def fail_second_of_two(request: dict[str, Any]) -> list[str]:
        return [
            await _fail_second(request["job_id"]),
            await _fail_second(request["job_id"]),
        ]

    @lg_entrypoint()
    async def retry_inner_entrypoint(request: dict[str, Any]) -> Any:
        inner = {"job_id": request["job_id"]}
        try:
            return await take_then_fail_entry.ainvoke(inner)
        except ActivityError:
            return await take_then_fail_entry.ainvoke(inner)

    @lg_entrypoint()
    async def other_config_after_failure(request: dict[str, Any]) -> str:
        try:
            return await take_unless_bad(request["job_id"], config="bad")
        except ValueError:
            return await take_unless_bad(request["job_id"], config="good")

    @lg_entrypoint()
    async def wait_then_take(request: dict[str, Any]) -> str:
        await _wait(request["job_id"], request["wait"])
        return await _next(request["job_id"])

    @lg_entrypoint()
    async def two_inner_entrypoints(request: dict[str, Any]) -> list[Any]:
        job_id = request["job_id"]
        return list(
            await asyncio.gather(
                wait_then_take.ainvoke({"job_id": job_id, "wait": True}),
                wait_then_take.ainvoke({"job_id": job_id, "wait": False}),
            )
        )

    @lg_entrypoint()
    async def take_ask_then_fail(request: dict[str, Any]) -> list[str]:
        job_id = request["job_id"]
        first = await _next(job_id)
        interrupt("continue?")
        return [first, await _fail_first(f"{job_id}-f")]

    @lg_entrypoint(checkpointer=InMemorySaver())
    async def take_then_fail_saved(request: dict[str, Any]) -> list[str]:
        job_id = request["job_id"]
        return [await _next(job_id), await _fail_first(f"{job_id}-f")]

    @lg_entrypoint()
    async def recover_inner(request: dict[str, Any]) -> Any:
        job_id = request["job_id"]
        config = RunnableConfig({"configurable": {"thread_id": job_id}})
        try:
            inner = {"job_id": job_id, "pair": (1, 2)}
            return await take_then_fail_saved.ainvoke(inner, config)
        except ActivityError:
            return await take_then_fail_saved.ainvoke(None, config)

    @lg_entrypoint()
    async def recover_inner_then_fail(request: dict[str, Any]) -> list[str]:
        job_id = request["job_id"]
        config = RunnableConfig({"configurable": {"thread_id": job_id}})
        try:
            values = await take_then_fail_saved.ainvoke({"job_id": job_id}, config)
        except ActivityError:
            values = await take_then_fail_saved.ainvoke(None, config)
        return [*values, await _fail_first(f"{job_id}-o")]

    @lg_entrypoint(checkpointer=InMemorySaver())
    async def take_then_fail_twice(request: dict[str, Any]) -> list[str]:
        job_id = request["job_id"]
        values = [await _next(job_id)]
        for _ in range(2):
            values.append(await _fail_twice(f"{job_id}-f"))
        return values

    @lg_entrypoint()
    async def recover_inner_twice(request: dict[str, Any]) -> Any:
        job_id = request["job_id"]
        config = RunnableConfig({"configurable": {"thread_id": f"{job_id}-1"}})
        for attempt in ({"job_id": job_id}, None):
            try:
                return await take_then_fail_twice.ainvoke(attempt, config)
            except ActivityError:
                pass
        config = RunnableConfig({"configurable": {"thread_id": f"{job_id}-2"}})
        return await take_then_fail_twice.ainvoke({"job_id": job_id}, config)

    return LangGraphPlugin(
        graphs={
            "value": _value_graph,
            "recover": _recover_graph,
            "recover-twice": _recover_twice_graph,
            "recover-thrice": _recover_thrice_graph,
            "cancel": _cancel_graph,
            "routes": _routes_graph,
            "approve-then-fail": _approve_then_fail_graph,
        },
        entrypoints={
            "two-branches": two_branches,
            "retry-on-failure": retry_on_failure,
            "retry-then-task": retry_then_task,
            "retry-task": retry_task,
            "two-subgraphs": two_subgraphs,
            "other-input-after-failure": other_input_after_failure,
            "fetch-with-client": fetch_with_client,
            "take-then-fail": take_then_fail_entry,
            "fail-second-of-two": fail_second_of_two,
            "retry-inner-entrypoint": retry_inner_entrypoint,
            "other-config-after-failure": other_config_after_failure,
            "wait-then-take": wait_then_take,
            "two-inner-entrypoints": two_inner_entrypoints,
            "take-ask-then-fail": take_ask_then_fail,
            "take-then-fail-saved": take_then_fail_saved,
            "recover-inner": recover_inner,
            "recover-inner-then-fail": recover_inner_then_fail,
            "take-then-fail-twice": take_then_fail_twice,
            "recover-inner-twice": recover_inner_twice,
        },
        tasks=[
            _wait,
            _next,
            _fail_first,
            _fail_second,
            _slow_first,
            _fail_twice,
            wait_then_call,
            take_then_fail,
            take_unless,
            take_unless_bad,
            wait_then_subgraph,
            fetch_with,
        ],
        activity_options={
            "wait_for_value": {"execute_in": "activity"},
            "next_value": {"execute_in": "activity"},
            "fail_first_call": {"execute_in": "activity"},
            "fail_second_call": {"execute_in": "activity"},
            "slow_first_call": {"execute_in": "activity"},
            "fail_twice": {"execute_in": "activity"},
            "wait_then_call": {"execute_in": "workflow"},
            "take_then_fail": {"execute_in": "workflow"},
            "take_unless": {"execute_in": "workflow"},
            "take_unless_bad": {"execute_in": "workflow"},
            "wait_then_subgraph": {"execute_in": "workflow"},
            "fetch_with": {"execute_in": "workflow"},
        },
        default_activity_options=_ACTIVITY_OPTIONS,
    )


@dataclass
class RepeatAcrossContinueAsNewInput:
    name: str
    job_id: str
    runs: int
    cache: dict[str, Any] | None = None
    seen: list[Any] = field(default_factory=list)


@workflow.defn
class RepeatAcrossContinueAsNewWorkflow:
    """Run k makes k + 1 identical calls, then continues-as-new with the cache.

    Returns the results of every run, in run order.
    """

    @workflow.run
    async def run(self, input: RepeatAcrossContinueAsNewInput) -> list[Any]:
        result = await entrypoint(input.name, cache=input.cache).ainvoke(
            {"job_id": input.job_id, "calls": len(input.seen) + 2}
        )
        seen = [*input.seen, result]
        if len(seen) < input.runs:
            workflow.continue_as_new(
                RepeatAcrossContinueAsNewInput(
                    name=input.name,
                    job_id=input.job_id,
                    runs=input.runs,
                    cache=cache(),
                    seen=seen,
                )
            )
        return seen


@workflow.defn
class ApproveAcrossContinueAsNewWorkflow:
    """Runs graph ``name``, resuming it with "yes" if it interrupts.

    If the resumed run fails, runs the graph again on a new thread. Then
    continues-as-new with the cache until it has run ``runs`` times.
    Returns whether each run was interrupted, and its answer.
    """

    @workflow.run
    async def run(self, input: RepeatAcrossContinueAsNewInput) -> list[Any]:
        app = graph(input.name, cache=input.cache).compile(checkpointer=InMemorySaver())
        result: Any = None
        interrupted = False
        for thread in ("1", "2"):
            config = RunnableConfig({"configurable": {"thread_id": thread}})
            result = await app.ainvoke({"job_id": input.job_id}, config)
            interrupted = "__interrupt__" in result
            try:
                if interrupted:
                    result = await app.ainvoke(Command(resume="yes"), config)
                break
            except ActivityError:
                pass
        seen = [*input.seen, [interrupted, result["answer"]]]
        if len(seen) < input.runs:
            workflow.continue_as_new(
                RepeatAcrossContinueAsNewInput(
                    name=input.name,
                    job_id=input.job_id,
                    runs=input.runs,
                    cache=cache(),
                    seen=seen,
                )
            )
        return seen


@workflow.defn
class RecoverAcrossContinueAsNewWorkflow:
    """Runs graph ``name``, resuming it from its checkpoint if it fails.

    If the resumed run fails too, runs the graph again on a new thread. Then
    continues-as-new with the cache until it has run ``runs`` times.
    Returns each run's value.
    """

    @workflow.run
    async def run(self, input: RepeatAcrossContinueAsNewInput) -> list[Any]:
        app = graph(input.name, cache=input.cache).compile(checkpointer=InMemorySaver())
        config = RunnableConfig({"configurable": {"thread_id": "1"}})
        try:
            result = await app.ainvoke({"job_id": input.job_id}, config)
        except ActivityError:
            try:
                result = await app.ainvoke(None, config)
            except ActivityError:
                result = await app.ainvoke(
                    {"job_id": input.job_id},
                    RunnableConfig({"configurable": {"thread_id": "2"}}),
                )
        seen = [*input.seen, result["value"]]
        if len(seen) < input.runs:
            workflow.continue_as_new(
                RepeatAcrossContinueAsNewInput(
                    name=input.name,
                    job_id=input.job_id,
                    runs=input.runs,
                    cache=cache(),
                    seen=seen,
                )
            )
        return seen


@workflow.defn
class CancelAcrossContinueAsNewWorkflow:
    """In its first run, cancels graph ``name`` after a second, then runs it again.

    Then continues-as-new with the cache until it has run ``runs`` times.
    Returns each run's value.
    """

    @workflow.run
    async def run(self, input: RepeatAcrossContinueAsNewInput) -> list[Any]:
        app = graph(input.name, cache=input.cache).compile()
        if not input.seen:
            attempt = asyncio.ensure_future(app.ainvoke({"job_id": input.job_id}))
            await asyncio.sleep(1)
            attempt.cancel()
            try:
                await attempt
            except asyncio.CancelledError:
                pass
        result = await app.ainvoke({"job_id": input.job_id})
        seen = [*input.seen, result["value"]]
        if len(seen) < input.runs:
            workflow.continue_as_new(
                RepeatAcrossContinueAsNewInput(
                    name=input.name,
                    job_id=input.job_id,
                    runs=input.runs,
                    cache=cache(),
                    seen=seen,
                )
            )
        return seen


@workflow.defn
class RetryInvocationAcrossContinueAsNewWorkflow:
    """Invokes entrypoint ``name`` again if an invocation fails.

    Then continues-as-new with the cache until it has run ``runs`` times.
    Returns each run's result.
    """

    @workflow.run
    async def run(self, input: RepeatAcrossContinueAsNewInput) -> list[Any]:
        app = entrypoint(input.name, cache=input.cache)
        try:
            result = await app.ainvoke({"job_id": input.job_id})
        except ActivityError:
            result = await app.ainvoke({"job_id": input.job_id})
        seen = [*input.seen, result]
        if len(seen) < input.runs:
            workflow.continue_as_new(
                RepeatAcrossContinueAsNewInput(
                    name=input.name,
                    job_id=input.job_id,
                    runs=input.runs,
                    cache=cache(),
                    seen=seen,
                )
            )
        return seen


@workflow.defn
class RecoverInvocationAcrossContinueAsNewWorkflow:
    """Invokes entrypoint ``name``, resuming it from its checkpoint if it fails.

    Then continues-as-new with the cache until it has run ``runs`` times.
    Returns each run's result.
    """

    @workflow.run
    async def run(self, input: RepeatAcrossContinueAsNewInput) -> list[Any]:
        app = entrypoint(input.name, cache=input.cache)
        config = RunnableConfig(
            {"configurable": {"thread_id": f"{input.job_id}-{len(input.seen)}"}}
        )
        try:
            result = await app.ainvoke({"job_id": input.job_id}, config)
        except ActivityError:
            result = await app.ainvoke(None, config)
        seen = [*input.seen, result]
        if len(seen) < input.runs:
            workflow.continue_as_new(
                RepeatAcrossContinueAsNewInput(
                    name=input.name,
                    job_id=input.job_id,
                    runs=input.runs,
                    cache=cache(),
                    seen=seen,
                )
            )
        return seen


@workflow.defn
class ApproveInvocationAcrossContinueAsNewWorkflow:
    """Like ``ApproveAcrossContinueAsNewWorkflow``, for entrypoint ``name``.

    Returns each run's result.
    """

    @workflow.run
    async def run(self, input: RepeatAcrossContinueAsNewInput) -> list[Any]:
        app = entrypoint(input.name, cache=input.cache).copy(
            {"checkpointer": InMemorySaver()}
        )
        result: Any = None
        for thread in ("1", "2"):
            config = RunnableConfig({"configurable": {"thread_id": thread}})
            result = await app.ainvoke({"job_id": input.job_id}, config)
            try:
                if "__interrupt__" in result:
                    result = await app.ainvoke(Command(resume="yes"), config)
                break
            except ActivityError:
                pass
        seen = [*input.seen, result]
        if len(seen) < input.runs:
            workflow.continue_as_new(
                RepeatAcrossContinueAsNewInput(
                    name=input.name,
                    job_id=input.job_id,
                    runs=input.runs,
                    cache=cache(),
                    seen=seen,
                )
            )
        return seen


@workflow.defn
class RouteAcrossContinueAsNewWorkflow:
    """Runs graph ``name`` and records the nodes its routers chose.

    Then continues-as-new with the cache until it has run ``runs`` times.
    Returns each run's routes.
    """

    @workflow.run
    async def run(self, input: RepeatAcrossContinueAsNewInput) -> list[Any]:
        app = graph(input.name, cache=input.cache).compile()
        result = await app.ainvoke({"job_id": input.job_id, "routes": []})
        seen = [*input.seen, result["routes"]]
        if len(seen) < input.runs:
            workflow.continue_as_new(
                RepeatAcrossContinueAsNewInput(
                    name=input.name,
                    job_id=input.job_id,
                    runs=input.runs,
                    cache=cache(),
                    seen=seen,
                )
            )
        return seen


async def _repeat_across_continue_as_new(
    client: Client,
    name: str,
    job_id: str,
    runs: int,
    plugin: LangGraphPlugin | None = None,
    workflow_cls: Any = RepeatAcrossContinueAsNewWorkflow,
) -> tuple[list[Any], list[WorkflowHistory]]:
    """Run ``name`` over ``runs`` runs; return each run's results and history.

    Each run's history is replayed.
    """
    plugin = plugin or _repeat_plugin()
    task_queue = f"repeat-runs-{uuid4()}"
    workflow_id = f"repeat-runs-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[workflow_cls],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            workflow_cls.run,
            RepeatAcrossContinueAsNewInput(name=name, job_id=job_id, runs=runs),
            id=workflow_id,
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        seen = await handle.result()

    histories: list[WorkflowHistory] = []
    run_id = handle.first_execution_run_id
    while run_id:
        history = await client.get_workflow_handle(
            workflow_id, run_id=run_id
        ).fetch_history()
        histories.append(history)
        last = history.events[-1]
        run_id = (
            last.workflow_execution_continued_as_new_event_attributes.new_execution_run_id
            if last.HasField("workflow_execution_continued_as_new_event_attributes")
            else None
        )
    assert len(histories) == runs

    replayer = Replayer(workflows=[workflow_cls], plugins=[plugin])
    for history in histories:
        await replayer.replay_workflow(history)
    return seen, histories


def _scheduled(history: WorkflowHistory) -> list[int]:
    """Return the positions of the run's ActivityTaskScheduled events."""
    return [
        i
        for i, event in enumerate(history.events)
        if event.HasField("activity_task_scheduled_event_attributes")
    ]


def _completed(history: WorkflowHistory) -> list[int]:
    """Return the positions of the run's ActivityTaskCompleted events."""
    return [
        i
        for i, event in enumerate(history.events)
        if event.HasField("activity_task_completed_event_attributes")
    ]


class PollState(TypedDict):
    job_id: str
    status: NotRequired[str]


def poll(state: PollState) -> dict[str, str]:
    return {"status": check_job(state["job_id"])}


async def _route(state: PollState) -> str:
    return END if state.get("status") == "done" else "poll"


@workflow.defn
class PollGraphWorkflow:
    @workflow.run
    async def run(self, job_id: str) -> Any:
        return await graph("poller").compile().ainvoke({"job_id": job_id})


@_needs_functional_api
async def test_repeated_task_call_runs_again(client: Client):
    """Each poll runs its Activity, so the loop sees the job finish."""
    job_id = f"job-{uuid4()}"
    plugin = _wait_for_job_plugin()
    task_queue = f"repeat-task-{uuid4()}"

    async with Worker(
        client, task_queue=task_queue, workflows=[WaitForJobWorkflow], plugins=[plugin]
    ):
        handle = await client.start_workflow(
            WaitForJobWorkflow.run,
            job_id,
            id=f"repeat-task-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        seen = await handle.result()

    assert (_polls[job_id], seen) == (3, ["running", "running", "done"])
    await Replayer(workflows=[WaitForJobWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )


async def test_repeated_node_call_runs_again(client: Client):
    """A node that routes back to itself with the same state runs each time.

    The second and third polls get the same state, so a cached third poll
    would never await an Activity, and the loop would never yield.
    """
    job_id = f"job-{uuid4()}"
    g = StateGraph(PollState)
    g.add_node("poll", poll, metadata={"execute_in": "activity"})
    g.add_edge(START, "poll")
    g.add_conditional_edges("poll", _route)
    plugin = LangGraphPlugin(
        graphs={"poller": g}, default_activity_options=_ACTIVITY_OPTIONS
    )
    task_queue = f"repeat-node-{uuid4()}"

    async with Worker(
        client, task_queue=task_queue, workflows=[PollGraphWorkflow], plugins=[plugin]
    ):
        handle = await client.start_workflow(
            PollGraphWorkflow.run,
            job_id,
            id=f"repeat-node-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=15),
        )
        result = await handle.result()

    assert (_polls[job_id], result) == (3, {"job_id": job_id, "status": "done"})
    await Replayer(workflows=[PollGraphWorkflow], plugins=[plugin]).replay_workflow(
        await handle.fetch_history()
    )


@_needs_functional_api
async def test_repeated_call_after_continue_as_new(client: Client):
    """After continue-as-new, the cache answers the call made before, and repeats run.

    The first run polls once. The next run polls until done: its first call
    gets the cached "running", and the two repeats run as Activities.
    """
    job_id = f"job-{uuid4()}"
    plugin = _wait_for_job_plugin()
    task_queue = f"repeat-continue-as-new-{uuid4()}"
    workflow_id = f"repeat-continue-as-new-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[WaitAcrossContinueAsNewWorkflow],
        plugins=[plugin],
    ):
        handle = await client.start_workflow(
            WaitAcrossContinueAsNewWorkflow.run,
            WaitAcrossContinueAsNewInput(job_id=job_id),
            id=workflow_id,
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        seen = await handle.result()

    assert (_polls[job_id], seen) == (3, ["running", "running", "done"])
    replayer = Replayer(workflows=[WaitAcrossContinueAsNewWorkflow], plugins=[plugin])
    await replayer.replay_workflow(await handle.fetch_history())
    await replayer.replay_workflow(
        await client.get_workflow_handle(workflow_id).fetch_history()
    )


@_needs_functional_api
async def test_repeated_calls_restored_in_order_after_continue_as_new(
    client: Client,
):
    """Completed repeats come back from the cache in order; only the new call runs.

    Run k makes k + 1 identical calls one after another, so each run after
    the first finds its earlier calls under the plain key and ``#1``, ``#2``,
    ... and schedules an Activity for its last call only. Every Activity
    returns a new value, so a result restored in the wrong place shows.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "repeat-in-sequence", job_id, runs=3
    )

    values = [f"{job_id}:{n}" for n in range(1, 5)]
    assert seen == [values[:2], values[:3], values[:4]]
    assert [len(_scheduled(h)) for h in histories] == [2, 1, 1]
    assert _calls[job_id] == 4


@_needs_functional_api
async def test_parallel_repeated_calls_after_continue_as_new(client: Client):
    """Identical calls made at once get separate results that survive continue-as-new.

    Run k starts k + 1 identical calls before awaiting any of them, so they
    take their occurrence numbers from one shared count while running as
    separate tasks. The first run's calls are scheduled together and return
    different values; each later run gets the earlier results back in the
    same places and schedules an Activity for its last call only.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "repeat-in-parallel", job_id, runs=3
    )

    scheduled = _scheduled(histories[0])
    assert len(scheduled) == 2
    assert max(scheduled) < min(_completed(histories[0]))
    assert sorted(seen[0]) == [f"{job_id}:1", f"{job_id}:2"]
    assert seen[1][:2] == seen[0]
    assert seen[2][:3] == seen[1]
    assert sorted(seen[2]) == [f"{job_id}:{n}" for n in range(1, 5)]
    assert [len(_scheduled(h)) for h in histories] == [2, 1, 1]
    assert _calls[job_id] == 4


@_needs_functional_api
async def test_parallel_callers_keep_their_results_after_continue_as_new(
    client: Client,
):
    """Identical calls from two parallel tasks are matched per task, not by arrival.

    Each workflow-side task awaits an Activity of its own and then makes the
    same call. In the first run, branch "a"'s Activity returns only after
    "b"'s call has run, so "b" makes the call first. After continue-as-new
    both Activities come from the cache, so "a" reaches the call first, and
    must still get its own earlier result.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "two-branches", job_id, runs=2, plugin=_callers_plugin()
    )

    first = {"a": f"{job_id}:2", "b": f"{job_id}:1"}
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_retry_after_failure_restored_after_continue_as_new(client: Client):
    """After continue-as-new, a call that failed before gives its place to its retry.

    The first run's first call fails and the entrypoint calls again. The
    next run's first call gets the retry's result from the cache and runs
    nothing.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "retry-on-failure", job_id, runs=2, plugin=_callers_plugin()
    )

    assert seen == [["failed", f"{job_id}:2"], [f"{job_id}:2"]]
    assert [len(_scheduled(h)) for h in histories] == [2, 0]
    assert _calls[job_id] == 2


@_needs_functional_api
async def test_task_after_retry_restored_after_continue_as_new(client: Client):
    """A workflow-side task called after a retried failure keeps its results.

    After continue-as-new the first call gets the retry's result without
    failing, so the entrypoint makes one call fewer before the task. The
    task's own calls must still find their results.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "retry-then-task", job_id, runs=2, plugin=_callers_plugin()
    )

    first = [f"{job_id}:2", f"{job_id}:3"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_retried_task_restored_after_continue_as_new(client: Client):
    """A retried workflow-side task gets its retry's results after continue-as-new.

    The task's first call succeeds and its second fails, so the entrypoint
    calls the task again. The next run's task call must get the retry's
    results, not the failed attempt's first value.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "retry-task", job_id, runs=2, plugin=_callers_plugin()
    )

    first = [f"{job_id}:2", f"{job_id}-f:2"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_failed_task_with_other_input_keeps_results_after_continue_as_new(
    client: Client,
):
    """A failed task call called again with other input keeps its own results.

    The first call takes a value and raises; the entrypoint then calls the
    task with other input. After continue-as-new the failed call must not
    take the second call's result, so the second call gets its value back
    from the cache and nothing runs.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "other-input-after-failure",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
    )

    assert seen == [f"{job_id}:2", f"{job_id}:2"]
    assert [len(_scheduled(h)) for h in histories] == [2, 0]
    assert _calls[job_id] == 2


@_needs_functional_api
async def test_node_resumed_from_checkpoint_keeps_results_after_continue_as_new(
    client: Client,
):
    """A failed node resumed from its checkpoint keeps its first call's result.

    The node's first call succeeds and its second fails. Resuming the graph
    from its checkpoint gets the first call's result from the checkpoint,
    without calling it, and runs the second again. After continue-as-new
    both come from the cache and nothing runs.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "recover",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
        workflow_cls=RecoverAcrossContinueAsNewWorkflow,
    )

    first = f"{job_id}:1 {job_id}-f:2"
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [3, 0]
    assert _calls[job_id] == 1


@_needs_functional_api
async def test_parallel_subgraph_calls_keep_their_results_after_continue_as_new(
    client: Client,
):
    """Two parallel tasks that invoke the same subgraph keep their own results.

    As in ``test_parallel_callers_keep_their_results_after_continue_as_new``,
    branch "b" reaches the subgraph first in the first run and "a" does after
    continue-as-new.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "two-subgraphs", job_id, runs=2, plugin=_callers_plugin()
    )

    first = {"a": f"{job_id}:2", "b": f"{job_id}:1"}
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_node_resumed_with_repeated_call_keeps_results_after_continue_as_new(
    client: Client,
):
    """A node resumed from its checkpoint keeps the places of its identical calls.

    The node makes the same call twice and the second fails. Resuming the
    graph from its checkpoint gets the first call's result from the
    checkpoint and runs the second again, which must take the second place,
    not the first. After continue-as-new both come from the cache and
    nothing runs.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "recover-twice",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
        workflow_cls=RecoverAcrossContinueAsNewWorkflow,
    )

    first = f"{job_id}:1 {job_id}:3"
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [3, 0]
    assert _calls[job_id] == 3


@_needs_functional_api
async def test_node_failing_again_after_resuming_keeps_results_after_continue_as_new(
    client: Client,
):
    """A node that fails again after resuming from its checkpoint is retried in full.

    The node's first call succeeds and its second fails. Resumed from its
    checkpoint, it gets the first call's result from the checkpoint, the
    second succeeds and the third fails. The graph then runs again on a new
    thread, which makes all three calls again. After continue-as-new all
    three get the last attempt's results, including the first call, which
    the resumed attempt did not make.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "recover-thrice",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
        workflow_cls=RecoverAcrossContinueAsNewWorkflow,
    )

    first = f"{job_id}:2 {job_id}-b:3 {job_id}-c:2"
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [7, 0]


@_needs_functional_api
async def test_cancelled_node_keeps_results_after_continue_as_new(client: Client):
    """A node cancelled after one of its calls finished is retried in full.

    The graph is cancelled while the node's second call runs, after its
    first returned, and the workflow runs the graph again. After
    continue-as-new the first call gets the retry's result, not the
    cancelled attempt's.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "cancel",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
        workflow_cls=CancelAcrossContinueAsNewWorkflow,
    )

    first = f"{job_id}:2 fast"
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_invocation_retried_after_failure_keeps_results_after_continue_as_new(
    client: Client,
):
    """An entrypoint invoked again after a failed invocation keeps the retry's results.

    The entrypoint's first call succeeds and its second fails, so the
    workflow invokes it again, which makes both calls again. After
    continue-as-new the one invocation gets both of the retry's results and
    runs nothing.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "take-then-fail",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
        workflow_cls=RetryInvocationAcrossContinueAsNewWorkflow,
    )

    first = [f"{job_id}:2", f"{job_id}-f:2"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_invocation_resumed_from_checkpoint_keeps_results_after_continue_as_new(
    client: Client,
):
    """An entrypoint resumed from its checkpoint keeps the restored call's result.

    The entrypoint makes the same call twice and the second fails, so the
    workflow resumes it from its checkpoint: LangGraph restores the first
    call without making it, and the second runs again. After
    continue-as-new the run gets the same two results and runs nothing.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "fail-second-of-two",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
        workflow_cls=RecoverInvocationAcrossContinueAsNewWorkflow,
    )

    first = [f"{job_id}:1", f"{job_id}:3"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [3, 0]


@_needs_functional_api
async def test_nested_invocation_retried_after_failure_keeps_results_after_continue_as_new(
    client: Client,
):
    """An entrypoint retried by an enclosing one keeps the retry's results.

    The outer entrypoint catches the inner one's failure and invokes it
    again, so the outer invocation never fails. After continue-as-new the
    run still gets both of the retry's results and runs nothing.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "retry-inner-entrypoint",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
    )

    first = [f"{job_id}:2", f"{job_id}-f:2"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_parallel_routers_keep_their_results_after_continue_as_new(
    client: Client,
):
    """Routers of parallel nodes that make the same call keep their own results.

    Node "a"'s Activity returns only after "b"'s router has made the call, so
    in the first run "b" routes to "b1" and "a" to "a2". After
    continue-as-new both nodes come from the cache and "a"'s router reaches
    the call first, but must still get its own earlier result.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "routes",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
        workflow_cls=RouteAcrossContinueAsNewWorkflow,
    )

    assert seen == [["a2", "b1"], ["a2", "b1"]]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_task_given_a_helper_keeps_results_after_continue_as_new(
    client: Client,
):
    """A workflow-side task given a helper object finds its results after continue-as-new.

    Each run makes a new helper, whose ``str()`` shows its address. That
    differs in every run and on every Worker, so it must not decide which
    results the task's calls get.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "fetch-with-client", job_id, runs=2, plugin=_callers_plugin()
    )

    assert seen == [f"{job_id}:1", f"{job_id}:1"]
    assert [len(_scheduled(h)) for h in histories] == [1, 0]


@_needs_functional_api
async def test_failed_task_with_other_config_keeps_results_after_continue_as_new(
    client: Client,
):
    """A task's own ``config`` argument is input, like any other.

    As in ``test_failed_task_with_other_input_keeps_results_after_continue_as_new``,
    with the two calls told apart only by ``config="bad"`` and ``config="good"``.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "other-config-after-failure",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
    )

    assert seen == [f"{job_id}:2", f"{job_id}:2"]
    assert [len(_scheduled(h)) for h in histories] == [2, 0]


@_needs_functional_api
async def test_parallel_nested_invocations_keep_their_results_after_continue_as_new(
    client: Client,
):
    """Two parallel invocations of an inner entrypoint keep their own results.

    As in ``test_parallel_callers_keep_their_results_after_continue_as_new``,
    the first invocation makes the shared call second in the first run and
    first after continue-as-new.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "two-inner-entrypoints", job_id, runs=2, plugin=_callers_plugin()
    )

    first = [f"{job_id}:2", f"{job_id}:1"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_invocation_failing_after_interrupt_keeps_results_after_continue_as_new(
    client: Client,
):
    """An entrypoint that fails after resuming from an interrupt is retried in full.

    LangGraph restores the call made before the interrupt, the resumed
    invocation fails, and the workflow invokes the entrypoint again on a new
    thread. After continue-as-new the first call gets the retry's result.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "take-ask-then-fail",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
        workflow_cls=ApproveInvocationAcrossContinueAsNewWorkflow,
    )

    first = [f"{job_id}:2", f"{job_id}-f:2"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_node_failing_after_interrupt_keeps_results_after_continue_as_new(
    client: Client,
):
    """A workflow-side node that fails after resuming from an interrupt is retried in full."""
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "approve-then-fail",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
        workflow_cls=ApproveAcrossContinueAsNewWorkflow,
    )

    first = [True, f"yes {job_id}:2 {job_id}-f:2"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [4, 0]


@_needs_functional_api
async def test_nested_invocation_resumed_from_checkpoint_keeps_results_after_continue_as_new(
    client: Client,
):
    """An entrypoint resumed by an enclosing one keeps the restored call's result.

    Its input holds a tuple, which LangGraph gives back from the checkpoint
    as a list when it resumes the invocation.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "recover-inner", job_id, runs=2, plugin=_callers_plugin()
    )

    first = [f"{job_id}:1", f"{job_id}-f:2"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [3, 0]


@_needs_functional_api
async def test_invocation_failing_after_nested_recovery_keeps_results_after_continue_as_new(
    client: Client,
):
    """An entrypoint that fails after an inner one resumed from its checkpoint.

    The inner entrypoint's second call fails, and it resumes from its
    checkpoint, which restores its first call. Then the outer entrypoint's
    own call fails, and the workflow invokes it again. After continue-as-new
    the inner first call gets the retry's result.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "recover-inner-then-fail",
        job_id,
        runs=2,
        plugin=_callers_plugin(),
        workflow_cls=RetryInvocationAcrossContinueAsNewWorkflow,
    )

    first = [f"{job_id}:2", f"{job_id}-f:3", f"{job_id}-o:2"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [7, 0]


@_needs_functional_api
async def test_invocation_failing_again_after_resuming_keeps_results_after_continue_as_new(
    client: Client,
):
    """Identical calls keep their own places after an invocation fails twice.

    The inner entrypoint fails in the same call before and after resuming
    from its checkpoint, so it is invoked again on a new thread, where that
    call and an identical one after it both run.
    """
    job_id = f"job-{uuid4()}"
    seen, histories = await _repeat_across_continue_as_new(
        client, "recover-inner-twice", job_id, runs=2, plugin=_callers_plugin()
    )

    first = [f"{job_id}:2", f"{job_id}-f:3", f"{job_id}-f:4"]
    assert seen == [first, first]
    assert [len(_scheduled(h)) for h in histories] == [6, 0]


class ApproveState(TypedDict):
    job_id: str
    answer: NotRequired[str]


def approve(state: ApproveState) -> dict[str, str]:
    """Take a new value, then ask for approval."""
    value = next_value(state["job_id"])
    return {"answer": f"{interrupt('approve?')} {value}"}


async def approve_then_fail(state: ApproveState) -> dict[str, str]:
    """Take a new value, ask for approval, then make a call that fails the first time."""
    first = await _next(state["job_id"])
    answer = interrupt("approve?")
    return {"answer": f"{answer} {first} {await _fail_first(state['job_id'] + '-f')}"}


_approve_then_fail_graph = StateGraph(ApproveState)
_approve_then_fail_graph.add_node(
    "approve", approve_then_fail, metadata={"execute_in": "workflow"}
)
_approve_then_fail_graph.add_edge(START, "approve")


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="interrupt() requires Python >= 3.11 for async context propagation",
)
async def test_interrupted_call_restored_after_continue_as_new(client: Client):
    """An interrupted call gives its place to the resumed call.

    The first run's node interrupts and is resumed, so its Activity runs
    twice. After continue-as-new the run gets the resumed answer from the
    cache: no interrupt, no Activity.
    """
    job_id = f"job-{uuid4()}"
    g = StateGraph(ApproveState)
    g.add_node("approve", approve, metadata={"execute_in": "activity"})
    g.add_edge(START, "approve")
    plugin = LangGraphPlugin(
        graphs={"approve": g}, default_activity_options=_ACTIVITY_OPTIONS
    )
    seen, histories = await _repeat_across_continue_as_new(
        client,
        "approve",
        job_id,
        runs=2,
        plugin=plugin,
        workflow_cls=ApproveAcrossContinueAsNewWorkflow,
    )

    assert seen == [[True, f"yes {job_id}:2"], [False, f"yes {job_id}:2"]]
    assert [len(_scheduled(h)) for h in histories] == [2, 0]
    assert _calls[job_id] == 2
