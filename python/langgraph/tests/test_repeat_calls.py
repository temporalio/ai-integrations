"""A node or task called again with the same input in one run executes again.

Native LangGraph runs every call. A poll loop that calls the same task, or a
node that routes back to itself, with the same input until a job is done must
see the job finish under the plugin too.
"""

import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import cache as once
from typing import Any
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


_wait = task(wait_for_value)
_next = task(next_value)
_fail_first = task(fail_first_call)


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
    async def two_subgraphs(request: dict[str, Any]) -> dict[str, str]:
        a = wait_then_subgraph(request["job_id"], True)
        b = wait_then_subgraph(request["job_id"], False)
        return {"a": await a, "b": await b}

    return LangGraphPlugin(
        graphs={"value": _value_graph},
        entrypoints={
            "two-branches": two_branches,
            "retry-on-failure": retry_on_failure,
            "retry-then-task": retry_then_task,
            "two-subgraphs": two_subgraphs,
        },
        tasks=[_wait, _next, _fail_first, wait_then_call, wait_then_subgraph],
        activity_options={
            "wait_for_value": {"execute_in": "activity"},
            "next_value": {"execute_in": "activity"},
            "fail_first_call": {"execute_in": "activity"},
            "wait_then_call": {"execute_in": "workflow"},
            "wait_then_subgraph": {"execute_in": "workflow"},
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

    Then continues-as-new with the cache until it has run ``runs`` times.
    Returns whether each run was interrupted, and its answer.
    """

    @workflow.run
    async def run(self, input: RepeatAcrossContinueAsNewInput) -> list[Any]:
        app = graph(input.name, cache=input.cache).compile(checkpointer=InMemorySaver())
        config = RunnableConfig({"configurable": {"thread_id": "1"}})
        result = await app.ainvoke({"job_id": input.job_id}, config)
        interrupted = "__interrupt__" in result
        if interrupted:
            result = await app.ainvoke(Command(resume="yes"), config)
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


class ApproveState(TypedDict):
    job_id: str
    answer: NotRequired[str]


def approve(state: ApproveState) -> dict[str, str]:
    """Take a new value, then ask for approval."""
    value = next_value(state["job_id"])
    return {"answer": f"{interrupt('approve?')} {value}"}


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
