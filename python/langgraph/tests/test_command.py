import operator
from datetime import timedelta
from typing import Annotated, Any, Literal
from uuid import uuid4

import pytest
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from langgraph.types import Command, Send, TimeoutPolicy
from typing_extensions import TypedDict

from temporalio import workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.converter import DataConverter
from temporalio.langgraph import LangGraphPlugin, graph
from temporalio.langgraph._activity import (
    ActivityOutput,
    _command_from_payload,
    _command_to_payload,
)
from temporalio.worker import Worker


class State(TypedDict):
    value: str


def node_a(state: State) -> Command[Literal["node_b"]]:
    return Command(update={"value": state["value"] + "a"}, goto="node_b")


def node_b(state: State) -> Command[Literal["__end__"]]:
    return Command(update={"value": state["value"] + "b"}, goto="__end__")


@workflow.defn
class CommandWorkflow:
    def __init__(self) -> None:
        self.app = graph("my-graph").compile()

    @workflow.run
    async def run(self, input: str) -> Any:
        return await self.app.ainvoke({"value": input})


async def test_command_goto_and_update(client: Client):
    g = StateGraph(State)
    g.add_node("node_a", node_a, metadata={"execute_in": "activity"})
    g.add_node("node_b", node_b, metadata={"execute_in": "activity"})
    g.add_edge(START, "node_a")

    task_queue = f"command-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CommandWorkflow],
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
            CommandWorkflow.run,
            "",
            id=f"test-command-{uuid4()}",
            task_queue=task_queue,
        )

    assert result == {"value": "ab"}


class FanOutState(TypedDict):
    mode: str
    results: Annotated[list[str], operator.add]


class WorkerState(TypedDict):
    item: str


def dispatch(state: FanOutState) -> Command | list[Command]:
    mode = state["mode"]
    if mode == "single":
        return Command(goto=Send("worker", {"item": "a"}))
    if mode == "list":
        return Command(
            goto=[Send("worker", {"item": "a"}), Send("worker", {"item": "b"})]
        )
    if mode == "mixed":
        return Command(goto=["tail", Send("worker", {"item": "a"})])
    if mode == "update":
        return Command(
            update={"results": ["dispatched"]},
            goto=Send("worker", {"item": "a"}),
        )
    if mode == "timeout":
        return Command(goto=Send("worker", {"item": "a"}, timeout=30))
    if mode == "command_list":
        return [
            Command(update={"results": ["listed"]}, goto=Send("worker", {"item": "a"}))
        ]
    raise ValueError(mode)


def worker(state: WorkerState) -> dict[str, list[str]]:
    return {"results": [state["item"].upper()]}


def tail(state: FanOutState) -> dict[str, list[str]]:
    return {"results": [state["mode"]]}


@workflow.defn
class CommandSendWorkflow:
    def __init__(self) -> None:
        self.app = graph("send-graph").compile()

    @workflow.run
    async def run(self, mode: str) -> Any:
        return await self.app.ainvoke({"mode": mode, "results": []})


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("single", ["A"]),
        ("list", ["A", "B"]),
        ("mixed", ["A", "mixed"]),
        ("update", ["A", "dispatched"]),
        ("timeout", ["A"]),
        ("command_list", ["A", "listed"]),
    ],
)
async def test_command_goto_send(client: Client, mode: str, expected: list[str]):
    g = StateGraph(FanOutState)
    g.add_node("dispatch", dispatch, metadata={"execute_in": "activity"})
    g.add_node("worker", worker, metadata={"execute_in": "activity"})
    g.add_node("tail", tail, metadata={"execute_in": "activity"})
    g.add_edge(START, "dispatch")
    g.add_edge("worker", END)
    g.add_edge("tail", END)

    task_queue = f"command-send-{uuid4()}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CommandSendWorkflow],
        plugins=[
            LangGraphPlugin(
                graphs={"send-graph": g},
                default_activity_options={
                    "start_to_close_timeout": timedelta(seconds=10),
                    "retry_policy": RetryPolicy(maximum_attempts=1),
                },
            )
        ],
    ):
        result = await client.execute_workflow(
            CommandSendWorkflow.run,
            mode,
            id=f"test-command-send-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=20),
        )

    assert sorted(result["results"]) == expected


def _round_trip(command: Command) -> Command:
    converter = DataConverter.default.payload_converter
    payloads = converter.to_payloads([_command_to_payload(command)])
    return _command_from_payload(converter.from_payloads(payloads)[0])


def test_command_payload_round_trips_send():
    timeout = TimeoutPolicy(
        run_timeout=1.5, idle_timeout=timedelta(seconds=2), refresh_on="heartbeat"
    )
    command = Command(
        update={"results": ["x"]},
        goto=["tail", Send("worker", {"item": "a"}, timeout=timeout)],
    )

    restored = _round_trip(command)

    assert restored == command
    assert isinstance(restored.goto, list)
    send = restored.goto[1]
    assert isinstance(send, Send)
    assert send.timeout == TimeoutPolicy(
        run_timeout=1.5, idle_timeout=2.0, refresh_on="heartbeat"
    )
    assert _round_trip(Command(goto=Send("worker", {"item": "a"}))) == Command(
        goto=Send("worker", {"item": "a"})
    )


@pytest.mark.parametrize(
    "command",
    [
        Command(update={"value": "a"}, goto="node_b"),
        Command(goto=["node_a", "node_b"]),
        Command(graph=Command.PARENT, update={"value": "a"}, goto=[]),
        Command(resume="yes", goto=[]),
    ],
)
def test_command_payload_reads_pre_send_format(command: Command):
    # Before Send support the Activity returned the Command dataclass itself.
    converter = DataConverter.default.payload_converter
    legacy = converter.to_payloads([ActivityOutput(langgraph_command=command)])
    output = converter.from_payloads(legacy, [ActivityOutput])[0]

    restored = _command_from_payload(output.langgraph_command)

    assert restored == command
    assert legacy == converter.to_payloads(
        [ActivityOutput(langgraph_command=_command_to_payload(command))]
    )
