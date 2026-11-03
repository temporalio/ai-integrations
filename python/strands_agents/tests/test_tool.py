import gc
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, List  # pyright: ignore[reportDeprecated]
from uuid import uuid4

import pytest
from strands import tool
from strands_tools import (  # pyright: ignore[reportMissingTypeStubs]
    calculator,
    file_read,
)

from temporalio import activity, workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowHistory
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents.workflow import activity_as_tool
from temporalio.worker import Replayer, Worker
from tests.common import get_activities
from tests.mock_model import MockModel


@tool
def letter_counter(word: str, letter: str) -> int:
    return word.lower().count(letter.lower())


@activity.defn(name="read_file")
async def read_file_activity(path: str) -> str:
    result = file_read.file_read(
        {
            "toolUseId": "read_file",
            "name": "file_read",
            "input": {"path": path, "mode": "view"},
        }
    )
    text = result["content"][0].get("text")
    assert text is not None
    return text


@activity.defn(name="track_parcel")
class TrackParcel:
    def __init__(self, carrier: str) -> None:
        self.carrier = carrier

    async def __call__(self, parcel_id: str) -> str:
        return f"parcel {parcel_id} via {self.carrier}"


class Shipping:
    def __init__(self, carrier: str) -> None:
        self.carrier = carrier

    @activity.defn(name="quote_parcel")
    async def quote(self, parcel_id: str) -> str:
        return f"quote for {parcel_id} via {self.carrier}"

    @staticmethod
    @activity.defn(name="price_parcel")
    async def price(
        self: str = "default-carrier",  # pyright: ignore[reportSelfClsParameterName]
        parcel_id: str = "default-parcel",
    ) -> str:
        return f"{self}:{parcel_id}"


@activity.defn(name="ship_parcel")
class ShipParcel:
    def __init__(self, parcel_id: str) -> None:
        self.parcel_id = parcel_id

    async def __call__(self, parcel_id: str, carrier: str = "road", /) -> str:
        return f"ship {parcel_id} by {carrier}"


@activity.defn(name="hand_over_parcel")
class HandOverParcel:
    async def __call__(self, agent: str, cls: str) -> str:
        return f"hand {cls} to {agent}"


@activity.defn(name="label_parcel")
class LabelParcel:
    @staticmethod
    async def __call__(self: str) -> str:  # pyright: ignore[reportSelfClsParameterName]
        return f"label {self}"


@activity.defn(name="weigh_parcel")
class WeighParcel:
    @classmethod
    async def __call__(cls, parcel_id: str) -> str:
        return f"weigh {parcel_id} with {cls.__name__}"


@activity.defn(name="stamp_parcel")
class StampParcel:
    def __init__(self, parcel_id: str) -> None:
        self.parcel_id = parcel_id

    async def __call__(self, _parcel_id: str) -> str:
        return f"stamp {_parcel_id}"


@dataclass
class Parcel:
    parcel_id: str


@activity.defn(name="load_parcel")
class LoadParcel:
    def __init__(self, parcels: list[Parcel]) -> None:
        self.parcels = parcels

    # A forward reference nested in an annotation resolves in this module.
    # typing.List, unlike list, makes it a ForwardRef that the worker's
    # converter resolves on Python 3.10 too.
    async def __call__(
        self,
        parcels: List["Parcel"],  # pyright: ignore[reportDeprecated]
    ) -> str:
        return "load " + ", ".join(p.parcel_id for p in parcels)


@activity.defn(name="sort_parcel")
class SortParcel:
    def __init__(self, parcels: list[Parcel]) -> None:
        self.parcels = parcels

    async def __call__(self, parcels: list["Parcel"]) -> str:
        return ", ".join(p.parcel_id for p in parcels)


@workflow.defn
class ToolWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=15),
            tools=[
                calculator,
                activity_as_tool(
                    read_file_activity,
                    start_to_close_timeout=timedelta(seconds=15),
                ),
                letter_counter,
                activity_as_tool(
                    TrackParcel, start_to_close_timeout=timedelta(seconds=15)
                ),
                *(
                    activity_as_tool(fn, start_to_close_timeout=timedelta(seconds=15))
                    for fn in (
                        Shipping.quote,
                        Shipping.price,
                        ShipParcel,
                        HandOverParcel,
                        LabelParcel,
                        WeighParcel,
                        StampParcel,
                        LoadParcel,
                    )
                ),
            ],
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await self.agent.invoke_async(prompt)
        return str(result)


async def test_tool(client: Client, tmp_path: Path):
    task_queue = f"test_tool-{uuid4()}"
    fixture = tmp_path / "greeting.txt"
    fixture.write_text("hello\n")

    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [
                    {"name": "read_file", "input": {"path": str(fixture)}},
                    {
                        "name": "calculator",
                        "input": {"expression": "3111696 / 74088"},
                    },
                    {
                        "name": "letter_counter",
                        "input": {"word": "strawberry", "letter": "R"},
                    },
                    {"name": "track_parcel", "input": {"parcel_id": "P-1"}},
                    {"name": "quote_parcel", "input": {"parcel_id": "P-2"}},
                    {"name": "price_parcel", "input": {"parcel_id": "P-8"}},
                    {"name": "ship_parcel", "input": {"parcel_id": "P-3"}},
                    {
                        "name": "ship_parcel",
                        "input": {"parcel_id": "P-4", "carrier": "rail"},
                    },
                    {
                        "name": "hand_over_parcel",
                        "input": {"agent": "Ana", "cls": "P-5"},
                    },
                    {"name": "label_parcel", "input": {"self": "P-6"}},
                    {"name": "weigh_parcel", "input": {"parcel_id": "P-7"}},
                    {"name": "stamp_parcel", "input": {"_parcel_id": "P-9"}},
                    {
                        "name": "load_parcel",
                        "input": {"parcels": [{"parcel_id": "P-10"}]},
                    },
                    "Done!",
                ]
            )
        }
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ToolWorkflow],
        activities=[
            read_file_activity,
            TrackParcel("acme-post"),
            Shipping("acme-post").quote,
            Shipping.price,
            ShipParcel("unused"),
            HandOverParcel(),
            LabelParcel(),
            WeighParcel(),
            StampParcel("unused"),
            LoadParcel([]),
        ],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            ToolWorkflow.run,
            "I have 3 requests:\n"
            f"1. Read the file at {fixture}\n"
            "2. Calculate 3111696 / 74088\n"
            '3. Tell me how many letter R\'s are in the word "strawberry"',
            id=f"test_tool_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == "Done!\n"

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "strands_get_model_config",
        "invoke_model",
        "read_file",
        "invoke_model",
        # calculator (in-workflow)
        "invoke_model",
        # letter_counter (in-workflow)
        "invoke_model",
        "track_parcel",
        "invoke_model",
        "quote_parcel",
        "invoke_model",
        "price_parcel",
        "invoke_model",
        "ship_parcel",
        "invoke_model",
        "ship_parcel",
        "invoke_model",
        "hand_over_parcel",
        "invoke_model",
        "label_parcel",
        "invoke_model",
        "weigh_parcel",
        "invoke_model",
        "stamp_parcel",
        "invoke_model",
        "load_parcel",
        "invoke_model",
    ]
    assert _activity_calls(client, history, "parcel") == [
        (["P-1"], "parcel P-1 via acme-post"),
        (["P-2"], "quote for P-2 via acme-post"),
        (["default-carrier", "P-8"], "default-carrier:P-8"),
        (["P-3", "road"], "ship P-3 by road"),
        (["P-4", "rail"], "ship P-4 by rail"),
        (["Ana", "P-5"], "hand P-5 to Ana"),
        (["P-6"], "label P-6"),
        (["P-7"], "weigh P-7 with WeighParcel"),
        (["P-9"], "stamp P-9"),
        ([[{"parcel_id": "P-10"}]], "load P-10"),
    ]

    cases: list[tuple[Callable[..., Any], list[str], list[str]]] = [
        (TrackParcel, ["parcel_id"], ["parcel_id"]),
        (Shipping.quote, ["parcel_id"], ["parcel_id"]),
        (Shipping.price, ["self", "parcel_id"], []),
        (ShipParcel, ["parcel_id", "carrier"], ["parcel_id"]),
        (HandOverParcel, ["agent", "cls"], ["agent", "cls"]),
        (LabelParcel, ["self"], ["self"]),
        (WeighParcel, ["parcel_id"], ["parcel_id"]),
        (StampParcel, ["_parcel_id"], ["_parcel_id"]),
        (LoadParcel, ["parcels"], ["parcels"]),
        (SortParcel, ["parcels"], ["parcels"]),
    ]
    for activity_fn, properties, required in cases:
        schema = activity_as_tool(
            activity_fn, start_to_close_timeout=timedelta(seconds=15)
        ).tool_spec["inputSchema"]["json"]
        assert list(schema["properties"]) == properties
        assert schema.get("required", []) == required
        if activity_fn in (LoadParcel, SortParcel):
            assert schema["$defs"]["Parcel"]["properties"] == {
                "parcel_id": {"title": "Parcel Id", "type": "string"}
            }

    await Replayer(
        workflows=[ToolWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


def _activity_calls(
    client: Client, history: WorkflowHistory, suffix: str
) -> list[tuple[list[Any], Any]]:
    """Return the inputs and result of each activity whose name ends with ``suffix``."""
    converter = client.data_converter.payload_converter
    inputs: dict[int, list[Any]] = {}
    calls: list[tuple[list[Any], Any]] = []
    for event in history.events:
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            scheduled = event.activity_task_scheduled_event_attributes
            if scheduled.activity_type.name.endswith(suffix):
                inputs[event.event_id] = converter.from_payloads(
                    scheduled.input.payloads
                )
        elif event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED:
            completed = event.activity_task_completed_event_attributes
            if completed.scheduled_event_id in inputs:
                [result] = converter.from_payloads(completed.result.payloads)
                calls.append((inputs[completed.scheduled_event_id], result))
    return calls


@activity.defn
class KeywordOnlyCall:
    async def __call__(self, parcel_id: str, *, carrier: str = "road") -> str:
        return parcel_id


@activity.defn
class VarArgsCall:
    async def __call__(self, parcel_id: str, *tags: str) -> str:
        return parcel_id


@activity.defn
class VarKeywordCall:
    async def __call__(self, parcel_id: str, **options: str) -> str:
        return parcel_id


@pytest.mark.parametrize(
    "activity_fn, message",
    [
        (KeywordOnlyCall, r"cannot take \*args, keyword-only .* \('carrier'\)"),
        (VarArgsCall, r"cannot take \*args, keyword-only .* \('tags'\)"),
        (VarKeywordCall, r"cannot take \*args, keyword-only .* \('options'\)"),
        (TrackParcel("acme-post"), "Pass the activity class TrackParcel"),
    ],
)
def test_activity_as_tool_refuses(activity_fn: Callable[..., Any], message: str):
    with pytest.raises(ValueError, match=message):
        activity_as_tool(activity_fn, start_to_close_timeout=timedelta(seconds=15))


def _local_shipping() -> tuple[weakref.ref[type], list[Callable[..., Any]]]:
    # A class a function defines can't be reached by name.
    class LocalShipping:
        @activity.defn(name="local_quote")
        async def quote(self, parcel_id: str) -> str:
            return parcel_id

        @staticmethod
        @activity.defn(name="local_price")
        async def price(
            self: str = "carrier",  # pyright: ignore[reportSelfClsParameterName]
            parcel_id: str = "parcel",
        ) -> str:
            return f"{self}:{parcel_id}"

    return weakref.ref(LocalShipping), [LocalShipping.quote, LocalShipping.price]


def test_activity_as_tool_refuses_method_of_unreachable_class():
    # Only the class tells a method from a staticmethod: the refusal doesn't
    # depend on whether the class is still alive.
    owner, activity_fns = _local_shipping()
    for collected in (False, True):
        if collected:
            gc.collect()
            assert owner() is None
        for activity_fn in activity_fns:
            with pytest.raises(ValueError, match="can't be reached by name"):
                activity_as_tool(
                    activity_fn, start_to_close_timeout=timedelta(seconds=15)
                )
