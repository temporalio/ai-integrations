import functools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
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
from tests.postponed_activities import LoadParcels


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


@activity.defn(name="ship_parcel")
class ShipParcel:
    def __init__(self, parcel_id: str) -> None:
        self.parcel_id = parcel_id

    async def __call__(self, parcel_id: str, carrier: str = "road", /) -> str:
        return f"ship {parcel_id} by {carrier}"


@activity.defn(name="weigh_parcel")
class WeighParcel:
    @classmethod
    async def __call__(cls, parcel_id: str) -> str:
        return f"weigh {parcel_id} with {cls.__name__}"


@activity.defn(name="locate_parcel")
async def locate_parcel(parcel_id: str, /) -> str:
    return f"locate {parcel_id}"


@activity.defn(name="describe_parcel")
class DescribeParcel:
    """Track a parcel through the carrier network.

    Args:
        carrier: The carrier the activity is built with.
    """

    def __init__(self, carrier: str) -> None:
        self.carrier = carrier

    async def __call__(self, parcel_id: str) -> str:
        """
        Args:
            parcel_id: The parcel to track.
        """
        return parcel_id


class _Quoting:
    async def __call__(self, parcel_id: str) -> str:
        """Quote shipping for a parcel."""
        return parcel_id


@activity.defn(name="quote_dataclass")
@dataclass
class QuoteDataclass(_Quoting):
    carrier: str = "road"


@activity.defn(name="doc_property")
class DocProperty:
    __doc__ = property(lambda self: "per-instance doc")  # pyright: ignore[reportAssignmentType]

    async def __call__(self, parcel_id: str) -> str:
        return parcel_id


@activity.defn(name="static_parcel")
class StaticParcel:
    @staticmethod
    async def __call__(parcel_id: str) -> str:
        return parcel_id


async def _route(_self: Any, origin: str, destination: str, parcel_id: str) -> str:
    return f"{parcel_id} from {origin} to {destination}"


@activity.defn(name="route_parcel")
class RouteParcel:
    __call__ = functools.partialmethod(_route, "SFO")


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
                        ShipParcel,
                        WeighParcel,
                        locate_parcel,
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
                    {"name": "ship_parcel", "input": {"parcel_id": "P-3"}},
                    {
                        "name": "ship_parcel",
                        "input": {"parcel_id": "P-4", "carrier": "rail"},
                    },
                    {"name": "weigh_parcel", "input": {"parcel_id": "P-7"}},
                    {"name": "locate_parcel", "input": {"parcel_id": "P-8"}},
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
            ShipParcel("unused"),
            WeighParcel(),
            locate_parcel,
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
        "ship_parcel",
        "invoke_model",
        "ship_parcel",
        "invoke_model",
        "weigh_parcel",
        "invoke_model",
        "locate_parcel",
        "invoke_model",
    ]
    assert _activity_calls(client, history, "parcel") == [
        (["P-1"], "parcel P-1 via acme-post"),
        (["P-2"], "quote for P-2 via acme-post"),
        (["P-3", "road"], "ship P-3 by road"),
        (["P-4", "rail"], "ship P-4 by rail"),
        (["P-7"], "weigh P-7 with WeighParcel"),
        (["P-8"], "locate P-8"),
    ]

    cases: list[tuple[Callable[..., Any], list[str], list[str], str]] = [
        (TrackParcel, ["parcel_id"], ["parcel_id"], "TrackParcel"),
        (Shipping.quote, ["parcel_id"], ["parcel_id"], "quote"),
        (ShipParcel, ["parcel_id", "carrier"], ["parcel_id"], "ShipParcel"),
        (WeighParcel, ["parcel_id"], ["parcel_id"], "WeighParcel"),
        (locate_parcel, ["parcel_id"], ["parcel_id"], "locate_parcel"),
        (
            DescribeParcel,
            ["parcel_id"],
            ["parcel_id"],
            "Track a parcel through the carrier network.",
        ),
        (QuoteDataclass, ["parcel_id"], ["parcel_id"], "Quote shipping for a parcel."),
        (DocProperty, ["parcel_id"], ["parcel_id"], "DocProperty"),
        (StaticParcel, ["parcel_id"], ["parcel_id"], "StaticParcel"),
        (
            RouteParcel,
            ["destination", "parcel_id"],
            ["destination", "parcel_id"],
            "RouteParcel",
        ),
        (LoadParcels, ["parcels"], ["parcels"], "LoadParcels"),
    ]
    for activity_fn, properties, required, description in cases:
        spec = activity_as_tool(
            activity_fn, start_to_close_timeout=timedelta(seconds=15)
        ).tool_spec
        schema = spec["inputSchema"]["json"]
        assert list(schema["properties"]) == properties
        assert schema.get("required", []) == required
        assert spec["description"] == description
        if activity_fn is DescribeParcel:
            assert schema["properties"]["parcel_id"]["description"] == (
                "The parcel to track."
            )

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


@activity.defn
class NoCall:
    pass


@activity.defn
class KeywordPartialCall:
    __call__ = functools.partialmethod(_route, origin="SFO")


@pytest.mark.parametrize(
    "activity_fn, message",
    [
        (KeywordOnlyCall, r"cannot take \*args, keyword-only .* \('carrier'\)"),
        (VarArgsCall, r"cannot take \*args, keyword-only .* \('tags'\)"),
        (VarKeywordCall, r"cannot take \*args, keyword-only .* \('options'\)"),
        (TrackParcel("acme-post"), "Pass the activity class TrackParcel"),
        (NoCall, "needs a __call__"),
        (KeywordPartialCall, "binds an argument by keyword"),
    ],
)
def test_activity_as_tool_refuses(activity_fn: Callable[..., Any], message: str):
    with pytest.raises(ValueError, match=message):
        activity_as_tool(activity_fn, start_to_close_timeout=timedelta(seconds=15))
