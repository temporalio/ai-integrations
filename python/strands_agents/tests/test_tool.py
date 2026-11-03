from collections.abc import Callable
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
from temporalio.client import Client
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
                activity_as_tool(
                    Shipping.quote, start_to_close_timeout=timedelta(seconds=15)
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
    ]
    spec = activity_as_tool(
        TrackParcel, start_to_close_timeout=timedelta(seconds=15)
    ).tool_spec
    assert list(spec["inputSchema"]["json"]["properties"]) == ["parcel_id"]

    await Replayer(
        workflows=[ToolWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


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
