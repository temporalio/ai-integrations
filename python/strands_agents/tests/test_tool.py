import asyncio
from datetime import timedelta
from pathlib import Path
from types import FrameType
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
from temporalio.strands_agents._temporal_agent import (
    _agent_loop_step,
    _called_by_sync_tool,
    _run_inline_when_called_by,
)
from temporalio.strands_agents.workflow import activity_as_tool
from temporalio.worker import Replayer, Worker
from tests.common import get_activities, get_tool_results
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
                    "Done!",
                ]
            )
        }
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ToolWorkflow],
        activities=[read_file_activity],
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
    ]
    # The model got the result of letter_counter, a sync tool run in the workflow.
    assert get_tool_results(history)[-1] == [{"text": "3"}]

    await Replayer(
        workflows=[ToolWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


@workflow.defn
class CrossTaskStreamWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=15),
            tools=[letter_counter],
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        stream = self.agent.stream_async(prompt)

        async def next_event() -> Any:
            return await anext(stream)

        result = None
        while True:
            try:
                # Take each event in a new task, as when multiplexing streams.
                event = await asyncio.create_task(next_event())
            except StopAsyncIteration:
                return str(result)
            result = event.get("result", result)


async def test_sync_tool_in_stream_across_tasks(client: Client):
    task_queue = f"test_sync_tool_in_stream_across_tasks-{uuid4()}"
    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [
                    {
                        "name": "letter_counter",
                        "input": {"word": "strawberry", "letter": "R"},
                    },
                    "Done!",
                ]
            )
        }
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CrossTaskStreamWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            CrossTaskStreamWorkflow.run,
            "How many R's are in strawberry?",
            id=f"test_sync_tool_in_stream_across_tasks_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == "Done!\n"

    history = await handle.fetch_history()
    # The sync tool ran in the workflow, though each step ran in another task.
    assert get_tool_results(history) == [[{"text": "3"}]]


def run_in_executor_from_other_framework(loop: asyncio.AbstractEventLoop) -> Any:
    return loop.run_in_executor(None, lambda: "other")


def called_by_other_framework(frame: FrameType) -> bool:
    return frame.f_code is run_in_executor_from_other_framework.__code__


# Strands' own sync tool call: asyncio.to_thread from strands.tools.decorator.
_strands_tool_module: dict[str, Any] = {"__name__": "strands.tools.decorator"}
exec(
    "import asyncio\n"
    "async def run_sync_tool(func):\n"
    "    return await asyncio.to_thread(func)\n",
    _strands_tool_module,
)


@pytest.mark.parametrize("strands_first", [True, False])
def test_sync_tool_dispatch_composes_with_other_framework(strands_first: bool):
    # Another plugin, such as Deep Agents, adds its caller to the same loop.
    loop = asyncio.new_event_loop()
    try:

        def no_threads(*_args: Any) -> Any:
            raise NotImplementedError

        setattr(loop, "run_in_executor", no_threads)
        callers = [_called_by_sync_tool, called_by_other_framework]
        for called_by in callers if strands_first else reversed(callers):
            _run_inline_when_called_by(loop, called_by)

        run_sync_tool = _strands_tool_module["run_sync_tool"]
        with _agent_loop_step():
            assert loop.run_until_complete(run_sync_tool(lambda: 42)) == 42
        other = run_in_executor_from_other_framework(loop)
        assert loop.run_until_complete(other) == "other"
        # Strands' call outside an agent loop step, and any other caller, fail.
        with pytest.raises(NotImplementedError):
            loop.run_until_complete(run_sync_tool(lambda: 42))
        with pytest.raises(NotImplementedError):
            loop.run_in_executor(None, lambda: "unrelated")
    finally:
        loop.close()
