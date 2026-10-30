import asyncio
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import Any, cast
from uuid import uuid4

from strands import tool
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterInvocationEvent, AfterToolCallEvent

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents.workflow import activity_as_hook
from temporalio.worker import Replayer, Worker
from tests.common import get_activities
from tests.mock_model import MockModel

# Module-level sink: written by the audit activity, read in assertions.
# Activity bodies run in worker context, not the sandbox, so a plain list is fine.
_AUDIT_LOG: list[str] = []


@activity.defn
async def audit_tool(tool_name: str) -> None:
    _AUDIT_LOG.append(tool_name)


@tool
def echo(text: str) -> str:
    return text


class AuditHook(HookProvider):
    def __init__(self) -> None:
        self.fired_events: list[str] = []

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(AfterToolCallEvent, self._sync_log)
        registry.add_callback(
            AfterToolCallEvent,
            activity_as_hook(
                audit_tool,
                activity_input=lambda event: event.tool_use["name"],
                start_to_close_timeout=timedelta(seconds=10),
            ),
        )

    def _sync_log(self, event: AfterToolCallEvent) -> None:
        # Deterministic in-workflow mutation: appends to per-workflow state.
        self.fired_events.append(event.tool_use["name"])


@workflow.defn
class HooksWorkflow:
    def __init__(self) -> None:
        self.hook = AuditHook()
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=15),
            tools=[echo],
            hooks=[self.hook],
        )

    @workflow.run
    async def run(self, prompt: str) -> list[str]:
        await self.agent.invoke_async(prompt)
        return self.hook.fired_events


async def test_hooks(client: Client):
    _AUDIT_LOG.clear()
    task_queue = f"test_hooks-{uuid4()}"
    plugin = StrandsPlugin(
        models={
            "mock": lambda: MockModel(
                [
                    {"name": "echo", "input": {"text": "hi"}},
                    "Done!",
                ]
            )
        }
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[HooksWorkflow],
        activities=[audit_tool],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            HooksWorkflow.run,
            "Say hi",
            id=f"test_hooks_{uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == ["echo"]

    assert _AUDIT_LOG == ["echo"]

    history = await handle.fetch_history()
    assert "audit_tool" in get_activities(history)

    await Replayer(
        workflows=[HooksWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)


@workflow.defn
class ClosedStreamWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        log: list[str] = []

        async def after_invocation(
            event: AfterInvocationEvent,  # pyright: ignore[reportUnusedParameter]
        ) -> None:
            log.append("start")
            await asyncio.sleep(0)
            log.append("end")

        agent = TemporalAgent(
            model="mock", start_to_close_timeout=timedelta(seconds=15)
        )
        agent.hooks.add_callback(AfterInvocationEvent, after_invocation)
        stream = cast(AsyncGenerator[Any, None], agent.stream_async("hi"))
        async for _event in stream:
            break
        await stream.aclose()
        return log


async def test_after_invocation_hook_finishes_when_stream_closed_early(client: Client):
    """Closing the stream early still lets an AfterInvocationEvent hook await."""
    task_queue = f"test_after_invocation_hook_finishes-{uuid4()}"
    plugin = StrandsPlugin(models={"mock": lambda: MockModel(["Done!"])})

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClosedStreamWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        assert await client.execute_workflow(
            ClosedStreamWorkflow.run,
            id=f"test_after_invocation_hook_finishes_{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        ) == ["start", "end"]
