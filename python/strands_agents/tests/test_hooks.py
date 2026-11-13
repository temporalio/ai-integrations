from collections.abc import AsyncGenerator
from datetime import timedelta
from uuid import uuid4

from strands import tool
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterInvocationEvent, AfterToolCallEvent

from temporalio import activity, workflow
from temporalio.api.enums.v1 import EventType
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
    def __init__(self) -> None:
        self.hook_finished = False
        self.agent = TemporalAgent(
            model="mock", start_to_close_timeout=timedelta(seconds=15)
        )
        self.agent.hooks.add_callback(AfterInvocationEvent, self._after_invocation)

    async def _after_invocation(self, event: AfterInvocationEvent) -> None:
        assert event.agent is self.agent
        await workflow.execute_activity(
            audit_tool,
            "after_invocation",
            start_to_close_timeout=timedelta(seconds=10),
        )
        self.hook_finished = True

    @workflow.run
    async def run(self, close_early: bool) -> bool:
        stream = self.agent.stream_async("Say hi")
        assert isinstance(stream, AsyncGenerator)
        async for _event in stream:
            if close_early:
                break
        await stream.aclose()
        return self.hook_finished


async def test_closed_stream_awaits_after_invocation_hooks(client: Client):
    """Closing the agent's stream early waits for its async AfterInvocationEvent hooks."""
    task_queue = f"test_closed_stream_awaits_after_invocation_hooks-{uuid4()}"
    plugin = StrandsPlugin(models={"mock": lambda: MockModel(["Hi!"])})

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ClosedStreamWorkflow],
        activities=[audit_tool],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        for close_early in (True, False):
            _AUDIT_LOG.clear()
            handle = await client.start_workflow(
                ClosedStreamWorkflow.run,
                close_early,
                id=f"test_closed_stream_{close_early}_{uuid4()}",
                task_queue=task_queue,
            )
            assert await handle.result()
            assert _AUDIT_LOG == ["after_invocation"]

            history = await handle.fetch_history()
            completed = [
                event
                for event in history.events
                if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED
            ]
            assert get_activities(history)[-1] == "audit_tool"
            assert len(completed) == len(get_activities(history))
            await Replayer(
                workflows=[ClosedStreamWorkflow], plugins=[plugin]
            ).replay_workflow(history)
