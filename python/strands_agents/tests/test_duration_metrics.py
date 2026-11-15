"""Duration metrics must survive cache eviction and explicit history replay."""

import asyncio
import time
from collections.abc import AsyncIterable
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from strands import Agent, tool
from strands.event_loop.streaming import stream_messages
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
from strands.telemetry.metrics import Trace
from strands.types.streaming import StreamEvent

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.strands_agents import StrandsPlugin, TemporalAgent
from temporalio.strands_agents.workflow import activity_as_tool
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner
from tests.common import get_activities
from tests.mock_model import MockModel


class SlowModel(MockModel):
    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[StreamEvent]:
        await asyncio.sleep(0.2)
        async for event in super().stream(*args, **kwargs):
            yield event
        yield {
            "metadata": {
                "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
                "metrics": {"latencyMs": 600},
            }
        }


@activity.defn
async def slow_metrics_tool() -> str:
    await asyncio.sleep(0.2)
    return "done"


@tool
async def native_metrics_tool() -> str:
    return await slow_metrics_tool()


@activity.defn
async def record_duration(value: dict[str, Any]) -> dict[str, Any]:
    return value


@activity.defn
async def wrong_duration(value: dict[str, Any]) -> dict[str, Any]:
    return value


class ToolDurationHook(HookProvider):
    def __init__(self) -> None:
        self.durations: list[float | None] = []

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterToolCallEvent, self.after_tool)

    def after_tool(self, event: AfterToolCallEvent) -> None:
        self.durations.append(event.duration)


def _timing(agent: Agent, hook: ToolDurationHook) -> dict[str, Any]:
    metrics = agent.event_loop_metrics
    summary = metrics.get_summary()
    return {
        "cycle_durations": metrics.cycle_durations,
        "total_duration": summary["total_duration"],
        "tool_duration": metrics.tool_metrics["slow_metrics_tool"].total_time,
        "hook_durations": hook.durations,
        "trace_times": [[trace.start_time, trace.end_time] for trace in metrics.traces],
        "tool_trace_times": [
            [child.start_time, child.end_time]
            for trace in metrics.traces
            for child in trace.children
        ],
        "latency_ms": metrics.accumulated_metrics["latencyMs"],
        "total_tokens": metrics.accumulated_usage["totalTokens"],
    }


@workflow.defn
class DurationMetricsWorkflow:
    @workflow.run
    async def run(self) -> dict[str, Any]:
        hook = ToolDurationHook()
        agent = TemporalAgent(
            model="slow",
            tools=[
                activity_as_tool(
                    slow_metrics_tool, start_to_close_timeout=timedelta(seconds=10)
                )
            ],
            hooks=[hook],
            callback_handler=None,
            start_to_close_timeout=timedelta(seconds=10),
        )
        start = workflow.time()
        trace = Trace("invocation")
        await agent.invoke_async("call the tool")
        trace.end()
        assert trace.start_time == start
        assert trace.end_time == workflow.time()

        snapshot = _timing(agent, hook)
        # Branching makes the clock observable to the Replayer's command checks.
        # The echoed snapshot also checks every timestamp, not just this branch.
        recorded = await workflow.execute_activity(
            record_duration if snapshot["total_duration"] > 0.4 else wrong_duration,
            snapshot,
            start_to_close_timeout=timedelta(seconds=10),
        )
        await workflow.sleep(0.01)
        assert _timing(agent, hook) == recorded
        return recorded


@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("sandboxed", [True, False])
async def test_durations_survive_replay(
    client: Client, cached: bool, sandboxed: bool
) -> None:
    plugin = StrandsPlugin(
        models={
            "slow": lambda: SlowModel(
                [{"name": "slow_metrics_tool", "input": {}}, "done"]
            )
        }
    )
    runner = SandboxedWorkflowRunner() if sandboxed else UnsandboxedWorkflowRunner()
    task_queue = f"duration-metrics-{uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[DurationMetricsWorkflow],
        activities=[slow_metrics_tool, record_duration, wrong_duration],
        plugins=[plugin],
        workflow_runner=runner,
        workflow_failure_exception_types=[AssertionError],
        max_cached_workflows=100 if cached else 0,
    ):
        handle = await client.start_workflow(
            DurationMetricsWorkflow.run,
            id=f"duration-metrics-{uuid4()}",
            task_queue=task_queue,
        )
        result = await handle.result()

    assert result["total_duration"] > 0.4
    assert result["tool_duration"] >= 0.18
    assert result["hook_durations"] == [result["tool_duration"]]
    assert result["latency_ms"] == 1200
    assert result["total_tokens"] == 14
    assert len(result["cycle_durations"]) == 2
    history = await handle.fetch_history()
    assert get_activities(history) == [
        "strands_get_model_config",
        "invoke_model",
        "slow_metrics_tool",
        "invoke_model",
        "record_duration",
    ]
    # A fresh plugin installs the same clock for replay, including when the
    # runner is unsandboxed. Activity bodies and their results are not rerun.
    await Replayer(
        workflows=[DurationMetricsWorkflow],
        plugins=[StrandsPlugin(models={})],
        workflow_runner=runner,
        workflow_failure_exception_types=[AssertionError],
    ).replay_workflow(history)


@workflow.defn
class FirstByteMetricsWorkflow:
    @workflow.run
    async def run(self) -> dict[str, Any]:
        start = workflow.time()
        metrics: dict[str, Any] = {}
        # A local model that yields after a Workflow timer exercises Strands'
        # inferred time-to-first-byte path without a provider or an exporter.
        async for event in stream_messages(SlowModel(["done"]), None, [], []):
            if "stop" in event:
                metrics = event["stop"][3]
        assert metrics["timeToFirstByteMs"] == int(1000 * (workflow.time() - start))
        assert metrics["timeToFirstByteMs"] >= 180
        assert metrics["latencyMs"] == 600
        return metrics


async def test_inferred_first_byte_time_survives_replay(client: Client) -> None:
    task_queue = f"first-byte-metrics-{uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[FirstByteMetricsWorkflow],
        plugins=[StrandsPlugin(models={})],
        max_cached_workflows=0,
        workflow_failure_exception_types=[AssertionError],
    ):
        handle = await client.start_workflow(
            FirstByteMetricsWorkflow.run,
            id=f"first-byte-metrics-{uuid4()}",
            task_queue=task_queue,
        )
        assert (await handle.result())["timeToFirstByteMs"] >= 180
    await Replayer(
        workflows=[FirstByteMetricsWorkflow],
        plugins=[StrandsPlugin(models={})],
        workflow_failure_exception_types=[AssertionError],
    ).replay_workflow(await handle.fetch_history())


async def test_native_strands_keeps_wall_clock_durations() -> None:
    # Installing a second plugin must be harmless, and must not change the
    # stdlib clocks used by Activities and native Strands in the same process.
    original_time, original_monotonic = time.time, time.monotonic
    plugin = StrandsPlugin(models={})
    plugin.configure_replayer({"workflow_runner": SandboxedWorkflowRunner()})
    plugin.configure_replayer({"workflow_runner": SandboxedWorkflowRunner()})
    assert time.time is original_time
    assert time.monotonic is original_monotonic
    start = time.time()
    hook = ToolDurationHook()
    result = await Agent(
        model=SlowModel([{"name": "native_metrics_tool", "input": {}}, "done"]),
        tools=[native_metrics_tool],
        hooks=[hook],
        callback_handler=None,
    ).invoke_async("hello")
    end = time.time()
    assert result.metrics.cycle_durations[0] >= 0.18
    assert result.metrics.tool_metrics["native_metrics_tool"].total_time >= 0.18
    assert hook.durations[0] is not None and hook.durations[0] >= 0.18
    assert start <= result.metrics.traces[0].start_time <= end
    assert result.metrics.traces[0].end_time is not None
    assert start <= result.metrics.traces[0].end_time <= end
