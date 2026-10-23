"""The segment benchmark Workflows: as in the hybrid prototype, and with Claude Code
tools as tool steps."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool

with workflow.unsafe.imports_passed_through():
    from tests.bench.workload import SCHEMA, segment_echo


@workflow.defn
class SegmentBenchmarkWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(
                    segment_echo,
                    name="echo",
                    input_schema=SCHEMA,
                    start_to_close_timeout=timedelta(seconds=30),
                )
            ],
            max_segments=None,
        )
        return await agent.run(prompt)


@workflow.defn
class BuiltinBenchmarkWorkflow:
    """Rounds of a Claude Code tool that runs as its own Activity (a tool step).

    The durable echo tool is offered as in ``SegmentBenchmarkWorkflow``, never called.
    """

    @workflow.run
    async def run(self, prompt: str, tools: str) -> str:
        agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(
                    segment_echo,
                    name="echo",
                    input_schema=SCHEMA,
                    start_to_close_timeout=timedelta(seconds=30),
                )
            ],
            # "Bash", "Write", or "Read,Edit": Read runs inside the segments.
            builtin_tools=tools.split(","),
            tool_activities=[t for t in tools.split(",") if t != "Read"],
            max_segments=None,
        )
        return await agent.run(prompt)
