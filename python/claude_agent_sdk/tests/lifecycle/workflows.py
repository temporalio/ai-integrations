"""Workflows that use the agent the ways the lifecycle tests check."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.claude_agent_sdk import AgentState, DurableClaudeAgent, activity_as_tool
from temporalio.common import RetryPolicy
from temporalio.exceptions import FailureError

with workflow.unsafe.imports_passed_through():
    from tests.endless.activities import count
    from tests.storage.workflows import fetch_document

FAST = RetryPolicy(initial_interval=timedelta(milliseconds=200), maximum_attempts=5)
COUNT = activity_as_tool(
    count, start_to_close_timeout=timedelta(seconds=30), retry_policy=FAST
)
FETCH = activity_as_tool(
    fetch_document, start_to_close_timeout=timedelta(seconds=60), retry_policy=FAST
)


@workflow.defn
class OneShotWorkflow:
    """The Quick start shape: ``run`` takes only the request, and nothing else is set."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(tools=[COUNT], max_segments=None)

    @workflow.run
    async def run(self, request: str) -> str:
        """Handle the request."""
        return await self.agent.run(request)


@workflow.defn
class UpdateChatWorkflow:
    """A chat over Updates: each ``ask`` returns Claude's answer to that message.

    Between messages, the run method continues as new when it is time.
    """

    @workflow.init
    def __init__(self, state: AgentState | None = None, auto: bool = False) -> None:
        self.lock = asyncio.Lock()
        self.agent = DurableClaudeAgent(
            tools=[COUNT],
            state=state,
            max_segments=None,
            auto_continue_as_new=auto,
            continue_as_new_after_events=40,
            continue_as_new_args=lambda s: [s, auto],
        )

    @workflow.run
    async def run(self, state: AgentState | None = None, auto: bool = False) -> None:
        """Serve messages; continue as new between them when the history is long."""
        del state, auto
        await workflow.wait_condition(
            lambda: self.agent.should_continue_as_new() and not self.lock.locked()
        )
        await self.agent.continue_as_new()

    @workflow.update
    async def ask(self, text: str) -> str:
        """One message, one answer."""
        async with self.lock:
            return await self.agent.run(text)

    @workflow.query
    def runs(self) -> int:
        """Workflow runs so far."""
        return self.agent.runs


@workflow.defn
class TasksWorkflow:
    """Runs prompts one after another; a failed task does not stop the next one."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(
            tools=[COUNT],
            max_segments=2,
            segment_retry_policy=RetryPolicy(maximum_attempts=2),
        )

    @workflow.run
    async def run(self, prompts: list[str]) -> list[str]:
        """Each prompt's answer, or ``error: <type>: ...``."""
        answers: list[str] = []
        for prompt in prompts:
            try:
                answers.append(await self.agent.run(prompt))
            except FailureError as err:  # ApplicationError or ActivityError
                answers.append(f"error: {type(err).__name__}: {err.message}")
        return answers


@workflow.defn
class SignalTaskWorkflow:
    """Wrong with auto Continue-As-New: runs the agent in a Signal handler."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(tools=[COUNT], auto_continue_as_new=True)
        self.answer: str | None = None

    @workflow.run
    async def run(self) -> str:
        """Wait for the answer the Signal handler produces."""
        await workflow.wait_condition(lambda: self.answer is not None)
        return self.answer or ""

    @workflow.signal
    async def ask(self, text: str) -> None:
        """Run a task in the handler."""
        self.answer = await self.agent.run(text)


@workflow.defn
class BigResultWorkflow:
    """Fetches large documents with live output on, continuing as new between them."""

    @workflow.init
    def __init__(self, prompt: str, state: AgentState | None = None) -> None:
        self.agent = DurableClaudeAgent(
            tools=[FETCH],
            state=state,
            max_segments=None,
            auto_continue_as_new=True,
            continue_as_new_after_events=12,
            live_output=True,
        )

    @workflow.run
    async def run(self, prompt: str, state: AgentState | None = None) -> str:
        """Run the task."""
        del state
        return await self.agent.run(prompt)

    @workflow.query
    def runs(self) -> int:
        """Workflow runs so far."""
        return self.agent.runs


@workflow.defn
class LateLiveOutputWorkflow:
    """Wrong: creates a live output agent after the Workflow was initialized."""

    @workflow.run
    async def run(self, request: str) -> str:
        """Fails its Workflow task: the stream's handlers must exist from the start."""
        agent = DurableClaudeAgent(tools=[COUNT], live_output=True)
        return await agent.run(request)
