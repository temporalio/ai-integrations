"""Long-running agents: one long task, and a chat that keeps one Claude session."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.claude_agent_sdk import AgentState, DurableClaudeAgent, activity_as_tool
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tests.endless.activities import count, publish

FAST = RetryPolicy(initial_interval=timedelta(milliseconds=200), maximum_attempts=5)


def _agent(state: AgentState | None, **options: Any) -> DurableClaudeAgent:
    tool: dict[str, Any] = {
        "start_to_close_timeout": timedelta(seconds=30),
        "retry_policy": FAST,
    }
    settings: dict[str, Any] = {
        "segment_timeout": timedelta(minutes=2),
        "segment_heartbeat_timeout": timedelta(seconds=30),
        "segment_retry_policy": FAST,
        "max_segments": None,
        **options,
    }
    return DurableClaudeAgent(
        tools=[
            activity_as_tool(count, **tool),
            activity_as_tool(publish, needs_approval=True, **tool),
        ],
        state=state,
        **settings,
    )


@dataclass
class TaskOptions:
    """Test knobs for LongTaskWorkflow.

    Attributes:
        continue_as_new: Whether the agent continues as new automatically.
        after_events: Fixed history length for Continue-As-New (None: server suggestion).
        live: Publish live output through Workflow Streams.
        segment_timeout: Seconds each model segment attempt may take (None: 2 minutes).
    """

    continue_as_new: bool = True
    after_events: int | None = None
    live: bool = False
    segment_timeout: float | None = None


@workflow.defn
class LongTaskWorkflow:
    """One long task: many tool calls in a single agent run."""

    @workflow.init
    def __init__(
        self, prompt: str, options: TaskOptions, state: AgentState | None = None
    ) -> None:
        extra: dict[str, Any] = {}
        if options.segment_timeout is not None:
            extra["segment_timeout"] = timedelta(seconds=options.segment_timeout)
        self.agent = _agent(
            state,
            auto_continue_as_new=options.continue_as_new,
            continue_as_new_after_events=options.after_events,
            continue_as_new_args=lambda s: [s.task_prompt, options, s],
            live_output=options.live,
            **extra,
        )

    @workflow.run
    async def run(
        self, prompt: str, options: TaskOptions, state: AgentState | None = None
    ) -> str:
        """Run the task."""
        del options, state
        return await self.agent.run(prompt)

    @workflow.update
    def review(self, tool_use_id: str, approved: bool) -> None:
        """Approve or reject a call."""
        self.agent.decide(tool_use_id, approved, "reviewer")

    @review.validator
    def check_review(self, tool_use_id: str, approved: bool) -> None:
        """Refuse decisions on calls that are not waiting."""
        del approved
        self.agent.validate_decision(tool_use_id)

    @workflow.signal
    def review_by_signal(self, tool_use_id: str, approved: bool) -> None:
        """Approve or reject a call without an Update (invalid decisions are ignored)."""
        self.agent.decide(tool_use_id, approved, "reviewer")

    @workflow.query
    def pending_approvals(self) -> list[dict[str, Any]]:
        """Calls waiting for a decision."""
        return self.agent.pending_approvals()

    @workflow.query
    def progress(self) -> dict[str, int]:
        """Totals across runs."""
        return {"runs": self.agent.runs, "tool_calls": self.agent.total_tool_calls}


@dataclass
class ChatState:
    """What a new chat run needs.

    Attributes:
        inbox: Messages not answered yet.
        replies: Answers so far.
        agent: The agent's state.
    """

    inbox: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    agent: AgentState | None = None


@workflow.defn
class ChatWorkflow:
    """A chat: every message is a new task on the same Claude session."""

    @workflow.init
    def __init__(self, stop_after: int, state: ChatState | None = None) -> None:
        self.inbox: list[str] = list(state.inbox) if state else []
        self.replies: list[str] = list(state.replies) if state else []
        self.agent = _agent(
            state.agent if state else None,
            auto_continue_as_new=True,
            continue_as_new_after_events=40,
            continue_as_new_args=lambda s: [
                stop_after,
                ChatState(inbox=list(self.inbox), replies=list(self.replies), agent=s),
            ],
        )

    @workflow.run
    async def run(self, stop_after: int, state: ChatState | None = None) -> list[str]:
        """Answer messages until ``stop_after`` replies were given."""
        del state
        if self.agent.busy:  # a task interrupted by Continue-As-New
            self.replies.append(await self.agent.run())
        while len(self.replies) < stop_after:
            await workflow.wait_condition(lambda: bool(self.inbox))
            if self.agent.should_continue_as_new():  # between messages
                await self.agent.continue_as_new()
            self.replies.append(await self.agent.run(self.inbox.pop(0)))
        return self.replies

    @workflow.signal
    def send(self, text: str) -> None:
        """A new message from the user."""
        self.inbox.append(text)

    @workflow.query
    def progress(self) -> dict[str, int]:
        """Totals across runs."""
        return {"runs": self.agent.runs, "tool_calls": self.agent.total_tool_calls}


@workflow.defn
class AutoChatWorkflow:
    """A chat that leaves Continue-As-New to the agent, which checks before every step,
    so it can hand over at the start of a message, before Claude sees it."""

    @workflow.init
    def __init__(self, stop_after: int, state: ChatState | None = None) -> None:
        self.inbox: list[str] = list(state.inbox) if state else []
        self.replies: list[str] = list(state.replies) if state else []
        self.agent = _agent(
            state.agent if state else None,
            auto_continue_as_new=True,
            continue_as_new_after_events=30,
            continue_as_new_args=lambda s: [
                stop_after,
                ChatState(inbox=list(self.inbox), replies=list(self.replies), agent=s),
            ],
        )

    @workflow.run
    async def run(self, stop_after: int, state: ChatState | None = None) -> list[str]:
        """Answer messages until ``stop_after`` replies were given."""
        del state
        if self.agent.busy:  # a message handed over by Continue-As-New
            self.replies.append(await self.agent.run())
        while len(self.replies) < stop_after:
            await workflow.wait_condition(lambda: bool(self.inbox))
            self.replies.append(await self.agent.run(self.inbox.pop(0)))
        return self.replies

    @workflow.signal
    def send(self, text: str) -> None:
        """A new message from the user."""
        self.inbox.append(text)

    @workflow.query
    def progress(self) -> dict[str, int]:
        """Totals across runs."""
        return {"runs": self.agent.runs, "tool_calls": self.agent.total_tool_calls}
