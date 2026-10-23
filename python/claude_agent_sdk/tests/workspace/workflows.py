"""A Workflow that runs its prompts as tasks of one agent, with Bash as a tool step.

Between tasks it can continue as new (``hand_over``), and a task that fails does not
stop the next one. It returns the agent's state, so a test sees the agent's folder
and its snapshot, and the tasks that failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio import workflow
from temporalio.claude_agent_sdk import AgentState, DurableClaudeAgent
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

FAST = RetryPolicy(initial_interval=timedelta(milliseconds=200), maximum_attempts=3)


@dataclass
class Tasks:
    """What the Workflow runs.

    Attributes:
        prompts: One task per prompt, in order.
        hand_over: Continue as new after each task but the last.
        failed: Tasks that failed so far (across runs).
    """

    prompts: list[str]
    hand_over: bool = False
    failed: list[str] = field(default_factory=list)


@dataclass
class Done:
    """What the Workflow returns.

    Attributes:
        state: The agent's state at the end.
        failed: The tasks that failed, with their error.
    """

    state: AgentState
    failed: list[str]


@workflow.defn
class FolderWorkflow:
    """One agent, its tasks in order."""

    @workflow.init
    def __init__(self, tasks: Tasks, state: AgentState | None = None) -> None:
        self.tasks = tasks
        self.agent = DurableClaudeAgent(
            builtin_tools=["Bash"],
            tool_activities=["Bash"],
            segment_retry_policy=FAST,
            tool_activity_retry_policy=FAST,
            segment_heartbeat_timeout=timedelta(seconds=10),
            state=state,
            continue_as_new_args=lambda s: [self.tasks, s],
        )

    @workflow.run
    async def run(self, tasks: Tasks, state: AgentState | None = None) -> Done:
        """Run the tasks; return the agent's state and the tasks that failed."""
        del tasks, state  # the same as ``__init__`` got
        while self.tasks.prompts:
            prompt = self.tasks.prompts.pop(0)
            try:
                await self.agent.run(prompt)
            except ApplicationError as err:
                self.tasks.failed.append(f"{prompt}: {err.message}")
            if self.tasks.hand_over and self.tasks.prompts:
                await self.agent.continue_as_new()
        return Done(self.agent.state(), list(self.tasks.failed))
