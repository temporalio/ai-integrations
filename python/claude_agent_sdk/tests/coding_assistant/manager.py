"""The manager: one long-lived workflow that hands each message from planner to coder to reviewer.

Claude Code's own subagents are disabled in this plugin because they are not durable, so the
handoff is the workflow's: each role is one Claude turn, and the workflow passes one role's
typed output into the next role's prompt. A worker lost mid-message resumes at the turn it was
in; finished roles are not run again.

* Each user message is an :meth:`CodingAssistant.ask` update. It returns the :class:`Review`.
* The planner and reviewer run in one-off sessions; the coder keeps one :class:`ClaudeSession`
  across messages, and its text streams to the ``assistant`` Workflow Stream topic.
* Each role's turn may spend ``max_budget_usd``. The coder's session stops at ``max_cost_usd``,
  and once it has, :meth:`CodingAssistant.ask` refuses new messages before any role runs.
* After ``messages_per_run`` messages the workflow continues as new, carrying the coder's
  session. The update validator refuses messages past that share, so a message never lands in
  a run that is about to continue as new; the client sends it again to reach the next run.
"""

from __future__ import annotations

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel

    from temporalio.claude_agent_sdk import (
        ClaudeDeferredTool,
        ClaudeSession,
        ClaudeSessionState,
        ClaudeTurnOptions,
    )
    from temporalio.contrib.workflow_streams import WorkflowStream

    from . import policy
    from .coder import implement
    from .planner import make_plan
    from .reviewer import Review, review_change

TOPIC = "assistant"
RUN_SHARE_TAKEN = "RunShareTaken"
SESSION_CAP_REACHED = "SessionCapReached"


class AssistantInput(BaseModel):
    """The workflow's input, and what it carries across continue-as-new."""

    session: ClaudeSessionState | None = None
    model: str | None = None
    max_budget_usd: float = 0.25
    max_cost_usd: float = 1.0
    messages_per_run: int = 2


@workflow.defn
class CodingAssistant:
    """Plans, makes and reviews each change the user asks for."""

    @workflow.init
    def __init__(self, input: AssistantInput) -> None:
        """Pick the coder's conversation up where the previous run left it, or start one."""
        self.stream = WorkflowStream()
        self._input = input
        self._coder = (
            ClaudeSession.restore(input.session)
            if input.session is not None
            else ClaudeSession(max_cost_usd=input.max_cost_usd)
        )
        self._decisions: list[str] = []
        self._taken = 0

    @workflow.run
    async def run(self, input: AssistantInput) -> None:
        """Take this run's share of messages, then continue as new."""
        await workflow.wait_condition(
            lambda: (
                self._taken >= input.messages_per_run
                and workflow.all_handlers_finished()
            )
        )
        workflow.continue_as_new(
            input.model_copy(update={"session": self._coder.snapshot()})
        )

    @workflow.update
    async def ask(self, task: str) -> Review:
        """One user message: plan it, make the change, review it."""
        self._taken += 1
        options = ClaudeTurnOptions(
            model=self._input.model,
            max_budget_usd=self._input.max_budget_usd,
            max_turns=8,
        )
        plan = await make_plan(task, options)

        async def approve(call: ClaudeDeferredTool) -> bool | tuple[bool, str]:
            verdict = policy.decide(
                call, files=plan.files, session_id=str(self._coder.state.session_id)
            )
            approved = verdict is True
            self._decisions.append(
                f"{call.name} {'approved' if approved else 'denied'}"
            )
            return verdict

        report = await implement(
            task,
            plan,
            session=self._coder,
            options=options,
            approve=approve,
            streaming_topic=TOPIC,
        )
        return await review_change(task, plan, report, options)

    @ask.validator
    def _admit(self, task: str) -> None:
        if self._taken >= self._input.messages_per_run:
            raise ApplicationError(
                f"this run has taken its share of messages; send {task!r} again for the "
                "next run",
                type=RUN_SHARE_TAKEN,
            )
        cap = self._coder.state.max_cost_usd
        if cap is not None and self._coder.state.cost_usd >= cap:
            raise ApplicationError(
                f"the session has spent its ${cap} cap", type=SESSION_CAP_REACHED
            )

    @workflow.query
    def decisions(self) -> list[str]:
        """Each native call this run decided, as ``"<tool> approved|denied"``."""
        return list(self._decisions)

    @workflow.query
    def cost_usd(self) -> float:
        """What the coder's session has spent."""
        return self._coder.state.cost_usd
