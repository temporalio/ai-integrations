"""Workflows and activities for the tests (own module: the Workflow sandbox re-imports it)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
    from temporalio.openai_codex import CodexApprovalDecision, CodexApprovalRequest
    from temporalio.openai_codex.workflow import (
        CodexSession,
        activity_as_tool,
        codex_tool,
    )


async def lookup(q: str) -> str:
    """Look up a ticket by id."""
    return f"ticket-{q}: resolved"


async def explode(reason: str) -> str:
    """Always fails."""
    raise ValueError(reason)


@activity.defn
async def record(note: str) -> str:
    """Record a note in the ledger (a real side effect: tests count executions)."""
    with open(os.environ["CODEX_TEST_LEDGER"], "a") as f:
        f.write(f"{note}|{activity.info().activity_id}\n")
    return f"recorded:{note}"


def new_session() -> CodexSession:
    """A session with one in-workflow tool, one Activity tool and one failing tool."""
    return CodexSession(
        native_tools=False,
        tools=[
            codex_tool(lookup),
            codex_tool(explode),
            activity_as_tool(record, start_to_close_timeout=timedelta(seconds=30)),
        ],
        instructions="Use the tools you are given.",
    )


@workflow.defn
class CodexWorkflow:
    """Runs one Codex turn."""

    def __init__(self) -> None:
        self.session = new_session()

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Run the turn and return Codex's answer."""
        return (await self.session.run(prompt)).text

    @workflow.query
    def state(self) -> dict[str, Any]:
        """The conversation state the Workflow holds."""
        return {
            "thread_id": self.session.thread_id,
            "rollout_lines": len(self.session.rollout.splitlines()),
            "tool_results": self.session.tool_results,
        }


@workflow.defn
class ObservedCodexWorkflow:
    """Runs one Codex turn with an observer context."""

    def __init__(self) -> None:
        self.session = new_session()

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Run the turn, streaming through the Worker's observer."""
        result = await self.session.run(prompt, observer_context={"turn": "t1"})
        return result.text


@dataclass
class NativeInput:
    """Input of :class:`NativeCodexWorkflow`."""

    prompt: str
    cwd: str
    # "approve" / "decline": decide immediately. "signal": wait for the `decide` Signal.
    # "none": configure no approval handler at all.
    mode: str = "approve"
    model: str | None = None
    approval_policy: str = "untrusted"
    heartbeat_seconds: float = 30


@workflow.defn
class NativeCodexWorkflow:
    """Runs one native Codex turn: Codex uses its own tools, the Workflow decides approvals."""

    @workflow.init
    def __init__(self, input: NativeInput) -> None:
        self.mode = input.mode
        self.asked: list[CodexApprovalRequest] = []
        self.signalled: dict[str, bool] = {}
        self.session = CodexSession(
            cwd=input.cwd,
            model=input.model,
            approval_policy=input.approval_policy,
            heartbeat_timeout=timedelta(seconds=input.heartbeat_seconds),
            approval_handler=None if input.mode == "none" else self._decide,
        )

    async def _decide(self, request: CodexApprovalRequest) -> CodexApprovalDecision:
        self.asked.append(request)
        if self.mode == "signal":
            await workflow.wait_condition(lambda: request.item_id in self.signalled)
            return CodexApprovalDecision(self.signalled[request.item_id])
        return CodexApprovalDecision(self.mode == "approve")

    @workflow.run
    async def run(self, input: NativeInput) -> str:
        """Run the turn and return Codex's answer."""
        return (
            await self.session.run(input.prompt, observer_context={"turn": "n1"})
        ).text

    @workflow.signal
    def decide(self, item_id: str, approved: bool) -> None:
        """Answer a pending approval."""
        self.signalled[item_id] = approved

    @workflow.query
    def pending(self) -> list[CodexApprovalRequest]:
        """Approvals asked and not yet answered."""
        return list(self.session.pending_approvals.values())

    @workflow.query
    def asked_requests(self) -> list[CodexApprovalRequest]:
        """Every approval the handler was asked about."""
        return self.asked


@workflow.defn
class ChatWorkflow:
    """A native Codex conversation: one turn per ``say`` Signal, all on one thread."""

    @workflow.init
    def __init__(self, cwd: str) -> None:
        self.session = CodexSession(cwd=cwd, approval_policy="on-request")
        self.prompts: list[str] = []
        self.replies: list[str] = []

    @workflow.run
    async def run(self, cwd: str) -> list[str]:  # pyright: ignore[reportUnusedParameter]
        """Answer each prompt in order; finish on an empty one."""
        while True:
            await workflow.wait_condition(lambda: len(self.prompts) > len(self.replies))
            prompt = self.prompts[len(self.replies)]
            if not prompt:
                return self.replies
            self.replies.append((await self.session.run(prompt)).text)

    @workflow.signal
    def say(self, prompt: str) -> None:
        """Queue a user message."""
        self.prompts.append(prompt)

    @workflow.query
    def answered(self) -> list[str]:
        """The replies so far."""
        return self.replies
