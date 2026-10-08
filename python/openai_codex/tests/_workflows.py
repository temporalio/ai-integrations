"""Workflows and activities for the tests (own module: the Workflow sandbox re-imports it)."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
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
