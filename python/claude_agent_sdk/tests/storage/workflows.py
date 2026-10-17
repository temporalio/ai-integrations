"""An agent that fetches a large document and reports its size."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool
from temporalio.claude_agent_sdk.testing import Final, HistoryItem, ToolCall
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tests.refund import shop


@activity.defn
async def fetch_document(args: dict[str, Any]) -> str:
    """Fetch a document of the requested size in KB."""
    shop.log_execution(
        "fetch_document", activity.info().activity_id, str(args.get("kb"))
    )
    return "<doc>" + "x" * (int(args["kb"]) * 1024) + "</doc>"


def fetch_policy(prompt: str, history: list[HistoryItem]) -> ToolCall | Final:
    """'fetch N KB': fetch it once, then report how many characters of it arrived."""
    match = re.search(r"fetch (\d+) KB", prompt)
    if not history:
        return ToolCall("fetch_document", {"kb": int(match.group(1)) if match else 1})
    last = history[-1]
    body = re.search(r"<doc>(x*)</doc>", str(last.content))
    if last.is_error or body is None:
        return Final(f"fetch failed: {str(last.content)[:200]}")
    return Final(f"got {len(body.group(1))} characters")


def pages_policy(prompt: str, history: list[HistoryItem]) -> ToolCall | Final:
    """'fetch N KB K times': fetch K documents, then report how much of them arrived."""
    match = re.search(r"fetch (\d+) KB (\d+) times", prompt)
    kb, times = (int(match.group(1)), int(match.group(2))) if match else (1, 1)
    done = [h for h in history if h.name == "fetch_document" and not h.is_error]
    if len(done) < times:
        return ToolCall("fetch_document", {"kb": kb})
    bodies = [re.search(r"<doc>(x*)</doc>", str(h.content)) for h in done]
    seen = sum(len(b.group(1)) for b in bodies if b is not None)
    return Final(f"saw {seen} characters in {len(done)} documents")


@workflow.defn
class FetchWorkflow:
    """Fetches one large document."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(
                    fetch_document,
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )
            ],
            segment_timeout=timedelta(minutes=2),
            segment_retry_policy=RetryPolicy(maximum_attempts=2),
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Run the request."""
        return await self.agent.run(prompt)


def parallel_fetch_policy(
    prompt: str, history: list[HistoryItem]
) -> list[ToolCall] | Final:
    """'fetch N KB K at once': K fetches in one message, then what arrived."""
    match = re.search(r"fetch (\d+) KB (\d+) at once", prompt)
    kb, k = (int(match.group(1)), int(match.group(2))) if match else (1, 1)
    done = [h for h in history if h.name == "fetch_document"]
    if not done:
        return [ToolCall("fetch_document", {"kb": kb}) for _ in range(k)]
    notes = sum(1 for h in done if h.is_error and "too large" in str(h.content))
    return Final(f"got {len(done) - notes} documents and {notes} notes")


@workflow.defn
class ParallelFetchWorkflow:
    """Fetches several large documents in one message."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(
            tools=[
                activity_as_tool(
                    fetch_document,
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )
            ],
            segment_timeout=timedelta(minutes=2),
            segment_retry_policy=RetryPolicy(maximum_attempts=2),
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Run the request."""
        return await self.agent.run(prompt)
