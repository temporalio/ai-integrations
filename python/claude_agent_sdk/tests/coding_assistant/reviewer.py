"""The reviewer: checks the coder's report against the plan and returns a typed review."""

from __future__ import annotations

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel

    from temporalio.claude_agent_sdk import ClaudeTurnOptions, run_claude_turn

    from .planner import Plan

SYSTEM_PROMPT = (
    "You review a code change. Given the task, the plan and the coder's report, say whether "
    "the change does what the task asks, which files it changed, and summarize it in one "
    "sentence."
)


class Review(BaseModel):
    """The change as the reviewer saw it: what one message to the assistant returns."""

    approved: bool
    files: list[str]
    summary: str


async def review_change(
    task: str, plan: Plan, report: str, options: ClaudeTurnOptions
) -> Review:
    """Review the coder's ``report`` in a one-off session."""
    reply = await run_claude_turn(
        f"Review: {task}\n\nPlan:\n{plan.model_dump_json(indent=2)}\n\n"
        f"Coder's report:\n{report}",
        options=options.model_copy(update={"system_prompt": SYSTEM_PROMPT}),
        output_type=Review,
    )
    if not isinstance(reply.output, Review):
        raise ApplicationError("the reviewer returned no review", non_retryable=True)
    return reply.output
