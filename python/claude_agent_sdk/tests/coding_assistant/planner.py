"""The planner: reads the ticket and returns a typed plan for the coder."""

from __future__ import annotations

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel

    from temporalio.claude_agent_sdk import ClaudeTurnOptions, run_claude_turn

    from .tools import lookup_ticket

SYSTEM_PROMPT = (
    "You plan small code changes. Read the ticket the task names with lookup_ticket. "
    "List only the files that must change, as paths relative to the repository root, "
    "and the steps that change them. List no files when nothing should be edited."
)


class Plan(BaseModel):
    """What the coder is asked to do. Its policy lets it edit only ``files``."""

    files: list[str]
    steps: list[str]


async def make_plan(task: str, options: ClaudeTurnOptions) -> Plan:
    """A plan for ``task``, in a one-off session: the planner keeps nothing between messages."""
    reply = await run_claude_turn(
        f"Plan: {task}",
        options=options.model_copy(update={"system_prompt": SYSTEM_PROMPT}),
        tools=[lookup_ticket],
        output_type=Plan,
    )
    if not isinstance(reply.output, Plan):
        raise ApplicationError("the planner returned no plan", non_retryable=True)
    return reply.output
