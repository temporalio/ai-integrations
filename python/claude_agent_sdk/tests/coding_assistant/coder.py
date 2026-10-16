"""The coder: carries out the plan in the repository, in one conversation across messages."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporalio.claude_agent_sdk import (
        READ_ONLY_NATIVE_TOOLS,
        ClaudeDeferredTool,
        ClaudeSession,
        ClaudeTurnOptions,
        run_claude_turn,
    )

    from .planner import Plan
    from .policy import SAFE_COMMANDS
    from .tools import comment_on_ticket, lookup_ticket

SYSTEM_PROMPT = (
    "You are a coder working in the repository in your working directory. Carry out the plan "
    "you are given, editing only its files with the Edit tool. The only shell commands you may "
    f"run, from your working directory and exactly as written, are: {sorted(SAFE_COMMANDS)}. "
    "When you changed something, post one comment on the ticket saying what with "
    "comment_on_ticket. Then reply in one or two sentences."
)
NATIVE_TOOLS = ["Read", "Glob", "Grep", "Edit", "Bash"]
AUTO_ALLOW = sorted(READ_ONLY_NATIVE_TOOLS.intersection(NATIVE_TOOLS))
"""Reads inside the session's directory run unasked; every other native call goes to policy."""


async def implement(
    task: str,
    plan: Plan,
    *,
    session: ClaudeSession,
    options: ClaudeTurnOptions,
    approve: Callable[[ClaudeDeferredTool], Awaitable[bool | tuple[bool, str]]],
    streaming_topic: str,
) -> str:
    """Carry out ``plan`` on ``session``; the coder's reply, streamed as Claude writes it."""
    reply = await run_claude_turn(
        f"Implement: {task}\n\nPlan:\n{plan.model_dump_json(indent=2)}",
        session=session,
        options=options.model_copy(update={"system_prompt": SYSTEM_PROMPT}),
        tools=[lookup_ticket, comment_on_ticket],
        native_tools=NATIVE_TOOLS,
        native_auto_allow=AUTO_ALLOW,
        approve=approve,
        streaming_topic=streaming_topic,
    )
    return reply.text or ""
