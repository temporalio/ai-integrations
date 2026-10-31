"""Agent middleware that lets :func:`run_deep_agent` continue as new mid-turn.

``run_deep_agent`` normally looks at history only after ``agent.ainvoke(...)``
returns, so one long turn (hundreds of model and tool calls) lands in one
history however long it gets. :class:`ContinueAsNewMiddleware` ends the agent
loop before its next model step once history is past the driver's limit; the
loop then returns a consistent transcript (every tool result is in) and the
driver carries it into a fresh run, where the agent picks up at that model step.
"""

from __future__ import annotations

from typing import Any

from temporalio import workflow
from temporalio.deepagents.workflow import _mid_turn, _past_limit

with workflow.unsafe.imports_passed_through():
    from langchain.agents.middleware import AgentMiddleware, hook_config
    from langchain_core.runnables.config import ensure_config


def _cut_requested() -> bool:
    """Count a model step of the driven turn; True once the turn should end.

    False when the agent is not driven by ``run_deep_agent``, and at the first
    model step of each ``run_deep_agent`` invocation, so a run that starts past
    the limit (a large carried transcript) still makes progress. Once true it
    stays true for the rest of the turn. It uses the same ``_past_limit`` check
    as the between-turn driver; the tests drive the ``continue_as_new_after``
    path, and the server-suggestion path is not separately replay-tested.
    """
    holder = _mid_turn.get()
    if holder is None:
        return False
    holder.model_steps += 1
    if (
        not holder.cut
        and holder.model_steps > 1
        and _past_limit(holder.continue_as_new_after)
    ):
        holder.cut = True
    return holder.cut


class ContinueAsNewMiddleware(AgentMiddleware):
    """End the agent loop before a model step when ``run_deep_agent`` should continue as new.

    Add it to the ``middleware=`` of the agent that :func:`run_deep_agent`
    drives. It only acts under the driver, using the driver's limit (the
    server's suggestion, or ``continue_as_new_after``); with a plain
    ``ainvoke`` it does nothing, so a turn is never cut without a driver to
    continue it. Async only (``abefore_model``), as the plugin's agents run.

    The cut ends the loop the way a finished turn ends (``jump_to: "end"``),
    so other middleware's ``after_agent`` hooks run at each cut, on the
    transcript so far. Sub-agents are not cut, also when it is on their
    ``middleware=``: their loop runs inside the parent's ``task`` call, and
    cutting one would hand the parent a partial answer. Other agents nested
    in the turn (an agent called from a tool) are not supported: they share
    the turn's step count.

    Only messages cross the cut, as with between-turn continue-as-new:
    state-backed files, todo state and custom middleware state do not cross as
    state, and each run starts with a fresh recursion limit.
    """

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Jump to the end of the loop once the driver asked for continue-as-new."""
        configurable = ensure_config().get("configurable") or {}
        # Deep Agents marks a sub-agent's run config this way (its task tool).
        if configurable.get("ls_agent_type") == "subagent":
            return None
        if _cut_requested():
            return {"jump_to": "end"}
        return None
