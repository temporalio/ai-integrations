"""Run a summarizing conversation manager's summary on the workflow event loop.

Strands' ``SummarizingConversationManager`` reduces the history in synchronous
code. It gets the summary from the agent's model through ``run_async``, which runs
the model on a new thread, or by calling its ``summarization_agent`` synchronously.
A ``TemporalModel`` cannot run on that thread: it needs the workflow event loop.

``TemporalAgent`` calls :func:`prepare_summary` just before Strands reduces the
history. It runs the manager's own reduction on copies to learn which messages the
manager will summarize, awaits that summary here, and hands it to the manager's
synchronous call.
"""

import asyncio
import copy
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.agent.conversation_manager.compression.context_compression import (
    generate_summary,
)
from strands.tools._tool_helpers import noop_tool
from strands.tools.registry import ToolRegistry
from strands.types.content import Message
from strands.types.tools import AgentTool

from temporalio.exceptions import is_cancelled_exception

# Attribute on the manager holding the prepared (messages, summary or error).
_PREPARED = "_temporal_prepared_summary"


class _SummaryRequest(BaseException):
    """Carries the messages a dry run would summarize.

    A ``BaseException`` so the manager's ``except Exception`` handlers let it through.
    """

    def __init__(self, messages: list[Message]) -> None:
        super().__init__()
        self.messages = messages


async def prepare_summary(
    agent: Agent, reduce: Callable[[SummarizingConversationManager, Any], Any]
) -> None:
    """Await the summary that ``reduce`` will ask the agent's manager for, if any.

    ``reduce`` makes the same call Strands is about to make, on the manager and
    agent it is given.
    """
    manager = agent.conversation_manager
    if not isinstance(manager, SummarizingConversationManager):
        return
    messages = _requested_messages(manager, agent, reduce)
    if messages is None:
        return
    try:
        result: Message | Exception = await _summarize(manager, agent, messages)
    except Exception as error:
        if _is_cancelled(error):
            # Strands handles a failed summary as an ordinary error, so the run
            # would not end cancelled.
            raise asyncio.CancelledError() from error
        # Strands raises or ignores a failed summary; hand it over to do that.
        result = error
    _hand_over(manager, messages, result)


def _requested_messages(
    manager: SummarizingConversationManager,
    agent: Agent,
    reduce: Callable[[SummarizingConversationManager, Any], Any],
) -> list[Message] | None:
    scratch = copy.copy(manager)

    def capture(messages: list[Message], _agent: Agent) -> Message:
        raise _SummaryRequest(messages)

    scratch._generate_summary = capture  # type: ignore[assignment]
    # The manager reads the messages and the model, and may mark messages as pinned.
    stub = SimpleNamespace(messages=copy.deepcopy(agent.messages), model=agent.model)
    try:
        reduce(scratch, stub)
    except _SummaryRequest as request:
        return request.messages
    except Exception:
        # Raised before any summary, such as when too few messages are left; the
        # manager's own call raises it too.
        pass
    return None


async def _summarize(
    manager: SummarizingConversationManager, agent: Agent, messages: list[Message]
) -> Message:
    if manager.summarization_agent is not None:
        return await _summarize_with_agent(manager.summarization_agent, messages)
    # As the manager does: newer Strands versions summarize with agent.aux_model.
    model = getattr(agent, "aux_model", agent.model)
    return await generate_summary(messages, model, manager.summarization_system_prompt)


async def _summarize_with_agent(summarizer: Agent, messages: list[Message]) -> Message:
    # The steps of SummarizingConversationManager._generate_summary_with_agent,
    # awaiting invoke_async instead of calling the agent synchronously.
    system_prompt = summarizer.system_prompt
    original_messages = summarizer.messages.copy()
    tool_registry = summarizer.tool_registry
    structured_output_model = getattr(
        summarizer, "_default_structured_output_model", None
    )
    try:
        if hasattr(summarizer, "_default_structured_output_model"):
            summarizer._default_structured_output_model = None
        if not summarizer.tool_names:
            registry = ToolRegistry()
            registry.register_tool(cast(AgentTool, noop_tool))
            summarizer.tool_registry = registry
        # A copy: the agent adds its turn to it, and the manager's call is matched
        # against these messages.
        summarizer.messages = copy.deepcopy(messages)
        result = await summarizer.invoke_async("Please summarize this conversation.")
        summary = result.message.copy()
        summary["role"] = "user"
        return summary
    finally:
        summarizer.system_prompt = system_prompt
        summarizer.messages = original_messages
        summarizer.tool_registry = tool_registry
        if hasattr(summarizer, "_default_structured_output_model"):
            summarizer._default_structured_output_model = structured_output_model


def _hand_over(
    manager: SummarizingConversationManager,
    messages: list[Message],
    result: Message | Exception,
) -> None:
    if _PREPARED not in vars(manager):
        generate = manager._generate_summary

        def prepared_summary(to_summarize: list[Message], agent: Agent) -> Message:
            prepared = vars(manager).get(_PREPARED)
            setattr(manager, _PREPARED, None)
            if prepared is None or prepared[0] != to_summarize:
                return generate(to_summarize, agent)
            if isinstance(prepared[1], Exception):
                raise prepared[1]
            return prepared[1]

        manager._generate_summary = prepared_summary  # type: ignore[assignment]
    setattr(manager, _PREPARED, (messages, result))


def _is_cancelled(error: BaseException | None) -> bool:
    # A summarization agent's model error reaches here wrapped in Strands' errors.
    while error is not None:
        if is_cancelled_exception(error):
            return True
        error = error.__cause__
    return False
