"""Scripted behavior that calls several tools in one message."""

from __future__ import annotations

import re

from temporalio.claude_agent_sdk.testing import Final, HistoryItem, ToolCall


def parallel_policy(
    prompt: str, history: list[HistoryItem]
) -> ToolCall | list[ToolCall] | Final:
    """'parallel N': count 1 to N in one message, then answer.

    With 'and publish', the same message also asks to publish (which needs approval).
    Steps a stopped task left without a result are asked for again.
    """
    match = re.search(r"parallel (\d+)", prompt)
    target = int(match.group(1)) if match else 0
    counted = {
        h.input.get("n") for h in history if h.name == "count" and not h.is_error
    }
    todo = [n for n in range(1, target + 1) if n not in counted]
    if not todo:
        return Final(f"counted {target} at once")
    calls = [ToolCall("count", {"n": n}) for n in todo]
    if "publish" in prompt and not any(h.name == "publish" for h in history):
        calls.insert(1, ToolCall("publish", {"n": target}))
    return calls
