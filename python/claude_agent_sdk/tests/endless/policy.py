"""Scripted behavior for the long-running agents."""

from __future__ import annotations

import re

from temporalio.claude_agent_sdk.testing import Final, HistoryItem, ToolCall


def count_policy(prompt: str, history: list[HistoryItem]) -> ToolCall | Final:
    """'count to N': call ``count`` until N steps ran in the whole session, then answer.

    With 'and publish' it also calls ``publish`` (which needs approval) at the end.
    """
    match = re.search(r"count to (\d+)", prompt)
    target = int(match.group(1)) if match else 0
    counted = sum(1 for h in history if h.name == "count" and not h.is_error)
    if counted < target:
        return ToolCall("count", {"n": counted + 1})
    if "publish" in prompt and not any(h.name == "publish" for h in history):
        return ToolCall("publish", {"n": target})
    return Final(f"counted to {target}")
