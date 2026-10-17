"""Scripted Claude behavior for the lifecycle tests."""

from __future__ import annotations

import re

from temporalio.claude_agent_sdk.testing import Final, HistoryItem, ToolCall


def big_input_policy(prompt: str, history: list[HistoryItem]) -> ToolCall | Final:
    """'big N': call ``count`` N times, each with a 300 KB argument."""
    match = re.search(r"big (\d+)", prompt)
    target = int(match.group(1)) if match else 0
    done = sum(1 for h in history if h.name == "count" and not h.is_error)
    if done < target:
        return ToolCall("count", {"n": done + 1, "blob": "x" * 300_000})
    return Final(f"counted to {target}")


def big_result_policy(prompt: str, history: list[HistoryItem]) -> ToolCall | Final:
    """'fetch N KB twice': two large results in a row, then an answer."""
    match = re.search(r"fetch (\d+) KB", prompt)
    kb = int(match.group(1)) if match else 1
    done = [h for h in history if h.name == "fetch_document" and not h.is_error]
    if len(done) < 2:
        return ToolCall("fetch_document", {"kb": kb})
    return Final(f"fetched {len(done)} documents")
