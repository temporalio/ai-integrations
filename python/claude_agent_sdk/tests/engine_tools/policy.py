"""Scripted behavior that calls Claude Code tools."""

from __future__ import annotations

import re

from temporalio.claude_agent_sdk.testing import Final, HistoryItem, ToolCall


def shell_policy(prompt: str, history: list[HistoryItem]) -> ToolCall | Final:
    """'run: <command>' runs it with Bash, 'note: <text>' calls the notes MCP tool,
    'count then run: <command>' counts first. The answer is what the tool returned."""
    count = re.match(r"count then run: (.*)", prompt, re.S)
    if count and not any(h.name == "count" for h in history):
        return ToolCall("count", {"n": 1})
    match = re.match(r"(?:count then )?run: (.*)", prompt, re.S)
    note = re.match(r"note: (.*)", prompt, re.S)
    name = "Bash" if match else "mcp__notes__add_note"
    done = [h for h in history if h.name == name]
    if not done:
        if match:
            return ToolCall("Bash", {"command": match.group(1), "description": "run"})
        return ToolCall(name, {"text": note.group(1) if note else ""})
    last = done[-1]
    return Final(f"{'error: ' if last.is_error else ''}{last.content}")


def together_policy(prompt: str, history: list[HistoryItem]) -> list[ToolCall] | Final:
    """'together: <command>' calls Bash and ``count`` in one message (in the order
    'bash first' or 'count first' says); a call that did not run is asked again."""
    command = prompt.split(": ", 1)[1]
    bash = [h for h in history if h.name == "Bash" and not h.is_error]
    counted = [h for h in history if h.name == "count" and not h.is_error]
    if bash and counted:
        return Final(f"{bash[-1].content} and counted {counted[-1].content['n']}")
    calls = []
    if not bash:
        calls.append(ToolCall("Bash", {"command": command, "description": "run"}))
    if not counted:
        calls.append(ToolCall("count", {"n": 1}))
    return calls if prompt.startswith("bash first") else calls[::-1]


def file_policy(prompt: str, history: list[HistoryItem]) -> list[ToolCall] | Final:
    """'files: <path>' reads the file, edits it together with a ``count``, edits it
    again, writes over it, and edits what it wrote. The answer says how each call
    went, in order."""
    path = prompt.split(": ", 1)[1]

    def edit(old: str, new: str) -> ToolCall:
        return ToolCall(
            "Edit", {"file_path": path, "old_string": old, "new_string": new}
        )

    plan = {
        0: [ToolCall("Read", {"file_path": path})],
        1: [edit("beta", "gamma"), ToolCall("count", {"n": 1})],
        3: [edit("gamma", "delta")],
        4: [ToolCall("Write", {"file_path": path, "content": "written\n"})],
        5: [edit("written", "rewritten")],
    }
    if len(history) in plan and not any(h.is_error for h in history):
        return plan[len(history)]
    return Final(
        " ".join(
            f"{h.name}:" + (f"error {str(h.content)[:120]}" if h.is_error else "ok")
            for h in history
        )
    )


def edit_policy(prompt: str, history: list[HistoryItem]) -> list[ToolCall] | Final:
    """'edit: <path>' edits the file together with a ``count``, then writes over it
    (for the scripted runner, which has no Read). The answer says how each call went,
    in order."""
    path = prompt.split(": ", 1)[1]
    edit = {"file_path": path, "old_string": "draft", "new_string": "final"}
    plan = {
        0: [ToolCall("Edit", edit), ToolCall("count", {"n": 1})],
        2: [ToolCall("Write", {"file_path": path, "content": "written\n"})],
    }
    if len(history) in plan and not any(h.is_error for h in history):
        return plan[len(history)]
    return Final(
        " ".join(f"{h.name}:" + ("error" if h.is_error else "ok") for h in history)
    )


def edit_once_policy(prompt: str, history: list[HistoryItem]) -> ToolCall | Final:
    """'edit once: <path>' reads the file, then adds a line under its first one (an
    edit that changes the file again each time it runs)."""
    path = prompt.split(": ", 1)[1]
    if not history:
        return ToolCall("Read", {"file_path": path})
    if len(history) == 1 and not history[0].is_error:
        return ToolCall(
            "Edit",
            {
                "file_path": path,
                "old_string": "first\n",
                "new_string": "first\nedited\n",
            },
        )
    return Final(
        " ".join(f"{h.name}:" + ("error" if h.is_error else "ok") for h in history)
    )
