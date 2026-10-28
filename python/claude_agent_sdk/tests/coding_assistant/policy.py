"""The coder's approval policy, an allowlist: approve only what is listed, deny the rest.

Reads inside the session's directory never reach it (they are auto-allowed). Everything else
does: an edit is approved only for a file the plan names, inside the session's directory, and a
command only when it is exactly one of ``SAFE_COMMANDS``. A denial tells Claude why.
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporalio.claude_agent_sdk import ClaudeDeferredTool

SAFE_COMMANDS = frozenset({"python -m pytest -q", "git diff", "git status"})


def decide(
    call: ClaudeDeferredTool, *, files: list[str], session_id: str
) -> bool | tuple[bool, str]:
    """Approve ``call`` if the allowlist covers it; otherwise deny it with the reason."""
    if call.name == "Edit":
        path = PurePosixPath(str(call.input.get("file_path", "")))
        if any(_is(path, file, session_id) for file in files):
            return True
        return False, f"You may edit only the files in the plan: {files}."
    if call.name == "Bash":
        command = str(call.input.get("command", ""))
        try:
            # Comparing tokens, not text, makes `git  diff` match and `git diff; rm x` not.
            if " ".join(shlex.split(command)) in SAFE_COMMANDS:
                return True
        except ValueError:
            pass
        return False, (
            f"Only these commands are allowed: {sorted(SAFE_COMMANDS)}. "
            "Tell the user what you would run instead."
        )
    return False, f"{call.name} is not allowed here."


def _is(path: PurePosixPath, file: str, session_id: str) -> bool:
    """Whether ``path`` names the plan's ``file``: relative to the session's directory, or
    absolute under it (Claude Code's Edit takes absolute paths, and the directory is named by
    the session id). The plan comes from a model, so its paths are checked too.
    """
    rel = PurePosixPath(file)
    if not rel.parts or rel.is_absolute() or ".." in rel.parts or ".." in path.parts:
        return False
    if not path.is_absolute():
        return path == rel
    return path.parts[-len(rel.parts) - 1 :] == (session_id, *rel.parts)
