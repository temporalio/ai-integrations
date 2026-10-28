# The committed replay histories of the Claude integration (histories/*.json), recorded by the
# offline FakeClaudeCLI scenario tests and replayed by test_claude_replay.py. A scenario test
# calls `record(name, history)`, which writes only under CLAUDE_RECORD_HISTORIES=1
# (`CLAUDE_RECORD_HISTORIES=1 make test PYTEST_ARGS=tests/test_claude_workflow.py`). Machine-specific values (temp and home directories, the
# checkout, the host name) are replaced by fixed placeholders, in JSON payloads too, so a file is
# the same whichever machine recorded it and leaks nothing about it.

from __future__ import annotations

import base64
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any

from temporalio.client import WorkflowHandle, WorkflowHistory

HISTORIES = Path(__file__).parents[1] / "histories"
RECORD_ENV = "CLAUDE_RECORD_HISTORIES"
REPO = Path(__file__).resolve().parents[4]

# Anything that would identify the recording machine. The replay test refuses them in every file.
FORBIDDEN = ("/Users/", "/home/", "/private/var/", "/var/folders/", "sk-ant-")


def _replacements() -> list[tuple[str, str]]:
    tmp = tempfile.gettempdir()
    pairs = {
        os.path.realpath(tmp): "/replay/tmp",
        tmp: "/replay/tmp",
        str(REPO): "/replay/repo",
        os.path.realpath(REPO): "/replay/repo",
        str(Path.home()): "/replay/user-home",
        socket.gethostname(): "replay-host",
    }
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        pairs[key] = "<redacted>"
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def scrub(text: str, replacements: list[tuple[str, str]] | None = None) -> str:
    replacements = _replacements() if replacements is None else replacements
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _scrub_tree(node: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(node, dict):
        if isinstance(node.get("data"), str) and isinstance(node.get("metadata"), dict):
            encoding = base64.b64decode(node["metadata"].get("encoding", "")).decode()
            if encoding.startswith("json/"):
                raw = base64.b64decode(node["data"]).decode()
                node = {
                    **node,
                    "data": base64.b64encode(
                        scrub(raw, replacements).encode()
                    ).decode(),
                }
        return {k: _scrub_tree(v, replacements) for k, v in node.items()}
    if isinstance(node, list):
        return [_scrub_tree(v, replacements) for v in node]
    if isinstance(node, str):
        return scrub(node, replacements)
    return node


def decoded(text: str) -> str:
    """The history's JSON with every JSON payload decoded, for searching it."""
    parts: list[str] = [text]

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("data"), str) and isinstance(
                node.get("metadata"), dict
            ):
                parts.append(base64.b64decode(node["data"]).decode(errors="replace"))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(json.loads(text))
    return "\n".join(parts)


async def record(name: str, history: WorkflowHistory | WorkflowHandle) -> None:
    """Write ``history`` as ``histories/<name>.json`` when recording; otherwise do nothing."""
    if os.environ.get(RECORD_ENV) != "1":
        return
    if isinstance(history, WorkflowHandle):
        history = await history.fetch_history()
    text = json.dumps(
        _scrub_tree(json.loads(history.to_json()), _replacements()), indent=2
    )
    if leaks := [f for f in FORBIDDEN if f in decoded(text)]:
        raise AssertionError(f"history {name} still holds {leaks} after scrubbing")
    HISTORIES.mkdir(exist_ok=True)
    (HISTORIES / f"{name}.json").write_text(text + "\n")
