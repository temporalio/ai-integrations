"""A random agent for the chaos tests.

Each run follows a seeded plan: a few messages, each with one to three calls (Edit,
Write, Overwrite, Bash, or the durable ``count``), every call with a token of its own.
The agent acts on each result as a careful agent would: it calls again a call that did
not run, does not run again a command or a write that may have run, reads a file
before it edits or writes over it, and reads it after an edit that may have run, to
see whether it did. It decides from the conversation alone, so a segment that runs
again after a crash decides the same way.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.helpers.fake_messages_api import FakeMessagesAPI, history_of

DID_NOT_RUN = ("did not run yet", "This call did not run")
"""A denial after a pause, and a tool step that failed before its call could start."""
MAY_HAVE_RUN = "may have run"
"""A tool step that failed after its call may have started."""
STALE = "modified since read"
"""Claude Code checked an edit again (anthropics/claude-code#99041): must never happen."""
TOOLS = ["Edit", "Edit", "Write", "Overwrite", "Bash", "count"]
"""What a call can be (Edit twice as often). Overwrite is a Write over a file Claude
read, the other case of #99041."""

_TOKEN = re.compile(r"\btok(\d+)\b")


@dataclass(frozen=True)
class Intent:
    """One thing the agent wants done, and the token that shows it was done."""

    tool: str  # Edit, Write, Overwrite, Bash or count
    n: int

    @property
    def token(self) -> str:
        return f"tok{self.n}"


def plan(seed: int) -> list[list[Intent]]:
    """The run's messages, each a list of calls."""
    rng = random.Random(seed)
    messages: list[list[Intent]] = []
    n = 0
    for _ in range(rng.randint(3, 5)):
        message = []
        for _ in range(rng.choice([1, 1, 2, 3])):
            n += 1
            message.append(Intent(rng.choice(TOOLS), n))
        messages.append(message)
    return messages


@dataclass
class Seen:
    """What the agent learned from a conversation."""

    status: dict[str, str] = field(default_factory=dict)  # done, retry or check
    ok: dict[str, str] = field(default_factory=dict)  # token -> tool, for successes
    fresh: set[str] = field(default_factory=set)  # files Claude Code knows as they are
    interrupted: int = 0  # results that say the call may have run
    stale: int = 0
    errors: list[str] = field(default_factory=list)


def _raw_results(body: dict[str, Any]) -> dict[str, str]:
    """Each tool result's text as Claude Code sent it, by ``tool_use_id``."""
    found: dict[str, str] = {}
    for message in body.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                text = (
                    " ".join(
                        str(b.get("text", "")) for b in inner if isinstance(b, dict)
                    )
                    if isinstance(inner, list)
                    else str(inner)
                )
                found[str(block.get("tool_use_id"))] = text
    return found


class ChaosAgent:
    """The fake model's policy for one run."""

    def __init__(self, seed: int, work: Path) -> None:
        self.seed = seed
        self.plan = plan(seed)
        self.work = work
        self.notes = str(work / "notes.txt")
        self.over = str(work / "over.txt")
        self.effects = (work / "effects.log").as_posix()
        self.api: FakeMessagesAPI | None = None
        self.final: dict[str, Any] | None = None
        """The request the agent answered with its final text."""

    def token_of(self, name: str, args: dict[str, Any]) -> str | None:
        """The token of a call (None for a Read, which has none)."""
        if name == "count":
            return f"tok{args.get('n')}"
        if name == "Edit":
            text = str(args.get("new_string"))
        elif name == "Write":
            text = str(args.get("content"))
        elif name == "Bash":
            text = str(args.get("command"))
        else:
            return None
        found = _TOKEN.search(text)
        return f"tok{found.group(1)}" if found else None

    def read(self, body: dict[str, Any]) -> Seen:
        """Go through the conversation's results in order."""
        _, _, history = history_of(body)
        raw = _raw_results(body)
        seen = Seen()
        for item in history:
            text = raw.get(item.id, str(item.content))
            if MAY_HAVE_RUN in text:
                seen.interrupted += 1
            if item.name == "Read":
                path = str(item.input.get("file_path"))
                if item.is_error:
                    seen.errors.append(f"Read {path}: {text[:200]}")
                    continue
                seen.fresh.add(path)
                if path == self.notes:
                    for pending, status in list(seen.status.items()):
                        if status == "check":  # an edit that may have run: did it?
                            ran = re.search(rf"\b{pending}\b", text) is not None
                            seen.status[pending] = "done" if ran else "retry"
                continue
            token = self.token_of(item.name, item.input)
            if token is None:
                continue
            path = str(item.input.get("file_path"))
            if not item.is_error:
                seen.status[token] = "done"
                seen.ok[token] = "Overwrite" if path == self.over else item.name
                continue
            if STALE in text:
                seen.stale += 1
                seen.fresh.discard(path)
                seen.status[token] = "retry"
            elif any(s in text for s in DID_NOT_RUN):
                seen.status[token] = "retry"
            elif MAY_HAVE_RUN in text:
                seen.fresh.discard(path)
                # An edit that may have run is checked by reading the file; a write
                # or a command that may have run is not run again.
                seen.status[token] = "check" if item.name == "Edit" else "done"
            else:
                seen.status[token] = "done"
                seen.errors.append(f"{item.name} {token}: {text[:200]}")
        return seen

    def call(self, intent: Intent) -> dict[str, Any]:
        assert self.api is not None
        token = intent.token
        if intent.tool == "Edit":
            args = {
                "file_path": self.notes,
                "old_string": "END",
                "new_string": f"{token}\nEND",
            }
            return self.api.call("Edit", args)
        if intent.tool == "Write":
            path = str(self.work / f"w_{token}.txt")
            return self.api.call("Write", {"file_path": path, "content": token})
        if intent.tool == "Overwrite":
            return self.api.call("Write", {"file_path": self.over, "content": token})
        if intent.tool == "Bash":
            command = f"echo {token} >> {self.effects}"
            return self.api.call("Bash", {"command": command, "description": "record"})
        return self.api.tool_use("count", {"n": intent.n})

    def decide(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        """The fake model's answer to one request."""
        assert self.api is not None
        seen = self.read(body)
        todo: list[Intent] = []
        for message in self.plan:
            todo = [i for i in message if seen.status.get(i.token) != "done"]
            if todo:
                break
        if not todo:
            self.final = body
            return [{"type": "text", "text": "FINAL"}]
        checking = any(seen.status.get(i.token) == "check" for i in todo)
        if checking or (
            any(i.tool == "Edit" for i in todo) and self.notes not in seen.fresh
        ):
            return [self.api.call("Read", {"file_path": self.notes})]
        if any(i.tool == "Overwrite" for i in todo) and self.over not in seen.fresh:
            return [self.api.call("Read", {"file_path": self.over})]
        return [self.call(i) for i in todo]
