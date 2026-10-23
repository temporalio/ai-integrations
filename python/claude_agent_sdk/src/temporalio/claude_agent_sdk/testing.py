"""Test helpers: a scripted stand-in for Claude that needs no engine and no API key.

``ScriptedClaude`` is a segment runner driven by a Python policy. Like the real
runner, it keeps each conversation in its Workflow by default, so a new Worker
process can continue a conversation that a crashed Worker started, with no shared
storage; or, given a folder, it keeps sessions there, like a session store. Each
segment ends at a checkpoint, and a segment that runs again starts over from the
checkpoint it was given, so a retry makes Claude decide again (with a new tool call
id).
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._conversation import external_storage_on, read_conversation
from ._events import emit
from ._models import (
    DeferredCall,
    SegmentInput,
    SegmentOutput,
    ToolOutcome,
    ToolStepInput,
)


@dataclass
class ToolCall:
    """A policy's decision to call a tool.

    Attributes:
        name: The durable tool to call, or a Claude Code tool in ``tool_activities``
            (such as ``Bash``), which ``engine_tools`` plays in its tool step.
        input: The call's arguments.
    """

    name: str
    input: dict[str, Any]


@dataclass
class Final:
    """A policy's final answer.

    Attributes:
        text: The answer.
    """

    text: str


@dataclass
class HistoryItem:
    """One finished tool call, as the policy sees it.

    Attributes:
        id: The ``tool_use_id``.
        name: The tool's name.
        input: The call's arguments.
        content: The tool's result.
        is_error: Whether the call failed.
        kind: ``durable``, or ``engine`` for a Claude Code tool that ran as its own
            Activity.
    """

    id: str
    name: str
    input: dict[str, Any]
    content: Any
    is_error: bool
    kind: str = "durable"


Policy = Callable[[str, "list[HistoryItem]"], "ToolCall | list[ToolCall] | Final"]
"""Decides the next step from the latest prompt and every finished tool call so far.

A list of calls is one message with several calls; they all run, and their results
arrive together.
"""


def _as_outcome(value: Any) -> ToolOutcome:
    if isinstance(value, ToolOutcome):
        return value
    return ToolOutcome(
        content=value.get("content"), is_error=bool(value.get("is_error"))
    )


def _new_state(prompt: str) -> dict[str, Any]:
    return {"prompt": prompt, "history": [], "pending": [], "turn": 0}


def _pending(state: dict[str, Any]) -> list[dict[str, Any]]:
    """The calls waiting for results (a folder from an older version kept one call)."""
    pending = state.get("pending")
    if isinstance(pending, dict):
        return [pending]
    return list(pending or [])


def _fold(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The conversation's state after ``entries`` (one entry per segment)."""
    state = _new_state("")
    for entry in entries:
        if "prompt" in entry:
            state["prompt"] = entry["prompt"]
        for call, result in zip(state["pending"], entry.get("results", [])):
            state["history"].append({**call, **result})
        state["pending"] = entry.get("calls", [])
        state["turn"] = entry["turn"]
    return state


class ScriptedClaude:
    """A segment runner that plays Claude with a Python policy."""

    def __init__(
        self,
        policy: Policy,
        state_dir: str | os.PathLike[str] | None = None,
        *,
        cost_per_segment: float = 0.01,
        think_seconds: float = 0.0,
        engine_tools: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    ) -> None:
        """Create the runner.

        Args:
            policy: Decides each step.
            state_dir: None (the default) keeps each conversation in its Workflow,
                like ``ClaudeAgentSdkRunner`` without a session store. A folder keeps
                sessions there instead, like a session store; share it between all
                Workers of a test.
            cost_per_segment: Cost reported for each segment, in USD.
            think_seconds: Delay per segment, to leave time for a test to crash a Worker.
            engine_tools: Stand-ins for Claude Code tools (such as ``Bash``) that run
                as their own Activities: a call to a name in the agent's
                ``tool_activities`` runs the function with the call's input in a tool
                step. Others return ``{"ran": <name>}``. To fail the step as a real
                one can, raise ``ApplicationError`` of type ``TOOL_CALL_NOT_RUN``
                (the call did not run) or ``TOOL_CALL_INTERRUPTED`` (it may have).
        """
        self._policy = policy
        self._engine_tools = dict(engine_tools or {})
        self._dir = Path(state_dir) if state_dir is not None else None
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._cost = cost_per_segment
        self._think = think_seconds

    def _path(self, session_id: str, checkpoint: str) -> Path:
        assert self._dir is not None
        return self._dir / session_id / f"{checkpoint}.json"

    def _load(self, session_id: str, checkpoint: str) -> dict[str, Any]:
        path = self._path(session_id, checkpoint)
        if not path.exists():
            raise RuntimeError(
                f"Unknown checkpoint {checkpoint} of session {session_id}"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, session_id: str, state: dict[str, Any]) -> str:
        """Store the state as a new checkpoint (never changed afterwards)."""
        assert self._dir is not None
        checkpoint = f"cp_{uuid.uuid4().hex[:12]}"
        folder = self._dir / session_id
        folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(tmp, self._path(session_id, checkpoint))
        return checkpoint

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """Run one segment of the scripted conversation.

        Args:
            inp: The segment input.
            attempt: The Activity attempt number; a retry gets a new tool call id.

        Returns:
            A pause at the policy's next tool call, or its final answer.

        Raises:
            RuntimeError: If the conversation is kept where this runner does not keep
                it, or the Workflow did not serve it (Temporal retries).
        """
        entries: list[dict[str, Any]] = []
        if self._dir is None:
            entries = await read_conversation(inp)
            problem = self._held_problem(inp, entries)
            if problem is not None:
                return SegmentOutput(
                    session_id=inp.session_id, is_error=True, error=problem
                )
            state = _fold(entries) if entries else _new_state(inp.prompt or "")
        else:
            if inp.transcript or (
                inp.conversation is not None and inp.conversation.entries
            ):
                raise RuntimeError(
                    f"Session {inp.session_id} is kept in its Workflow, but this "
                    "ScriptedClaude keeps sessions in a folder. Retrying."
                )
            if inp.checkpoint is None:
                state = _new_state(inp.prompt or "")
            else:
                state = self._load(inp.session_id, inp.checkpoint)
        change: dict[str, Any] = {}  # what this segment adds to the conversation
        if inp.prompt is not None:
            state["prompt"] = inp.prompt  # a new task on the same session
            change["prompt"] = inp.prompt
        pending = _pending(state)
        missing = [call["id"] for call in pending if call["id"] not in inp.injected]
        if missing:
            return SegmentOutput(
                session_id=inp.session_id,
                is_error=True,
                error=f"The paused call {', '.join(missing)} got no result",
            )
        if pending:
            results = []
            for call in pending:
                outcome = _as_outcome(inp.injected[call["id"]])
                result = {"content": outcome.content, "is_error": outcome.is_error}
                state["history"].append({**call, **result})
                results.append(result)
            state["pending"] = []
            change["results"] = results
        if self._think:
            await asyncio.sleep(self._think)
        action = self._policy(
            state["prompt"], [HistoryItem(**h) for h in state["history"]]
        )
        state["turn"] += 1
        change["turn"] = state["turn"]
        calls: list[dict[str, Any]] = []
        answer: str | None = None
        if isinstance(action, Final):
            emit({"type": "text", "text": action.text})  # what Claude writes
            answer = action.text
            change["final"] = answer
        else:
            retry = f"_{attempt}" if attempt > 1 else ""
            durable = {t.name for t in inp.tools}
            for n, act in enumerate(action if isinstance(action, list) else [action]):
                letter = chr(ord("a") + n - 1) if n else ""  # the first id as before
                engine = act.name not in durable and any(
                    fnmatch.fnmatchcase(act.name, p) for p in inp.tool_activities
                )
                calls.append(
                    {
                        "id": (
                            f"toolu_{inp.session_id[:8]}_{state['turn']:02d}"
                            f"{letter}{retry}"
                        ),
                        "name": act.name,
                        "input": act.input,
                        "kind": "engine" if engine else "durable",
                    }
                )
            state["pending"] = calls
            change["calls"] = calls
        if self._dir is not None:
            checkpoint = self._save(inp.session_id, state)
        else:
            checkpoint = f"cp_{uuid.uuid4().hex[:12]}"
        out = SegmentOutput(
            session_id=inp.session_id,
            result=answer,
            deferred=DeferredCall(**calls[0]) if calls else None,
            checkpoint=checkpoint,
            cost_usd=self._cost,
            external_storage=external_storage_on(),
            siblings=[DeferredCall(**call) for call in calls[1:]],
        )
        if self._dir is None:
            out.transcript_keep = len(entries)
            out.transcript_add = [{"uuid": checkpoint, **change}]
        return out

    async def run_tool_step(self, step: ToolStepInput, attempt: int) -> ToolOutcome:
        """Run a Claude Code tool call that runs as its own Activity, with its stand-in.

        Args:
            step: The call.
            attempt: The Activity attempt number.

        Returns:
            The stand-in's result.
        """
        del attempt
        run = self._engine_tools.get(step.call.name)
        if run is None:
            return ToolOutcome({"ran": step.call.name})
        result = run(step.call.input)
        if asyncio.iscoroutine(result):
            result = await result
        return result if isinstance(result, ToolOutcome) else ToolOutcome(result)

    @staticmethod
    def _held_problem(inp: SegmentInput, entries: list[dict[str, Any]]) -> str | None:
        """Check that the conversation the Workflow holds ends at the checkpoint.

        Raises:
            RuntimeError: If the Workflow holds no conversation for a session that
                has a checkpoint: it is kept in a folder this runner does not have.
        """
        if inp.checkpoint is None:
            if entries:
                return "The Workflow holds a conversation but no checkpoint"
            return None
        if not entries:
            raise RuntimeError(
                f"The Workflow holds no conversation for session {inp.session_id} "
                f"(checkpoint {inp.checkpoint}): it is kept in a folder this "
                "ScriptedClaude does not have. Retrying."
            )
        if entries[-1].get("uuid") != inp.checkpoint:
            return (
                f"The conversation the Workflow holds ends at "
                f"{entries[-1].get('uuid')}, not at checkpoint {inp.checkpoint}"
            )
        return None


__all__ = ["Final", "HistoryItem", "Policy", "ScriptedClaude", "ToolCall"]
