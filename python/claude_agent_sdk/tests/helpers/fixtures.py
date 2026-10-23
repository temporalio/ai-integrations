"""Stream-json fixtures and the hook-script helpers shared by the native-tool tests.

Builders live in :mod:`temporalio.claude_agent_sdk.testing`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import pytest

from temporalio.claude_agent_sdk import _native
from temporalio.claude_agent_sdk._models import ClaudeTurnInput
from temporalio.claude_agent_sdk.testing import (
    MODEL,
    SESSION,
    START,
    STATUS_REQUESTING,
    SYSTEM_INIT,
    FakeClaudeCLI,
    answer,
    assistant,
    error_turn,
    message_end,
    message_start,
    native_round,
    result,
    stream,
    structured_turn,
    text_round,
    tool_round,
    tool_turn,
)

__all__ = [
    "API_ERROR",
    "COMPLETED_ONLY",
    "INTERRUPTED_RESULT",
    "MODEL",
    "POSIX_ONLY",
    "POSIX_ONLY_REASON",
    "SESSION",
    "START",
    "STATUS_REQUESTING",
    "STREAMING_UNTIL_INTERRUPT",
    "STRUCTURED",
    "SYSTEM_INIT",
    "TOOL_ROUND_TRIP",
    "answer",
    "assistant",
    "completed_cli",
    "error_turn",
    "hook_script",
    "hook_state",
    "hook_verdict",
    "message_end",
    "message_start",
    "native_round",
    "result",
    "stream",
    "structured_turn",
    "text_round",
    "tool_round",
    "tool_turn",
]


def hook_script(
    state_dir: Path, tool_use_id: Any, tool_input: Any = None, **extra: Any
) -> subprocess.CompletedProcess[str]:
    """Run the command hook once, the way the CLI would."""
    data = {
        "hook_event_name": "PreToolUse",
        "tool_name": extra.pop("tool_name", "Bash"),
        "tool_input": tool_input if tool_input is not None else {"command": "ls"},
        "tool_use_id": tool_use_id,
        **extra,
    }
    return subprocess.run(
        [sys.executable, "-I", "-S", str(_native.HOOK_SCRIPT), str(state_dir)],
        input=json.dumps(data),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def hook_verdict(proc: subprocess.CompletedProcess[str]) -> tuple[str, str] | None:
    """The permission decision the hook printed, or None when it made none."""
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout:
        return None
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    return out["permissionDecision"], out["permissionDecisionReason"]


def hook_state(tmp_path: Path, **input: Any) -> Path:
    """A native-tool state directory prepared the way the Activity prepares one."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    input.setdefault("native_tools", ["Bash"])
    _native.prepare_state(
        state_dir, ClaudeTurnInput(prompt="p", session_id=str(uuid.uuid4()), **input)
    )
    return state_dir


def completed_cli(tmp_path: Path) -> FakeClaudeCLI:
    """A fake CLI under ``tmp_path / "cli"`` that answers every prompt with one message."""
    return FakeClaudeCLI(tmp_path / "cli", turn=COMPLETED_ONLY)


POSIX_ONLY_REASON = (
    "the fake CLI, its process groups, and POSIX file modes do not exist on Windows"
)
POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason=POSIX_ONLY_REASON)

RATE_LIMIT = {
    "type": "rate_limit_event",
    "rate_limit_info": {
        "status": "allowed",
        "resetsAt": 1793491200,
        "rateLimitType": "overage",
        "overageStatus": "allowed",
        "overageResetsAt": 1793491200,
        "isUsingOverage": False,
        "overageInUse": True,
    },
    "session_id": SESSION,
    "uuid": "rl-1",
}

# Dict literals of different shapes infer as list[object].
_START = cast(list[dict[str, Any]], START)

# Two model rounds: thinking + text + one tool call, then a text answer. Interleaved with
# messages the observer must drop: system init/status/session state, a rate-limit event, the tool
# result, an unknown stream event, and a Claude-internal subagent's text (parent_tool_use_id set).
TOOL_ROUND_TRIP: list[dict[str, Any]] = [
    *_START,
    STATUS_REQUESTING,
    message_start("msg_1", 10, cache_read_input_tokens=5),
    stream(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        }
    ),
    stream(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "User wants "},
        }
    ),
    RATE_LIMIT,
    stream(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "weather."},
        }
    ),
    stream(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        }
    ),
    assistant(
        "msg_1",
        [{"type": "thinking", "thinking": "User wants weather.", "signature": "sig"}],
    ),
    stream({"type": "content_block_stop", "index": 0}),
    stream(
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        }
    ),
    stream(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Let me "},
        }
    ),
    stream({"type": "ping"}),
    stream(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "check."},
        }
    ),
    assistant("msg_1", [{"type": "text", "text": "Let me check."}]),
    stream({"type": "content_block_stop", "index": 1}),
    stream(
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {},
            },
        }
    ),
    stream(
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        }
    ),
    stream(
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": ' "Paris"}'},
        }
    ),
    assistant(
        "msg_1",
        [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ],
    ),
    stream({"type": "content_block_stop", "index": 2}),
    *message_end("tool_use", 10, 20, thinking_tokens=6, cache_read_input_tokens=5),
    {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}
            ],
        },
        "parent_tool_use_id": None,
        "session_id": SESSION,
        "uuid": "u-1",
    },
    stream(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "SUBAGENT TEXT"},
        },
        parent="toolu_task",
    ),
    assistant(
        "msg_sub", [{"type": "text", "text": "SUBAGENT TEXT"}], parent="toolu_task"
    ),
    {"type": "system", "subtype": "some_future_subtype", "session_id": SESSION},
    *text_round("msg_2", ["It's ", "sunny."], input_tokens=40, output_tokens=7),
    result(text="Let me check.It's sunny."),
]

# A model round whose stream carries no text deltas (only the completed message) — the
# completed text is the fallback and must be published exactly once.
COMPLETED_ONLY: list[dict[str, Any]] = [
    *_START,
    STATUS_REQUESTING,
    message_start("msg_c", 3),
    assistant(
        "msg_c",
        [{"type": "text", "text": "Hello from the completed message."}],
        input_tokens=3,
    ),
    *message_end("end_turn", 3, 6),
    result(text="Hello from the completed message."),
]

API_ERROR = error_turn()
"""Claude Code's shape for an API failure; the real CLI then exits 1 (a ResultError)."""

STRUCTURED = structured_turn({"city": "Paris", "temp_f": 72})
"""Recorded shape: with --json-schema the reply is a synthetic `StructuredOutput` tool call."""

# The first half of a turn; the fake CLI then waits for an interrupt and answers with this result.
STREAMING_UNTIL_INTERRUPT: list[dict[str, Any]] = [
    *_START,
    STATUS_REQUESTING,
    message_start("msg_i", 8),
    stream(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    ),
    stream(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Partial"},
        }
    ),
]
INTERRUPTED_RESULT = result(
    text=None,
    subtype="error_during_execution",
    is_error=True,
    terminal_reason="aborted_streaming",
)
