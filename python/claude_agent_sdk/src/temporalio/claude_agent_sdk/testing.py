"""Offline testing for the Claude Agent SDK integration.

Script a turn and point the plugin at the fake CLI::

    from temporalio.claude_agent_sdk import ClaudeAgentPlugin
    from temporalio.claude_agent_sdk.testing import FakeClaudeCLI, text_turn

    cli = FakeClaudeCLI(tmp_path, turn=text_turn("Hello!"))
    plugin = ClaudeAgentPlugin(cli_path=cli.cli_path)

Builders return the stdout lines of (part of) one turn in Claude Code 2.1.294 wire order:
``text_turn``, ``tool_turn``, ``structured_turn`` and ``error_turn`` are whole turns, and the
``*_round`` builders, ``answer`` and ``result`` compose new ones. For crash and resume tests,
``FakeClaudeCLI(dir, plans=...)`` models the CLI's session persistence and ``crash_at``
(``before_prompt_persisted``, ``after_tool_call``, ``after_tool_run`` or ``mid_answer``) stops it
where a test kills the worker.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "FAKE_CLI_SCRIPT",
    "MODEL",
    "SESSION",
    "START",
    "FakeClaudeCLI",
    "answer",
    "error_turn",
    "message_end",
    "message_start",
    "native_round",
    "result",
    "structured_turn",
    "text_round",
    "text_turn",
    "tool_round",
    "tool_turn",
]

FAKE_CLI_SCRIPT = Path(__file__).with_name("_fake_cli.py")
"""The fake CLI itself: ``python _fake_cli.py <scenario.json> <claude args...>``."""


class FakeClaudeCLI:
    """A scripted Claude Code CLI for offline tests; pass ``cli_path`` to the plugin.

    ``scenario`` is written as JSON next to an executable wrapper in ``directory``: ``turn`` (the
    stdout lines after any prompt), ``turns`` (``[[prompt prefix, turn], ...]``), and the keys
    documented in ``_fake_cli.py`` (``log``, ``effects``, ``state``, ``exit_code``, ...). The fake
    writes ``{"pid", "argv"}`` of its last run to ``record_path``. POSIX only, like the CLI.
    """

    def __init__(self, directory: str | Path, **scenario: Any) -> None:
        """Write the scenario and an executable wrapper under ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.record_path = directory / "record.json"
        scenario_file = directory / "scenario.json"
        scenario_file.write_text(
            json.dumps({"record": str(self.record_path), **scenario})
        )
        wrapper = directory / "claude"
        command = shlex.join([sys.executable, str(FAKE_CLI_SCRIPT), str(scenario_file)])
        wrapper.write_text(f'#!/bin/sh\nexec {command} "$@"\n')
        wrapper.chmod(0o755)
        self.cli_path = str(wrapper)


SESSION = "11111111-2222-4333-8444-555555555555"
MODEL = "claude-haiku-5-5"
"""The model id the CLI reports in ``message_start`` (requested as the alias ``haiku``)."""


def _uuid(tag: str) -> str:
    return f"se-{tag}"


def stream(event: dict[str, Any], *, parent: str | None = None) -> dict[str, Any]:
    return {
        "type": "stream_event",
        "event": event,
        "session_id": SESSION,
        "parent_tool_use_id": parent,
        "uuid": _uuid(str(id(event))),
    }


def _usage(input_tokens: int, output_tokens: int, **extra: Any) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": output_tokens,
        "service_tier": "standard",
        **extra,
    }


def message_start(message_id: str, input_tokens: int, **usage: Any) -> dict[str, Any]:
    """Return a ``message_start`` stream event for one model round."""
    return stream(
        {
            "type": "message_start",
            "message": {
                "model": MODEL,
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "stop_reason": None,
                "usage": _usage(input_tokens, 1, **usage),
            },
        }
    )


def message_end(
    stop_reason: str,
    input_tokens: int,
    output_tokens: int,
    *,
    thinking_tokens: int = 0,
    **usage: Any,
) -> list[dict[str, Any]]:
    """The round's `message_delta` (its real usage) and `message_stop`."""
    return [
        stream(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {
                    **_usage(input_tokens, output_tokens, **usage),
                    "output_tokens_details": {"thinking_tokens": thinking_tokens},
                },
            }
        ),
        stream({"type": "message_stop"}),
    ]


def assistant(
    message_id: str | None,
    content: list[dict[str, Any]],
    *,
    error: str | None = None,
    parent: str | None = None,
    input_tokens: int = 0,
) -> dict[str, Any]:
    """A completed message for ONE content block, with the stale `message_start` usage."""
    return {
        "type": "assistant",
        "message": {
            "model": MODEL if error is None else "<synthetic>",
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "stop_reason": None,
            "usage": _usage(input_tokens, 1),
        },
        "parent_tool_use_id": parent,
        "session_id": SESSION,
        "uuid": f"as-{message_id}-{len(content)}",
        **({"error": error} if error else {}),
    }


def result(
    *,
    text: str | None = "done",
    is_error: bool = False,
    subtype: str = "success",
    structured_output: Any = None,
    terminal_reason: str = "completed",
    stop_reason: str = "end_turn",
    **extra: Any,
) -> dict[str, Any]:
    """Return a terminal ``result`` event for one turn."""
    return {
        "type": "result",
        "subtype": subtype,
        "duration_ms": 1200,
        "duration_api_ms": 1100,
        "is_error": is_error,
        "num_turns": 2,
        "session_id": SESSION,
        "stop_reason": stop_reason,
        "result": text,
        "structured_output": structured_output,
        "terminal_reason": terminal_reason,
        "total_cost_usd": 0.0123,
        "usage": {
            **_usage(50, 27, cache_read_input_tokens=5),
            "output_tokens_details": {"thinking_tokens": 4},
        },
        "permission_denials": [],
        "uuid": "res-1",
        **extra,
    }


SYSTEM_INIT = {
    "type": "system",
    "subtype": "init",
    "cwd": f"/workspace/{SESSION}",
    "session_id": SESSION,
    "tools": [],
    "mcp_servers": [],
    "model": MODEL,
    "permissionMode": "dontAsk",
    "apiKeySource": "none",
    "claude_code_version": "2.1.294",
    "uuid": "sys-init",
}
SESSION_RUNNING = {
    "type": "system",
    "subtype": "session_state_changed",
    "state": "running",
    "sdk_host_only": True,
    "session_id": SESSION,
    "uuid": "sys-running",
}
STATUS_REQUESTING = {
    "type": "system",
    "subtype": "status",
    "status": "requesting",
    "session_id": SESSION,
    "uuid": "sys-status",
}
START = [SESSION_RUNNING, SYSTEM_INIT]
"""What the CLI writes before the first round of every turn."""


def _text_block(
    message_id: str, index: int, chunks: list[str], input_tokens: int
) -> list[dict[str, Any]]:
    return [
        stream(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        *[
            stream(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": c},
                }
            )
            for c in chunks
        ],
        assistant(
            message_id,
            [{"type": "text", "text": "".join(chunks)}],
            input_tokens=input_tokens,
        ),
        stream({"type": "content_block_stop", "index": index}),
    ]


def _tool_block(
    message_id: str, index: int, tool_use_id: str, name: str, tool_input: dict[str, Any]
) -> list[dict[str, Any]]:
    block = {"type": "tool_use", "id": tool_use_id, "name": name, "input": {}}
    return [
        stream({"type": "content_block_start", "index": index, "content_block": block}),
        stream(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": ""},
            }
        ),
        stream(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(tool_input),
                },
            }
        ),
        {
            **assistant(message_id, [{**block, "input": tool_input}]),
            "uuid": f"as-{tool_use_id}",
        },
        stream({"type": "content_block_stop", "index": index}),
    ]


def text_round(
    message_id: str, chunks: list[str], *, input_tokens: int, output_tokens: int
):
    """One model round that only writes text."""
    return [
        STATUS_REQUESTING,
        message_start(message_id, input_tokens),
        *_text_block(message_id, 0, chunks, input_tokens),
        *message_end("end_turn", input_tokens, output_tokens),
    ]


def tool_round(
    message_id: str, calls: list[tuple[str, str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """One model round that calls workflow tools in parallel. ``calls`` are (tool_use_id,
    name, input). Each call's `tools/call` goes out as soon as its block completes, mid-message,
    and the fake CLI emits the tool results after `message_stop`.
    """
    lines: list[dict[str, Any]] = [STATUS_REQUESTING, message_start(message_id, 9)]
    for index, (tool_use_id, name, tool_input) in enumerate(calls):
        lines += _tool_block(
            message_id, index, tool_use_id, f"mcp__temporal__{name}", tool_input
        )
        lines.append(
            {
                "fake": "mcp_calls",
                "calls": [{"id": tool_use_id, "name": name, "arguments": tool_input}],
            }
        )
    return [*lines, *message_end("tool_use", 9, 5)]


def native_round(
    message_id: str,
    tool_use_id: str,
    name: str,
    tool_input: dict[str, Any],
    **call: Any,
):
    """One model round that says a line, then calls one native tool, then the fake CLI's
    directive to gate (and maybe run) it.
    """
    text = _text_block(message_id, 0, ["Running it."], 9)
    return [
        STATUS_REQUESTING,
        message_start(message_id, 9),
        *text,
        *_tool_block(message_id, 1, tool_use_id, name, tool_input),
        *message_end("tool_use", 9, 5),
        {
            "fake": "native_call",
            "id": tool_use_id,
            "name": name,
            "input": tool_input,
            **call,
        },
    ]


def answer(text: str, message_id: str = "msg_answer") -> list[dict[str, Any]]:
    """A closing text round and the result."""
    return [
        *text_round(message_id, [text], input_tokens=20, output_tokens=3),
        result(text=text),
    ]


def tool_turn(
    calls: list[tuple[str, str, dict[str, Any]]], *, answer: str = "All done."
) -> list[dict[str, Any]]:
    """A whole turn: one parallel tool round, then a text answer and the result."""
    return [
        *START,
        *tool_round("msg_tools_{{P}}", calls),
        *text_round("msg_answer_{{P}}", [answer], input_tokens=20, output_tokens=3),
        result(text=answer),
    ]


def text_turn(text: str) -> list[dict[str, Any]]:
    """A whole turn that answers ``text``."""
    return [*START, *answer(text)]  # type: ignore


def structured_turn(output: dict[str, Any]) -> list[dict[str, Any]]:
    """A whole turn answering an output schema the way Claude Code does: a synthetic
    ``StructuredOutput`` tool call and its tool result, with ``output`` as the structured
    output and its JSON as the result text.
    """
    return [
        *START,  # type: ignore
        STATUS_REQUESTING,
        message_start("msg_s", 12),
        *_tool_block("msg_s", 0, "toolu_so", "StructuredOutput", output),
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "toolu_so",
                        "type": "tool_result",
                        "content": "Structured output provided successfully",
                    }
                ],
            },
            "parent_tool_use_id": None,
            "session_id": SESSION,
            "uuid": "u-so",
        },
        *message_end("tool_use", 12, 9),
        result(
            text=json.dumps(output, separators=(",", ":")),
            structured_output=output,
            stop_reason="tool_use",
        ),
    ]


def error_turn(
    message: str = "API Error: 529 overloaded", *, status: int = 529
) -> list[dict[str, Any]]:
    """A whole turn failing the way Claude Code reports an API error: a synthesized assistant
    message carrying ``error``, then an ``is_error`` result. Pair it with ``exit_code=1`` in
    the scenario, as the real CLI then exits non-zero.
    """
    return [
        *START,  # type: ignore
        assistant(None, [{"type": "text", "text": message}], error="server_error"),
        result(
            text=message,
            is_error=True,
            terminal_reason="api_error",
            api_error_status=status,
        ),
    ]
