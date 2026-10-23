"""The workload of the hybrid benchmark (prototype/claude-agent-hybrid, 7798d31,
tests/hybrid), on this plugin's current API: one durable echo tool, 20 rounds."""

from __future__ import annotations

from typing import Any

from temporalio import activity

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "n": {"type": "integer"},
        "approval": {"type": "boolean"},
        "delay": {"type": "number", "minimum": 0},
    },
    "required": ["n"],
    "additionalProperties": False,
}


@activity.defn(name="echo")
async def segment_echo(arguments: dict[str, Any]) -> dict[str, int]:
    """Echo one number with no external side effects."""
    return {"n": arguments["n"]}
