"""The durable tool of the warm engine tests: echo a number, and count each run."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from temporalio import activity

RAN: defaultdict[str, list[int]] = defaultdict(list)
"""The numbers each Workflow echoed, in order: every call must run exactly once."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}


@activity.defn(name="echo")
async def echo(arguments: dict[str, Any]) -> dict[str, Any]:
    """Echo one number."""
    RAN[activity.info().workflow_id or ""].append(int(arguments["n"]))
    return {"n": arguments["n"]}
