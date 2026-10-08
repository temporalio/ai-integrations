"""Deterministic durable tools for native batch tests."""

from pathlib import Path
from typing import Any

from temporalio import activity


@activity.defn
async def record(arguments: dict[str, Any]) -> str:
    """Append the requested text and return it."""
    with Path(arguments["path"]).open("a") as stream:
        stream.write(arguments["text"] + "\n")
    return arguments["text"]
