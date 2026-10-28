"""Constants shared across the Claude Agent SDK integration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Final

STDIN_LABELS: Final[Mapping[str, str]] = {
    "user": "prompt",
    "control_request": "request",
    "control_response": "reply",
}

CONNECT: Final = "claude-agent-sdk-connect"
WRITE: Final = "claude-agent-sdk-write"
END_INPUT: Final = "claude-agent-sdk-end-input"
READ: Final = "claude-agent-sdk-read"
CLOSE: Final = "claude-agent-sdk-close"

WORKER_GONE_TIMEOUT: Final = timedelta(seconds=30)
