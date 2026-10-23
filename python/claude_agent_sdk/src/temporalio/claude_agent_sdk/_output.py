"""Sends the CLI's stdout to its Workflow: a Signal for each message."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

from temporalio.client import WorkflowHandle
from temporalio.contrib.workflow_streams import WorkflowStreamClient

from ._payloads import CliOutput


class OutputSender:
    """Sends each line of the CLI's stdout to its Workflow as a :class:`CliOutput` Signal.

    With ``stream``, a stream_event line is published to its ``topic`` instead. :attr:`start` is the byte offset the Workflow has all of the
    output up to, which a retried Activity resumes from. It passes a published
    line only once the stream flushed it, so a retry may publish a line again,
    but never drops one.
    """

    def __init__(
        self,
        workflow: WorkflowHandle[Any, Any],
        offset: int,
        stream: WorkflowStreamClient | None = None,
        topic: str = "",
    ) -> None:
        """Send the output from byte ``offset``."""
        self._workflow = workflow
        self._stream = stream
        self._topic = stream.topic(topic) if stream is not None else None
        # The bytes after ``end`` that don't make a whole line yet.
        self._partial = b""
        # Lines up to ``end`` were sent or published; up to ``start``, Signalled.
        self.start = self.end = offset
        self._unflushed = False

    async def forward(self, stdout: AsyncIterator[bytes]) -> None:
        """Send ``stdout``'s lines as they come, until it ends."""
        async for chunk in stdout:
            await self.add(chunk)

    async def add(self, data: bytes) -> None:
        """Add ``data``, sending each line it completes."""
        *whole, self._partial = (self._partial + data).split(b"\n")
        for line in whole:
            await self._send(line)

    async def add_rest(self, data: bytes) -> None:
        """Add stdout re-read from byte ``end``, which repeats the partial line."""
        self._partial = b""
        await self.add(data)

    async def _send(self, line: bytes) -> None:
        end = self.end + len(line) + 1  # The newline.
        message = _parse(line)
        if self._topic is not None and message and message["type"] == "stream_event":
            self._topic.publish(message)
            self._unflushed = True
        else:
            if self._stream is not None and self._unflushed:
                # The offset is about to pass the published lines.
                await self._stream.flush()
                self._unflushed = False
            kind, label = _classify(message)
            text = line.decode(errors="replace")
            output = CliOutput(self.start, end, text, summary=label)
            await self._workflow.signal(kind, output)
            self.start = end
        self.end = end


def _parse(line: bytes) -> dict[str, Any] | None:
    """The message on ``line``, or ``None`` if it has none, such as a debug line."""
    try:
        message: Any = json.loads(line)
    except ValueError:
        return None
    if not isinstance(message, dict) or "type" not in message:
        return None
    return cast(dict[str, Any], message)


def _classify(message: dict[str, Any] | None) -> tuple[str, str]:
    """The kind of a stdout message, one of OUTPUT_KINDS, and a label for the Temporal UI."""
    if message is None:
        return "other", "not a message"
    kind = str(message.get("type"))
    if kind in ("control_request", "control_response"):
        body = message.get("request") or message.get("response") or {}
        # can_use_tool names its tool, and mcp_message its server.
        about = body.get("tool_name") or body.get("server_name")
        subtype = body.get("subtype")
        return kind, f"{kind}: {subtype} {about}" if about else f"{kind}: {subtype}"
    if kind == "control_cancel_request":
        return "control_request", kind
    if kind == "result":
        return "result", f"result: {message.get('subtype')}"
    if kind == "system":
        return "other", f"system: {message.get('subtype')}"
    if kind not in ("assistant", "user"):
        return "other", kind
    content = (message.get("message") or {}).get("content")
    blocks: list[dict[str, Any]] = (
        [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
    )
    tools = [str(b.get("name")) for b in blocks if b.get("type") == "tool_use"]
    if tools:
        return "tool_use", f"tool_use: {', '.join(tools)}"
    if any(b.get("type") == "tool_result" for b in blocks):
        return "tool_result", "tool_result"
    if kind == "user":
        return "other", kind
    return "assistant", f"assistant: {', '.join(str(b.get('type')) for b in blocks)}"
