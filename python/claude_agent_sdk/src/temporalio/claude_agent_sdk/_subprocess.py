"""The default connector: Claude Code in a subprocess of the Worker, run by the Claude Agent SDK's own transport."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKError
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from temporalio.service import PayloadLimitsConfig

from ._types import ConnectActivityInput, Output


class SubprocessConnector:
    """Run Claude Code in a subprocess of the Worker, as the Claude Agent SDK's ``SubprocessCLITransport`` does."""

    async def connect(self, request: ConnectActivityInput) -> _SubprocessConnection:
        """Start Claude Code."""
        connection = _SubprocessConnection(request)
        await connection.connect()
        return connection


class _SubprocessConnection(SubprocessCLITransport):
    """The Claude Agent SDK's transport, run with the flags the Workflow built, its stdout read straight from the pipe."""

    def __init__(self, request: ConnectActivityInput) -> None:
        self._args = request.args
        self._next: asyncio.Future[dict[str, Any] | None] | None = None
        self._reading = asyncio.Lock()

        # The Claude Agent SDK reads stderr on its own task and passes each line
        # to this callback as it arrives, not when a read asks for it. Keep the
        # lines until the next read returns them, or the Workflow's stderr
        # callback misses them, including why Claude Code failed.
        self._stderr: list[Output] = []

        options = ClaudeAgentOptions(
            cli_path=request.cli_path,
            cwd=request.cwd,
            env=request.env,
            user=request.user,
            max_buffer_size=request.max_buffer_size,
            stderr=self._stderr.append,
        )

        super().__init__(prompt="", options=options)

        # Runs nothing until the first read asks for a message.
        self._messages = self.read_messages()

    def _build_command(self) -> list[str]:
        if self._cli_path is None:
            raise RuntimeError(
                "connect() builds the command only after finding Claude Code"
            )

        return [self._cli_path, *self._args]

    async def read(self) -> list[Output]:
        async with self._reading:
            output: list[Output] = []
            size = 2  # The surrounding JSON list.
            limit = PayloadLimitsConfig().payloads_warn_size

            def fits(item: Output) -> bool:
                nonlocal size
                next_size = size + len(json.dumps(item).encode())
                if output:
                    next_size += 2  # The JSON comma and space.
                if output and next_size > limit:
                    return False
                size = next_size
                return True

            try:
                # Shielded, so a cancelled attempt leaves the message to its retry.
                message = await asyncio.shield(self._pending())
            except ClaudeSDKError:
                # Claude Code failed. Return its stderr first, if any; the next
                # read raises it again.
                if not self._stderr:
                    raise
                message = None

            while self._stderr and fits(self._stderr[0]):
                output.append(self._stderr.pop(0))

            if not self._stderr:
                try:
                    while message is not None and fits(message):
                        self._next = None
                        output.append(message)
                        message = await self._next_message()
                except ClaudeSDKError:
                    # Claude Code failed. The next read raises it again, after this
                    # one returns what came before it.
                    pass

            return output

    async def close(self) -> None:
        pending = self._next
        if pending is not None:
            pending.cancel()

        try:
            await super().close()
        finally:
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)

    def _pending(self) -> asyncio.Future[dict[str, Any] | None]:
        """Claude Code's next message, read from the pipe once and kept until a read returns it."""
        if self._next is None:
            self._next = asyncio.ensure_future(anext(self._messages, None))
        return self._next

    async def _next_message(self) -> dict[str, Any] | None:
        """Claude Code's next message if it already wrote it, else None."""
        pending = self._pending()
        done, _ = await asyncio.wait([pending], timeout=0)
        return pending.result() if done else None
