"""The default connector: Claude Code in a subprocess of the Worker, run by the Claude Agent SDK's own transport."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKError, CLIConnectionError
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from ._types import ConnectActivityInput, Output


class SubprocessConnector:
    """Run Claude Code in a subprocess of the Worker, as the Claude Agent SDK's ``SubprocessCLITransport`` does."""

    async def connect(self, request: ConnectActivityInput) -> _SubprocessConnection:
        """Start Claude Code."""
        connection = _SubprocessConnection(request)
        await connection.connect()
        return connection


@dataclass
class _Exit:
    """Claude Code exited: with status 0, or with ``error``."""

    error: ClaudeSDKError | None


class _SubprocessConnection(SubprocessCLITransport):
    """The Claude Agent SDK's transport, run with the flags the Workflow built.

    Stdout messages and stderr lines go into one queue in the order they
    arrive. A read returns what is queued.
    """

    def __init__(self, request: ConnectActivityInput) -> None:
        self._args = request.args
        self._output: asyncio.Queue[Output | _Exit] = asyncio.Queue()

        options = ClaudeAgentOptions(
            cli_path=request.cli_path,
            cwd=request.cwd,
            env=request.env,
            user=request.user,
            max_buffer_size=request.max_buffer_size,
            stderr=self._output.put_nowait,
        )

        super().__init__(prompt="", options=options)

    def _build_command(self) -> list[str]:
        if self._cli_path is None:
            raise RuntimeError(
                "connect() builds the command only after finding Claude Code"
            )

        return [self._cli_path, *self._args]

    async def connect(self) -> None:
        await super().connect()
        self._pump = asyncio.create_task(self._copy_stdout())

    async def _copy_stdout(self) -> None:
        try:
            async for message in self.read_messages():
                self._output.put_nowait(message)
            ended = _Exit(None)
        except ClaudeSDKError as error:
            ended = _Exit(error)
        except Exception as error:
            # Such as invalid UTF-8, which the Claude Agent SDK lets through.
            ended = _Exit(
                CLIConnectionError(f"Failed to read Claude Code's output: {error}")
            )

        # Stdout can close before the last stderr lines are read.
        if self._stderr_task is not None:
            await self._stderr_task.wait()
        self._output.put_nowait(ended)

    async def read(self) -> list[Output]:
        output: list[Output] = []
        item = await self._output.get()
        while not isinstance(item, _Exit):
            output.append(item)
            if self._output.empty():
                return output
            item = self._output.get_nowait()

        # Claude Code exited. Keep the exit queued so every later read sees it,
        # and raise its error only once the output before it is returned.
        self._output.put_nowait(item)
        if item.error is not None and not output:
            raise item.error
        return output

    async def close(self) -> None:
        # The pump keeps draining stdout while Claude Code wraps up.
        await super().close()
        self._pump.cancel()
        await asyncio.gather(self._pump, return_exceptions=True)
