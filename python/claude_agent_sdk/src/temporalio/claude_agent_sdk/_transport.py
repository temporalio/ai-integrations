"""A Claude Agent SDK ``Transport`` that a Workflow uses to run Claude Code in a sandbox."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, Transport
from claude_agent_sdk.types import SDKControlResponse

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.contrib.workflow_streams import WorkflowTopicHandle
from temporalio.exceptions import ActivityError

from ._client import SandboxClient
from ._command import ClaudeCodeCommand
from ._payloads import (
    END_OF_INPUT,
    OUTPUT_KINDS,
    CliOutput,
    SandboxSession,
)

# How the Temporal UI names the stdin lines an Activity sends.
_STDIN_LABELS = {
    "user": "prompt",
    "control_request": "request",
    "control_response": "reply",
}


class TemporalTransport(Transport):
    """Runs Claude Code in a sandbox, over its own stdin and stdout.

    Construct it in a Workflow and pass it to the Claude Agent SDK's
    ``ClaudeSDKClient``. Each Activity gets the sandbox from its factory, on
    every attempt: a lost sandbox fails only the Activity that reached for it,
    and its retry asks the factory again. The transport fails only if an
    Activity fails for good.
    """

    def __init__(
        self,
        options: ClaudeAgentOptions,
        *,
        sandbox: str,
        task_queue: str | None = None,
        start_to_close_timeout: timedelta = timedelta(minutes=5),
        heartbeat_timeout: timedelta = timedelta(seconds=10),
        retry_policy: RetryPolicy | None = None,
        stream: WorkflowTopicHandle[Any] | None = None,
    ) -> None:
        """Prepare the transport. The sandbox starts on ``connect``.

        Args:
            options: The same options passed to ``ClaudeSDKClient``. The
                transport builds Claude Code's command line from them.
            sandbox: The name the sandbox provider is registered under on
                ``ClaudeAgentSDKPlugin``.
            task_queue: The task queue for the Activities. Defaults to the
                Workflow's.
            start_to_close_timeout: For the Activities that start the sandbox,
                launch Claude Code, write its input, and close the session.
                Starting a sandbox gets at least 30 minutes.
            heartbeat_timeout: How soon a lost Worker is detected, for every
                Activity.
            retry_policy: For every Activity. ``None`` means Temporal's default,
                which retries forever; watching the output then retries at
                most 5 seconds apart, since the Claude Agent SDK waits on it.
                Each retry builds the sandbox again with its factory.
            stream: A topic of the Workflow's ``WorkflowStream``, for
                ``options.include_partial_messages``. Each ``stream_event``
                goes to it, for observers outside the Workflow; the Workflow
                doesn't get them.

        Raises:
            ValueError: If ``options.include_partial_messages`` is set without
                ``stream``, or ``stream`` without it. Or if ``options.sandbox``
                is set with ``options.settings`` as a file path: the Claude
                Agent SDK merges the two by reading the file, which is in the
                sandbox, not the Workflow.
        """
        if options.include_partial_messages != (stream is not None):
            raise ValueError(
                "TemporalTransport needs a stream if and only if "
                "options.include_partial_messages is set"
            )
        self._command = ClaudeCodeCommand(options)
        self._plugins = [plugin["path"] for plugin in options.plugins]

        # One transport per Workflow: its output comes on these Signals.
        for kind in OUTPUT_KINDS:
            workflow.set_signal_handler(kind, self._on_output)
        self._sandbox = SandboxClient(
            sandbox=sandbox,
            first_execution_run_id=workflow.info().first_execution_run_id,
            task_queue=task_queue,
            start_to_close_timeout=start_to_close_timeout,
            heartbeat_timeout=heartbeat_timeout,
            retry_policy=retry_policy,
            stream_topic=stream.name if stream is not None else None,
        )

        # Messages for the CLI's stdin, and for the Claude Agent SDK.
        self._outbox: list[dict[str, Any]] = []
        self._inbox: deque[dict[str, Any]] = deque()
        # How far into the CLI's stdout the Workflow has the lines.
        self._stdout_offset = 0
        # The stdin files written, or being written.
        self._stdin_files = 0
        # The Claude Agent SDK's requests the CLI hasn't answered: its ID for each,
        # by the ID the CLI got. Its IDs are random, so the Workflow picks the ones
        # the CLI gets, and replay sends the same.
        self._sdk_ids: dict[str, str] = {}
        self._requests_sent = 0
        self._tasks: list[asyncio.Task[None]] = []
        # The connect Activity that starts the sandbox. Once it started, close
        # cancels it and ends the sandbox.
        self._connecting: asyncio.Task[SandboxSession] | None = None
        self._connected = False
        self._closed = False
        # The Claude Agent SDK called end_input.
        self._ended = False
        # The CLI's stdin got the end of the input.
        self._input_ended = False
        self._exited = False
        # The CLI exited: read_messages ends.
        self._finished = False
        self._failure: BaseException | None = None

    async def connect(self) -> None:
        """Start the sandbox and the CLI in it."""
        self._connected = True
        try:
            self._connecting = asyncio.create_task(
                self._sandbox.connect(self._command.env(), self._plugins)
            )
            sandbox = await self._connecting
            args = self._command.args(sandbox.config_dir)
            session = await self._sandbox.launch(sandbox, args)
        except BaseException:
            # The Claude Agent SDK closes a transport only once it connected.
            await self.close()
            raise
        self._tasks.append(asyncio.create_task(self._watch_output(session)))
        self._tasks.append(asyncio.create_task(self._write_stdin(session)))

    async def write(self, data: str) -> None:
        """Send the Claude Agent SDK's prompts, requests, and answers to the CLI."""
        for line in data.splitlines():
            if not line.strip():
                continue
            message: Any = json.loads(line)
            if message["type"] == "control_request":
                self._requests_sent += 1
                request_id = f"temporal_{self._requests_sent}"
                self._sdk_ids[request_id] = message["request_id"]
                message = {**message, "request_id": request_id}
            self._outbox.append(message)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """The CLI's messages, each once. Ends on close, or once the input has ended and the CLI is done."""
        while True:
            await workflow.wait_condition(
                lambda: (
                    bool(self._inbox)
                    or self._failure is not None
                    or self._closed
                    or self._finished
                )
            )
            if self._inbox:
                yield self._inbox.popleft()
            elif self._failure is not None:
                raise self._failure
            else:
                await self.close()
                return

    async def end_input(self) -> None:
        """End the CLI's stdin, as the Claude Agent SDK closes it: the CLI exits once it is done."""
        self._ended = True

    async def close(self) -> None:
        """Stop the CLI and end the sandbox. The session's storage stays."""
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            task.cancel()
        if self._connecting is not None:
            # Connecting can retry forever: cancel it, then end the sandbox.
            self._connecting.cancel()
            await asyncio.wait([self._connecting])
            await self._sandbox.close()

    def is_ready(self) -> bool:
        """Whether the transport is connected and not closed."""
        return self._connected and not self._closed

    def _on_output(self, output: CliOutput) -> None:
        """Handle the Signal with a line of the CLI's stdout: take each line once, in order.

        Takes only the next line. Any other is a line sent again, or one a stale
        Activity attempt sent ahead, which its retry sends again in order.
        """
        if output.start != self._stdout_offset:
            return
        self._stdout_offset = output.end
        try:
            message: Any = json.loads(output.line)
            kind = message["type"]
        except (ValueError, TypeError, KeyError):
            # Not a protocol message, such as "[SandboxDebug] ..." lines. The
            # Claude Agent SDK skips them too.
            return
        if kind == "control_response":
            self._on_response(message)
        else:
            self._inbox.append(message)

    def _on_response(self, message: SDKControlResponse) -> None:
        """Hand the CLI's answer to one of the Claude Agent SDK's requests back, under its ID."""
        response = message["response"]
        sdk_id = self._sdk_ids.pop(response["request_id"], None)
        if sdk_id is not None:
            self._inbox.append(
                {**message, "response": {**response, "request_id": sdk_id}}
            )

    async def _watch_output(self, session: SandboxSession) -> None:
        """Watch the CLI's output, even while idle, until it exits. Then read_messages ends.

        Claude Code can start a turn on its own, such as when a background task
        finishes.
        """
        try:
            await self._sandbox.watch_output(session, self._stdout_offset)
            self._finished = True
        except ActivityError as error:
            self._fail(error)
        finally:
            self._exited = True

    async def _write_stdin(self, session: SandboxSession) -> None:
        """Write the outbox to the CLI's stdin in order, then the end of input once the Claude Agent SDK ends it."""
        while True:
            await workflow.wait_condition(
                lambda: (
                    self._exited
                    or bool(self._outbox)
                    or (self._ended and not self._input_ended)
                )
            )
            if self._exited:
                return
            if self._outbox:
                messages, self._outbox = self._outbox, []
                text = "".join(f"{json.dumps(message)}\n" for message in messages)
                kinds: list[str] = [message["type"] for message in messages]
                labels = dict.fromkeys(_STDIN_LABELS.get(k, k) for k in kinds)
                summary = f"Send {', '.join(labels)}"
            else:
                text, self._input_ended = f"{END_OF_INPUT}\n", True
                summary = "End input"
            self._stdin_files += 1
            number = self._stdin_files
            try:
                await self._sandbox.write_stdin(session, number, text, summary)
            except ActivityError as error:
                self._fail(error)
                return

    def _fail(self, error: BaseException) -> None:
        """Fail the transport, unless it is closed: an Activity that close cancelled is no failure."""
        if not self._closed and self._failure is None:
            self._failure = error
