"""A Claude Agent SDK ``Transport`` for Workflows, which reaches Claude Code through Activities."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    CLIConnectionError,
    Transport,
)
from claude_agent_sdk import __version__ as sdk_version
from claude_agent_sdk._internal.transport.subprocess_cli import (
    _SDK_READS_SESSION_STATE_ENV,
    SubprocessCLITransport,
)
from claude_agent_sdk.types import _configure_can_use_tool

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.contrib.workflow_streams import WorkflowTopicHandle
from temporalio.exceptions import (
    ActivityError,
    is_cancelled_exception,
)

from ._constants import (
    CLOSE,
    CONNECT,
    CONNECTION_LOST,
    END_INPUT,
    READ,
    STDIN_LABELS,
    WORKER_GONE_TIMEOUT,
    WRITE,
)
from ._errors import sdk_error
from ._types import (
    CloseActivityInput,
    ConnectActivityInput,
    Connection,
    EndInputActivityInput,
    Output,
    ReadActivityInput,
    ReadResult,
    TransportActivityOptions,
    WriteActivityInput,
)


class TemporalTransport(Transport):
    """A Claude Agent SDK ``Transport`` for Workflows.

    It raises the same errors as the Claude Agent SDK's subprocess transport. If
    the Worker that reaches Claude Code is gone, it raises ``CLIConnectionError``;
    to continue the conversation, start a new client with ``options.resume`` set
    to the session ID.
    """

    def __init__(
        self,
        options: ClaudeAgentOptions,
        *,
        activity_options: TransportActivityOptions | None = None,
        stream: WorkflowTopicHandle[Any] | None = None,
    ) -> None:
        """Prepare the transport. Claude Code starts on ``connect``.

        Args:
            options: The same options passed to ``ClaudeSDKClient``.
            activity_options: Options for each Activity.
            stream: A topic of the Workflow's ``WorkflowStream``.

        Raises:
            ValueError: If ``options.include_partial_messages`` is set without
                ``stream``, or ``stream`` without it.
        """
        if options.include_partial_messages != (stream is not None):
            raise ValueError(
                f"{type(self).__name__} needs a stream if and only if "
                "options.include_partial_messages is set"
            )

        if not _is_json_object(options.settings):
            raise ValueError(
                f"{type(self).__name__} needs options.settings to be a JSON "
                "object, not a file path"
            )

        self._options = options
        self._activity_options: TransportActivityOptions = activity_options or {}

        self._connection: Connection | None = None

        self._received = 0

        # The Worker connected to Claude Code shut down or went away. Skip any
        # Activity on its task queue rather than wait out its timeout.
        self._connection_lost = False

        self._stream = stream

        self._write_lock = asyncio.Lock()

        # The Claude Agent SDK picks random request IDs, which replay wouldn't
        # reproduce. Send IDs from workflow.uuid4 instead, and map each back to
        # the Claude Agent SDK's when the CLI answers.
        self._sdk_ids: dict[str, str] = {}

    async def connect(self) -> None:
        """Start Claude Code."""
        options: dict[str, Any] = {
            "summary": "Start Claude Code",
            "start_to_close_timeout": timedelta(minutes=5),
            "heartbeat_timeout": timedelta(seconds=10),
            **self._activity_options.get("connect", {}),
        }

        try:
            self._connection = await workflow.execute_activity(
                CONNECT, self._connect_input(), result_type=Connection, **options
            )
        except ActivityError as error:
            raise self._sdk_error(error)  # noqa: B904 - sdk_error sets the cause.

    async def write(self, data: str) -> None:
        """Send the Claude Agent SDK's prompts, requests, and answers to the CLI."""
        async with self._write_lock:
            if self._connection is None:
                raise CLIConnectionError("ProcessTransport is not ready for writing")

            message = json.loads(data)
            kind: str = message["type"]

            # The request id needs to be swapped for a deterministic one for Replay
            if kind == "control_request":
                request_id = str(workflow.uuid4())
                self._sdk_ids[request_id] = message["request_id"]
                data = f"{json.dumps({**message, 'request_id': request_id})}\n"
            elif kind == "user" and "uuid" not in message:
                # Claude Code skips a prompt whose uuid it already has, so a
                # retried write doesn't run it again.
                data = f"{json.dumps({**message, 'uuid': str(workflow.uuid4())})}\n"

            options: dict[str, Any] = {
                "start_to_close_timeout": timedelta(minutes=5),
                "heartbeat_timeout": timedelta(seconds=10),
                "summary": f"Send {STDIN_LABELS.get(kind, kind)}",
                "schedule_to_start_timeout": WORKER_GONE_TIMEOUT,
                **self._activity_options.get("write", {}),
                "task_queue": self._connection.task_queue,
            }
            await self._execute(
                WRITE, WriteActivityInput(self._connection.handle, data), options
            )

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield each of the CLI's messages once until it exits, including turns Claude Code starts on its own."""
        if self._connection is None:
            raise CLIConnectionError("Not connected")

        options: dict[str, Any] = {
            "start_to_close_timeout": timedelta(minutes=5),
            "retry_policy": RetryPolicy(backoff_coefficient=1.0),
            "heartbeat_timeout": timedelta(seconds=10),
            "summary": "Read output",
            "schedule_to_start_timeout": WORKER_GONE_TIMEOUT,
            **self._activity_options.get("read", {}),
            "task_queue": self._connection.task_queue,
        }

        while True:
            result: ReadResult = await self._execute(
                READ,
                ReadActivityInput(self._connection.handle, self._received),
                options,
                result_type=ReadResult,
            )

            if not result.output:  # Claude Code has stopped
                break

            self._received += len(result.output)

            for message in self._handle_output(result.output):
                yield message

    async def end_input(self) -> None:
        """End the CLI's stdin, after the writes before it."""
        async with self._write_lock:
            if self._connection is None:
                return

            options: dict[str, Any] = {
                "start_to_close_timeout": timedelta(minutes=5),
                "heartbeat_timeout": timedelta(seconds=10),
                "summary": "End input",
                "schedule_to_start_timeout": WORKER_GONE_TIMEOUT,
                **self._activity_options.get("end_input", {}),
                "task_queue": self._connection.task_queue,
            }

            await self._execute(
                END_INPUT, EndInputActivityInput(self._connection.handle), options
            )

    async def close(self) -> None:
        """Stop the CLI and everything it started."""
        if self._connection is None:
            return

        options: dict[str, Any] = {
            "start_to_close_timeout": timedelta(minutes=5),
            "heartbeat_timeout": timedelta(seconds=10),
            "summary": "Close Claude Code",
            "schedule_to_start_timeout": WORKER_GONE_TIMEOUT,
            **self._activity_options.get("close", {}),
            "task_queue": self._connection.task_queue,
        }
        if not self._connection_lost:
            try:
                await workflow.execute_activity(
                    CLOSE, CloseActivityInput(self._connection.handle), **options
                )
            except ActivityError as error:
                if is_cancelled_exception(error):
                    raise
        # Only now, so a close that was cancelled can run again.
        self._connection = None

    async def _execute(
        self, activity: str, arg: Any, options: dict[str, Any], **result: Any
    ) -> Any:
        """Run an Activity on the connected Worker, unless the connection is lost."""
        if self._connection_lost:
            raise CLIConnectionError(CONNECTION_LOST)
        try:
            return await workflow.execute_activity(activity, arg, **result, **options)
        except ActivityError as error:
            raise self._sdk_error(error)  # noqa: B904 - sdk_error sets the cause.

    def _sdk_error(self, error: ActivityError) -> BaseException:
        """The Claude Agent SDK's error for a failed Activity, noting whether the connection to Claude Code is lost.

        A cancellation stays as it is.
        """
        if is_cancelled_exception(error):
            return error
        sdk = sdk_error(error)
        if isinstance(sdk, CLIConnectionError) and str(sdk) == CONNECTION_LOST:
            self._connection_lost = True
        return sdk

    def is_ready(self) -> bool:
        """Whether Claude Code started and the transport isn't closed."""
        return self._connection is not None

    def _handle_output(self, outputs: list[Output]) -> Iterator[dict[str, Any]]:
        """Handle a batch of the CLI's output, and yield the messages in it."""
        for output in outputs:
            if isinstance(output, str):
                if self._options.stderr is not None:
                    self._options.stderr(output)
                continue

            kind = output.get("type")
            if kind == "stream_event" and self._stream is not None:
                self._stream.publish(output)
                continue

            if kind != "control_response":
                yield output
                continue

            # Swap the deterministic request ID back to the Claude Agent SDK's
            response = output["response"]
            sdk_id = self._sdk_ids.pop(response["request_id"], None)
            if sdk_id is None:
                raise RuntimeError(
                    f"Claude Code answered unknown request {response['request_id']}"
                )
            yield {**output, "response": {**response, "request_id": sdk_id}}

    def _connect_input(self) -> ConnectActivityInput:
        """Claude Code's command line and environment, built from the options as in the Claude Agent SDK."""
        options = _configure_can_use_tool(self._options)
        # The Claude Agent SDK builds the command line only inside its
        # subprocess transport, so that code is borrowed here. Replace this once
        # the Python Claude Agent SDK has a hook like the TypeScript one's
        # spawnClaudeCodeProcess. It needs a CLI path, which only the connector
        # can find, so it gets a placeholder that is then dropped.
        placeholder = dataclasses.replace(options, cli_path="claude")
        args = SubprocessCLITransport(prompt="", options=placeholder)._build_command()[
            1:
        ]

        env = {
            "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
            **options.env,
            "CLAUDE_AGENT_SDK_VERSION": sdk_version,
        }

        # As in the Claude Agent SDK, set it unless options.env has it, in any
        # letter case. It makes Claude Code emit the session_state_changed
        # messages the Claude Agent SDK uses to tell when a run is over.
        if not any(key.upper() == _SDK_READS_SESSION_STATE_ENV for key in env):
            env[_SDK_READS_SESSION_STATE_ENV] = "1"

        if options.enable_file_checkpointing:
            env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"

        return ConnectActivityInput(
            args,
            env,
            cli_path=str(options.cli_path) if options.cli_path else None,
            cwd=str(options.cwd) if options.cwd else None,
            user=options.user,
            max_buffer_size=options.max_buffer_size,
        )


def _is_json_object(settings: str | None) -> bool:
    """Whether ``settings`` is unset or a JSON object, which the Claude Agent SDK merges without reading a file."""
    if settings is None:
        return True
    try:
        return isinstance(json.loads(settings), dict)
    except ValueError:
        return False
