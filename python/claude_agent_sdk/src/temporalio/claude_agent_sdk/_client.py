"""The transport's sandbox, reached from the Workflow through the sandbox Activities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TypeVar

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.types import MethodAsyncSingleParam
from temporalio.workflow import ActivityCancellationType

from ._activities import SandboxActivities
from ._payloads import (
    CloseRequest,
    ConnectRequest,
    LaunchRequest,
    SandboxSession,
    StdinWrite,
    WatchRequest,
)

# The connect Activity's minimum start-to-close timeout: starting a sandbox can be slow.
_CONNECT_TIMEOUT = timedelta(minutes=30)
# Unlimited retries, since the CLI can run for days, at most 5s apart, since the
# Claude Agent SDK waits on the output.
_WATCH_RETRY_POLICY = RetryPolicy(maximum_interval=timedelta(seconds=5))

_Arg = TypeVar("_Arg")
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class SandboxClient:
    """Runs the sandbox Activities for one transport, with its options."""

    sandbox: str
    """The name the sandbox provider is registered under."""
    first_execution_run_id: str
    """The key of the session's storage."""
    task_queue: str | None
    start_to_close_timeout: timedelta
    heartbeat_timeout: timedelta
    retry_policy: RetryPolicy | None
    stream_topic: str | None
    """The Workflow Stream topic stream_event lines are published to, if any."""

    async def connect(self, env: dict[str, str], plugins: list[str]) -> SandboxSession:
        """Start a sandbox on the session's storage. Cancelling it waits for the Activity to stop."""
        request = ConnectRequest(
            self.sandbox, self.first_execution_run_id, env, plugins
        )
        return await workflow.execute_activity_method(
            SandboxActivities.connect,
            request,
            summary="Start sandbox",
            task_queue=self.task_queue,
            # Starting a sandbox can take minutes, such as to pull its image.
            start_to_close_timeout=max(self.start_to_close_timeout, _CONNECT_TIMEOUT),
            heartbeat_timeout=self.heartbeat_timeout,
            retry_policy=self.retry_policy,
            # Close waits for the cancellation, then ends the sandbox it started.
            cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
        )

    async def launch(self, session: SandboxSession, args: list[str]) -> SandboxSession:
        """Launch the CLI."""
        request = LaunchRequest(session, args)
        return await self._execute(
            SandboxActivities.launch, request, "Start Claude Code"
        )

    async def write_stdin(
        self, session: SandboxSession, number: int, text: str, summary: str
    ) -> None:
        """Write ``text`` as the CLI's stdin file ``number``."""
        write = StdinWrite(session, number, text)
        await self._execute(SandboxActivities.write_stdin, write, summary)

    async def watch_output(self, session: SandboxSession, offset: int) -> None:
        """Signal the CLI's stdout from byte ``offset`` until it exits."""
        request = WatchRequest(session, offset, self.stream_topic)
        await workflow.execute_activity_method(
            SandboxActivities.watch_output,
            request,
            summary="Stream output",
            task_queue=self.task_queue,
            # Runs as long as the CLI, possibly days waiting on a person. Heartbeats
            # find a lost Worker; a retry resumes from the heartbeated offset.
            start_to_close_timeout=timedelta(days=365),
            heartbeat_timeout=self.heartbeat_timeout,
            retry_policy=self.retry_policy or _WATCH_RETRY_POLICY,
        )

    async def close(self) -> None:
        """End the sandbox. The session's storage stays."""
        request = CloseRequest(self.sandbox, self.first_execution_run_id)
        await self._execute(SandboxActivities.close, request, "Stop sandbox")

    async def _execute(
        self,
        activity: MethodAsyncSingleParam[SandboxActivities, _Arg, _Result],
        arg: _Arg,
        summary: str,
    ) -> _Result:
        return await workflow.execute_activity_method(
            activity,
            arg,
            summary=summary,
            task_queue=self.task_queue,
            start_to_close_timeout=self.start_to_close_timeout,
            heartbeat_timeout=self.heartbeat_timeout,
            retry_policy=self.retry_policy,
        )
