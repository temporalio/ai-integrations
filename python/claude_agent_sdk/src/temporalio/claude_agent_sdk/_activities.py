"""The Activities :class:`TemporalTransport` runs to start Claude Code and carry its stdin and stdout."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
from collections.abc import AsyncIterator, Callable, Mapping, Sequence

from temporalio import activity
from temporalio.contrib.workflow_streams import WorkflowStreamClient

from ._errors import claude_code_exited, unknown_sandbox
from ._home import SandboxHome
from ._output import OutputSender
from ._payloads import (
    CLOSE,
    CONNECT,
    LAUNCH,
    WATCH_OUTPUT,
    WRITE_STDIN,
    CloseRequest,
    ConnectRequest,
    LaunchRequest,
    SandboxSession,
    StdinWrite,
    WatchRequest,
)
from ._sandbox import Sandbox, SandboxContext, SandboxFactory
from ._sandbox_cli import SandboxCli, write_plugins


class SandboxActivities:
    """Activities that start Claude Code in a sandbox and carry its stdin and stdout."""

    def __init__(self, sandboxes: Mapping[str, SandboxFactory]) -> None:
        """Bind the factories that build the sandboxes, by the name a :class:`TemporalTransport` asks for."""
        self._sandboxes = dict(sandboxes)

    @activity.defn(name=CONNECT)
    async def connect(self, request: ConnectRequest) -> SandboxSession:
        """Start a sandbox for the session, on its storage, and write the plugins into it."""
        sandbox = await self._sandbox(request.sandbox, request.first_execution_run_id)
        # The config dir on the session's storage, outside the workspace.
        env = {"CLAUDE_CONFIG_DIR": SandboxHome(sandbox.home).config, **request.env}
        async with _heartbeating():
            await sandbox.start(env=env)
            plugin_dirs = await write_plugins(sandbox, request.plugins)
        return SandboxSession(
            request.sandbox,
            env["CLAUDE_CONFIG_DIR"],
            plugin_dirs,
            first_execution_run_id=request.first_execution_run_id,
        )

    @activity.defn(name=LAUNCH)
    async def launch(self, request: LaunchRequest) -> SandboxSession:
        """Start the CLI in the sandbox, with its stdin and stdout in files."""
        cli = await self._cli(request.session)
        async with _heartbeating():
            pid, boot = await cli.launch(request.cli_args)
        return dataclasses.replace(request.session, cli_pid=pid, cli_boot=boot)

    @activity.defn(name=WRITE_STDIN)
    async def write_stdin(self, write: StdinWrite) -> None:
        """Write text as one of the CLI's stdin files."""
        cli = await self._cli(write.session)
        async with _heartbeating():
            await cli.write_stdin(write.number, write.text.encode())

    @activity.defn(name=WATCH_OUTPUT)
    async def watch_output(self, request: WatchRequest) -> None:
        """Send the CLI's stdout to the Workflow as Signals until the CLI exits.

        With a stream topic, stream_event lines are published to it instead.

        Heartbeats the byte offset it has sent up to, and resumes from it when
        retried. Fails with ``CLAUDE_CODE_EXITED``, and the CLI's stderr, if the
        CLI exited with a nonzero status. A CLI that never exited, such as one
        lost with its sandbox, is retried as the retry policy allows.
        """
        cli = await self._cli(request.session)
        info = activity.info()
        details: Sequence[int] = info.heartbeat_details
        workflow = activity.client().get_workflow_handle(
            info.workflow_id or "", run_id=info.workflow_run_id
        )
        offset = details[0] if details else request.offset
        async with contextlib.AsyncExitStack() as stack:
            stream = None
            if request.stream_topic is not None:
                stream = await stack.enter_async_context(
                    WorkflowStreamClient.from_within_activity()
                )
            output = OutputSender(workflow, offset, stream, request.stream_topic or "")
            await stack.enter_async_context(_heartbeating(lambda: output.start))
            await output.forward(cli.follow_stdout(output.end))
            # tail may not have printed the last lines before it was stopped.
            await output.add_rest(await cli.read_stdout(output.end))
        status = await cli.exit_status()
        if status is None:
            # The output broke off, or the CLI was lost: retried from the heartbeat's offset.
            raise RuntimeError("Claude Code's output ended without a recorded exit")
        if status != 0:
            raise claude_code_exited(status, await cli.log())

    @activity.defn(name=CLOSE)
    async def close(self, request: CloseRequest) -> None:
        """End the sandbox, and the CLI with it. The session's storage stays."""
        sandbox = await self._sandbox(request.sandbox, request.first_execution_run_id)
        async with _heartbeating():
            await sandbox.terminate()

    async def _cli(self, session: SandboxSession) -> SandboxCli:
        """The CLI in the session's sandbox, from the factory."""
        sandbox = await self._sandbox(session.sandbox, session.first_execution_run_id)
        return SandboxCli(sandbox, session)

    async def _sandbox(self, name: str, first_execution_run_id: str) -> Sandbox:
        """The sandbox the factory registered as ``name`` builds for this Activity."""
        factory = self._sandboxes.get(name)
        if factory is None:
            raise unknown_sandbox(name)
        info = activity.info()
        context = SandboxContext(
            name,
            info.namespace,
            info.workflow_id or "",
            info.workflow_run_id or "",
            first_execution_run_id,
        )
        sandbox = factory(context)
        if inspect.isawaitable(sandbox):
            sandbox = await sandbox
        return sandbox


@contextlib.asynccontextmanager
async def _heartbeating(
    progress: Callable[[], object] | None = None,
) -> AsyncIterator[None]:
    """Heartbeat every second while the block runs, so Temporal soon notices if this Worker is lost.

    With ``progress``, each heartbeat records ``progress()``, such as how far
    into the stdout the Workflow has the output, and a retry of the Activity
    resumes from it.
    """

    async def heartbeat() -> None:
        while True:
            activity.heartbeat(*([progress()] if progress else []))
            await asyncio.sleep(1)

    heartbeats = asyncio.create_task(heartbeat())
    try:
        yield
    finally:
        heartbeats.cancel()
