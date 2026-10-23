"""A Docker Sandboxes :class:`Sandbox`: one sandbox per session, which keeps the session's files."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shlex
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from temporalio.claude_agent_sdk import ExecResult, Sandbox, SandboxError

_SBX_TIMEOUT = 60.0
# Creating the first sandbox from an image pulls it, and running one boots a microVM.
_START_TIMEOUT = 600.0

# Create the workspace $1 and home $2, and write the env file $3 from stdin,
# readable only by its owner.
_SETUP = 'mkdir -p "$1" "$2" "$(dirname "$3")" && umask 077 && cat > "$3"'

# Print the command's pid, then run it as that process. Killing sbx exec leaves
# the command running in the sandbox, so a stream closed early kills it by pid.
_PID_THEN_EXEC = ["sh", "-c", 'echo $$ && exec "$@"', "sh"]
_KILL = 'kill "$1"'


class DockerSandbox(Sandbox):
    """Runs a Workflow's Claude Code CLI in a Docker Sandbox, a microVM managed by ``sbx``.

    Each session has one sandbox, named after it, and its files, ``workspace``
    and ``home``, are in the sandbox's own filesystem. :meth:`terminate` stops
    the sandbox and keeps them; the next :meth:`start` starts it again. The
    sandbox may reach any host.
    :meth:`delete_session` removes the sandbox and its files. ``sbx``
    keeps sandboxes on the machine that runs it, so every Worker must run the
    same ``sbx``, for example through ``sbx_command``.

    Args:
        key: Names the sandbox: every ``DockerSandbox`` with the same key reaches the same one,
            for example ``SandboxContext.first_execution_run_id``.
        env: Variables, such as credentials, that :meth:`credentials` returns.
        cli_path: Claude Code executable inside the sandbox.
        create_args: Extra ``sbx create`` arguments, for example ``["--cpus", "2"]``.
        workspace: Directory in the sandbox the CLI works in.
        home: Directory in the sandbox for the CLI's config, the plugins, and the CLI's stdin and stdout.
        sbx_command: The command that runs ``sbx``, for example
            ``["ssh", "build-host", "sbx"]`` to run it on another host.
    """

    def __init__(
        self,
        key: str,
        *,
        env: Mapping[str, str] | None = None,
        cli_path: str = "claude",
        create_args: Sequence[str] = (),
        workspace: str = "/home/agent/workspace",
        home: str = "/home/agent/.claude-agent",
        sbx_command: Sequence[str] = ("sbx",),
    ) -> None:
        """Reach the sandbox named by ``key`` through the local ``sbx``.

        Args:
            key: Names the sandbox.
            env: Variables to add to the sandbox's environment.
            cli_path: The Claude Code executable in the sandbox.
            create_args: Extra arguments for ``sbx create``.
            workspace: The session's workspace in the sandbox.
            home: The session's ``home`` in the sandbox.
            sbx_command: The command that runs ``sbx``, put before each of its
                arguments.
        """
        self._name = _name(key)
        self._env = dict(env or {})
        self._cli_path = cli_path
        self._create_args = list(create_args)
        self._workspace = workspace
        self._home = home
        self._sbx_command = list(sbx_command)
        source = f'. {shlex.quote(self._env_file)} && exec "$@"'
        self._with_env = ["sh", "-c", source, "sh"]

    @property
    def cli_path(self) -> str:
        """The Claude Code executable in the sandbox."""
        return self._cli_path

    @property
    def home(self) -> str:
        """Where the CLI's config, the plugins, and the CLI's stdin and stdout live."""
        return self._home

    @property
    def _env_file(self) -> str:
        """The sandbox's environment, outside ``home``. Each :meth:`start` writes it again."""
        return f"{self._home.rstrip('/')}.env"

    async def credentials(self) -> Mapping[str, str]:
        """Return variables to add to the environment of the sandbox, which :meth:`start` is starting.

        Returns ``env`` by default. Override to register the sandbox with a
        credential proxy, for example with ``sbx secret set --sandbox``, and
        return only non-secret variables such as ``ANTHROPIC_BASE_URL``. They
        override same-named variables :meth:`start` was given, and are written to
        a file next to ``home``, never passed on an ``sbx`` command line.
        """
        return self._env

    async def start(self, *, env: Mapping[str, str], **kwargs: Any) -> None:
        """Start the session's sandbox, creating it the first time, and stop it first if it runs.

        Stopping it ends everything an earlier start left running, such as the
        CLI of a lost Worker or of the run before a reset, so only one CLI at a
        time writes the session's files. The files stay.
        """
        if await self._status() is None:
            # With no host path, the session's files are in the sandbox's own filesystem.
            create = [
                "create",
                "--name",
                self._name,
                "--no-progress",
                *self._create_args,
            ]
            _check(await self._sbx([*create, "claude"], timeout=_START_TIMEOUT))
        else:
            await self.terminate()
        # Let the sandbox reach any host, so a person approving each Bash call is
        # the only gate. Scoped to this sandbox; adding it again does nothing.
        allow = ["policy", "allow", "network", "--sandbox", self._name, "**"]
        _check(await self._sbx(allow))
        # Without --detach, the sandbox stops, and the CLI with it, between commands.
        _check(
            await self._sbx(
                ["run", "--detach", "--name", self._name], timeout=_START_TIMEOUT
            )
        )

        script = _env_script({**env, **await self.credentials()})
        # In /, since the workspace may not exist until setup creates it.
        sbx_exec = ["exec", "-i", "-w", "/", self._name]
        setup = ["sh", "-c", _SETUP, "sh", self._workspace, self._home, self._env_file]
        _check(await self._sbx([*sbx_exec, *setup], script.encode()))

    async def exec(
        self, argv: Sequence[str], *, input: bytes = b"", **kwargs: Any
    ) -> ExecResult:
        """Run ``argv`` in the workspace, with the environment :meth:`start` was given."""
        # No cap, so a long command runs as long as the Activity allows.
        result = await self._sbx(
            ["exec", "-i", "-w", self._workspace, self._name, *self._with_env, *argv],
            input,
            timeout=None,
        )
        if result.exit_code and await self._status() != "running":
            raise SandboxError(f"Sandbox {self._name} is not running")
        return result

    async def stream(self, argv: Sequence[str], **kwargs: Any) -> AsyncIterator[bytes]:
        """Run ``argv`` in the workspace, and yield its stdout as it comes."""
        process = await asyncio.create_subprocess_exec(
            *self._sbx_command,
            "exec",
            "-w",
            self._workspace,
            self._name,
            *self._with_env,
            *_PID_THEN_EXEC,
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert process.stdout is not None
        pid = ""
        try:
            pid = (await process.stdout.readline()).decode().strip()
            while chunk := await process.stdout.read(1 << 16):
                yield chunk
            await process.wait()
        finally:
            # Also on cancellation, so neither sbx nor the command outlives the Activity.
            if process.returncode is None:
                await self._stop_stream(process, pid)
        if process.returncode and await self._status() != "running":
            raise SandboxError(f"Sandbox {self._name} is not running")

    async def write_files(
        self, files: Mapping[str, tuple[bytes, int]], **kwargs: Any
    ) -> None:
        """Write each file through ``exec``, one at a time. Each appears whole: written aside, then moved into place."""
        for path, (content, mode) in files.items():
            write = (
                'mkdir -p "$(dirname "$1")" && cat > "$1.$$.tmp" '
                '&& chmod "$2" "$1.$$.tmp" && mv "$1.$$.tmp" "$1"'
            )
            argv = ["sh", "-c", write, "sh", path, f"{mode:o}"]
            result = await self.exec(argv, input=content)
            _check(result)

    async def terminate(self, **kwargs: Any) -> None:
        """Stop the sandbox, and everything in it. Its files stay. Does nothing if it is gone."""
        # sbx stop fails on a missing sandbox.
        result = await self._sbx(["stop", self._name])
        if result.exit_code and await self._status() is not None:
            _check(result)

    async def delete_session(self) -> None:
        """Remove the session's sandbox, with its workspace and home.

        The transport never calls it. Call it, for example from an Activity
        that builds this sandbox from the factory, once the session is done.
        Does nothing if the sandbox is gone.
        """
        # sbx rm fails on a missing sandbox, even with --force.
        result = await self._sbx(["rm", "--force", self._name])
        if result.exit_code and await self._status() is not None:
            _check(result)

    async def _stop_stream(self, process: asyncio.subprocess.Process, pid: str) -> None:
        """Kill the ``sbx exec`` of a stream, and the command it ran, by its pid in the sandbox."""
        process.kill()
        await process.wait()
        if not pid:
            return
        # Not a failure of the stream: at worst, the command runs on. OSError includes
        # a missing sbx and _sbx's TimeoutError, and RuntimeError a failed sbx ls.
        with contextlib.suppress(OSError, RuntimeError):
            # sbx exec would start a stopped sandbox again, though the command stopped with it.
            if await self._status() == "running":
                await self._sbx(["exec", self._name, "sh", "-c", _KILL, "sh", pid])

    async def _status(self) -> str | None:
        """The sandbox's status, such as ``running`` or ``stopped``, or ``None`` if ``sbx`` has no such sandbox.

        A failed ``sbx exec`` does not say which. ``sbx exec`` starts a stopped
        sandbox first, but the CLI stopped with it.
        """
        result = await self._sbx(["ls", "--json"])
        if result.exit_code:
            # Not a SandboxError, which says the sandbox is gone: the Activity retries.
            raise RuntimeError(f"sbx ls failed: {_stderr(result)}")
        sandboxes: list[dict[str, object]] = json.loads(result.stdout)
        found = [sandbox for sandbox in sandboxes if sandbox.get("name") == self._name]
        return str(found[0].get("status")) if found else None

    async def _sbx(
        self,
        args: Sequence[str],
        input: bytes = b"",
        timeout: float | None = _SBX_TIMEOUT,
    ) -> ExecResult:
        # A missing sbx raises FileNotFoundError, not a SandboxError: the Activity retries.
        process = await asyncio.create_subprocess_exec(
            *self._sbx_command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(input), timeout)
        except asyncio.TimeoutError:
            # Not a SandboxError, which says the sandbox is gone: the Activity retries.
            raise TimeoutError(f"Timed out: sbx {args[0]}") from None
        finally:
            # Also on cancellation, so no sbx process outlives the Activity.
            if process.returncode is None:
                process.kill()
                await process.wait()
        # communicate() returned, so the process has exited.
        assert process.returncode is not None
        return ExecResult(process.returncode, stdout, stderr)


def _name(key: str) -> str:
    """The name of the sandbox for ``key``."""
    return f"claude-agent-{key}"


def _env_script(env: Mapping[str, str]) -> str:
    """A shell script that exports ``env``."""
    for key in env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Not an environment variable name: {key!r}")
    return "".join(f"export {key}={shlex.quote(value)}\n" for key, value in env.items())


def _check(result: ExecResult) -> None:
    """Raise the ``sbx`` command's stderr as a :class:`SandboxError` if it failed."""
    if result.exit_code:
        raise SandboxError(_stderr(result))


def _stderr(result: ExecResult) -> str:
    return result.stderr.decode(errors="replace").strip()
