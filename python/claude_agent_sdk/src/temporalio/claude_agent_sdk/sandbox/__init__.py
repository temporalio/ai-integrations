"""Provider-neutral remote execution for one Claude turn.

A :class:`SandboxRuntime` starts a fresh sandbox per attempt and only the CLI process moves;
the worker keeps the ``ClaudeSDKClient``, tools, streaming and heartbeats. The sandbox
filesystem is the session: after each turn the Activity snapshots it, the workflow stores that
opaque id, and the next attempt starts from it. :meth:`SandboxRuntime.terminate_strays` is the
remote counterpart of the local process-group sweep. This is not Claude Code's Bash sandbox,
which is a native-tool setting.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import io
import json
import logging
import shlex
import tarfile
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from temporalio.claude_agent_sdk._native import (
    HOOK_SCRIPT,
    NATIVE_STATE_DIR_NAME,
    NATIVE_TOOLS_VERIFIED,
    Answer,
    _write_atomic,
    merge_native_calls,
    settle_transcript_file,
)
from temporalio.claude_agent_sdk._sdk import load_sdk

logger = logging.getLogger(__name__)

_DEFAULT_MAX_MESSAGE_BYTES = 1024 * 1024
_STDERR_TAIL_BYTES = 4096
# SIGKILL, SIGTERM, and 128, which Modal reports when a sandbox is terminated under a command.
_KILLED_EXIT_CODES = frozenset({137, 143, 128, -9, -15})
_EXIT_GRACE_SECONDS = 10.0
_TERM_GRACE_SECONDS = 10.0
# ``cd`` into the session directory (creating it on the first turn) and record the CLI pid
# beside the native-tool state, where the CLI's Bash and file tools cannot write (see
# :func:`cli_pidfile`). ``exec`` replaces the shell, so the pidfile names the CLI.
_PIDFILE = f'"${{1%/*}}/{NATIVE_STATE_DIR_NAME}/${{1##*/}}.cli.pid"'
LAUNCH_SCRIPT = (
    f'mkdir -p "$1" "${{1%/*}}/{NATIVE_STATE_DIR_NAME}" && cd "$1" && '
    f'echo $$ > {_PIDFILE} && shift && exec "$@"'
)

SANDBOX_HOOK_NAME = "_native_hook.py"


class SandboxError(Exception):
    """The sandbox cannot run or commit this turn. The message says why."""


class SandboxLost(Exception):
    """The sandbox or its CLI died from outside while the turn ran.

    The workflow can run the turn again from the last snapshot it committed.
    """


class SandboxProcess(Protocol):
    """One command in a sandbox. Stdin and the output streams are bytes."""

    async def write(self, data: bytes) -> None:
        """Write ``data`` to stdin and flush it."""
        ...

    async def end_input(self) -> None:
        """Close stdin."""
        ...

    def stdout(self) -> AsyncIterator[bytes]:
        """Output chunks, ending when the command's stdout closes. A broken stream raises."""
        ...

    def stderr(self) -> AsyncIterator[bytes]:
        """Stderr chunks. A broken stream may end the iterator."""
        ...

    async def wait(self) -> int:
        """The exit code once the command has ended."""
        ...


class Sandbox(Protocol):
    """One started sandbox: run commands, snapshot the filesystem, then end it."""

    async def exec(
        self, argv: Sequence[str], *, env: Mapping[str, str | None] | None = None
    ) -> SandboxProcess:
        """Start ``argv`` with no shell. ``env`` is added to the sandbox's own environment.

        The runtime must be able to run ``sh``.
        """
        ...

    async def snapshot(self) -> str:
        """Persist the filesystem. The id is what a later ``SandboxRuntime.start`` resumes from."""
        ...

    async def terminate(self) -> None:
        """End the sandbox and any command in it. Idempotent if it is already gone."""
        ...


class SandboxRuntime(Protocol):
    """Where a turn's CLI runs. Configuration lives on the worker, not in workflow history."""

    @property
    def cli_path(self) -> str:
        """The Claude Code executable inside the sandbox."""
        ...

    @property
    def supports_native_tools(self) -> bool:
        """Whether Claude's built-in tools can run inside this isolation.

        Fail closed: ``False`` unless the runtime's isolation can start Claude Code's Bash
        sandbox (bubblewrap and socat on Linux) and the image has ``python3`` and ``tar``.
        """
        ...

    def cwd(self, session_id: str) -> str:
        """The session working directory. The same path on every sandbox of this runtime."""
        ...

    async def start(
        self,
        *,
        session_id: str | None,
        snapshot: str | None,
        generation: int | None = None,
    ) -> Sandbox:
        """A new sandbox for one attempt.

        ``snapshot`` is a previous :meth:`Sandbox.snapshot` id, or None to start from the
        runtime's base image. Tag it with ``session_id`` and, when ``generation`` is set, with
        that lease generation (:func:`generation_is_stray`) so :meth:`terminate_strays` can
        tell this attempt from an older or newer one.
        """
        ...

    async def terminate_strays(self, session_id: str, *, generation: int) -> None:
        """Terminate sandboxes earlier attempts left for ``session_id``, and wait until they end.

        Only a sandbox whose generation tag :func:`generation_is_stray` says is older than
        ``generation``. An equal or newer sandbox, including one created while this list was
        in flight, is left running. Does not delete snapshots. A snapshot named by workflow
        state must survive.
        """
        ...


GENERATION_TAG = "claude-agent-generation"
"""Sandbox tag whose value is the attempt's lease generation, as a decimal string."""


def generation_is_stray(found: str | None, caller: int) -> bool:
    """Whether a sandbox tagged ``found`` is older than the attempt calling ``terminate_strays``.

    A missing tag is a sandbox from before generations were recorded, so it is a stray. An
    equal or greater generation is this attempt or a newer one and must be left alone: a list
    that returns after a newer sandbox was created still carries that newer tag. A tag that is
    not an integer is not treated as older, because this caller cannot prove it is.
    """
    if found is None or found == "":
        return True
    try:
        return int(found) < caller
    except ValueError:
        return False


class _SandboxProcess:
    """The kill handle the Activity uses. Killing the CLI terminates its sandbox."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.returncode: int | None = None
        self._sandbox = sandbox
        self.terminating: asyncio.Task[Any] | None = None

    def kill(self) -> None:
        """Schedule termination. Safe to call more than once."""
        if self.terminating is None:
            self.terminating = asyncio.ensure_future(self._sandbox.terminate())


class SandboxTransport:
    """SDK ``Transport`` for a CLI started in a sandbox.

    Duck-typed: subclassing ``claude_agent_sdk.Transport`` would import the SDK when this package
    is imported. ``options`` is the turn's ``ClaudeAgentOptions``.
    """

    def __init__(self, sandbox: Sandbox, cli_path: str, options: Any) -> None:
        """Bind one sandbox and the options the CLI is started with."""
        self._sandbox = sandbox
        self._cwd = str(options.cwd)
        self._options = options
        self._max_message = options.max_buffer_size or _DEFAULT_MAX_MESSAGE_BYTES
        self._process: _SandboxProcess = _SandboxProcess(sandbox)
        self._exec: SandboxProcess | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = b""
        self._write_lock = asyncio.Lock()
        # The SDK writes the prompt and control responses from different tasks. Modal's stdin
        # writer resends a stale offset if those overlap, so writes are serialized.
        self._ready = False
        self.stuck = False
        """True once close() terminated the sandbox because the CLI did not exit."""
        self.lost = False
        """True once the sandbox or its CLI was killed from outside."""
        self._stopped = False
        self._command = _cli_command(cli_path, options)

    def environment(self) -> dict[str, str | None]:
        """The CLI environment: SDK defaults, then the options. Not the worker's environment."""
        env: dict[str, str | None] = {
            "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
            **self._options.env,
            "CLAUDE_AGENT_SDK_VERSION": load_sdk().__version__,
        }
        reads_state = _sdk_subprocess_module()._SDK_READS_SESSION_STATE_ENV
        if not any(key.upper() == reads_state for key in env):
            env[reads_state] = "1"
        env["PWD"] = self._cwd
        return env

    async def connect(self) -> None:
        """Start the CLI in the sandbox."""
        try:
            self._exec = await self._sandbox.exec(
                ["sh", "-c", LAUNCH_SCRIPT, "sh", self._cwd, *self._command],
                env=self.environment(),
            )
        except Exception as exc:
            self.lost = True
            raise SandboxLost(f"it could not start Claude Code: {exc}") from exc
        self._stderr_task = asyncio.ensure_future(self._drain_stderr(self._exec))
        self._ready = True

    async def write(self, data: str) -> None:
        """Write one CLI message. Concurrent callers wait for the previous write."""
        if not self._ready or self._exec is None:
            raise load_sdk().CLIConnectionError(
                "SandboxTransport is not ready for writing"
            )
        try:
            async with self._write_lock:
                await self._exec.write(data.encode())
        except Exception as exc:
            # Mark lost before raising. A write that fails mid-turn must not be snapshotted:
            # the commit boundary is the flag ``snapshot`` reads, and the exception alone is
            # too late if a handler snapshots first.
            self.lost = True
            self._ready = False
            raise SandboxLost(
                f"it could not write to the sandboxed CLI: {exc}"
            ) from exc

    async def end_input(self) -> None:
        """Close the CLI's stdin, bounded so a hung write cannot hold ``close``."""
        if self._exec is not None:
            try:
                async with asyncio.timeout(_EXIT_GRACE_SECONDS), self._write_lock:
                    await self._exec.end_input()
            except Exception:  # noqa: BLE001 - already closed, or a write holding the lock hung
                logger.warning(
                    "closing the sandboxed CLI's stdin failed", exc_info=True
                )

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """NDJSON messages from the CLI's stdout."""
        return self._read_messages()

    async def _read_messages(self) -> AsyncIterator[dict[str, Any]]:
        sdk = load_sdk()
        if self._exec is None:
            raise sdk.CLIConnectionError("Not connected")
        pending = b""
        try:
            chunks = self._exec.stdout()
        except Exception as exc:
            self.lost = True
            raise SandboxLost(f"its output stream failed: {exc!r}") from exc
        while True:
            try:
                chunk = await chunks.__anext__()
            except StopAsyncIteration:
                break
            except Exception as exc:
                self.lost = True
                raise SandboxLost(f"its output stream failed: {exc!r}") from exc
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for line in lines:
                if message := self._parse(line, complete=True):
                    yield message
            if len(pending) > self._max_message:
                raise sdk.CLIJSONDecodeError(
                    f"JSON message exceeded maximum buffer size of {self._max_message} bytes",
                    ValueError(
                        f"Buffer size {len(pending)} exceeds limit {self._max_message}"
                    ),
                )
        if message := self._parse(pending):
            yield message
        try:
            code = await self._exec.wait()
        except Exception as exc:
            self.lost = True
            raise SandboxLost(f"its CLI could not be observed: {exc}") from exc
        self._process.returncode = code
        if self._stopped and code in _KILLED_EXIT_CODES:
            return
        if code in _KILLED_EXIT_CODES:
            self.lost = True
            raise SandboxLost(f"its CLI was killed (exit code {code})")
        if code:
            raise sdk.ProcessError(
                f"Command failed with exit code {code}",
                exit_code=code,
                stderr=self._stderr_tail.decode(errors="replace"),
            )

    def _parse(self, line: bytes, *, complete: bool = False) -> dict[str, Any] | None:
        """One NDJSON object, or None for a blank line or other CLI chatter.

        A ``complete`` line that does not parse loses the sandbox, as the SDK's transport raises:
        the message it held may be a tool call or the result. Only a tail cut off at EOF is
        dropped.
        """
        line = line.strip()
        if not line.startswith(b"{"):
            return None
        try:
            return json.loads(line)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            if not complete:
                return None
            self.lost = True
            raise SandboxLost(f"its CLI wrote a corrupt message: {exc}") from exc

    async def _drain_stderr(self, process: SandboxProcess) -> None:
        try:
            async for chunk in process.stderr():
                self._stderr_tail = (self._stderr_tail + chunk)[-_STDERR_TAIL_BYTES:]
                if callback := self._options.stderr:
                    for text in chunk.decode(errors="replace").splitlines():
                        callback(text)
        except Exception:  # noqa: BLE001 - diagnostics only
            logger.debug("reading the sandboxed CLI's stderr failed", exc_info=True)

    async def close(self) -> None:
        """End stdin, SIGTERM a CLI that lingers, and terminate the sandbox only if it will not exit."""
        self._ready = False
        await self.end_input()
        observed: BaseException | None = None
        if self._exec is not None and self._process.returncode is None:
            try:
                async with asyncio.timeout(_EXIT_GRACE_SECONDS):
                    self._process.returncode = await self._exec.wait()
            except TimeoutError as exc:
                logger.warning(
                    "the sandboxed CLI did not exit (%r); sending it SIGTERM", exc
                )
                if not await self._sigterm_cli():
                    logger.warning(
                        "the sandboxed CLI ignored SIGTERM; terminating its sandbox"
                    )
                    self.stuck = True
                    self._process.kill()
            except Exception as exc:  # noqa: BLE001 - the process could not be observed
                self.lost = True
                observed = exc
                logger.warning(
                    "the sandboxed CLI could not be observed (%r); sending it SIGTERM",
                    exc,
                )
                if not await self._sigterm_cli():
                    logger.warning(
                        "the sandboxed CLI ignored SIGTERM; terminating its sandbox"
                    )
                    self.stuck = True
                    self._process.kill()
        if self._process.terminating is not None:
            await asyncio.gather(self._process.terminating, return_exceptions=True)
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
        if observed is not None:
            raise SandboxLost(
                f"its CLI could not be observed: {observed}"
            ) from observed

    async def stop(self) -> None:
        """End the CLI now, without waiting for stdin EOF. ``close`` still cleans up."""
        if self._exec is not None and self._process.returncode is None:
            self._stopped = True
            await self._sigterm_cli()

    async def _sigterm_cli(self) -> bool:
        """Whether SIGTERM ended the CLI. The sandbox is terminated only if it will not go."""
        assert self._exec is not None
        try:
            async with asyncio.timeout(_TERM_GRACE_SECONDS):
                pidfile = shlex.quote(cli_pidfile(self._cwd))
                kill = await self._sandbox.exec(
                    ["sh", "-c", f"kill -TERM $(cat {pidfile})"]
                )
                await kill.wait()
                self._process.returncode = await self._exec.wait()
        except Exception:  # noqa: BLE001 - TimeoutError, or the sandbox is gone
            return False
        return True

    def is_ready(self) -> bool:
        """Whether ``write`` can accept a message."""
        return self._ready


@functools.cache
def _sdk_subprocess_module() -> Any:
    """The SDK subprocess transport, loaded on first use.

    Importing ``claude_agent_sdk`` at module import would make workflow imports of this package
    depend on the SDK. The private builder is covered by the same SDK and CLI pin as native tools.
    """
    load_sdk()
    return importlib.import_module(
        "claude_agent_sdk._internal.transport.subprocess_cli"
    )


def _cli_command(cli_path: str, options: Any) -> list[str]:
    """The CLI argv from the SDK's own builder.

    ``ClaudeSDKClient.connect`` routes ``can_use_tool`` over stdio only on the options of the
    transport it builds itself, so a custom transport has to ask for it.
    """
    prompt_tool = "stdio" if options.can_use_tool else None
    builder = _sdk_subprocess_module().SubprocessCLITransport(
        prompt="",
        options=replace(
            options, cli_path=cli_path, permission_prompt_tool_name=prompt_tool
        ),
    )
    return builder._build_command()  # type: ignore[no-any-return]


async def replace_dir(sandbox: Sandbox, path: str, files: Mapping[str, bytes]) -> None:
    """Make ``path`` hold exactly ``files`` (relative name to content), via one ``tar`` exec."""
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    process = await sandbox.exec(
        ["sh", "-c", 'rm -rf "$1" && mkdir -p "$1" && tar -x -C "$1"', "sh", path]
    )
    await process.write(archive.getvalue())
    await process.end_input()
    if code := await process.wait():
        raise SandboxError(f"could not write {path} in the sandbox (exit {code})")


async def read_dir(sandbox: Sandbox, path: str) -> dict[str, bytes]:
    """Regular files under ``path`` (relative name to content), via one ``tar`` exec."""
    process = await sandbox.exec(["tar", "-c", "-C", path, "."])
    data = b"".join([chunk async for chunk in process.stdout()])
    if code := await process.wait():
        raise SandboxError(f"could not read {path} in the sandbox (exit {code})")
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {
            member.name.removeprefix("./"): extracted.read()
            for member in tar.getmembers()
            if member.isfile() and (extracted := tar.extractfile(member)) is not None
        }


def sandbox_state_dir(cwd: str, session_id: str) -> str:
    """Native-tool state inside the sandbox, beside the session directory so the shell cannot write it."""
    return f"{cwd.rsplit('/', 1)[0]}/{NATIVE_STATE_DIR_NAME}/{session_id}"


def cli_pidfile(cwd: str) -> str:
    """Where :data:`LAUNCH_SCRIPT` records the CLI pid for the session directory ``cwd``."""
    parent, name = cwd.rsplit("/", 1)
    return f"{parent}/{NATIVE_STATE_DIR_NAME}/{name}.cli.pid"


def sandbox_hook_command(remote_dir: str) -> str:
    """The command hook inside the sandbox. The image's ``python3`` runs the pushed script."""
    return shlex.join(
        ["python3", "-I", "-S", f"{remote_dir}/{SANDBOX_HOOK_NAME}", remote_dir]
    )


async def push_state(sandbox: Sandbox, remote_dir: str, state_dir: Path) -> None:
    """Replace the sandbox state directory with this host's files and the hook script."""
    files = {
        path.relative_to(state_dir).as_posix(): path.read_bytes()
        for path in state_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    files[SANDBOX_HOOK_NAME] = HOOK_SCRIPT.read_bytes()
    try:
        await replace_dir(sandbox, remote_dir, files)
    except SandboxError:
        raise
    except Exception as exc:
        raise SandboxLost(f"it could not receive the native-tool state: {exc}") from exc


async def settle_sandbox_transcript(
    sandbox: Sandbox, cwd: str, session_id: str, answer: Answer | None = None
) -> None:
    """``_native.settle_local_transcript`` for the session's transcript inside the sandbox.

    The file is where the CLI there keeps it: its config directory, keyed by ``cwd``. A sandbox
    without one (a fresh session) is left alone.
    """
    key = load_sdk().project_key_for_directory(Path(cwd))
    path = f"${{CLAUDE_CONFIG_DIR:-$HOME/.claude}}/projects/{key}/{session_id}.jsonl"
    try:
        read = await sandbox.exec(["sh", "-c", f'cat "{path}"'])
        data = b"".join([chunk async for chunk in read.stdout()])
        if (
            await read.wait()
            or (settled := settle_transcript_file(data, answer)) is None
        ):
            return
        write = await sandbox.exec(
            ["sh", "-c", f'cat > "{path}.tmp" && mv "{path}.tmp" "{path}"']
        )
        await write.write(settled)
        await write.end_input()
        code = await write.wait()
    except SandboxLost:
        raise
    except Exception as exc:
        raise SandboxLost(f"it could not settle the session transcript: {exc}") from exc
    if code:
        raise SandboxError(f"could not settle the session transcript (exit {code})")


async def pull_state(sandbox: Sandbox, remote_dir: str, state_dir: Path) -> None:
    """Copy the hook's ledger back to the host. The decision ledger replaces the local one."""
    files = await read_dir(sandbox, remote_dir)
    if ledger_text := files.get("ledger.jsonl"):
        _write_atomic(state_dir / "ledger.jsonl", ledger_text.decode())
    if calls_text := files.get("calls.json"):
        merge_native_calls(state_dir, json.loads(calls_text)["calls"])


async def merge_remote_native_state(
    sandbox: Sandbox, remote_dir: str, state_dir: Path
) -> None:
    """Fold the snapshot's native ledger into ``state_dir`` before the host copy is pushed.

    ``test -d`` exits 0 when the directory is present and 1 when it is absent. An absent
    directory is a first turn and is left for :func:`push_state` to create. Any other exit, or
    a failure to run or observe the probe, is a lost sandbox: the caller must not replace the
    directory.
    """
    try:
        probe = await sandbox.exec(["sh", "-c", 'test -d "$1"', "sh", remote_dir])
        code = await probe.wait()
    except SandboxLost:
        raise
    except Exception as exc:
        raise SandboxLost(
            f"it could not check the native-tool state directory: {exc}"
        ) from exc
    if code == 1:
        return
    if code != 0:
        raise SandboxLost(
            f"it could not check the native-tool state directory (exit {code})"
        )
    await pull_state(sandbox, remote_dir, state_dir)


class SandboxTurn:
    """One attempt: start from the last snapshot, run a transport, snapshot, then terminate."""

    def __init__(self, runtime: SandboxRuntime, sandbox: Sandbox) -> None:
        """Bind a runtime and the sandbox it just started."""
        self.runtime = runtime
        self.sandbox = sandbox
        self.transport: SandboxTransport | None = None
        self.ledger_unreadable = False
        """True once this attempt could not read the sandbox native ledger. ``snapshot`` refuses."""

    @classmethod
    async def start(
        cls,
        runtime: SandboxRuntime,
        *,
        session_id: str | None,
        snapshot: str | None,
        generation: int | None = None,
        owns: Callable[[], Awaitable[None]] | None = None,
    ) -> SandboxTurn:
        """Sweep older sandboxes, start one for ``generation``, and check the CLI version.

        ``owns`` is the attempt's lease check. It runs after the sweep and again after create.
        A failure after create terminates only the sandbox this call just started.
        """
        if session_id is not None:
            if generation is None:
                raise SandboxError(
                    "a sandbox for a session needs the attempt's lease generation"
                )
            await runtime.terminate_strays(session_id, generation=generation)
            if owns is not None:
                await owns()
        sandbox = await runtime.start(
            session_id=session_id, snapshot=snapshot, generation=generation
        )
        if owns is not None:
            try:
                await owns()
            except BaseException:
                await _terminate_quietly(sandbox)
                raise
        turn = cls(runtime, sandbox)
        try:
            await turn._check_cli()
        except BaseException:
            await turn.terminate()
            raise
        return turn

    async def _check_cli(self) -> None:
        """Refuse a CLI that is not a verified pair with the installed SDK.

        The transport builds argv with SDK internals that pin covers, for every sandboxed turn.
        """
        cli = self.runtime.cli_path
        try:
            process = await self.sandbox.exec([cli, "--version"])
            out = b"".join([chunk async for chunk in process.stdout()])
            code = await process.wait()
        except Exception as exc:
            raise SandboxLost(f"it could not check the CLI's version: {exc}") from exc
        version = out.decode(errors="replace").split()[0] if out.split() else None
        if code or (load_sdk().__version__, version) not in NATIVE_TOOLS_VERIFIED:
            raise SandboxError(
                f"the sandbox's CLI ({cli!r}) reports version {version!r} (exit {code}); the "
                f"installed claude-agent-sdk {load_sdk().__version__} is verified only with "
                f"CLIs in {sorted(NATIVE_TOOLS_VERIFIED)}"
            )

    async def config_dir(self) -> str | None:
        """The CLI's config directory as the sandbox resolves it (``CLAUDE_CONFIG_DIR``, else
        ``$HOME/.claude``, with the provider's secrets and image environment), or None when that
        is not an absolute path.
        """
        try:
            process = await self.sandbox.exec(
                ["sh", "-c", 'printf %s "${CLAUDE_CONFIG_DIR:-$HOME/.claude}"']
            )
            out = b"".join([chunk async for chunk in process.stdout()])
            code = await process.wait()
        except Exception as exc:
            raise SandboxLost(
                f"it could not resolve the CLI's config directory: {exc}"
            ) from exc
        if code:
            raise SandboxLost(
                f"it could not resolve the CLI's config directory (exit {code})"
            )
        path = out.decode(errors="replace")
        return path if path.startswith("/") else None

    def new_transport(self, options: Any) -> SandboxTransport:
        """The transport for this attempt's CLI. One per turn."""
        self.transport = SandboxTransport(self.sandbox, self.runtime.cli_path, options)
        return self.transport

    async def snapshot(self) -> str:
        """The filesystem snapshot id. Raises when the CLI was killed or would not exit.

        A session resumed from an older snapshot would otherwise forget this turn.
        """
        if self.ledger_unreadable:
            raise SandboxError(
                "its native-tool ledger could not be read, so there is no snapshot"
            )
        if self.transport is not None and (self.transport.stuck or self.transport.lost):
            raise SandboxError(
                "its CLI was killed or would not exit, so there is no snapshot"
            )
        try:
            return await self.sandbox.snapshot()
        except Exception as exc:
            raise SandboxError(
                f"the sandbox could not be snapshotted: {exc!r}"
            ) from exc

    def kill(self) -> None:
        """Terminate the sandbox without waiting."""
        if self.transport is not None:
            self.transport._process.kill()  # noqa: SLF001 - one handle, one termination
        else:
            asyncio.ensure_future(self.sandbox.terminate())

    async def terminate(self) -> None:
        """End the sandbox. A sandbox that is already gone is ignored."""
        await _terminate_quietly(self.sandbox)


async def _terminate_quietly(sandbox: Sandbox) -> None:
    """End ``sandbox``. One that is already gone is ignored."""
    try:
        await sandbox.terminate()
    except Exception as exc:  # noqa: BLE001 - already gone
        logger.debug("terminating the Claude sandbox failed: %r", exc)


__all__ = [
    "LAUNCH_SCRIPT",
    "Sandbox",
    "SandboxError",
    "SandboxLost",
    "SandboxProcess",
    "SandboxRuntime",
    "SandboxTransport",
    "GENERATION_TAG",
    "SandboxTurn",
    "generation_is_stray",
    "merge_remote_native_state",
    "pull_state",
    "push_state",
    "sandbox_hook_command",
    "sandbox_state_dir",
    "settle_sandbox_transcript",
]
