"""Public-facing provider that pairs a name with a real sandbox client."""

from __future__ import annotations

import abc
import io
from collections.abc import Callable, Collection, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agents.sandbox import Manifest
from agents.sandbox.errors import SandboxError
from agents.sandbox.session.sandbox_client import BaseSandboxClient
from agents.sandbox.session.sandbox_session import SandboxSession
from agents.sandbox.session.sandbox_session_state import SandboxSessionState

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents._temporal_worker_env_ref import (
    AllowAllWorkerEnvVars,
    _snapshot_resolvable_env_vars,
)
from temporalio.openai_agents.sandbox._temporal_activity_models import (
    CreateSessionArgs,
    ExecArgs,
    HydrateWorkspaceArgs,
    OnHostSandboxSessionState,
    PersistWorkspaceArgs,
    PersistWorkspaceResult,
    PtyExecStartArgs,
    PtyExecUpdateResult,
    PtyWriteStdinArgs,
    ReadArgs,
    ReadResult,
    ResumeSessionArgs,
    RunningArgs,
    RunningResult,
    SessionResult,
    StartArgs,
    StopArgs,
    WriteArgs,
    _HasState,
)
from temporalio.openai_agents.sandbox._temporal_activity_models import (
    ExecResult as ExecResultModel,
)
from temporalio.openai_agents.sandbox._temporal_worker_env_value import (
    _resolvable_worker_env_vars_scope,
)


@contextmanager
def _translate_sandbox_errors() -> Generator[None, None, None]:
    # Temporal retries every activity exception by default, so only a SandboxError
    # the library has classified as terminal (retryable is False) is turned into a
    # non-retryable ApplicationError.
    try:
        yield
    except SandboxError as e:
        if e.retryable is False:
            raise ApplicationError(
                str(e), type=str(e.error_code), non_retryable=True
            ) from e
        raise


@contextmanager
def _sandbox_activity_scope(
    resolvable_worker_env_vars: frozenset[str] | AllowAllWorkerEnvVars,
) -> Generator[None, None, None]:
    with _resolvable_worker_env_vars_scope(resolvable_worker_env_vars):
        with _translate_sandbox_errors():
            yield


# Backends that run the sandbox on the Worker's own machine.
_ON_HOST_BACKEND_IDS = frozenset({"docker", "unix_local"})


def _recorded_task_queue(state: SandboxSessionState) -> str | None:
    """The Task Queue ``state`` records, if any."""
    return state.task_queue if isinstance(state, OnHostSandboxSessionState) else None


def _with_trusted_mounts(
    state: SandboxSessionState, trusted_manifest: Manifest | None
) -> SandboxSessionState:
    """Return ``state`` with the mount credentials of ``trusted_manifest``.

    The data converter drops them from the state, and a backend's ``resume()``
    refuses a state without them.
    """
    if trusted_manifest is None:
        return state
    return state.rebind_persisted_mount_authority(
        trusted_manifest, provider_backend_id=state.type
    )


class _SandboxProviderBase(abc.ABC):
    """The Activities shared by both sandbox client providers."""

    def __init__(self, name: str, client: BaseSandboxClient[Any]) -> None:
        self._name = name
        self._client = client
        self._sessions: dict[str, SandboxSession] = {}

    @property
    def name(self) -> str:
        """The provider name used as an activity-name prefix."""
        return self._name

    @abc.abstractmethod
    async def _session(self, args: _HasState) -> SandboxSession:
        """Return the live session the arguments' state names."""
        ...

    def _created_task_queue(self) -> str | None:
        """The Task Queue that a session the running create Activity makes records in its state."""
        return None

    def _session_result(
        self, session: SandboxSession, task_queue: str | None
    ) -> SessionResult:
        """Return the session's state, recording ``task_queue`` in it when given one."""
        state = session.state
        if task_queue is not None:
            state = OnHostSandboxSessionState.model_validate(
                {**state.model_dump(mode="json"), "task_queue": task_queue}
            )
        return SessionResult(state=state, supports_pty=session.supports_pty())

    @abc.abstractmethod
    def _own_activities(
        self, resolvable: frozenset[str] | AllowAllWorkerEnvVars
    ) -> list[Callable[..., Any]]:
        """Return the resume, client delete and session shutdown Activities."""
        ...

    def _get_activities(
        self, resolvable_worker_env_vars: Collection[str] | AllowAllWorkerEnvVars
    ) -> Sequence[Callable[..., Any]]:
        """Return all activity callables for registration with a Temporal Worker."""
        prefix = self._name
        resolvable = _snapshot_resolvable_env_vars(resolvable_worker_env_vars)

        # -- Client-level operations (lifecycle) --

        @activity.defn(name=f"{prefix}-sandbox_client_create")
        async def create_session(args: CreateSessionArgs) -> SessionResult:
            with _sandbox_activity_scope(resolvable):
                session = await self._client.create(
                    snapshot=args.snapshot_spec,
                    manifest=args.manifest,
                    options=args.client_options,
                )
                result = self._session_result(session, self._created_task_queue())
                # Held under the id the Workflow gets back: an on-host state is
                # the backend's serialized state, which can carry another id,
                # as Docker's does with mount credentials.
                self._sessions[str(result.state.session_id)] = session
                return result

        # -- Session-level operations (I/O and lifecycle) --

        @activity.defn(name=f"{prefix}-sandbox_session_exec")
        async def exec_(args: ExecArgs) -> ExecResultModel:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                result = await session.exec(
                    *args.command,
                    timeout=args.timeout,
                    shell=args.shell,
                    user=args.user,
                )
                return ExecResultModel(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.exit_code,
                )

        @activity.defn(name=f"{prefix}-sandbox_session_read")
        async def read(args: ReadArgs) -> ReadResult:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                handle = await session.read(Path(args.path))
                return ReadResult(data=handle.read())

        @activity.defn(name=f"{prefix}-sandbox_session_write")
        async def write(args: WriteArgs) -> None:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                await session.write(Path(args.path), io.BytesIO(args.data))
                return None

        @activity.defn(name=f"{prefix}-sandbox_session_running")
        async def running(args: RunningArgs) -> RunningResult:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                return RunningResult(is_running=await session.running())

        @activity.defn(name=f"{prefix}-sandbox_session_persist_workspace")
        async def persist_workspace(
            args: PersistWorkspaceArgs,
        ) -> PersistWorkspaceResult:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                stream = await session.persist_workspace()
                return PersistWorkspaceResult(data=stream.read())

        @activity.defn(name=f"{prefix}-sandbox_session_hydrate_workspace")
        async def hydrate_workspace(args: HydrateWorkspaceArgs) -> None:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                await session.hydrate_workspace(io.BytesIO(args.data))
                return None

        @activity.defn(name=f"{prefix}-sandbox_session_pty_exec_start")
        async def pty_exec_start(args: PtyExecStartArgs) -> PtyExecUpdateResult:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                update = await session.pty_exec_start(
                    *args.command,
                    timeout=args.timeout,
                    shell=args.shell,
                    user=args.user,
                    tty=args.tty,
                    yield_time_s=args.yield_time_s,
                    max_output_tokens=args.max_output_tokens,
                )
                return PtyExecUpdateResult(
                    process_id=update.process_id,
                    output=update.output,
                    exit_code=update.exit_code,
                    original_token_count=update.original_token_count,
                )

        @activity.defn(name=f"{prefix}-sandbox_session_pty_write_stdin")
        async def pty_write_stdin(args: PtyWriteStdinArgs) -> PtyExecUpdateResult:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                update = await session.pty_write_stdin(
                    session_id=args.session_id,
                    chars=args.chars,
                    yield_time_s=args.yield_time_s,
                    max_output_tokens=args.max_output_tokens,
                )
                return PtyExecUpdateResult(
                    process_id=update.process_id,
                    output=update.output,
                    exit_code=update.exit_code,
                    original_token_count=update.original_token_count,
                )

        @activity.defn(name=f"{prefix}-sandbox_session_start")
        async def start(args: StartArgs) -> SessionResult:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                await session.start()
                return self._session_result(session, _recorded_task_queue(args.state))

        @activity.defn(name=f"{prefix}-sandbox_session_stop")
        async def session_stop(args: StopArgs) -> SessionResult:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                await session.stop()
                return self._session_result(session, _recorded_task_queue(args.state))

        return [
            create_session,
            exec_,
            read,
            write,
            running,
            persist_workspace,
            hydrate_workspace,
            pty_exec_start,
            pty_write_stdin,
            start,
            session_stop,
            *self._own_activities(resolvable),
        ]


class SandboxClientProvider(_SandboxProviderBase):
    """A named sandbox client provider for Temporal workflows.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Wraps a ``BaseSandboxClient`` with a unique name so that multiple
    sandbox backends can be registered on a single Temporal worker.  Each
    provider gets its own set of Temporal activities whose names are prefixed
    with the provider name, allowing them to coexist on the same task queue.

    Any Worker can serve any of its sandboxes, so it is for hosted backends.
    Docker and Unix-local run the sandbox on the Worker's own machine, so it
    refuses them; use :class:`OnHostSandboxClientProvider` for those.

    On the **worker side**, pass one or more providers to the plugin::

        plugin = OpenAIAgentsPlugin(
            sandbox_clients=[
                SandboxClientProvider("daytona", DaytonaSandboxClient()),
            ],
        )

    On the **workflow side**, reference a provider by name via
    :func:`temporalio.openai_agents.workflow.temporal_sandbox_client`::

        run_config = RunConfig(
            sandbox=SandboxRunConfig(
                client=temporal_sandbox_client("daytona"),
                ...
            ),
        )

    Args:
        name: A unique name for this sandbox backend (e.g. ``"daytona"``).
            Must match the name used on the workflow side.
        client: The real ``BaseSandboxClient`` that performs sandbox
            lifecycle and I/O operations on the worker.

    Raises:
        ValueError: If ``client`` is a Docker or Unix-local client.
    """

    def __init__(self, name: str, client: BaseSandboxClient[Any]) -> None:
        """Initialize the provider."""
        if client.backend_id in _ON_HOST_BACKEND_IDS:
            raise ValueError(
                f"Sandbox client provider {name!r}: the {client.backend_id!r} backend "
                "runs the sandbox on the Worker's own machine, where other Workers "
                "can't reach it. Use OnHostSandboxClientProvider for it, with one "
                "Worker or several."
            )
        super().__init__(name, client)

    async def _session(self, args: _HasState) -> SandboxSession:
        key = str(args.state.session_id)
        if key not in self._sessions:
            self._sessions[key] = await self._client.resume(args.state)
        return self._sessions[key]

    def _own_activities(
        self, resolvable: frozenset[str] | AllowAllWorkerEnvVars
    ) -> list[Callable[..., Any]]:
        prefix = self._name

        @activity.defn(name=f"{prefix}-sandbox_client_resume")
        async def resume_session(args: ResumeSessionArgs) -> SessionResult:
            with _sandbox_activity_scope(resolvable):
                session = await self._client.resume(
                    _with_trusted_mounts(args.state, args.trusted_manifest)
                )
                self._sessions[str(session.state.session_id)] = session
                return self._session_result(session, None)

        @activity.defn(name=f"{prefix}-sandbox_client_delete")
        async def delete_session(args: StopArgs) -> None:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                await self._client.delete(session)
                return None

        @activity.defn(name=f"{prefix}-sandbox_session_shutdown")
        async def session_shutdown(args: StopArgs) -> None:
            key = str(args.state.session_id)
            session = self._sessions.get(key)
            if session is None:
                return None
            try:
                with _sandbox_activity_scope(resolvable):
                    await session.shutdown()
            except ApplicationError:
                # Terminal failure: the session is dead, so evict it before
                # re-raising. A retryable error instead propagates with the
                # entry kept so the activity's retry can still shut it down.
                del self._sessions[key]
                raise
            del self._sessions[key]
            return None

        return [resume_session, delete_session, session_shutdown]
