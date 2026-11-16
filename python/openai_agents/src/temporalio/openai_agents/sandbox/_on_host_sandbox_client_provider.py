"""Provider for sandbox backends that run on the Worker's own machine."""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any

from agents.sandbox.session.sandbox_client import BaseSandboxClient
from agents.sandbox.session.sandbox_session import SandboxSession
from agents.sandbox.session.sandbox_session_state import SandboxSessionState

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents._temporal_worker_env_ref import AllowAllWorkerEnvVars
from temporalio.openai_agents.sandbox._sandbox_client_provider import (
    _sandbox_activity_scope,
    _SandboxProviderBase,
)
from temporalio.openai_agents.sandbox._temporal_activity_models import (
    OnHostSandboxSessionState,
    ResumeSessionArgs,
    SessionResult,
    StopArgs,
    _HasState,
)

SANDBOX_LOST_ERROR_TYPE = "SandboxLost"
"""The ``ApplicationError`` type of an operation on a sandbox whose Worker process is gone."""


class OnHostSandboxClientProvider(_SandboxProviderBase):
    """A named sandbox client provider for backends that run the sandbox on the Worker's own machine.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Use it for Docker and Unix-local. Creating a sandbox runs on the run's usual
    Task Queue, so any Worker can create one. The Worker process that creates it
    keeps the live session and records its own Task Queue,
    ``<name>@<worker_name>``, in the session state, and every later operation on
    that sandbox goes to that queue, which an extra Worker in the same process
    polls. The plugin starts that Worker with the main one and stops it with the
    main one.

    The state the Workflow keeps carries the backend's own state. The process
    rebuilds a session it doesn't hold only when OpenAI's SDK resumes it, as
    after an approval pause, a continue-as-new or a restart of this Worker, and
    only with the backend's own ``resume()``. Any other operation on a session it
    doesn't hold, such as one a restart interrupted, fails with a non-retryable
    ``ApplicationError`` of type ``SandboxLost``, except delete, which does
    nothing.

    On the **worker side**::

        plugin = OpenAIAgentsPlugin(
            sandbox_clients=[
                OnHostSandboxClientProvider(
                    "docker", DockerSandboxClient(), worker_name="build-box-1"
                ),
            ],
        )

    On the **workflow side**, reference it like any provider:
    ``temporal_sandbox_client("docker")``.

    Args:
        name: A unique name for this sandbox backend. Must match the name used
            on the workflow side.
        client: The real ``BaseSandboxClient`` that performs sandbox lifecycle
            and I/O operations on the worker.
        worker_name: Names this Worker process. Only one running Worker may use
            a name: give each Worker on a machine its own, and let the old
            process exit before a restart starts the new one. Defaults to the
            machine's hostname.
    """

    def __init__(
        self,
        name: str,
        client: BaseSandboxClient[Any],
        *,
        worker_name: str | None = None,
    ) -> None:
        """Initialize the provider."""
        super().__init__(name, client)
        if worker_name is None:
            worker_name = socket.gethostname()
        self._task_queue = f"{name}@{worker_name}"

    @property
    def task_queue(self) -> str:
        """The Task Queue this process's sandboxes are served on: ``<name>@<worker_name>``."""
        return self._task_queue

    async def _session(self, args: _HasState) -> SandboxSession:
        session = self._sessions.get(str(args.state.session_id))
        if session is not None:
            return session
        queue = (
            args.state.task_queue
            if isinstance(args.state, OnHostSandboxSessionState)
            else self._task_queue
        )
        raise ApplicationError(
            f"Sandbox session '{args.state.session_id}' is lost: the Worker process "
            f"that created it on Task Queue '{queue}' has stopped.",
            type=SANDBOX_LOST_ERROR_TYPE,
            # A retry would land on this same process.
            non_retryable=True,
        )

    def _session_result(self, session: SandboxSession) -> SessionResult:
        state = session.state
        return SessionResult(
            state=OnHostSandboxSessionState.model_validate(
                {
                    **state.model_dump(mode="json"),
                    "type": "temporal_on_host",
                    "backend_type": state.type,
                    "task_queue": self._task_queue,
                }
            ),
            supports_pty=session.supports_pty(),
        )

    def _own_activities(
        self, resolvable: frozenset[str] | AllowAllWorkerEnvVars
    ) -> list[Callable[..., Any]]:
        prefix = self._name

        @activity.defn(name=f"{prefix}-sandbox_client_resume")
        async def resume_session(args: ResumeSessionArgs) -> SessionResult:
            with _sandbox_activity_scope(resolvable):
                session = self._sessions.get(str(args.state.session_id))
                if session is None:
                    if not isinstance(args.state, OnHostSandboxSessionState):
                        # A state this provider didn't return carries nothing to
                        # rebuild from, so this raises SandboxLost.
                        return self._session_result(await self._session(args))
                    session = await self._client.resume(_backend_state(args.state))
                    self._sessions[str(session.state.session_id)] = session
                return self._session_result(session)

        @activity.defn(name=f"{prefix}-sandbox_client_delete")
        async def delete_session(args: StopArgs) -> None:
            key = str(args.state.session_id)
            session = self._sessions.get(key)
            if session is None:
                return None
            try:
                with _sandbox_activity_scope(resolvable):
                    await self._client.delete(session)
            except ApplicationError:
                # Terminal failure: stop holding the session. A retryable error
                # keeps it so the activity's retry can still delete it.
                del self._sessions[key]
                raise
            del self._sessions[key]
            return None

        # Unlike SandboxClientProvider, shutdown keeps the session: the delete
        # that follows it needs the live session to remove the sandbox.
        @activity.defn(name=f"{prefix}-sandbox_session_shutdown")
        async def session_shutdown(args: StopArgs) -> None:
            with _sandbox_activity_scope(resolvable):
                session = await self._session(args)
                await session.shutdown()
                return None

        return [resume_session, delete_session, session_shutdown]


def _backend_state(state: OnHostSandboxSessionState) -> SandboxSessionState:
    """Return the backend's own state, which ``state`` carries."""
    payload = state.model_dump(mode="json")
    del payload["task_queue"]
    payload["type"] = payload.pop("backend_type")
    return SandboxSessionState.parse(payload)
