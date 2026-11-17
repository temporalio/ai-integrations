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
    _recorded_task_queue,
    _sandbox_activity_scope,
    _SandboxProviderBase,
    _with_trusted_mounts,
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
    Task Queue, so any Worker can create one. The Worker that creates it keeps
    the live session and records the Task Queue of an extra Worker beside it,
    ``<Task Queue>/<name>@<worker_name>``, in the session state. Every later
    operation on that sandbox goes to that queue. The plugin starts one extra
    Worker for each main Worker, with the main Worker's Activity settings, and
    stops it with the main one.

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
        worker_name: Names this Worker process. For each Task Queue, only one
            running process may use a name: give each process on a machine
            that serves the same Task Queue its own, and let the old process
            exit before a restart starts the new one. Workers in one process
            can share a name. Defaults to the machine's hostname.
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
        self._worker_name = (
            worker_name if worker_name is not None else socket.gethostname()
        )

    def _task_queue_for(self, main_task_queue: str) -> str:
        """The Task Queue of the extra Worker beside a main Worker on ``main_task_queue``."""
        return f"{main_task_queue}/{self._name}@{self._worker_name}"

    def _created_task_queue(self) -> str:
        # The create Activity runs on the main Worker, so its own queue names
        # the extra Worker beside it.
        return self._task_queue_for(activity.info().task_queue)

    async def _session(self, args: _HasState) -> SandboxSession:
        session = self._sessions.get(str(args.state.session_id))
        if session is not None:
            return session
        queue = _recorded_task_queue(args.state)
        where = "" if queue is None else f" on Task Queue '{queue}'"
        raise ApplicationError(
            f"Sandbox session '{args.state.session_id}' is lost: the Worker process "
            f"that created it{where} has stopped.",
            type=SANDBOX_LOST_ERROR_TYPE,
            # A retry would land on this same process.
            non_retryable=True,
        )

    def _own_activities(
        self, resolvable: frozenset[str] | AllowAllWorkerEnvVars
    ) -> list[Callable[..., Any]]:
        prefix = self._name

        @activity.defn(name=f"{prefix}-sandbox_client_resume")
        async def resume_session(args: ResumeSessionArgs) -> SessionResult:
            with _sandbox_activity_scope(resolvable):
                # Resume runs on the extra Worker, so it keeps the recorded queue.
                queue = _recorded_task_queue(args.state)
                session = self._sessions.get(str(args.state.session_id))
                if session is not None:
                    return self._session_result(session, queue)
                if not isinstance(args.state, OnHostSandboxSessionState):
                    # A state this provider didn't return carries nothing to
                    # rebuild from, so this raises SandboxLost.
                    return self._session_result(await self._session(args), None)
                session = await self._client.resume(
                    _with_trusted_mounts(
                        _backend_state(args.state), args.trusted_manifest
                    )
                )
                result = self._session_result(session, queue)
                # Held under the id the Workflow gets back, as at create.
                self._sessions[str(result.state.session_id)] = session
                return result

        @activity.defn(name=f"{prefix}-sandbox_client_delete")
        async def delete_session(args: StopArgs) -> None:
            key = str(args.state.session_id)
            session = self._sessions.get(key)
            if session is None:
                return None
            try:
                with _sandbox_activity_scope(resolvable):
                    await self._client.delete(session)
            except ApplicationError as err:
                # A retryable error keeps the session, so the Activity's retry
                # can still delete the sandbox.
                if err.non_retryable:
                    self._sessions.pop(key, None)
                raise
            self._sessions.pop(key, None)
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
    return SandboxSessionState.parse(payload)
