"""Temporal-aware sandbox client that dispatches lifecycle operations as activities."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

from agents.sandbox import Manifest
from agents.sandbox.session.sandbox_client import (
    BaseSandboxClient,
    BaseSandboxClientOptions,
)
from agents.sandbox.session.sandbox_session import SandboxSession
from agents.sandbox.session.sandbox_session_state import SandboxSessionState
from agents.sandbox.snapshot import SnapshotBase, SnapshotSpec, SnapshotSpecUnion
from pydantic.type_adapter import TypeAdapter

from temporalio import workflow
from temporalio.openai_agents._errors import AgentsWorkflowError
from temporalio.openai_agents.sandbox._temporal_activity_models import (
    CreateSessionArgs,
    ResumeSessionArgs,
    SessionResult,
    StopArgs,
    _parse_session_state,
)
from temporalio.openai_agents.sandbox._temporal_sandbox_session import (
    TemporalSandboxSession,
    _sent_trusted_manifest,
    _session_activity_config,
)
from temporalio.workflow import ActivityConfig


class TemporalSandboxClient(BaseSandboxClient[BaseSandboxClientOptions]):
    """Stateless client that dispatches all lifecycle operations as Temporal activities.

    No inner client is needed -- session creation, resumption, and deletion are
    all handled by activities whose names are prefixed with the provider
    ``name`` (e.g. ``"daytona-sandbox_create_session"``).  The real
    ``BaseSandboxClient`` lives inside :class:`SandboxClientProvider` or
    :class:`OnHostSandboxClientProvider` on the worker. Creation runs on the
    configured Task Queue, and operations on a sandbox from an
    ``OnHostSandboxClientProvider`` go to the Task Queue in its state.

    Users should never need to instantiate this directly -- use
    :func:`temporalio.openai_agents.workflow.temporal_sandbox_client`
    instead.

    Args:
        name: The name of the :class:`SandboxClientProvider` registered on the
            worker.  Used as an activity-name prefix so that the correct
            sandbox backend is targeted.
        config: Optional activity configuration for controlling timeouts,
            retries, etc.  Defaults to a 5-minute ``start_to_close_timeout``.
        backend_id: The client's ``backend_id``. Defaults to ``name``. See
            :func:`temporalio.openai_agents.workflow.temporal_sandbox_client`.
    """

    def __init__(
        self,
        name: str,
        config: ActivityConfig | None = None,
        *,
        backend_id: str | None = None,
    ) -> None:
        """Initialize the client."""
        self._name = name
        self._config: ActivityConfig = config or ActivityConfig(
            start_to_close_timeout=timedelta(minutes=5),
        )
        self.backend_id = backend_id or name
        # The manifest with mount credentials of each session this client
        # returned, for the delete Activity: OpenAI's SDK passes delete() a
        # wrapper of the session.
        self._trusted_manifests: dict[uuid.UUID, Manifest] = {}

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: BaseSandboxClientOptions,
    ) -> SandboxSession:
        """Create a new sandbox session via activity."""
        _reject_host_path_grants(manifest)
        result: SessionResult = await workflow.execute_activity(
            f"{self._name}-sandbox_client_create",
            arg=CreateSessionArgs(
                snapshot_spec=TypeAdapter(SnapshotSpecUnion).validate_python(snapshot)
                if isinstance(snapshot, SnapshotSpec)
                else snapshot,
                manifest=manifest,
                client_options=options,
            ),
            result_type=SessionResult,
            **self._config,
        )
        return self._new_session(result, manifest)

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        """Resume an existing sandbox session via activity."""
        _reject_host_path_grants(state.manifest)
        # Serializing the state drops the mount credentials the Runner rebound
        # into it, so only then does its manifest carry them to the Worker. A
        # state with nothing rebound sends none.
        trusted_manifest = state.manifest if state.mount_authority_rebound else None
        result: SessionResult = await workflow.execute_activity(
            f"{self._name}-sandbox_client_resume",
            arg=ResumeSessionArgs(state=state, trusted_manifest=trusted_manifest),
            result_type=SessionResult,
            **_session_activity_config(self._config, state),
        )
        return self._new_session(result, trusted_manifest)

    async def delete(self, session: TemporalSandboxSession) -> TemporalSandboxSession:  # type: ignore[override]
        """Delete a sandbox session via activity."""
        state = session.state
        await workflow.execute_activity(
            f"{self._name}-sandbox_client_delete",
            arg=StopArgs(
                state=state,
                trusted_manifest=_sent_trusted_manifest(
                    state, self._trusted_manifests.get(state.session_id)
                ),
            ),
            **_session_activity_config(self._config, state),
        )
        self._trusted_manifests.pop(state.session_id, None)
        return session

    def _new_session(
        self, result: SessionResult, trusted_manifest: Manifest | None
    ) -> SandboxSession:
        if trusted_manifest is not None:
            self._trusted_manifests[result.state.session_id] = trusted_manifest
        return self._wrap_session(
            TemporalSandboxSession(
                name=self._name,
                config=self._config,
                state=result.state,
                supports_pty_flag=result.supports_pty,
                trusted_manifest=trusted_manifest,
            ),
            # Real instrumentation runs in the activity in the real client session.
            instrumentation=None,
        )

    def deserialize_session_state(self, payload: dict[str, Any]) -> SandboxSessionState:
        """Deserialize a session state from a dict."""
        return _parse_session_state(payload)


def _reject_host_path_grants(manifest: Manifest | None) -> None:
    if manifest is None:
        return
    # Sandbox-side paths only: this message reaches the workflow failure event.
    bound = [g.path for g in manifest.extra_path_grants if g.host_path is not None]
    if bound:
        raise AgentsWorkflowError(
            "Sandbox path grants with a host_path are not supported by the Temporal OpenAI "
            f"Agents plugin (found: {', '.join(bound)}). A grant's host_path is written "
            "into the activity argument in plaintext. Remove host_path from these grants."
        )
