"""Pydantic models for Temporal sandbox activity arguments and results.

Using ``pydantic_data_converter`` on the Temporal client means these models are
serialized/deserialized automatically. Each activity receives a single typed
model instance rather than a positional arg list.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from typing import Annotated, Any, cast

from agents.sandbox import Manifest
from agents.sandbox.session.sandbox_client import BaseSandboxClientOptions
from agents.sandbox.session.sandbox_session_state import SandboxSessionState
from agents.sandbox.snapshot import SnapshotBase, SnapshotSpecUnion
from agents.sandbox.types import User
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    SerializeAsAny,
    field_validator,
)


def _coerce_bytes(v: Any) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, str):
        return b64decode(v)
    raise ValueError(f"Expected bytes or base64 string, got {type(v)}")


# Bytes type that is stored as raw bytes in Python but base64-encoded in JSON,
# ensuring lossless serialization of arbitrary binary data through pydantic.
JsonSafeBytes = Annotated[
    bytes,
    BeforeValidator(_coerce_bytes),
    PlainSerializer(lambda v: b64encode(v).decode("ascii"), return_type=str),
]

# ---------------------------------------------------------------------------
# Session state of a sandbox pinned to the Worker process that created it.
# ---------------------------------------------------------------------------


class OnHostSandboxSessionState(SandboxSessionState):
    """The Workflow's state of a sandbox created by an ``OnHostSandboxClientProvider``.

    It is the backend's own state, with the backend's ``type`` and its other
    fields kept as undeclared fields, plus the Task Queue of the Worker process
    holding the sandbox. The Workflow sends every operation on the sandbox to
    that queue, and a Worker with that queue can rebuild the session with the
    backend's ``resume()`` when OpenAI's SDK resumes it. Its ``type`` isn't a
    ``Literal``, so OpenAI doesn't register it by ``type``: the plugin parses a
    state that has a ``task_queue`` as this class.
    """

    # Pydantic merges this with the base model's config.
    model_config = ConfigDict(extra="allow")

    task_queue: str


def _parse_session_state(value: object) -> SandboxSessionState:
    """Parse a session state, as an ``OnHostSandboxSessionState`` when it has a ``task_queue``."""
    if isinstance(value, dict) and "task_queue" in value:
        return OnHostSandboxSessionState.model_validate(value)
    return SandboxSessionState.parse(value)


# ---------------------------------------------------------------------------
# Shared base for all argument models that carry a session state field.
# ---------------------------------------------------------------------------


class _HasState(BaseModel):
    state: SerializeAsAny[SandboxSessionState]

    @field_validator("state", mode="before")
    @classmethod
    def _coerce_state(cls, value: object) -> SandboxSessionState:
        return _parse_session_state(value)


# ---------------------------------------------------------------------------
# Argument models (workflow -> activity)
# ---------------------------------------------------------------------------


class ExecArgs(_HasState):
    """Arguments for exec activity."""

    command: list[str]
    timeout: float | None = None
    shell: bool | list[str] = True
    user: str | User | None = None


class ReadArgs(_HasState):
    """Arguments for read activity."""

    path: str


class WriteArgs(_HasState):
    """Arguments for write activity."""

    path: str
    data: JsonSafeBytes


class RunningArgs(_HasState):
    """Arguments for running check activity."""

    pass


class PersistWorkspaceArgs(_HasState):
    """Arguments for persist workspace activity."""

    pass


class HydrateWorkspaceArgs(_HasState):
    """Arguments for hydrate workspace activity."""

    data: JsonSafeBytes


class PtyExecStartArgs(_HasState):
    """Arguments for PTY exec start activity."""

    command: list[str]
    timeout: float | None = None
    shell: bool | list[str] = True
    user: str | User | None = None
    tty: bool = False
    yield_time_s: float | None = None
    max_output_tokens: int | None = None


class PtyWriteStdinArgs(_HasState):
    """Arguments for PTY write stdin activity."""

    session_id: int
    chars: str
    yield_time_s: float | None = None
    max_output_tokens: int | None = None


class StartArgs(_HasState):
    """Arguments for start activity."""

    pass


class StopArgs(_HasState):
    """Arguments for stop activity."""

    pass


# ---------------------------------------------------------------------------
# Result models (activity -> workflow)
# ---------------------------------------------------------------------------


class ExecResult(BaseModel):
    """Result of an exec activity."""

    stdout: JsonSafeBytes
    stderr: JsonSafeBytes
    exit_code: int


class PtyExecUpdateResult(BaseModel):
    """Result of a PTY exec activity."""

    process_id: int | None
    output: JsonSafeBytes
    exit_code: int | None
    original_token_count: int | None


class ReadResult(BaseModel):
    """Result of a read activity."""

    data: JsonSafeBytes


class RunningResult(BaseModel):
    """Result of a running check activity."""

    is_running: bool


class PersistWorkspaceResult(BaseModel):
    """Result of a persist workspace activity."""

    data: JsonSafeBytes


# ---------------------------------------------------------------------------
# Session lifecycle models (create / resume)
# ---------------------------------------------------------------------------


class CreateSessionArgs(BaseModel):
    """Arguments for create session activity."""

    snapshot_spec: SnapshotSpecUnion | SerializeAsAny[SnapshotBase] | None = None
    manifest: Manifest | None = None
    client_options: SerializeAsAny[BaseSandboxClientOptions] | None = None

    @field_validator("snapshot_spec", mode="before")
    @classmethod
    def _coerce_snapshot_spec(
        cls, value: object
    ) -> SnapshotSpecUnion | SnapshotBase | None:
        if value is None or isinstance(value, SnapshotBase):
            return value
        # SnapshotBase subclasses always carry an `id` field;
        # SnapshotSpec subclasses do not.  Use that to distinguish
        # serialized SnapshotBase dicts from SnapshotSpecUnion dicts.
        if isinstance(value, dict) and "id" in value:
            return SnapshotBase.parse(value)
        return cast(SnapshotSpecUnion | None, value)

    @field_validator("client_options", mode="before")
    @classmethod
    def _coerce_client_options(cls, value: object) -> BaseSandboxClientOptions | None:
        if value is None:
            return None
        return BaseSandboxClientOptions.parse(value)


class ResumeSessionArgs(_HasState):
    """Arguments for resume session activity."""

    # The state's manifest with the mount credentials that serializing the state
    # drops. The Worker rebinds them into the state for the backend's resume().
    trusted_manifest: Manifest | None = None


class SessionResult(_HasState):
    """Result of create/resume -- session state + capabilities."""

    supports_pty: bool
