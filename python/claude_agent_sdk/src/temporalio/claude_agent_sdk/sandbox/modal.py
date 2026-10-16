"""Modal implementation of :class:`~temporalio.claude_agent_sdk.sandbox.SandboxRuntime`.

Importing this module imports ``modal``. Install the ``modal`` extra::

    pip install 'temporalio-claude-agent-sdk[modal]'

Each sandbox is tagged with its session and lease generation; :meth:`terminate_strays` reads
tags with ``get_tags`` (Modal 1.6 has no ``tags`` attribute) and terminates only older
generations. A snapshot is a Modal image id kept for ``snapshot_ttl``; Modal cannot delete one.
Credentials are ``modal.Secret`` objects on this runtime, never in workflow payloads.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

from temporalio.claude_agent_sdk.sandbox import (
    GENERATION_TAG,
    SandboxError,
    SandboxLost,
    generation_is_stray,
)

try:
    import modal
except ImportError as exc:
    raise ImportError(
        "Modal sandbox execution needs the `modal` package: "
        "pip install 'temporalio-claude-agent-sdk[modal]'"
    ) from exc

logger = logging.getLogger(__name__)

SESSION_TAG = "claude-agent-session"


def _modal() -> Any:
    """The ``modal`` module. Tests replace this with a fake client."""
    return modal


def _transient(modal_client: Any) -> tuple[type[BaseException], ...]:
    """Provider failures a retry can outlast: transport, timeout, and service errors. Auth,
    permission, and configuration errors are not, and retrying them would never end.
    """
    errors = modal_client.exception
    return (
        OSError,  # includes the builtin ConnectionError and TimeoutError
        errors.ConnectionError,
        errors.TimeoutError,
        errors.InternalFailure,
        errors.InternalError,
        errors.ServiceError,
        errors.ResourceExhaustedError,
    )


@dataclass(frozen=True)
class ModalSandboxRuntime:
    """Run each turn's CLI in a Modal Sandbox started from ``image``.

    Args:
        image: A ``modal.Image`` with the pinned Claude Code CLI. Later turns start from the
            previous snapshot instead. Include ``bubblewrap``, ``socat``, ``python3``, and
            ``tar`` before setting ``runtime="vm"`` for native tools.
        secrets: ``modal.Secret`` objects. One must provide the CLI's credentials, for example
            ``modal.Secret.from_name("my-anthropic-secret")`` with ``ANTHROPIC_API_KEY``. The
            CLI receives nothing from the worker environment.
        app_name: Modal App the sandboxes belong to. Created when missing.
        cli_path: The CLI inside ``image``.
        env: Extra non-secret variables for every sandbox, besides the turn options.
        runtime: ``"gvisor"`` (Modal's default) or ``"vm"``. Native Claude tools need ``"vm"``:
            Claude Code's bubblewrap does not start under gVisor.
        timeout: Maximum sandbox lifetime. The turn must finish inside it.
        workspace_root: Parent of each session directory in the sandbox.
        snapshot_timeout: Bound on one ``snapshot_filesystem`` call.
        snapshot_ttl: How long Modal keeps a snapshot, and so how long a session can resume.
        outbound_domain_allowlist: Modal's ``Sandbox.create`` egress allowlist. ``None`` (the
            default) leaves egress open; ``("api.anthropic.com",)``, plus the host of any
            ``ANTHROPIC_BASE_URL``, limits it to the model API.
    """

    image: Any
    secrets: tuple[Any, ...] = ()
    app_name: str = "temporal-claude-agent-sdk"
    cli_path: str = "claude"
    env: Mapping[str, str] = field(default_factory=dict)
    runtime: Literal["gvisor", "vm"] | None = None
    timeout: timedelta = timedelta(hours=1)
    workspace_root: str = "/workspace"
    snapshot_timeout: timedelta = timedelta(minutes=2)
    snapshot_ttl: timedelta = timedelta(days=30)
    outbound_domain_allowlist: tuple[str, ...] | None = None

    @property
    def supports_native_tools(self) -> bool:
        """Claude Code's bubblewrap starts under Modal's ``vm`` runtime, not under gVisor."""
        return self.runtime == "vm"

    def cwd(self, session_id: str) -> str:
        """The session directory ``<workspace_root>/<session_id>``."""
        return f"{self.workspace_root.rstrip('/')}/{session_id}"

    async def start(
        self,
        *,
        session_id: str | None,
        snapshot: str | None,
        generation: int | None = None,
    ) -> ModalSandbox:
        """Create one sandbox from ``snapshot`` or from ``image``.

        A transient provider failure is :class:`SandboxLost`: nothing ran, and the turn can be
        tried again. Any other is :class:`SandboxError`, which rejects the turn.
        """
        modal_client = _modal()
        egress = (
            {"outbound_domain_allowlist": list(self.outbound_domain_allowlist)}
            if self.outbound_domain_allowlist is not None
            else {}
        )
        tags = None
        if session_id is not None:
            tags = {SESSION_TAG: session_id}
            if generation is not None:
                tags[GENERATION_TAG] = str(generation)
        try:
            app = await modal_client.App.lookup.aio(
                self.app_name, create_if_missing=True
            )
            sandbox = await modal_client.Sandbox.create.aio(
                app=app,
                image=modal_client.Image.from_id(snapshot) if snapshot else self.image,
                secrets=list(self.secrets),
                env=dict(self.env),
                timeout=int(self.timeout.total_seconds()),
                runtime=self.runtime,
                tags=tags,
                **egress,
            )
        except modal_client.exception.NotFoundError as exc:
            if snapshot is None:
                raise
            raise SandboxError(
                f"snapshot {snapshot} is expired or missing ({exc}), so the session cannot "
                f"resume from it; start a new session (snapshots are kept for snapshot_ttl="
                f"{self.snapshot_ttl}, raise it to keep idle sessions longer)"
            ) from exc
        except _transient(modal_client) as exc:
            raise SandboxLost(f"Modal could not start a sandbox: {exc!r}") from exc
        except Exception as exc:
            raise SandboxError(f"Modal could not start a sandbox: {exc!r}") from exc
        return ModalSandbox(sandbox, self)

    async def terminate_strays(self, session_id: str, *, generation: int) -> None:
        """Terminate this session's sandboxes older than ``generation``. Snapshots stay.

        A transient provider failure is :class:`SandboxLost`, before this attempt started
        anything; any other is :class:`SandboxError`.
        """
        modal_client = _modal()
        try:
            app = await modal_client.App.lookup.aio(
                self.app_name, create_if_missing=True
            )
            async for old in modal_client.Sandbox.list.aio(
                app_id=app.app_id, tags={SESSION_TAG: session_id}
            ):
                tags = await old.get_tags.aio()
                found = tags.get(GENERATION_TAG) if isinstance(tags, dict) else None
                if not generation_is_stray(found, generation):
                    continue
                logger.warning(
                    "terminating Claude sandbox %s left by an earlier attempt",
                    old.object_id,
                )
                await old.terminate.aio(wait=True)
        except _transient(modal_client) as exc:
            raise SandboxLost(
                f"Modal could not end this session's earlier sandboxes: {exc!r}"
            ) from exc
        except Exception as exc:
            raise SandboxError(
                f"Modal could not end this session's earlier sandboxes: {exc!r}"
            ) from exc


class ModalSandbox:
    """One ``modal.Sandbox``."""

    def __init__(self, sandbox: Any, runtime: ModalSandboxRuntime) -> None:
        """Bind a created sandbox to the runtime that configured its snapshot limits."""
        self._sandbox = sandbox
        self._runtime = runtime

    async def exec(
        self, argv: Sequence[str], *, env: Mapping[str, str | None] | None = None
    ) -> ModalProcess:
        """Start ``argv``. Byte streams, not text."""
        return ModalProcess(await self._sandbox.exec.aio(*argv, env=env, text=False))

    async def snapshot(self) -> str:
        """The Modal image id of this filesystem. Opaque to the workflow."""
        image = await self._sandbox.snapshot_filesystem.aio(
            int(self._runtime.snapshot_timeout.total_seconds()),
            ttl=int(self._runtime.snapshot_ttl.total_seconds()),
        )
        return str(image.object_id)

    async def terminate(self) -> None:
        """Terminate and wait. Idempotent at the call site if the sandbox is already gone."""
        await self._sandbox.terminate.aio(wait=True)


class ModalProcess:
    """One ``sandbox.exec`` process."""

    def __init__(self, process: Any) -> None:
        """Bind Modal's container process."""
        self._process = process

    async def write(self, data: bytes) -> None:
        """Write stdin and drain it."""
        self._process.stdin.write(data)
        await self._process.stdin.drain.aio()

    async def end_input(self) -> None:
        """Send stdin EOF and drain it."""
        self._process.stdin.write_eof()
        await self._process.stdin.drain.aio()

    def stdout(self) -> AsyncIterator[bytes]:
        """The process stdout iterator."""
        return self._process.stdout.__aiter__()  # type: ignore[no-any-return]

    def stderr(self) -> AsyncIterator[bytes]:
        """The process stderr iterator."""
        return self._process.stderr.__aiter__()  # type: ignore[no-any-return]

    async def wait(self) -> int:
        """The process exit code."""
        return await self._process.wait.aio()  # type: ignore[no-any-return]
