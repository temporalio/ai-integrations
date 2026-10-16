"""Modal adapter against a fake ``modal`` client. No Modal credentials.

The provider-neutral behavior is in ``test_claude_sandbox.py``. This file checks only
what the Modal adapter adds: create arguments, tags, snapshot image ids, and
``NotFoundError`` for an expired snapshot.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import claude_agent_sdk
import pytest

from temporalio.claude_agent_sdk import sandbox as sandbox_module
from temporalio.claude_agent_sdk._activity import ClaudeTurnActivity
from temporalio.claude_agent_sdk._models import (
    CLAUDE_SANDBOX_LOST_ERROR_TYPE,
    CLAUDE_TURN_REJECTED_ERROR_TYPE,
    ClaudeTurnInput,
)
from temporalio.claude_agent_sdk.sandbox import GENERATION_TAG
from temporalio.claude_agent_sdk.sandbox.modal import (
    SESSION_TAG,
    ModalSandbox,
    ModalSandboxRuntime,
)
from temporalio.claude_agent_sdk.testing import FakeClaudeCLI
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from .helpers import fake_sandbox
from .helpers import fixtures as fx

pytestmark = fx.POSIX_ONLY

CLI_VERSION = "2.1.294"


class _Aio:
    """Modal's wrapper: ``fn.aio(...)`` is the awaitable form."""

    def __init__(self, fn: Any) -> None:
        self.aio = fn


class _Chunks:
    def __init__(self, stream: asyncio.StreamReader) -> None:
        self._stream = stream

    @classmethod
    def of(cls, data: bytes) -> _Chunks:
        stream = asyncio.StreamReader()
        stream.feed_data(data)
        stream.feed_eof()
        return cls(stream)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while chunk := await self._stream.read(65536):
            yield chunk


class _Exec:
    """A fake container process over a local subprocess."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdin and process.stdout and process.stderr
        self.process = process
        self.stdin = types.SimpleNamespace(
            write=process.stdin.write,
            write_eof=process.stdin.write_eof,
            drain=_Aio(process.stdin.drain),
        )
        self.stdout = _Chunks(process.stdout)
        self.stderr = _Chunks(process.stderr)
        self.wait = _Aio(process.wait)


class _Version:
    """What ``cli --version`` prints."""

    def __init__(self, text: str) -> None:
        async def wait() -> int:
            return 0

        self.stdout = _Chunks.of(text.encode())
        self.wait = _Aio(wait)


class FakeSandbox:
    """One fake ``modal.Sandbox``."""

    def __init__(
        self, modal: FakeModal, image: Any, tags: dict[str, str] | None, **kw: Any
    ) -> None:
        self.modal = modal
        self.image = image
        self._tags = dict(tags or {})
        self.create_kwargs = kw
        self.execs: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.terminated = 0
        self.object_id = f"sb-{len(modal.sandboxes) + 1}"
        self.terminate = _Aio(self._terminate)
        self.exec = _Aio(self._exec)
        self.get_tags = _Aio(self._get_tags)
        self.snapshot_filesystem = _Aio(self._snapshot)

    async def _get_tags(self) -> dict[str, str]:
        return dict(self._tags)

    async def _terminate(self, *, wait: bool = False) -> None:
        del wait
        self.terminated += 1
        for process in self.modal.processes.get(self.object_id, []):
            if process.returncode is None:
                process.kill()

    async def _exec(self, *argv: str, **kw: Any) -> Any:
        self.execs.append((argv, kw))
        if argv[1:] == ("--version",):
            return _Version(self.modal.cli_version)
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ["PATH"], **(kw.get("env") or {})},
        )
        self.modal.processes.setdefault(self.object_id, []).append(process)
        return _Exec(process)

    async def _snapshot(self, timeout: int, *, ttl: int) -> Any:
        if self.modal.snapshot_error is not None:
            raise self.modal.snapshot_error
        image_id = f"im-snapshot-{len(self.modal.snapshots) + 1}"
        self.modal.snapshots.append((self.object_id, timeout, ttl))
        return types.SimpleNamespace(object_id=image_id)


class NotFoundError(Exception):
    """``modal.exception.NotFoundError``."""


class AuthError(Exception):
    """``modal.exception.AuthError``."""


_TRANSIENT = {
    name: type(name, (Exception,), {"__doc__": f"``modal.exception.{name}``."})
    for name in (
        "ConnectionError",
        "TimeoutError",
        "InternalFailure",
        "InternalError",
        "ServiceError",
        "ResourceExhaustedError",
    )
}


class FakeModal:
    """The slice of ``modal`` the runtime calls."""

    def __init__(self) -> None:
        self.cli_version = f"{CLI_VERSION} (Claude Code)\n"
        self.sandboxes: list[FakeSandbox] = []
        self.processes: dict[str, list[asyncio.subprocess.Process]] = {}
        self.snapshots: list[tuple[str, int, int]] = []
        self.snapshot_error: Exception | None = None
        self.gone_images: set[str] = set()
        self.exception = types.SimpleNamespace(
            NotFoundError=NotFoundError, AuthError=AuthError, **_TRANSIENT
        )
        self.App = types.SimpleNamespace(lookup=_Aio(self._lookup))
        self.Image = types.SimpleNamespace(from_id=lambda image_id: f"image:{image_id}")
        self.Sandbox = types.SimpleNamespace(
            create=_Aio(self._create), list=_Aio(self._list)
        )

    async def _lookup(self, name: str, *, create_if_missing: bool) -> Any:
        del create_if_missing
        return types.SimpleNamespace(app_id=f"ap-{name}")

    async def _create(
        self, *, image: Any, tags: dict[str, str] | None = None, **kw: Any
    ) -> Any:
        if image in self.gone_images:
            raise NotFoundError(f"{image} has expired")
        sandbox = FakeSandbox(self, image, tags, **kw)
        self.sandboxes.append(sandbox)
        return sandbox

    async def _list(
        self, *, app_id: str, tags: dict[str, str]
    ) -> AsyncIterator[FakeSandbox]:
        del app_id
        for sandbox in self.sandboxes:
            if not sandbox.terminated and all(
                sandbox._tags.get(key) == value for key, value in tags.items()
            ):
                yield sandbox


@pytest.fixture(autouse=True)
def _skip_version_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")


@pytest.fixture(autouse=True)
def _verified_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sandbox_module,
        "NATIVE_TOOLS_VERIFIED",
        frozenset({(claude_agent_sdk.__version__, CLI_VERSION)}),
    )


@pytest.fixture
def modal(monkeypatch: pytest.MonkeyPatch) -> FakeModal:
    fake = FakeModal()
    monkeypatch.setattr(
        "temporalio.claude_agent_sdk.sandbox.modal._modal", lambda: fake
    )
    return fake


def _runtime(tmp_path: Path, cli: FakeClaudeCLI, **kw: Any) -> ModalSandboxRuntime:
    return ModalSandboxRuntime(
        image="image:base",
        cli_path=cli.cli_path,
        workspace_root=str(tmp_path / "workspace"),
        **kw,
    )


def _activity(runtime: ModalSandboxRuntime, tmp_path: Path) -> ClaudeTurnActivity:
    return ClaudeTurnActivity(sandbox=runtime, workspace_root=tmp_path / "host")


def _input(**overrides: Any) -> ClaudeTurnInput:
    overrides.setdefault("session_id", fx.SESSION)
    return ClaudeTurnInput(prompt="Hello", **overrides)


async def test_an_egress_allowlist_reaches_sandbox_create(
    tmp_path: Path, modal: FakeModal
) -> None:
    runtime = _runtime(
        tmp_path,
        fx.completed_cli(tmp_path),
        outbound_domain_allowlist=("api.anthropic.com",),
    )
    await ActivityEnvironment().run(_activity(runtime, tmp_path).run_turn, _input())
    (sandbox,) = modal.sandboxes
    assert sandbox.create_kwargs["outbound_domain_allowlist"] == ["api.anthropic.com"]


async def test_a_turn_returns_the_modal_image_id(
    tmp_path: Path, modal: FakeModal
) -> None:
    cli = fx.completed_cli(tmp_path)
    runtime = _runtime(tmp_path, cli, secrets=("secret",), env={"EXTRA": "1"})

    result = await ActivityEnvironment().run(
        _activity(runtime, tmp_path).run_turn, _input()
    )

    assert result.text == "Hello from the completed message."
    assert result.sandbox_snapshot == "im-snapshot-1"
    (sandbox,) = modal.sandboxes
    assert sandbox.image == "image:base"
    assert await sandbox.get_tags.aio() == {
        SESSION_TAG: fx.SESSION,
        GENERATION_TAG: "1",
    }
    with pytest.raises(AttributeError):
        sandbox.tags  # type: ignore[attr-defined]
    assert sandbox.create_kwargs["secrets"] == ["secret"]
    assert sandbox.create_kwargs["env"] == {"EXTRA": "1"}
    assert "outbound_domain_allowlist" not in sandbox.create_kwargs
    assert modal.snapshots == [("sb-1", 120, 30 * 24 * 3600)]
    assert sandbox.terminated == 1
    argv, kw = sandbox.execs[-1]
    assert argv[:3] == ("sh", "-c", sandbox_module.LAUNCH_SCRIPT)
    assert argv[4] == f"{runtime.workspace_root}/{fx.SESSION}"
    assert argv[5] == cli.cli_path
    assert f"--session-id={fx.SESSION}" in argv
    assert f"--resume={fx.SESSION}" not in argv
    assert kw["env"]["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"
    assert "PATH" not in kw["env"]


async def test_the_next_turn_starts_from_the_image_id(
    tmp_path: Path, modal: FakeModal
) -> None:
    cli = fx.completed_cli(tmp_path)
    activity = _activity(_runtime(tmp_path, cli), tmp_path)
    first = await ActivityEnvironment().run(activity.run_turn, _input())

    await ActivityEnvironment().run(
        activity.run_turn, _input(resume=True, sandbox_snapshot=first.sandbox_snapshot)
    )

    second = modal.sandboxes[1]
    assert second.image == f"image:{first.sandbox_snapshot}"
    assert f"--resume={fx.SESSION}" in second.execs[-1][0]
    assert modal.snapshots[-1][0] == "sb-2"


async def test_a_live_stray_is_terminated_through_the_adapter(
    tmp_path: Path, modal: FakeModal
) -> None:
    cli = fx.completed_cli(tmp_path)
    lost = await modal.Sandbox.create.aio(image="x", tags={SESSION_TAG: fx.SESSION})
    other = await modal.Sandbox.create.aio(
        image="x", tags={SESSION_TAG: "someone-else"}
    )
    process = await asyncio.create_subprocess_exec(
        "sleep",
        "30",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    modal.processes.setdefault(lost.object_id, []).append(process)

    try:
        assert lost.terminated == 0
        assert process.returncode is None
        await ActivityEnvironment().run(
            _activity(_runtime(tmp_path, cli), tmp_path).run_turn, _input()
        )
        assert lost.terminated == 1
        assert await lost.get_tags.aio() == {SESSION_TAG: fx.SESSION}
        assert process.returncode is not None
        assert other.terminated == 0
        assert modal.snapshots  # the committed snapshot is not deleted
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_an_unparsable_generation_tag_is_not_terminated(modal: FakeModal) -> None:
    runtime = ModalSandboxRuntime(image="image:base")
    weird = await modal.Sandbox.create.aio(
        image="x", tags={SESSION_TAG: fx.SESSION, GENERATION_TAG: "nope"}
    )

    await runtime.terminate_strays(fx.SESSION, generation=2)

    assert weird.terminated == 0
    assert await weird.get_tags.aio() == {
        SESSION_TAG: fx.SESSION,
        GENERATION_TAG: "nope",
    }


async def test_an_older_sweep_leaves_a_newer_sandbox(modal: FakeModal) -> None:
    runtime = ModalSandboxRuntime(image="image:base")
    newer = await modal.Sandbox.create.aio(
        image="x", tags={SESSION_TAG: fx.SESSION, GENERATION_TAG: "2"}
    )
    same = await modal.Sandbox.create.aio(
        image="x", tags={SESSION_TAG: fx.SESSION, GENERATION_TAG: "2"}
    )
    older = await modal.Sandbox.create.aio(
        image="x", tags={SESSION_TAG: fx.SESSION, GENERATION_TAG: "1"}
    )
    untagged = await modal.Sandbox.create.aio(image="x", tags={SESSION_TAG: fx.SESSION})

    await runtime.terminate_strays(fx.SESSION, generation=2)

    assert newer.terminated == 0
    assert same.terminated == 0
    assert older.terminated == 1
    assert untagged.terminated == 1
    assert modal.snapshots == []


async def test_an_expired_snapshot_is_non_retryable(
    tmp_path: Path, modal: FakeModal
) -> None:
    cli = fx.completed_cli(tmp_path)
    modal.gone_images.add("image:im-expired")

    with pytest.raises(ApplicationError) as failed:
        await ActivityEnvironment().run(
            _activity(_runtime(tmp_path, cli), tmp_path).run_turn,
            _input(resume=True, sandbox_snapshot="im-expired"),
        )

    assert failed.value.non_retryable
    assert failed.value.type == CLAUDE_TURN_REJECTED_ERROR_TYPE
    assert "snapshot im-expired is expired or missing" in str(failed.value)
    assert "start a new session" in str(failed.value)


async def test_a_missing_base_image_is_not_blamed_on_a_snapshot(
    tmp_path: Path, modal: FakeModal
) -> None:
    cli = fx.completed_cli(tmp_path)
    modal.gone_images.add("image:base")

    with pytest.raises(NotFoundError):
        await ActivityEnvironment().run(
            _activity(_runtime(tmp_path, cli), tmp_path).run_turn, _input()
        )


@pytest.mark.parametrize("call", ["lookup", "create", "list", "get_tags"])
async def test_a_transient_provider_error_is_a_lost_sandbox(
    tmp_path: Path, modal: FakeModal, call: str
) -> None:
    """Nothing ran, so the turn is retryable from the snapshot it started from."""

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise ConnectionError("Modal is unavailable")

    if call == "get_tags":
        await modal.Sandbox.create.aio(image="x", tags={SESSION_TAG: fx.SESSION})
        modal.sandboxes[0].get_tags = _Aio(unavailable)
    else:
        target = modal.App if call == "lookup" else modal.Sandbox
        setattr(target, call, _Aio(unavailable))

    with pytest.raises(ApplicationError) as lost:
        await ActivityEnvironment().run(
            _activity(
                _runtime(tmp_path, fx.completed_cli(tmp_path)), tmp_path
            ).run_turn,
            _input(sandbox_snapshot="im-old"),
        )

    assert lost.value.type == CLAUDE_SANDBOX_LOST_ERROR_TYPE
    assert not lost.value.non_retryable
    assert "Modal is unavailable" in str(lost.value)


@pytest.mark.parametrize("call", ["create", "list"])
async def test_a_provider_auth_error_is_not_a_lost_sandbox(
    tmp_path: Path, modal: FakeModal, call: str
) -> None:
    """Retrying cannot fix credentials or configuration, so the turn is not resumed."""

    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise AuthError("token rejected")

    setattr(modal.Sandbox, call, _Aio(denied))

    with pytest.raises(Exception) as failed:
        await ActivityEnvironment().run(
            _activity(
                _runtime(tmp_path, fx.completed_cli(tmp_path)), tmp_path
            ).run_turn,
            _input(sandbox_snapshot="im-old"),
        )

    assert getattr(failed.value, "type", None) != CLAUDE_SANDBOX_LOST_ERROR_TYPE
    assert "token rejected" in str(failed.value)


def _parameter_names(method: Any) -> tuple[str, ...]:
    return tuple(
        name for name in inspect.signature(method).parameters if name != "self"
    )


def test_alternate_runtimes_match_the_protocol() -> None:
    """Modal and the in-process fake expose the same runtime and sandbox operations."""
    for name in ("cwd", "start", "terminate_strays"):
        expected = _parameter_names(getattr(sandbox_module.SandboxRuntime, name))
        assert _parameter_names(getattr(ModalSandboxRuntime, name)) == expected
        assert (
            _parameter_names(getattr(fake_sandbox.FakeSandboxRuntime, name)) == expected
        )
    for name in ("exec", "snapshot", "terminate"):
        expected = _parameter_names(getattr(sandbox_module.Sandbox, name))
        assert _parameter_names(getattr(ModalSandbox, name)) == expected
        assert _parameter_names(getattr(fake_sandbox.FakeSandbox, name)) == expected
    modal_runtime = ModalSandboxRuntime(image="x")
    fake_runtime = fake_sandbox.FakeSandboxRuntime("cli", "/work")
    assert isinstance(modal_runtime.supports_native_tools, bool)
    assert isinstance(fake_runtime.supports_native_tools, bool)


@pytest.mark.parametrize(
    ("runtime", "supported"), [("vm", True), ("gvisor", False), (None, False)]
)
def test_only_the_vm_runtime_supports_native_tools(
    runtime: Literal["gvisor", "vm"] | None, supported: bool
) -> None:
    assert (
        ModalSandboxRuntime(image="x", runtime=runtime).supports_native_tools
        is supported
    )


_ = (_skip_version_probe, _verified_pair)
