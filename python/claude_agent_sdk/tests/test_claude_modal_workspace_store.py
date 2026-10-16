"""ModalVolumeWorkspaceStore offline, against an in-memory fake of the Modal surface it uses
(Volume.listdir/read_file/batch_upload/remove_file, Dict.put/get/pop, each as ``.aio``): the
shared-dir store's properties plus Modal's failure windows (a lost Dict.put response, a retried
marker upload, a writer that dies holding a claim).
"""

from __future__ import annotations

import asyncio
import random
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from temporalio.claude_agent_sdk import ModalVolumeWorkspaceStore, _modal_workspace
from temporalio.claude_agent_sdk._workspace import WorkspaceIntegrityError

from .helpers.fixtures import POSIX_ONLY
from .test_claude_workspace import SID, make_workspace, tree

modal = pytest.importorskip("modal", reason="the modal extra is not installed")

pytestmark = POSIX_ONLY


class Crash(Exception):
    """A worker dying at an injected point."""


async def _jitter() -> None:
    # Every remote call yields, sometimes more than once, so concurrent writers interleave.
    for _ in range(random.randint(1, 3)):
        await asyncio.sleep(0)


class FakeVolume:
    """modal.Volume's client API over a dict of path -> bytes. A batch commits all its files at
    once when it closes. The no-overwrite check is not atomic with the write: Modal documents
    last-write-wins and no locking, so the fake grants no more."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.crash_on_commit: str | None = None
        self.lost_response: str | None = None
        self.listdir = SimpleNamespace(aio=self._listdir)
        self.read_file = SimpleNamespace(aio=self._read_file)
        self.batch_upload = SimpleNamespace(aio=self._batch_upload)
        self.remove_file = SimpleNamespace(aio=self._remove_file)

    async def _listdir(self, path: str) -> list[Any]:
        await _jitter()
        prefix = path.strip("/") + "/"
        children = {
            p[len(prefix) :].split("/")[0] for p in self.files if p.startswith(prefix)
        }
        if not children:
            raise modal.exception.NotFoundError(f"No such file or directory: {path}")
        return [SimpleNamespace(path=prefix + c) for c in sorted(children)]

    async def _read_file(self, path: str) -> AsyncIterator[bytes]:
        await _jitter()
        try:
            data = self.files[path.strip("/")]
        except KeyError:
            raise FileNotFoundError(path) from None
        for i in range(0, len(data), 64 * 1024):
            yield data[i : i + 64 * 1024]

    @asynccontextmanager
    async def _batch_upload(self, force: bool = False) -> AsyncIterator[Any]:
        staged: list[tuple[Any, str]] = []
        yield SimpleNamespace(
            put_file=lambda local, remote: staged.append((local, remote))
        )
        uploads = {
            remote.strip("/"): Path(local).read_bytes()
            if isinstance(local, str)
            else local.read()
            for local, remote in staged
        }
        if self.crash_on_commit and any(self.crash_on_commit in p for p in uploads):
            raise Crash("worker died uploading")
        exists = not force and any(p in self.files for p in uploads)
        await _jitter()  # check, then write: not a compare-and-set
        if exists:
            raise FileExistsError(f"{list(uploads)} already exist")
        self.files.update(uploads)
        if self.lost_response and any(self.lost_response in p for p in uploads):
            self.lost_response = None
            raise FileExistsError("retried after a lost response")

    async def _remove_file(self, path: str, recursive: bool = False) -> None:
        assert recursive, "the store removes a session's directory"
        await _jitter()
        prefix = path.strip("/")
        doomed = [p for p in self.files if p == prefix or p.startswith(prefix + "/")]
        if not doomed:
            raise FileNotFoundError(path)
        for p in doomed:
            del self.files[p]


class FakeDict:
    """modal.Dict's put/get/pop; put(skip_if_exists=True) is atomic, as Modal's is."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.lose_put_response = False
        self.crash_after_put = False
        self.put = SimpleNamespace(aio=self._put)
        self.get = SimpleNamespace(aio=self._get)
        self.pop = SimpleNamespace(aio=self._pop)

    async def _put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        await _jitter()
        if skip_if_exists and key in self.data:
            return False
        self.data[key] = value
        if self.crash_after_put:
            self.crash_after_put = False
            raise Crash("worker died after claiming")
        if self.lose_put_response:
            self.lose_put_response = False
            return False
        return True

    async def _get(self, key: str, default: Any = None) -> Any:
        await _jitter()
        return self.data.get(key, default)

    async def _pop(self, key: str, default: Any = None) -> Any:
        await _jitter()
        return self.data.pop(key, default)


@pytest.fixture
def backend() -> tuple[FakeVolume, FakeDict]:
    return FakeVolume(), FakeDict()


async def test_round_trip_and_an_unchanged_workspace_keeps_its_revision(
    tmp_path: Path, backend: tuple[FakeVolume, FakeDict]
) -> None:
    store = ModalVolumeWorkspaceStore(*backend)
    assert await store.revisions(SID) == [] and await store.latest(SID) is None
    source = make_workspace(tmp_path / "a" / SID)

    assert (await store.persist(SID, source)).revision == 1
    assert (await store.persist(SID, source)).revision == 1
    target = tmp_path / "b" / SID
    await ModalVolumeWorkspaceStore(*backend).restore(SID, 1, target)

    assert tree(target) == tree(source)


async def test_concurrent_writers_get_distinct_revisions(
    tmp_path: Path, backend: tuple[FakeVolume, FakeDict]
) -> None:
    random.seed(7)
    sources = [
        make_workspace(tmp_path / f"w{i}", marker=f"writer {i}") for i in range(6)
    ]
    stores = [ModalVolumeWorkspaceStore(*backend) for _ in range(3)]
    snapshots = await asyncio.gather(
        *(stores[i % 3].persist(SID, s) for i, s in enumerate(sources))
    )
    assert sorted(s.revision for s in snapshots) == [1, 2, 3, 4, 5, 6]
    for snapshot, source in zip(snapshots, sources, strict=True):
        restored = tmp_path / f"r{snapshot.revision}"
        await stores[0].restore(SID, snapshot.revision, restored)
        assert tree(restored) == tree(source)


async def test_a_writer_that_dies_holding_a_claim_costs_one_number(
    tmp_path: Path, backend: tuple[FakeVolume, FakeDict]
) -> None:
    volume, claims = backend
    store = ModalVolumeWorkspaceStore(volume, claims)
    first = make_workspace(tmp_path / "a", marker="one")
    await store.persist(SID, first)
    claims.crash_after_put = True
    with pytest.raises(Crash):
        await store.persist(SID, make_workspace(tmp_path / "b", marker="two"))
    assert await store.revisions(SID) == [1], "the dead writer committed nothing"

    third = await store.persist(SID, make_workspace(tmp_path / "c", marker="three"))

    assert third.revision == 3 and await store.revisions(SID) == [1, 3]


async def test_lost_responses_do_not_cost_a_revision(
    tmp_path: Path, backend: tuple[FakeVolume, FakeDict]
) -> None:
    volume, claims = backend
    store = ModalVolumeWorkspaceStore(volume, claims)
    claims.lose_put_response = True
    volume.lost_response = "revisions/1.json"

    assert (await store.persist(SID, make_workspace(tmp_path / "a"))).revision == 1
    assert await store.revisions(SID) == [1]


async def test_a_corrupt_or_missing_snapshot_is_refused_and_the_target_kept(
    tmp_path: Path, backend: tuple[FakeVolume, FakeDict]
) -> None:
    volume, _ = backend
    store = ModalVolumeWorkspaceStore(*backend)
    source = make_workspace(tmp_path / "a")
    snapshot = await store.persist(SID, source)
    target = make_workspace(tmp_path / "target", marker="keep")
    before = tree(target)
    blob = f"{SID}/blobs/{snapshot.sha256}.tar.gz"

    data = bytearray(volume.files[blob])
    data[len(data) // 2] ^= 0xFF
    volume.files[blob] = bytes(data)
    with pytest.raises(WorkspaceIntegrityError, match="corrupt"):
        await store.restore(SID, 1, target)
    del volume.files[blob]
    with pytest.raises(WorkspaceIntegrityError, match="missing its archive"):
        await store.restore(SID, 1, target)

    assert tree(target) == before


async def test_delete_starts_over_and_unsafe_session_ids_are_refused(
    tmp_path: Path, backend: tuple[FakeVolume, FakeDict]
) -> None:
    volume, claims = backend
    store = ModalVolumeWorkspaceStore(volume, claims)
    source = make_workspace(tmp_path / "a")
    await store.persist(SID, source)

    await store.delete(SID)

    assert volume.files == {} and claims.data == {}
    with pytest.raises(KeyError):
        await store.get(SID, 1)
    assert (await store.persist(SID, source)).revision == 1
    with pytest.raises(ValueError, match="invalid session id"):
        await store.persist("../escape", source)


def test_from_name_builds_lazy_handles_and_explains_a_missing_modal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    fake = SimpleNamespace(
        Volume=SimpleNamespace(
            from_name=lambda n, **kw: calls.append(("volume", n, kw))
        ),
        Dict=SimpleNamespace(from_name=lambda n, **kw: calls.append(("dict", n, kw))),
    )
    monkeypatch.setattr(_modal_workspace, "_modal", lambda: fake)
    ModalVolumeWorkspaceStore.from_name("ws", environment_name="dev")
    assert [(kind, name) for kind, name, _ in calls] == [
        ("volume", "ws"),
        ("dict", "ws-claims"),
    ]
    assert all(kw["environment_name"] == "dev" for _, _, kw in calls)

    monkeypatch.setitem(sys.modules, "modal", None)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(ImportError, match=r"temporalio-claude-agent-sdk\[modal\]"):
        ModalVolumeWorkspaceStore.from_name("ws")


def test_importing_the_package_does_not_import_modal() -> None:
    code = (
        "import sys, temporalio.claude_agent_sdk as c; "
        "assert c.ModalVolumeWorkspaceStore; assert 'modal' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
