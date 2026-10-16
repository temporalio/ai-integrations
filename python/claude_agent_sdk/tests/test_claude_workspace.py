"""Workspace store round trips, atomicity, restore refusal, and wiring on the turn Activity."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import stat
import tarfile
import threading
import uuid
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from temporalio.claude_agent_sdk import _workspace
from temporalio.claude_agent_sdk._activity import ClaudeTurnActivity
from temporalio.claude_agent_sdk._models import (
    CLAUDE_RESULT_ERROR_TYPE,
    CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE,
    CLAUDE_WORKSPACE_RESTORE_ERROR_TYPE,
    ClaudeDeferredTool,
    ClaudeNativeDecision,
    ClaudeTurnInput,
)
from temporalio.claude_agent_sdk._workspace import (
    DEFAULT_IGNORE,
    SharedDirWorkspaceStore,
    WorkspaceIntegrityError,
    WorkspaceTooLarge,
)
from temporalio.claude_agent_sdk.testing import FakeClaudeCLI
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from .helpers import fixtures as fx
from .helpers.fixtures import POSIX_ONLY

pytestmark = POSIX_ONLY

SID = "0b6a3c3e-6f1e-4d2a-9a8e-2f4a1c9d7e10"


def tree(root: Path) -> dict[str, tuple]:
    out: dict[str, tuple] = {".": ("dir", stat.S_IMODE(root.stat().st_mode))}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        mode = stat.S_IMODE(p.lstat().st_mode)
        if p.is_symlink():
            out[rel] = ("link", os.readlink(p))
        elif p.is_dir():
            out[rel] = ("dir", mode)
        else:
            out[rel] = ("file", p.read_bytes(), mode)
    return out


def make_workspace(root: Path, marker: str = "v1") -> Path:
    (root / "src/pkg/deep").mkdir(parents=True)
    (root / "empty/nested-empty").mkdir(parents=True)
    (root / "src/pkg/deep/notes.md").write_text(f"# {marker}\n")
    (root / "blob.bin").write_bytes(bytes(range(256)) * 64)
    run = root / "run.sh"
    run.write_text("#!/bin/sh\necho hi\n")
    run.chmod(0o755)
    (root / "secret").write_text("private")
    (root / "secret").chmod(0o600)
    (root / "link-to-notes").symlink_to("src/pkg/deep/notes.md")
    root.chmod(0o755)
    return root


def store_at(tmp_path: Path, **kwargs: Any) -> SharedDirWorkspaceStore:
    return SharedDirWorkspaceStore(tmp_path / "shared-store", **kwargs)


def install_archive(
    store_root: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]
) -> int:
    """Commit a hand-built archive as revision 1, following the documented layout."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    raw = buf.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    session = store_root / SID
    (session / "blobs").mkdir(parents=True)
    (session / "revisions").mkdir()
    (session / "blobs" / f"{digest}.tar.gz").write_bytes(raw)
    marker = {
        "session_id": SID,
        "revision": 1,
        "sha256": digest,
        "size": len(raw),
        "created_at": 0.0,
    }
    (session / "revisions" / "1.json").write_text(json.dumps(marker))
    return 1


def member(
    name: str, type: bytes = tarfile.REGTYPE, data: bytes = b"x", linkname: str = ""
) -> tuple[tarfile.TarInfo, bytes | None]:
    info = tarfile.TarInfo(name)
    info.type = type
    info.linkname = linkname
    info.mode = 0o755 if type == tarfile.DIRTYPE else 0o644
    if type == tarfile.REGTYPE:
        info.size = len(data)
        return info, data
    return info, None


async def test_round_trip_onto_another_workers_path(tmp_path: Path) -> None:
    source = make_workspace(tmp_path / "worker-a" / SID)
    store = store_at(tmp_path)

    revision = (await store.persist(SID, source)).revision
    target = tmp_path / "worker-b" / "elsewhere" / SID  # parent does not exist yet
    await store.restore(SID, revision, target)

    assert revision == 1
    assert tree(target) == tree(source)
    assert os.access(target / "run.sh", os.X_OK)
    assert (target / "empty/nested-empty").is_dir()


async def test_restore_replaces_stale_content_and_none_is_a_no_op(
    tmp_path: Path,
) -> None:
    source = make_workspace(tmp_path / "a")
    store = store_at(tmp_path)
    r1 = (await store.persist(SID, source)).revision
    target = tmp_path / "b"
    target.mkdir()
    (target / "stale.txt").write_text("from an older revision")

    pass
    assert (target / "stale.txt").exists()
    await store.restore(SID, r1, target)
    assert tree(target) == tree(source)
    assert not any(
        p.name.startswith(".b.") for p in tmp_path.iterdir()
    )  # no staging left behind


async def test_unchanged_workspace_keeps_its_revision(tmp_path: Path) -> None:
    source = make_workspace(tmp_path / "ws")
    store = store_at(tmp_path)
    assert (await store.persist(SID, source)).revision == 1
    assert (await store.persist(SID, source)).revision == 1
    (source / "new.txt").write_text("change")
    assert (await store.persist(SID, source)).revision == 2
    assert await store.revisions(SID) == [1, 2]
    assert (await store.latest(SID)).revision == 2  # type: ignore[union-attr]


async def test_concurrent_writers_get_distinct_revisions(tmp_path: Path) -> None:
    sources = [
        make_workspace(tmp_path / f"w{i}", marker=f"writer {i}") for i in range(8)
    ]
    # Two store objects on one root, as two workers mounting the same share.
    stores = [store_at(tmp_path), store_at(tmp_path)]

    snapshots = await asyncio.gather(
        *(stores[i % 2].persist(SID, s) for i, s in enumerate(sources))
    )
    revisions = [snap.revision for snap in snapshots]

    assert sorted(revisions) == list(range(1, 9))
    for revision, source in zip(revisions, sources):
        target = tmp_path / f"restored-{revision}"
        await stores[0].restore(SID, revision, target)
        assert tree(target) == tree(source)


async def test_a_crash_before_commit_leaves_the_previous_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_workspace(tmp_path / "ws")
    store = store_at(tmp_path)
    r1 = (await store.persist(SID, source)).revision
    before = tree(source)
    (source / "src/pkg/deep/notes.md").write_text("half-done edit\n")

    def crash(*_args: object, **_kwargs: Any) -> None:
        raise RuntimeError("worker died")

    monkeypatch.setattr("temporalio.claude_agent_sdk._workspace.os.link", crash)
    with pytest.raises(RuntimeError):
        (await store.persist(SID, source)).revision
    monkeypatch.undo()

    def crash_mid_pack(_source: Path, _entries: object, out: BinaryIO) -> None:
        out.write(b"\x1f\x8b partial")
        raise RuntimeError("worker died")

    monkeypatch.setattr(_workspace, "pack_workspace", crash_mid_pack)
    with pytest.raises(RuntimeError):
        (await store.persist(SID, source)).revision
    monkeypatch.undo()

    # A SIGKILL skips cleanup: debris in tmp/ and a stray marker name must not count.
    (tmp_path / "shared-store" / SID / "tmp" / "debris.tar.gz").write_bytes(b"junk")
    (tmp_path / "shared-store" / SID / "revisions" / "2.json.tmp").write_text("{")

    assert await store.revisions(SID) == [r1]
    target = tmp_path / "other-worker"
    await store.restore(SID, r1, target)
    assert tree(target) == before


async def test_a_corrupt_snapshot_is_detected_and_the_target_kept(
    tmp_path: Path,
) -> None:
    store = store_at(tmp_path)
    revision = (await store.persist(SID, make_workspace(tmp_path / "ws"))).revision
    (blob,) = (tmp_path / "shared-store" / SID / "blobs").iterdir()
    raw = bytearray(blob.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    blob.write_bytes(bytes(raw))
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("untouched")

    with pytest.raises(WorkspaceIntegrityError, match="corrupt"):
        await store.restore(SID, revision, target)
    assert tree(target) == {
        ".": tree(target)["."],
        "keep.txt": ("file", b"untouched", 0o644),
    }


async def test_a_missing_snapshot_blob_fails_and_an_older_revision_still_restores(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ws"
    source.mkdir()
    (source / "notes.txt").write_text("v1")
    store = store_at(tmp_path)
    first = await store.persist(SID, source)
    (source / "notes.txt").write_text("v2")
    second = await store.persist(SID, source)
    assert (first.revision, second.revision) == (1, 2)
    (tmp_path / "shared-store" / SID / "blobs" / f"{second.sha256}.tar.gz").unlink()
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("untouched")
    kept = tree(target)

    with pytest.raises(FileNotFoundError):
        await store.restore(SID, second.revision, target)

    assert tree(target) == kept
    assert await store.revisions(SID) == [1, 2]
    await store.restore(SID, first.revision, target)
    assert (target / "notes.txt").read_text() == "v1"
    assert not (target / "keep.txt").exists()


@pytest.mark.parametrize(
    "members",
    [
        pytest.param([member("../escape")], id="dotdot"),
        pytest.param([member("/tmp/abs-escape")], id="absolute"),
        pytest.param([member("a/../../escape")], id="nested-dotdot"),
        pytest.param(
            [member("etc", tarfile.SYMTYPE, linkname="/etc")], id="symlink-absolute"
        ),
        pytest.param(
            [member("up", tarfile.SYMTYPE, linkname="../../outside")], id="symlink-up"
        ),
        pytest.param(
            [member("door", tarfile.SYMTYPE, linkname=".."), member("door/escape")],
            id="write-through-symlink",
        ),
        pytest.param(
            [member("hard", tarfile.LNKTYPE, linkname="/etc/passwd")], id="hardlink-out"
        ),
        pytest.param([member("dev", tarfile.CHRTYPE)], id="device"),
        pytest.param([member("pipe", tarfile.FIFOTYPE)], id="fifo"),
    ],
)
async def test_restore_refuses_entries_that_escape_the_target(
    tmp_path: Path,
    members: list[tuple[tarfile.TarInfo, bytes | None]],
) -> None:
    store_root = tmp_path / "shared-store"
    revision = install_archive(store_root, [member("fine.txt"), *members])
    store = SharedDirWorkspaceStore(store_root)
    target = tmp_path / "w" / "ws"
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("untouched")
    outside_before = sorted(p.name for p in tmp_path.iterdir())

    with pytest.raises(WorkspaceIntegrityError, match="refused"):
        await store.restore(SID, revision, target)

    assert sorted(p.name for p in target.iterdir()) == ["keep.txt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == outside_before
    assert sorted(p.name for p in (tmp_path / "w").iterdir()) == ["ws"]
    assert not Path("/tmp/abs-escape").exists()


async def test_persist_leaves_out_symlinks_that_leave_the_workspace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ws"
    (source / "sub").mkdir(parents=True)
    (source / "real.txt").write_text("ok")
    (source / "inside").symlink_to("real.txt")
    (source / "sub" / "sibling").symlink_to("../real.txt")
    (source / "abs").symlink_to("/etc")
    (source / "sub" / "up").symlink_to("../../outside")
    store = store_at(tmp_path)

    revision = (await store.persist(SID, source)).revision
    target = tmp_path / "target"
    await store.restore(SID, revision, target)

    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*")) == [
        "inside",
        "real.txt",
        "sub",
        "sub/sibling",
    ]


async def test_ignore_patterns_and_the_native_state_dir_are_excluded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ws"
    (source / ".native-tool-state/decisions").mkdir(parents=True)
    (source / ".native-tool-state/decisions/toolu_1.json").write_text(
        '{"decision": "approved"}'
    )
    (source / "node_modules/pkg").mkdir(parents=True)
    (source / "node_modules/pkg/index.js").write_text("x")
    (source / "logs").mkdir()
    (source / "logs/run.log").write_text("x")
    (source / "keep.txt").write_text("kept")
    store = store_at(
        tmp_path,
        ignore=(*DEFAULT_IGNORE, ".native-tool-state", "node_modules", "*.log"),
    )

    revision = (await store.persist(SID, source)).revision
    target = tmp_path / "target"
    await store.restore(SID, revision, target)

    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*")) == [
        "keep.txt",
        "logs",
    ]


async def test_size_and_entry_bounds_are_enforced(tmp_path: Path) -> None:
    source = tmp_path / "ws"
    source.mkdir()
    (source / "big.bin").write_bytes(b"\0" * 1001)

    with pytest.raises(WorkspaceTooLarge, match="1000 bytes"):
        (await store_at(tmp_path, max_bytes=1000).persist(SID, source)).revision
    assert not (
        tmp_path / "shared-store" / SID
    ).exists()  # refused before writing anything

    for i in range(5):
        (source / f"f{i}").write_text("x")
    with pytest.raises(WorkspaceTooLarge, match="5 entries"):
        (await store_at(tmp_path, max_entries=5).persist(SID, source)).revision

    # A snapshot over the reader's own bound (a tampered store, or a decompression bomb).
    revision = (await store_at(tmp_path).persist(SID, source)).revision
    with pytest.raises(WorkspaceTooLarge):
        await store_at(tmp_path, max_bytes=1000).restore(SID, revision, tmp_path / "t")
    assert not (tmp_path / "t").exists()


async def test_delete_and_unsafe_session_ids(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    (await store.persist(SID, make_workspace(tmp_path / "ws"))).revision
    await store.delete(SID)
    assert await store.latest(SID) is None
    assert await store.revisions(SID) == []
    with pytest.raises(KeyError):
        await store.get(SID, 1)
    await store.delete(SID)  # idempotent

    for bad in ("../x", "a/b", ".hidden", ""):
        with pytest.raises(ValueError, match="invalid session id"):
            await store.revisions(bad)


COMMAND = {"command": "echo hi > out.txt"}
CALL = ClaudeDeferredTool(tool_use_id="toolu_x", name="Bash", input=COMMAND)


class _FailingStore(SharedDirWorkspaceStore):
    async def persist(self, session_id: str, source_dir: Path) -> Any:
        raise OSError("the share is gone")


def _activity(tmp_path: Path, store: Any, turns: list[list[Any]]) -> ClaudeTurnActivity:
    cli = FakeClaudeCLI(
        tmp_path / "cli",
        turns=turns,
        log=str(tmp_path / "log.jsonl"),
        state=str(tmp_path / "state.json"),
        effects=str(tmp_path / "effects.jsonl"),
    )
    return ClaudeTurnActivity(
        cli_path=cli.cli_path,
        workspace_root=tmp_path / "ws",
        enable_native_tools=True,
        workspace_store=store,
    )


async def _seed(
    store: SharedDirWorkspaceStore, tmp_path: Path, session_id: str, text: str
) -> int:
    seed = tmp_path / f"seed-{text}"
    seed.mkdir()
    (seed / "notes.txt").write_text(text)
    return (await store.persist(session_id, seed)).revision


async def test_cancelled_restore_cannot_overwrite_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel while the real restore thread is paused, then let a successor write."""
    store = SharedDirWorkspaceStore(tmp_path / "store")
    session_id = str(uuid.uuid4())
    old = await _seed(store, tmp_path, session_id, "old")
    new = await _seed(store, tmp_path, session_id, "successor")
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()
    restore = store._restore

    def held(sid: str, revision: int, target: Path) -> Any:
        if revision != old:
            return restore(sid, revision, target)
        loop.call_soon_threadsafe(entered.set)
        try:
            if not release.wait(10):
                raise AssertionError("test did not release restore thread")
            return restore(sid, revision, target)
        finally:
            finished.set()

    monkeypatch.setattr(store, "_restore", held)
    cli = fx.completed_cli(tmp_path)
    activity = ClaudeTurnActivity(
        cli_path=cli.cli_path, workspace_root=tmp_path / "ws", workspace_store=store
    )
    env = ActivityEnvironment()
    first = asyncio.create_task(
        env.run(
            activity.run_turn,
            ClaudeTurnInput(prompt="hi", session_id=session_id, workspace_revision=old),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        env.cancel()
        done, _ = await asyncio.wait({first}, timeout=0.75)
        released_before_drain = bool(done)
        if not released_before_drain:
            release.set()
            await asyncio.gather(first, return_exceptions=True)
        await asyncio.wait_for(
            ActivityEnvironment().run(
                activity.run_turn,
                ClaudeTurnInput(
                    prompt="hi", session_id=session_id, workspace_revision=new
                ),
            ),
            5,
        )
        target = tmp_path / "ws" / session_id / "notes.txt"
        assert target.read_text() == "successor"
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        assert target.read_text() == "successor", (
            f"cancelled restore overwrote successor with {target.read_text()!r}; "
            f"Activity released before drain={released_before_drain}"
        )
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 2)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)


async def test_the_recorded_revision_is_restored_before_the_cli_and_the_run_is_persisted(
    tmp_path: Path,
    verified: None,
) -> None:
    assert verified is None
    store = SharedDirWorkspaceStore(tmp_path / "store")
    session_id = str(uuid.uuid4())
    recorded = await _seed(store, tmp_path, session_id, "recorded")
    await _seed(
        store, tmp_path, session_id, "a zombie's later write"
    )  # the store's latest
    deferring = [
        ["q", [fx.SYSTEM_INIT, *fx.native_round("msg_n", "toolu_x", "Bash", COMMAND)]],
        ["Continue.", fx.answer("Done.")],
    ]
    activity = _activity(tmp_path, store, deferring)
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                session_id: {
                    "id": "toolu_x",
                    "name": "Bash",
                    "input": COMMAND,
                    "writes": {"out.txt": "hi\n"},
                }
            }
        )
    )

    result = await ActivityEnvironment().run(
        activity.run_turn,
        ClaudeTurnInput(
            prompt="Continue.",
            session_id=session_id,
            resume=True,
            native_tools=["Bash"],
            native_decision=ClaudeNativeDecision(call=CALL, approved=True),
            workspace_revision=recorded,
        ),
    )

    assert result.text == "Done." and result.workspace_revision == 3
    restored = tmp_path / "elsewhere"
    await store.restore(session_id, 3, restored)
    # The CLI ran in the recorded revision (a restore after it would have wiped out.txt).
    assert {p.name: p.read_text() for p in restored.iterdir()} == {
        "notes.txt": "recorded",
        "out.txt": "hi\n",
    }
    assert [(c.tool_use_id, c.status) for c in result.native_calls] == [
        ("toolu_x", "finished")
    ]


async def test_an_unchanged_workspace_keeps_its_revision(
    tmp_path: Path, verified: None
) -> None:
    assert verified is None
    store = SharedDirWorkspaceStore(tmp_path / "store")
    session_id = str(uuid.uuid4())
    recorded = await _seed(store, tmp_path, session_id, "v1")
    activity = _activity(tmp_path, store, [["hello", fx.answer("Hi.")]])

    result = await ActivityEnvironment().run(
        activity.run_turn,
        ClaudeTurnInput(
            prompt="hello",
            session_id=session_id,
            workspace_revision=recorded,
        ),
    )

    assert result.workspace_revision == recorded
    assert (tmp_path / "ws" / session_id / "notes.txt").read_text() == "v1"


async def test_a_persist_failure_fails_the_turn_with_its_own_error(
    tmp_path: Path, verified: None
) -> None:
    assert verified is None
    activity = _activity(
        tmp_path,
        _FailingStore(tmp_path / "store"),
        [["hello", fx.answer("Hi.")]],
    )

    with pytest.raises(ApplicationError) as raised:
        await ActivityEnvironment().run(
            activity.run_turn,
            ClaudeTurnInput(
                prompt="hello",
                session_id=str(uuid.uuid4()),
                workspace_revision=None,
            ),
        )

    assert (
        raised.value.type == CLAUDE_WORKSPACE_PERSIST_ERROR_TYPE
        and raised.value.non_retryable
    )
    assert "the share is gone" in raised.value.message
    assert raised.value.details[0]["total_cost_usd"] == 0.0123  # the run still spent


async def test_a_persist_failure_after_a_failed_run_keeps_the_runs_error(
    tmp_path: Path, verified: None
) -> None:
    assert verified is None
    activity = _activity(
        tmp_path, _FailingStore(tmp_path / "store"), [["hello", fx.error_turn()]]
    )

    with pytest.raises(ApplicationError) as raised:
        await ActivityEnvironment().run(
            activity.run_turn,
            ClaudeTurnInput(prompt="hello", session_id=str(uuid.uuid4())),
        )

    assert raised.value.type == CLAUDE_RESULT_ERROR_TYPE
    assert "overloaded" in raised.value.message
    assert raised.value.details[0]["total_cost_usd"] == 0.0123


async def test_a_missing_recorded_revision_fails_before_the_cli_starts(
    tmp_path: Path,
    verified: None,
) -> None:
    assert verified is None
    activity = _activity(
        tmp_path,
        SharedDirWorkspaceStore(tmp_path / "store"),
        [["hello", fx.answer("Hi.")]],
    )

    with pytest.raises(ApplicationError) as raised:
        await ActivityEnvironment().run(
            activity.run_turn,
            ClaudeTurnInput(
                prompt="hello",
                session_id=str(uuid.uuid4()),
                workspace_revision=7,
            ),
        )

    assert (
        raised.value.type == CLAUDE_WORKSPACE_RESTORE_ERROR_TYPE
        and raised.value.non_retryable
    )
    assert not (tmp_path / "log.jsonl").exists(), "the CLI never ran"


async def test_a_missing_snapshot_blob_fails_before_the_cli_starts(
    tmp_path: Path,
    verified: None,
) -> None:
    assert verified is None
    store = SharedDirWorkspaceStore(tmp_path / "store")
    session_id = str(uuid.uuid4())
    recorded = await _seed(store, tmp_path, session_id, "recorded")
    snapshot = await store.get(session_id, recorded)
    (tmp_path / "store" / session_id / "blobs" / f"{snapshot.sha256}.tar.gz").unlink()
    activity = _activity(tmp_path, store, [["hello", fx.answer("Hi.")]])

    with pytest.raises(ApplicationError) as raised:
        await ActivityEnvironment().run(
            activity.run_turn,
            ClaudeTurnInput(
                prompt="hello",
                session_id=session_id,
                workspace_revision=recorded,
            ),
        )

    assert (
        raised.value.type == CLAUDE_WORKSPACE_RESTORE_ERROR_TYPE
        and raised.value.non_retryable
    )
    assert not (tmp_path / "log.jsonl").exists(), "the CLI never ran"
    assert not (tmp_path / "ws" / session_id / "notes.txt").exists()
