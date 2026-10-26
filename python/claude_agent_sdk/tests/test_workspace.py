"""Workspace snapshots: an agent's folder saved to a store and restored on any Worker.

These tests drive the runner's workspaces directly, with a ``FileWorkspaceStore``. A
"Worker on another machine" is a second ``Workspaces`` with a root of its own (paths
in a snapshot are relative to the agent's folder). The real engine and the Workflow
are in ``test_workspace_failover.py``.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import (
    WORKSPACE_CORRUPT,
    WORKSPACE_INVALID,
    WORKSPACE_NOT_FOUND,
    WORKSPACE_TOO_LARGE,
    ClaudeAgentSdkRunner,
    FileWorkspaceStore,
    Workspace,
    _workspace,
)
from temporalio.claude_agent_sdk._workspace import CONTROL, Workspaces
from temporalio.exceptions import ApplicationError

POSIX = sys.platform != "win32"
OLD_NS = 631_152_000_123_456_700  # 1990-01-01, to 100 ns (what NTFS keeps)


def can_symlink(where: Path) -> bool:
    probe = where / ".symlink-probe"
    try:
        os.symlink("x", probe)
    except OSError:
        return False
    probe.unlink()
    return True


def workspaces(root: Path, store: Any, **options: Any) -> Workspaces:
    return Workspaces(
        str(root),
        store,
        max_bytes=options.pop("max_bytes", None),
        exclude=options.pop("exclude", ()),
    )


def describe(folder: Path) -> dict[str, Any]:
    """Every entry of a folder, as a snapshot keeps it."""
    found: dict[str, Any] = {}
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder).as_posix()
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            found[rel] = ("link", os.readlink(path))
        elif stat.S_ISDIR(st.st_mode):
            found[rel] = ("dir", stat.S_IMODE(st.st_mode) if POSIX else None)
        elif stat.S_ISREG(st.st_mode):
            found[rel] = (
                "file",
                hashlib.sha256(path.read_bytes()).hexdigest(),
                stat.S_IMODE(st.st_mode) if POSIX else None,
                st.st_mtime_ns,
            )
    return found


async def save(ws: Workspaces, snapshot: str | None, change: Any = None) -> str:
    """One step: restore ``snapshot``, run ``change(folder)``, save."""
    async with ws.use(Workspace("agent", snapshot)) as use:
        if change is not None:
            change(Path(use.folder))
        return await use.save()


def make_varied(folder: Path, links: bool) -> None:
    (folder / "src" / "deep" / "er").mkdir(parents=True)
    (folder / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (folder / "src" / "deep" / "er" / "data.bin").write_bytes(bytes(range(256)) * 9)
    (folder / "empty").mkdir()
    (folder / ".hidden").write_bytes(b"\x00secret\xff")
    (folder / "run.sh").write_text("#!/bin/sh\necho run\n", encoding="utf-8")
    (folder / "café").mkdir()
    (folder / "café" / "日本.txt").write_text("unicode\n", encoding="utf-8")
    (folder / "old.txt").write_text("old\n", encoding="utf-8")
    os.utime(folder / "old.txt", ns=(OLD_NS, OLD_NS))
    (folder / "locked").mkdir()
    (folder / "locked" / "inside.txt").write_text("in\n", encoding="utf-8")
    (folder / "readonly.txt").write_text("ro\n", encoding="utf-8")
    if POSIX:
        os.chmod(folder / "run.sh", 0o755)
        os.chmod(folder / "readonly.txt", 0o444)
        os.chmod(folder / "locked", 0o555)
    else:
        os.chmod(folder / "readonly.txt", stat.S_IREAD)
    if links:
        os.symlink("src/main.py", folder / "to-main")
        os.symlink("/nowhere/at/all", folder / "dangling")
        os.symlink("src", folder / "to-src", target_is_directory=True)


async def test_a_folder_comes_back_exactly(tmp_path: Path) -> None:
    """Files (bytes, mode, mtime to the nanosecond), empty and read-only directories,
    hidden files, non-ASCII names and links (never followed) come back as they were
    on another Worker, and the restored folder saves to the same snapshot."""
    store = FileWorkspaceStore(tmp_path / "store")
    one = workspaces(tmp_path / "w1", store)
    links = can_symlink(tmp_path)
    snapshot = await save(one, None, lambda f: make_varied(f, links))
    two = workspaces(tmp_path / "w2", store)
    async with two.use(Workspace("agent", snapshot)) as use:
        assert use.changed
        again = await use.save()
    original = describe(tmp_path / "w1" / "agent")
    assert describe(tmp_path / "w2" / "agent") == original
    assert original["old.txt"][3] == OLD_NS
    assert again == snapshot  # the same folder, the same key
    if links:
        assert original["to-main"] == ("link", "src/main.py")
        assert original["dangling"] == ("link", "/nowhere/at/all")


async def test_a_restore_changes_only_what_differs(tmp_path: Path) -> None:
    """A folder changed in every way comes back to the snapshot: extra entries go,
    kinds swap back, bytes, modes and times are fixed. Only files whose bytes
    differ are read from the store; excluded paths and special files stay."""
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w", store, exclude=["node_modules", "*.log"])
    folder = tmp_path / "w" / "agent"
    links = can_symlink(tmp_path)

    def first(f: Path) -> None:
        for name in ("same.txt", "bytes.txt", "mode.txt", "time.txt", "to-dir"):
            (f / name).write_text(f"{name}\n", encoding="utf-8")
        (f / "to-file").mkdir()
        (f / "to-file" / "x.txt").write_text("x\n", encoding="utf-8")
        (f / "from-link").write_text("real\n", encoding="utf-8")

    snapshot = await save(ws, None, first)
    before = describe(folder)

    (folder / "extra.txt").write_text("extra\n", encoding="utf-8")
    (folder / "extra-dir" / "sub").mkdir(parents=True)
    (folder / "extra-dir" / "sub" / "f.txt").write_text("f\n", encoding="utf-8")
    (folder / "bytes.txt").write_text("BYTES.txt\n", encoding="utf-8")  # same size
    if POSIX:
        os.chmod(folder / "mode.txt", 0o600)
    os.utime(folder / "time.txt", ns=(OLD_NS, OLD_NS))
    (folder / "to-dir").unlink()
    (folder / "to-dir").mkdir()
    (folder / "to-dir" / "y.txt").write_text("y\n", encoding="utf-8")
    (folder / "to-file" / "x.txt").unlink()
    (folder / "to-file").rmdir()
    (folder / "to-file").write_text("now a file\n", encoding="utf-8")
    if links:
        (folder / "from-link").unlink()
        os.symlink("same.txt", folder / "from-link")
    (folder / "node_modules" / "pkg").mkdir(parents=True)
    (folder / "node_modules" / "pkg" / "index.js").write_text("js", encoding="utf-8")
    (folder / "build.log").write_text("log\n", encoding="utf-8")
    if sys.platform != "win32":
        os.mkfifo(folder / "pipe")

    calls = ws.store_calls
    async with ws.use(Workspace("agent", snapshot)) as use:
        assert use.changed
    after = describe(folder)
    for kept in ("node_modules", "node_modules/pkg", "node_modules/pkg/index.js"):
        assert kept in after
        del after[kept]
    del after["build.log"]
    assert after == before
    if POSIX:
        assert stat.S_ISFIFO((folder / "pipe").lstat().st_mode)
    # bytes.txt, to-dir and to-file were read again; from-link too if it was a link.
    assert ws.store_calls - calls == 3 + (1 if links else 0)


async def test_in_sync_reads_nothing_and_a_change_saves_only_its_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder as this Worker left it needs no store call and no file read (its
    files' stat is enough); one changed file deep down saves that file and the trees
    on its path, nothing else."""
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w", store)

    def tree(f: Path) -> None:
        for i in range(20):
            (f / "a" / "b").mkdir(parents=True, exist_ok=True)
            (f / f"top{i}.txt").write_text(f"top {i}\n", encoding="utf-8")
            (f / "a" / "b" / f"deep{i}.txt").write_text(f"deep {i}\n", encoding="utf-8")

    def age(f: Path) -> None:  # past the racy window: the next step trusts the stat
        for path in f.rglob("*.txt"):
            os.utime(path, ns=(OLD_NS, OLD_NS))

    first = await save(ws, None, tree)
    first = await save(ws, first, age)  # a step that set the old times
    reads: list[str] = []
    for name in ("_hash_file", "_stage"):
        real = getattr(_workspace, name)

        def counted(*args: Any, real: Any = real, name: str = name) -> Any:
            reads.append(f"{name} {args[0]}")
            return real(*args)

        monkeypatch.setattr(_workspace, name, counted)
    calls = ws.store_calls
    async with ws.use(Workspace("agent", first)) as use:
        assert not use.changed
        assert await use.save() == first
    assert ws.store_calls == calls  # nothing read, nothing saved
    assert reads == []  # and no file read either

    second = await save(
        ws,
        first,
        lambda f: (f / "a" / "b" / "deep3.txt").write_text(
            "changed\n", encoding="utf-8"
        ),
    )
    assert second != first
    assert ws.store_calls - calls == 4  # the file, and the trees a/b, a and the root


async def test_a_missing_or_corrupt_object_stops_the_step(tmp_path: Path) -> None:
    """A snapshot that history accepted must be readable: a missing object or one
    that does not match its key stops the step, not retryable, before the folder
    changes."""
    store = FileWorkspaceStore(tmp_path / "store")
    one = workspaces(tmp_path / "w1", store)

    def make(f: Path) -> None:
        (f / "notes.txt").write_bytes(b"notes\n")  # the bytes hashed below
        (f / "sub").mkdir()

    snapshot = await save(one, None, make)
    sha = hashlib.sha256(b"notes\n").hexdigest()
    obj = tmp_path / "store" / "agent" / sha[:2] / sha

    obj.write_bytes(b"NOTES\n")  # corrupt
    two = workspaces(tmp_path / "w2", store)
    (tmp_path / "w2" / "agent").mkdir(parents=True)
    (tmp_path / "w2" / "agent" / "mine.txt").write_text("mine\n", encoding="utf-8")
    with pytest.raises(ApplicationError) as caught:
        await save(two, snapshot)
    assert caught.value.type == WORKSPACE_CORRUPT and caught.value.non_retryable
    # Nothing changed in the folder: no file removed, no directory made.
    assert [p.name for p in (tmp_path / "w2" / "agent").iterdir()] == ["mine.txt"]

    obj.unlink()  # missing
    three = workspaces(tmp_path / "w3", store)
    with pytest.raises(ApplicationError) as caught:
        await save(three, snapshot)
    assert caught.value.type == WORKSPACE_NOT_FOUND and caught.value.non_retryable
    assert "agent/" + sha in str(caught.value)

    root = tmp_path / "store" / "agent" / snapshot[:2] / snapshot
    root.write_bytes(b"{}")  # a tree that is not the one saved
    four = workspaces(tmp_path / "w4", store)
    with pytest.raises(ApplicationError) as caught:
        await save(four, snapshot)
    assert caught.value.type == WORKSPACE_CORRUPT


class FlakyStore(FileWorkspaceStore):
    """Fails the first ``failures`` loads with an outage."""

    def __init__(self, root: Path, failures: int) -> None:
        super().__init__(root)
        self.failures = failures
        self.loads = 0

    async def load(self, key: str, path: str) -> None:
        self.loads += 1
        if self.loads <= self.failures:
            raise ConnectionError("the store is down")
        await super().load(key, path)


async def test_a_store_outage_is_tried_three_times_then_left_to_temporal(
    tmp_path: Path,
) -> None:
    store = FileWorkspaceStore(tmp_path / "store")
    snapshot = await save(
        workspaces(tmp_path / "w1", store),
        None,
        lambda f: (f / "a.txt").write_text("a\n", encoding="utf-8"),
    )
    flaky = FlakyStore(tmp_path / "store", failures=2)
    await save(workspaces(tmp_path / "w2", flaky), snapshot)  # third time works
    assert (tmp_path / "w2" / "agent" / "a.txt").read_text(encoding="utf-8") == "a\n"
    down = FlakyStore(tmp_path / "store", failures=100)
    with pytest.raises(ConnectionError):  # retryable: Temporal runs the step again
        await save(workspaces(tmp_path / "w3", down), snapshot)
    assert down.loads == 3


async def test_limits_and_unreadable_files_are_clear_errors(tmp_path: Path) -> None:
    store = FileWorkspaceStore(tmp_path / "store")
    small = workspaces(tmp_path / "w1", store, max_bytes=10)
    with pytest.raises(ApplicationError) as caught:
        await save(small, None, lambda f: (f / "big.txt").write_bytes(b"x" * 11))
    assert caught.value.type == WORKSPACE_TOO_LARGE and caught.value.non_retryable
    assert "workspace_max_bytes" in str(caught.value)

    with pytest.raises(ApplicationError) as caught:
        async with small.use(Workspace("../escape")):
            pass
    assert caught.value.type == WORKSPACE_INVALID

    if sys.platform != "win32" and os.geteuid() != 0:  # root reads any file
        ws = workspaces(tmp_path / "w2", store)

        def unreadable(f: Path) -> None:
            (f / "secret").write_text("s", encoding="utf-8")
            os.chmod(f / "secret", 0)

        with pytest.raises(ApplicationError) as caught:
            await save(ws, None, unreadable)
        assert caught.value.type == WORKSPACE_INVALID
        assert "workspace_exclude" in str(caught.value)


def put_tree(store_root: Path, entries: dict[str, Any]) -> str:
    """Write a tree object straight into a folder store; return its key."""
    data = json.dumps(
        {"entries": entries, "format": 1},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    sha = hashlib.sha256(data).hexdigest()
    path = store_root / "agent" / sha[:2] / sha
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha


@pytest.mark.parametrize(
    "name",
    ["..", ".", "a/b", "", "a\u0000b"],
    ids=["up", "dot", "slash", "empty", "nul"],
)
async def test_a_tree_that_would_leave_the_folder_is_refused(
    tmp_path: Path, name: str
) -> None:
    """A tree whose names leave the folder (written by a broken or hostile store
    client, with valid keys) is refused before the folder changes."""
    store_root = tmp_path / "store"
    blob = b"pwned\n"
    sha = hashlib.sha256(blob).hexdigest()
    (store_root / "agent" / sha[:2]).mkdir(parents=True)
    (store_root / "agent" / sha[:2] / sha).write_bytes(blob)
    root = put_tree(
        store_root,
        {
            name: {
                "type": "file",
                "sha256": sha,
                "size": len(blob),
                "mode": 0o644,
                "mtime_ns": OLD_NS,
            }
        },
    )
    ws = workspaces(tmp_path / "w", FileWorkspaceStore(store_root))
    folder = tmp_path / "w" / "agent"
    folder.mkdir(parents=True)
    (folder / "mine.txt").write_text("mine\n", encoding="utf-8")
    with pytest.raises(ApplicationError) as caught:
        await save(ws, root)
    assert caught.value.type == WORKSPACE_CORRUPT and caught.value.non_retryable
    assert sorted(p.name for p in folder.iterdir()) == ["mine.txt"]
    assert not any(
        p.read_bytes() == blob
        for p in tmp_path.rglob("*")
        if p.is_file() and "store" not in p.parts
    )


async def test_a_tree_with_another_shape_is_refused(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    ws = workspaces(tmp_path / "w", FileWorkspaceStore(store_root))
    for entries in (
        {
            "x": {
                "type": "file",
                "sha256": "0" * 64,
                "size": -1,
                "mode": 0,
                "mtime_ns": 0,
            }
        },
        {"x": {"type": "dir", "sha256": "z" * 64, "mode": 0o755}},
        {"x": {"type": "link", "target": ""}},
        {"x": {"type": "socket"}},
        {"x": {"type": "link", "target": "t", "extra": 1}},
    ):
        with pytest.raises(ApplicationError) as caught:
            await save(ws, put_tree(store_root, entries))
        assert caught.value.type == WORKSPACE_CORRUPT


async def test_a_local_link_never_takes_a_write_out_of_the_folder(
    tmp_path: Path,
) -> None:
    """A link the folder holds where the snapshot has a directory is replaced by the
    directory: the files go inside the folder, never where the link points."""
    if not can_symlink(tmp_path):
        pytest.skip("this machine cannot create symbolic links")
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w", store)

    def make(f: Path) -> None:
        (f / "conf").mkdir()
        (f / "conf" / "settings.txt").write_text("mine\n", encoding="utf-8")

    snapshot = await save(ws, None, make)
    outside = tmp_path / "outside"
    outside.mkdir()
    folder = tmp_path / "w" / "agent"
    (folder / "conf" / "settings.txt").unlink()
    (folder / "conf").rmdir()
    os.symlink(outside, folder / "conf", target_is_directory=True)
    async with ws.use(Workspace("agent", snapshot)) as use:
        assert use.changed
    assert not (folder / "conf").is_symlink()
    assert (folder / "conf" / "settings.txt").read_text(encoding="utf-8") == "mine\n"
    assert list(outside.iterdir()) == []


async def test_the_folder_itself_must_not_be_a_link(tmp_path: Path) -> None:
    if not can_symlink(tmp_path):
        pytest.skip("this machine cannot create symbolic links")
    (tmp_path / "w").mkdir()
    (tmp_path / "elsewhere").mkdir()
    os.symlink(
        tmp_path / "elsewhere", tmp_path / "w" / "agent", target_is_directory=True
    )
    ws = workspaces(tmp_path / "w", FileWorkspaceStore(tmp_path / "store"))
    with pytest.raises(ApplicationError) as caught:
        await save(ws, None)
    assert caught.value.type == WORKSPACE_INVALID


async def test_a_read_only_directory_is_opened_for_the_restore_and_closed_again(
    tmp_path: Path,
) -> None:
    """A directory without write permission still gets its files replaced and its
    extra files removed, and keeps its mode (run as a user without root, where
    permission bits apply)."""
    if not POSIX:
        pytest.skip("Windows has no directory write permission bit")
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w", store)
    locked = tmp_path / "w" / "agent" / "locked"

    def make(f: Path) -> None:
        (f / "locked").mkdir()
        (f / "locked" / "a.txt").write_text("saved\n", encoding="utf-8")
        os.chmod(f / "locked", 0o555)

    snapshot = await save(ws, None, make)
    os.chmod(locked, 0o755)
    (locked / "a.txt").unlink()
    (locked / "a.txt").write_text("other\n", encoding="utf-8")
    (locked / "extra.txt").write_text("extra\n", encoding="utf-8")
    os.chmod(locked, 0o555)
    async with ws.use(Workspace("agent", snapshot)) as use:
        assert use.changed
    assert sorted(p.name for p in locked.iterdir()) == ["a.txt"]
    assert (locked / "a.txt").read_text(encoding="utf-8") == "saved\n"
    assert stat.S_IMODE(locked.stat().st_mode) == 0o555
    os.chmod(locked, 0o755)  # so the test's folder can be removed


async def test_other_bytes_with_the_same_size_and_time_are_found(
    tmp_path: Path,
) -> None:
    """A change that keeps the file's size and puts its time back (``touch -d``)
    still shows: its ctime changed. The next step restores the saved bytes."""
    if not POSIX:
        pytest.skip(
            "Windows reports no change time: such a change shows by size or time"
        )
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w", store)
    path = tmp_path / "w" / "agent" / "notes.txt"

    def write(f: Path) -> None:
        (f / "notes.txt").write_text("saved\n", encoding="utf-8")
        os.utime(f / "notes.txt", ns=(OLD_NS, OLD_NS))

    snapshot = await save(ws, None, write)
    time.sleep(0.05)  # a later ctime, also on coarse clocks
    path.write_text("SAVED\n", encoding="utf-8")
    os.utime(path, ns=(OLD_NS, OLD_NS))
    async with ws.use(Workspace("agent", snapshot)) as use:
        assert use.changed
    assert path.read_text(encoding="utf-8") == "saved\n"


async def test_excluded_paths_are_never_saved_restored_or_removed(
    tmp_path: Path,
) -> None:
    store = FileWorkspaceStore(tmp_path / "store")
    exclude = ["node_modules", "*.log", "build/*", "./cache/"]
    one = workspaces(tmp_path / "w1", store, exclude=exclude)

    def make(f: Path) -> None:
        for rel in (
            "keep.txt",
            "node_modules/a/b.js",
            "src/node_modules/c.js",
            "src/run.log",
            "build/out.o",
            "cache/x",
            "src/build/kept.txt",
        ):
            (f / rel).parent.mkdir(parents=True, exist_ok=True)
            (f / rel).write_text(rel, encoding="utf-8")

    snapshot = await save(one, None, make)
    two = workspaces(tmp_path / "w2", store, exclude=exclude)
    (tmp_path / "w2" / "agent" / "cache").mkdir(parents=True)
    (tmp_path / "w2" / "agent" / "cache" / "local").write_text("l", encoding="utf-8")
    await save(two, snapshot)
    got = sorted(
        p.relative_to(tmp_path / "w2" / "agent").as_posix()
        for p in (tmp_path / "w2" / "agent").rglob("*")
        if p.is_file()
    )
    assert got == ["cache/local", "keep.txt", "src/build/kept.txt"]


async def test_two_holders_of_a_folder_take_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two steps of one agent never use its folder at once: in one Worker process
    (an attempt that still runs), even where file locks do not keep one process's
    holders apart (``flock`` on NFS is a lock of the process), and across processes
    on one machine."""
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w", store)
    events: list[str] = []

    async def hold(name: str, seconds: float) -> None:
        async with ws.use(Workspace("agent")):
            events.append(f"{name} in")
            await asyncio.sleep(seconds)
            events.append(f"{name} out")

    with monkeypatch.context() as patched:
        patched.setattr(_workspace, "_lock_file", lambda fd: None)
        patched.setattr(_workspace, "_unlock_file", lambda fd: None)
        await asyncio.gather(hold("a", 0.3), hold("b", 0.0))
    assert events in (
        ["a in", "a out", "b in", "b out"],
        ["b in", "b out", "a in", "a out"],
    )

    lock = tmp_path / "w" / CONTROL / "agent.lock"
    # Another Worker process holds the folder until the test closes its input.
    holder = (
        "import sys, msvcrt\n"
        "f = open(sys.argv[1], 'r+b')\n"
        "msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)\n"
        "print('held', flush=True)\n"
        "sys.stdin.read()\n"
        if sys.platform == "win32"
        else "import sys, fcntl\n"
        "f = open(sys.argv[1], 'r+b')\n"
        "fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "print('held', flush=True)\n"
        "sys.stdin.read()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", holder, str(lock)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    entered = asyncio.Event()

    async def step() -> None:
        async with ws.use(Workspace("agent")):
            entered.set()

    try:
        assert proc.stdout is not None and proc.stdin is not None
        assert proc.stdout.readline().strip() == b"held"
        task = asyncio.ensure_future(step())
        await asyncio.sleep(0.5)
        assert not entered.is_set()  # it waits for the other process
        proc.stdin.close()  # the other process ends, and its lock with it
        await asyncio.wait_for(task, 30)
    finally:
        proc.kill()
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
    assert entered.is_set()


async def test_the_folder_store_keeps_objects_whole(tmp_path: Path) -> None:
    store = FileWorkspaceStore(tmp_path / "store")
    source = tmp_path / "source"
    source.write_bytes(b"bytes")
    key = "agent/" + hashlib.sha256(b"bytes").hexdigest()
    await store.save(key, str(source))
    await store.save(key, str(source))  # again: harmless
    await store.load(key, str(tmp_path / "copy"))
    assert (tmp_path / "copy").read_bytes() == b"bytes"
    with pytest.raises(KeyError):
        await store.load("agent/" + "0" * 64, str(tmp_path / "none"))
    for bad in ("agent", "../x/" + "0" * 64, "agent/" + "A" * 64, "/abs/" + "0" * 64):
        with pytest.raises(ValueError):
            await store.load(bad, str(tmp_path / "none"))
    left = [p.name for p in (tmp_path / "store").rglob("*") if p.name.endswith(".tmp")]
    assert left == []


def test_the_runner_checks_its_workspace_settings(tmp_path: Path) -> None:
    store = FileWorkspaceStore(tmp_path / "store")

    class NotAStore:
        def save(self, key: str, path: str) -> None:
            del key, path

    class NotAsync(NotAStore):
        def load(self, key: str, path: str) -> None:
            del key, path

    with pytest.raises(TypeError):
        ClaudeAgentSdkRunner(cwd=str(tmp_path), workspace_store=NotAStore())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ClaudeAgentSdkRunner(cwd=str(tmp_path), workspace_store=NotAsync())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ClaudeAgentSdkRunner(
            cwd=str(tmp_path), workspace_store=store, workspace_exclude="node_modules"
        )
    with pytest.raises(ValueError):
        ClaudeAgentSdkRunner(
            cwd=str(tmp_path), workspace_store=store, workspace_max_bytes=0
        )
    with pytest.raises(ValueError):
        ClaudeAgentSdkRunner(
            cwd=str(tmp_path), workspace_store=store, workspace_exclude=["/"]
        )


async def test_a_special_file_in_the_way_of_an_entry_is_replaced(
    tmp_path: Path,
) -> None:
    """A scan leaves FIFOs and sockets out, but where the snapshot has an entry, they
    make way for it."""
    if sys.platform == "win32" or not can_symlink(tmp_path):
        pytest.skip("needs FIFOs and symbolic links")
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w", store)

    def make(f: Path) -> None:
        (f / "d").mkdir()
        os.symlink("d", f / "l")
        (f / "x.txt").write_text("x\n", encoding="utf-8")

    snapshot = await save(ws, None, make)
    folder = tmp_path / "w" / "agent"
    for name in ("d", "l", "x.txt"):
        path = folder / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            path.rmdir()
        os.mkfifo(path)
    async with ws.use(Workspace("agent", snapshot)) as use:
        assert use.changed
    assert (folder / "d").is_dir() and not (folder / "d").is_symlink()
    assert os.readlink(folder / "l") == "d"
    assert (folder / "x.txt").read_text(encoding="utf-8") == "x\n"


async def test_the_control_folder_must_not_be_a_link(tmp_path: Path) -> None:
    """Locks and temporary files go to ``<cwd>/.temporal-workspaces``; a link there
    (made by a command, say) stops the step instead of being followed."""
    if not can_symlink(tmp_path):
        pytest.skip("this machine cannot create symbolic links")
    (tmp_path / "w").mkdir()
    (tmp_path / "elsewhere").mkdir()
    os.symlink(
        tmp_path / "elsewhere", tmp_path / "w" / CONTROL, target_is_directory=True
    )
    ws = workspaces(tmp_path / "w", FileWorkspaceStore(tmp_path / "store"))
    with pytest.raises(ApplicationError) as caught:
        await save(ws, None)
    assert caught.value.type == WORKSPACE_INVALID
    assert list((tmp_path / "elsewhere").iterdir()) == []


async def test_a_restore_never_changes_another_name_of_a_file(tmp_path: Path) -> None:
    """A file the folder shares with another name (a hard link, as package caches
    make) gets a copy of its own before its time or mode is fixed: the other name,
    inside or outside the folder, stays as it was."""
    if not POSIX:
        pytest.skip("hard links with their own modes are a POSIX matter here")
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w", store)
    folder = tmp_path / "w" / "agent"
    outside = tmp_path / "cache.txt"

    def make(f: Path) -> None:
        (f / "a.txt").write_text("shared\n", encoding="utf-8")
        os.chmod(f / "a.txt", 0o644)
        os.utime(f / "a.txt", ns=(OLD_NS, OLD_NS))

    snapshot = await save(ws, None, make)
    os.link(folder / "a.txt", outside)  # the cache's name for the same file
    os.chmod(outside, 0o600)  # changes both names
    later = OLD_NS + 10**9
    os.utime(outside, ns=(later, later))
    async with ws.use(Workspace("agent", snapshot)) as use:
        assert use.changed
    inside = (folder / "a.txt").stat()
    assert (stat.S_IMODE(inside.st_mode), inside.st_mtime_ns) == (0o644, OLD_NS)
    kept = outside.stat()
    assert (stat.S_IMODE(kept.st_mode), kept.st_mtime_ns) == (0o600, later)
    assert inside.st_ino != kept.st_ino
    async with ws.use(Workspace("agent", snapshot)) as use:
        assert not use.changed  # it stays restored


async def test_a_folder_that_cannot_be_read_is_rolled_back(tmp_path: Path) -> None:
    """A command that takes the Worker's access to a directory away: saving it fails
    with a clear error, not retryable; restoring removes it and makes the snapshot's
    one again (run as a user without root, where permission bits apply)."""
    if sys.platform == "win32":
        pytest.skip("POSIX permission bits")
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w", store)
    folder = tmp_path / "w" / "agent"

    def make(f: Path) -> None:
        (f / "d").mkdir()
        (f / "d" / "x.txt").write_text("x\n", encoding="utf-8")

    snapshot = await save(ws, None, make)
    saved_mode = stat.S_IMODE((folder / "d").stat().st_mode)
    try:
        for mode in (0o000, 0o300, 0o600):
            os.chmod(folder / "d", mode)
            if os.geteuid() != 0:  # root reads any folder
                with pytest.raises(ApplicationError) as caught:
                    async with ws.use(Workspace("agent", snapshot)) as use:
                        os.chmod(folder / "d", mode)  # the step's command
                        await use.save()
                assert caught.value.type == WORKSPACE_INVALID
                assert "workspace_exclude" in str(caught.value)
                os.chmod(folder / "d", mode)
            async with ws.use(Workspace("agent", snapshot)):
                pass
            assert stat.S_IMODE((folder / "d").stat().st_mode) == saved_mode
            assert (folder / "d" / "x.txt").read_text(encoding="utf-8") == "x\n"
    finally:
        if (folder / "d").exists():
            os.chmod(folder / "d", 0o755)  # so the test's folder can be removed


async def test_a_new_exclude_pattern_leaves_old_snapshots_paths_alone(
    tmp_path: Path,
) -> None:
    """A snapshot saved before a pattern was added still holds what it now leaves
    out: a restore neither writes those paths nor touches what the folder has there."""
    store = FileWorkspaceStore(tmp_path / "store")
    before = workspaces(tmp_path / "w1", store)

    def make(f: Path) -> None:
        (f / "node_modules" / "pkg").mkdir(parents=True)
        (f / "node_modules" / "pkg" / "index.js").write_text("old", encoding="utf-8")
        (f / "build").mkdir()
        (f / "build" / "out.o").write_text("old", encoding="utf-8")
        (f / "src.txt").write_text("src\n", encoding="utf-8")

    snapshot = await save(before, None, make)
    for root in ("w1", "w2"):  # the same disk, and a new one
        after = workspaces(tmp_path / root, store, exclude=["node_modules", "/build/*"])
        folder = tmp_path / root / "agent"
        (folder / "node_modules").mkdir(parents=True, exist_ok=True)
        (folder / "node_modules" / "local.js").write_text("local", encoding="utf-8")
        (folder / "build").mkdir(exist_ok=True)
        (folder / "build" / "out.o").write_text("local", encoding="utf-8")
        await save(after, snapshot)
        assert (folder / "src.txt").read_text(encoding="utf-8") == "src\n"
        assert (folder / "node_modules" / "local.js").read_text(
            encoding="utf-8"
        ) == "local"
        assert (folder / "build" / "out.o").read_text(encoding="utf-8") == "local"
        if root == "w2":
            assert not (folder / "node_modules" / "pkg").exists()


async def test_links_where_the_runner_works_are_never_followed(tmp_path: Path) -> None:
    """A link put where the runner keeps a workspace's temporary files is removed,
    not followed; the agent's folder replaced by a link stops the save instead of
    saving what the link points to."""
    if not can_symlink(tmp_path):
        pytest.skip("this machine cannot create symbolic links")
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w", store)
    outside = tmp_path / "outside"
    (outside / "deep").mkdir(parents=True)
    (outside / "deep" / "f.txt").write_text("f", encoding="utf-8")
    if POSIX:
        os.chmod(outside / "deep", 0o755)
    snapshot = await save(ws, None)
    temp = tmp_path / "w" / CONTROL / "agent.tmp"
    shutil.rmtree(temp, ignore_errors=True)
    os.symlink(outside, temp, target_is_directory=True)
    await save(ws, snapshot)
    assert (outside / "deep" / "f.txt").read_text(encoding="utf-8") == "f"
    if POSIX:
        assert stat.S_IMODE((outside / "deep").stat().st_mode) == 0o755

    def swap(f: Path) -> None:  # the step's command
        shutil.rmtree(f)
        os.symlink(outside, f, target_is_directory=True)

    with pytest.raises(ApplicationError) as caught:
        await save(ws, snapshot, swap)
    assert caught.value.type == WORKSPACE_INVALID
    assert ws.store_calls == 1  # the empty folder's tree; nothing of the link's


async def test_a_cancelled_step_waits_for_its_file_work(tmp_path: Path) -> None:
    """File work in a thread cannot be stopped: a cancelled step lets it finish
    before it goes on (and unlocks the folder)."""
    done = tmp_path / "done"

    def slow() -> None:
        time.sleep(0.5)
        done.write_text("x", encoding="utf-8")

    task = asyncio.ensure_future(_workspace._in_thread(slow))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert done.exists()


@pytest.mark.parametrize("bad", ["aux", "con.txt", "abc.", "-x", ".hidden", "a/b", ""])
async def test_a_workspace_id_must_be_a_plain_name_everywhere(
    tmp_path: Path, bad: str
) -> None:
    ws = workspaces(tmp_path / "w", FileWorkspaceStore(tmp_path / "store"))
    with pytest.raises(ApplicationError) as caught:
        async with ws.use(Workspace(bad)):
            pass
    assert caught.value.type == WORKSPACE_INVALID


async def test_patterns_with_a_leading_slash_match_from_the_folder(
    tmp_path: Path,
) -> None:
    store = FileWorkspaceStore(tmp_path / "store")
    ws = workspaces(tmp_path / "w1", store, exclude=["/cache", "./tmp/"])

    def make(f: Path) -> None:
        for rel in ("cache/a", "tmp/b", "src/cache/c", "src/tmp/d"):
            (f / rel).parent.mkdir(parents=True, exist_ok=True)
            (f / rel).write_text(rel, encoding="utf-8")

    snapshot = await save(ws, None, make)
    other = workspaces(tmp_path / "w2", store)
    await save(other, snapshot)
    got = sorted(
        p.relative_to(tmp_path / "w2" / "agent").as_posix()
        for p in (tmp_path / "w2" / "agent").rglob("*")
        if p.is_file()
    )
    assert got == ["src/cache/c", "src/tmp/d"]


async def test_an_agent_folder_closed_to_the_worker_is_opened_again(
    tmp_path: Path,
) -> None:
    """A command that takes the Worker's access to the agent's folder itself away
    (``chmod 0 .``): the next step gives it back and goes on."""
    if sys.platform == "win32":
        pytest.skip("POSIX permission bits")
    ws = workspaces(tmp_path / "w", FileWorkspaceStore(tmp_path / "store"))
    folder = tmp_path / "w" / "agent"
    snapshot = await save(
        ws, None, lambda f: (f / "a.txt").write_text("a\n", encoding="utf-8")
    )
    os.chmod(folder, 0)
    await save(ws, snapshot)
    assert (folder / "a.txt").read_text(encoding="utf-8") == "a\n"


async def test_a_file_the_worker_cannot_read_is_replaced_by_a_restore(
    tmp_path: Path,
) -> None:
    """A command that takes the Worker's read access to a file away: the next step
    replaces the file with the snapshot's (run as a user without root, where
    permission bits apply)."""
    if sys.platform == "win32":
        pytest.skip("POSIX permission bits")
    ws = workspaces(tmp_path / "w", FileWorkspaceStore(tmp_path / "store"))
    folder = tmp_path / "w" / "agent"
    snapshot = await save(
        ws, None, lambda f: (f / "notes.txt").write_text("notes\n", encoding="utf-8")
    )
    os.chmod(folder / "notes.txt", 0)
    async with ws.use(Workspace("agent", snapshot)) as use:
        assert use.changed
    assert (folder / "notes.txt").read_text(encoding="utf-8") == "notes\n"


async def test_a_restore_that_may_not_change_the_folder_stops_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory of another user (left by a container, say) that the restore must
    change: the step stops with an error that says so, not retryable."""
    ws = workspaces(tmp_path / "w", FileWorkspaceStore(tmp_path / "store"))
    snapshot = await save(
        ws, None, lambda f: (f / "out.o").write_text("out\n", encoding="utf-8")
    )
    (tmp_path / "w" / "agent" / "out.o").write_text("OUT\n", encoding="utf-8")

    def refused(staged: str, path: str, entry: Any) -> None:
        del staged, entry
        raise PermissionError(errno.EACCES, "Permission denied", path)

    monkeypatch.setattr(_workspace, "_place", refused)
    with pytest.raises(ApplicationError) as caught:
        await save(ws, snapshot)
    assert caught.value.type == WORKSPACE_INVALID and caught.value.non_retryable
    assert "another user" in str(caught.value)


async def test_a_link_put_where_the_lock_file_goes_is_not_followed(
    tmp_path: Path,
) -> None:
    if not can_symlink(tmp_path) or sys.platform == "win32":
        pytest.skip("needs symbolic links and O_NOFOLLOW")
    (tmp_path / "w" / CONTROL).mkdir(parents=True)
    target = tmp_path / "elsewhere.txt"
    os.symlink(target, tmp_path / "w" / CONTROL / "agent.lock")
    ws = workspaces(tmp_path / "w", FileWorkspaceStore(tmp_path / "store"))
    await save(ws, None)
    assert not target.exists()
    assert not (tmp_path / "w" / CONTROL / "agent.lock").is_symlink()


class HangingStore(FileWorkspaceStore):
    """A store whose loads never return."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.cancelled = False

    async def load(self, key: str, path: str) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def test_a_cancelled_step_gives_a_store_call_a_grace_period_then_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileWorkspaceStore(tmp_path / "store")
    snapshot = await save(
        workspaces(tmp_path / "w1", store),
        None,
        lambda f: (f / "a.txt").write_text("a\n", encoding="utf-8"),
    )
    monkeypatch.setattr(_workspace, "_STORE_GRACE", 0.3)
    hanging = HangingStore(tmp_path / "store")
    task = asyncio.ensure_future(save(workspaces(tmp_path / "w2", hanging), snapshot))
    await asyncio.sleep(0.3)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 10
    assert hanging.cancelled
