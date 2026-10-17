"""Workspace failover: each agent's folder follows it to any Worker.

With ``ClaudeAgentSdkRunner(workspace_store=...)``, each agent works in a folder of its
own, ``<cwd>/<id>``, the same path on every Worker. After each segment and tool step,
the runner saves what changed in the folder to the store before the Activity
returns: each file under the SHA-256 of its bytes, and each directory as a small JSON
tree that lists its entries, like git. The Activity returns the root tree's key, so
the snapshot becomes the agent's current one only when the Workflow records that
result: an attempt that does not complete (its Worker died or was cut off, it timed
out) can save objects, but never make them current. Before each segment and tool
step, the runner makes the Worker's folder match the current snapshot: it reads and
checks every object it needs first, then removes, writes and fixes only what differs.

Kept: regular files (their bytes, permission bits and modification time, to the
nanosecond), directories (permission bits), symbolic links (their target, never
followed), empty directories and hidden files. A hard link is kept as a file. Left
out: sockets, FIFOs, devices, other file systems mounted inside, and paths that match
``workspace_exclude``. Directory modification times are not kept: adding or removing a
file changes them, and tools go by file times.

Based on Brian Strauch's workspace snapshot in the hybrid prototype (``_workspace.py``
on prototype/claude-agent-hybrid), which kept one managed file in a shared database;
and on Cameron Osborne's research for cold restoration (immutable bundles, admitted
through Temporal's history; here, content-addressed objects).
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import stat
import sys
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import (
    Any,
    NamedTuple,
    Protocol,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

from temporalio import activity
from temporalio.exceptions import ApplicationError

from ._models import (
    WORKSPACE_CORRUPT,
    WORKSPACE_INVALID,
    WORKSPACE_NOT_FOUND,
    WORKSPACE_TOO_LARGE,
    Workspace,
    windows_device_name,
    workspace_id_ok,
)

CONTROL = ".temporal-workspaces"
"""Folder under the runner's ``cwd`` with each workspace's lock file and temporary
files (on the same file system as the workspaces, so a file moves into place at once)."""
FORMAT = 1
"""Version of the tree format (``_tree_bytes``)."""
DEFAULT_MAX_BYTES = 1 << 30
"""Default of ``workspace_max_bytes``: 1 GiB of files."""

_SHA_CHARS = frozenset("0123456789abcdef")
_CHUNK = 1 << 20
_RACY_NS = 2_000_000_000
"""A file changed less than this long before a scan is hashed again at the next one
(as git does): a change within the file system's clock tick can leave its size and
times as they were."""
_PARALLEL = 8
"""Files hashed, saved or loaded at once."""
_STORE_TRIES = 3
_STORE_GRACE = 30.0
"""How long a cancelled step waits for a store call to return before it cancels it."""
_INDEXES = 256
"""Workspaces whose last known state a runner keeps (the others are hashed again)."""
_TREES = 4096
"""Parsed trees a runner keeps (they never change: a tree's key is its hash)."""
_CHANGING_TRIES = 3
_NAME_SURROGATE = 0x20000000
"""Windows reparse tags with this bit name another path (junctions, mount points)."""
_SHARING_VIOLATION = 32
"""Windows error: the file is open in another process that does not share it."""
_LOCK_HELD = frozenset(
    {
        errno.EACCES,
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        getattr(errno, "EDEADLK", errno.EAGAIN),
        getattr(errno, "EDEADLOCK", errno.EAGAIN),
    }
)
"""Errors of a lock attempt that mean another holder has the lock."""

_logger = logging.getLogger(__name__)
T = TypeVar("T")


@runtime_checkable
class WorkspaceStore(Protocol):
    """Where a runner keeps workspace snapshots: immutable objects, by key.

    A key is ``<workspace id>/<sha256>``: the agent's folder name, and the SHA-256 of
    the object's bytes (hexadecimal). An object never changes, so a store may skip a
    ``save`` for a key it has, and keep objects as long as any snapshot that uses
    them may be restored (the plugin removes none; remove a workspace's objects, all
    under its ``<workspace id>/`` prefix, once its agent is done).

    Both methods are async, and may be called for many keys at once. Retry short
    outages inside them: the runner tries each call 3 times, then fails the step.
    """

    async def save(self, key: str, path: str) -> None:
        """Keep the bytes of the file at ``path`` under ``key``.

        Return only once they are durable: the Activity's result names them next.
        """
        ...

    async def load(self, key: str, path: str) -> None:
        """Write the bytes kept under ``key`` to a new file at ``path``.

        Raises:
            KeyError: If the store has no object under ``key``.
        """
        ...


class FileWorkspaceStore:
    """A ``WorkspaceStore`` in a folder.

    For tests, one machine, or a shared mount (NFS, EFS, SMB) that every Worker
    reaches. Each object is a file, ``<root>/<workspace id>/<first 2 hex
    digits>/<sha256>``, written to a temporary file, flushed to disk, and moved into
    place (and on Linux and macOS its folder flushed too), so a reader never sees
    part of one.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Create the store.

        Args:
            root: The folder that holds the objects. Created when needed.
        """
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        workspace, _, digest = key.partition("/")
        if not (workspace_id_ok(workspace) and _is_sha(digest)):
            raise ValueError(f"Not a workspace object key: {key!r}")
        return self._root / workspace / digest[:2] / digest

    async def save(self, key: str, path: str) -> None:
        """Copy the file at ``path`` into the store, unless it has ``key`` already."""
        await asyncio.to_thread(self._save, self._path(key), path)

    async def load(self, key: str, path: str) -> None:
        """Copy the object under ``key`` to ``path``; ``KeyError`` if there is none."""
        await asyncio.to_thread(self._load, key, self._path(key), path)

    @staticmethod
    def _save(target: Path, source: str) -> None:
        if target.is_file():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temp)
            with open(temp, "rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(temp, target)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temp.unlink()
        if sys.platform != "win32":  # the new name is durable too
            fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    @staticmethod
    def _load(key: str, source: Path, target: str) -> None:
        try:
            shutil.copyfile(source, target)
        except FileNotFoundError:
            if not source.is_file():
                raise KeyError(key) from None
            raise


# ---- what a snapshot holds ----


class _File(NamedTuple):
    sha: str
    size: int
    mode: int
    mtime_ns: int


class _Dir(NamedTuple):
    mode: int


class _Link(NamedTuple):
    target: str


_Entry = _File | _Dir | _Link


class _Local(NamedTuple):
    """An entry found in the folder."""

    kind: str  # "file", "dir" or "link"
    st: os.stat_result
    target: str | None  # a link's target


def _is_sha(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA_CHARS


def _mode(st: os.stat_result) -> int:
    return stat.S_IMODE(st.st_mode) & 0o777


if sys.platform == "win32":

    def _stat_key(st: os.stat_result) -> tuple[int, ...]:
        """What tells that a file may have changed since it was hashed.

        Size, time and mode only: a folder listing on Windows gives neither the
        file's id (``st_ino`` is 0) nor its change time (``st_ctime`` is the creation
        time there, and the change time from ``os.fstat`` on Python 3.12+).
        """
        return (st.st_size, st.st_mtime_ns, st.st_mode)

else:

    def _stat_key(st: os.stat_result) -> tuple[int, ...]:
        """What tells that a file may have changed since it was hashed (as git's index).

        Its ctime changes with any change, even one that sets the modification time
        back.
        """
        return (
            st.st_size,
            st.st_mtime_ns,
            st.st_ctime_ns,
            st.st_ino,
            st.st_dev,
            st.st_mode,
        )


def _reparse_point(st: os.stat_result) -> bool:
    """A junction or a mount point: another path, which is never followed (Windows).

    Symbolic links are links; reparse points that do not name another path (such
    as deduplicated or cloud files) are files.
    """
    if sys.platform == "win32":
        return bool(st.st_reparse_tag & _NAME_SURROGATE) and not stat.S_ISLNK(
            st.st_mode
        )
    return False


def _other_device(st: os.stat_result, device: int) -> bool:
    """A directory on another file system (Linux, macOS).

    On Windows a mounted volume is a reparse point, and a folder listing gives no
    device.
    """
    return sys.platform != "win32" and st.st_dev != device


def _kind(entry: _Entry) -> str:
    if isinstance(entry, _File):
        return "file"
    return "dir" if isinstance(entry, _Dir) else "link"


def _depth(path: str) -> int:
    return path.count("/") + 1 if path else 0


def _parents(path: str) -> Iterable[str]:
    """``a`` and ``a/b`` for ``a/b/c``."""
    parts = path.split("/")
    for i in range(1, len(parts)):
        yield "/".join(parts[:i])


def _corrupt(problem: str) -> ApplicationError:
    return ApplicationError(
        f"The agent's workspace snapshot cannot be restored: {problem}. The object "
        "does not match what was saved, so the step stops.",
        type=WORKSPACE_CORRUPT,
        non_retryable=True,
    )


def _invalid(problem: str) -> ApplicationError:
    return ApplicationError(problem, type=WORKSPACE_INVALID, non_retryable=True)


def _check_name(name: Any) -> str:
    """A name in a stored tree: one path component that stays inside its folder."""
    if (
        not isinstance(name, str)
        or name in ("", ".", "..")
        or "/" in name
        or "\0" in name
    ):
        raise _corrupt(f"a tree holds an invalid name {name!r}")
    if sys.platform == "win32" and (
        "\\" in name or ":" in name or name[-1] in ". " or windows_device_name(name)
    ):
        raise _corrupt(f"a tree holds a name Windows cannot use: {name!r}")
    return name


def _tree_bytes(entries: dict[str, Any]) -> bytes:
    """A directory's tree: canonical JSON, so the same folder always gets the same key."""
    return json.dumps(
        {"entries": entries, "format": FORMAT},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_tree(data: bytes) -> dict[str, tuple[_Entry, str | None]]:
    """A stored tree's entries, each with its own tree's key if it is a directory.

    Raises:
        ApplicationError: Of type ``WorkspaceCorrupt`` if the tree is not valid.
    """
    try:
        doc = json.loads(data)
    except ValueError:
        raise _corrupt("a tree is not valid JSON") from None
    if (
        not isinstance(doc, dict)
        or doc.get("format") != FORMAT
        or not isinstance(doc.get("entries"), dict)
        or set(doc) != {"entries", "format"}
    ):
        raise _corrupt("a tree has another format")
    parsed: dict[str, tuple[_Entry, str | None]] = {}
    for name, item in doc["entries"].items():
        _check_name(name)
        if not isinstance(item, dict):
            raise _corrupt(f"entry {name!r} is not an object")
        kind = item.get("type")
        mode = item.get("mode")
        if kind == "file":
            sha, size, mtime = (
                item.get("sha256"),
                item.get("size"),
                item.get("mtime_ns"),
            )
            if not (
                _is_sha(sha)
                and _is_int(size)
                and size >= 0
                and _is_int(mtime)
                and _is_int(mode)
                and 0 <= mode <= 0o777
                and set(item) == {"type", "sha256", "size", "mode", "mtime_ns"}
            ):
                raise _corrupt(f"file entry {name!r} is not valid")
            parsed[name] = (_File(sha, size, mode, mtime), None)
        elif kind == "dir":
            sha = item.get("sha256")
            if not (
                _is_sha(sha)
                and _is_int(mode)
                and 0 <= mode <= 0o777
                and set(item) == {"type", "sha256", "mode"}
            ):
                raise _corrupt(f"directory entry {name!r} is not valid")
            parsed[name] = (_Dir(mode), sha)
        elif kind == "link":
            target = item.get("target")
            if not (
                isinstance(target, str)
                and target
                and "\0" not in target
                and set(item) == {"type", "target"}
            ):
                raise _corrupt(f"link entry {name!r} is not valid")
            parsed[name] = (_Link(target), None)
        else:
            raise _corrupt(f"entry {name!r} has an unknown type")
    return parsed


def _build(entries: dict[str, _Entry]) -> tuple[str, dict[str, bytes]]:
    """The trees of a folder: its root's key, and every tree's bytes by key."""
    children: dict[str, dict[str, _Entry]] = {"": {}}
    for path, entry in entries.items():
        parent, _, name = path.rpartition("/")
        children.setdefault(parent, {})[name] = entry
        if isinstance(entry, _Dir):
            children.setdefault(path, {})
    keys: dict[str, str] = {}
    blobs: dict[str, bytes] = {}
    for folder in sorted(children, key=_depth, reverse=True):
        listing: dict[str, Any] = {}
        for name, entry in children[folder].items():
            path = f"{folder}/{name}" if folder else name
            if isinstance(entry, _File):
                listing[name] = {
                    "type": "file",
                    "sha256": entry.sha,
                    "size": entry.size,
                    "mode": entry.mode,
                    "mtime_ns": entry.mtime_ns,
                }
            elif isinstance(entry, _Dir):
                listing[name] = {
                    "type": "dir",
                    "sha256": keys[path],
                    "mode": entry.mode,
                }
            else:
                listing[name] = {"type": "link", "target": entry.target}
        data = _tree_bytes(listing)
        key = hashlib.sha256(data).hexdigest()
        keys[folder] = key
        blobs[key] = data
    return keys[""], blobs


# ---- the folder on this Worker ----


class _Pattern(NamedTuple):
    glob: str
    by_path: bool  # matched against the path from the agent's folder, not the name


def _pattern(raw: Any) -> _Pattern:
    """A ``workspace_exclude`` pattern.

    With a ``/`` at its start or in its middle, it matches the path from the agent's
    folder; with none, or only a trailing one, a name at any depth.
    """
    if not isinstance(raw, str) or "\0" in raw:
        raise ValueError(f"Not a workspace_exclude pattern: {raw!r}")
    glob = raw.removeprefix("./").strip("/")
    if not glob:
        raise ValueError(f"Not a workspace_exclude pattern: {raw!r}")
    return _Pattern(glob, raw.startswith(("/", "./")) or "/" in glob)


def _excluded(path: str, name: str, patterns: Sequence[_Pattern]) -> bool:
    return any(
        fnmatch.fnmatchcase(path if p.by_path else name, p.glob) for p in patterns
    )


def _scan(
    root: str, patterns: Sequence[_Pattern], unreadable: set[str] | None = None
) -> dict[str, _Local]:
    """Every entry in the folder that a snapshot keeps, by its path from the folder.

    An entry removed while the scan runs is left out.

    Args:
        root: The agent's folder.
        patterns: What ``workspace_exclude`` leaves out.
        unreadable: Collects the directories that cannot be listed; without it, one
            stops the scan.

    Raises:
        ApplicationError: Of type ``WorkspaceInvalid`` if a directory cannot be
            listed and ``unreadable`` is None.
    """
    found: dict[str, _Local] = {}
    device = os.lstat(root).st_dev
    stack = [("", root)]
    while stack:
        prefix, folder = stack.pop()
        here: dict[str, _Local] = {}
        try:
            for item in list(os.scandir(folder)):
                path = f"{prefix}/{item.name}" if prefix else item.name
                if _excluded(path, item.name, patterns):
                    continue
                try:
                    st = item.stat(follow_symlinks=False)
                    if stat.S_ISLNK(st.st_mode):
                        here[path] = _Local("link", st, os.readlink(item.path))
                    elif _reparse_point(st):
                        continue  # a junction or a mount point: never followed
                    elif stat.S_ISDIR(st.st_mode):
                        if not _other_device(st, device):  # else mounted here
                            here[path] = _Local("dir", st, None)
                    elif stat.S_ISREG(st.st_mode):
                        here[path] = _Local("file", st, None)
                    # Sockets, FIFOs and devices are left out.
                except FileNotFoundError:
                    continue  # removed meanwhile
        except FileNotFoundError:
            found.pop(prefix, None)  # removed meanwhile
            continue
        except PermissionError as err:  # it cannot be listed, or its entries read
            if getattr(err, "winerror", None) == _SHARING_VIOLATION:
                raise RuntimeError(
                    f"{folder} holds a file open in another process that does not "
                    "share it. Retrying."
                ) from None
            if unreadable is None:
                raise _invalid(
                    f"The agent's workspace cannot be saved: the folder {folder} "
                    f"cannot be read ({err.strerror}). Give the Worker access to it, "
                    "or leave it out with workspace_exclude."
                ) from None
            unreadable.add(prefix)
            continue
        found.update(here)
        stack.extend(
            (path, os.path.join(folder, path.rpartition("/")[2]))
            for path, local in here.items()
            if local.kind == "dir"
        )
    return found


def _open_regular(path: str, *, saving: bool = True) -> int | None:
    """Open the regular file at ``path``, without following a link or waiting.

    Args:
        path: The file.
        saving: Whether the file is read to be saved: one that cannot be read then
            stops the step. Otherwise (a restore), it reads as one that differs.

    Returns:
        Its descriptor; None if it is gone or is something else now (a process
        changed the folder meanwhile), or cannot be read and is not being saved.

    Raises:
        ApplicationError: Of type ``WorkspaceInvalid`` if it cannot be read, saving.
        RuntimeError: If another process has it open and does not share it (Windows;
            Temporal retries).
    """
    if not hasattr(os, "O_NOFOLLOW"):  # Windows: check first
        try:
            if not stat.S_ISREG(os.lstat(path).st_mode):
                return None
        except FileNotFoundError:
            return None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)  # a FIFO put there now does not block
        | getattr(os, "O_BINARY", 0)
    )
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as err:
        if err.errno == errno.ELOOP:
            return None  # a link now
        if getattr(err, "winerror", None) == _SHARING_VIOLATION:
            raise RuntimeError(
                f"{path} is open in another process that does not share it. Retrying."
            ) from None
        if isinstance(err, PermissionError):
            if not saving:
                return None
            raise _invalid(
                f"The agent's workspace cannot be saved: {path} cannot be read "
                f"({err.strerror}). Give the Worker read access to it, or leave it out "
                "with workspace_exclude."
            ) from None
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        return None
    return fd


def _hash_file(path: str) -> tuple[str, int] | None:
    """The SHA-256 and size of a regular file.

    None if it is not one (any more), or cannot be read: then it differs from any
    snapshot's file, and a restore replaces it.
    """
    fd = _open_regular(path, saving=False)
    if fd is None:
        return None
    digest = hashlib.sha256()
    size = 0
    try:
        while chunk := os.read(fd, _CHUNK):
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest(), size


def _stage(path: str, staged: str) -> tuple[str, int, os.stat_result] | None:
    """Copy a file to ``staged`` while hashing it, so the bytes saved are the bytes hashed.

    Returns:
        The SHA-256 and size of the copy, and the file's ``stat`` before it was read
        (what the next scan compares with); None if the file is gone.
    """
    for attempt in range(_CHANGING_TRIES):
        fd = _open_regular(path)
        if fd is None:
            _warn(
                f"{path} was removed or replaced while the agent's workspace was saved."
            )
            return None
        try:
            before = os.fstat(fd)
            digest = hashlib.sha256()
            size = 0
            with open(staged, "wb") as out:
                while chunk := os.read(fd, _CHUNK):
                    digest.update(chunk)
                    out.write(chunk)
                    size += len(chunk)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if _stat_key(before) == _stat_key(after) and size == before.st_size:
            return digest.hexdigest(), size, before
        if attempt + 1 == _CHANGING_TRIES:
            _warn(
                f"{path} kept changing while the agent's workspace was saved (a "
                "process writes to it); its snapshot holds the bytes read last."
            )
            return digest.hexdigest(), size, before
    raise AssertionError("unreachable")


def _make_writable(folder: str, changed: dict[str, int]) -> None:
    """Let the Worker add and remove entries in ``folder``; remember its mode."""
    if sys.platform == "win32":
        return  # a directory's mode does not keep its owner out there
    st = os.lstat(folder)
    wanted = stat.S_IRWXU
    if stat.S_ISDIR(st.st_mode) and st.st_mode & wanted != wanted:
        changed.setdefault(folder, stat.S_IMODE(st.st_mode))
        os.chmod(folder, stat.S_IMODE(st.st_mode) | wanted)


def _open_up(folder: str) -> None:
    """Let the Worker list and remove everything under the directory ``folder``.

    Links are never followed, and a directory's mode is changed only if it keeps
    the Worker out.

    Raises:
        ApplicationError: Of type ``WorkspaceInvalid`` if another file system is
            mounted under it, or a junction is in it: removing the folder would
            remove what they hold.
    """
    device = os.lstat(folder).st_dev
    stack = [folder]
    while stack:
        current = stack.pop()
        if sys.platform != "win32":
            st = os.lstat(current)
            if not stat.S_ISDIR(st.st_mode):
                continue  # replaced meanwhile: rmtree refuses a link
            if st.st_mode & stat.S_IRWXU != stat.S_IRWXU:
                os.chmod(current, stat.S_IMODE(st.st_mode) | stat.S_IRWXU)
        with os.scandir(current) as listing:
            for item in listing:
                st = item.stat(follow_symlinks=False)
                if stat.S_ISLNK(st.st_mode):
                    continue  # removed, never followed
                if _reparse_point(st) or (
                    stat.S_ISDIR(st.st_mode) and _other_device(st, device)
                ):
                    raise _invalid(
                        f"The agent's workspace holds {item.path}, another file system "
                        "or a junction, where the snapshot has nothing: the step stops "
                        "rather than remove what it holds. Leave it out with "
                        "workspace_exclude, or remove it."
                    )
                if stat.S_ISDIR(st.st_mode):
                    stack.append(item.path)
                elif sys.platform == "win32" and not st.st_mode & stat.S_IWRITE:
                    os.chmod(item.path, stat.S_IWRITE)  # Windows keeps read-only files


def _remove(path: str) -> None:
    """Remove whatever is at ``path``.

    A directory goes with all it holds; anything else by its name (a link, never
    what it points to).
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode) and not _reparse_point(st):
        _open_up(path)
        shutil.rmtree(path)
    elif stat.S_ISDIR(st.st_mode):
        os.rmdir(path)  # a junction: removed, never followed
    else:
        try:
            os.unlink(path)
        except PermissionError:
            if sys.platform != "win32":
                raise
            os.chmod(path, stat.S_IWRITE)  # a read-only file, on Windows
            os.unlink(path)


def _clear(path: str) -> None:
    """Remove what a scan leaves out where an entry must go.

    A socket, a FIFO or a junction goes; a directory stays.
    """
    with contextlib.suppress(FileNotFoundError):
        st = os.lstat(path)
        if not stat.S_ISDIR(st.st_mode) or _reparse_point(st):
            _remove(path)


def _place(staged: str, path: str, entry: _File) -> None:
    """Give a checked copy the entry's time and mode, then move it into place."""
    os.utime(staged, ns=(entry.mtime_ns, entry.mtime_ns))
    os.chmod(staged, entry.mode)
    if sys.platform == "win32" and os.path.isfile(path):
        os.chmod(path, stat.S_IWRITE)  # Windows does not replace a read-only file
    try:
        os.replace(staged, path)
    except OSError as err:
        if err.errno != errno.EXDEV:
            raise
        shutil.move(staged, path)  # the temporary folder is on another file system


def _warn(message: str) -> None:
    text = f"temporalio.claude_agent_sdk: {message}"
    if activity.in_activity():
        activity.logger.warning(text)
    else:
        _logger.warning(text)


async def _finished(work: Awaitable[T], grace: float | None = None) -> T:
    """Await ``work``; if cancelled, let it finish first, then go on cancelling.

    File work in a thread cannot be stopped, and a store call writes into the
    workspace's temporary folder: the folder is unlocked only once they are done.

    Args:
        work: What to await.
        grace: How long a cancelled step waits for it (seconds; None: until it is
            done). A store call still running then is cancelled.
    """
    task = asyncio.ensure_future(work)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        loop = asyncio.get_running_loop()
        end = None if grace is None else loop.time() + grace
        while not task.done():
            left = None if end is None else end - loop.time()
            if left is not None and left <= 0:
                task.cancel()
                break
            try:
                await asyncio.wait([task], timeout=left)
            except asyncio.CancelledError:
                continue
        if task.done() and not task.cancelled():
            task.exception()  # retrieved: the cancellation goes on
        raise


async def _in_thread(function: Callable[..., T], *args: Any) -> T:
    """Run blocking file work in a thread (see ``_finished``)."""
    return await _finished(asyncio.to_thread(function, *args))


async def _all(work: Iterable[Awaitable[T]]) -> list[T]:
    """Run at most ``_PARALLEL`` at once; once all ended, raise the first failure."""
    gate = asyncio.Semaphore(_PARALLEL)

    async def one(item: Awaitable[T]) -> T:
        async with gate:
            return await item

    done = await asyncio.gather(*(one(w) for w in work), return_exceptions=True)
    for item in done:
        if isinstance(item, BaseException):
            raise item
    return done  # type: ignore[return-value]


class _Index:
    """What this Worker process knows of a workspace's folder since it last synced it.

    Attributes:
        snapshot: The snapshot the folder matched then (None: empty).
        entries: That snapshot's entries, by path.
        stats: Each file's ``stat`` then (``_stat_key``).
        racy: Files to hash again at the next scan (``_RACY_NS``).
        known: Keys of the snapshot's objects, which the store has.
    """

    def __init__(self) -> None:
        self.snapshot: str | None = None
        self.entries: dict[str, _Entry] = {}
        self.stats: dict[str, tuple[int, ...]] = {}
        self.racy: set[str] = set()
        self.known: set[str] = set()


class _Plan(NamedTuple):
    remove: list[str]
    mkdir: list[str]
    fetch: dict[str, _File]
    meta: dict[str, _File]  # the bytes are right: set the time and mode
    link: dict[str, str]
    modes: dict[str, int]  # directories whose mode to set last

    def empty(self) -> bool:
        return not (
            self.remove
            or self.mkdir
            or self.fetch
            or self.meta
            or self.link
            or self.modes
        )


class Workspaces:
    """A runner's agent folders and their snapshots (see the module's description)."""

    def __init__(
        self,
        root: str,
        store: WorkspaceStore,
        *,
        max_bytes: int | None,
        exclude: Sequence[str],
    ) -> None:
        """Set up workspaces under ``root`` (the runner's ``cwd``)."""
        self.root = os.path.abspath(root)
        self.store = store
        self.max_bytes = max_bytes
        self.patterns = tuple(_pattern(p) for p in exclude)
        self._indexes: OrderedDict[str, _Index] = OrderedDict()
        self._trees: OrderedDict[str, dict[str, tuple[_Entry, str | None]]] = (
            OrderedDict()
        )
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}
        self.store_calls = 0
        """Calls to the store so far (tests check what a step reads and writes)."""

    def folder(self, workspace_id: str) -> str:
        """The agent's own folder: ``<cwd>/<id>``."""
        return os.path.join(self.root, workspace_id)

    @contextlib.asynccontextmanager
    async def use(self, workspace: Workspace) -> AsyncIterator[_Use]:
        """Lock the agent's folder and make it match ``workspace.snapshot``.

        Yields:
            The folder, to run the step in and then save (``_Use.save``).

        Raises:
            ApplicationError: Not retryable: of type ``WorkspaceNotFound`` if an
                object of the snapshot is not in the store, ``WorkspaceCorrupt`` if one
                does not match its key, ``WorkspaceInvalid`` if the workspace id is
                not a plain name or the folder cannot be used.
        """
        if not workspace_id_ok(workspace.id):
            raise _invalid(
                f"Workspace id {workspace.id!r} is not a plain folder name (letters, "
                "digits, '.', '_', '-', at most 128, starting and ending with a "
                "letter or a digit)."
            )
        if workspace.snapshot is not None and not _is_sha(workspace.snapshot):
            raise _corrupt(f"snapshot {workspace.snapshot!r} is not a SHA-256")
        lock, users = self._locks.get(workspace.id, (asyncio.Lock(), 0))
        self._locks[workspace.id] = (lock, users + 1)
        try:
            async with lock:
                fd = await self._file_lock(workspace.id)
                try:
                    use = _Use(self, workspace.id)
                    await use.sync(workspace.snapshot)
                    yield use
                finally:
                    _unlock_file(fd)
                    os.close(fd)
        finally:
            lock, users = self._locks[workspace.id]
            if users == 1:
                del self._locks[workspace.id]
            else:
                self._locks[workspace.id] = (lock, users - 1)

    async def _file_lock(self, workspace_id: str) -> int:
        """The lock that keeps other Worker processes on this machine out of the folder."""
        control = os.path.join(self.root, CONTROL)
        await _in_thread(os.makedirs, control, 0o700, True)
        st = os.lstat(control)
        if not stat.S_ISDIR(st.st_mode) or _reparse_point(st):
            raise _invalid(
                f"{control} is not a plain directory (a link?): the runner keeps its "
                "workspace locks and temporary files there."
            )
        path = os.path.join(control, f"{workspace_id}.lock")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as err:
            if err.errno != errno.ELOOP:
                raise
            os.unlink(path)  # a link put there: removed, never followed
            fd = os.open(path, flags, 0o600)
        try:
            while True:
                try:
                    _lock_file(fd)
                    return fd
                except OSError as err:
                    if err.errno not in _LOCK_HELD:
                        raise _invalid(
                            f"Cannot lock {path} ({err}): the runner's cwd must be on "
                            "a file system with file locks."
                        ) from None
                await asyncio.sleep(0.1)  # an earlier attempt still runs here
        except BaseException:
            os.close(fd)
            raise

    def index(self, workspace_id: str) -> _Index | None:
        """What this process knows of the workspace's folder, if anything."""
        found = self._indexes.get(workspace_id)
        if found is not None:
            self._indexes.move_to_end(workspace_id)
        return found

    def keep_index(self, workspace_id: str, index: _Index) -> None:
        """Remember what this process knows of the folder (the newest ``_INDEXES``)."""
        self._indexes[workspace_id] = index
        self._indexes.move_to_end(workspace_id)
        while len(self._indexes) > _INDEXES:
            self._indexes.popitem(last=False)

    async def call_store(self, save: bool, key: str, path: str) -> None:
        """One store call, tried ``_STORE_TRIES`` times (a missing object at once)."""
        for attempt in range(_STORE_TRIES):
            self.store_calls += 1
            try:
                if save:
                    await _finished(self.store.save(key, path), _STORE_GRACE)
                else:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(path)
                    await _finished(self.store.load(key, path), _STORE_GRACE)
                return
            except KeyError:
                raise
            except Exception:
                if attempt + 1 == _STORE_TRIES:
                    raise
                await asyncio.sleep(0.2 * 2**attempt)

    def cached_tree(self, key: str) -> dict[str, tuple[_Entry, str | None]] | None:
        """A tree this process read already, by its store key (it never changes)."""
        found = self._trees.get(key)
        if found is not None:
            self._trees.move_to_end(key)
        return found

    def keep_tree(self, key: str, tree: dict[str, tuple[_Entry, str | None]]) -> None:
        """Remember a tree read from the store (the newest ``_TREES``)."""
        self._trees[key] = tree
        while len(self._trees) > _TREES:
            self._trees.popitem(last=False)


class _Use:
    """One segment's or tool step's hold on an agent's folder."""

    def __init__(self, workspaces: Workspaces, workspace_id: str) -> None:
        self._ws = workspaces
        self.id = workspace_id
        self.folder = workspaces.folder(workspace_id)
        self._temp = os.path.join(workspaces.root, CONTROL, f"{workspace_id}.tmp")
        self.changed = False
        """Whether the folder had to change to match the snapshot (``sync``)."""
        self._index: _Index | None = None

    def _key(self, sha: str) -> str:
        return f"{self.id}/{sha}"

    def _temp_path(self) -> str:
        return os.path.join(self._temp, uuid.uuid4().hex)

    def _abs(self, path: str) -> str:
        return os.path.join(self.folder, *path.split("/"))

    def _parent(self, path: str) -> str:
        return os.path.dirname(self._abs(path))

    def _check_folder(self) -> None:
        """The agent's folder must be a plain directory the Worker can use.

        Raises:
            ApplicationError: Of type ``WorkspaceInvalid`` if it is a link, a junction
                or something else.
        """
        st = os.lstat(self.folder)
        if not stat.S_ISDIR(st.st_mode) or _reparse_point(st):
            raise _invalid(
                f"The agent's workspace {self.folder} is not a plain directory (a "
                "link, a junction or a file): the step stops rather than go through "
                "it."
            )
        if sys.platform != "win32" and st.st_mode & stat.S_IRWXU != stat.S_IRWXU:
            os.chmod(self.folder, stat.S_IMODE(st.st_mode) | stat.S_IRWXU)

    def _prepare(self) -> None:
        """Make the folder, and an empty temporary folder (a step that died left it)."""
        _remove(self._temp)
        os.mkdir(self._temp, 0o700)
        with contextlib.suppress(FileExistsError):
            os.mkdir(self.folder)
        self._check_folder()

    async def sync(self, wanted: str | None) -> None:
        """Make the folder match snapshot ``wanted`` (None: an empty folder)."""
        await _in_thread(self._prepare)
        unreadable: set[str] = set()
        seen = await _in_thread(_scan, self.folder, self._ws.patterns, unreadable)
        if "" in unreadable:
            raise _invalid(
                f"The agent's workspace {self.folder} cannot be read: does it belong to "
                "another user?"
            )
        index = self._ws.index(self.id)
        if index is not None and index.snapshot == wanted:
            if not unreadable and await self._as_left(seen, index):
                self._index = index
                return
            target, known = index.entries, index.known
        else:
            target, known = await self._load(wanted)
        plan = await self._plan(seen, target, index, unreadable)
        if not plan.empty():
            self.changed = True
            await self._apply(plan)
            seen = await _in_thread(_scan, self.folder, self._ws.patterns, None)
            _check_restored(seen, target)
        fresh = _Index()
        fresh.snapshot = wanted
        fresh.entries = target
        fresh.known = set(known)
        now = time.time_ns()
        for path, local in seen.items():
            if local.kind == "file":
                fresh.stats[path] = _stat_key(local.st)
                if local.st.st_mtime_ns >= now - _RACY_NS:
                    fresh.racy.add(path)
        self._index = fresh
        self._ws.keep_index(self.id, fresh)

    async def _as_left(self, seen: dict[str, _Local], index: _Index) -> bool:
        """Whether the folder is as this process left it (``index``).

        Only files whose ``stat`` changed (or that changed just before it was taken)
        are hashed again; any other difference is a change.
        """
        suspects = _suspects(seen, index)
        if suspects is None:
            return False
        if suspects:
            hashed = await _all(
                _in_thread(_hash_file, self._abs(path)) for path in suspects
            )
            for path, got in zip(suspects, hashed):
                entry = index.entries[path]
                if (
                    got is None
                    or not isinstance(entry, _File)
                    or got != (entry.sha, entry.size)
                ):
                    return False
            now = time.time_ns()
            for path in suspects:
                st = seen[path].st
                index.stats[path] = _stat_key(st)
                if st.st_mtime_ns < now - _RACY_NS:
                    index.racy.discard(path)
        return True

    @staticmethod
    def _indexed_sha(path: str, local: _Local, index: _Index | None) -> str | None:
        """A file's SHA-256 as this process last saw it.

        Only if its ``stat`` says it is unchanged since.
        """
        if (
            index is None
            or path in index.racy
            or index.stats.get(path) != _stat_key(local.st)
        ):
            return None
        entry = index.entries.get(path)
        return entry.sha if isinstance(entry, _File) else None

    async def _tree(self, sha: str) -> dict[str, tuple[_Entry, str | None]]:
        key = self._key(sha)
        found = self._ws.cached_tree(key)
        if found is not None:
            return found
        path = self._temp_path()
        await self._fetch_checked(sha, path, None)
        try:
            data = await _in_thread(Path(path).read_bytes)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
        tree = _parse_tree(data)
        self._ws.keep_tree(key, tree)
        return tree

    async def _fetch_checked(self, sha: str, path: str, size: int | None) -> None:
        """Load an object to ``path`` and check it (twice, then give up)."""
        for _ in range(2):
            try:
                await self._ws.call_store(False, self._key(sha), path)
            except KeyError:
                raise ApplicationError(
                    f"The agent's workspace cannot be restored: object "
                    f"{self._key(sha)} is not in the workspace store. The step stops: "
                    "a snapshot that history accepted must be readable (was it "
                    "removed from the store?).",
                    type=WORKSPACE_NOT_FOUND,
                    non_retryable=True,
                ) from None
            got = await _in_thread(_hash_file, path)
            if got is not None and got[0] == sha and (size is None or got[1] == size):
                return
        raise _corrupt(f"object {self._key(sha)} does not match its key")

    async def _load(self, snapshot: str | None) -> tuple[dict[str, _Entry], set[str]]:
        """A snapshot's entries by path, and the keys of its objects.

        Paths that ``workspace_exclude`` leaves out now are left out here too (a
        snapshot saved before the pattern was added has them): a restore never
        writes them, as a scan never sees them.
        """
        if snapshot is None:
            return {}, set()
        entries: dict[str, _Entry] = {}
        known = {snapshot}
        level = [("", snapshot)]
        while level:
            trees = await _all(self._tree(key) for _, key in level)
            following: list[tuple[str, str]] = []
            for (prefix, _), tree in zip(level, trees):
                for name, (entry, key) in tree.items():
                    path = f"{prefix}/{name}" if prefix else name
                    if _excluded(path, name, self._ws.patterns):
                        continue
                    entries[path] = entry
                    if key is not None:
                        following.append((path, key))
                        known.add(key)
                    elif isinstance(entry, _File):
                        known.add(entry.sha)
            level = following
        return entries, known

    async def _plan(
        self,
        seen: dict[str, _Local],
        target: dict[str, _Entry],
        index: _Index | None,
        unreadable: set[str],
    ) -> _Plan:
        """What to change in the folder so that it matches ``target``.

        A directory that cannot be listed is removed (and made again if the snapshot
        has it).
        """
        remove: list[str] = []
        removed_dirs: set[str] = set()
        for path in sorted(seen, key=_depth):
            if any(p in removed_dirs for p in _parents(path)):
                continue  # goes with its directory
            local = seen[path]
            want = target.get(path)
            if (
                path in unreadable
                or want is None
                or _kind(want) != local.kind
                or (isinstance(want, _Link) and want.target != local.target)
            ):
                remove.append(path)
                if local.kind == "dir":
                    removed_dirs.add(path)
        gone = set(remove)

        def present(path: str) -> _Local | None:
            if path in gone or any(p in removed_dirs for p in _parents(path)):
                return None
            return seen.get(path)

        mkdir: list[str] = []
        fetch: dict[str, _File] = {}
        meta: dict[str, _File] = {}
        link: dict[str, str] = {}
        modes: dict[str, int] = {}
        compare: list[tuple[str, _Local, _File]] = []
        for path, want in target.items():
            here = present(path)
            if isinstance(want, _Dir):
                if here is None:
                    mkdir.append(path)
                    modes[path] = want.mode
                elif _mode(here.st) != want.mode:
                    modes[path] = want.mode
            elif isinstance(want, _Link):
                if here is None:
                    link[path] = want.target
            elif here is None or here.st.st_size != want.size:
                fetch[path] = want
            else:
                compare.append((path, here, want))
        unknown = [p for p, here, _ in compare if not self._indexed_sha(p, here, index)]
        hashed = await _all(_in_thread(_hash_file, self._abs(p)) for p in unknown)
        found = {p: got[0] for p, got in zip(unknown, hashed) if got is not None}
        for path, here, want in compare:
            sha = found.get(path) or self._indexed_sha(path, here, index)
            if sha != want.sha:
                fetch[path] = want
            elif _mode(here.st) != want.mode or here.st.st_mtime_ns != want.mtime_ns:
                meta[path] = want
        mkdir.sort(key=_depth)
        remove.sort(key=_depth, reverse=True)
        return _Plan(remove, mkdir, fetch, meta, link, modes)

    async def _apply(self, plan: _Plan) -> None:
        """Change the folder as planned.

        Every object is read from the store and checked first, so a missing or a
        corrupt one stops the step before anything in the folder changes. Then:
        remove, create directories, place files, make links, fix files' times and
        modes, and directories' modes last.
        """
        by_sha: dict[str, list[tuple[str, _File]]] = {}
        for path, entry in plan.fetch.items():
            by_sha.setdefault(entry.sha, []).append((path, entry))
        staged: dict[str, str] = {}

        async def fetch(sha: str, size: int) -> None:
            staged[sha] = self._temp_path()
            await self._fetch_checked(sha, staged[sha], size)

        await _all(fetch(sha, uses[0][1].size) for sha, uses in by_sha.items())
        opened: dict[str, int] = {}  # directories made writable for now: their mode

        def steps() -> None:
            for path in plan.remove:
                _make_writable(self._parent(path), opened)
                _remove(self._abs(path))
            for path in plan.mkdir:
                _make_writable(self._parent(path), opened)
                _clear(self._abs(path))
                os.mkdir(self._abs(path), 0o700)
            for sha, uses in by_sha.items():
                for i, (path, entry) in enumerate(uses):
                    copy = staged[sha]
                    if i < len(uses) - 1:
                        copy = self._temp_path()
                        shutil.copyfile(staged[sha], copy)
                    _make_writable(self._parent(path), opened)
                    _place(copy, self._abs(path), entry)
            for path, target in plan.link.items():
                _make_writable(self._parent(path), opened)
                absolute = self._abs(path)
                _clear(absolute)
                is_dir = os.path.isdir(os.path.join(os.path.dirname(absolute), target))
                try:
                    os.symlink(target, absolute, target_is_directory=is_dir)
                except OSError as err:
                    if sys.platform != "win32":
                        raise
                    raise _invalid(
                        f"The agent's workspace holds a symbolic link ({path}) that "
                        f"this Worker cannot create ({err}): on Windows, turn on "
                        "Developer Mode or run the Worker with the right to create "
                        "symbolic links."
                    ) from None
            for path, entry in plan.meta.items():
                absolute = self._abs(path)
                if os.lstat(absolute).st_nlink > 1:
                    # Another name shares this file (inside the folder or outside, as
                    # a package cache does): give this name a copy of its own.
                    copy = self._temp_path()
                    shutil.copyfile(absolute, copy)
                    _make_writable(self._parent(path), opened)
                    _place(copy, absolute, entry)
                    continue
                os.chmod(absolute, entry.mode | stat.S_IWUSR)
                os.utime(absolute, ns=(entry.mtime_ns, entry.mtime_ns))
                os.chmod(absolute, entry.mode)
            final = dict(opened)
            for path, mode in plan.modes.items():
                final[self._abs(path)] = mode
            for folder in sorted(final, key=lambda p: p.count(os.sep), reverse=True):
                os.chmod(folder, final[folder])

        def change() -> None:
            try:
                steps()
            except PermissionError as err:
                if getattr(err, "winerror", None) == _SHARING_VIOLATION:
                    raise RuntimeError(
                        f"{err.filename} is open in another process that does not "
                        "share it. Retrying."
                    ) from None
                raise _invalid(
                    f"The agent's workspace cannot be restored: {err.filename} cannot "
                    f"be changed ({err.strerror}); does it belong to another user? "
                    "Remove it, or leave it out with workspace_exclude."
                ) from None

        await _in_thread(change)

    async def save(self) -> str:
        """Save what changed in the folder; return the new snapshot's key.

        Raises:
            ApplicationError: Not retryable: of type ``WorkspaceTooLarge`` if the
                folder's files are larger than ``workspace_max_bytes``, or
                ``WorkspaceInvalid`` if the folder is not a plain directory any more,
                or a file or a directory in it cannot be read.
        """
        assert self._index is not None, "sync first"
        index = self._index
        started = time.time_ns()
        await _in_thread(self._check_folder)
        seen = await _in_thread(_scan, self.folder, self._ws.patterns, None)
        total = sum(local.st.st_size for local in seen.values() if local.kind == "file")
        limit = self._ws.max_bytes
        if limit is not None and total > limit:
            raise ApplicationError(
                f"The agent's workspace ({self.folder}) holds {total:,} bytes of files, "
                f"more than workspace_max_bytes ({limit:,}), so it was not saved. "
                "Leave large files out with workspace_exclude (for example "
                "node_modules or build output), or raise the limit.",
                type=WORKSPACE_TOO_LARGE,
                non_retryable=True,
            )
        entries: dict[str, _Entry] = {}
        stats: dict[str, tuple[int, ...]] = {}
        hashing: list[str] = []
        for path, local in seen.items():
            if local.kind == "dir":
                entries[path] = _Dir(_mode(local.st))
            elif local.kind == "link":
                assert local.target is not None
                entries[path] = _Link(local.target)
            else:
                key = _stat_key(local.st)
                old = index.entries.get(path)
                if (
                    isinstance(old, _File)
                    and index.stats.get(path) == key
                    and path not in index.racy
                ):
                    entries[path] = old
                    stats[path] = key
                else:
                    hashing.append(path)
        saving: set[str] = set()

        async def put_file(path: str) -> tuple[_File, tuple[int, ...]] | None:
            staged = self._temp_path()
            done = await _in_thread(_stage, self._abs(path), staged)
            try:
                if done is None:
                    return None  # removed while saving
                sha, size, before = done
                if sha not in index.known and sha not in saving:
                    saving.add(sha)
                    await self._ws.call_store(True, self._key(sha), staged)
                return _File(sha, size, _mode(before), before.st_mtime_ns), _stat_key(
                    before
                )
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(staged)

        hashed = await _all(put_file(path) for path in hashing)
        for path, result in zip(hashing, hashed):
            if result is not None:
                entries[path], stats[path] = result
        root, trees = _build(entries)

        async def put_tree(sha: str, data: bytes) -> None:
            staged = self._temp_path()
            await _in_thread(Path(staged).write_bytes, data)
            try:
                await self._ws.call_store(True, self._key(sha), staged)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(staged)

        await _all(
            put_tree(sha, data) for sha, data in trees.items() if sha not in index.known
        )
        fresh = _Index()
        fresh.snapshot = root
        fresh.entries = entries
        fresh.stats = stats
        # The keys of this snapshot's objects only: what the next step can skip.
        fresh.known = {e.sha for e in entries.values() if isinstance(e, _File)}
        fresh.known.update(trees)
        fresh.racy = {
            path
            for path, entry in entries.items()
            if isinstance(entry, _File) and entry.mtime_ns >= started - _RACY_NS
        }
        self._index = fresh
        self._ws.keep_index(self.id, fresh)
        return root


def _suspects(seen: dict[str, _Local], index: _Index) -> list[str] | None:
    """The files to hash again to tell whether the folder is as ``index`` has it.

    Returns:
        None if it is not (an entry added, removed, or changed in kind, size, mode,
        time or target); else the files whose ``stat`` changed otherwise, or that
        changed just before it was taken (``_RACY_NS``).
    """
    if len(seen) != len(index.entries):
        return None
    suspects: list[str] = []
    for path, local in seen.items():
        want = index.entries.get(path)
        if want is None or _kind(want) != local.kind:
            return None
        if isinstance(want, _File):
            st = local.st
            if (st.st_size, _mode(st), st.st_mtime_ns) != (
                want.size,
                want.mode,
                want.mtime_ns,
            ):
                return None
            if path in index.racy or index.stats.get(path) != _stat_key(st):
                suspects.append(path)
        elif isinstance(want, _Dir):
            if want.mode != _mode(local.st):
                return None
        elif want.target != local.target:
            return None
    return suspects


def _check_restored(seen: dict[str, _Local], target: dict[str, _Entry]) -> None:
    """After a restore: the folder holds exactly the snapshot's entries.

    Raises:
        RuntimeError: If something else changed it meanwhile (Temporal retries).
    """
    problem = None
    if set(seen) != set(target):
        extra = sorted(set(seen) - set(target))[:3]
        missing = sorted(set(target) - set(seen))[:3]
        problem = f"extra {extra}, missing {missing}"
    else:
        for path, local in seen.items():
            want = target[path]
            if (
                _kind(want) != local.kind
                or (
                    isinstance(want, _File)
                    and (want.size, want.mode) != (local.st.st_size, _mode(local.st))
                )
                or (isinstance(want, _Dir) and want.mode != _mode(local.st))
                or (isinstance(want, _Link) and want.target != local.target)
            ):
                problem = f"{path} differs"
                break
    if problem is not None:
        raise RuntimeError(
            "The agent's workspace changed while it was restored (another process "
            f"writes to it?): {problem}. Retrying."
        )


if sys.platform == "win32":
    import msvcrt

    def _lock_file(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock_file(fd: int) -> None:
        with contextlib.suppress(OSError):
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_file(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_file(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
