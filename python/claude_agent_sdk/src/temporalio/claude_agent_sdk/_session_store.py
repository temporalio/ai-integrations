"""A SessionStore for the Claude Agent SDK that keeps transcripts in a folder."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import SessionKey, SessionStoreEntry

_CACHED_FILES = 256  # transcripts whose entry ids are kept in memory
_HINT = 64  # characters of the session id that start a file name, to help people
_LOCK_WAIT = 20.0  # seconds; below the SDK's 60 s per append, so it can retry
_BLOCK = 65536  # bytes read at a time when looking back for the last newline
_BINARY = getattr(os, "O_BINARY", 0)

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int, shared: bool) -> bool:
        del shared  # Windows locks are always exclusive
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EDEADLK):
                return False
            raise
        return True

    def _unlock(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int, shared: bool) -> bool:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def _locked(path: Path, *, shared: bool = False) -> Iterator[None]:
    """Hold the lock file next to ``path``, which every process's store respects."""
    lock = f"{path}.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | _BINARY, 0o644)
    try:
        deadline = time.monotonic() + _LOCK_WAIT
        while not _try_lock(fd, shared):
            if time.monotonic() > deadline:  # not TimeoutError: the SDK retries this
                raise OSError(
                    errno.EBUSY, f"{lock} stayed locked for {_LOCK_WAIT:.0f} s"
                )
            time.sleep(0.005)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def _cut_torn_tail(path: Path) -> None:
    """Remove the half line a writer left if it died in the middle of an append.

    Called with the lock held. Writers hold it for their whole write, so a last line
    without its newline can only come from a writer that died. Appending after it
    would join the two into one broken line.
    """
    try:
        handle = path.open("rb+")
    except FileNotFoundError:
        return
    with handle:
        end = handle.seek(0, os.SEEK_END)
        if end == 0:
            return
        handle.seek(end - 1)
        if handle.read(1) == b"\n":
            return
        keep = end - 1
        while keep > 0:  # back to the last newline, a block at a time
            start = max(0, keep - _BLOCK)
            handle.seek(start)
            newline = handle.read(keep - start).rfind(b"\n")
            if newline >= 0:
                keep = start + newline + 1
                break
            keep = start
        handle.truncate(keep)


def _write_all(fd: int, data: bytes) -> None:
    """Write all of ``data``: one write may take only part of it."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _complete_entries(text: str) -> list[Any]:
    """The entries of a JSON Lines text, ignoring a last line a writer did not finish."""
    complete = text[: text.rfind("\n") + 1]
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


class FileSessionStore:
    """Keeps Claude session transcripts as JSON Lines files in one folder.

    Made for tests, local development and a single machine: every Worker that may
    resume a session must see the same folder. In production, use a store backed by
    a database or object storage (the Claude Agent SDK repository has example
    stores for S3, Redis and Postgres). It implements the two required
    ``SessionStore`` methods, ``append`` and ``load``, and treats each entry's
    ``uuid`` as an idempotency key, as the SDK asks. File work runs in a thread, so
    it does not block the Worker's event loop.

    Each transcript is one file, named by a hash of its whole session key, with a
    lock file next to it. An append holds the lock alone; loads share it (on
    Windows, one at a time). So Worker processes on one machine can share the
    folder, and an append first removes the half line of a writer that died.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Create the store.

        Args:
            root: The folder; it is created if missing.
        """
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # Entry ids per file, with the file size they were read at.
        self._seen: OrderedDict[Path, tuple[int, set[str]]] = OrderedDict()
        self._lock = threading.Lock()  # guards _seen: file work runs in threads

    def _path(self, key: SessionKey) -> Path:
        # The SDK treats every part of the key as opaque, so the name is a hash of
        # the whole key: no two keys share a file, whatever their characters, and
        # names differ in more than case. The session id only helps people.
        whole = json.dumps([key["project_key"], key["session_id"], key.get("subpath")])
        digest = hashlib.sha256(whole.encode("utf-8")).hexdigest()
        hint = re.sub(r"[^a-z0-9_-]", "_", key["session_id"].lower())[:_HINT]
        return self._root / f"{hint}-{digest}.jsonl"

    def _uuids(self, path: Path) -> set[str]:
        """A copy of the entry ids in ``path``, read again if another process wrote.

        Called with the transcript's lock held.
        """
        size = path.stat().st_size if path.exists() else 0
        with self._lock:
            cached = self._seen.get(path)
            if cached is not None and cached[0] == size:
                self._seen.move_to_end(path)
                return set(cached[1])
        found: set[str] = set()
        if size:
            for entry in _complete_entries(path.read_text(encoding="utf-8")):
                uid = entry.get("uuid")
                if uid:
                    found.add(uid)
        self._remember(path, size, found)
        return set(found)

    def _remember(self, path: Path, size: int, uuids: set[str]) -> None:
        with self._lock:
            self._seen[path] = (size, uuids)
            self._seen.move_to_end(path)
            while len(self._seen) > _CACHED_FILES:
                self._seen.popitem(last=False)

    def _append(self, path: Path, entries: list[SessionStoreEntry]) -> None:
        with _locked(path):  # one append per transcript at a time, in any process
            _cut_torn_tail(path)
            seen = self._uuids(path)
            lines: list[str] = []
            for entry in entries:
                uid = entry.get("uuid")
                if uid and uid in seen:
                    continue
                lines.append(json.dumps(entry, separators=(",", ":")) + "\n")
                if uid:
                    seen.add(uid)
            if lines:
                flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | _BINARY
                fd = os.open(path, flags, 0o644)
                try:
                    _write_all(fd, "".join(lines).encode("utf-8"))
                finally:
                    os.close(fd)
            # Only now: if the write failed, the SDK retries the batch, so its ids
            # must not count as stored.
            self._remember(path, path.stat().st_size if path.exists() else 0, seen)

    async def append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        """Append transcript entries, skipping entries whose uuid is already stored.

        Args:
            key: The session key (``project_key``, ``session_id``, optional ``subpath``).
            entries: The entries to add.
        """
        await asyncio.to_thread(self._append, self._path(key), entries)

    def _load(self, path: Path) -> list[SessionStoreEntry] | None:
        if not path.exists():
            return None
        with _locked(path, shared=True):  # never in the middle of an append
            text = path.read_text(encoding="utf-8")
        return cast("list[SessionStoreEntry]", _complete_entries(text))

    async def load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        """Load a session's entries.

        Args:
            key: The session key.

        Returns:
            The entries in order, or None if the session is unknown.
        """
        return await asyncio.to_thread(self._load, self._path(key))
