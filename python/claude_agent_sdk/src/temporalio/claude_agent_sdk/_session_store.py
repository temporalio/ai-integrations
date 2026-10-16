"""A file-backed ``claude_agent_sdk.SessionStore``."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


class FileSessionStore:
    """A ``SessionStore`` that keeps each session's transcript as a JSON-lines file under
    ``root``: ``<root>/<project_key>/<session_id>.jsonl`` (a subagent's subpath is appended
    to the name).

    Every append holds an exclusive ``flock`` on the session's lock file, skips entries whose
    ``uuid`` the file already holds (the protocol's idempotency key, so a retried batch lands
    once), and replaces the file with an fsynced copy, so a reader sees each batch whole or not
    at all. Each append rewrites the session's file: fine for transcripts of a few megabytes;
    use a database-backed store beyond that.

    One host: any local directory. Several workers: ``root`` must be a filesystem every worker
    mounts with working ``flock`` and atomic ``rename`` (local disk, NFSv4 with locking; not
    SMB or object-storage FUSE mounts). POSIX only.

    Args:
        root: The store's directory; created on first append.
    """

    def __init__(self, root: str | Path) -> None:
        """Store transcripts under ``root``. See the class docstring."""
        self.root = Path(root)

    async def append(self, key: Any, entries: list[Any]) -> None:
        """Append ``entries`` to the session's transcript, skipping uuids it already holds."""
        await asyncio.to_thread(self._append, self._path(key), entries)

    async def load(self, key: Any) -> list[Any] | None:
        """The session's transcript entries in append order, or None when there are none."""
        return await asyncio.to_thread(self._load, self._path(key))

    def _path(self, key: Any) -> Path:
        name = str(key["session_id"])
        if subpath := key.get("subpath"):
            name = f"{name}__{str(subpath).replace('/', '_')}"
        project = str(key["project_key"])
        for part in (project, name):
            if part in ("", ".", "..") or "/" in part or "\\" in part or "\0" in part:
                raise ValueError(f"unsafe session key component {part!r}")
        return self.root / project / f"{name}.jsonl"

    def _append(self, path: Path, entries: list[Any]) -> None:
        if sys.platform == "win32":
            raise NotImplementedError(
                "FileSessionStore needs flock, which Windows lacks"
            )
        # Imported here: Windows has no fcntl, and the package must still import there.
        import fcntl

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            held = self._load(path) or []
            seen = {e.get("uuid") for e in held if isinstance(e, dict)} - {None}
            new = []
            for entry in entries:
                uid = entry.get("uuid") if isinstance(entry, dict) else None
                if uid is not None and uid in seen:
                    continue
                seen.add(uid)
                new.append(entry)
            if not new:
                return
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(e) + "\n" for e in [*held, *new])
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

    @staticmethod
    def _load(path: Path) -> list[Any] | None:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        return [json.loads(line) for line in text.splitlines() if line.strip()] or None
