"""Worker-side checkpoints for a stopped Codex app-server."""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

_REFERENCE_PREFIX = "openai-codex:1:"
_THREAD_ID = re.compile(r"[a-zA-Z0-9_-]+")
_DATABASE = re.compile(r"(?:state|goals|memories(?:_v2)?|queue)_\d+\.sqlite(?:-wal)?")
_ROLLOUT = re.compile(r"rollout-[a-zA-Z0-9_-]+\.jsonl(?:\.zst)?")

_MEMORY = re.compile(
    r"memories/(?:MEMORY|memory_summary|raw_memories|rollout_summaries/[a-zA-Z0-9_-]+)\.md"
)


class SessionCheckpoint:
    """Select and restore thread state from a dedicated CODEX_HOME.

    Call checkpoint only once the app-server has shut down. Storage callbacks
    run on the worker and must create immutable objects and verify their
    contents before returning from a load. Only the reference crosses into
    the Workflow.
    """

    def __init__(
        self, *, include_optional: Callable[[str, str, int], bool] | None = None
    ) -> None:
        """Set the optional-artifact predicate (relative path, kind, byte size)."""
        self._include_optional = include_optional

    @staticmethod
    def reference(key: str) -> str:
        """Wrap a storage key holding Codex's native files in a v1 reference."""
        if not key or any(part in {"", ".", ".."} for part in key.split("/")):
            raise ValueError("Invalid Codex checkpoint storage key")
        return _REFERENCE_PREFIX + key

    @staticmethod
    def _key(reference: str) -> str:
        if not reference.startswith(_REFERENCE_PREFIX):
            raise ValueError("Unsupported Codex checkpoint reference")
        key = reference[len(_REFERENCE_PREFIX) :]
        SessionCheckpoint.reference(key)
        return key

    def checkpoint(
        self,
        home: Path,
        save: Callable[[Mapping[str, Path], str | None], str],
        *,
        thread_id: str,
        previous: str | None = None,
    ) -> str:
        """Save the selected native files, returning a versioned reference."""
        parent = self._key(previous) if previous is not None else None
        return self.reference(save(self._select(home, thread_id), parent))

    def restore(
        self,
        staged_home: Path,
        reference: str,
        load: Callable[[Path, str], None],
        *,
        thread_id: str,
    ) -> None:
        """Stage verified state for activation at the same absolute CODEX_HOME."""
        key = self._key(reference)
        with tempfile.TemporaryDirectory(dir=staged_home.parent) as directory:
            source = Path(directory)
            load(source, key)
            for name, path in self._select(source, thread_id).items():
                destination = staged_home / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

    def _select(self, home: Path, thread_id: str) -> dict[str, Path]:
        if not _THREAD_ID.fullmatch(thread_id):
            raise ValueError("Invalid Codex checkpoint thread ID")
        if home.is_symlink():
            raise ValueError("Codex checkpoint home must be a real directory")
        files = {}
        found_thread = False
        for path in sorted(home.rglob("*")):
            if not _regular_file(path):
                continue
            relative = path.relative_to(home)
            name = relative.as_posix()
            if name == "session_index.jsonl" or _DATABASE.fullmatch(name):
                files[name] = path
            elif _is_rollout(relative):
                # The app-server prefers the plain rollout if compression left
                # both representations on disk.
                if name.endswith(".zst") and _regular_file(path.with_suffix("")):
                    continue
                rollout = relative.name.removesuffix(".zst").removesuffix(".jsonl")
                if rollout.endswith("-" + thread_id):
                    found_thread |= path.stat().st_size > 0
                files[name] = path
            elif _MEMORY.fullmatch(name):
                if self._include_optional is None or self._include_optional(
                    name, "memory-note", path.stat().st_size
                ):
                    files[name] = path
        if not found_thread:
            raise FileNotFoundError("Codex checkpoint is missing its thread rollout")
        for name in files:
            if name.endswith(".sqlite-wal") and name.removesuffix("-wal") not in files:
                raise ValueError("Codex checkpoint has a WAL without its database")
        return files


def _regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def _is_rollout(relative: Path) -> bool:
    folders = relative.parts[:-1]
    if folders == ("archived_sessions",):
        return _ROLLOUT.fullmatch(relative.name) is not None
    return (
        len(folders) == 4
        and folders[0] == "sessions"
        and re.fullmatch(r"[0-9]{4}/[0-9]{2}/[0-9]{2}", "/".join(folders[1:]))
        is not None
        and _ROLLOUT.fullmatch(relative.name) is not None
    )
