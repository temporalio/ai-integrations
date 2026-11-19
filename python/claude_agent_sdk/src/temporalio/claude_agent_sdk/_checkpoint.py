"""Worker-side checkpoints for a stopped Claude Code session."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from claude_agent_sdk import (
    SessionKey,
    SessionListSubkeysKey,
    SessionStoreEntry,
    project_key_for_directory,
)

_REFERENCE_PREFIX = "claude-agent-sdk:1:"
_SESSION_ID = re.compile(r"[a-zA-Z0-9_-]+")
_SUBAGENT = re.compile(r"agent-[a-zA-Z0-9_-]+\.(?:jsonl|meta\.json)")
_TOOL_RESULT = re.compile(r"[a-zA-Z0-9_-]+\.(?:txt|json)")


class SessionCheckpoint:
    """Select and restore session state on a worker with a dedicated config dir.

    Call checkpoint only once Claude Code has shut down. Storage callbacks run
    on the worker and must create immutable objects and verify their contents
    before returning from a load. Only the reference crosses into the Workflow.
    """

    def __init__(
        self, *, include_optional: Callable[[str, str, int], bool] | None = None
    ) -> None:
        """Set the optional-artifact predicate (relative path, kind, byte size)."""
        self._include_optional = include_optional

    @staticmethod
    def reference(key: str) -> str:
        """Wrap a storage key holding Claude's native files in a v1 reference."""
        if not key or any(part in {"", ".", ".."} for part in key.split("/")):
            raise ValueError("Invalid Claude checkpoint storage key")
        return _REFERENCE_PREFIX + key

    @staticmethod
    def _key(reference: str) -> str:
        if not reference.startswith(_REFERENCE_PREFIX):
            raise ValueError("Unsupported Claude checkpoint reference")
        key = reference[len(_REFERENCE_PREFIX) :]
        SessionCheckpoint.reference(key)
        return key

    def checkpoint(
        self,
        home: Path,
        save: Callable[[Mapping[str, Path], str | None], str],
        *,
        cwd: str | Path,
        session_id: str,
        previous: str | None = None,
    ) -> str:
        """Save the selected native files, returning a versioned reference."""
        parent = self._key(previous) if previous is not None else None
        project = project_key_for_directory(cwd)
        return self.reference(save(self._select(home, project, session_id), parent))

    async def restore(
        self,
        staged_home: Path,
        reference: str,
        load: Callable[[Path, str], None],
        *,
        cwd: str | Path,
        session_id: str,
        reply_id: str | None = None,
    ) -> None:
        """Materialize verified transcripts before starting TemporalTransport.

        Use a fresh staging directory. The caller can activate it at the stable
        CLAUDE_CONFIG_DIR once the restore has completed.
        """
        key = self._key(reference)
        project = project_key_for_directory(cwd)
        with tempfile.TemporaryDirectory(dir=staged_home.parent) as directory:
            source = Path(directory)
            await asyncio.to_thread(load, source, key)
            files = await asyncio.to_thread(self._select, source, project, session_id)
            store = _TranscriptStore(files)
            session: SessionKey = {"project_key": project, "session_id": session_id}
            entries = await store.load(session)
            if not entries or (
                reply_id is not None
                and not any(entry.get("uuid") == reply_id for entry in entries)
            ):
                raise ValueError("Claude checkpoint is missing its saved resume point")

            names = [_transcript_path(session)]
            for subpath in await store.list_subkeys(session):
                subkey: SessionKey = {**session, "subpath": subpath}
                await store.load(subkey)
                names.append(_transcript_path(subkey))

            # Copy the exact bytes that load validated so later appends can
            # reuse the original transcript prefix in incremental checkpoints.
            await asyncio.to_thread(_materialize, staged_home, files, names)

    def _select(self, home: Path, project: str, session_id: str) -> dict[str, Path]:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError("Invalid Claude checkpoint session ID")
        files = {}
        main = f"projects/{project}/{session_id}.jsonl"
        root = f"projects/{project}/{session_id}/"
        if home.is_symlink():
            raise ValueError("Claude checkpoint home must be a real directory")
        for path in sorted(home.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(home).as_posix()
            if relative == main:
                files[relative] = path
            elif relative.startswith(root):
                folder, _, name = relative[len(root) :].partition("/")
                if folder == "subagents" and _SUBAGENT.fullmatch(name):
                    files[relative] = path
                elif folder == "tool-results" and _TOOL_RESULT.fullmatch(name):
                    if self._include_optional is None or self._include_optional(
                        relative, "tool-result", path.stat().st_size
                    ):
                        files[relative] = path
        if main not in files or files[main].stat().st_size == 0:
            raise FileNotFoundError("Claude checkpoint is missing its main transcript")
        return files


def _transcript_path(key: SessionKey) -> str:
    root = f"projects/{key['project_key']}/{key['session_id']}"
    subpath = key.get("subpath")
    return f"{root}/{subpath}.jsonl" if subpath else f"{root}.jsonl"


def _read_entries(path: Path) -> list[SessionStoreEntry]:
    entries = []
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                entry = json.loads(line)
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("type"), str
                ):
                    raise ValueError("Invalid Claude checkpoint transcript entry")
                entries.append(cast("SessionStoreEntry", cast("object", entry)))
    return entries


class _TranscriptStore:
    """A read-only, checkpoint-scoped view of the CLI's raw JSONL files."""

    def __init__(self, files: Mapping[str, Path]) -> None:
        self._files = files

    async def append(self, _key: SessionKey, _entries: list[SessionStoreEntry]) -> None:
        raise TypeError("A confirmed Claude checkpoint is read-only")

    async def load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        path = self._files.get(_transcript_path(key))
        return (
            await asyncio.to_thread(_read_entries, path) if path is not None else None
        )

    async def list_subkeys(self, key: SessionListSubkeysKey) -> list[str]:
        root = f"projects/{key['project_key']}/{key['session_id']}/"
        return [
            name[len(root) : -len(".jsonl")]
            for name in self._files
            if name.startswith(root + "subagents/") and name.endswith(".jsonl")
        ]


def _materialize(home: Path, files: Mapping[str, Path], transcripts: list[str]) -> None:
    for name, source in files.items():
        if name.endswith(".jsonl") and name not in transcripts:
            continue
        destination = home / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
