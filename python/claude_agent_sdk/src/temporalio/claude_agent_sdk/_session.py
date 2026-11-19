"""Select and restore Claude Code session files on a Worker."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from claude_agent_sdk import (
    SessionKey,
    SessionListSubkeysKey,
    SessionStoreEntry,
    project_key_for_directory,
)

_SESSION_ID = re.compile(r"[a-zA-Z0-9_-]+")
_SUBAGENT = re.compile(r"agent-[a-zA-Z0-9_-]+\.(?:jsonl|meta\.json)")
_TOOL_RESULT = re.compile(r"[a-zA-Z0-9_-]+\.(?:txt|json)")


def session_files(
    config_dir: Path,
    *,
    cwd: str | Path,
    session_id: str,
    include_optional: Callable[[str, str, int], bool] | None = None,
) -> dict[str, Path]:
    """Select the native files after Claude Code exits.

    The optional predicate receives the relative path, artifact kind, and
    byte size. Required transcript files are always included.
    """
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError("Invalid Claude checkpoint session ID")
    if config_dir.is_symlink():
        raise ValueError("Claude checkpoint home must be a real directory")

    project = project_key_for_directory(cwd)
    main = f"projects/{project}/{session_id}.jsonl"
    root = f"projects/{project}/{session_id}/"
    files = {}
    for path in sorted(config_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(config_dir).as_posix()
        if relative == main:
            files[relative] = path
        elif relative.startswith(root):
            folder, _, name = relative[len(root) :].partition("/")
            if folder == "subagents" and _SUBAGENT.fullmatch(name):
                files[relative] = path
            elif folder == "tool-results" and _TOOL_RESULT.fullmatch(name):
                if include_optional is None or include_optional(
                    relative, "tool-result", path.stat().st_size
                ):
                    files[relative] = path
    if main not in files or files[main].stat().st_size == 0:
        raise FileNotFoundError("Claude checkpoint is missing its main transcript")
    return files


async def restore_session(
    staged_home: Path,
    *,
    cwd: str | Path,
    session_id: str,
    reply_id: str | None = None,
    include_optional: Callable[[str, str, int], bool] | None = None,
) -> None:
    """Prepare a verified archive in a fresh staging directory before launch.

    The caller unpacks the archive into staging, then activates the directory
    at the stable CLAUDE_CONFIG_DIR once this hook succeeds.
    """
    files = await asyncio.to_thread(
        session_files,
        staged_home,
        cwd=cwd,
        session_id=session_id,
        include_optional=include_optional,
    )
    store = _TranscriptStore(files)
    session: SessionKey = {
        "project_key": project_key_for_directory(cwd),
        "session_id": session_id,
    }
    entries = await store.load(session)
    if not entries or (
        reply_id is not None
        and not any(entry.get("uuid") == reply_id for entry in entries)
    ):
        raise ValueError("Claude checkpoint is missing its saved resume point")

    for subpath in await store.list_subkeys(session):
        await store.load({**session, "subpath": subpath})

    await asyncio.to_thread(_discard_unselected, staged_home, set(files.values()))


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


def _discard_unselected(home: Path, files: set[Path]) -> None:
    # Work from leaves to root so empty directories can also be removed.
    for path in sorted(home.rglob("*"), reverse=True):
        if path in files:
            continue
        if path.is_dir() and not path.is_symlink():
            if not any(path.iterdir()):
                path.rmdir()
        else:
            path.unlink()
