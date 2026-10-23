"""The SQLite session store of the hybrid benchmark (prototype/claude-agent-hybrid,
tests/hybrid/store.py), with only what the SDK calls: append, load, list_subkeys."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import SessionKey, SessionStoreEntry


class TranscriptStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS entries (seq INTEGER PRIMARY KEY, "
                "project TEXT, session TEXT, subpath TEXT, uuid TEXT, data TEXT, "
                "UNIQUE(project, session, subpath, uuid))"
            )

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def _append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        with self.connect() as db:
            db.executemany(
                "INSERT OR IGNORE INTO entries(project,session,subpath,uuid,data) "
                "VALUES (?,?,?,?,?)",
                [
                    (
                        key["project_key"],
                        key["session_id"],
                        key.get("subpath", ""),
                        e.get("uuid"),
                        json.dumps(e),
                    )
                    for e in entries
                ],
            )

    async def append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        await asyncio.to_thread(self._append, key, entries)

    def _load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        with self.connect() as db:
            rows = db.execute(
                "SELECT data FROM entries WHERE project=? AND session=? "
                "AND subpath=? ORDER BY seq",
                (key["project_key"], key["session_id"], key.get("subpath", "")),
            ).fetchall()
        return cast("list[SessionStoreEntry]", [json.loads(r[0]) for r in rows]) or None

    async def load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        return await asyncio.to_thread(self._load, key)

    async def list_subkeys(self, key: Any) -> list[str]:
        def read() -> list[str]:
            with self.connect() as db:
                return [
                    r[0]
                    for r in db.execute(
                        "SELECT DISTINCT subpath FROM entries WHERE project=? "
                        "AND session=? AND subpath!=''",
                        (key["project_key"], key["session_id"]),
                    )
                ]

        return await asyncio.to_thread(read)
