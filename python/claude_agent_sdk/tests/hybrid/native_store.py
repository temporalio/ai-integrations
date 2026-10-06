"""Recoverable workspace and original native tool results for probes."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import SessionKey, SessionStoreEntry

from temporalio.claude_agent_sdk._process import workspace_lock
from temporalio.claude_agent_sdk._workspace import restore, snapshot
from tests.hybrid.models import Attempt, Call
from tests.hybrid.store import TranscriptStore


class NativeStore(TranscriptStore):
    """SQLite is a shared test service; each Worker materializes a disposable copy.

    The managed tree contains regular files and directories. Links and special
    files fail publication. External effects cannot be rolled back with the tree.
    """

    def __init__(self, root: Path, phase: str = "live") -> None:
        super().__init__(root / "store.db")
        self.workspace = root / "workspace"
        self.lock_path = root / "workspace.lock"
        self.tools = {"Read", "Edit", "Write", "Bash"}
        self.mcp_servers: dict[str, Any] = {}
        self.phase = phase
        self.owner = ""
        self.held = asyncio.Event()
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS workspace (id INTEGER PRIMARY KEY, "
                "owner TEXT, burst INTEGER, attempt INTEGER, version INTEGER, files TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS native_calls (id TEXT PRIMARY KEY, "
                "name TEXT, arguments TEXT, owner TEXT, phase TEXT, staged TEXT, "
                "result TEXT, version INTEGER)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS native_events (seq INTEGER PRIMARY KEY, "
                "id TEXT, phase TEXT, owner TEXT)"
            )

    def initialize(self, text: str) -> None:
        files = {
            "note.txt": {
                "data": base64.b64encode(text.encode()).decode(),
                "mode": 0o640,
                "mtime_ns": time.time_ns(),
            }
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO workspace VALUES (1,'',-1,0,0,?)", (json.dumps(files),)
            )

    def checkout(self, attempt: Attempt) -> None:
        with workspace_lock(self.lock_path):
            self._checkout(attempt)

    def _checkout(self, attempt: Attempt) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT owner,burst,attempt,files FROM workspace WHERE id=1"
            ).fetchone()
            assert row is not None
            if (attempt.burst, attempt.number) < (row[1], row[2]):
                raise RuntimeError("stale workspace checkout")
            if (attempt.burst, attempt.number) == (row[1], row[2]) and row[
                0
            ] != attempt.token:
                raise RuntimeError("conflicting workspace owner")
            db.execute(
                "UPDATE workspace SET owner=?,burst=?,attempt=? WHERE id=1",
                (attempt.token, attempt.burst, attempt.number),
            )
            files = json.loads(row[3])
        self.owner = attempt.token
        restore(self.workspace, files)

    def prepare(self, call: Call) -> None:
        if call.name not in self.tools or call.subpath:
            raise RuntimeError("unsupported native workspace call")
        if call.name in {"Read", "Edit", "Write"}:
            path = Path(call.arguments["file_path"]).resolve()
            if self.workspace not in path.parents:
                raise RuntimeError("native probe must stay in its managed workspace")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            owner = db.execute("SELECT owner FROM workspace WHERE id=1").fetchone()[0]
            if owner != call.attempt.token:
                raise RuntimeError("stale native executor")
            previous = db.execute(
                "SELECT name,arguments,result,phase,owner FROM native_calls WHERE id=?",
                (call.id,),
            ).fetchone()
            arguments = json.dumps(call.arguments, sort_keys=True)
            if previous and (previous[0], previous[1]) != (call.name, arguments):
                raise RuntimeError("conflicting native execution ID")
            if previous and previous[2] is not None:
                raise RuntimeError("native engine tried to repeat a completed call")
            if previous and (
                previous[3] != "requested" or previous[4] != call.attempt.token
            ):
                raise RuntimeError(
                    "native effect may have executed; reconcile its original ID before retrying"
                )
            db.execute(
                "INSERT OR IGNORE INTO native_calls "
                "VALUES (?,?,?,?,'requested',NULL,NULL,NULL)",
                (call.id, call.name, arguments, call.attempt.token),
            )

    def permit(self, call: Call) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            owner = db.execute("SELECT owner FROM workspace WHERE id=1").fetchone()[0]
            if owner != call.attempt.token:
                raise RuntimeError(
                    "native executor was lost without a committed result"
                )
            row = db.execute(
                "SELECT name,arguments,owner,phase,result FROM native_calls WHERE id=?",
                (call.id,),
            ).fetchone()
            if row is None or (row[0], row[1], row[2]) != (
                call.name,
                json.dumps(call.arguments, sort_keys=True),
                call.attempt.token,
            ):
                raise RuntimeError("missing or conflicting native execution request")
            if row[4] is not None or row[3] in {
                "permitted",
                "executed",
                "held-before-execution",
                "effect-without-result",
            }:
                # Rejoin the original live executor without issuing another permission.
                return
            if row[3] != "requested":
                raise RuntimeError(
                    "native effect may have executed; reconcile its original ID before retrying"
                )
            db.execute(
                "UPDATE native_calls SET phase='permitted' WHERE id=?", (call.id,)
            )

    def mark(self, tid: str, phase: str) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute("SELECT owner FROM workspace WHERE id=1").fetchone()[0]
                != self.owner
            ):
                raise RuntimeError("stale workspace writer")
            db.execute("UPDATE native_calls SET phase=? WHERE id=?", (phase, tid))
            db.execute(
                "INSERT INTO native_events(id,phase,owner) VALUES (?,?,?)",
                (tid, phase, self.owner),
            )

    def stage(self, tid: str) -> None:
        files = snapshot(self.workspace)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute("SELECT owner FROM workspace WHERE id=1").fetchone()[0]
                != self.owner
            ):
                raise RuntimeError("stale workspace snapshot")
            db.execute(
                "UPDATE native_calls SET staged=? WHERE id=?", (json.dumps(files), tid)
            )

    def execution(self, tid: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT name,phase,result,version FROM native_calls WHERE id=?", (tid,)
            ).fetchone()
        return (
            None
            if row is None
            else {
                "name": row[0],
                "phase": row[1],
                "result": json.loads(row[2]) if row[2] else None,
                "version": row[3],
            }
        )

    def executions(self) -> dict[str, dict[str, Any]]:
        with self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM native_calls")]
        return {tid: row for tid in ids if (row := self.execution(tid)) is not None}

    def durable_text(self) -> str:
        with self.connect() as db:
            files = json.loads(
                db.execute("SELECT files FROM workspace WHERE id=1").fetchone()[0]
            )
        return base64.b64decode(files["note.txt"]["data"]).decode()

    def capture(self, entries: list[SessionStoreEntry]) -> list[str]:
        """Commit the actual native tool_result and its staged workspace together.

        The transcript mirror can lag this commit. Recovery must join the
        Activity outcome and persist the exact original result content before
        launching a replacement CLI.
        """
        completed = []
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for entry in entries:
                message = cast(dict[str, Any], entry.get("message", {}))
                content = message.get("content", [])
                if not isinstance(content, list):
                    continue
                for block in content:
                    if (
                        not isinstance(block, dict)
                        or block.get("type") != "tool_result"
                    ):
                        continue
                    tid = block["tool_use_id"]
                    row = db.execute(
                        "SELECT owner,staged,result,name,phase FROM native_calls WHERE id=?",
                        (tid,),
                    ).fetchone()
                    if row is None:
                        continue
                    result = json.dumps(block, sort_keys=True)
                    if row[2] is not None:
                        if row[2] != result:
                            raise RuntimeError("conflicting original native result")
                        continue
                    owner, version = db.execute(
                        "SELECT owner,version FROM workspace WHERE id=1"
                    ).fetchone()
                    if owner != self.owner or row[0] != self.owner:
                        raise RuntimeError("stale native result writer")
                    if row[1] is None:
                        raise RuntimeError("native result has no workspace snapshot")
                    if row[4] != "executed":
                        raise RuntimeError(
                            "native result has no execution acknowledgment"
                        )
                    version += 1
                    db.execute(
                        "UPDATE workspace SET files=?,version=? WHERE id=1",
                        (row[1], version),
                    )
                    db.execute(
                        "UPDATE native_calls SET result=?,version=?,phase='committed' WHERE id=?",
                        (result, version, tid),
                    )
                    db.execute(
                        "INSERT INTO native_events(id,phase,owner) VALUES (?,'committed',?)",
                        (tid, self.owner),
                    )
                    completed.append(row[3])
        return completed

    async def append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        names = await asyncio.to_thread(self.capture, entries)
        if any(self.phase.startswith(name + "-after-commit") for name in names):
            await self.held.wait()
        await super().append(key, entries)
