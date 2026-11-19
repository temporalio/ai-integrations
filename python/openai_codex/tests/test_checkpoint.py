import shutil
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

from temporalio.openai_codex import SessionCheckpoint


def test_checkpoint_keeps_wal_and_one_native_rollout(tmp_path: Path) -> None:
    home, saved, staged = (tmp_path / name for name in ("home", "backup", "staged"))
    rollout = "sessions/2026/11/19/rollout-2026-11-19T01-02-03-current.jsonl"
    archived = "archived_sessions/rollout-2026-11-19T01-02-03-other.jsonl.zst"
    optional = "memories/MEMORY.md"
    contents = {
        rollout: b'{ "type":"session_meta" }\n',
        rollout + ".zst": b"stale compressed copy",
        archived: b"archived rollout",
        optional: b"optional",
        "session_index.jsonl": b'{"id":"current"}\n',
        "auth.json": b"secret",
        "config.toml": b"settings",
        "logs_2.sqlite": b"logs",
        "thread_history_1.sqlite": b"materialized copy",
        "thread-writer-locks/lock": b"process lock",
        "skills/.system/tool/SKILL.md": b"bundled skill",
    }
    for name, content in contents.items():
        path = home / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    # Capture a consistent database and WAL left by a stopped process.
    db = tmp_path / "state_5.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE threads (id TEXT)")
        conn.execute("INSERT INTO threads VALUES ('current')")
        conn.commit()
        for name in (db.name, db.name + "-wal", db.name + "-shm"):
            shutil.copy2(tmp_path / name, home / name)
    conn.close()

    seen = []

    def include(name: str, kind: str, size: int) -> bool:
        seen.append((name, kind, size))
        return False

    def save(files: Mapping[str, Path], parent: str | None) -> str:
        assert parent is None
        assert set(files) == {
            rollout,
            archived,
            db.name,
            db.name + "-wal",
            "session_index.jsonl",
        }
        for name, path in files.items():
            target = saved / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        return "checkpoint"

    def load(directory: Path, key: str) -> None:
        assert key == "checkpoint"
        shutil.copytree(saved, directory, dirs_exist_ok=True)

    checkpoint = SessionCheckpoint(include_optional=include)
    ref = checkpoint.checkpoint(home, save, thread_id="current")
    assert seen == [(optional, "memory-note", len(contents[optional]))]
    checkpoint.restore(staged, ref, load, thread_id="current")
    assert (staged / rollout).read_bytes() == contents[rollout]
    assert not (staged / (rollout + ".zst")).exists()
    assert not (staged / "auth.json").exists()
    conn = sqlite3.connect(staged / db.name)
    try:
        assert conn.execute("SELECT id FROM threads").fetchone() == ("current",)
    finally:
        conn.close()

    with pytest.raises(FileNotFoundError, match="thread rollout"):
        checkpoint.restore(staged, ref, load, thread_id="unrelated")
    (saved / db.name).unlink()
    with pytest.raises(ValueError, match="WAL without its database"):
        checkpoint.restore(staged, ref, load, thread_id="current")
    with pytest.raises(ValueError, match="Unsupported Codex checkpoint"):
        checkpoint.restore(
            staged, "claude-agent-sdk:1:checkpoint", load, thread_id="current"
        )
