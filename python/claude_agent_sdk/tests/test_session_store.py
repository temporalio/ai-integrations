"""FileSessionStore follows the SessionStore contracts, checked with the SDK's own suite."""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

from claude_agent_sdk import SessionKey, SessionStore
from claude_agent_sdk.testing import run_session_store_conformance

from temporalio.claude_agent_sdk import FileSessionStore


async def test_file_session_store_passes_the_sdk_conformance_suite(
    tmp_path: Path,
) -> None:
    fresh = itertools.count()

    def make_store() -> SessionStore:
        # SessionStore's type includes the optional methods (list, delete), which
        # FileSessionStore does not implement; the suite skips their contracts.
        return FileSessionStore(tmp_path / f"store-{next(fresh)}")  # type: ignore[return-value]

    await run_session_store_conformance(make_store)


_APPEND_ONE_BY_ONE = """
import asyncio, sys
from temporalio.claude_agent_sdk import FileSessionStore

async def main():
    store = FileSessionStore(sys.argv[1])
    key = {"project_key": "project", "session_id": "session"}
    for i in range(int(sys.argv[3])):
        await store.append(key, [{"type": "user", "uuid": f"{sys.argv[2]}-{i}"}])

asyncio.run(main())
"""


async def test_worker_processes_can_append_to_one_session_at_the_same_time(
    tmp_path: Path,
) -> None:
    names, count = ("a", "b", "c"), 40
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _APPEND_ONE_BY_ONE, str(tmp_path), name, str(count)]
        )
        for name in names
    ]
    try:
        codes = [worker.wait(timeout=60) for worker in workers]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
    assert codes == [0] * len(names)
    key = SessionKey(project_key="project", session_id="session")
    uuids = [
        str(entry.get("uuid"))
        for entry in await FileSessionStore(tmp_path).load(key) or []
    ]
    assert len(uuids) == len(names) * count
    for name in names:  # each process's entries, in the order it appended them
        assert [u for u in uuids if u.startswith(f"{name}-")] == [
            f"{name}-{i}" for i in range(count)
        ]
