"""The packaged FileSessionStore: idempotent, whole-batch appends under concurrency."""

from __future__ import annotations

import asyncio
import itertools
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk.testing import run_session_store_conformance

from temporalio.claude_agent_sdk import FileSessionStore
from temporalio.claude_agent_sdk._lease import (
    FileLeaseStore,
    fenced_session_store,
    session_lease,
)

from .helpers.fixtures import POSIX_ONLY

pytestmark = POSIX_ONLY

KEY = {"project_key": "-work-s1", "session_id": "s1"}


def _entry(uid: str) -> dict[str, object]:
    return {"uuid": uid, "type": "user", "message": {"content": uid}}


async def test_a_retried_batch_lands_once_and_loads_in_order(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    assert await store.load(KEY) is None

    await store.append(KEY, [_entry("a"), _entry("b")])
    await store.append(KEY, [_entry("b"), _entry("c")])

    assert [e["uuid"] for e in await store.load(KEY) or []] == ["a", "b", "c"]
    assert sorted(p.name for p in (tmp_path / "-work-s1").iterdir()) == [
        "s1.jsonl",
        "s1.lock",
    ], "no temp file is left behind"


async def test_file_store_passes_the_sdk_conformance_suite(tmp_path: Path) -> None:
    fresh = itertools.count()
    await run_session_store_conformance(
        lambda: cast(Any, FileSessionStore(tmp_path / str(next(fresh))))
    )


async def test_the_lease_fenced_store_passes_the_sdk_conformance_suite(
    tmp_path: Path,
) -> None:
    fresh = itertools.count()
    leases = FileLeaseStore(tmp_path / "leases")
    async with session_lease(leases, "s1", timedelta(seconds=30)) as lease:
        assert lease is not None
        await run_session_store_conformance(
            lambda: fenced_session_store(
                FileSessionStore(tmp_path / str(next(fresh))), lease
            )
        )


async def test_concurrent_appends_lose_nothing(tmp_path: Path) -> None:
    stores = [FileSessionStore(tmp_path) for _ in range(4)]
    await asyncio.gather(
        *(
            stores[i % 4].append(KEY, [_entry(f"{i}-{j}") for j in range(5)])
            for i in range(20)
        )
    )
    assert len(await stores[0].load(KEY) or []) == 100


async def test_a_subagent_subpath_is_its_own_file(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    await store.append({**KEY, "subpath": "subagents/agent-1"}, [_entry("x")])
    assert await store.load(KEY) is None
    assert (tmp_path / "-work-s1" / "s1__subagents_agent-1.jsonl").exists()


@pytest.mark.parametrize(
    "key",
    [
        {"project_key": "..", "session_id": "s1"},
        {"project_key": "p", "session_id": "../escape"},
        {"project_key": "a/b", "session_id": "s1"},
    ],
)
async def test_a_key_that_would_leave_the_root_is_refused(
    tmp_path: Path, key: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        await FileSessionStore(tmp_path / "root").append(key, [_entry("x")])
    assert not (tmp_path / "root").exists()
