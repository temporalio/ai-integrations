"""Workspace tree recovery, snapshot rejection and effect retry fencing."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import cast

import pytest
from claude_agent_sdk import SessionStoreEntry

from temporalio.claude_agent_sdk._process import workspace_lock
from temporalio.claude_agent_sdk._workspace import restore, snapshot
from tests.hybrid.models import Attempt, Call
from tests.hybrid.native_store import NativeStore


def test_tree_follows_replacement_worker(tmp_path: Path) -> None:
    store = NativeStore(tmp_path)
    store.initialize("old")
    attempt = Attempt(0, 1, "old-worker")
    store.checkout(attempt)
    nested = store.workspace / "nested"
    nested.mkdir()
    (nested / "binary.dat").write_bytes(bytes(range(256)))
    (nested / "empty").mkdir()
    executable = store.workspace / "run.sh"
    executable.write_text("echo restored\n")
    executable.chmod(0o750)
    os.utime(executable, ns=(1234567890000000000, 1234567890000000000))
    (store.workspace / "note.txt").unlink()
    call = Call(
        "bash-original", "Bash", {"command": "tree mutation"}, attempt, "original"
    )
    store.prepare(call)
    store.permit(call)
    store.mark(call.id, "executed")
    store.stage(call.id)
    expected = snapshot(store.workspace)
    store.capture(
        cast(
            list[SessionStoreEntry],
            [
                {
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "content": "done",
                            }
                        ]
                    }
                }
            ],
        )
    )
    shutil.rmtree(store.workspace)
    replacement = NativeStore(tmp_path)
    replacement.checkout(Attempt(0, 2, "replacement"))
    assert snapshot(replacement.workspace) == expected
    assert not (replacement.workspace / "note.txt").exists()
    assert (replacement.workspace / "nested/empty").is_dir()
    assert replacement.execution(call.id) == store.execution(call.id)
    with pytest.raises(RuntimeError, match="stale workspace snapshot"):
        store.stage(call.id)


@pytest.mark.parametrize(
    "name",
    [
        "../escaped",
        "/absolute",
        "a/../../escaped",
        "a\\escaped",
        "a//escaped",
        "a/file",
    ],
)
def test_corrupt_tree_leaves_workspace_unchanged(tmp_path: Path, name: str) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "kept").write_text("original")
    original = snapshot(root)
    invalid = {name: {"data": "", "mode": 0o600, "mtime_ns": 1}}
    with pytest.raises(RuntimeError, match="invalid stored workspace"):
        restore(root, invalid)
    assert snapshot(root) == original


def test_links_and_oversized_files_fail_publication(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = tmp_path / "outside"
    target.write_text("external")
    link = root / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("platform cannot create symlinks")
    with pytest.raises(RuntimeError, match="not a regular file"):
        snapshot(root)
    link.unlink()
    (root / "large").write_bytes(b"123")
    with pytest.raises(RuntimeError, match="byte limit"):
        snapshot(root, max_bytes=2)


@pytest.mark.parametrize("name", ["Bash", "mcp__external__charge"])
def test_ambiguous_effect_cannot_be_issued_again(tmp_path: Path, name: str) -> None:
    store = NativeStore(tmp_path)
    store.tools.add(name)
    store.initialize("old")
    attempt = Attempt(0, 1, "first")
    store.checkout(attempt)
    call = Call(
        "original-id", name, {"command": "external effect"}, attempt, "original"
    )
    store.prepare(call)
    store.permit(call)
    store.permit(call)  # An Activity retry rejoins the same live executor.
    with pytest.raises(RuntimeError, match="reconcile its original ID"):
        store.prepare(call)  # A second engine permission is forbidden.
    store.mark(call.id, "executed")
    store.permit(call)
    assert store.execution(call.id)["phase"] == "executed"  # type: ignore[index]
    replacement = NativeStore(tmp_path)
    replacement.tools.add(name)
    replacement.checkout(Attempt(0, 2, "replacement"))
    with pytest.raises(RuntimeError, match="lost without a committed result"):
        replacement.permit(call)


def test_live_executor_prevents_workspace_checkout(tmp_path: Path) -> None:
    store = NativeStore(tmp_path)
    store.initialize("old")
    store.checkout(Attempt(0, 1, "first"))
    before = snapshot(store.workspace)
    with workspace_lock(store.lock_path):
        with pytest.raises(OSError):
            store.checkout(Attempt(0, 2, "replacement"))
    assert snapshot(store.workspace) == before
