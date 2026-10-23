from __future__ import annotations

from pathlib import Path

import pytest

from temporalio.claude_agent_sdk._sandbox_cli import write_plugins
from tests.helpers.memory_sandbox import MemorySandbox


async def test_plugins_are_written_with_their_content_and_mode(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    (plugin / "hooks").mkdir(parents=True)
    (plugin / "plugin.json").write_text("{}")
    (plugin / "hooks" / "check.sh").write_text("#!/bin/sh\n")
    (plugin / "hooks" / "check.sh").chmod(0o755)
    (tmp_path / "shared.md").write_text("shared")
    (plugin / "linked.md").symlink_to(tmp_path / "shared.md")
    sandbox = MemorySandbox()
    await sandbox.start(env={})

    dirs = await write_plugins(sandbox, [str(plugin)])

    root = f"{sandbox.home}/plugins/0"
    assert dirs == [root]
    assert {path: bytes(content) for path, content in sandbox.files.items()} == {
        f"{root}/plugin.json": b"{}",
        f"{root}/hooks/check.sh": b"#!/bin/sh\n",
        # A link is written with its target's content.
        f"{root}/linked.md": b"shared",
    }
    assert sandbox.modes[f"{root}/hooks/check.sh"] == 0o755


async def test_no_plugins_writes_nothing() -> None:
    sandbox = MemorySandbox()
    await sandbox.start(env={})

    assert await write_plugins(sandbox, []) == []
    assert sandbox.files == {}


async def test_a_plugin_that_is_not_a_directory_fails(tmp_path: Path) -> None:
    sandbox = MemorySandbox()
    await sandbox.start(env={})

    with pytest.raises(NotADirectoryError):
        await write_plugins(sandbox, [str(tmp_path / "missing")])
