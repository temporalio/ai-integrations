import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest
from claude_agent_sdk import project_key_for_directory

from temporalio.claude_agent_sdk import SessionCheckpoint


async def test_checkpoint_preserves_raw_transcripts_and_selects_optional_files(
    tmp_path: Path,
) -> None:
    home, backup, staged = (tmp_path / name for name in ("home", "backup", "staged"))
    project = f"projects/{project_key_for_directory(tmp_path)}"
    main = f"{project}/session.jsonl"
    child = f"{project}/session/subagents/agent-child.jsonl"
    optional = f"{project}/session/tool-results/tool-output.txt"
    contents = {
        main: b'{ "type": "assistant", "uuid":"reply", "extra":[1,true] }\n',
        child: b'{"type":"user","unknown":"\\u2022"}\n',
        child.removesuffix(".jsonl") + ".meta.json": b'{"agentType":"analyst"}',
        optional: b"large output",
        f"{project}/another.jsonl": b"unrelated session",
        "settings.json": b"settings",
        ".credentials.json": b"secret",
        "debug/session.txt": b"logs",
    }
    for name, content in contents.items():
        path = home / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (home / child.replace("child", "link")).symlink_to(home / ".credentials.json")

    seen = []

    def select(name: str, kind: str, size: int) -> bool:
        seen.append((name, kind, size))
        return False

    def save(files: Mapping[str, Path], parent: str | None) -> str:
        assert parent == "first"
        assert set(files) == {main, child, child.removesuffix(".jsonl") + ".meta.json"}
        for name, path in files.items():
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        return "second"

    def load(directory: Path, key: str) -> None:
        assert key == "second"
        shutil.copytree(backup, directory, dirs_exist_ok=True)

    checkpoint = SessionCheckpoint(include_optional=select)
    reference = checkpoint.checkpoint(
        home,
        save,
        cwd=tmp_path,
        session_id="session",
        previous=checkpoint.reference("first"),
    )
    assert seen == [(optional, "tool-result", len(contents[optional]))]
    await checkpoint.restore(
        staged, reference, load, cwd=tmp_path, session_id="session", reply_id="reply"
    )
    assert {
        path.relative_to(staged).as_posix(): path.read_bytes()
        for path in staged.rglob("*")
        if path.is_file()
    } == {
        name: contents[name]
        for name in (main, child, child.removesuffix(".jsonl") + ".meta.json")
    }
    with pytest.raises(ValueError, match="saved resume point"):
        await checkpoint.restore(
            staged,
            reference,
            load,
            cwd=tmp_path,
            session_id="session",
            reply_id="unrelated",
        )
    with pytest.raises(ValueError, match="Unsupported Claude checkpoint"):
        await checkpoint.restore(
            staged,
            "claude-agent-sdk:2:second",
            load,
            cwd=tmp_path,
            session_id="session",
        )
