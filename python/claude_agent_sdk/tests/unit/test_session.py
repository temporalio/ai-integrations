import shutil
from pathlib import Path

import pytest
from claude_agent_sdk import project_key_for_directory

from temporalio.claude_agent_sdk import restore_session, session_files


async def test_checkpoint_preserves_raw_transcripts_and_selects_optional_files(
    tmp_path: Path,
) -> None:
    home, staged = tmp_path / "home", tmp_path / "staged"
    project = f"projects/{project_key_for_directory(tmp_path)}"
    main = f"{project}/session.jsonl"
    child = f"{project}/session/subagents/agent-child.jsonl"
    optional = f"{project}/session/tool-results/tool-output.txt"
    metadata = child.removesuffix(".jsonl") + ".meta.json"
    contents = {
        main: b'{ "type": "assistant", "uuid":"reply", "extra":[1,true] }\n',
        child: b'{"type":"user","unknown":"\\u2022"}\n',
        metadata: b'{"agentType":"analyst"}',
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

    files = session_files(
        home, cwd=tmp_path, session_id="session", include_optional=select
    )
    assert set(files) == {main, child, metadata}
    assert seen == [(optional, "tool-result", len(contents[optional]))]
    assert optional in session_files(home, cwd=tmp_path, session_id="session")

    # A restored full-home archive can still hold unrelated or sensitive files.
    shutil.copytree(home, staged, symlinks=True)
    await restore_session(
        staged,
        cwd=tmp_path,
        session_id="session",
        reply_id="reply",
        include_optional=select,
    )
    assert {
        path.relative_to(staged).as_posix(): path.read_bytes()
        for path in staged.rglob("*")
        if path.is_file()
    } == {name: contents[name] for name in (main, child, metadata)}
    assert not (staged / child.replace("child", "link")).is_symlink()
    with pytest.raises(ValueError, match="saved resume point"):
        await restore_session(
            staged, cwd=tmp_path, session_id="session", reply_id="unrelated"
        )
    (staged / child).write_text("malformed")
    with pytest.raises(ValueError):
        await restore_session(
            staged, cwd=tmp_path, session_id="session", reply_id="reply"
        )
