"""``record_histories.py`` keeps existing recordings unless told otherwise."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests import record_histories


async def test_record_histories_refuses_an_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    existing = tmp_path / "tool_loop.json"
    existing.write_text("{}", encoding="utf-8")
    calls: list[str] = []

    async def recorder(_client: Any) -> Any:
        calls.append("recorded")
        raise AssertionError("no recorder runs when every file is refused")

    monkeypatch.setattr(record_histories, "HISTORIES", tmp_path)
    monkeypatch.setattr(record_histories, "RECORDERS", {"tool_loop.json": recorder})
    # Refused: the file stays, the reason points at the policy, no server starts.
    assert await record_histories.main([]) == 1
    assert existing.read_text(encoding="utf-8") == "{}"
    assert calls == []
    out = capsys.readouterr().out
    assert "refusing to overwrite" in out and "tests/histories/README.md" in out
    assert "--overwrite" in out and "--suffix" in out


def test_record_histories_suffix_names_a_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(record_histories, "HISTORIES", tmp_path)
    assert (
        record_histories.target_path("tool_loop.json", "")
        == tmp_path / "tool_loop.json"
    )
    assert (
        record_histories.target_path("tool_loop.json", "0.2")
        == tmp_path / "tool_loop-0.2.json"
    )
    assert record_histories.parse_args(
        ["--overwrite", "input shape changed"]
    ).overwrite == ("input shape changed")
    assert record_histories.parse_args([]).overwrite is None
