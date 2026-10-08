"""The Codex login (auth.json) handling: copied into a fresh home, written back only when refreshed."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from temporalio.exceptions import ApplicationError
from temporalio.openai_codex._activity import (  # pyright: ignore[reportPrivateUsage]
    _copy_auth_in,
    _sync_auth_back,
)


def make_login(
    tmp_path: Path, content: bytes = b'{"tokens": {"refresh_token": "one"}}'
) -> Path:
    source = tmp_path / "auth.json"
    source.write_bytes(content)
    source.chmod(0o600)
    return source


def test_the_login_is_copied_into_the_fresh_home_privately(tmp_path: Path) -> None:
    source = make_login(tmp_path)
    home = tmp_path / "home"
    home.mkdir()

    _copy_auth_in(str(source), str(home))

    copy = home / "auth.json"
    assert copy.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(copy.stat().st_mode) == 0o600


def test_an_unchanged_login_is_not_written_back(tmp_path: Path) -> None:
    source = make_login(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    _copy_auth_in(str(source), str(home))
    before = source.stat().st_mtime_ns

    _sync_auth_back(str(source), str(home))

    assert source.stat().st_mtime_ns == before


def test_refreshed_tokens_are_written_back_so_the_original_keeps_working(
    tmp_path: Path,
) -> None:
    source = make_login(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    _copy_auth_in(str(source), str(home))
    # Codex refreshes the tokens inside the throwaway home...
    (home / "auth.json").write_bytes(b'{"tokens": {"refresh_token": "two"}}')

    _sync_auth_back(str(source), str(home))

    # ...and the original must now hold the new refresh token, still private, with no temp file left.
    assert source.read_bytes() == b'{"tokens": {"refresh_token": "two"}}'
    assert stat.S_IMODE(source.stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_a_missing_login_fails_clearly(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    with pytest.raises(ApplicationError, match="does not exist"):
        _copy_auth_in(str(tmp_path / "nope.json"), str(home))
    assert os.listdir(home) == []
