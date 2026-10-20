"""Read-only runfiles can be merged without losing executable permissions."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from installed_runner import _copy_writable


@pytest.mark.parametrize("mode", [0o444, 0o555])
def test_readonly_files_can_be_replaced(tmp_path: Path, mode: int) -> None:
    target = tmp_path / "installed"
    for name in ["first", "second"]:
        source = tmp_path / name
        source.write_text(name)
        source.chmod(mode)
        _copy_writable(source, target)
        assert target.read_text() == name
        assert target.stat().st_mode & stat.S_IWUSR
        if os.name != "nt":
            assert target.stat().st_mode & 0o111 == mode & 0o111
