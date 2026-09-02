from __future__ import annotations

from pathlib import Path

import pytest

import matrix
from conftest import make_python_plugin


def test_matrix_is_ubuntu_min_max_plus_macos_and_windows_at_max() -> None:
    m = matrix.build_matrix(["3.10", "3.12", "3.14"])
    assert [(e["os"], e["runtime"]) for e in m["include"]] == [
        ("ubuntu-latest", "3.10"), ("ubuntu-latest", "3.14"), ("macos-latest", "3.14"), ("windows-latest", "3.14"),
    ]
    dist_cells = [e for e in m["include"] if e["dist"]]
    assert dist_cells == [{"os": "ubuntu-latest", "runtime": "3.14", "time_skipping": True, "dist": True}]


def test_single_version_still_covers_three_operating_systems() -> None:
    assert [e["os"] for e in matrix.build_matrix(["3.12"])["include"]] == ["ubuntu-latest", "macos-latest", "windows-latest"]


def test_errors() -> None:
    with pytest.raises(ValueError):
        matrix.build_matrix([])


def test_cli_reads_plugin_toml(tmp_path: Path) -> None:
    d = make_python_plugin(tmp_path, "fakeplug")
    out = tmp_path / "out"
    result = matrix.main(["--plugin-dir", str(d), "--github-output", str(out)])
    assert result["include"][0]["runtime"] == "3.10"
    assert out.read_text().startswith('matrix={"include":')
