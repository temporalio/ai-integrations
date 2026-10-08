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
    assert dist_cells == [{"os": "ubuntu-latest", "runtime": "3.14", "dist": True}]


def test_single_version_still_covers_three_operating_systems() -> None:
    assert [e["os"] for e in matrix.build_matrix(["3.12"])["include"]] == ["ubuntu-latest", "macos-latest", "windows-latest"]


def test_java_boot_variants_have_one_artifact_producer() -> None:
    cells = matrix.build_matrix(["17", "21"], ["4.0.8", "4.1.1"])["include"]
    assert len(cells) == 8
    assert {(c["runtime"], c["os"]) for c in cells} == {
        ("17", "ubuntu-latest"), ("21", "ubuntu-latest"), ("21", "macos-latest"), ("21", "windows-latest")}
    assert [c["spring_boot"] for c in cells if c["dist"]] == ["4.0.8"]


def test_errors() -> None:
    with pytest.raises(ValueError):
        matrix.build_matrix([])


def test_cli_reads_plugin_toml(tmp_path: Path) -> None:
    d = make_python_plugin(tmp_path, "fakeplug")
    out = tmp_path / "out"
    result = matrix.main(["--plugin-dir", str(d), "--github-output", str(out)])
    assert result["include"][0]["runtime"] == "3.10"
    assert out.read_text().startswith('matrix={"include":')


def test_cli_preserves_go_patch_floor_and_selects_all_platforms(tmp_path: Path) -> None:
    plugin = tmp_path / "go" / "fakeplug"
    plugin.mkdir(parents=True)
    (plugin / "plugin.toml").write_text('[ci]\nruntime-versions = ["1.26.5", "1.27"]\n')
    result = matrix.main(["--plugin-dir", str(plugin)])
    assert [(cell["os"], cell["runtime"]) for cell in result["include"]] == [
        ("ubuntu-latest", "1.26.5"),
        ("ubuntu-latest", "1.27"),
        ("macos-latest", "1.27"),
        ("windows-latest", "1.27"),
    ]
    assert [cell["runtime"] for cell in result["include"] if cell["dist"]] == ["1.27"]
