from __future__ import annotations

from pathlib import Path

import pytest

import matrix
from conftest import make_python_plugin


def test_pr_profile_is_ubuntu_min_and_max() -> None:
    m = matrix.build_matrix(["3.10", "3.12", "3.14"], "pr")
    assert m == {"include": [
        {"os": "ubuntu-latest", "runtime": "3.10", "time_skipping": False, "dist": False},
        {"os": "ubuntu-latest", "runtime": "3.14", "time_skipping": True, "dist": True},
    ]}


def test_full_profile_adds_macos_and_windows_at_max() -> None:
    m = matrix.build_matrix(["3.10", "3.14"], "full")
    oses = [(e["os"], e["runtime"]) for e in m["include"]]
    assert oses == [("ubuntu-latest", "3.10"), ("ubuntu-latest", "3.14"), ("macos-latest", "3.14"), ("windows-latest", "3.14")]
    assert sum(e["dist"] for e in m["include"]) == 1 and sum(e["time_skipping"] for e in m["include"]) == 1


def test_single_version_collapses_to_one_ubuntu_cell() -> None:
    assert len(matrix.build_matrix(["3.12"], "pr")["include"]) == 1


def test_errors() -> None:
    with pytest.raises(ValueError):
        matrix.build_matrix([], "pr")
    with pytest.raises(ValueError):
        matrix.build_matrix(["3.10"], "nope")


def test_cli_reads_plugin_toml(tmp_path: Path) -> None:
    d = make_python_plugin(tmp_path, "fakeplug")
    out = tmp_path / "out"
    result = matrix.main(["--plugin-dir", str(d), "--profile", "pr", "--github-output", str(out)])
    assert result["include"][0]["runtime"] == "3.10"
    assert out.read_text().startswith('matrix={"include":')
