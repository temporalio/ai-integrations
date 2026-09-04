"""Build a real (tiny) namespace-package plugin with uv_build and exercise check_wheel.py and smoke.py end to end."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import check_wheel
import smoke
from conftest import build_plugin, init_repo, make_python_plugin, uv_available

pytestmark = pytest.mark.skipif(not uv_available(), reason="uv is required to build the fixture wheel")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = init_repo(tmp_path_factory.mktemp("built") / "repo")
    plugin = make_python_plugin(root, "fakeplug", dependencies=[])  # no deps: smoke venvs stay tiny
    dist = build_plugin(plugin)
    return root, plugin, dist


def test_check_wheel_passes_for_a_correct_build(built: tuple[Path, Path, Path]) -> None:
    root, plugin, dist = built
    problems = check_wheel.check(plugin, dist, root)
    # The fixture has no temporalio dependency on purpose; that is the only expected complaint.
    assert problems == ["METADATA Requires-Dist must include temporalio"], problems


def test_check_wheel_verifies_license_bytes_and_namespace(built: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, plugin, dist = built
    other_root = tmp_path / "other"
    other_root.mkdir()
    (other_root / "LICENSE").write_text("different license text\n")
    problems = check_wheel.check(plugin, dist, other_root)
    assert any("differs from the repository root LICENSE" in p for p in problems)
    assert any("sdist LICENSE differs" in p for p in problems)
    assert not any("must not ship" in p for p in problems)


def test_check_wheel_requires_exactly_one_artifact_each(built: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, plugin, _ = built
    problems = check_wheel.check(plugin, tmp_path, root)
    assert len(problems) == 2 and all("expected exactly one" in p for p in problems)


def test_smoke_orchestrator_passes(built: tuple[Path, Path, Path]) -> None:
    _, plugin, dist = built
    rc = smoke.main(["--plugin", str(plugin), "--dist", str(dist)])
    assert rc == 0


def test_smoke_in_env_detects_editable_and_version_mismatch(built: tuple[Path, Path, Path], tmp_path: Path) -> None:
    _, plugin, dist = built
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "--quiet", str(venv)], check=True)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = {**os.environ, "UV_LINK_MODE": "copy"}
    subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(py), "--editable", str(plugin)], check=True, env=env)
    r = subprocess.run([str(py), smoke.__file__, "--in-env", "--coordinate", "temporalio-fakeplug", "--root-api", "temporalio.contrib.fakeplug"],
                       env={**env, "EXPECTED_VERSION": "0.1.0rc1"}, capture_output=True, text=True)
    assert r.returncode == 1 and "editable mode" in r.stdout

    wheel = next(dist.glob("*.whl"))
    subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(py), "--reinstall", str(wheel)], check=True, env=env)
    r = subprocess.run([str(py), smoke.__file__, "--in-env", "--coordinate", "temporalio-fakeplug", "--root-api", "temporalio.contrib.fakeplug"],
                       env={**env, "EXPECTED_VERSION": "9.9.9"}, capture_output=True, text=True)
    assert r.returncode == 1 and "installed version" in r.stdout
    r = subprocess.run([str(py), smoke.__file__, "--in-env", "--coordinate", "temporalio-fakeplug", "--root-api", "temporalio.contrib.fakeplug"],
                       env={**env, "EXPECTED_VERSION": "0.1.0-rc1"}, capture_output=True, text=True)  # non-canonical spelling is normalised
    assert r.returncode == 0, r.stdout


def test_smoke_in_env_detects_overwritten_and_stray_files(built: tuple[Path, Path, Path], tmp_path: Path) -> None:
    _, plugin, dist = built
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "--quiet", str(venv)], check=True)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wheel = next(dist.glob("*.whl"))
    subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(py), str(wheel)], check=True, env={**os.environ, "UV_LINK_MODE": "copy"})
    site = subprocess.run([str(py), "-c", "import temporalio.contrib.fakeplug as m, pathlib; print(pathlib.Path(m.__file__).parent)"],
                          check=True, capture_output=True, text=True).stdout.strip()
    pkg_dir = Path(site)
    (pkg_dir / "_impl.py").write_text("VALUE = 'tampered'\n")
    r = subprocess.run([str(py), smoke.__file__, "--in-env", "--coordinate", "temporalio-fakeplug", "--root-api", "temporalio.contrib.fakeplug"],
                       env={**os.environ, "EXPECTED_VERSION": "0.1.0rc1"}, capture_output=True, text=True)
    assert r.returncode == 1 and "differs from the hash" in r.stdout
    subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(py), "--reinstall", str(wheel)], check=True, env={**os.environ, "UV_LINK_MODE": "copy"})
    (pkg_dir / "stray.py").write_text("x = 1\n")
    r = subprocess.run([str(py), smoke.__file__, "--in-env", "--coordinate", "temporalio-fakeplug", "--root-api", "temporalio.contrib.fakeplug"],
                       env={**os.environ, "EXPECTED_VERSION": "0.1.0rc1"}, capture_output=True, text=True)
    assert r.returncode == 1 and "not owned by" in r.stdout


def test_normalize_version() -> None:
    assert smoke.normalize_version("1.0.0-RC1") == "1.0.0rc1"
    assert smoke.normalize_version("v1.0.0.dev3") == "1.0.0.dev3"
    assert smoke.normalize_version("1.0.0.post1") == "1.0.0.post1"
    assert smoke.versions_equal("1.0.0rc1", "1.0.0-rc1")
    assert not smoke.versions_equal("1.0.0", "1.0.1")


def test_requirement_name_is_exact_not_prefix() -> None:
    assert check_wheel._requirement_name("temporalio[opentelemetry,pydantic]>=1.32.0,<2; python_version >= '3.10'") == "temporalio"
    assert check_wheel._requirement_name("temporalio-mcp>=0.1,<0.2") == "temporalio-mcp"
    assert check_wheel._requirement_name("not a requirement !!") == ""
