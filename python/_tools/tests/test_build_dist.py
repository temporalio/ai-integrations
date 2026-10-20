"""Distribution staging must release its working directory before cleanup."""

from __future__ import annotations

import sys
from pathlib import Path

import build_dist
import pytest


@pytest.mark.parametrize("fail_sdist", [False, True])
def test_build_restores_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_sdist: bool
) -> None:
    project = tmp_path / "project"
    package = project / "src" / "example"
    package.mkdir(parents=True)
    manifest = project / "pyproject.toml"
    manifest.write_text(
        '[project]\nname = "example"\nversion = "0.0.0"\n'
        '[build-system]\nrequires = ["uv_build"]\nbuild-backend = "uv_build"\n'
    )
    source = package / "__init__.py"
    source.write_text('"""Example package."""\n')
    dist = tmp_path / "dist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dist",
            "--project",
            str(project),
            "--version",
            "0.1.0",
            "--wheel",
            str(dist / "example-0.1.0-py3-none-any.whl"),
            "--sdist",
            str(dist / "example-0.1.0.tar.gz"),
            "--source",
            str(manifest),
            "--source",
            str(source),
        ],
    )
    if fail_sdist:

        def fail_build(_directory: str) -> str:
            raise RuntimeError("sdist build failed")

        monkeypatch.setattr(build_dist.uv_build, "build_sdist", fail_build)
    cwd = Path.cwd()
    if fail_sdist:
        with pytest.raises(RuntimeError, match="sdist build failed"):
            build_dist.main()
    else:
        build_dist.main()
        assert (dist / "example-0.1.0-py3-none-any.whl").is_file()
        assert (dist / "example-0.1.0.tar.gz").is_file()
    assert Path.cwd() == cwd
