"""Shared fixtures: a temporary git repository shaped like ai-integrations with one fake plugin."""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REQUIRED_VERSION = ">=0.12.5,<0.13"
LICENSE_TEXT = "MIT License\n\nCopyright (c) 2026 Temporal Technologies Inc.\n\nPermission is hereby granted...\n"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "ci@example.com")
    git(root, "config", "user.name", "CI")
    git(root, "config", "commit.gpgsign", "false")
    (root / "LICENSE").write_text(LICENSE_TEXT)
    (root / "README.md").write_text("# repo\n")
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "pyproject.toml").write_text(
        f'[project]\nname = "scripts"\nversion = "0"\n\n[tool.uv]\npackage = false\nrequired-version = "{REQUIRED_VERSION}"\n'
    )
    return root


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def make_python_plugin(
    root: Path,
    name: str = "fakeplug",
    *,
    version: str = "0.1.0rc1",
    maturity: str = "experimental",
    dependencies: list[str] | None = None,
    allow_final: bool = False,
) -> Path:
    classifier = {
        "ga": "Development Status :: 5 - Production/Stable",
        "preview": "Development Status :: 4 - Beta",
        "experimental": "Development Status :: 3 - Alpha",
    }[maturity]
    coordinate = "temporalio-" + name.replace("_", "-")
    d = root / "python" / name
    pkg = d / "src" / "temporalio" / "contrib" / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text('"""Fake plugin."""\n\n__all__ = ["hello"]\n\n\ndef hello() -> str:\n    return "hi"\n')
    (pkg / "_impl.py").write_text("VALUE = 1\n")
    (pkg / "py.typed").write_text("")
    deps = dependencies if dependencies is not None else ["temporalio>=1.32.0,<2"]
    deps_toml = ", ".join(f'"{x}"' for x in deps)
    (d / "pyproject.toml").write_text(textwrap.dedent(f'''
        [project]
        name = "{coordinate}"
        version = "{version}"
        description = "fake"
        requires-python = ">=3.10"
        readme = "README.md"
        license = "MIT"
        license-files = ["LICENSE"]
        classifiers = ["{classifier}", "Typing :: Typed"]
        dependencies = [{deps_toml}]

        [build-system]
        requires = ["uv_build>=0.12.5,<0.13"]
        build-backend = "uv_build"

        [tool.uv.build-backend]
        module-name = "temporalio.contrib.{name}"

        [tool.uv]
        required-version = "{REQUIRED_VERSION}"
        exclude-newer = "2 weeks"
        exclude-newer-package = {{ temporalio = false }}
    ''').lstrip())
    (d / "plugin.toml").write_text(textwrap.dedent(f'''
        [plugin]
        name = "{name}"
        language = "python"
        coordinate = "{coordinate}"
        registry = "pypi"
        root-api = "temporalio.contrib.{name}"
        maturity = "{maturity}"
        status = "active"
        docs = "https://docs.temporal.io/"
        upstream = "temporalio/sdk-python:temporalio/contrib/{name}"

        [release]
        allow-final = {str(allow_final).lower()}

        [ci]
        runtime-versions = ["3.10", "3.14"]

        [smoke]
        imports = ["temporalio.contrib.{name}._impl"]
    ''').lstrip())
    (d / "Makefile").write_text("DIST := " + coordinate + "\ninclude ../_shared/python.mk\n")
    (d / "README.md").write_text("# fake\n\nSee https://github.com/temporalio/ai-integrations for details.\n")
    (d / "uv.lock").write_text("version = 1\n")
    lic = d / "LICENSE"
    if lic.exists() or lic.is_symlink():
        lic.unlink()
    os.symlink("../../LICENSE", lic)
    return d


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo")


@pytest.fixture
def plugin_repo(repo: Path) -> Path:
    make_python_plugin(repo)
    commit_all(repo, "add fake plugin")
    return repo


def uv_available() -> bool:
    return shutil.which("uv") is not None


def build_plugin(plugin_dir: Path) -> Path:
    dist = plugin_dir / "dist"
    subprocess.run(["uv", "build", "--no-sources", "--out-dir", str(dist), str(plugin_dir)], check=True, capture_output=True, text=True)
    return dist
