"""Run checks against an unpacked installed wheel, never the plugin's source imports."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

from python.runfiles import runfiles


def _copy_writable(source: str | Path, target: str | Path) -> str | Path:
    copied = shutil.copy2(source, target)
    path = Path(copied)
    path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
    return copied


def _run_checks(root: Path, wheel: Path, mode: str, test_file: str | None) -> int:
    os.chdir(root)
    if mode == "test":
        import pytest

        assert test_file is not None
        # Bazel schedules modules independently; keep one fixture scope per module.
        return pytest.main(["-c", "pyproject.toml", test_file])
    if mode == "package":
        import importlib.metadata as metadata

        from tests.helpers.plugin_meta import load_plugin_meta
        from tests.helpers.provenance import check_provenance

        meta = load_plugin_meta(root)
        check_provenance(meta.coordinate, meta.package_relpath)
        importlib.import_module(meta.root_api)
        assert metadata.distribution(meta.coordinate).version in wheel.name
        return 0
    commands = [
        ["ruff", "check", "--select", "I"],
        ["ruff", "format", "--check"],
        ["pyright", "--pythonpath", sys.executable],
        ["mypy"],
        ["basedpyright", "--pythonpath", sys.executable],
        ["pydocstyle", "--ignore-decorators=overload", "src"],
    ]
    for module, *arguments in commands:
        if module == "ruff":
            executable = Path(sys.executable).parent / (
                "ruff.exe" if os.name == "nt" else "ruff"
            )
            command = [str(executable), *arguments]
        else:
            command = [sys.executable, "-m", module, *arguments]
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--sdk-wheel", required=True)
    parser.add_argument("--extra-wheel", action="append", default=[])
    parser.add_argument("--mode", choices=["test", "lint", "package"], required=True)
    parser.add_argument("--prepared", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test", help="One test module, relative to the plugin directory")
    args = parser.parse_args()
    if args.mode == "test" and not args.test:
        parser.error("test mode requires --test")
    resolver = runfiles.Create()

    def locate(value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or path.is_file():
            return path.absolute()
        resolved = resolver.Rlocation(value)
        if not resolved:
            raise FileNotFoundError(f"Missing Bazel runfile: {value}")
        return Path(resolved)

    root = Path(resolver.Rlocation("_main/" + args.project + "/pyproject.toml")).parent
    wheel = locate(args.wheel)
    if args.prepared:
        return _run_checks(root, wheel, args.mode, args.test)
    server = os.environ.get("TEMPORAL_CLI_PATH")
    node = os.environ.get("NODE_BINARY")
    if node:
        os.environ["PATH"] = (
            str(locate(node).parent) + os.pathsep + os.environ.get("PATH", "")
        )
    if server:
        os.environ["TEMPORAL_CLI_PATH"] = str(locate(server))
    with tempfile.TemporaryDirectory(
        prefix="installed-plugin-", dir=os.environ.get("TEST_TMPDIR")
    ) as directory:
        environment = Path(directory) / "venv"
        venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
        if os.name == "nt":
            installed = environment / "Lib" / "site-packages"
            interpreter = environment / "Scripts" / "python.exe"
        else:
            installed = (
                environment
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            interpreter = environment / "bin" / "python"
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        dependency_paths = [
            Path(path)
            for path in sys.path
            if Path(path).name == "site-packages"
            and Path(path).is_dir()
            and not (Path(path) / "_bazel_site_init.py").is_file()
        ]
        # Merge the independently locked installed wheels into one environment,
        # matching a normal installation for regular and namespace packages.
        for dependency_path in dependency_paths:
            # Runfiles may be read-only. Keep merged files writable so overlapping
            # wheel paths can be replaced, including the SDK and plugin below.
            shutil.copytree(
                dependency_path,
                installed,
                dirs_exist_ok=True,
                copy_function=_copy_writable,
            )
            binaries = dependency_path.parent / "bin"
            if binaries.is_dir():
                for binary in binaries.iterdir():
                    if binary.is_file():
                        _copy_writable(binary, scripts / binary.name)
        for dependency in [args.sdk_wheel, *args.extra_wheel]:
            with zipfile.ZipFile(locate(dependency)) as sdk:
                sdk.extractall(installed)
        with zipfile.ZipFile(wheel) as archive:
            for entry in archive.infolist():
                if (
                    Path(entry.filename).is_absolute()
                    or ".." in Path(entry.filename).parts
                ):
                    raise ValueError("wheel contains an unsafe path")
                if ".data" in entry.filename.split("/")[0]:
                    raise ValueError(
                        "integration wheels must not contain wheel .data paths"
                    )
            archive.extractall(installed)
        # Native extensions stay loaded until their process exits on Windows.
        # Run checks in the venv's interpreter, then remove it from the parent.
        env = {
            **os.environ,
            "PATH": str(scripts) + os.pathsep + os.environ.get("PATH", ""),
            "VIRTUAL_ENV": str(environment),
            "PYTHONPATH": os.pathsep.join([str(installed), str(root), *sys.path]),
        }
        if os.name == "nt":
            # Bazel sets HOME; Python's Windows expanduser uses USERPROFILE.
            env["USERPROFILE"] = env.get("HOME", directory)
        return subprocess.run(
            [
                str(interpreter),
                str(Path(__file__).absolute()),
                *sys.argv[1:],
                "--prepared",
            ],
            cwd=root,
            env=env,
            check=False,
        ).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
