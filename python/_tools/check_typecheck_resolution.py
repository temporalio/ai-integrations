"""Prove that Pyright and mypy resolve a unique source-only module."""

import argparse
import os
import subprocess
from pathlib import Path

import tomllib
from uv_tool import uv_binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-dir", required=True)
    args = parser.parse_args()
    project = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")) / args.plugin_dir
    api = tomllib.loads((project / "plugin.toml").read_text())["plugin"]["root-api"]
    probe = project / "src" / Path(*api.split(".")) / "_resolution_probe.py"
    test = project / "tests/test__resolution_probe.py"
    if probe.exists() or test.exists():
        raise FileExistsError("Refusing to overwrite resolution probes")
    try:
        probe.write_text('RESOLUTION_PROBE = "src"\n')
        test.write_text(
            f'from {api}._resolution_probe import RESOLUTION_PROBE\n\ndef test_resolution_probe() -> None:\n    assert RESOLUTION_PROBE == "src"\n'
        )
        for tool in ["pyright", "mypy"]:
            subprocess.run(
                [uv_binary(), "run", "--locked", tool, str(test)],
                cwd=project,
                env={**os.environ, "UV_NO_EDITABLE": "1"},
                check=True,
            )
    finally:
        probe.unlink(missing_ok=True)
        test.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
