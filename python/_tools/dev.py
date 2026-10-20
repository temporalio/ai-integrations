"""Explicit workspace development operations; Bazel builds never mutate lockfiles."""

import argparse
import os
import subprocess
from pathlib import Path

from uv_tool import uv_binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["lock", "update", "venv", "format"])
    parser.add_argument("--plugin-dir", required=True)
    args = parser.parse_args()
    project = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")) / args.plugin_dir
    env = {**os.environ, "UV_NO_EDITABLE": "1", "UV_LINK_MODE": "copy"}
    command = [uv_binary()]
    if args.action == "lock":
        command += ["lock"]
    elif args.action == "update":
        command += ["lock", "--upgrade"]
    elif args.action == "venv":
        command += ["sync", "--locked"]
    else:
        command += ["run", "--locked", "ruff", "format", "."]
    subprocess.run(command, cwd=project, env=env, check=True)


if __name__ == "__main__":
    main()
