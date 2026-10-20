"""Python CI operations, with build metadata supplied by Bazel."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from python.runfiles import runfiles


def main() -> int:
    if os.environ.get("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(os.environ["BUILD_WORKSPACE_DIRECTORY"])
    action, *arguments = sys.argv[1:]
    if action == "matrix":
        import argparse

        from _common import write_github_output

        parser = argparse.ArgumentParser()
        parser.add_argument("--plugin-dir", required=True)
        parser.add_argument("--github-output")
        args = parser.parse_args(arguments)
        metadata = json.loads(
            Path(
                runfiles.Create().Rlocation(
                    "+plugin_metadata+plugin_metadata/plugins.json"
                )
            ).read_text()
        )
        value = json.dumps(metadata[args.plugin_dir]["matrix"], separators=(",", ":"))
        print("matrix=" + value)
        write_github_output(
            args.github_output or os.environ.get("GITHUB_OUTPUT"), {"matrix": value}
        )
        return 0
    if action == "select":
        import detect_changes

        result = detect_changes.main(arguments)
        return 0
    if action == "conventions":
        import check_conventions

        return check_conventions.main(arguments)
    if action == "nightly":
        import nightly_report

        return nightly_report.main(arguments)
    raise ValueError(f"unknown Python CI action {action!r}")


if __name__ == "__main__":
    sys.exit(main())
