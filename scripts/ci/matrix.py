#!/usr/bin/env python3
"""Emit the GitHub Actions test matrix for one plugin.

Reads `[ci] runtime-versions` from the plugin's plugin.toml (first = minimum,
last = maximum supported runtime) and emits the same matrix for every run,
pull requests included:

  ubuntu-latest x {min, max}, macos-latest and windows-latest at max

`dist` marks the ubuntu/max cell: Python builds and uploads distributions there,
and Go runs race detection there. Output: matrix={"include":[...]}.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import compact_json, load_toml, write_github_output  # noqa: E402

def build_matrix(runtime_versions: list[str], spring_boot_versions: list[str] | None = None) -> dict[str, list[dict[str, object]]]:
    if not runtime_versions:
        raise ValueError("[ci] runtime-versions must list at least one version")
    versions = [str(v) for v in runtime_versions]
    lo, hi = versions[0], versions[-1]
    include: list[dict[str, object]] = []
    if lo != hi:
        include.append({"os": "ubuntu-latest", "runtime": lo, "dist": False})
    include.append({"os": "ubuntu-latest", "runtime": hi, "dist": True})
    for runner in ("macos-latest", "windows-latest"):
        include.append({"os": runner, "runtime": hi, "dist": False})
    if spring_boot_versions:
        include = [dict(cell, spring_boot=boot, dist=bool(cell["dist"] and i == 0))
                   for i, boot in enumerate(spring_boot_versions) for cell in include]
    return {"include": include}


def main(argv: list[str] | None = None) -> dict[str, list[dict[str, object]]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--github-output", default=None)
    args = parser.parse_args(argv)

    meta = load_toml(args.plugin_dir / "plugin.toml")
    versions = meta.get("ci", {}).get("runtime-versions", [])
    matrix = build_matrix(versions, meta.get("ci", {}).get("spring-boot-versions"))
    text = compact_json(matrix)
    print(f"matrix={text}")
    write_github_output(args.github_output or os.environ.get("GITHUB_OUTPUT"), {"matrix": text})
    return matrix


if __name__ == "__main__":
    main()
