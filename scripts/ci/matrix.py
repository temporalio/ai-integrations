#!/usr/bin/env python3
"""Emit the GitHub Actions test matrix for one Python plugin.

Reads `[ci] runtime-versions` from the plugin's plugin.toml (first = minimum,
last = maximum supported interpreter) and the requested profile:

  pr    ubuntu-latest x {min, max}
  full  pr + macos-latest and windows-latest at max

`time_skipping` and `dist` are true only for the ubuntu/max cell, which also
uploads the built distributions. Output: matrix={"include":[...]}.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import compact_json, load_toml, write_github_output  # noqa: E402

PROFILES = ("pr", "full")


def build_matrix(runtime_versions: list[str], profile: str) -> dict[str, list[dict[str, object]]]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    if not runtime_versions:
        raise ValueError("[ci] runtime-versions must list at least one version")
    versions = [str(v) for v in runtime_versions]
    lo, hi = versions[0], versions[-1]
    include: list[dict[str, object]] = []
    if lo != hi:
        include.append({"os": "ubuntu-latest", "runtime": lo, "time_skipping": False, "dist": False})
    include.append({"os": "ubuntu-latest", "runtime": hi, "time_skipping": True, "dist": True})
    if profile == "full":
        for runner in ("macos-latest", "windows-latest"):
            include.append({"os": runner, "runtime": hi, "time_skipping": False, "dist": False})
    return {"include": include}


def main(argv: list[str] | None = None) -> dict[str, list[dict[str, object]]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="pr")
    parser.add_argument("--github-output", default=None)
    args = parser.parse_args(argv)

    meta = load_toml(args.plugin_dir / "plugin.toml")
    versions = meta.get("ci", {}).get("runtime-versions", [])
    matrix = build_matrix(versions, args.profile)
    text = compact_json(matrix)
    print(f"matrix={text}")
    write_github_output(args.github_output or os.environ.get("GITHUB_OUTPUT"), {"matrix": text})
    return matrix


if __name__ == "__main__":
    main()
