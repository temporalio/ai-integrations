"""Build the declared uv_build distributions without installing or resolving anything."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from contextlib import chdir
from pathlib import Path

import uv_build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()
    wheel, sdist = args.wheel.absolute(), args.sdist.absolute()
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="python-dist-") as directory:
        root = Path(directory)
        # The builder's Python 3.14 transition owns this platform wheel. Resolve
        # its installed executable instead of selecting it in the plugin's runtime.
        binary_dir = Path(uv_build.__file__).parents[2] / "bin"
        binary = binary_dir / ("uv-build.exe" if os.name == "nt" else "uv-build")
        if not binary.is_file():
            raise FileNotFoundError("The pinned uv_build executable is missing")
        os.environ["PATH"] = str(binary_dir) + os.pathsep + os.environ.get("PATH", "")
        for filename in args.source:
            source = Path(filename)
            relative = source.relative_to(args.project)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        manifest = root / "pyproject.toml"
        manifest.write_text(
            re.sub(
                r'(?m)^version\s*=\s*"[^"]+"',
                f'version = "{args.version}"',
                manifest.read_text(),
                count=1,
            )
        )
        with chdir(root):
            actual_wheel = Path(uv_build.build_wheel(str(wheel.parent)))
            actual_sdist = Path(uv_build.build_sdist(str(sdist.parent)))
        if actual_wheel.name != wheel.name or actual_sdist.name != sdist.name:
            raise ValueError(
                f"unexpected distribution names: {actual_wheel}, {actual_sdist}"
            )


if __name__ == "__main__":
    main()
