"""Locate the pinned uv wheel executable in Bazel's installed wheel layout."""

import os
from pathlib import Path

import uv


def uv_binary() -> str:
    candidate = (
        Path(uv.__file__).parents[2] / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    )
    if candidate.is_file():
        return str(candidate.absolute())
    return uv.find_uv_bin()
