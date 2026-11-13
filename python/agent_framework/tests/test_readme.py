"""The README's Python blocks must compile as written."""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"

_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _python_blocks() -> list[str]:
    return _BLOCK.findall(README.read_text(encoding="utf-8"))


def test_readme_has_python_blocks() -> None:
    assert len(_python_blocks()) >= 3, "expected the quickstart and feature examples"


def test_readme_blocks_compile() -> None:
    for index, block in enumerate(_python_blocks()):
        compile(block, f"<README block {index}>", "exec")
