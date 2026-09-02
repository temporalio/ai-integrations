"""The installed, non-editable copy of this plugin must be byte-identical to ``src/``.

Complements the provenance guard in ``conftest.py``: that guard proves the installed files match
the distribution's RECORD; this test proves the distribution was built from this checkout.
"""

import hashlib
import importlib.metadata as importlib_metadata
import os
from pathlib import Path

from tests.helpers.plugin_meta import load_plugin_meta

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
META = load_plugin_meta(PLUGIN_ROOT)


def _digests(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)).replace(os.sep, "/"): hashlib.sha256(path.read_bytes()).digest()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
    }


def test_installed_matches_source() -> None:
    src_dir = PLUGIN_ROOT / "src" / META.package_relpath
    installed_dir = Path(importlib_metadata.distribution(META.coordinate).locate_file(META.package_relpath))
    expected = _digests(src_dir)
    actual = _digests(installed_dir)
    if not META.allow_final or os.environ.get("ALLOW_OVERLAP_WITH_CORE") == "1":
        # temporalio<=1.32 ships a README.md inside the package directory.
        actual.pop("README.md", None)
    assert set(actual) == set(expected), (
        f"installed file set differs from src/: missing={sorted(set(expected) - set(actual))} "
        f"extra={sorted(set(actual) - set(expected))}; run `make sync`"
    )
    mismatched = sorted(rel for rel in expected if expected[rel] != actual[rel])
    assert not mismatched, f"installed files differ from src/: {mismatched}; run `make sync`"
