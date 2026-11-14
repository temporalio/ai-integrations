"""Packaging checks must catch missing runtime code and accidental test shipments."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from check_npm_package import check_package


def _package(tmp_path: Path, change: str = "") -> Path:
    plugin = tmp_path / "plugin"
    (plugin / "src").mkdir(parents=True)
    (plugin / "dist").mkdir()
    source = b"export const value = 1;\n"
    (plugin / "src/index.ts").write_bytes(source)
    manifest = {"name": "@temporalio/fake", "version": "0.0.0", "main": "lib/index.js",
                "types": "lib/index.d.ts", "exports": {".": {"require": "./lib/index.js"}}}
    (plugin / "package.json").write_text(json.dumps(manifest))
    files = {"package/package.json": json.dumps(manifest).encode(), "package/LICENSE": b"MIT\n",
             "package/src/index.ts": source, "package/lib/index.js": b"exports.value = 1;",
             "package/lib/index.d.ts": b"export declare const value = 1;"}
    if change == "missing-code":
        del files["package/lib/index.js"]
    elif change == "test-leak":
        files["package/lib/__tests__/test.js"] = b"test();"
    elif change == "source-drift":
        files["package/src/index.ts"] = b"export const value = 2;"
    elif change == "license-drift":
        files["package/LICENSE"] = b"other license"
    with tarfile.open(plugin / "dist/fake.tgz", "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return plugin


def test_valid_npm_package(tmp_path: Path) -> None:
    assert check_package(_package(tmp_path), b"MIT\n") == []


def test_missing_runtime_entrypoint(tmp_path: Path) -> None:
    errors = check_package(_package(tmp_path, "missing-code"), b"MIT\n")
    assert any("missing compiled entrypoint" in error for error in errors)


def test_tests_do_not_ship(tmp_path: Path) -> None:
    errors = check_package(_package(tmp_path, "test-leak"), b"MIT\n")
    assert any("test or dependency file shipped" in error for error in errors)


def test_built_source_matches_checkout(tmp_path: Path) -> None:
    errors = check_package(_package(tmp_path, "source-drift"), b"MIT\n")
    assert any("packaged source differs" in error for error in errors)


def test_license_matches_repository(tmp_path: Path) -> None:
    errors = check_package(_package(tmp_path, "license-drift"), b"MIT\n")
    assert any("LICENSE differs" in error for error in errors)
