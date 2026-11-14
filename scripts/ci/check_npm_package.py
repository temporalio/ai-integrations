#!/usr/bin/env python3
"""Verify an npm tarball contains built entrypoints and the plugin's source and license."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path


def check_package(plugin_dir: Path, license_bytes: bytes) -> list[str]:
    errors: list[str] = []
    tarballs = sorted((plugin_dir / "dist").glob("*.tgz"))
    if len(tarballs) != 1:
        return ["expected exactly one npm tarball in dist"]
    with tarfile.open(tarballs[0], "r:gz") as archive:
        files = {member.name: archive.extractfile(member).read()
                 for member in archive.getmembers() if member.isfile()}
    expected = json.loads((plugin_dir / "package.json").read_text())
    package = json.loads(files.get("package/package.json", b"{}"))
    for key in ("name", "version", "dependencies", "peerDependencies", "exports"):
        if package.get(key) != expected.get(key):
            errors.append(f"packaged {key} differs from the tested manifest")
    if files.get("package/LICENSE") != license_bytes:
        errors.append("packaged LICENSE differs from the repository LICENSE")
    for path in files:
        if "__tests__" in Path(path).parts or "node_modules" in Path(path).parts or "/tests/" in path:
            errors.append(f"test or dependency file shipped: {path}")

    def targets(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [target for entry in value.values() for target in targets(entry)]
        if isinstance(value, list):
            return [target for entry in value for target in targets(entry)]
        return []

    for target in targets([package.get("main"), package.get("types"), package.get("exports", {})]):
        if f"package/{target.removeprefix('./')}" not in files:
            errors.append(f"missing compiled entrypoint: {target}")
    for source in (plugin_dir / "src").rglob("*.ts"):
        if "__tests__" in source.parts:
            continue
        relative = source.relative_to(plugin_dir).as_posix()
        if files.get(f"package/{relative}") != source.read_bytes():
            errors.append(f"packaged source differs from checkout: {relative}")
        compiled = source.relative_to(plugin_dir / "src").with_suffix(".js").as_posix()
        if f"package/lib/{compiled}" not in files:
            errors.append(f"missing compiled source: lib/{compiled}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    args = parser.parse_args()
    errors = check_package(args.plugin_dir.resolve(), Path(__file__).resolve().parents[2].joinpath("LICENSE").read_bytes())
    if errors:
        print("\n".join(f"FAIL: {error}" for error in errors))
        return 1
    print("OK: npm package contents, entrypoints, source and LICENSE verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
