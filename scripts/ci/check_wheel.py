#!/usr/bin/env python3
"""Verify the built wheel and sdist of a Python plugin.

  * exactly one wheel and one sdist in --dist
  * wheel ships <root-api>/__init__.py and py.typed, every .py under src/, and
    NO __init__.py for any parent package (temporalio/, temporalio/contrib/):
    those belong to the SDK wheel and sharing them would corrupt installs
  * METADATA: Name == coordinate, Version == pyproject version, License-Expression
    MIT (or legacy License: MIT), Requires-Dist includes temporalio
  * <dist-info>/licenses/LICENSE and <sdist>/LICENSE are byte-identical to the
    repository root LICENSE (catches a missing symlink target or a Windows stub)
  * sdist ships pyproject.toml, README.md, LICENSE and every .py under src/
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_toml, repo_root  # noqa: E402


def _metadata(text: str) -> dict[str, list[str]]:
    headers: dict[str, list[str]] = {}
    for line in text.split("\n\n", 1)[0].splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            headers.setdefault(key.strip(), []).append(value.strip())
    return headers


def check(plugin_dir: Path, dist: Path, root: Path | None = None) -> list[str]:
    problems: list[str] = []
    root = root or repo_root()
    meta = load_toml(plugin_dir / "plugin.toml")["plugin"]
    project = load_toml(plugin_dir / "pyproject.toml")["project"]
    coordinate = meta["coordinate"]
    root_api = meta["root-api"]
    version = project["version"]
    pkg_path = root_api.replace(".", "/")
    parents = []
    parts = pkg_path.split("/")
    for i in range(1, len(parts)):
        parents.append("/".join(parts[:i]) + "/__init__.py")
    src = plugin_dir / "src"
    src_py = sorted(str(p.relative_to(src)).replace("\\", "/") for p in src.rglob("*.py"))
    license_bytes = (root / "LICENSE").read_bytes()

    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1:
        problems.append(f"expected exactly one wheel in {dist}, found {[w.name for w in wheels]}")
    if len(sdists) != 1:
        problems.append(f"expected exactly one sdist in {dist}, found {[s.name for s in sdists]}")
    if problems:
        return problems

    with zipfile.ZipFile(wheels[0]) as whl:
        names = set(whl.namelist())
        for required in (f"{pkg_path}/__init__.py", f"{pkg_path}/py.typed"):
            if required not in names:
                problems.append(f"wheel missing {required}")
        for parent in parents:
            if parent in names:
                problems.append(f"wheel must not ship {parent} (owned by the temporalio SDK wheel)")
        for rel in src_py:
            if rel not in names:
                problems.append(f"wheel missing source file {rel}")
        dist_info = sorted(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
        if len(dist_info) != 1:
            problems.append("wheel must contain exactly one .dist-info/METADATA")
            return problems
        info = dist_info[0]
        headers = _metadata(whl.read(f"{info}/METADATA").decode("utf-8"))
        if canonicalize_name(headers.get("Name", [""])[0]) != canonicalize_name(coordinate):
            problems.append(f"METADATA Name {headers.get('Name')} != coordinate {coordinate}")
        try:
            if Version(headers.get("Version", ["0"])[0]) != Version(version):
                problems.append(f"METADATA Version {headers.get('Version')} != pyproject version {version}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"METADATA Version unparsable: {exc}")
        if "MIT" not in headers.get("License-Expression", []) and "MIT" not in headers.get("License", []):
            problems.append("METADATA must declare License-Expression: MIT")
        if not any(r.split(";")[0].strip().startswith("temporalio") for r in headers.get("Requires-Dist", [])):
            problems.append("METADATA Requires-Dist must include temporalio")
        license_name = f"{info}/licenses/LICENSE"
        if license_name not in names:
            problems.append(f"wheel missing {license_name}")
        elif whl.read(license_name) != license_bytes:
            problems.append(f"wheel {license_name} differs from the repository root LICENSE (symlink not followed?)")

    with tarfile.open(sdists[0], "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        tops = {n.split("/")[0] for n in members}
        if len(tops) != 1:
            problems.append(f"sdist must have a single top-level directory, found {sorted(tops)}")
            return problems
        top = tops.pop()
        for required in ("pyproject.toml", "README.md", "LICENSE"):
            if f"{top}/{required}" not in members:
                problems.append(f"sdist missing {required}")
        for rel in src_py:
            if f"{top}/src/{rel}" not in members:
                problems.append(f"sdist missing src/{rel}")
        lic = members.get(f"{top}/LICENSE")
        if lic is not None:
            if not lic.isfile():
                problems.append("sdist LICENSE must be a regular file with the license text (symlink must be dereferenced)")
            else:
                fh = tar.extractfile(lic)
                if fh is None or fh.read() != license_bytes:
                    problems.append("sdist LICENSE differs from the repository root LICENSE")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)
    problems = check(args.plugin_dir.resolve(), args.dist.resolve(), args.repo_root.resolve() if args.repo_root else None)
    if problems:
        print(f"FAIL: {len(problems)} problem(s) with the built distributions:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: wheel and sdist in {args.dist} pass all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
