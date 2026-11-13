#!/usr/bin/env python3
"""Verify a .NET plugin's packed NuGet package and symbol package against the tested source.

The ubuntu/max cell of _dotnet-plugin.yml runs this after `make build`, and the release pipeline
publishes the same files. Mirrors check_java_dist.py: exactly one `.nupkg` and one `.snupkg` named
`<coordinate>.<version>`; nuspec identity, authors, file license, packed README, icon, Source Link
repository metadata (the commit must equal --commit when given) and a `Temporalio` dependency with
a lower bound in one dependency group per csproj target framework; packed LICENSE byte-identical to
the root LICENSE, README to the plugin README and icon to dotnet/_shared/icon.png; `lib/<tfm>/`
holds the assembly and its XML documentation; no debug symbols, no test project output and no
signature in the tested package; the symbol package carries the PDB for every target framework.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

REPOSITORY_URL = "https://github.com/temporalio/ai-integrations"
SIGNATURE = ".signature.p7s"


def csproj_frameworks(csproj: Path) -> list[str]:
    """Target framework monikers a csproj builds (TargetFramework or the TargetFrameworks list)."""
    frameworks: list[str] = []
    for element in ET.parse(csproj).getroot().iter():
        if element.tag == "TargetFrameworks" and element.text:
            frameworks.extend(part.strip() for part in element.text.split(";") if part.strip())
        elif element.tag == "TargetFramework" and element.text and element.text.strip():
            frameworks.append(element.text.strip())
    if not frameworks:
        raise ValueError(f"{csproj}: no TargetFramework(s)")
    return frameworks


def nuspec_framework(tfm: str) -> str:
    """Name a nuspec dependency group uses for a csproj moniker (net5.0 and later keep the moniker)."""
    if match := re.fullmatch(r"netstandard(\d+\.\d+)", tfm):
        return f".NETStandard{match.group(1)}"
    if match := re.fullmatch(r"netcoreapp(\d+\.\d+)", tfm):
        return f".NETCoreApp{match.group(1)}"
    if match := re.fullmatch(r"net(\d)(\d)(\d)?", tfm):
        return ".NETFramework" + ".".join(part for part in match.groups() if part)
    return tfm


def has_lower_bound(version_range: str) -> bool:
    """`1.22.0`, `[1.22.0, )` and `[1.22.0]` have one; `(, 2.0.0)` and an empty range do not."""
    text = version_range.strip()
    if not text:
        return False
    if text[0] in "[(":
        return bool(text[1:].rstrip("])").split(",")[0].strip())
    return True


def read_nuspec(archive: zipfile.ZipFile) -> ET.Element:
    names = [name for name in archive.namelist() if name.endswith(".nuspec") and "/" not in name]
    if len(names) != 1:
        raise ValueError(f"{archive.filename}: expected exactly one root nuspec, found {names}")
    metadata = ET.fromstring(archive.read(names[0])).find("{*}metadata")
    if metadata is None:
        raise ValueError(f"{archive.filename}: nuspec has no metadata element")
    return metadata


def nuspec_text(metadata: ET.Element, tag: str) -> str:
    return (metadata.findtext(f"{{*}}{tag}") or "").strip()


def check(plugin_dir: Path, version: str, dist: Path | None = None, repo_root: Path | None = None, commit: str | None = None) -> None:
    plugin_dir = plugin_dir.resolve()
    repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    meta = tomllib.loads((plugin_dir / "plugin.toml").read_text(encoding="utf-8"))
    coordinate = meta["plugin"]["coordinate"]
    dist = (dist or plugin_dir / "dist").resolve()
    base = f"{coordinate}.{version}"
    expected = {f"{base}.nupkg", f"{base}.snupkg"}
    actual = {path.name for path in dist.iterdir()}
    if actual != expected or any(not path.is_file() or path.is_symlink() for path in dist.iterdir()):
        raise ValueError(f"publication files differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    frameworks = csproj_frameworks(plugin_dir / "src" / coordinate / f"{coordinate}.csproj")
    packed_files = {
        "LICENSE": (repo_root / "LICENSE").read_bytes(),
        "README.md": (plugin_dir / "README.md").read_bytes(),
        "icon.png": (repo_root / "dotnet" / "_shared" / "icon.png").read_bytes(),
    }
    with zipfile.ZipFile(dist / f"{base}.nupkg") as archive:
        names = archive.namelist()
        metadata = read_nuspec(archive)
        for field, value in (("id", coordinate), ("version", version), ("authors", "Temporal"),
                             ("readme", "README.md"), ("icon", "icon.png")):
            if nuspec_text(metadata, field) != value:
                raise ValueError(f"nuspec {field} must be {value!r}, got {nuspec_text(metadata, field)!r}")
        license_element = metadata.find("{*}license")
        if license_element is None or license_element.get("type") != "file" or (license_element.text or "").strip() != "LICENSE":
            raise ValueError('nuspec license must be <license type="file">LICENSE</license>')
        repository = metadata.find("{*}repository")
        if repository is None or repository.get("type") != "git" or repository.get("url") != REPOSITORY_URL or not repository.get("commit"):
            raise ValueError(f"nuspec repository must be type git at {REPOSITORY_URL} with a Source Link commit")
        if commit and repository.get("commit") != commit:
            raise ValueError(f"nuspec repository commit {repository.get('commit')} differs from the tested commit {commit}")
        groups = metadata.findall("{*}dependencies/{*}group")
        found = {group.get("targetFramework", "") for group in groups}
        wanted = {nuspec_framework(tfm) for tfm in frameworks}
        if found != wanted:
            raise ValueError(f"dependency groups {sorted(found)} differ from the csproj target frameworks {sorted(wanted)}")
        for group in groups:
            temporalio = [dep for dep in group.findall("{*}dependency") if dep.get("id", "").lower() == "temporalio"]
            if len(temporalio) != 1 or not has_lower_bound(temporalio[0].get("version", "")):
                raise ValueError(f"{group.get('targetFramework')}: exactly one Temporalio dependency with a lower bound is required")
        for name, payload in packed_files.items():
            if name not in names:
                raise ValueError(f"nupkg is missing {name}")
            if archive.read(name) != payload:
                raise ValueError(f"packaged {name} differs from the repository file")
        for tfm in frameworks:
            for suffix in (".dll", ".xml"):
                if f"lib/{tfm}/{coordinate}{suffix}" not in names:
                    raise ValueError(f"nupkg is missing lib/{tfm}/{coordinate}{suffix}")
        for name in names:
            if name.endswith(".pdb"):
                raise ValueError(f"{name}: debug symbols belong in the snupkg")
            if ".tests." in name.lower():
                raise ValueError(f"{name}: test project output shipped")
            if name == SIGNATURE:
                raise ValueError("the tested package must be unsigned; nuget.org adds the repository signature")
    with zipfile.ZipFile(dist / f"{base}.snupkg") as archive:
        names = archive.namelist()
        metadata = read_nuspec(archive)
        if nuspec_text(metadata, "id") != coordinate or nuspec_text(metadata, "version") != version:
            raise ValueError("snupkg nuspec id or version differs from the nupkg")
        for tfm in frameworks:
            if f"lib/{tfm}/{coordinate}.pdb" not in names:
                raise ValueError(f"snupkg is missing lib/{tfm}/{coordinate}.pdb")
    print(f"OK: {coordinate} {version}: package identity, license, README, icon, Source Link, dependencies and symbols")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--commit", help="Source Link commit the package must record (git rev-parse HEAD on CI)")
    args = parser.parse_args()
    try:
        check(args.plugin_dir, args.version, args.dist, args.repo_root, args.commit)
    except (ValueError, OSError, KeyError, ET.ParseError, zipfile.BadZipFile) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
