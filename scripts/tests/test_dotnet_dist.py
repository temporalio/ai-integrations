"""check_dotnet_dist.py accepts the packed pair the shared build produces and rejects drift."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from check_dotnet_dist import check, has_lower_bound, nuspec_framework

COORDINATE = "Temporalio.Fakedotnet"
VERSION = "0.0.1-rc.1"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
LICENSE = b"MIT license text\n"
ICON = b"\x89PNG fake icon"
README = b"# fake\n\nhttps://github.com/temporalio/ai-integrations\n"


def nuspec(*, version: str = VERSION, license_type: str = "file", groups: tuple[str, ...] = ("net10.0",),
           temporalio: str | None = "1.22.0", commit: str = COMMIT, icon: str = "icon.png") -> str:
    dependency = f'<dependency id="Temporalio" version="{temporalio}" exclude="Build,Analyzers" />' if temporalio else ""
    dependencies = "".join(f'<group targetFramework="{group}">{dependency}</group>' for group in groups)
    return f'''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">
  <metadata>
    <id>{COORDINATE}</id>
    <version>{version}</version>
    <authors>Temporal</authors>
    <license type="{license_type}">LICENSE</license>
    <icon>{icon}</icon>
    <readme>README.md</readme>
    <repository type="git" url="https://github.com/temporalio/ai-integrations" commit="{commit}" />
    <dependencies>{dependencies}</dependencies>
  </metadata>
</package>'''


def write_package(path: Path, nuspec_text: str, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{COORDINATE}.nuspec", nuspec_text)
        for name, payload in entries.items():
            archive.writestr(name, payload)


def rewrite(path: Path, *, drop: tuple[str, ...] = (), add: dict[str, bytes] | None = None) -> None:
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist() if name not in drop}
    entries.update(add or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def publication(tmp_path: Path, **nuspec_overrides: object) -> tuple[Path, Path]:
    """A plugin directory and repository root with a valid nupkg and snupkg pair in dist/."""
    root = tmp_path / "repo"
    (root / "dotnet" / "_shared").mkdir(parents=True)
    (root / "LICENSE").write_bytes(LICENSE)
    (root / "dotnet" / "_shared" / "icon.png").write_bytes(ICON)
    plugin = root / "dotnet" / "fakedotnet"
    (plugin / "src" / COORDINATE).mkdir(parents=True)
    (plugin / "plugin.toml").write_text(f'[plugin]\ncoordinate = "{COORDINATE}"\n')
    (plugin / "README.md").write_bytes(README)
    (plugin / "src" / COORDINATE / f"{COORDINATE}.csproj").write_text(
        '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework></PropertyGroup></Project>'
    )
    dist = plugin / "dist"
    dist.mkdir()
    write_package(dist / f"{COORDINATE}.{VERSION}.nupkg", nuspec(**nuspec_overrides), {
        "LICENSE": LICENSE, "README.md": README, "icon.png": ICON,
        f"lib/net10.0/{COORDINATE}.dll": b"MZ", f"lib/net10.0/{COORDINATE}.xml": b"<doc/>",
    })
    write_package(dist / f"{COORDINATE}.{VERSION}.snupkg", nuspec(**nuspec_overrides), {f"lib/net10.0/{COORDINATE}.pdb": b"pdb"})
    return plugin, root


def test_valid_publication(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plugin, root = publication(tmp_path)
    check(plugin, VERSION, repo_root=root, commit=COMMIT)
    assert capsys.readouterr().out.startswith(f"OK: {COORDINATE} {VERSION}")
    check(plugin, VERSION, repo_root=root)  # the commit is optional outside CI


@pytest.mark.parametrize("fault,message", [
    ("version", "nuspec version"),
    ("license-expression", "license must be"),
    ("license-drift", "packaged LICENSE differs"),
    ("readme-drift", "packaged README.md differs"),
    ("icon-drift", "packaged icon.png differs"),
    ("missing-xml", "missing lib/net10.0/Temporalio.Fakedotnet.xml"),
    ("pdb-in-nupkg", "debug symbols belong in the snupkg"),
    ("test-output", "test project output shipped"),
    ("signed", "must be unsigned"),
    ("extra-file", "publication files differ"),
    ("missing-snupkg", "publication files differ"),
    ("missing-temporalio", "Temporalio dependency with a lower bound"),
    ("open-lower-bound", "Temporalio dependency with a lower bound"),
    ("group-framework", "dependency groups"),
    ("commit", "differs from the tested commit"),
    ("no-commit", "Source Link commit"),
    ("snupkg-pdb", "snupkg is missing lib/net10.0/Temporalio.Fakedotnet.pdb"),
    ("snupkg-version", "snupkg nuspec id or version differs"),
])
def test_publication_rejects_invalid_artifacts(tmp_path: Path, fault: str, message: str) -> None:
    overrides: dict[str, object] = {
        "version": {"version": "0.0.0"}, "license-expression": {"license_type": "expression"},
        "missing-temporalio": {"temporalio": None}, "open-lower-bound": {"temporalio": "(, 2.0.0)"},
        "group-framework": {"groups": ("netstandard2.0",)}, "commit": {"commit": "feedfacefeedfacefeedfacefeedfacefeedface"},
        "no-commit": {"commit": ""},
    }.get(fault, {})
    plugin, root = publication(tmp_path, **overrides)
    dist = plugin / "dist"
    nupkg = dist / f"{COORDINATE}.{VERSION}.nupkg"
    snupkg = dist / f"{COORDINATE}.{VERSION}.snupkg"
    if fault == "license-drift":
        rewrite(nupkg, add={"LICENSE": b"other license\n"})
    elif fault == "readme-drift":
        rewrite(nupkg, add={"README.md": b"# other\n"})
    elif fault == "icon-drift":
        rewrite(nupkg, add={"icon.png": b"other"})
    elif fault == "missing-xml":
        rewrite(nupkg, drop=(f"lib/net10.0/{COORDINATE}.xml",))
    elif fault == "pdb-in-nupkg":
        rewrite(nupkg, add={f"lib/net10.0/{COORDINATE}.pdb": b"pdb"})
    elif fault == "test-output":
        rewrite(nupkg, add={f"lib/net10.0/{COORDINATE}.Tests.dll": b"MZ"})
    elif fault == "signed":
        rewrite(nupkg, add={".signature.p7s": b"signature"})
    elif fault == "extra-file":
        (dist / "notes.txt").write_text("extra\n")
    elif fault == "missing-snupkg":
        snupkg.unlink()
    elif fault == "snupkg-pdb":
        rewrite(snupkg, drop=(f"lib/net10.0/{COORDINATE}.pdb",))
    elif fault == "snupkg-version":
        rewrite(snupkg, add={f"{COORDINATE}.nuspec": nuspec(version="0.0.0").encode()})
    with pytest.raises(ValueError, match=message):
        check(plugin, VERSION, repo_root=root, commit=COMMIT)


def test_multi_target_packages_need_every_framework(tmp_path: Path) -> None:
    plugin, root = publication(tmp_path, groups=("net10.0", ".NETStandard2.0"))
    csproj = plugin / "src" / COORDINATE / f"{COORDINATE}.csproj"
    csproj.write_text(csproj.read_text().replace("<TargetFramework>net10.0</TargetFramework>",
                                                 "<TargetFrameworks>net10.0;netstandard2.0</TargetFrameworks>"))
    with pytest.raises(ValueError, match="missing lib/netstandard2.0/Temporalio.Fakedotnet.dll"):
        check(plugin, VERSION, repo_root=root, commit=COMMIT)


@pytest.mark.parametrize("tfm,expected", [
    ("net10.0", "net10.0"), ("net8.0", "net8.0"), ("netstandard2.0", ".NETStandard2.0"),
    ("netcoreapp3.1", ".NETCoreApp3.1"), ("net462", ".NETFramework4.6.2"), ("net48", ".NETFramework4.8"),
])
def test_nuspec_group_names(tfm: str, expected: str) -> None:
    assert nuspec_framework(tfm) == expected


@pytest.mark.parametrize("version_range,expected", [
    ("1.22.0", True), ("[1.22.0, )", True), ("[1.22.0]", True), ("[1.22.0, 2.0.0)", True),
    ("(, 2.0.0)", False), ("", False), ("   ", False),
])
def test_lower_bounds(version_range: str, expected: bool) -> None:
    assert has_lower_bound(version_range) is expected
