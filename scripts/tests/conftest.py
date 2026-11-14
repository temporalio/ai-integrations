"""Shared fixtures: a temporary git repository shaped like ai-integrations with one fake plugin."""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_VERSION = ">=0.13.0,<0.14"
LICENSE_TEXT = "MIT License\n\nCopyright (c) 2026 Temporal Technologies Inc.\n\nPermission is hereby granted...\n"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "ci@example.com")
    git(root, "config", "user.name", "CI")
    git(root, "config", "commit.gpgsign", "false")
    (root / "LICENSE").write_text(LICENSE_TEXT)
    (root / ".gitignore").write_text(".venv/\ndist/\n")
    (root / "README.md").write_text("# repo\n")
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "pyproject.toml").write_text(
        f'[project]\nname = "scripts"\nversion = "0"\n\n[tool.uv]\npackage = false\nrequired-version = "{REQUIRED_VERSION}"\n'
    )
    return root


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def make_python_plugin(
    root: Path,
    name: str = "fakeplug",
    *,
    version: str = "0.0.0",
    maturity: str = "pre-release",
    dependencies: list[str] | None = None,
) -> Path:
    classifier = {
        "generally-available": "Development Status :: 5 - Production/Stable",
        "public-preview": "Development Status :: 4 - Beta",
        "pre-release": "Development Status :: 3 - Alpha",
    }[maturity]
    coordinate = "temporalio-" + name.replace("_", "-")
    d = root / "python" / name
    pkg = d / "src" / "temporalio" / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text('"""Fake plugin."""\n\n__all__ = ["hello"]\n\n\ndef hello() -> str:\n    return "hi"\n')
    (pkg / "_impl.py").write_text("VALUE = 1\n")
    (pkg / "py.typed").write_text("")
    deps = dependencies if dependencies is not None else ["temporalio>=1.36.0,<2"]
    deps_toml = ", ".join(f'"{x}"' for x in deps)
    (d / "pyproject.toml").write_text(textwrap.dedent(f'''
        [project]
        name = "{coordinate}"
        version = "{version}"
        description = "fake"
        requires-python = ">=3.11"
        readme = "README.md"
        license = "MIT"
        license-files = ["LICENSE"]
        classifiers = ["{classifier}", "Typing :: Typed"]
        dependencies = [{deps_toml}]

        [build-system]
        requires = ["uv_build>=0.12.5,<0.13"]
        build-backend = "uv_build"

        [tool.uv.build-backend]
        module-name = "temporalio.{name}"

        [tool.uv]
        required-version = "{REQUIRED_VERSION}"
        exclude-newer = "2 weeks"
        exclude-newer-package = {{ temporalio = false }}
    ''').lstrip())
    (d / "plugin.toml").write_text(textwrap.dedent(f'''
        [plugin]
        name = "{name}"
        language = "python"
        coordinate = "{coordinate}"
        registry = "pypi"
        root-api = "temporalio.{name}"
        maturity = "{maturity}"
        docs = "https://docs.temporal.io/"

        [ci]
        runtime-versions = ["3.11", "3.15"]

        [smoke]
        imports = ["temporalio.{name}._impl"]
    ''').lstrip())
    (d / "Makefile").write_text("DIST := " + coordinate + "\ninclude ../_shared/python.mk\n")
    (d / "README.md").write_text("# fake\n\nSee https://github.com/temporalio/ai-integrations for details.\n")
    (d / "uv.lock").write_text("version = 1\n")
    lic = d / "LICENSE"  # committed copy, byte-identical to the root LICENSE
    if lic.exists() or lic.is_symlink():
        lic.unlink()
    lic.write_text(LICENSE_TEXT)
    return d


def dotnet_lock(packages: dict[str, str], framework: str = "net10.0") -> str:
    """A packages.lock.json with one direct entry per package (NuGet lock format version 2)."""
    entries = {
        package: {"type": "Direct", "requested": f"[{version}, )", "resolved": version, "contentHash": "sha512-fake"}
        for package, version in packages.items()
    }
    return json.dumps({"version": 2, "dependencies": {framework: entries}}, indent=2) + "\n"


def make_dotnet_plugin(
    root: Path,
    name: str = "fakedotnet",
    coordinate: str = "Temporalio.Fakedotnet",
    *,
    maturity: str = "pre-release",
) -> Path:
    """Write a .NET plugin that satisfies check_conventions.py, plus the shared files it imports."""
    shared = root / "dotnet" / "_shared"
    shared.mkdir(parents=True, exist_ok=True)
    for source in ("dotnet.props", "dotnet.mk"):
        (shared / source).write_bytes((REPO_ROOT / "dotnet" / "_shared" / source).read_bytes())
    template = root / "dotnet" / "_template"
    template.mkdir(parents=True, exist_ok=True)
    (template / ".editorconfig").write_bytes((REPO_ROOT / "dotnet" / "_template" / ".editorconfig").read_bytes())
    d = root / "dotnet" / name
    lib = d / "src" / coordinate
    tests = d / "tests" / f"{coordinate}.Tests"
    lib.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    (d / "plugin.toml").write_text(textwrap.dedent(f'''
        [plugin]
        name = "{name}"
        language = "dotnet"
        coordinate = "{coordinate}"
        registry = "nuget"
        root-api = "{coordinate}"
        maturity = "{maturity}"
        docs = "https://github.com/temporalio/ai-integrations/tree/main/dotnet/{name}"

        [ci]
        runtime-versions = ["10.0"]

        [smoke]
        imports = ["{coordinate}.FakedotnetPlugin"]
    ''').lstrip())
    (d / "Directory.Build.props").write_text(
        '<Project>\n  <Import Project="$(MSBuildThisFileDirectory)../_shared/dotnet.props" />\n</Project>\n'
    )
    (d / f"{coordinate}.slnx").write_text(
        f'<Solution>\n  <Project Path="src/{coordinate}/{coordinate}.csproj" />\n'
        f'  <Project Path="tests/{coordinate}.Tests/{coordinate}.Tests.csproj" />\n</Solution>\n'
    )
    (d / "global.json").write_text(json.dumps({
        "sdk": {"version": "10.0.100", "rollForward": "latestFeature", "allowPrerelease": False},
        "test": {"runner": "Microsoft.Testing.Platform"},
    }, indent=2) + "\n")
    (d / "nuget.config").write_text(textwrap.dedent('''
        <?xml version="1.0" encoding="utf-8"?>
        <configuration>
          <packageSources>
            <clear />
            <add key="nuget.org" value="https://api.nuget.org/v3/index.json" protocolVersion="3" minPublishAgeHours="336" />
          </packageSources>
          <minPublishAgeExceptions>
            <package pattern="Temporalio" />
            <package pattern="Temporalio.*" />
          </minPublishAgeExceptions>
        </configuration>
    ''').lstrip())
    (d / ".editorconfig").write_bytes((template / ".editorconfig").read_bytes())
    (d / "Makefile").write_text(f"PACKAGE := {coordinate}\ninclude ../_shared/dotnet.mk\n")
    (d / "README.md").write_text(f"# {coordinate}\n\nSee https://github.com/temporalio/ai-integrations for details.\n")
    (d / "LICENSE").write_bytes((root / "LICENSE").read_bytes())
    (lib / f"{coordinate}.csproj").write_text(textwrap.dedent(f'''
        <Project Sdk="Microsoft.NET.Sdk">
          <PropertyGroup>
            <TargetFramework>net10.0</TargetFramework>
            <PackageId>{coordinate}</PackageId>
            <RootNamespace>{coordinate}</RootNamespace>
            <Version>0.0.0</Version>
            <IsPackable>true</IsPackable>
            <Description>Fake plugin (Pre-release)</Description>
            <PackageLicenseFile>LICENSE</PackageLicenseFile>
            <PackageReadmeFile>README.md</PackageReadmeFile>
            <PackageProjectUrl>https://github.com/temporalio/ai-integrations/tree/main/dotnet/{name}</PackageProjectUrl>
          </PropertyGroup>
          <ItemGroup>
            <None Include="../../LICENSE" Pack="true" PackagePath="" />
            <None Include="../../README.md" Pack="true" PackagePath="" />
          </ItemGroup>
          <ItemGroup>
            <PackageReference Include="Temporalio" Version="1.22.0" />
            <PackageReference Include="StyleCop.Analyzers" Version="1.3.180" PrivateAssets="all" />
          </ItemGroup>
        </Project>
    ''').lstrip())
    (lib / "FakedotnetPlugin.cs").write_text(
        f"namespace {coordinate};\n\n/// <summary>Fixture plugin used to exercise the shared .NET build logic.</summary>\n"
        f"public sealed class FakedotnetPlugin\n{{\n    /// <summary>Gets the plugin name.</summary>\n"
        f"    public string Name {{ get; }} = \"{coordinate}\";\n}}\n"
    )
    (lib / "packages.lock.json").write_text(dotnet_lock({"Temporalio": "1.22.0", "StyleCop.Analyzers": "1.3.180"}))
    (tests / f"{coordinate}.Tests.csproj").write_text(textwrap.dedent(f'''
        <Project Sdk="Microsoft.NET.Sdk">
          <PropertyGroup>
            <TargetFramework>net10.0</TargetFramework>
            <OutputType>Exe</OutputType>
            <UseMicrosoftTestingPlatformRunner>true</UseMicrosoftTestingPlatformRunner>
            <GenerateDocumentationFile>false</GenerateDocumentationFile>
          </PropertyGroup>
          <ItemGroup>
            <ProjectReference Include="../../src/{coordinate}/{coordinate}.csproj" />
            <PackageReference Include="xunit.v3" Version="4.0.3" />
            <PackageReference Include="Microsoft.Testing.Extensions.TrxReport" Version="2.5.3" />
            <PackageReference Include="StyleCop.Analyzers" Version="1.3.180" PrivateAssets="all" />
            <Using Include="Xunit" />
          </ItemGroup>
        </Project>
    ''').lstrip())
    (tests / "FakedotnetPluginTests.cs").write_text(
        f"namespace {coordinate}.Tests;\n\npublic class FakedotnetPluginTests\n{{\n    [Fact]\n"
        f"    public void Plugin_Has_Name()\n    {{\n        Assert.Equal(\"{coordinate}\", new FakedotnetPlugin().Name);\n    }}\n}}\n"
    )
    (tests / "packages.lock.json").write_text(dotnet_lock({
        "xunit.v3": "4.0.3", "Microsoft.Testing.Extensions.TrxReport": "2.5.3", "StyleCop.Analyzers": "1.3.180",
    }))
    return d


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo")


@pytest.fixture
def dotnet_repo(repo: Path) -> Path:
    make_dotnet_plugin(repo)
    commit_all(repo, "add fake .NET plugin")
    return repo


@pytest.fixture
def plugin_repo(repo: Path) -> Path:
    make_python_plugin(repo)
    commit_all(repo, "add fake plugin")
    return repo


def uv_available() -> bool:
    return shutil.which("uv") is not None


def build_plugin(plugin_dir: Path) -> Path:
    dist = plugin_dir / "dist"
    subprocess.run(["uv", "build", "--no-sources", "--out-dir", str(dist), str(plugin_dir)], check=True, capture_output=True, text=True)
    return dist
