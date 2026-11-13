"""Every .NET plugin follows the dotnet/<name>/ contract the shared build files assume."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from check_conventions import Checker
from conftest import make_dotnet_plugin

PLUGIN = "dotnet/fakedotnet"
LIB = "src/Temporalio.Fakedotnet/Temporalio.Fakedotnet.csproj"
TESTS = "tests/Temporalio.Fakedotnet.Tests/Temporalio.Fakedotnet.Tests.csproj"


def edit(repo: Path, rel: str, old: str, new: str) -> None:
    path = repo / PLUGIN / rel
    text = path.read_text()
    assert old in text, (rel, old)
    path.write_text(text.replace(old, new))


def test_fake_dotnet_plugin_passes(dotnet_repo: Path) -> None:
    assert Checker(dotnet_repo).run() == []


def test_shared_files_without_plugins_pass(repo: Path) -> None:
    shutil.rmtree(make_dotnet_plugin(repo))
    assert Checker(repo).run() == []


CASES: dict[str, tuple[Callable[[Path], None], str]] = {
    "coordinate-letters": (
        lambda r: edit(r, "plugin.toml", 'coordinate = "Temporalio.Fakedotnet"', 'coordinate = "Temporalio.Other"'),
        "coordinate must be 'Temporalio.fakedotnet' up to casing"),
    "root-api": (
        lambda r: edit(r, "plugin.toml", 'root-api = "Temporalio.Fakedotnet"', 'root-api = "Temporalio.Fakedotnet.Core"'),
        "root-api must be 'Temporalio.Fakedotnet'"),
    "registry": (lambda r: edit(r, "plugin.toml", 'registry = "nuget"', 'registry = "pypi"'), "registry must be 'nuget'"),
    "runtime-versions": (lambda r: edit(r, "plugin.toml", 'runtime-versions = ["10.0"]', 'runtime-versions = ["10"]'),
                         "major.minor"),
    "smoke-imports": (lambda r: edit(r, "plugin.toml", "Temporalio.Fakedotnet.FakedotnetPlugin", "Other.Plugin"),
                      "smoke imports must name types under Temporalio.Fakedotnet"),
    "package-id": (lambda r: edit(r, LIB, "<PackageId>Temporalio.Fakedotnet</PackageId>", "<PackageId>Temporalio.Other</PackageId>"),
                   "PackageId must be 'Temporalio.Fakedotnet'"),
    "version": (lambda r: edit(r, LIB, "<Version>0.0.0</Version>", "<Version>0.1.0</Version>"), "development placeholder"),
    "license-expression": (
        lambda r: edit(r, LIB, "<PackageLicenseFile>LICENSE</PackageLicenseFile>",
                       "<PackageLicenseFile>LICENSE</PackageLicenseFile><PackageLicenseExpression>MIT</PackageLicenseExpression>"),
        "no PackageLicenseExpression"),
    "aot": (lambda r: edit(r, LIB, "<IsPackable>true</IsPackable>", "<IsPackable>true</IsPackable><IsAotCompatible>true</IsAotCompatible>"),
            "IsAotCompatible is not allowed"),
    "weaker-policy": (
        lambda r: edit(r, LIB, "<IsPackable>true</IsPackable>", "<IsPackable>true</IsPackable><TreatWarningsAsErrors>false</TreatWarningsAsErrors>"),
        "TreatWarningsAsErrors must stay true"),
    "missing-temporalio": (lambda r: edit(r, LIB, '    <PackageReference Include="Temporalio" Version="1.22.0" />\n', ""),
                           "PackageReference to Temporalio"),
    "stylecop-not-private": (lambda r: edit(r, LIB, 'Version="1.3.180" PrivateAssets="all"', 'Version="1.3.180"'),
                             'StyleCop.Analyzers must be referenced with PrivateAssets="all"'),
    "vstest": (
        lambda r: edit(r, TESTS, "<Using Include=", '<PackageReference Include="Microsoft.NET.Test.Sdk" Version="18.0.0" />\n    <Using Include='),
        "Microsoft.NET.Test.Sdk is a VSTest package"),
    "mtp-runner-missing": (
        lambda r: edit(r, TESTS, "    <UseMicrosoftTestingPlatformRunner>true</UseMicrosoftTestingPlatformRunner>\n", ""),
        "UseMicrosoftTestingPlatformRunner must be true"),
    "tests-packable": (lambda r: edit(r, TESTS, "<OutputType>Exe</OutputType>", "<OutputType>Exe</OutputType><IsPackable>true</IsPackable>"),
                       "must not be packable"),
    "project-reference-outside": (
        lambda r: edit(r, TESTS, "../../src/Temporalio.Fakedotnet/Temporalio.Fakedotnet.csproj", "../../../other/src/Other/Other.csproj"),
        "points outside the plugin"),
    "global-json-roll-forward": (lambda r: edit(r, "global.json", '"rollForward": "latestFeature"', '"rollForward": "latestMajor"'),
                                 "rollForward must be 'latestFeature'"),
    "global-json-prerelease": (lambda r: edit(r, "global.json", '"allowPrerelease": false', '"allowPrerelease": true'),
                               "allowPrerelease must be false"),
    "global-json-floor": (lambda r: edit(r, "global.json", '"version": "10.0.100"', '"version": "9.0.100"'), "floor channel 10.0"),
    "global-json-runner": (lambda r: edit(r, "global.json", '"runner": "Microsoft.Testing.Platform"', '"runner": "VSTest"'),
                           "test.runner must be 'Microsoft.Testing.Platform'"),
    "editorconfig-drift": (lambda r: (r / PLUGIN / ".editorconfig").write_text((r / PLUGIN / ".editorconfig").read_text() + "\n# drift\n"),
                           "differs from canonical dotnet/_template/.editorconfig"),
    "props-import": (lambda r: (r / PLUGIN / "Directory.Build.props").write_text("<Project>\n  <PropertyGroup />\n</Project>\n"),
                     "must start with the shared import"),
    "nuget-config-cooldown": (lambda r: edit(r, "nuget.config", ' minPublishAgeHours="336"', ""), "must carry minPublishAgeHours"),
    "nuget-config-exception": (lambda r: edit(r, "nuget.config", '    <package pattern="Temporalio*" />\n', ""),
                               "must exempt the Temporalio* pattern"),
    "nuget-config-extra-source": (
        lambda r: edit(r, "nuget.config", "  </packageSources>", '    <add key="mirror" value="https://mirror.example/v3/index.json" />\n  </packageSources>'),
        "must be the only package source"),
    "nuget-config-clear": (lambda r: edit(r, "nuget.config", "    <clear />\n", ""), "must start with <clear />"),
    "makefile-package": (lambda r: edit(r, "Makefile", "PACKAGE := Temporalio.Fakedotnet", "PACKAGE := Temporalio.Other"),
                         "must set PACKAGE := Temporalio.Fakedotnet"),
    "makefile-include": (lambda r: edit(r, "Makefile", "include ../_shared/dotnet.mk", "include ../_shared/other.mk"),
                         "must include ../_shared/dotnet.mk"),
    "readme-relative-link": (lambda r: (r / PLUGIN / "README.md").write_text("see [root](../../README.md)\n"), "relative link"),
    "license-drift": (lambda r: (r / PLUGIN / "LICENSE").write_text("MIT\n"), "LICENSE differs from root"),
    "changelog": (lambda r: (r / PLUGIN / "CHANGELOG.md").write_text("# changes\n"), "no changelog files"),
    "second-slnx": (lambda r: (r / PLUGIN / "Other.slnx").write_text("<Solution />\n"), "exactly one Temporalio.Fakedotnet.slnx"),
    "slnx-missing-project": (
        lambda r: edit(r, "Temporalio.Fakedotnet.slnx", '  <Project Path="tests/Temporalio.Fakedotnet.Tests/Temporalio.Fakedotnet.Tests.csproj" />\n', ""),
        "must list src/Temporalio.Fakedotnet/Temporalio.Fakedotnet.csproj and tests/"),
    "lock-missing": (lambda r: (r / PLUGIN / "src/Temporalio.Fakedotnet/packages.lock.json").unlink(),
                     "missing src/Temporalio.Fakedotnet/packages.lock.json"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_contract_breach_is_one_violation(dotnet_repo: Path, case: str) -> None:
    mutate, expected = CASES[case]
    mutate(dotnet_repo)
    violations = Checker(dotnet_repo).run()
    assert len(violations) == 1, violations
    assert expected in violations[0]


def test_coordinate_prefix_casing_is_reported_first(dotnet_repo: Path) -> None:
    meta = dotnet_repo / PLUGIN / "plugin.toml"
    meta.write_text(meta.read_text().replace("Temporalio.Fakedotnet", "temporalio.Fakedotnet"))
    violations = Checker(dotnet_repo).run()
    assert violations[0] == (f"{PLUGIN}: plugin.toml coordinate 'temporalio.Fakedotnet' must keep the Temporalio. "
                             "prefix casing and be a valid NuGet ID")


def test_xunit_v2_is_rejected_and_xunit_v3_required(dotnet_repo: Path) -> None:
    edit(dotnet_repo, TESTS, 'Include="xunit.v3"', 'Include="xunit"')
    violations = Checker(dotnet_repo).run()
    assert any("must reference xunit.v3" in v for v in violations)
    assert any("xunit is a VSTest package" in v for v in violations)


def test_uncommitted_required_files_are_reported(dotnet_repo: Path) -> None:
    import subprocess
    subprocess.run(["git", "-C", str(dotnet_repo), "rm", "-q", "--cached", f"{PLUGIN}/tests/Temporalio.Fakedotnet.Tests/packages.lock.json"], check=True)
    assert Checker(dotnet_repo).run() == [f"{PLUGIN}: tests/Temporalio.Fakedotnet.Tests/packages.lock.json must be committed"]


def test_language_root_build_files_are_forbidden(dotnet_repo: Path) -> None:
    for name in ("Directory.Build.props", "global.json", "packages.lock.json"):
        (dotnet_repo / "dotnet" / name).write_text("{}\n")
    violations = Checker(dotnet_repo).run()
    assert len(violations) == 3, violations
    assert any("dotnet/Directory.Build.props: language-root build files are forbidden" in v for v in violations)
    assert any("dotnet/global.json: language-root build files are forbidden" in v for v in violations)
    assert any("dotnet/packages.lock.json: language-level lockfiles are forbidden" in v for v in violations)


def test_missing_shared_files_are_reported(dotnet_repo: Path) -> None:
    (dotnet_repo / "dotnet/_shared/dotnet.props").unlink()
    (dotnet_repo / "dotnet/_template/.editorconfig").unlink()
    violations = Checker(dotnet_repo).run()
    assert "dotnet/_shared/dotnet.props: missing shared build conventions" in violations
    assert "dotnet/_template/.editorconfig: missing canonical editorconfig" in violations


def test_coordinate_casing_in_plugin_toml_is_canonical(repo: Path) -> None:
    from conftest import commit_all
    make_dotnet_plugin(repo, "agent-framework", "Temporalio.AgentFramework")
    commit_all(repo, "add agent framework")
    assert Checker(repo).run() == []
