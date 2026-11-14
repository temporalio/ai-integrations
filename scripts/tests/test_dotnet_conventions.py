"""Every .NET plugin follows the dotnet/<name>/ contract the shared build files assume."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import check_conventions
import pytest
from check_conventions import Checker
from conftest import REPO_ROOT, make_dotnet_plugin

PLUGIN = "dotnet/fakedotnet"
LIB = "src/Temporalio.Extensions.Fakedotnet/Temporalio.Extensions.Fakedotnet.csproj"
TESTS = "tests/Temporalio.Extensions.Fakedotnet.Tests/Temporalio.Extensions.Fakedotnet.Tests.csproj"


def edit(repo: Path, rel: str, old: str, new: str) -> None:
    path = repo / PLUGIN / rel
    text = path.read_text()
    assert old in text, (rel, old)
    path.write_text(text.replace(old, new))


def write_root_props(repo: Path, body: str) -> None:
    (repo / PLUGIN / "Directory.Build.props").write_text(
        '<Project>\n  <Import Project="$(MSBuildThisFileDirectory)../_shared/dotnet.props" />\n'
        f"  <PropertyGroup>\n    {body}\n  </PropertyGroup>\n</Project>\n"
    )


def write_extra_project(repo: Path, body: str) -> None:
    project = repo / PLUGIN / "src/Temporalio.Extensions.Fakedotnet.Extra/Temporalio.Extensions.Fakedotnet.Extra.csproj"
    project.parent.mkdir(parents=True, exist_ok=True)
    project.write_text(
        '<Project Sdk="Microsoft.NET.Sdk">\n  <PropertyGroup>\n    <TargetFramework>net10.0</TargetFramework>\n'
        f"    {body}\n  </PropertyGroup>\n  <ItemGroup>\n"
        '    <PackageReference Include="StyleCop.Analyzers" Version="1.3.180" PrivateAssets="all" />\n'
        "  </ItemGroup>\n</Project>\n"
    )


def test_fake_dotnet_plugin_passes(dotnet_repo: Path) -> None:
    assert Checker(dotnet_repo).run() == []


def test_shared_files_without_plugins_pass(repo: Path) -> None:
    shutil.rmtree(make_dotnet_plugin(repo))
    assert Checker(repo).run() == []


CASES: dict[str, tuple[Callable[[Path], None], str]] = {
    "coordinate-letters": (
        lambda r: edit(r, "plugin.toml", 'coordinate = "Temporalio.Extensions.Fakedotnet"', 'coordinate = "Temporalio.Other"'),
        "coordinate must be 'Temporalio.Extensions.fakedotnet' up to casing"),
    "root-api": (
        lambda r: edit(r, "plugin.toml", 'root-api = "Temporalio.Extensions.Fakedotnet"', 'root-api = "Temporalio.Extensions.Fakedotnet.Core"'),
        "root-api must be 'Temporalio.Extensions.Fakedotnet'"),
    "registry": (lambda r: edit(r, "plugin.toml", 'registry = "nuget"', 'registry = "pypi"'), "registry must be 'nuget'"),
    "runtime-versions": (lambda r: edit(r, "plugin.toml", 'runtime-versions = ["10.0"]', 'runtime-versions = ["10"]'),
                         "major.minor"),
    "smoke-imports": (lambda r: edit(r, "plugin.toml", "Temporalio.Extensions.Fakedotnet.FakedotnetPlugin", "Other.Plugin"),
                      "smoke imports must name types under Temporalio.Extensions.Fakedotnet"),
    "package-id": (lambda r: edit(r, LIB, "<PackageId>Temporalio.Extensions.Fakedotnet</PackageId>", "<PackageId>Temporalio.Other</PackageId>"),
                   "PackageId must be 'Temporalio.Extensions.Fakedotnet'"),
    "version": (lambda r: edit(r, LIB, "<Version>0.0.0</Version>", "<Version>0.1.0</Version>"), "development placeholder"),
    "license-file": (
        lambda r: edit(r, LIB, "<PackageLicenseExpression>MIT</PackageLicenseExpression>",
                       "<PackageLicenseExpression>MIT</PackageLicenseExpression><PackageLicenseFile>LICENSE</PackageLicenseFile>"),
        "no PackageLicenseFile"),
    "license-other": (
        lambda r: edit(r, LIB, "<PackageLicenseExpression>MIT</PackageLicenseExpression>", "<PackageLicenseExpression>Apache-2.0</PackageLicenseExpression>"),
        "PackageLicenseExpression must be MIT"),
    "license-packed": (
        lambda r: edit(r, LIB, '<None Include="../../README.md" Pack="true" PackagePath="" />',
                       '<None Include="../../README.md" Pack="true" PackagePath="" /><None Include="../../LICENSE" Pack="true" PackagePath="" />'),
        "do not pack a LICENSE file"),
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
        lambda r: edit(r, TESTS, "../../src/Temporalio.Extensions.Fakedotnet/Temporalio.Extensions.Fakedotnet.csproj", "../../../other/src/Other/Other.csproj"),
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
                     "must be the unconditioned shared import"),
    "nuget-config-cooldown": (lambda r: edit(r, "nuget.config", ' minPublishAgeHours="336"', ""), "must carry minPublishAgeHours"),
    "nuget-config-exception": (lambda r: edit(r, "nuget.config", '    <package pattern="Temporalio.*" />\n', ""),
                               "must exempt exactly Temporalio and Temporalio.*"),
    "nuget-config-exception-prefix": (
        lambda r: edit(r, "nuget.config", '    <package pattern="Temporalio" />\n    <package pattern="Temporalio.*" />\n',
                       '    <package pattern="Temporalio*" />\n'),
        "must exempt exactly Temporalio and Temporalio.*"),
    "nuget-config-extra-source": (
        lambda r: edit(r, "nuget.config", "  </packageSources>", '    <add key="mirror" value="https://mirror.example/v3/index.json" />\n  </packageSources>'),
        "must be the only package source"),
    "nuget-config-clear": (lambda r: edit(r, "nuget.config", "    <clear />\n", ""), "must start with <clear />"),
    "makefile-package": (lambda r: edit(r, "Makefile", "PACKAGE := Temporalio.Extensions.Fakedotnet", "PACKAGE := Temporalio.Other"),
                         "must set PACKAGE := Temporalio.Extensions.Fakedotnet"),
    "makefile-include": (lambda r: edit(r, "Makefile", "include ../_shared/dotnet.mk", "include ../_shared/other.mk"),
                         "must include ../_shared/dotnet.mk"),
    "readme-relative-link": (lambda r: (r / PLUGIN / "README.md").write_text("see [root](../../README.md)\n"), "relative link"),
    "license-drift": (lambda r: (r / PLUGIN / "LICENSE").write_text("MIT\n"), "LICENSE differs from root"),
    "changelog": (lambda r: (r / PLUGIN / "CHANGELOG.md").write_text("# changes\n"), "no changelog files"),
    "second-slnx": (lambda r: (r / PLUGIN / "Other.slnx").write_text("<Solution />\n"), "exactly one Temporalio.Extensions.Fakedotnet.slnx"),
    "slnx-missing-project": (
        lambda r: edit(r, "Temporalio.Extensions.Fakedotnet.slnx", '  <Project Path="tests/Temporalio.Extensions.Fakedotnet.Tests/Temporalio.Extensions.Fakedotnet.Tests.csproj" />\n', ""),
        "must list src/Temporalio.Extensions.Fakedotnet/Temporalio.Extensions.Fakedotnet.csproj and tests/"),
    "lock-missing": (lambda r: (r / PLUGIN / "src/Temporalio.Extensions.Fakedotnet/packages.lock.json").unlink(),
                     "missing src/Temporalio.Extensions.Fakedotnet/packages.lock.json"),
    "two-runtimes": (
        lambda r: (edit(r, "plugin.toml", 'runtime-versions = ["10.0"]', 'runtime-versions = ["8.0", "10.0"]'),
                   edit(r, LIB, "<TargetFramework>net10.0</TargetFramework>", "<TargetFramework>netstandard2.0</TargetFramework>")),
        "test project must target net8.0;net10.0"),
    "library-newer-than-oldest-runtime": (
        lambda r: (edit(r, "plugin.toml", 'runtime-versions = ["10.0"]', 'runtime-versions = ["8.0", "10.0"]'),
                   edit(r, TESTS, "<TargetFramework>net10.0</TargetFramework>", "<TargetFrameworks>net8.0;net10.0</TargetFrameworks>")),
        "TargetFramework must be netstandard2.0 or net8.0"),
    "root-props-weaker-policy": (lambda r: write_root_props(r, "<TreatWarningsAsErrors>false</TreatWarningsAsErrors>"),
                                 "Directory.Build.props: TreatWarningsAsErrors must stay true"),
    "root-props-analyzers-off": (lambda r: write_root_props(r, "<EnableNETAnalyzers>false</EnableNETAnalyzers>"),
                                 "Directory.Build.props: EnableNETAnalyzers must stay true"),
    "root-props-version": (lambda r: write_root_props(r, "<Version>0.1.0</Version>"), "Directory.Build.props: Version must be"),
    "root-props-aot": (lambda r: write_root_props(r, "<PublishAot>true</PublishAot>"), "Directory.Build.props: PublishAot is not allowed"),
    "root-props-license-expression": (lambda r: write_root_props(r, "<PackageLicenseExpression>MIT</PackageLicenseExpression>"),
                                      "Directory.Build.props: the license belongs in the library csproj"),
    "root-props-packable": (lambda r: write_root_props(r, "<IsPackable>true</IsPackable>"), "would pack the test project too"),
    "nested-props": (lambda r: (r / PLUGIN / "src/Directory.Build.props").write_text("<Project />\n"),
                     "src/Directory.Build.props: a nearer Directory.Build.props replaces the plugin-root one"),
    "nested-props-in-project": (lambda r: (r / PLUGIN / "tests/Temporalio.Extensions.Fakedotnet.Tests/Directory.Build.props").write_text("<Project />\n"),
                                "tests/Temporalio.Extensions.Fakedotnet.Tests/Directory.Build.props: a nearer Directory.Build.props"),
    "targets": (lambda r: (r / PLUGIN / "Directory.Build.targets").write_text("<Project />\n"),
                "Directory.Build.targets: Directory.Build.targets is evaluated after the csproj"),
    "nested-targets": (lambda r: (r / PLUGIN / "src/Temporalio.Extensions.Fakedotnet/Directory.Build.targets").write_text("<Project />\n"),
                       "src/Temporalio.Extensions.Fakedotnet/Directory.Build.targets: Directory.Build.targets is evaluated after"),
    "packages-props": (lambda r: (r / PLUGIN / "Directory.Packages.props").write_text("<Project />\n"),
                       "Directory.Packages.props: Directory.Packages.props moves package versions out of the csproj"),
    "target-framework-floor": (lambda r: edit(r, LIB, "<TargetFramework>net10.0</TargetFramework>", "<TargetFramework>net9.0</TargetFramework>"),
                               "TargetFramework(s) must include net10.0"),
    "project-url": (lambda r: edit(r, LIB, "tree/main/dotnet/fakedotnet</PackageProjectUrl>", "tree/main/dotnet/other</PackageProjectUrl>"),
                    "PackageProjectUrl must be https://github.com/temporalio/ai-integrations/tree/main/dotnet/fakedotnet"),
    "library-project-reference": (
        lambda r: edit(r, LIB, '  <ItemGroup>\n    <PackageReference Include="Temporalio"',
                       '  <ItemGroup>\n    <ProjectReference Include="../Temporalio.Extensions.Fakedotnet.Extra/Temporalio.Extensions.Fakedotnet.Extra.csproj" />\n'
                       '    <PackageReference Include="Temporalio"'),
        "ProjectReference in the packed library"),
    "extra-project-weaker-policy": (lambda r: write_extra_project(r, "<TreatWarningsAsErrors>false</TreatWarningsAsErrors>"),
                                    "Temporalio.Extensions.Fakedotnet.Extra.csproj: TreatWarningsAsErrors must stay true"),
    "extra-project-packable": (lambda r: write_extra_project(r, "<IsPackable>true</IsPackable>"), "only src/<Coordinate> is packable"),
    "root-props-analysis-level-none": (lambda r: write_root_props(r, "<AnalysisLevel>none</AnalysisLevel>"),
                                       "Directory.Build.props: AnalysisLevel must stay 10.0"),
    "code-analysis-warnings": (
        lambda r: edit(r, LIB, "<IsPackable>true</IsPackable>",
                       "<IsPackable>true</IsPackable><CodeAnalysisTreatWarningsAsErrors>false</CodeAnalysisTreatWarningsAsErrors>"),
        "CodeAnalysisTreatWarningsAsErrors must stay true"),
    "msbuild-warnings": (
        lambda r: edit(r, TESTS, "<OutputType>Exe</OutputType>",
                       "<OutputType>Exe</OutputType><MSBuildTreatWarningsAsErrors>false</MSBuildTreatWarningsAsErrors>"),
        "MSBuildTreatWarningsAsErrors must stay true"),
    "conditioned-import": (
        lambda r: (r / PLUGIN / "Directory.Build.props").write_text(
            '<Project>\n  <Import Project="$(MSBuildThisFileDirectory)../_shared/dotnet.props" Condition="\'$(CI)\' == \'true\'" />\n</Project>\n'),
        "must be the unconditioned shared import"),
    # The bypasses MSBuild would honor and a text-level check would miss.
    "props-xmlns": (
        lambda r: (r / PLUGIN / "Directory.Build.props").write_text(
            '<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">\n'
            '  <Import Project="$(MSBuildThisFileDirectory)../_shared/dotnet.props" />\n'
            "  <PropertyGroup>\n    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>\n  </PropertyGroup>\n</Project>\n"),
        "Directory.Build.props: TreatWarningsAsErrors must stay true"),
    "conditioned-second-assignment": (
        lambda r: edit(r, LIB, "<IsPackable>true</IsPackable>",
                       '<IsPackable>true</IsPackable><TreatWarningsAsErrors>true</TreatWarningsAsErrors>'
                       '<TreatWarningsAsErrors Condition="false">false</TreatWarningsAsErrors>'),
        "TreatWarningsAsErrors must stay true"),
    "csproj-import": (
        lambda r: edit(r, LIB, '<Project Sdk="Microsoft.NET.Sdk">\n', '<Project Sdk="Microsoft.NET.Sdk">\n  <Import Project="../extra.props" />\n'),
        "Import of '../extra.props' adds build settings outside dotnet/_shared"),
    "props-second-import": (
        lambda r: (r / PLUGIN / "Directory.Build.props").write_text(
            '<Project>\n  <Import Project="$(MSBuildThisFileDirectory)../_shared/dotnet.props" />\n  <Import Project="extra.props" />\n</Project>\n'),
        "Directory.Build.props: Import of 'extra.props' adds build settings outside dotnet/_shared"),
    "run-analyzers-off": (lambda r: edit(r, LIB, "<IsPackable>true</IsPackable>", "<IsPackable>true</IsPackable><RunAnalyzers>false</RunAnalyzers>"),
                          "RunAnalyzers must stay true"),
    "package-validation-off": (lambda r: edit(r, LIB, "<IsPackable>true</IsPackable>", "<IsPackable>true</IsPackable><EnablePackageValidation>false</EnablePackageValidation>"),
                               "EnablePackageValidation must stay true"),
    "nuget-audit-off": (lambda r: edit(r, LIB, "<IsPackable>true</IsPackable>", "<IsPackable>true</IsPackable><NuGetAudit>false</NuGetAudit>"),
                        "NuGetAudit must stay true"),
    "restore-locked-mode-off": (
        lambda r: edit(r, TESTS, "<OutputType>Exe</OutputType>", "<OutputType>Exe</OutputType><RestoreLockedMode>false</RestoreLockedMode>"),
        "RestoreLockedMode must stay true"),
    "globalconfig-discovery-on": (
        lambda r: write_root_props(r, "<DiscoverGlobalAnalyzerConfigFiles>true</DiscoverGlobalAnalyzerConfigFiles>"),
        "DiscoverGlobalAnalyzerConfigFiles must stay false"),
    "globalconfig-in-plugin": (lambda r: (r / PLUGIN / ".globalconfig").write_text("is_global = true\n"),
                               ".globalconfig: a .globalconfig would change analyzer severities"),
    "globalconfig-in-src": (lambda r: (r / PLUGIN / "src/Temporalio.Extensions.Fakedotnet/.globalconfig").write_text("is_global = true\n"),
                            "src/Temporalio.Extensions.Fakedotnet/.globalconfig: a .globalconfig would change analyzer severities"),
    "build-rsp": (lambda r: (r / PLUGIN / "Directory.Build.rsp").write_text("-p:TreatWarningsAsErrors=false\n"),
                  "Directory.Build.rsp: Directory.Build.rsp adds MSBuild switches"),
    "nested-editorconfig": (lambda r: (r / PLUGIN / "src/.editorconfig").write_text("root = true\n"),
                            "src/.editorconfig: a nested .editorconfig overrides"),
    "configure-await-false": (
        lambda r: (r / PLUGIN / "src/Temporalio.Extensions.Fakedotnet/Awaits.cs").write_text(
            "namespace Temporalio.Extensions.Fakedotnet;\n\npublic static class Awaits\n{\n    public static async Task RunAsync() => await Task.Yield().ConfigureAwait(false);\n}\n"),
        "src/Temporalio.Extensions.Fakedotnet/Awaits.cs:5: ConfigureAwait(false) leaves the Workflow scheduler"),
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
    meta.write_text(meta.read_text().replace("Temporalio.Extensions.Fakedotnet", "temporalio.Extensions.Fakedotnet"))
    violations = Checker(dotnet_repo).run()
    assert violations[0] == (f"{PLUGIN}: plugin.toml coordinate 'temporalio.Extensions.Fakedotnet' must keep the Temporalio.Extensions. "
                             "prefix casing and be a valid NuGet ID")


def test_build_output_is_not_scanned_for_directory_files(dotnet_repo: Path) -> None:
    stale = dotnet_repo / PLUGIN / "src/Temporalio.Extensions.Fakedotnet/obj/Directory.Build.props"
    stale.parent.mkdir(parents=True)
    stale.write_text("<Project />\n")
    assert Checker(dotnet_repo).run() == []


def test_shared_import_is_matched_as_xml_not_text(dotnet_repo: Path) -> None:
    (dotnet_repo / PLUGIN / "Directory.Build.props").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<!-- plugin build settings -->\n<Project>\n'
        '    <Import Project="$(MSBuildThisFileDirectory)..\\_shared\\dotnet.props"/>\n'
        "    <PropertyGroup>\n        <Description>Fake plugin (Pre-release)</Description>\n    </PropertyGroup>\n</Project>\n"
    )
    assert Checker(dotnet_repo).run() == []


def test_makefile_include_may_carry_a_comment(dotnet_repo: Path) -> None:
    (dotnet_repo / PLUGIN / "Makefile").write_text("PACKAGE := Temporalio.Extensions.Fakedotnet\ninclude ../_shared/dotnet.mk # shared targets\n")
    assert Checker(dotnet_repo).run() == []


def test_extra_projects_under_the_policy_pass(dotnet_repo: Path) -> None:
    write_extra_project(dotnet_repo, "<Description>Benchmarks</Description>")
    assert Checker(dotnet_repo).run() == []


def test_shared_props_keep_advisories_warnings_and_pin_the_analysis_level() -> None:
    props = ET.parse(REPO_ROOT / "dotnet" / "_shared" / "dotnet.props").getroot()
    properties = {prop.tag: (prop.text or "").strip() for group in props.iter("PropertyGroup") for prop in group}
    assert {"NU1900", "NU1901", "NU1902", "NU1903", "NU1904"} <= set(properties["WarningsNotAsErrors"].split(";"))
    assert properties["AnalysisLevel"] == check_conventions.DOTNET_ANALYSIS_LEVEL == "10.0"
    # The policy the check enforces on plugins is the policy the shared props declare.
    for name, value in check_conventions.DOTNET_POLICY_PROPERTIES.items():
        if name in properties:
            assert properties[name].lower() == value, name


def test_root_props_may_add_plugin_specific_properties(dotnet_repo: Path) -> None:
    write_root_props(dotnet_repo, "<Description>Fake plugin (Pre-release)</Description>")
    assert Checker(dotnet_repo).run() == []


def test_xunit_v2_is_rejected_and_xunit_v3_required(dotnet_repo: Path) -> None:
    edit(dotnet_repo, TESTS, 'Include="xunit.v3"', 'Include="xunit"')
    violations = Checker(dotnet_repo).run()
    assert any("must reference xunit.v3" in v for v in violations)
    assert any("xunit is a VSTest package" in v for v in violations)


def test_uncommitted_required_files_are_reported(dotnet_repo: Path) -> None:
    import subprocess
    subprocess.run(["git", "-C", str(dotnet_repo), "rm", "-q", "--cached", f"{PLUGIN}/tests/Temporalio.Extensions.Fakedotnet.Tests/packages.lock.json"], check=True)
    assert Checker(dotnet_repo).run() == [f"{PLUGIN}: tests/Temporalio.Extensions.Fakedotnet.Tests/packages.lock.json must be committed"]


def test_language_root_build_files_are_forbidden(dotnet_repo: Path) -> None:
    for name in ("Directory.Build.props", "global.json", "packages.lock.json", ".globalconfig", "Directory.Build.rsp", ".editorconfig"):
        (dotnet_repo / "dotnet" / name).write_text("{}\n")
    violations = Checker(dotnet_repo).run()
    assert len(violations) == 6, violations
    assert any("dotnet/Directory.Build.props: language-root build files are forbidden" in v for v in violations)
    assert any("dotnet/global.json: language-root build files are forbidden" in v for v in violations)
    assert any("dotnet/packages.lock.json: language-level lockfiles are forbidden" in v for v in violations)
    for name in (".globalconfig", "Directory.Build.rsp", ".editorconfig"):
        assert any(f"dotnet/{name}: language-root build files are forbidden" in v for v in violations), name


def test_configure_await_false_is_reported_only_under_src(dotnet_repo: Path) -> None:
    (dotnet_repo / PLUGIN / "tests/Temporalio.Extensions.Fakedotnet.Tests/Awaits.cs").write_text(
        "public static class Awaits\n{\n    public static async Task RunAsync() => await Task.Yield().ConfigureAwait(false);\n}\n")
    assert Checker(dotnet_repo).run() == []


def test_shared_props_do_not_discover_globalconfigs() -> None:
    props = ET.parse(REPO_ROOT / "dotnet" / "_shared" / "dotnet.props").getroot()
    properties = {prop.tag: (prop.text or "").strip() for group in props.iter("PropertyGroup") for prop in group}
    assert properties["DiscoverGlobalAnalyzerConfigFiles"] == "false"
    includes = [item.get("Include", "") for group in props.iter("ItemGroup") for item in group if item.tag == "EditorConfigFiles"]
    assert len(includes) == 2 and all("$(MSBuildThisFileDirectory)analyzers" in include for include in includes)


def test_format_and_lint_exclude_the_configure_await_fixer() -> None:
    mk = (REPO_ROOT / "dotnet" / "_shared" / "dotnet.mk").read_text()
    assert mk.count("--exclude-diagnostics CA2007") == 2, "format and lint both skip CA2007"
    config = (REPO_ROOT / "dotnet" / "_shared" / "analyzers.globalconfig").read_text()
    assert "dotnet_diagnostic.CA2007.severity = none" in config


def test_missing_shared_files_are_reported(dotnet_repo: Path) -> None:
    (dotnet_repo / "dotnet/_shared/dotnet.props").unlink()
    (dotnet_repo / "dotnet/_template/.editorconfig").unlink()
    violations = Checker(dotnet_repo).run()
    assert "dotnet/_shared/dotnet.props: missing shared build conventions" in violations
    assert "dotnet/_template/.editorconfig: missing canonical editorconfig" in violations


def test_coordinate_casing_in_plugin_toml_is_canonical(repo: Path) -> None:
    from conftest import commit_all
    make_dotnet_plugin(repo, "microsoft-agent-framework", "Temporalio.Extensions.MicrosoftAgentFramework")
    commit_all(repo, "add agent framework")
    assert Checker(repo).run() == []
