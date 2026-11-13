#!/usr/bin/env python3
"""Repository-invariant checks for temporalio/ai-integrations.

Runs on every PR (job `conventions`) and nightly. Exits non-zero
with a list of violations. Passes on a repository with zero plugins.

Checks (see AGENTS.md, "Repository invariants" and "Python conventions"):
  * plugin folder names never end in `-plugin` / `_plugin`
  * no language-level lockfiles (python/uv.lock, typescript/pnpm-lock.yaml, ...)
  * python/_shared contains configuration and make logic, never Python source or stubs
  * every Python plugin has pyproject.toml, uv.lock, plugin.toml, Makefile, README.md,
    and {root-api}/{__init__.py,py.typed} under src/
  * every Java plugin loads the shared plugin repositories from its initial pluginManagement block
  * NO src/temporalio/__init__.py and NO src/temporalio/contrib/__init__.py (namespace invariant)
  * LICENSE is a committed regular file byte-identical to the root LICENSE; pyproject declares license = "MIT"
    and license-files = ["LICENSE"]; no CHANGELOG*, no smoke_test.py in the plugin dir
  * plugin.toml schema and agreement with pyproject.toml (name/coordinate/root-api/
    maturity classifier/requires-python floor/module-name/required-version)
  * no [tool.uv.sources] path or workspace entries
  * the project's published README has no relative markdown links (PyPI renders it)
  * every .NET plugin follows the dotnet/<name>/ contract (AGENTS.md, ".NET conventions"): a
    Directory.Build.props that starts by importing dotnet/_shared/dotnet.props and whose body keeps
    the shared policy, no other MSBuild directory file anywhere in the plugin, one <Coordinate>.slnx,
    one SDK channel, global.json and nuget.config policy, the canonical .editorconfig, the library and
    MTP test csproj rules, a committed packages.lock.json beside each csproj
  * no Directory.Build.props, Directory.Build.targets, Directory.Packages.props, global.json or
    nuget.config at dotnet/ itself (MSBuild would apply them to every plugin silently)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    LANGUAGES,
    Plugin,
    discover_plugins,
    load_toml,
    repo_root,
)

MATURITY_CLASSIFIER = {
    "generally-available": "Development Status :: 5 - Production/Stable",
    "public-preview": "Development Status :: 4 - Beta",
    "pre-release": "Development Status :: 3 - Alpha",
}
REGISTRIES = {"python": "pypi", "typescript": "npm", "java": "maven", "go": "goproxy", "dotnet": "nuget"}
LANGUAGE_LOCKFILES = ("uv.lock", "pnpm-lock.yaml", "package-lock.json", "yarn.lock", "go.sum", "gradle.lockfile", "packages.lock.json")
# MSBuild imports the nearest of these above a project, so a copy at the language root would apply to every plugin.
LANGUAGE_ROOT_FORBIDDEN = {
    "dotnet": ("Directory.Build.props", "Directory.Build.targets", "Directory.Packages.props", "global.json", "nuget.config"),
}
RELATIVE_LINK = re.compile(r"\]\((\.\.?/)")
STANDARD_TEST_SUPPORT = {
    "tests/helpers/environment.py": "tests/helpers/environment.py.tmpl",
    "tests/helpers/plugin_meta.py": "tests/helpers/plugin_meta.py.tmpl",
    "tests/helpers/provenance.py": "tests/helpers/provenance.py.tmpl",
    "tests/test_env.py": "tests/test_env.py.tmpl",
    "tests/test_installed_matches_source.py": "tests/test_installed_matches_source.py.tmpl",
}
STANDARD_TYPESCRIPT_TEST_SUPPORT = {
    "src/__tests__/helpers/environment.ts": "src/__tests__/helpers/environment.ts.tmpl",
    "src/__tests__/test-environment-startup.test.ts": "src/__tests__/test-environment-startup.test.ts.tmpl",
}
PYTHON_DEVELOPMENT_VERSION = "0.0.0"
JAVA_PLUGIN_MANAGEMENT = """pluginManagement {
    apply from: '../_shared/settings.gradle'
}
"""
DOTNET_DEVELOPMENT_VERSION = "0.0.0"
DOTNET_PREFIX = "Temporalio."
DOTNET_SHARED_IMPORT_PATH = "$(MSBuildThisFileDirectory)../_shared/dotnet.props"
DOTNET_SHARED_IMPORT = f'<Project>\n  <Import Project="{DOTNET_SHARED_IMPORT_PATH}" />\n'
# Keep equal to dotnet/_shared/dotnet.props and the plugins' global.json floor channel.
DOTNET_ANALYSIS_LEVEL = "10.0"
DOTNET_REPOSITORY_TREE = "https://github.com/temporalio/ai-integrations/tree/main"
# MSBuild composes every project from these files along the directory chain: a nearer Directory.Build.props
# replaces the plugin-root one (and its shared import), Directory.Build.targets is evaluated after the csproj
# and overrides it, Directory.Packages.props moves versions out of the csproj files. The plugin-root
# Directory.Build.props is the only one allowed.
DOTNET_DIRECTORY_FILES = {
    "Directory.Build.props": "a nearer Directory.Build.props replaces the plugin-root one and drops the shared policy",
    "Directory.Build.targets": "Directory.Build.targets is evaluated after the csproj and overrides the shared policy",
    "Directory.Packages.props": "Directory.Packages.props moves package versions out of the csproj files",
}
DOTNET_BUILD_OUTPUT = {"bin", "obj"}
NUGET_ID = re.compile(r"\w+([_.-]\w+)*")
NUGET_ORG = "https://api.nuget.org/v3/index.json"
# Microsoft.Testing.Platform mode rejects VSTest projects; `xunit` is the v2 package.
DOTNET_VSTEST_PACKAGES = ("Microsoft.NET.Test.Sdk", "xunit.runner.visualstudio", "xunit", "coverlet.collector")
# These add SDK-version-pinned implicit PackageReferences, which break locked restores across SDK patches.
DOTNET_FORBIDDEN_PROPERTIES = ("IsAotCompatible", "PublishAot", "PublishTrimmed", "IsTrimmable")
# Shared policy from dotnet/_shared/dotnet.props that a csproj may not weaken.
DOTNET_POLICY_PROPERTIES = {
    "TreatWarningsAsErrors": "true", "RestorePackagesWithLockFile": "true", "Nullable": "enable",
    "GenerateDocumentationFile": "true", "IncludeSymbols": "true", "PublishRepositoryUrl": "true",
    "EnableNETAnalyzers": "true", "AnalysisMode": "allenabledbydefault", "AnalysisLevel": DOTNET_ANALYSIS_LEVEL,
    "CodeAnalysisTreatWarningsAsErrors": "true", "MSBuildTreatWarningsAsErrors": "true",
}


class Checker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.violations: list[str] = []

    def fail(self, msg: str) -> None:
        self.violations.append(msg)

    # -- helpers -----------------------------------------------------------------
    def tracked_files(self, rel_dir: str) -> set[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(self.root), "ls-files", "--", rel_dir],
                check=True, capture_output=True, text=True,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return set()
        return {line.strip() for line in out.splitlines() if line.strip()}

    def scripts_required_version(self) -> str | None:
        path = self.root / "scripts" / "pyproject.toml"
        if not path.is_file():
            return None
        return load_toml(path).get("tool", {}).get("uv", {}).get("required-version")

    # -- checks ------------------------------------------------------------------
    def check_language_roots(self, discovered: dict[str, list[Plugin]]) -> None:
        for language in LANGUAGES:
            lang_dir = self.root / language
            if not lang_dir.is_dir():
                continue
            for lock in LANGUAGE_LOCKFILES:
                if (lang_dir / lock).exists():
                    self.fail(f"{language}/{lock}: language-level lockfiles are forbidden; each plugin owns its own")
            for forbidden in LANGUAGE_ROOT_FORBIDDEN.get(language, ()):
                if (lang_dir / forbidden).exists():
                    self.fail(f"{language}/{forbidden}: language-root build files are forbidden; each plugin owns its own "
                              f"and imports {language}/_shared explicitly")
            for child in lang_dir.iterdir():
                if child.is_dir() and not child.name.startswith(("_", ".")):
                    if child.name.endswith(("-plugin", "_plugin")):
                        self.fail(f"{language}/{child.name}: plugin folders must not end in -plugin/_plugin")
                    if child not in [p.path for p in discovered[language]]:
                        self.fail(f"{language}/{child.name}: directory has no {language} manifest; non-plugin folders must start with `_`")
        shared_python = self.root / "python" / "_shared"
        if shared_python.is_dir():
            for tracked in sorted(self.tracked_files("python/_shared")):
                path = self.root / tracked
                if path.suffix in {".py", ".pyi", ".pyw", ".pyx"}:
                    self.fail(
                        f"{path.relative_to(self.root)}: Python code must be duplicated into each plugin, not shared"
                    )

    def check_python_plugin(self, plugin: Plugin) -> None:
        d = plugin.path
        rel = plugin.rel
        name = plugin.name
        for required in ("pyproject.toml", "uv.lock", "plugin.toml", "Makefile", "README.md"):
            if not (d / required).is_file():
                self.fail(f"{rel}: missing {required}")

        pyproject: dict[str, Any] | None = None
        meta: dict[str, Any] | None = None
        if (d / "pyproject.toml").is_file() and (d / "plugin.toml").is_file():
            try:
                pyproject = load_toml(d / "pyproject.toml")
                meta = load_toml(d / "plugin.toml")
            except Exception as exc:  # noqa: BLE001
                self.fail(f"{rel}: cannot parse TOML: {exc}")

        root_api = (
            meta.get("plugin", {}).get("root-api")
            if isinstance(meta, dict) and isinstance(meta.get("plugin"), dict)
            else None
        )
        package_parts = (
            root_api.split(".")
            if isinstance(root_api, str)
            and root_api == f"temporalio.{name}"
            else ["temporalio", name]
        )
        package_rel = Path("src", *package_parts)
        pkg = d / package_rel
        if not (pkg / "__init__.py").is_file():
            self.fail(f"{rel}: missing {package_rel}/__init__.py")
        if not (pkg / "py.typed").is_file():
            self.fail(f"{rel}: missing {package_rel}/py.typed")
        for forbidden in ("src/temporalio/__init__.py", "src/temporalio/contrib/__init__.py"):
            if (d / forbidden).exists():
                self.fail(f"{rel}: {forbidden} must not exist (namespace invariant; the SDK owns these packages)")

        tracked = {p.split("/", 2)[-1] for p in self.tracked_files(rel)}
        # LICENSE: every plugin ships the license text in its wheel and sdist, so each plugin directory
        # carries a committed copy that must stay byte-identical to the root LICENSE (`cp LICENSE python/<name>/`).
        license_path = d / "LICENSE"
        if not license_path.exists() or not stat.S_ISREG(license_path.lstat().st_mode):
            self.fail(f"{rel}: LICENSE must be a regular file")
        elif "LICENSE" not in tracked:
            self.fail(f"{rel}: LICENSE must be committed (copy the root LICENSE: `cp LICENSE {rel}/LICENSE`)")
        elif license_path.read_bytes() != (self.root / "LICENSE").read_bytes():
            self.fail(f"{rel}: LICENSE differs from the root LICENSE; re-copy it")
        for f in sorted(tracked):
            base = f.rsplit("/", 1)[-1]
            if "/" not in f and base.upper().startswith("CHANGELOG"):
                self.fail(f"{rel}: {f} is committed; release notes are generated from history, no changelog files")
            if base == "smoke_test.py":
                self.fail(f"{rel}: {f} is committed; the generic scripts/ci/smoke.py replaces per-plugin smoke scripts")

        if pyproject is None or meta is None:
            return
        self.check_plugin_toml(plugin, meta, pyproject)
        self.check_pyproject(plugin, pyproject)
        self.check_readme(plugin, pyproject)
        self.check_standard_test_support(plugin)

    def check_standard_test_support(self, plugin: Plugin) -> None:
        template_rel_root = f"{plugin.language}/_template"
        template_root = self.root / template_rel_root
        if not template_root.is_dir():
            return
        support_files = {
            "python": STANDARD_TEST_SUPPORT,
            "typescript": STANDARD_TYPESCRIPT_TEST_SUPPORT,
        }[plugin.language]
        for plugin_rel, template_rel in support_files.items():
            plugin_path = plugin.path / plugin_rel
            template_path = template_root / template_rel
            if not plugin_path.is_file():
                self.fail(f"{plugin.rel}: missing standard test support file {plugin_rel}")
            elif not template_path.is_file():
                self.fail(f"{template_rel_root}/{template_rel}: missing canonical test support template")
            elif plugin_path.read_bytes() != template_path.read_bytes():
                self.fail(
                    f"{plugin.rel}/{plugin_rel}: differs from canonical {template_rel_root}/{template_rel}"
                )

    def check_java_plugin(self, plugin: Plugin) -> None:
        d, rel = plugin.path, plugin.rel
        for required in ("build.gradle", "settings.gradle", "plugin.toml", "README.md", "LICENSE",
                         "gradlew", "gradlew.bat", "gradle/wrapper/gradle-wrapper.jar",
                         "gradle/wrapper/gradle-wrapper.properties"):
            if not (d / required).is_file():
                self.fail(f"{rel}: missing {required}")
        settings = d / "settings.gradle"
        if settings.is_file() and not settings.read_text().lstrip().startswith(JAVA_PLUGIN_MANAGEMENT):
            self.fail(
                f"{rel}/settings.gradle: must start with the shared pluginManagement block:\n"
                f"{JAVA_PLUGIN_MANAGEMENT.rstrip()}"
            )
        if not (self.root / "java/_shared/settings.gradle").is_file():
            self.fail("java/_shared/settings.gradle: missing shared plugin repository configuration")
        if not (d / "plugin.toml").is_file():
            return
        meta = load_toml(d / "plugin.toml")
        p = meta.get("plugin", {})
        for key, expected in {"name": plugin.name, "language": "java", "registry": "maven",
                              "coordinate": f"io.temporal:{plugin.name}"}.items():
            if p.get(key) != expected:
                self.fail(f"{rel}: plugin.toml {key} must be {expected!r}")
        root_api = p.get("root-api", "")
        if not isinstance(root_api, str) or not root_api.startswith("io.temporal."):
            self.fail(f"{rel}: root-api must be under io.temporal")
        if p.get("maturity") not in MATURITY_CLASSIFIER:
            self.fail(f"{rel}: invalid maturity")
        ci = meta.get("ci", {})
        runtimes = ci.get("runtime-versions", [])
        if not runtimes or not all(isinstance(v, str) and v.isdigit() for v in runtimes):
            self.fail(f"{rel}: runtime-versions must contain Java major versions")
        for table in (p, ci):
            for banned in ("owners", "secrets", "live-secrets"):
                if banned in table:
                    self.fail(f"{rel}: plugin.toml must not contain {banned!r}")
        for boot in ci.get("spring-boot-versions", []):
            if not re.fullmatch(r"\d+\.\d+\.\d+", str(boot)):
                self.fail(f"{rel}: invalid Spring Boot version {boot!r}")
            elif not (d / f"gradle/dependency-locks/spring-boot-{boot}.lockfile").is_file():
                self.fail(f"{rel}: missing dependency lock for Spring Boot {boot}")
        if not ci.get("spring-boot-versions") and not (
            d / "gradle/dependency-locks/dependencies.lockfile"
        ).is_file():
            self.fail(f"{rel}: missing gradle/dependency-locks/dependencies.lockfile")
        imports = meta.get("smoke", {}).get("imports", [])
        if not imports or not all(isinstance(i, str) and i.startswith(root_api + ".") for i in imports):
            self.fail(f"{rel}: smoke imports must name classes under root-api")
        license_path = d / "LICENSE"
        if not license_path.is_file() or license_path.is_symlink():
            self.fail(f"{rel}: LICENSE must be a regular file")
        elif f"{rel}/LICENSE" not in self.tracked_files(rel):
            self.fail(f"{rel}: LICENSE must be committed")
        elif license_path.read_bytes() != (self.root / "LICENSE").read_bytes():
            self.fail(f"{rel}: LICENSE differs from root")
        wrapper = d / "gradle/wrapper/gradle-wrapper.properties"
        if wrapper.is_file() and not re.search(r"^distributionSha256Sum=[a-f0-9]{64}$", wrapper.read_text(), re.M):
            self.fail(f"{rel}: Gradle distribution checksum must be pinned")
        if any(path.name.upper().startswith("CHANGELOG") for path in d.iterdir()):
            self.fail(f"{rel}: release notes are generated; no changelog files")

    def check_typescript_plugin(self, plugin: Plugin) -> None:
        d, rel = plugin.path, plugin.rel
        for required in ("package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "plugin.toml",
                         "tsconfig.json", "Makefile", "README.md", "LICENSE"):
            if not (d / required).is_file():
                self.fail(f"{rel}: missing {required}")
        if not (d / "plugin.toml").is_file():
            return
        package = json.loads((d / "package.json").read_text())
        meta = load_toml(d / "plugin.toml")
        coordinate = f"@temporalio/{plugin.name}"
        p = meta.get("plugin", {})
        for key, expected in {"name": plugin.name, "language": "typescript", "registry": "npm",
                              "coordinate": coordinate, "root-api": coordinate}.items():
            if p.get(key) != expected:
                self.fail(f"{rel}: plugin.toml {key} must be {expected!r}")
        if package.get("name") != coordinate or package.get("version") != "0.0.0":
            self.fail(f"{rel}: package.json must name {coordinate} with development version 0.0.0")
        if p.get("maturity") not in MATURITY_CLASSIFIER:
            self.fail(f"{rel}: invalid maturity")
        versions = meta.get("ci", {}).get("runtime-versions", [])
        floor = package.get("engines", {}).get("node", "").removeprefix(">=")
        # Majors select the newest release of each line. The first entry may name the
        # floor's major.minor instead, so the minimum cell runs the floor itself when a
        # plugin's dependencies require more than the repository minimum.
        floor_parts = floor.split(".")
        if (not versions or not all(isinstance(version, str) and version.isdigit() for version in versions[1:])
                or not isinstance(versions[0], str)
                or versions[0] not in (floor_parts[0], ".".join(floor_parts[:2]))):
            self.fail(f"{rel}: runtime-versions must list Node majors starting with the Node engine floor's major "
                      "(or its major.minor)")
        if meta.get("smoke", {}).get("imports") != [coordinate]:
            self.fail(f"{rel}: smoke imports must name the root package")
        for section in ("dependencies", "peerDependencies", "optionalDependencies", "devDependencies"):
            for name, version in package.get(section, {}).items():
                if name == "@temporalio/test-helpers" and section == "devDependencies" and version == "link:tests/helpers":
                    continue
                if version.startswith(("workspace:", "link:", "file:")):
                    self.fail(f"{rel}: {section} {name} must use a published registry coordinate")
        license_path = d / "LICENSE"
        if not license_path.is_file() or license_path.is_symlink():
            self.fail(f"{rel}: LICENSE must be a regular file")
        elif f"{rel}/LICENSE" not in self.tracked_files(rel):
            self.fail(f"{rel}: LICENSE must be committed")
        elif license_path.read_bytes() != (self.root / "LICENSE").read_bytes():
            self.fail(f"{rel}: LICENSE differs from root")
        if any(path.name.upper().startswith("CHANGELOG") for path in d.iterdir()):
            self.fail(f"{rel}: release notes are generated; no changelog files")
        self.check_standard_test_support(plugin)

    def check_plugin_toml(self, plugin: Plugin, meta: dict[str, Any], pyproject: dict[str, Any]) -> None:
        rel = plugin.rel
        p = meta.get("plugin")
        if not isinstance(p, dict):
            self.fail(f"{rel}: plugin.toml missing [plugin] table")
            return
        expected_coordinate = "temporalio-" + plugin.name.replace("_", "-")
        root_api = p.get("root-api")
        expected_root_api = "temporalio." + plugin.name
        if p.get("name") != plugin.name:
            self.fail(f"{rel}: plugin.toml name {p.get('name')!r} must equal the folder name {plugin.name!r}")
        if p.get("language") != plugin.language:
            self.fail(f"{rel}: plugin.toml language must be {plugin.language!r}")
        if p.get("coordinate") != expected_coordinate:
            self.fail(f"{rel}: plugin.toml coordinate must be {expected_coordinate!r} (got {p.get('coordinate')!r})")
        if p.get("registry") != REGISTRIES[plugin.language]:
            self.fail(f"{rel}: plugin.toml registry must be {REGISTRIES[plugin.language]!r}")
        if root_api != expected_root_api:
            self.fail(f"{rel}: plugin.toml root-api must be {expected_root_api!r} (got {root_api!r})")
        maturity = p.get("maturity")
        if maturity not in MATURITY_CLASSIFIER:
            self.fail(f"{rel}: plugin.toml maturity must be one of {sorted(MATURITY_CLASSIFIER)}")
        for banned in ("owners", "live-secrets", "secrets"):
            if banned in p or banned in meta.get("ci", {}):
                self.fail(f"{rel}: plugin.toml must not contain {banned!r} (ownership is CODEOWNERS; CI has no secrets)")
        versions = meta.get("ci", {}).get("runtime-versions")
        if not isinstance(versions, list) or not versions:
            self.fail(f"{rel}: plugin.toml [ci] runtime-versions must be a non-empty list")
            versions = []
        smoke_imports = meta.get("smoke", {}).get("imports", [])
        if smoke_imports and not all(
            isinstance(i, str) and isinstance(root_api, str) and i.startswith(root_api)
            for i in smoke_imports
        ):
            self.fail(f"{rel}: plugin.toml [smoke] imports must be modules under {root_api}")

        project = pyproject.get("project", {})
        if project.get("name") != p.get("coordinate"):
            self.fail(f"{rel}: pyproject project.name {project.get('name')!r} must equal plugin.toml coordinate {p.get('coordinate')!r}")
        classifiers = project.get("classifiers", [])
        status_classifiers = [c for c in classifiers if c.startswith("Development Status ::")]
        expected_classifier = MATURITY_CLASSIFIER.get(maturity or "")
        if expected_classifier and status_classifiers != [expected_classifier]:
            self.fail(f"{rel}: classifiers must contain exactly {expected_classifier!r} for maturity {maturity!r} (got {status_classifiers})")
        requires = project.get("requires-python", "")
        m = re.search(r">=\s*(\d+\.\d+)", requires)
        if not m:
            self.fail(f"{rel}: requires-python must contain a >=X.Y floor (got {requires!r})")
        elif versions and str(versions[0]) != m.group(1):
            self.fail(f"{rel}: first [ci] runtime-versions entry {versions[0]!r} must equal the requires-python floor {m.group(1)!r}")
        module_name = pyproject.get("tool", {}).get("uv", {}).get("build-backend", {}).get("module-name")
        if module_name != root_api:
            self.fail(f"{rel}: [tool.uv.build-backend] module-name must be {root_api!r} (got {module_name!r})")

    def check_pyproject(self, plugin: Plugin, pyproject: dict[str, Any]) -> None:
        rel = plugin.rel
        project = pyproject.get("project", {})
        if project.get("version") != PYTHON_DEVELOPMENT_VERSION:
            self.fail(
                f"{rel}: pyproject [project] version must be the tag-authoritative development placeholder "
                f"{PYTHON_DEVELOPMENT_VERSION!r} (got {project.get('version')!r})"
            )
        if project.get("license") != "MIT":
            self.fail(f"{rel}: pyproject [project] license must be the SPDX expression \"MIT\"")
        if project.get("license-files") != ["LICENSE"]:
            self.fail(f"{rel}: pyproject [project] license-files must be [\"LICENSE\"]")
        if any(c.startswith("License ::") for c in project.get("classifiers", [])):
            self.fail(f"{rel}: drop License :: classifiers; PEP 639 uses the license expression instead")
        uv_cfg = pyproject.get("tool", {}).get("uv", {})
        for dep, source in uv_cfg.get("sources", {}).items():
            entries = source if isinstance(source, list) else [source]
            for entry in entries:
                if isinstance(entry, dict) and ("path" in entry or "workspace" in entry):
                    self.fail(f"{rel}: [tool.uv.sources] {dep} uses a path/workspace source; cross-plugin deps must use published coordinates")
        expected_rv = self.scripts_required_version()
        if expected_rv and uv_cfg.get("required-version") != expected_rv:
            self.fail(f"{rel}: [tool.uv] required-version must equal scripts/pyproject.toml's {expected_rv!r} (got {uv_cfg.get('required-version')!r})")
        exclude_newer_pkg = uv_cfg.get("exclude-newer-package", {})
        if "exclude-newer" in uv_cfg and exclude_newer_pkg.get("temporalio") is not False:
            self.fail(f"{rel}: exclude-newer is set but temporalio is not exempted (`exclude-newer-package = {{ temporalio = false }}`)")

    def check_readme(self, plugin: Plugin, pyproject: dict[str, Any]) -> None:
        configured = pyproject.get("project", {}).get("readme", "README.md")
        filename = configured.get("file") if isinstance(configured, dict) else configured
        if not isinstance(filename, str) or Path(filename).is_absolute() or ".." in Path(filename).parts:
            self.fail(f"{plugin.rel}: project.readme must name a file inside the plugin directory")
            return
        self.check_readme_links(plugin, filename, "PyPI")

    def check_readme_links(self, plugin: Plugin, filename: str, renderer: str) -> None:
        readme = plugin.path / filename
        if not readme.is_file():
            self.fail(f"{plugin.rel}: published README {filename} is missing")
            return
        for lineno, line in enumerate(readme.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if RELATIVE_LINK.search(line):
                self.fail(f"{plugin.rel}/{filename}:{lineno}: relative link; use absolute https://github.com/... URLs ({renderer} renders this file)")

    # -- .NET ---------------------------------------------------------------------
    def check_dotnet_plugin(self, plugin: Plugin) -> None:
        d, rel = plugin.path, plugin.rel
        if not (self.root / "dotnet/_shared/dotnet.props").is_file():
            self.fail("dotnet/_shared/dotnet.props: missing shared build conventions")
        if not (d / "plugin.toml").is_file():
            self.fail(f"{rel}: missing plugin.toml")
            return
        meta = load_toml(d / "plugin.toml")
        p = meta.get("plugin", {})
        coordinate = p.get("coordinate")
        # NuGet IDs are case-insensitive: the folder fixes the letters, plugin.toml fixes the casing,
        # and every other rule below is phrased in terms of that coordinate.
        expected = DOTNET_PREFIX + plugin.name.replace("-", "")
        if not isinstance(coordinate, str) or coordinate.lower() != expected.lower():
            self.fail(f"{rel}: plugin.toml coordinate must be {expected!r} up to casing (got {coordinate!r})")
            return
        if not coordinate.startswith(DOTNET_PREFIX) or not NUGET_ID.fullmatch(coordinate) or len(coordinate) > 100:
            self.fail(f"{rel}: plugin.toml coordinate {coordinate!r} must keep the {DOTNET_PREFIX} prefix casing and be a valid NuGet ID")
        for key, value in {"name": plugin.name, "language": "dotnet", "registry": "nuget", "root-api": coordinate}.items():
            if p.get(key) != value:
                self.fail(f"{rel}: plugin.toml {key} must be {value!r}")
        if p.get("maturity") not in MATURITY_CLASSIFIER:
            self.fail(f"{rel}: invalid maturity")
        ci = meta.get("ci", {})
        for table in (p, ci):
            for banned in ("owners", "secrets", "live-secrets"):
                if banned in table:
                    self.fail(f"{rel}: plugin.toml must not contain {banned!r}")
        runtimes = ci.get("runtime-versions", [])
        if not runtimes or not all(isinstance(v, str) and re.fullmatch(r"\d+\.\d+", v) for v in runtimes):
            self.fail(f"{rel}: runtime-versions must list .NET SDK channels as major.minor strings")
            runtimes = []
        elif len(runtimes) != 1:
            # global.json resolves the floor channel with latestFeature in every cell, so a second channel would
            # build with the floor SDK and test nothing new; a multi-channel matrix needs the workflow to inject an
            # SDK above the floor, which waits for a plugin that needs one.
            self.fail(f"{rel}: runtime-versions must name exactly one SDK channel until the workflow can inject "
                      "an SDK above the global.json floor")
        imports = meta.get("smoke", {}).get("imports", [])
        if not imports or not all(isinstance(i, str) and i.startswith(coordinate + ".") for i in imports):
            self.fail(f"{rel}: smoke imports must name types under {coordinate}")

        library = f"src/{coordinate}/{coordinate}.csproj"
        tests = f"tests/{coordinate}.Tests/{coordinate}.Tests.csproj"
        tracked = {path.split("/", 2)[-1] for path in self.tracked_files(rel)}
        for name in ("Directory.Build.props", "global.json", "nuget.config", ".editorconfig", "Makefile", "README.md",
                     f"{coordinate}.slnx", library, f"src/{coordinate}/packages.lock.json", tests,
                     f"tests/{coordinate}.Tests/packages.lock.json"):
            if not (d / name).is_file():
                self.fail(f"{rel}: missing {name}")
            elif name not in tracked:
                self.fail(f"{rel}: {name} must be committed")
        solutions = sorted(path.name for path in d.glob("*.slnx"))
        if solutions != [f"{coordinate}.slnx"]:
            self.fail(f"{rel}: exactly one {coordinate}.slnx is allowed at the plugin root (found {solutions})")
        if (d / f"{coordinate}.slnx").is_file():
            try:
                paths = {(project.get("Path") or "").replace("\\", "/") for project in ET.parse(d / f"{coordinate}.slnx").getroot().iter("Project")}
            except ET.ParseError as exc:
                self.fail(f"{rel}/{coordinate}.slnx: invalid XML ({exc})")
            else:
                if not {library, tests} <= paths:
                    self.fail(f"{rel}/{coordinate}.slnx: must list {library} and {tests}")
        projects: list[Path] = []
        for directory, folders, files in os.walk(d):
            folders[:] = sorted(folder for folder in folders if folder not in DOTNET_BUILD_OUTPUT and not folder.startswith("."))
            for name in sorted(files):
                if name.endswith(".csproj"):
                    projects.append(Path(directory) / name)
                if name in DOTNET_DIRECTORY_FILES and not (name == "Directory.Build.props" and Path(directory) == d):
                    nested = (Path(directory) / name).relative_to(d).as_posix()
                    self.fail(f"{rel}/{nested}: {DOTNET_DIRECTORY_FILES[name]}; the plugin-root Directory.Build.props is the only "
                              "MSBuild directory file a plugin may have")
        props = d / "Directory.Build.props"
        if props.is_file():
            self.check_dotnet_build_props(plugin, props)
        self.check_dotnet_global_json(plugin, runtimes)
        self.check_dotnet_nuget_config(plugin)
        template = self.root / "dotnet/_template/.editorconfig"
        if not template.is_file():
            self.fail("dotnet/_template/.editorconfig: missing canonical editorconfig")
        elif (d / ".editorconfig").is_file() and (d / ".editorconfig").read_bytes() != template.read_bytes():
            self.fail(f"{rel}/.editorconfig: differs from canonical dotnet/_template/.editorconfig")
        if (d / library).is_file():
            self.check_dotnet_library(plugin, coordinate, d / library, f"net{runtimes[0]}" if runtimes else None)
        if (d / tests).is_file():
            self.check_dotnet_tests(plugin, d / tests)
        canonical = {(d / library).resolve(), (d / tests).resolve()}
        for project in projects:
            if project.resolve() not in canonical:
                self.check_dotnet_extra_project(plugin, project)
        makefile = d / "Makefile"
        if makefile.is_file():
            lines = [line.strip() for line in makefile.read_text(encoding="utf-8").splitlines()]
            # A trailing comment on the assignment would leave a space in the value; the include line may carry one.
            package = next((m.group(1) for m in (re.fullmatch(r"PACKAGE\s*:?=\s*(\S+)", line) for line in lines) if m), None)
            if package != coordinate:
                self.fail(f"{rel}/Makefile: must set PACKAGE := {coordinate}")
            if "include ../_shared/dotnet.mk" not in [line.split("#", 1)[0].strip() for line in lines]:
                self.fail(f"{rel}/Makefile: must include ../_shared/dotnet.mk")
        license_path = d / "LICENSE"
        if not license_path.is_file() or license_path.is_symlink():
            self.fail(f"{rel}: LICENSE must be a regular file")
        elif "LICENSE" not in tracked:
            self.fail(f"{rel}: LICENSE must be committed")
        elif license_path.read_bytes() != (self.root / "LICENSE").read_bytes():
            self.fail(f"{rel}: LICENSE differs from root")
        if any(path.name.upper().startswith("CHANGELOG") for path in d.iterdir()):
            self.fail(f"{rel}: release notes are generated; no changelog files")
        self.check_readme_links(plugin, "README.md", "nuget.org")

    def check_dotnet_global_json(self, plugin: Plugin, runtimes: list[str]) -> None:
        path = plugin.path / "global.json"
        if not path.is_file():
            return
        rel = f"{plugin.rel}/global.json"
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.fail(f"{rel}: invalid JSON ({exc})")
            return
        sdk = settings.get("sdk", {}) if isinstance(settings, dict) else {}
        version = str(sdk.get("version", ""))
        if runtimes and ".".join(version.split(".")[:2]) != runtimes[0]:
            self.fail(f"{rel}: sdk.version {version!r} must be in the floor channel {runtimes[0]} (first [ci] runtime-versions entry)")
        if sdk.get("rollForward") != "latestFeature":
            self.fail(f"{rel}: sdk.rollForward must be 'latestFeature' so a newer major SDK is never used silently")
        if sdk.get("allowPrerelease") is not False:
            self.fail(f"{rel}: sdk.allowPrerelease must be false")
        if not isinstance(settings, dict) or settings.get("test", {}).get("runner") != "Microsoft.Testing.Platform":
            self.fail(f"{rel}: test.runner must be 'Microsoft.Testing.Platform'")

    def check_dotnet_nuget_config(self, plugin: Plugin) -> None:
        path = plugin.path / "nuget.config"
        if not path.is_file():
            return
        rel = f"{plugin.rel}/nuget.config"
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            self.fail(f"{rel}: invalid XML ({exc})")
            return
        sources = root.find("packageSources")
        children = list(sources) if sources is not None else []
        if not children or children[0].tag != "clear":
            self.fail(f"{rel}: packageSources must start with <clear /> so a developer's feeds never reach the lock")
        adds = [child for child in children if child.tag == "add"]
        if [add.get("value") for add in adds] != [NUGET_ORG]:
            self.fail(f"{rel}: nuget.org ({NUGET_ORG}) must be the only package source")
        elif not (adds[0].get("minPublishAgeHours") or "").isdigit() or int(adds[0].get("minPublishAgeHours", "0")) <= 0:
            self.fail(f"{rel}: the nuget.org source must carry minPublishAgeHours (the dependency cooldown)")
        exceptions = root.find("minPublishAgeExceptions")
        patterns = [] if exceptions is None else [entry.get("pattern") for entry in exceptions.findall("package")]
        # The package-source-mapping shape: the SDK itself plus its dotted children; a Temporalio* prefix
        # would also exempt TemporalioCommunity.* and every other ID that happens to start with the word.
        if sorted(pattern for pattern in patterns if pattern) != ["Temporalio", "Temporalio.*"]:
            self.fail(f"{rel}: minPublishAgeExceptions must exempt exactly Temporalio and Temporalio.*")

    def _dotnet_project(self, plugin: Plugin, csproj: Path) -> tuple[dict[str, str], list[tuple[str, dict[str, str]]]] | None:
        """Parse a csproj into (properties, items) with lower-cased names; conditions are ignored, the last value wins."""
        try:
            root = ET.parse(csproj).getroot()
        except ET.ParseError as exc:
            self.fail(f"{plugin.rel}/{csproj.relative_to(plugin.path).as_posix()}: invalid XML ({exc})")
            return None
        properties: dict[str, str] = {}
        for group in root.iter("PropertyGroup"):
            for prop in group:
                properties[prop.tag.lower()] = (prop.text or "").strip()
        items: list[tuple[str, dict[str, str]]] = []
        for group in root.iter("ItemGroup"):
            for item in group:
                metadata = {key.lower(): value for key, value in item.attrib.items()}
                metadata.update({child.tag.lower(): (child.text or "").strip() for child in item})
                items.append((item.tag.lower(), metadata))
        return properties, items

    def check_dotnet_build_props(self, plugin: Plugin, props: Path) -> None:
        """The first element is the unconditioned shared import; the body below it applies to every project."""
        rel = f"{plugin.rel}/Directory.Build.props"
        try:
            root = ET.parse(props).getroot()
        except ET.ParseError as exc:
            self.fail(f"{rel}: invalid XML ({exc})")
            return
        first = next(iter(root), None)
        target = (first.get("Project") or "").replace("\\", "/") if first is not None else ""
        if (root.tag.rsplit("}", 1)[-1] != "Project" or first is None or first.tag.rsplit("}", 1)[-1] != "Import"
                or target != DOTNET_SHARED_IMPORT_PATH or first.get("Condition") is not None):
            self.fail(f"{rel}: the first element must be the unconditioned shared import:\n{DOTNET_SHARED_IMPORT.rstrip()}")
        parsed = self._dotnet_project(plugin, props)
        if parsed is None:
            return
        properties, items = parsed
        if "version" in properties and properties["version"] != DOTNET_DEVELOPMENT_VERSION:
            self.fail(f"{rel}: Version must be the tag-authoritative development placeholder "
                      f"{DOTNET_DEVELOPMENT_VERSION!r} (got {properties['version']!r})")
        if "packagelicenseexpression" in properties or "packagelicenseurl" in properties:
            self.fail(f"{rel}: ship the license text; no PackageLicenseExpression or PackageLicenseUrl")
        if properties.get("ispackable", "").lower() == "true":
            self.fail(f"{rel}: IsPackable here would pack the test project too; set it in the library csproj")
        self._dotnet_common_rules(plugin, props, properties, items, DOTNET_POLICY_PROPERTIES, require_stylecop=False)

    def _dotnet_common_rules(self, plugin: Plugin, csproj: Path, properties: dict[str, str],
                             items: list[tuple[str, dict[str, str]]], policy: dict[str, str], *,
                             require_stylecop: bool = True) -> None:
        rel = f"{plugin.rel}/{csproj.relative_to(plugin.path).as_posix()}"
        for name, value in policy.items():
            actual = properties.get(name.lower())
            if actual is not None and actual.lower() != value:
                self.fail(f"{rel}: {name} must stay {value} (shared policy from dotnet/_shared/dotnet.props)")
        for name in DOTNET_FORBIDDEN_PROPERTIES:
            if name.lower() in properties:
                self.fail(f"{rel}: {name} is not allowed; it adds SDK-pinned implicit package references that break locked restores")
        references = {metadata.get("include", ""): metadata for tag, metadata in items if tag == "packagereference"}
        stylecop = next((metadata for include, metadata in references.items() if include.lower() == "stylecop.analyzers"), None)
        if require_stylecop and (stylecop is None or stylecop.get("privateassets", "").lower() != "all"):
            self.fail(f"{rel}: StyleCop.Analyzers must be referenced with PrivateAssets=\"all\"")
        for include in references:
            if "/" in include or "\\" in include:
                self.fail(f"{rel}: PackageReference {include!r} must name a published coordinate, not a path")
        for tag, metadata in items:
            target = metadata.get("include") if tag == "projectreference" else metadata.get("hintpath") if tag == "reference" else None
            if target and not (csproj.parent / target.replace("\\", "/")).resolve().is_relative_to(plugin.path.resolve()):
                self.fail(f"{rel}: {target} points outside the plugin; cross-plugin dependencies use published coordinates")

    def check_dotnet_library(self, plugin: Plugin, coordinate: str, csproj: Path, floor: str | None) -> None:
        parsed = self._dotnet_project(plugin, csproj)
        if parsed is None:
            return
        properties, items = parsed
        rel = f"{plugin.rel}/{csproj.relative_to(plugin.path).as_posix()}"
        if properties.get("packageid") != coordinate:
            self.fail(f"{rel}: PackageId must be {coordinate!r} (got {properties.get('packageid')!r})")
        if properties.get("version") != DOTNET_DEVELOPMENT_VERSION:
            self.fail(f"{rel}: Version must be the tag-authoritative development placeholder "
                      f"{DOTNET_DEVELOPMENT_VERSION!r} (got {properties.get('version')!r})")
        if properties.get("ispackable", "").lower() != "true":
            self.fail(f"{rel}: IsPackable must be true")
        frameworks = [part.strip() for part in properties.get("targetframeworks", "").split(";") if part.strip()]
        frameworks += [properties["targetframework"]] if properties.get("targetframework") else []
        if floor and floor not in frameworks:
            self.fail(f"{rel}: TargetFramework(s) must include {floor} (the [ci] runtime-versions channel)")
        if properties.get("packagelicensefile") != "LICENSE":
            self.fail(f"{rel}: PackageLicenseFile must be LICENSE")
        if "packagelicenseexpression" in properties or "packagelicenseurl" in properties:
            self.fail(f"{rel}: ship the license text; no PackageLicenseExpression or PackageLicenseUrl")
        if properties.get("packagereadmefile") != "README.md":
            self.fail(f"{rel}: PackageReadmeFile must be README.md")
        if properties.get("packageprojecturl") != f"{DOTNET_REPOSITORY_TREE}/dotnet/{plugin.name}":
            self.fail(f"{rel}: PackageProjectUrl must be {DOTNET_REPOSITORY_TREE}/dotnet/{plugin.name}")
        if any(tag == "projectreference" for tag, _ in items):
            self.fail(f"{rel}: a ProjectReference in the packed library becomes a nuspec dependency on a coordinate that "
                      "is not on nuget.org; merge the code into the library or publish it as its own plugin")
        packed = {metadata.get("include", "").replace("\\", "/") for tag, metadata in items
                  if tag == "none" and metadata.get("pack", "").lower() == "true"}
        for name in ("../../LICENSE", "../../README.md"):
            if name not in packed:
                self.fail(f"{rel}: must pack {name} (<None Include=\"{name}\" Pack=\"true\" PackagePath=\"\" />)")
        temporalio = next((metadata for tag, metadata in items if tag == "packagereference" and metadata.get("include", "").lower() == "temporalio"), None)
        if temporalio is None or not temporalio.get("version"):
            self.fail(f"{rel}: a PackageReference to Temporalio with an explicit Version is required")
        self._dotnet_common_rules(plugin, csproj, properties, items, DOTNET_POLICY_PROPERTIES)

    def check_dotnet_extra_project(self, plugin: Plugin, csproj: Path) -> None:
        """Any other project the plugin carries (a sample, a benchmark) builds under the same policy."""
        parsed = self._dotnet_project(plugin, csproj)
        if parsed is None:
            return
        properties, items = parsed
        rel = f"{plugin.rel}/{csproj.relative_to(plugin.path).as_posix()}"
        if properties.get("ispackable", "").lower() == "true":
            self.fail(f"{rel}: only src/<Coordinate> is packable; a second package needs its own plugin")
        policy = {name: value for name, value in DOTNET_POLICY_PROPERTIES.items() if name != "GenerateDocumentationFile"}
        self._dotnet_common_rules(plugin, csproj, properties, items, policy)

    def check_dotnet_tests(self, plugin: Plugin, csproj: Path) -> None:
        parsed = self._dotnet_project(plugin, csproj)
        if parsed is None:
            return
        properties, items = parsed
        rel = f"{plugin.rel}/{csproj.relative_to(plugin.path).as_posix()}"
        if properties.get("usemicrosofttestingplatformrunner", "").lower() != "true":
            self.fail(f"{rel}: UseMicrosoftTestingPlatformRunner must be true (xunit.v3 under Microsoft.Testing.Platform)")
        if properties.get("outputtype", "").lower() != "exe":
            self.fail(f"{rel}: OutputType must be Exe (Microsoft.Testing.Platform test projects are executables)")
        if properties.get("ispackable", "").lower() == "true":
            self.fail(f"{rel}: the test project must not be packable")
        references = {metadata.get("include", "").lower() for tag, metadata in items if tag == "packagereference"}
        for required in ("xunit.v3", "Microsoft.Testing.Extensions.TrxReport"):
            if required.lower() not in references:
                self.fail(f"{rel}: must reference {required}")
        for banned in DOTNET_VSTEST_PACKAGES:
            if banned.lower() in references:
                self.fail(f"{rel}: {banned} is a VSTest package; Microsoft.Testing.Platform mode rejects it")
        policy = {name: value for name, value in DOTNET_POLICY_PROPERTIES.items() if name != "GenerateDocumentationFile"}
        self._dotnet_common_rules(plugin, csproj, properties, items, policy)


    def run(self) -> list[str]:
        discovered = discover_plugins(self.root)
        self.check_language_roots(discovered)
        for plugin in discovered["python"]:
            self.check_python_plugin(plugin)
        for plugin in discovered["java"]:
            self.check_java_plugin(plugin)
        for plugin in discovered["typescript"]:
            self.check_typescript_plugin(plugin)
        for plugin in discovered["dotnet"]:
            self.check_dotnet_plugin(plugin)
        return self.violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)
    violations = Checker(repo_root(args.repo_root)).run()
    if violations:
        print(f"FAIL: {len(violations)} convention violation(s):")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("OK: repository conventions satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
