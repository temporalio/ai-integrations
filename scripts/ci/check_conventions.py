#!/usr/bin/env python3
"""Repository-invariant checks for temporalio/ai-integrations.

Runs on every PR (job `conventions`) and nightly (`--nightly`). Exits non-zero
with a list of violations. Passes on a repository with zero plugins.

Checks (see AGENTS.md, "Repository invariants" and "Python conventions"):
  * plugin folder names never end in `-plugin` / `_plugin`
  * no language-level lockfiles (python/uv.lock, typescript/pnpm-lock.yaml, ...)
  * every Python plugin has pyproject.toml, uv.lock, plugin.toml, Makefile, README.md,
    src/temporalio/contrib/<name>/{__init__.py,py.typed}
  * NO src/temporalio/__init__.py and NO src/temporalio/contrib/__init__.py (namespace invariant)
  * LICENSE is a committed regular file byte-identical to the root LICENSE; pyproject declares license = "MIT"
    and license-files = ["LICENSE"]; no CHANGELOG*, no smoke_test.py in the plugin dir
  * plugin.toml schema and agreement with pyproject.toml (name/coordinate/root-api/
    maturity classifier/requires-python floor/module-name/required-version)
  * no [tool.uv.sources] path or workspace entries
  * README has no relative markdown links (PyPI renders the README)
  * PR context: a PR with more than 20 commits must carry the `history-import` label
  * --nightly: coordinates with [release] allow-final = false must not exist on PyPI yet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LANGUAGES, Plugin, discover_plugins, load_toml, repo_root  # noqa: E402

MATURITY_CLASSIFIER = {
    "ga": "Development Status :: 5 - Production/Stable",
    "preview": "Development Status :: 4 - Beta",
    "experimental": "Development Status :: 3 - Alpha",
}
REGISTRIES = {"python": "pypi", "typescript": "npm", "java": "maven", "go": "goproxy"}
LANGUAGE_LOCKFILES = ("uv.lock", "pnpm-lock.yaml", "package-lock.json", "yarn.lock", "go.sum", "gradle.lockfile")
RELATIVE_LINK = re.compile(r"\]\((\.\.?/)")
MAX_PR_COMMITS_WITHOUT_LABEL = 20
HISTORY_IMPORT_LABEL = "history-import"


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
            for child in lang_dir.iterdir():
                if child.is_dir() and not child.name.startswith(("_", ".")):
                    if child.name.endswith(("-plugin", "_plugin")):
                        self.fail(f"{language}/{child.name}: plugin folders must not end in -plugin/_plugin")
                    if child not in [p.path for p in discovered[language]]:
                        self.fail(f"{language}/{child.name}: directory has no {language} manifest; non-plugin folders must start with `_`")

    def check_python_plugin(self, plugin: Plugin) -> None:
        d = plugin.path
        rel = plugin.rel
        name = plugin.name
        pkg = d / "src" / "temporalio" / "contrib" / name
        for required in ("pyproject.toml", "uv.lock", "plugin.toml", "Makefile", "README.md"):
            if not (d / required).is_file():
                self.fail(f"{rel}: missing {required}")
        if not (pkg / "__init__.py").is_file():
            self.fail(f"{rel}: missing src/temporalio/contrib/{name}/__init__.py")
        if not (pkg / "py.typed").is_file():
            self.fail(f"{rel}: missing src/temporalio/contrib/{name}/py.typed")
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

        if not (d / "pyproject.toml").is_file() or not (d / "plugin.toml").is_file():
            return
        try:
            pyproject = load_toml(d / "pyproject.toml")
            meta = load_toml(d / "plugin.toml")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"{rel}: cannot parse TOML: {exc}")
            return
        self.check_plugin_toml(plugin, meta, pyproject)
        self.check_pyproject(plugin, pyproject)
        self.check_readme(plugin)

    def check_plugin_toml(self, plugin: Plugin, meta: dict[str, Any], pyproject: dict[str, Any]) -> None:
        rel = plugin.rel
        p = meta.get("plugin")
        if not isinstance(p, dict):
            self.fail(f"{rel}: plugin.toml missing [plugin] table")
            return
        expected_coordinate = "temporalio-" + plugin.name.replace("_", "-")
        expected_root_api = "temporalio.contrib." + plugin.name
        if p.get("name") != plugin.name:
            self.fail(f"{rel}: plugin.toml name {p.get('name')!r} must equal the folder name {plugin.name!r}")
        if p.get("language") != plugin.language:
            self.fail(f"{rel}: plugin.toml language must be {plugin.language!r}")
        if p.get("coordinate") != expected_coordinate:
            self.fail(f"{rel}: plugin.toml coordinate must be {expected_coordinate!r} (got {p.get('coordinate')!r})")
        if p.get("registry") != REGISTRIES[plugin.language]:
            self.fail(f"{rel}: plugin.toml registry must be {REGISTRIES[plugin.language]!r}")
        if p.get("root-api") != expected_root_api:
            self.fail(f"{rel}: plugin.toml root-api must be {expected_root_api!r}")
        maturity = p.get("maturity")
        if maturity not in MATURITY_CLASSIFIER:
            self.fail(f"{rel}: plugin.toml maturity must be one of {sorted(MATURITY_CLASSIFIER)}")
        for banned in ("owners", "live-secrets", "secrets"):
            if banned in p or banned in meta.get("ci", {}):
                self.fail(f"{rel}: plugin.toml must not contain {banned!r} (ownership is CODEOWNERS; CI has no secrets)")
        release = meta.get("release", {})
        if not isinstance(release.get("allow-final"), bool):
            self.fail(f"{rel}: plugin.toml [release] allow-final must be a boolean")
        versions = meta.get("ci", {}).get("runtime-versions")
        if not isinstance(versions, list) or not versions:
            self.fail(f"{rel}: plugin.toml [ci] runtime-versions must be a non-empty list")
            versions = []
        smoke_imports = meta.get("smoke", {}).get("imports", [])
        if smoke_imports and not all(isinstance(i, str) and i.startswith(expected_root_api) for i in smoke_imports):
            self.fail(f"{rel}: plugin.toml [smoke] imports must be modules under {expected_root_api}")

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
        if module_name != expected_root_api:
            self.fail(f"{rel}: [tool.uv.build-backend] module-name must be {expected_root_api!r} (got {module_name!r})")

    def check_pyproject(self, plugin: Plugin, pyproject: dict[str, Any]) -> None:
        rel = plugin.rel
        project = pyproject.get("project", {})
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

    def check_readme(self, plugin: Plugin) -> None:
        readme = plugin.path / "README.md"
        if not readme.is_file():
            return
        for lineno, line in enumerate(readme.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if RELATIVE_LINK.search(line):
                self.fail(f"{plugin.rel}/README.md:{lineno}: relative link; use absolute https://github.com/... URLs (PyPI renders this file)")

    def check_pr_context(self) -> None:
        if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
            return
        try:
            commits = int(os.environ.get("PR_COMMITS", "0") or 0)
        except ValueError:
            commits = 0
        try:
            labels = json.loads(os.environ.get("PR_LABELS", "[]") or "[]")
        except json.JSONDecodeError:
            labels = []
        if commits > MAX_PR_COMMITS_WITHOUT_LABEL and HISTORY_IMPORT_LABEL not in labels:
            self.fail(
                f"PR has {commits} commits without the `{HISTORY_IMPORT_LABEL}` label; only history imports/re-syncs may carry that many commits, and they must be merged with a merge commit"
            )

    def check_nightly(self, plugins: list[Plugin]) -> None:
        for plugin in plugins:
            meta_path = plugin.path / "plugin.toml"
            if not meta_path.is_file():
                continue
            meta = load_toml(meta_path)
            if meta.get("release", {}).get("allow-final") is not False:
                continue
            coordinate = meta.get("plugin", {}).get("coordinate")
            if not coordinate:
                continue
            url = f"https://pypi.org/pypi/{coordinate}/json"
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                    status = resp.status
            except urllib.error.HTTPError as exc:
                status = exc.code
            except urllib.error.URLError as exc:
                print(f"::warning::{plugin.rel}: could not query PyPI ({exc}); skipping name-squat watch", file=sys.stderr)
                continue
            if status == 200:
                self.fail(f"{plugin.rel}: {coordinate} already exists on PyPI while [release] allow-final = false (name-squat or unexpected publish)")
            elif status != 404:
                print(f"::warning::{plugin.rel}: PyPI returned {status} for {coordinate}", file=sys.stderr)

    def run(self, nightly: bool) -> list[str]:
        discovered = discover_plugins(self.root)
        self.check_language_roots(discovered)
        for plugin in discovered["python"]:
            self.check_python_plugin(plugin)
        self.check_pr_context()
        if nightly:
            self.check_nightly(discovered["python"])
        return self.violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--nightly", action="store_true", help="also run the PyPI name-squat watch")
    args = parser.parse_args(argv)
    violations = Checker(repo_root(args.repo_root)).run(nightly=args.nightly)
    if violations:
        print(f"FAIL: {len(violations)} convention violation(s):")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("OK: repository conventions satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
