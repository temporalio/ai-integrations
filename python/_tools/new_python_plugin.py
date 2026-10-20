#!/usr/bin/env python3
"""Scaffold a new Python plugin directory from ``python/_template``.

Usage:
    bazel run //python/_tools:new_plugin -- NAME --description "..." [--coordinate NAME] \
        [--maturity pre-release|public-preview|generally-available]

Standard library only; Python 3.11+.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(
    os.environ.get("BUILD_WORKSPACE_DIRECTORY", Path(__file__).resolve().parents[2])
)
TEMPLATE = REPO_ROOT / "python" / "_template"
MATURITY_CLASSIFIER = {
    "generally-available": "Development Status :: 5 - Production/Stable",
    "public-preview": "Development Status :: 4 - Beta",
    "pre-release": "Development Status :: 3 - Alpha",
}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "name", help="folder name under python/, snake_case (e.g. openai_agents)"
    )
    parser.add_argument(
        "--description", required=True, help="one-line package description"
    )
    parser.add_argument(
        "--coordinate", help="PyPI name; default temporalio-<name with _ -> ->"
    )
    parser.add_argument(
        "--maturity", choices=sorted(MATURITY_CLASSIFIER), default="pre-release"
    )
    parser.add_argument(
        "--existing",
        action="store_true",
        help="add missing files to an existing plugin dir",
    )
    return parser.parse_args(argv)


def validate_name(name: str) -> None:
    if not NAME_RE.match(name):
        sys.exit(f"error: name must match {NAME_RE.pattern} (snake_case), got {name!r}")
    if name.startswith("_"):
        sys.exit(
            "error: names starting with '_' are reserved for non-plugin directories"
        )
    if name.endswith(("_plugin", "-plugin")):
        sys.exit(
            "error: plugin folders must not end in -plugin/_plugin (design doc acceptance rule)"
        )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    name = args.name
    validate_name(name)
    coordinate = args.coordinate or "temporalio-" + name.replace("_", "-")
    module = f"temporalio.{name}"
    substitutions = {
        "__NAME__": name,
        "__COORDINATE__": coordinate,
        "__MODULE__": module,
        "__MODULE_PATH__": module.replace(".", os.sep),
        "__DESCRIPTION__": args.description,
        "__MATURITY__": args.maturity,
        "__MATURITY_CLASSIFIER__": MATURITY_CLASSIFIER[args.maturity],
        "__DOCS_SLUG__": name.replace("_", "-"),
    }

    target = REPO_ROOT / "python" / name
    if target.exists():
        sys.exit(f"error: {target} already exists")

    written: list[Path] = []
    for src in sorted(TEMPLATE.rglob("*")):
        if src.is_dir() or "__pycache__" in src.parts or src.suffix == ".pyc":
            continue
        rel = str(src.relative_to(TEMPLATE))
        for placeholder in ("__NAME__", "__MODULE_PATH__"):
            rel = rel.replace(placeholder, substitutions[placeholder])
        if rel.endswith(".tmpl"):
            rel = rel[: -len(".tmpl")]
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            os.symlink(os.readlink(src), dest)
        else:
            text = src.read_text(encoding="utf-8")
            for placeholder, value in substitutions.items():
                text = text.replace(placeholder, value)
            dest.write_text(text, encoding="utf-8")
        written.append(dest)

    license_dest = target / "LICENSE"
    if not license_dest.exists():
        license_dest.write_bytes(
            (REPO_ROOT / "LICENSE").read_bytes()
        )  # committed copy; conventions keep it identical
        written.append(license_dest)

    for path in written:
        print(f"wrote   {path.relative_to(REPO_ROOT)}")

    print(
        f"""
Next steps for python/{name} ({coordinate}, maturity={args.maturity}):
  1. Fill in [project].dependencies in pyproject.toml (declare exactly what is imported).
  2. bazel run //python/_tools:dev -- lock --plugin-dir python/{name}   # creates uv.lock; commit it
  3. Add an independent pip.parse hub in python/_deps.MODULE.bazel (minimum and maximum Python).
  4. bazel mod deps --lockfile_mode=update   # review and commit MODULE.bazel.lock
  5. bazel test //python/{name}:lint //python/{name}:tests
  6. Cover provider behavior with deterministic local models or transports; CI has no provider secrets.
  7. Add a row to the plugin table in README.md.
  8. Ask a PyPI org admin for pending trusted publishers for "{coordinate}" on test.pypi.org
     (environment testpypi) and pypi.org (environment pypi), both bound to
     temporalio/ai-integrations, workflow release-python.yml.
  9. Open the PR. CI discovers the plugin from plugin.toml; no workflow edits are needed.
"""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
