#!/usr/bin/env python3
"""Decide which plugins CI must run for the current event.

Stdlib only (runs with the runner's system python3 before uv is installed).

Rules (see AGENTS.md, "CI architecture"):
  * push to main, schedule, workflow_dispatch, FORCE_ALL, unknown events, or an
    unusable diff base           -> every plugin of every language
  * .github/** or scripts/ci/**  -> every plugin of every language
  * <lang>/<plugin>/**           -> that plugin
  * any other <lang>/** file     -> every plugin of that language (shared resources)
  * scripts/** (non-ci)          -> no plugins; reported as scripts_only when nothing else changed
  * anything else (root files)   -> nothing

Outputs (GITHUB_OUTPUT and stdout): python, typescript, java, go (sorted JSON
arrays, literal [] when empty), any, scripts_only, mode (all|diff|none).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LANGUAGES, compact_json, discover_plugins, repo_root, write_github_output  # noqa: E402

ALL_EVENTS = {"push", "schedule", "workflow_dispatch"}
ZERO_SHA = "0" * 40


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def changed_files(root: Path, base: str, head: str, three_dot: bool) -> list[str]:
    spec = f"{base}...{head}" if three_dot else f"{base} {head}"
    out = _git(root, "diff", "--name-only", "--no-renames", *spec.split(" "))
    return [line.strip() for line in out.splitlines() if line.strip()]


def select(
    files: list[str], plugins: dict[str, list[str]]
) -> tuple[dict[str, set[str]], bool, bool]:
    """Apply the path rules. Returns (selected-by-language, run_all, scripts_only)."""
    selected: dict[str, set[str]] = {lang: set() for lang in LANGUAGES}
    scripts_touched = False
    other_touched = False
    for f in files:
        parts = f.split("/")
        top = parts[0]
        if top == ".github" or f == "LICENSE":
            # Workflow/tooling changes affect every plugin; so does the root LICENSE, which every
            # plugin packages through its LICENSE symlink.
            return selected, True, False
        if top == "scripts":
            if len(parts) > 1 and parts[1] == "ci":
                return selected, True, False
            scripts_touched = True
            continue
        if top in plugins:
            if len(parts) >= 3 and parts[1] in plugins[top]:
                selected[top].add(parts[1])
            else:
                selected[top].update(plugins[top])
            other_touched = True
            continue
        # Other root-level files (README.md, AGENTS.md, ...) select nothing.
    return selected, False, scripts_touched and not other_touched and not any(selected.values())


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--base", default=None, help="diff base ref (defaults from the event)")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--all", action="store_true", help="select every plugin")
    parser.add_argument("--github-output", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print only; do not write GITHUB_OUTPUT")
    args = parser.parse_args(argv)

    root = repo_root(args.repo_root)
    discovered = discover_plugins(root)
    plugins = {lang: [p.name for p in ps] for lang, ps in discovered.items()}

    mode = "diff"
    run_all = args.all or os.environ.get("FORCE_ALL", "").lower() in {"1", "true", "yes"}
    files: list[str] = []
    if not run_all:
        event = args.event
        try:
            if event == "pull_request":
                base = args.base or f"origin/{os.environ.get('GITHUB_BASE_REF', 'main')}"
                files = changed_files(root, base, args.head, three_dot=True)
            elif event == "merge_group":
                base = args.base or os.environ.get("MERGE_GROUP_BASE_SHA", "")
                head = os.environ.get("MERGE_GROUP_HEAD_SHA") or args.head
                if not base:
                    raise ValueError("MERGE_GROUP_BASE_SHA missing")
                files = changed_files(root, base, head, three_dot=False)
            elif not event and args.base:  # local use: detect_changes.py --base origin/main
                files = changed_files(root, args.base, args.head, three_dot=True)
            else:  # push, schedule, workflow_dispatch, unknown events, or no diff base
                run_all = True
        except (subprocess.CalledProcessError, ValueError) as exc:  # fail safe: run everything
            print(f"::warning::change detection could not compute a diff ({exc}); running everything", file=sys.stderr)
            run_all = True

    scripts_only = False
    if run_all:
        selected = {lang: set(names) for lang, names in plugins.items()}
        mode = "all"
    else:
        selected, all_from_rules, scripts_only = select(files, plugins)
        if all_from_rules:
            selected = {lang: set(names) for lang, names in plugins.items()}
            mode = "all"
        elif not any(selected.values()):
            mode = "none"

    result: dict[str, object] = {lang: sorted(selected[lang]) for lang in LANGUAGES}
    result["any"] = any(selected.values())
    result["scripts_only"] = scripts_only
    result["mode"] = mode

    outputs = {lang: compact_json(result[lang]) for lang in LANGUAGES}
    outputs["any"] = "true" if result["any"] else "false"
    outputs["scripts_only"] = "true" if scripts_only else "false"
    outputs["mode"] = mode
    for key, value in outputs.items():
        print(f"{key}={value}")
    if files and not run_all:
        print(f"changed files considered: {len(files)}", file=sys.stderr)
    if not args.dry_run:
        write_github_output(args.github_output or os.environ.get("GITHUB_OUTPUT"), outputs)
    return result


if __name__ == "__main__":
    main()
