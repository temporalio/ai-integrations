#!/usr/bin/env python3
"""Report nightly CI and Dependabot update failures as GitHub issues.

Reads the jobs of the current workflow run through `gh api`, maps reusable
workflow job names such as `Python (openai_agents) / ...`, `Go (googleadk) / ...`
and `.NET (microsoft-agent-framework) / ...` to a plugin, and keeps exactly one open issue
per failing plugin labelled `nightly`.
Passing plugins with an open issue get a comment and are closed.

With --dependabot, also track Dependabot update jobs separately. Security-update
failures are relevant only while a matching alert remains open. Remember the
last failure in the issue so expired Actions history cannot imply recovery.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from _common import discover_plugins, repo_root

# Job names come from ci.yml (`name: Python`, `name: Java`, `name: Go`, `name: TypeScript`, `name: .NET`)
# joined with the reusable workflow's job name; test_nightly_report.py asserts the two stay in step.
JOB_RE = re.compile(r"^(?P<language>Python|Java|Go|TypeScript|\.NET) \((?P<plugin>[^)]+)\) / ")
# Issue keys are `<language root folder>/<plugin>`: `.lower()` of the job's language, except `.NET`, whose root is `dotnet`.
KEY_PREFIX = {".NET": "dotnet"}
LABEL = "nightly"
FAILED = {"failure", "timed_out"}
DEPENDABOT_LABEL = "dependabot-failure"
DEPENDABOT_PATH = "dynamic/dependabot/dependabot-updates"
DEPENDABOT_FAILED = FAILED | {"startup_failure", "action_required"}
DEPENDABOT_RE = re.compile(r"^(?P<manager>\S+) in (?P<directory>/\S*)(?: for (?P<dependency>.+))?$")
DEPENDABOT_STATE_RE = re.compile(r"<!-- dependabot-failures: (.+) -->")
ECOSYSTEMS = {
    "uv": "pip", "pip": "pip", "npm_and_yarn": "npm", "npm": "npm",
    "gradle": "maven", "maven": "maven", "gomod": "go",
    "github_actions": "github-actions",
}


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def _json_documents(text: str) -> list:
    """Parse the concatenated JSON documents that `gh api --paginate` emits (one per page)."""
    decoder = json.JSONDecoder()
    docs: list = []
    idx = 0
    text = text.strip()
    while idx < len(text):
        doc, end = decoder.raw_decode(text, idx)
        docs.append(doc)
        idx = end
        while idx < len(text) and text[idx].isspace():
            idx += 1
    return docs


def classify(jobs: list[dict]) -> tuple[set[str], set[str]]:
    failing: set[str] = set()
    seen: set[str] = set()
    for job in jobs:
        m = JOB_RE.match(job.get("name", ""))
        if not m or not job.get("conclusion"):
            continue
        language = m.group("language")
        plugin = f"{KEY_PREFIX.get(language, language.lower())}/{m.group('plugin')}"
        seen.add(plugin)
        if job["conclusion"] in FAILED:
            failing.add(plugin)
    return failing, seen - failing


def title_for(plugin: str) -> str:
    return f"nightly: {plugin} failing (latest)"


def _api_items(endpoint: str, field: str | None = None) -> list[dict]:
    items: list[dict] = []
    for page in _json_documents(_gh("api", endpoint, "--paginate")):
        items.extend(page[field] if field else page)
    return items


def _run_label(run: dict) -> str:
    return re.sub(r" - Update #\d+$", "", run.get("display_title") or run.get("name") or "Dependabot update")


def _package_name(ecosystem: str, name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower() if ecosystem == "pip" else name


def dependabot_key(run: dict) -> tuple[str, ...]:
    label = _run_label(run)
    match = DEPENDABOT_RE.fullmatch(label)
    if not match:
        return (label,)
    manager, directory, dependency = match.group("manager", "directory", "dependency")
    ecosystem = ECOSYSTEMS.get(manager, manager)
    return ecosystem, directory.rstrip("/"), _package_name(ecosystem, dependency or "")


def _run_order(run: dict) -> tuple[str, int, int]:
    return run.get("run_started_at") or run.get("created_at") or "", run["id"], run.get("run_attempt") or 1


def dependabot_plugins(run: dict, alerts: list[dict], plugins: set[str]) -> set[str]:
    match = DEPENDABOT_RE.fullmatch(_run_label(run))
    if not match:
        # Surface format changes and setup failures rather than silently dropping them.
        return {"repository"}
    manager, directory, dependency = match.group("manager", "directory", "dependency")
    if dependency is None:
        return {plugin for plugin in plugins if fnmatchcase(f"/{plugin}", directory)} or {
            "scripts" if directory == "/scripts" else "repository"
        }

    ecosystem = ECOSYSTEMS.get(manager, manager)
    affected: set[str] = set()
    for alert in alerts:
        dep = alert["dependency"]
        package = dep["package"]
        manifest = PurePosixPath(dep["manifest_path"])
        if (package["ecosystem"] != ecosystem
                or _package_name(ecosystem, package["name"]) != _package_name(ecosystem, dependency)
                or not fnmatchcase(f"/{manifest}", directory.rstrip("/") + "/*")
                or alert.get("state", "open") != "open"):
            continue
        # A failure predating a newly discovered alert is not an attempt to fix it.
        if alert.get("created_at") and _run_order(run)[0] and alert["created_at"] > _run_order(run)[0]:
            continue
        candidate = "/".join(manifest.parts[:2])
        if candidate in plugins:
            affected.add(candidate)
        else:
            affected.add("scripts" if manifest.parts[0] == "scripts" else "repository")
    return affected


def dependabot_failures(runs: list[dict], alerts: list[dict], plugins: set[str]) -> dict[str, list[dict]]:
    latest: dict[tuple[str, ...], dict] = {}
    for run in sorted(runs, key=_run_order):
        if run.get("status") == "completed" and run.get("conclusion") in DEPENDABOT_FAILED | {"success"}:
            latest[dependabot_key(run)] = run
    failures: dict[str, list[dict]] = {}
    for run in latest.values():
        if run["conclusion"] in DEPENDABOT_FAILED:
            for plugin in dependabot_plugins(run, alerts, plugins):
                failures.setdefault(plugin, []).append(run)
    return failures


def _dependabot_state(body: str) -> list[dict] | None:
    match = DEPENDABOT_STATE_RE.search(body)
    if not match:
        return None
    try:
        state = json.loads(match[1])
    except json.JSONDecodeError:
        return None
    if not isinstance(state, list) or not all(
        isinstance(run, dict) and isinstance(run.get("id"), int)
        and isinstance(run.get("name"), str) and isinstance(run.get("html_url"), str)
        and run.get("status") == "completed" and run.get("conclusion") in DEPENDABOT_FAILED for run in state
    ):
        return None
    return state


def _dependabot_body(plugin: str, runs: list[dict]) -> str:
    state = [
        {"name": _run_label(run), **{key: run.get(key) for key in (
            "id", "html_url", "status", "conclusion", "created_at", "run_started_at", "run_attempt",
        )}}
        for run in sorted(runs, key=dependabot_key)
    ]
    links = "\n".join(f"- `{run['name']}`: [{run['conclusion']}]({run['html_url']})" for run in state)
    return (
        f"Dependabot updates are failing for `{plugin}`.\n\n{links}\n\n"
        "Checked nightly by `scripts/ci/nightly_report.py`. This issue closes when the updates succeed "
        "or the related security alerts are resolved. Unchanged failures do not post daily comments.\n\n"
        f"<!-- dependabot-failures: {json.dumps(state, sort_keys=True)} -->"
    )


def _with_body(*args: str, body: str) -> str:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "body.md"
        path.write_text(body, encoding="utf-8")
        return _gh(*args, "--body-file", str(path))


def report_dependabot(repo: str, *, dry_run: bool = False) -> None:
    workflows = _api_items(f"repos/{repo}/actions/workflows?per_page=100", "workflows")
    runs = [
        run for workflow in workflows if workflow["path"] == DEPENDABOT_PATH
        for run in _api_items(f"repos/{repo}/actions/workflows/{workflow['id']}/runs?per_page=100", "workflow_runs")
    ]
    alerts = _api_items(f"repos/{repo}/dependabot/alerts?state=open&per_page=100")
    issues = json.loads(_gh("issue", "list", "--repo", repo, "--label", DEPENDABOT_LABEL,
                            "--state", "open", "--limit", "200", "--json", "number,title,body") or "[]")
    previous = {issue["number"]: _dependabot_state(issue.get("body") or "") for issue in issues}
    # Live records win ties, including a rerun of a previously failed Actions run.
    history = [run for state in previous.values() if state is not None for run in state] + runs
    plugins = {plugin.rel for group in discover_plugins(repo_root()).values() for plugin in group}
    failures = dependabot_failures(history, alerts, plugins)
    print(f"dependabot failing: {sorted(failures)}")
    for plugin, failed in sorted(failures.items()):
        for run in failed:
            print(f"  {plugin}: {_run_label(run)}: {run['html_url']}")
    if dry_run:
        return

    if failures:
        _gh("label", "create", DEPENDABOT_LABEL, "--repo", repo, "--force", "--color", "B60205",
            "--description", "Dependabot dependency updates are failing")
    by_title = {issue["title"]: issue for issue in issues}
    for plugin, failed in sorted(failures.items()):
        title = f"dependabot: {plugin} updates failing"
        body = _dependabot_body(plugin, failed)
        issue = by_title.get(title)
        if issue is None:
            _with_body("issue", "create", "--repo", repo, "--title", title, "--label", DEPENDABOT_LABEL, body=body)
        else:
            if issue.get("body") != body:
                _with_body("issue", "edit", str(issue["number"]), "--repo", repo, body=body)
            old = {dependabot_key(run): run["conclusion"] for run in previous[issue["number"]] or []}
            new = {dependabot_key(run): run["conclusion"] for run in failed}
            if old != new:
                _with_body("issue", "comment", str(issue["number"]), "--repo", repo,
                           body="The failed Dependabot updates have changed.\n\n" + body.split("<!--", 1)[0])
    active_titles = {f"dependabot: {plugin} updates failing" for plugin in failures}
    for issue in issues:
        if issue["title"] not in active_titles and previous[issue["number"]] is not None:
            _gh("issue", "close", str(issue["number"]), "--repo", repo, "--comment",
                "Dependabot updates have recovered, or the related security alerts are resolved.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dependabot", action="store_true", help="Also report Dependabot update failures")
    args = parser.parse_args(argv)
    if not args.repo or not args.run_id:
        parser.error("--repo and --run-id are required")

    jobs: list[dict] = []
    for page in _json_documents(_gh("api", f"repos/{args.repo}/actions/runs/{args.run_id}/jobs?per_page=100", "--paginate")):
        jobs.extend(page.get("jobs", []))
    failing, passing = classify(jobs)
    run_url = f"https://github.com/{args.repo}/actions/runs/{args.run_id}"
    print(f"failing: {sorted(failing)}\npassing: {sorted(passing)}")
    if args.dry_run:
        if args.dependabot:
            report_dependabot(args.repo, dry_run=True)
        return 0

    subprocess.run(["gh", "label", "create", LABEL, "--repo", args.repo, "--force", "--color", "B60205",
                    "--description", "Nightly checks with latest allowed dependencies failing"], check=False, capture_output=True)
    open_issues = json.loads(_gh("issue", "list", "--repo", args.repo, "--label", LABEL, "--state", "open", "--limit", "200", "--json", "number,title") or "[]")
    by_title = {i["title"]: i["number"] for i in open_issues}

    for plugin in sorted(failing):
        title = title_for(plugin)
        body = (f"The nightly check with latest allowed dependencies is failing for `{plugin}`.\n\n"
                f"Latest run: {run_url}\n\nThis issue is updated automatically by `scripts/ci/nightly_report.py`; "
                "it closes itself when the plugin is green again.")
        if title in by_title:
            _gh("issue", "comment", str(by_title[title]), "--repo", args.repo, "--body", f"Still failing: {run_url}")
            print(f"updated #{by_title[title]}: {title}")
        else:
            out = _gh("issue", "create", "--repo", args.repo, "--title", title, "--label", LABEL, "--body", body)
            print(f"opened {out.strip()}: {title}")
    for plugin in sorted(passing):
        title = title_for(plugin)
        if title in by_title:
            _gh("issue", "close", str(by_title[title]), "--repo", args.repo, "--comment", f"Green again: {run_url}")
            print(f"closed #{by_title[title]}: {title}")
    if args.dependabot:
        report_dependabot(args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
