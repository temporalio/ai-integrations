#!/usr/bin/env python3
"""Open, update, or close one GitHub issue per failing (plugin, lane) after a nightly run.

Reads the jobs of the current workflow run through `gh api`, maps reusable
workflow job names such as `Python (openai_agents) / openai_agents (ubuntu-latest, py3.14)`
or `Python (lowest-direct, advisory) (openai_agents) / ...` to a (lane, plugin)
pair, and keeps exactly one open issue per failing pair labelled `nightly`.
Passing pairs with an open issue get a comment and are closed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

JOB_RE = re.compile(r"^(?P<lane>Python(?: \(lowest-direct, advisory\))?) \((?P<plugin>[^)]+)\) / ")
LANE_KEY = {"Python": "latest", "Python (lowest-direct, advisory)": "lowest-direct"}
LABEL = "nightly"
FAILED = {"failure", "timed_out"}


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


def classify(jobs: list[dict]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    failing: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for job in jobs:
        m = JOB_RE.match(job.get("name", ""))
        if not m or not job.get("conclusion"):
            continue
        key = (LANE_KEY.get(m.group("lane"), m.group("lane")), m.group("plugin"))
        seen.add(key)
        if job["conclusion"] in FAILED:
            failing.add(key)
    return failing, seen - failing


def title_for(lane: str, plugin: str) -> str:
    return f"nightly: {plugin} failing ({lane})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--dry-run", action="store_true")
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
        return 0

    subprocess.run(["gh", "label", "create", LABEL, "--repo", args.repo, "--force", "--color", "B60205",
                    "--description", "Nightly dependency lanes (latest / lowest-direct) failing"], check=False, capture_output=True)
    open_issues = json.loads(_gh("issue", "list", "--repo", args.repo, "--label", LABEL, "--state", "open", "--limit", "200", "--json", "number,title") or "[]")
    by_title = {i["title"]: i["number"] for i in open_issues}

    for lane, plugin in sorted(failing):
        title = title_for(lane, plugin)
        body = (f"The nightly `{lane}` dependency lane is failing for `{plugin}`.\n\n"
                f"Latest run: {run_url}\n\nThis issue is updated automatically by `scripts/ci/nightly_report.py`; "
                "it closes itself when the lane is green again.")
        if title in by_title:
            _gh("issue", "comment", str(by_title[title]), "--repo", args.repo, "--body", f"Still failing: {run_url}")
            print(f"updated #{by_title[title]}: {title}")
        else:
            out = _gh("issue", "create", "--repo", args.repo, "--title", title, "--label", LABEL, "--body", body)
            print(f"opened {out.strip()}: {title}")
    for lane, plugin in sorted(passing):
        title = title_for(lane, plugin)
        if title in by_title:
            _gh("issue", "close", str(by_title[title]), "--repo", args.repo, "--comment", f"Green again: {run_url}")
            print(f"closed #{by_title[title]}: {title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
