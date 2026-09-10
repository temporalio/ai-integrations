"""nightly_report.py maps workflow job names to (lane, plugin) pairs; ci.yml owns those names."""

from __future__ import annotations

from pathlib import Path

import nightly_report
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _job(name: str, conclusion: str | None) -> dict:
    return {"name": name, "conclusion": conclusion}


def test_classify_separates_lanes_and_plugins() -> None:
    failing, passing = nightly_report.classify(
        [
            _job("Python (openai_agents) / openai_agents (ubuntu-latest, py3.14)", "success"),
            _job("Python (openai_agents) / openai_agents (ubuntu-latest, py3.10)", "failure"),
            _job("Python (lowest-direct) (openai_agents) / openai_agents (ubuntu-latest, py3.10)", "failure"),
            _job("Python (lowest-direct) (mcp) / mcp (macos-latest, py3.14)", "success"),
            _job("Python (mcp) / matrix", "success"),
            _job("Python (mcp) / mcp (windows-latest, py3.14)", None),
            _job("Conventions and tooling tests", "failure"),
            _job("ci-status", "failure"),
        ]
    )
    assert failing == {("latest", "openai_agents"), ("lowest-direct", "openai_agents")}
    assert passing == {("lowest-direct", "mcp"), ("latest", "mcp")}


def test_lane_names_match_ci_workflow_job_names() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    lane_jobs = {
        job["name"]
        for job in workflow["jobs"].values()
        if isinstance(job, dict) and str(job.get("uses", "")).endswith("_python-plugin.yml")
    }
    assert lane_jobs == set(nightly_report.LANE_KEY), (
        "ci.yml renamed a Python lane job; update JOB_RE/LANE_KEY in scripts/ci/nightly_report.py"
    )
    for lane in lane_jobs:
        match = nightly_report.JOB_RE.match(f"{lane} (fakeplug) / fakeplug (ubuntu-latest, py3.14)")
        assert match is not None and match.group("plugin") == "fakeplug"
