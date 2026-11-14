"""nightly_report.py maps workflow job names to plugins; ci.yml owns those names."""

from __future__ import annotations

import json
from pathlib import Path

import nightly_report
import pytest
import release_tool
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _job(name: str, conclusion: str | None) -> dict:
    return {"name": name, "conclusion": conclusion}


def test_classify_aggregates_results_by_plugin() -> None:
    failing, passing = nightly_report.classify(
        [
            _job(
                "Python (openai_agents) / openai_agents (ubuntu-latest, py3.15)",
                "success",
            ),
            _job(
                "Python (openai_agents) / openai_agents (ubuntu-latest, py3.11)",
                "failure",
            ),
            _job("Python (mcp) / mcp (macos-latest, py3.15)", "success"),
            _job("Python (mcp) / matrix", "success"),
            _job("Python (mcp) / mcp (windows-latest, py3.15)", None),
            _job("Conventions and tooling tests", "failure"),
            _job("ci-status", "failure"),
        ]
    )
    assert failing == {"python/openai_agents"}
    assert passing == {"python/mcp"}


def test_report_matches_ci_workflow_job_names() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    )
    for language, expected_name in (
        ("python", "Python"),
        ("java", "Java"),
        ("go", "Go"),
        ("typescript", "TypeScript"),
        ("dotnet", ".NET"),
    ):
        names = {
            job["name"]
            for job in workflow["jobs"].values()
            if isinstance(job, dict)
            and str(job.get("uses", "")).endswith(f"_{language}-plugin.yml")
        }
        assert names == {expected_name}, (
            f"ci.yml renamed a {language} job; update JOB_RE in scripts/ci/nightly_report.py"
        )
        match = nightly_report.JOB_RE.match(f"{expected_name} (fakeplug) / matrix")
        assert match is not None and match.group("plugin") == "fakeplug"
        assert match.group("language") == expected_name
        # The release workflows' check-nightly looks the issue up by the title this report gives it.
        (key,), _ = nightly_report.classify([_job(f"{expected_name} (fakeplug) / matrix", "failure")])
        assert nightly_report.title_for(key) == release_tool.nightly_issue_title(language, "fakeplug")
    assert {"python", "java", "go", "typescript", "dotnet"}.issubset(
        workflow["jobs"]["nightly-report"]["needs"]
    )
    report = workflow["jobs"]["nightly-report"]
    assert report["permissions"]["vulnerability-alerts"] == "read"
    assert any(step.get("run") == "python3 scripts/ci/nightly_report.py --dependabot" for step in report["steps"])


def test_typescript_and_python_langsmith_have_separate_nightly_issues() -> None:
    failing, passing = nightly_report.classify(
        [
            _job(
                "TypeScript (langsmith) / langsmith (ubuntu-latest, node24)", "failure"
            ),
            _job("Python (langsmith) / langsmith (ubuntu-latest, py3.15)", "success"),
        ]
    )
    assert failing == {"typescript/langsmith"}
    assert passing == {"python/langsmith"}


def test_classify_go_failures_and_keep_language_identities_separate() -> None:
    failing, passing = nightly_report.classify(
        [
            _job("Python (shared) / shared (ubuntu-latest, py3.15)", "success"),
            _job("Go (shared) / matrix", "success"),
            _job("Go (shared) / shared (ubuntu-latest, go1.26.6)", "failure"),
            _job("Go (shared) / shared (windows-latest, go1.27)", "success"),
            _job("Go (recovered) / recovered (macos-latest, go1.27)", "success"),
            _job("Go (pending) / pending (ubuntu-latest, go1.27)", None),
        ]
    )
    assert failing == {"go/shared"}
    assert passing == {"python/shared", "go/recovered"}


def test_java_failures_are_aggregated_by_plugin() -> None:
    failing, passing = nightly_report.classify(
        [
            _job("Java (spring-ai) / spring-ai (windows-latest, java25)", "failure"),
            _job("Java (spring-ai) / matrix", "success"),
        ]
    )
    assert failing == {"java/spring-ai"}
    assert not passing


def test_dotnet_failures_use_the_language_root_key() -> None:
    failing, passing = nightly_report.classify(
        [
            _job(".NET (microsoft-agent-framework) / microsoft-agent-framework (windows-latest, dotnet10.0)", "failure"),
            _job(".NET (microsoft-agent-framework) / matrix", "success"),
            _job(".NET (other) / other (ubuntu-latest, dotnet10.0)", "success"),
        ]
    )
    assert failing == {"dotnet/microsoft-agent-framework"}
    assert passing == {"dotnet/other"}
    assert nightly_report.title_for("dotnet/microsoft-agent-framework") == release_tool.nightly_issue_title("dotnet", "microsoft-agent-framework")


@pytest.mark.parametrize("conclusion", ["failure", "success"])
@pytest.mark.parametrize("existing_issue", [False, True])
@pytest.mark.parametrize("language", ["Python", "Java"])
def test_report_creates_or_reuses_issue_with_qualified_title(
    monkeypatch: pytest.MonkeyPatch, conclusion: str, existing_issue: bool,
    language: str,
) -> None:
    repo = "temporalio/ai-integrations"
    run_url = f"https://github.com/{repo}/actions/runs/123"
    key = f"{language.lower()}/langsmith"
    title = f"nightly: {key} failing (latest)"
    issues = [{"number": 500, "title": "nightly: typescript/langsmith failing (latest)"}]
    if existing_issue:
        issues.append({"number": 444, "title": title})
    calls: list[tuple[str, ...]] = []

    def gh(*args: str) -> str:
        calls.append(args)
        if args[0] == "api":
            return json.dumps({"jobs": [_job(f"{language} (langsmith) / matrix", conclusion)]})
        if args[:2] == ("issue", "list"):
            return json.dumps(issues)
        return ""

    monkeypatch.setattr(nightly_report, "_gh", gh)
    monkeypatch.setattr(nightly_report.subprocess, "run", lambda *args, **kwargs: None)
    assert nightly_report.main(["--repo", repo, "--run-id", "123"]) == 0
    actions = calls[2:]
    if not existing_issue:
        if conclusion == "failure":
            assert len(actions) == 1
            assert actions[0][:6] == ("issue", "create", "--repo", repo, "--title", title)
        else:
            assert actions == []
        return
    if conclusion == "failure":
        assert actions == [("issue", "comment", "444", "--repo", repo, "--body", f"Still failing: {run_url}")]
    else:
        assert actions == [("issue", "close", "444", "--repo", repo, "--comment", f"Green again: {run_url}")]
