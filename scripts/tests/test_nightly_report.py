"""nightly_report.py maps workflow job names to plugins; ci.yml owns those names."""

from __future__ import annotations

from pathlib import Path

import nightly_report
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
    assert failing == {"openai_agents"}
    assert passing == {"mcp"}


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
    assert passing == {"langsmith"}


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
    assert passing == {"shared", "go/recovered"}


def test_java_failures_are_aggregated_by_plugin() -> None:
    failing, passing = nightly_report.classify(
        [
            _job("Java (spring-ai) / spring-ai (windows-latest, java25)", "failure"),
            _job("Java (spring-ai) / matrix", "success"),
        ]
    )
    assert failing == {"spring-ai"}
    assert not passing


def test_dotnet_failures_use_the_language_root_key() -> None:
    failing, passing = nightly_report.classify(
        [
            _job(".NET (agent-framework) / agent-framework (windows-latest, dotnet10.0)", "failure"),
            _job(".NET (agent-framework) / matrix", "success"),
            _job(".NET (other) / other (ubuntu-latest, dotnet10.0)", "success"),
        ]
    )
    assert failing == {"dotnet/agent-framework"}
    assert passing == {"dotnet/other"}
    assert nightly_report.title_for("dotnet/agent-framework") == release_tool.nightly_issue_title("dotnet", "agent-framework")
