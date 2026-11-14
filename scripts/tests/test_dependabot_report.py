"""Dependabot failures notify once, survive missing history, and close on recovery."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import nightly_report
import pytest
from _common import Plugin


def run(number: int, dependency: str | None = "mcp", *, directory: str = "/python/fakeplug",
        manager: str = "uv", conclusion: str | None = "failure", status: str = "completed", **extra: object) -> dict:
    target = f" for {dependency}" if dependency else ""
    return {
        "id": number, "name": f"{manager} in {directory}{target} - Update #{number}",
        "html_url": f"https://github.com/owner/repo/actions/runs/{number}",
        "status": status, "conclusion": conclusion, "run_attempt": 1,
        "created_at": "2026-11-01T12:00:00Z", "run_started_at": "2026-11-01T12:00:00Z", **extra,
    }


def alert(package: str = "mcp", manifest: str = "python/fakeplug/uv.lock", ecosystem: str = "pip", **extra: object) -> dict:
    return {"dependency": {"package": {"ecosystem": ecosystem, "name": package}, "manifest_path": manifest},
            "state": "open", "created_at": "2026-10-01T00:00:00Z", **extra}


def test_latest_result_is_per_dependency_and_directory() -> None:
    failed = run(1)
    runs = [run(4, conclusion="success", directory="/python/another"),
            run(3, None, conclusion="success"), run(2, "pydantic", conclusion="success"), failed]
    assert nightly_report.dependabot_failures(runs, [alert()], {"python/fakeplug"}) == {"python/fakeplug": [failed]}
    assert nightly_report.dependabot_failures(runs + [run(5, conclusion="success")], [alert()], {"python/fakeplug"}) == {}


@pytest.mark.parametrize("status,conclusion", [
    ("queued", None), ("in_progress", None), ("completed", "cancelled"),
    ("completed", "neutral"), ("completed", "skipped"),
])
def test_unfinished_or_cancelled_retry_cannot_clear_a_failure(status: str, conclusion: str | None) -> None:
    failed = run(1)
    runs = [failed, run(2, status=status, conclusion=conclusion)]
    assert nightly_report.dependabot_failures(runs, [alert()], {"python/fakeplug"}) == {"python/fakeplug": [failed]}


def test_rerun_order_uses_the_attempt_start_time() -> None:
    runs = [run(2), run(1, conclusion="success", run_attempt=2, run_started_at="2026-11-02T00:00:00Z")]
    assert nightly_report.dependabot_failures(runs, [alert()], {"python/fakeplug"}) == {}


@pytest.mark.parametrize("conclusion", sorted(nightly_report.DEPENDABOT_FAILED))
def test_reports_error_conclusions(conclusion: str) -> None:
    failed = run(1, conclusion=conclusion)
    assert nightly_report.dependabot_failures([failed], [alert()], {"python/fakeplug"}) == {"python/fakeplug": [failed]}


@pytest.mark.parametrize("alerts", [
    [], [alert(state="fixed")], [alert(state="dismissed")], [alert(package="another")],
    [alert(manifest="python/another/uv.lock")], [alert(created_at="2026-11-02T00:00:00Z")],
])
def test_security_failure_requires_a_matching_open_alert_from_before_the_attempt(alerts: list[dict]) -> None:
    assert nightly_report.dependabot_failures([run(1)], alerts, {"python/fakeplug"}) == {}


def test_python_package_names_are_normalized_in_alerts_and_retries() -> None:
    failed = run(1, "Some_Package")
    alerts = [alert("some.package")]
    assert nightly_report.dependabot_failures([failed], alerts, {"python/fakeplug"}) == {"python/fakeplug": [failed]}
    assert nightly_report.dependabot_failures(
        [failed, run(2, "some-package", manager="pip", conclusion="success")], alerts, {"python/fakeplug"},
    ) == {}


@pytest.mark.parametrize("manager,directory,package,manifest,ecosystem,plugin", [
    ("npm_and_yarn", "/typescript/fakeplug", "@hono/node-server", "typescript/fakeplug/pnpm-lock.yaml", "npm", "typescript/fakeplug"),
    ("gradle", "/java/fakeplug", "org.example:core", "java/fakeplug/gradle/dependency-locks/boot.lockfile", "maven", "java/fakeplug"),
    ("gomod", "/go/fakeplug", "golang.org/x/net", "go/fakeplug/go.mod", "go", "go/fakeplug"),
    ("github_actions", "/", "actions/checkout", ".github/workflows/ci.yml", "github-actions", "repository"),
    ("uv", "/scripts", "packaging", "scripts/uv.lock", "pip", "scripts"),
])
def test_maps_ecosystems_nested_manifests_and_repository_failures(
    manager: str, directory: str, package: str, manifest: str, ecosystem: str, plugin: str,
) -> None:
    failed = run(1, package, directory=directory, manager=manager)
    assert nightly_report.dependabot_failures([failed], [alert(package, manifest, ecosystem)], {plugin}) == {plugin: [failed]}


def test_version_update_globs_cover_all_matching_plugins_without_security_alerts() -> None:
    failed = run(1, None, directory="/typescript/*", manager="npm_and_yarn")
    plugins = {"typescript/one", "typescript/two", "python/one"}
    assert nightly_report.dependabot_failures([failed], [], plugins) == {
        "typescript/one": [failed], "typescript/two": [failed],
    }


@pytest.mark.parametrize("name", ["New Dependabot job format - Update #1", None, ""])
def test_unrecognized_job_names_are_reported_at_repository_level(name: str | None) -> None:
    failed = run(1, name=name)
    assert nightly_report.dependabot_failures([failed], [], set()) == {"repository": [failed]}


@pytest.fixture
def reporter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    data = {"runs": [run(1)], "alerts": [alert()], "issues": [], "writes": [], "api": [], "fail_api": False}
    monkeypatch.setattr(nightly_report, "discover_plugins", lambda _: {"python": [Plugin("python", "fakeplug", tmp_path)]})

    def gh(*args: str) -> str:
        if args[0] == "api":
            data["api"].append(args)
            assert args[-1] == "--paginate"
            endpoint = args[1]
            if endpoint.endswith("/actions/workflows?per_page=100"):
                return json.dumps({"workflows": []}) + "\n" + json.dumps({"workflows": [
                    {"id": 99, "path": nightly_report.DEPENDABOT_PATH},
                    {"id": 100, "path": "dynamic/dependabot/update-graph"},
                ]})
            if endpoint.endswith("/actions/workflows/99/runs?per_page=100"):
                return json.dumps({"workflow_runs": data["runs"][:1]}) + json.dumps({"workflow_runs": data["runs"][1:]})
            if "/dependabot/alerts?" in endpoint:
                if data["fail_api"]:
                    raise subprocess.CalledProcessError(1, args)
                return json.dumps(data["alerts"][:1]) + "\n" + json.dumps(data["alerts"][1:])
            if endpoint.endswith("/actions/runs/123/jobs?per_page=100"):
                return '{"jobs": []}'
            pytest.fail(f"Unexpected API endpoint: {endpoint}")
        if args[:2] == ("issue", "list"):
            assert args[args.index("--label") + 1] == nightly_report.DEPENDABOT_LABEL
            return json.dumps(data["issues"])
        body = None
        if "--body-file" in args:
            body = Path(args[args.index("--body-file") + 1]).read_text()
        data["writes"].append((args[:2], body))
        if args[:2] == ("issue", "create"):
            data["issues"].append({"number": 42, "title": args[args.index("--title") + 1], "body": body})
        elif args[:2] == ("issue", "edit"):
            data["issues"][0]["body"] = body
        elif args[:2] == ("issue", "close"):
            data["issues"].clear()
        return ""

    monkeypatch.setattr(nightly_report, "_gh", gh)
    return data


def test_creates_a_separate_issue_with_failed_run_links_and_state(reporter: dict) -> None:
    nightly_report.report_dependabot("owner/repo")
    issue, = reporter["issues"]
    assert issue["title"] == "dependabot: python/fakeplug updates failing"
    assert run(1)["html_url"] in issue["body"]
    assert nightly_report._dependabot_state(issue["body"])[0]["id"] == 1
    assert [args for args, _ in reporter["writes"]] == [("label", "create"), ("issue", "create")]


@pytest.mark.parametrize("repeat", ["same", "new-run", "new-dependency", "new-conclusion", "success", "alert-resolved"])
def test_repeated_reports_only_notify_when_failures_change(reporter: dict, repeat: str) -> None:
    nightly_report.report_dependabot("owner/repo")
    reporter["writes"].clear()
    if repeat == "new-run":
        reporter["runs"].append(run(2))
    elif repeat == "new-dependency":
        reporter["runs"].append(run(2, "another"))
        reporter["alerts"].append(alert("another"))
    elif repeat == "new-conclusion":
        reporter["runs"].append(run(2, conclusion="timed_out"))
    elif repeat == "success":
        reporter["runs"].append(run(2, conclusion="success"))
    elif repeat == "alert-resolved":
        reporter["alerts"].clear()
    nightly_report.report_dependabot("owner/repo")
    actions = [args for args, _ in reporter["writes"] if args[0] == "issue"]
    expected = {
        "same": [], "new-run": [("issue", "edit")],
        "new-dependency": [("issue", "edit"), ("issue", "comment")],
        "new-conclusion": [("issue", "edit"), ("issue", "comment")],
        "success": [("issue", "close")], "alert-resolved": [("issue", "close")],
    }
    assert actions == expected[repeat]


def test_missing_actions_history_preserves_failure_until_alert_resolution(reporter: dict) -> None:
    nightly_report.report_dependabot("owner/repo")
    reporter["runs"].clear()
    reporter["writes"].clear()
    nightly_report.report_dependabot("owner/repo")
    assert not any(args[0] == "issue" for args, _ in reporter["writes"])
    reporter["alerts"].clear()
    nightly_report.report_dependabot("owner/repo")
    assert ("issue", "close") in [args for args, _ in reporter["writes"]]


def test_partial_recovery_keeps_the_issue_open_and_notifies_of_the_change(reporter: dict) -> None:
    reporter["runs"].append(run(2, "another"))
    reporter["alerts"].append(alert("another"))
    nightly_report.report_dependabot("owner/repo")
    reporter["writes"].clear()
    reporter["runs"].append(run(3, conclusion="success"))
    nightly_report.report_dependabot("owner/repo")
    assert [args for args, _ in reporter["writes"] if args[0] == "issue"] == [("issue", "edit"), ("issue", "comment")]
    assert [item["id"] for item in nightly_report._dependabot_state(reporter["issues"][0]["body"])] == [2]


def test_live_successful_rerun_replaces_the_saved_failure(reporter: dict) -> None:
    nightly_report.report_dependabot("owner/repo")
    reporter["writes"].clear()
    reporter["runs"][0] = run(1, conclusion="success", run_attempt=2)
    nightly_report.report_dependabot("owner/repo")
    assert ("issue", "close") in [args for args, _ in reporter["writes"]]


@pytest.mark.parametrize("body", ["Human notes", "<!-- dependabot-failures: invalid -->", "<!-- dependabot-failures: [null] -->"])
def test_unknown_or_malformed_issue_state_does_not_cause_automatic_closure(reporter: dict, body: str) -> None:
    reporter["issues"].append({"number": 42, "title": "dependabot: python/fakeplug updates failing", "body": body})
    reporter["runs"].clear()
    reporter["alerts"].clear()
    nightly_report.report_dependabot("owner/repo")
    assert reporter["writes"] == []


def test_failed_api_read_does_not_mutate_issues(reporter: dict) -> None:
    nightly_report.report_dependabot("owner/repo")
    reporter["writes"].clear()
    reporter["fail_api"] = True
    with pytest.raises(subprocess.CalledProcessError):
        nightly_report.report_dependabot("owner/repo")
    assert reporter["writes"] == []


def test_cli_dry_run_reads_all_dependabot_pages_without_writes(reporter: dict, capsys: pytest.CaptureFixture[str]) -> None:
    reporter["runs"].insert(0, run(2, conclusion="success"))
    assert nightly_report.main(["--repo", "owner/repo", "--run-id", "123", "--dependabot", "--dry-run"]) == 0
    assert "dependabot failing: []" in capsys.readouterr().out
    assert reporter["writes"] == []
