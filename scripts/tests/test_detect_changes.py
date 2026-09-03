from __future__ import annotations

from pathlib import Path

import detect_changes
from conftest import commit_all, git, make_python_plugin


def _setup(repo: Path) -> str:
    make_python_plugin(repo, "alpha")
    make_python_plugin(repo, "beta")
    (repo / "python" / "_shared").mkdir(parents=True)
    (repo / "python" / "_shared" / "python.mk").write_text("all:\n")
    (repo / "python" / "README.md").write_text("python\n")
    ts = repo / "typescript" / "gamma"
    ts.mkdir(parents=True)
    (ts / "package.json").write_text("{}")
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")
    (repo / "scripts" / "ci").mkdir(parents=True)
    (repo / "scripts" / "ci" / "x.py").write_text("x = 1\n")
    (repo / "scripts" / "release").mkdir(parents=True)
    (repo / "scripts" / "release" / "y.py").write_text("y = 1\n")
    return commit_all(repo, "base")


def _run(repo: Path, base: str, *touch: str, event: str = "pull_request") -> dict:
    for rel in touch:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(p.read_text() + "\n# change\n" if p.exists() else "new\n")
    commit_all(repo, "change")
    return detect_changes.main(["--repo-root", str(repo), "--event", event, "--base", base, "--dry-run"])


def test_dependency_change_marks_plugin_for_the_lowest_lane(repo: Path) -> None:
    base = _setup(repo)
    r = _run(repo, base, "python/alpha/pyproject.toml")
    assert r["python"] == ["alpha"] and r["python_deps"] == ["alpha"]
    r = _run(repo, base, "python/beta/src/temporalio/contrib/beta/_impl.py")
    assert r["python"] == ["alpha", "beta"] and r["python_deps"] == ["alpha"]  # cumulative diff vs base


def test_plugin_file_selects_only_that_plugin(repo: Path) -> None:
    base = _setup(repo)
    r = _run(repo, base, "python/alpha/src/temporalio/contrib/alpha/_impl.py")
    assert r["python"] == ["alpha"] and r["typescript"] == [] and r["any"] is True and r["mode"] == "diff"


def test_shared_language_file_selects_all_plugins_of_that_language(repo: Path) -> None:
    base = _setup(repo)
    assert _run(repo, base, "python/README.md")["python"] == ["alpha", "beta"]
    assert _run(repo, base, "python/_shared/python.mk")["python"] == ["alpha", "beta"]
    assert _run(repo, base, "python/_shared/python.mk")["typescript"] == []


def test_github_or_scripts_ci_selects_everything(repo: Path) -> None:
    base = _setup(repo)
    r = _run(repo, base, ".github/workflows/ci.yml")
    assert r["python"] == ["alpha", "beta"] and r["typescript"] == ["gamma"] and r["mode"] == "all"
    r = _run(repo, base, "scripts/ci/x.py")
    assert r["mode"] == "all"


def test_other_scripts_select_nothing_and_report_scripts_only(repo: Path) -> None:
    base = _setup(repo)
    r = _run(repo, base, "scripts/release/y.py")
    assert r["python"] == [] and r["any"] is False and r["scripts_only"] is True and r["mode"] == "none"


def test_root_files_select_nothing(repo: Path) -> None:
    base = _setup(repo)
    r = _run(repo, base, "README.md")
    assert r["any"] is False and r["scripts_only"] is False and r["mode"] == "none"


def test_new_plugin_added_in_the_change_is_selected(repo: Path) -> None:
    base = _setup(repo)
    make_python_plugin(repo, "delta")
    commit_all(repo, "add delta")
    r = detect_changes.main(["--repo-root", str(repo), "--event", "pull_request", "--base", base, "--dry-run"])
    assert r["python"] == ["delta"]


def test_push_schedule_dispatch_and_unknown_events_run_everything(repo: Path) -> None:
    base = _setup(repo)
    for event in ("push", "schedule", "workflow_dispatch", "mystery"):
        r = detect_changes.main(["--repo-root", str(repo), "--event", event, "--base", base, "--dry-run"])
        assert r["mode"] == "all" and r["python"] == ["alpha", "beta"], event


def test_unusable_base_falls_back_to_everything(repo: Path) -> None:
    _setup(repo)
    r = detect_changes.main(["--repo-root", str(repo), "--event", "pull_request", "--base", "0" * 40, "--dry-run"])
    assert r["mode"] == "all"


def test_github_output_format(repo: Path, tmp_path: Path) -> None:
    base = _setup(repo)
    out = tmp_path / "out.txt"
    (repo / "python" / "alpha" / "README.md").write_text("# alpha\n")
    commit_all(repo, "readme")
    detect_changes.main(["--repo-root", str(repo), "--event", "pull_request", "--base", base, "--github-output", str(out)])
    text = out.read_text()
    assert 'python=["alpha"]' in text and "typescript=[]" in text and "any=true" in text and "mode=diff" in text
    assert git(repo, "status", "--porcelain") == ""


def test_root_license_fans_out_to_every_plugin(repo: Path) -> None:
    base = _setup(repo)
    (repo / "LICENSE").write_text("MIT\n")
    commit_all(repo, "add license")
    r = _run(repo, base, "LICENSE")
    assert r["python"] == ["alpha", "beta"] and r["typescript"] == ["gamma"]
