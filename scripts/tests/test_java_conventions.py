"""Every Java plugin must load the shared plugin repository configuration early."""

from __future__ import annotations

from pathlib import Path

import pytest
from check_conventions import Checker
from conftest import commit_all


@pytest.fixture
def java_repo(repo: Path) -> Path:
    plugin = repo / "java/example"
    plugin.mkdir(parents=True)
    (plugin / "plugin.toml").write_text('''[plugin]
name = "example"
language = "java"
coordinate = "io.temporal:example"
root-api = "io.temporal.example"
registry = "maven"
maturity = "pre-release"
[ci]
runtime-versions = ["17", "25"]
[smoke]
imports = ["io.temporal.example.Public"]
''')
    for name in ("build.gradle", "README.md", "gradlew", "gradlew.bat",
                 "gradle/wrapper/gradle-wrapper.jar", "gradle/dependency-locks/dependencies.lockfile"):
        path = plugin / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (plugin / "gradle/wrapper/gradle-wrapper.properties").write_text(
        "distributionSha256Sum=" + "0" * 64 + "\n"
    )
    (plugin / "settings.gradle").write_text("""pluginManagement {
    apply from: '../_shared/settings.gradle'
}

rootProject.name = 'example'
""")
    (plugin / "LICENSE").write_bytes((repo / "LICENSE").read_bytes())
    shared = repo / "java/_shared/settings.gradle"
    shared.parent.mkdir(parents=True)
    shared.write_bytes((Path(__file__).parents[2] / "java/_shared/settings.gradle").read_bytes())
    commit_all(repo, "add independent Java plugin")
    return repo


def test_new_java_plugin_uses_shared_repositories(java_repo: Path) -> None:
    assert Checker(java_repo).run() == []


@pytest.mark.parametrize("settings", [
    "rootProject.name = 'example'\n",
    """pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}
""",
    "apply from: '../_shared/settings.gradle'\n",
    """rootProject.name = 'example'
pluginManagement {
    apply from: '../_shared/settings.gradle'
}
""",
    """pluginManagement {
    repositories { gradlePluginPortal() }
    apply from: '../_shared/settings.gradle'
}
""",
])
def test_java_plugin_cannot_bypass_shared_repositories(java_repo: Path, settings: str) -> None:
    (java_repo / "java/example/settings.gradle").write_text(settings)
    violations = Checker(java_repo).run()
    assert len(violations) == 1
    assert "java/example/settings.gradle: must start with the shared pluginManagement block" in violations[0]


def test_java_plugin_requires_shared_settings(java_repo: Path) -> None:
    (java_repo / "java/_shared/settings.gradle").unlink()
    assert Checker(java_repo).run() == [
        "java/_shared/settings.gradle: missing shared plugin repository configuration"
    ]


def test_java_plugin_without_boot_matrix_requires_a_dependency_lock(java_repo: Path) -> None:
    (java_repo / "java/example/gradle/dependency-locks/dependencies.lockfile").unlink()
    commit_all(java_repo, "drop the dependency lock")
    failures = Checker(java_repo).run()
    assert any("dependencies.lockfile" in failure for failure in failures), failures
