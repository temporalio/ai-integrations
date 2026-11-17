"""Only dependency download failures get bounded retries, in fresh Gradle processes."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import resolve_java_dependencies as resolver

DNS_FAILURE = """Execution failed for task ':resolveDependencies'.
> Could not resolve all files for configuration ':compileClasspath'.
  > Could not GET 'https://repo.maven.apache.org/maven2/example.pom'.
    > repo.maven.apache.org: nodename nor servname provided, or not known
Caused by: java.net.UnknownHostException: repo.maven.apache.org
"""


class FakeGradle:
    def __init__(self, outcomes: list[tuple[int, str]]) -> None:
        self.outcomes = iter(outcomes)
        self.calls: list[list[str]] = []
        self.directories: list[Path] = []

    def __call__(self, command: list[str], *, cwd: Path, **kwargs: object) -> FakeGradleProcess:
        self.calls.append(command)
        self.directories.append(cwd)
        return FakeGradleProcess(*next(self.outcomes))


class FakeGradleProcess:
    def __init__(self, returncode: int, output: str) -> None:
        self.returncode = returncode
        self.stdout = io.StringIO(output)

    def __enter__(self) -> FakeGradleProcess:
        return self

    def __exit__(self, *args: object) -> None:
        self.stdout.close()

    def wait(self) -> int:
        return self.returncode


@pytest.mark.parametrize("cause", [
    "java.net.UnknownHostException: repo.maven.apache.org",
    "nodename nor servname provided, or not known",
    "Temporary failure in name resolution",
    "java.net.SocketTimeoutException: Read timed out",
    "Connection reset",
    "Received status code 408 from server",
    "Received status code 429 from server",
    "Received status code 503 from server",
])
def test_transient_repository_failures_are_retryable(cause: str) -> None:
    assert resolver.is_transient_download_failure(f"Could not GET 'https://repo.maven.apache.org/example.pom'.\n> {cause}")


@pytest.mark.parametrize("failure", [
    "Received status code 401 from server",
    "Received status code 403 from server",
    "Received status code 404 from server",
    "Could not find example:missing:1.0",
    "javax.net.ssl.SSLHandshakeException: PKIX path building failed",
    "Dependency verification failed: checksum mismatch",
    "Dependency lock state for configuration ':compileClasspath' is out of date",
])
def test_permanent_failures_take_precedence_over_a_transient_error(failure: str) -> None:
    assert not resolver.is_transient_download_failure(DNS_FAILURE + failure)


def test_test_failure_mentioning_network_is_not_retried() -> None:
    assert not resolver.is_transient_download_failure("Execution failed for task ':test'.\nUnknownHostException test FAILED")


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_uses_the_native_gradle_wrapper_from_the_plugin_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str,
) -> None:
    directory = tmp_path / "plugin with spaces"
    directory.mkdir()
    fake = FakeGradle([(0, "BUILD SUCCESSFUL\n")])
    monkeypatch.setattr(resolver.sys, "platform", platform)
    monkeypatch.setattr(resolver.subprocess, "Popen", fake)
    assert resolver.resolve(directory, "locked", "", "0.0.0") == 0
    assert fake.directories == [directory.resolve()]
    if platform == "win32":
        assert fake.calls[0][:4] == ["cmd.exe", "/d", "/c", "gradlew.bat"]
        assert "bash" not in fake.calls[0]
    else:
        assert fake.calls[0][:2] == ["bash", str(directory / "gradlew")]


@pytest.mark.parametrize("mode", ["locked", "latest"])
def test_dns_retries_keep_the_dependency_mode_and_never_run_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str,
) -> None:
    fake = FakeGradle([(1, DNS_FAILURE), (1, DNS_FAILURE), (0, "BUILD SUCCESSFUL\n")])
    delays: list[int] = []
    monkeypatch.setattr(resolver.subprocess, "Popen", fake)
    monkeypatch.setattr(resolver.time, "sleep", delays.append)
    assert resolver.resolve(tmp_path, mode, "4.1.1", "0.1.0-RC1") == 0
    assert delays == [10, 30]
    assert len(fake.calls) == 3
    assert fake.calls[0] == fake.calls[1] == fake.calls[2]
    command = fake.calls[0]
    assert "--no-daemon" in command
    assert f"-PdependencyMode={mode}" in command
    assert "-PspringBootVersion=4.1.1" in command
    assert "-PreleaseVersion=0.1.0-RC1" in command
    assert not {"test", "spotlessCheck", "stageDist"}.intersection(command)
    if mode == "locked":
        assert command[-1] == "resolveDependencies"
        assert "--write-locks" not in command and "--refresh-dependencies" not in command
    else:
        assert command[-3:] == ["resolveAndLockAll", "--write-locks", "--refresh-dependencies"]
    output = capsys.readouterr().out
    assert "UnknownHostException" in output and "attempt 3/3" in output and "BUILD SUCCESSFUL" in output


@pytest.mark.parametrize("outcomes,expected_calls,expected_delays,exitcode", [
    ([(7, DNS_FAILURE)] * 3, 3, [10, 30], 7),
    ([(2, "Could not GET 'https://repo.maven.apache.org/example.pom': status code 403")], 1, [], 2),
    ([(3, "Could not resolve dependencies: lock file mismatch")], 1, [], 3),
    ([(0, "BUILD SUCCESSFUL")], 1, [], 0),
    ([(-15, DNS_FAILURE)], 1, [], -15),
])
def test_resolution_stops_on_success_permanent_error_cancellation_or_retry_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    outcomes: list[tuple[int, str]], expected_calls: int, expected_delays: list[int], exitcode: int,
) -> None:
    fake = FakeGradle(outcomes)
    delays: list[int] = []
    monkeypatch.setattr(resolver.subprocess, "Popen", fake)
    monkeypatch.setattr(resolver.time, "sleep", delays.append)
    assert resolver.resolve(tmp_path, "locked", "", "0.0.0") == exitcode
    assert len(fake.calls) == expected_calls
    assert delays == expected_delays


def test_retry_starts_a_new_wrapper_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the real subprocess path and combined stdout/stderr, with no Java or network.
    (tmp_path / "gradlew").write_text('''#!/usr/bin/env bash
count=0
if [ -f attempts ]; then read -r count < attempts; fi
count=$((count + 1))
printf '%s\\n' "$count" > attempts
if [ "$count" -lt 3 ]; then
  printf '%s\\n' "Could not resolve dependencies: java.net.UnknownHostException: repo.maven.apache.org" >&2
  exit 1
fi
''')
    (tmp_path / "gradlew.bat").write_text('''@echo off
set /a count=0
if exist attempts set /p count=<attempts
set /a count+=1
>attempts echo %count%
if %count% LSS 3 (
  1>&2 echo Could not resolve dependencies: java.net.UnknownHostException: repo.maven.apache.org
  exit /b 1
)
exit /b 0
''')
    delays: list[int] = []
    monkeypatch.setattr(resolver.time, "sleep", delays.append)
    assert resolver.resolve(tmp_path, "locked", "", "0.0.0") == 0
    assert (tmp_path / "attempts").read_text().strip() == "3"
    assert delays == [10, 30]
