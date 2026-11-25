"""A test run retries only a verified outage of its own Temporal server."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import run_temporal_tests as launcher


@pytest.mark.parametrize(
    ("shutdown_status", "health", "retry"),
    [
        (0, [], False),
        (1, [True], False),
        (1, [False, True], False),
        (1, [False, False], True),
        (130, [], False),
    ],
)
def test_failed_tests_require_sustained_server_outage(monkeypatch, shutdown_status, health, retry):
    tests = MagicMock()
    tests.__enter__.return_value = tests
    tests.poll.return_value = shutdown_status
    probe = Mock(side_effect=health)
    monkeypatch.setattr(launcher.subprocess, "Popen", Mock(return_value=tests))
    monkeypatch.setattr(launcher, "server_healthy", probe)
    monkeypatch.setattr(launcher.time, "sleep", Mock())

    status, outage = launcher.run_tests(Path("temporal"), Mock(), "127.0.0.1:1234", ["test"])

    assert status == shutdown_status
    assert bool(outage) == retry
    assert probe.call_count == len(health)
    assert launcher.subprocess.Popen.call_args.kwargs["env"][launcher.TEST_ADDRESS] == "127.0.0.1:1234"


def test_server_exit_stops_the_active_test_runner(monkeypatch):
    tests = MagicMock()
    tests.__enter__.return_value = tests
    tests.poll.return_value = None
    stop = Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", Mock(return_value=tests))
    monkeypatch.setattr(launcher, "stop_tests", stop)
    server = Mock()
    server.poll.return_value = 23

    status, outage = launcher.run_tests(Path("temporal"), server, "127.0.0.1:1234", ["test"])

    assert status != 0
    assert outage == "Temporal dev server exited with status 23 during tests"
    stop.assert_called_once_with(tests)


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ([(0, None)], 0),
        ([(2, None)], 2),
        ([(1, "server crashed"), (0, None)], 0),
        ([(1, "server crashed"), (2, "server unreachable")], 2),
    ],
)
def test_launcher_runs_at_most_two_attempts(monkeypatch, capsys, outcomes, expected):
    monkeypatch.delenv(launcher.TEST_ADDRESS, raising=False)
    monkeypatch.setattr(sys, "argv", ["launcher", "--", "test", "--flag"])
    monkeypatch.setattr(launcher, "download_cli", Mock(return_value=Path("temporal")))
    seen_logs = []

    def attempt(cli, host, flags, command, log):
        assert command == ["test", "--flag"]
        log.write_text(f"server log {log.name}\n", encoding="utf-8")
        seen_logs.append(log.name)
        return outcomes[len(seen_logs) - 1]

    monkeypatch.setattr(launcher, "test_attempt", attempt)

    assert launcher.main() == expected
    assert seen_logs == [f"server-{index}.log" for index in range(1, len(outcomes) + 1)]
    errors = capsys.readouterr().err
    if len(outcomes) == 2:
        assert "server log server-1.log" in errors
        assert "Retrying tests once" in errors
    else:
        assert "Retrying tests" not in errors


@pytest.mark.parametrize(
    ("variable", "value", "args"),
    [
        (launcher.TEST_ADDRESS, "127.0.0.1:1234", []),
        ("TEMPORAL_SERVICE_ADDRESS", "127.0.0.1:1234", ["--typescript-helper-args", "[]"]),
        ("TEMPORAL_TEST_ENV_CONFIG_SERVER", "true", ["--typescript-helper-args", "[]"]),
    ],
)
def test_external_server_has_no_local_lifecycle(monkeypatch, variable, value, args):
    monkeypatch.delenv(launcher.TEST_ADDRESS, raising=False)
    monkeypatch.setenv(variable, value)
    monkeypatch.setattr(sys, "argv", ["launcher", *args, "--", "test"])
    monkeypatch.setattr(launcher, "download_cli", Mock(side_effect=AssertionError("unexpected CLI download")))
    call = Mock(return_value=2)
    monkeypatch.setattr(launcher.subprocess, "call", call)

    assert launcher.main() == 2
    call.assert_called_once_with(["test"])
