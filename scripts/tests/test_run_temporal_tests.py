"""A test run retries only a verified outage of its own Temporal server."""

from __future__ import annotations

import hashlib
import http.client
import io
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import run_temporal_tests as launcher


@pytest.fixture
def cli_archive(tmp_path, monkeypatch):
    binary = b"test CLI executable"
    executable = "temporal.exe" if os.name == "nt" else "temporal"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as package:
        member = tarfile.TarInfo(executable)
        member.size = len(binary)
        package.addfile(member, io.BytesIO(binary))
    content = archive.getvalue()
    monkeypatch.setenv("LOCALAPPDATA" if os.name == "nt" else "XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(launcher, "CLI_DIGESTS", {"linux_amd64": hashlib.sha256(content).hexdigest()})
    monkeypatch.setattr(launcher.time, "sleep", Mock())
    return binary, content


@pytest.mark.parametrize("error", [ConnectionResetError("connection reset"), http.client.IncompleteRead(b"partial")])
def test_download_restarts_after_a_response_body_failure(monkeypatch, cli_archive, error):
    binary, content = cli_archive
    incomplete = MagicMock()
    incomplete.__enter__.return_value = incomplete
    incomplete.read.side_effect = [b"partial archive", error]
    download = Mock(side_effect=[incomplete, io.BytesIO(content)])
    monkeypatch.setattr(launcher.urllib.request, "urlopen", download)

    assert launcher.download_cli().read_bytes() == binary
    assert download.call_count == 2
    assert launcher.download_cli().read_bytes() == binary
    assert download.call_count == 2


def test_download_checksum_mismatch_is_terminal(monkeypatch, cli_archive):
    download = Mock(return_value=io.BytesIO(b"corrupt archive"))
    monkeypatch.setattr(launcher.urllib.request, "urlopen", download)

    with pytest.raises(ValueError, match="checksum mismatch"):
        launcher.download_cli()
    download.assert_called_once()


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
    monkeypatch.setenv(launcher.TEST_ADDRESS, "127.0.0.1:4567")
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
    monkeypatch.setenv(launcher.TEST_ADDRESS, "127.0.0.1:4567")
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


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal and process-group semantics")
@pytest.mark.parametrize("signal_group", [False, True])
def test_sigterm_stops_server_runner_and_worker(tmp_path, signal_group):
    machine = launcher.platform.machine().lower()
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(machine, machine)
    target = f"{launcher.platform.system().lower()}_{arch}"
    if target not in launcher.CLI_DIGESTS:
        pytest.skip(f"The launcher does not download a CLI for {target}")
    cli = tmp_path / "ai-integrations" / "temporal-cli" / launcher.CLI_DIGESTS[target] / "temporal"
    cli.parent.mkdir(parents=True)
    server_pid = tmp_path / "server.pid"
    server = tmp_path / "cli.py"
    server.write_text(f"""\
import os, signal, socket, sys
from pathlib import Path

if sys.argv[1:3] == ['server', 'start-dev']:
    port = int(sys.argv[sys.argv.index('--port') + 1])
    with socket.create_server(('127.0.0.1', port)):
        Path({str(server_pid)!r}).write_text(str(os.getpid()))
        signal.pause()
else:
    print('SERVING')
""")
    cli.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(server))} "$@"\n')
    cli.chmod(0o755)

    test_pids = tmp_path / "tests.pid"
    runner = tmp_path / "runner.py"
    runner.write_text(f"""\
import os, signal, subprocess, sys
from pathlib import Path

worker = subprocess.Popen([sys.executable, '-c', 'import signal; signal.pause()'])

def stop(*_):
    worker.wait(timeout=3)
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)
tmp = Path({str(test_pids.with_suffix('.tmp'))!r})
tmp.write_text(f'{{os.getpid()}} {{worker.pid}}')
tmp.replace({str(test_pids)!r})
signal.pause()
""")
    log = tmp_path / "launcher.log"
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path), "TEMPORAL_TEST_IP": "127.0.0.1"}
    command = [sys.executable, launcher.__file__, "--", sys.executable, str(runner)]
    with log.open("wb") as output, subprocess.Popen(
        command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
    ) as process:
        try:
            deadline = time.monotonic() + 15
            while not test_pids.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert test_pids.exists(), log.read_text()
            if signal_group:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            assert process.wait(timeout=12) == 128 + signal.SIGTERM, log.read_text()
            assert "Retrying tests" not in log.read_text()
            for pid in (server_pid.read_text(), *test_pids.read_text().split()):
                with pytest.raises(ProcessLookupError):
                    os.kill(int(pid), 0)
        except BaseException:
            if process.poll() is None:
                process.kill()
            for path in (server_pid, test_pids):
                if path.exists():
                    for pid in path.read_text().split():
                        with suppress(ProcessLookupError):
                            os.kill(int(pid), signal.SIGKILL)
            raise
