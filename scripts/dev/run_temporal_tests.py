#!/usr/bin/env python3
"""Run a plugin's tests against a dev server owned by this invocation."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

CLI_VERSION = "1.10.1"
# Archive digests from the corresponding temporalio/cli GitHub release.
CLI_DIGESTS = {
    "darwin_amd64": "8ec6c513ad2177de4eb15d2cb17a4fe5070e6078afd7f80bdfcc37fae459d3c4",
    "darwin_arm64": "94f7f982b26438e8abbde6000ba471b694be6a2413b31f6162f87a9f9c48c738",
    "linux_amd64": "4c1500e2ce2d4de136e5fe8852d9b7f82eb62f19b6b383b3e0c040519a9d20b1",
    "linux_arm64": "2f748e9411b37c9cabb57f1f08e37765388c12b5534432c7aea28b593df9145d",
    "windows_amd64": "36a9c7355cce5c8bcf9469c1a679bef764545c62f154570b2a35e2a8fc61f4e2",
    "windows_arm64": "4665694b5eda0ebf786a12d256500d147e1a158d621e38e05d5d91ca3f190841",
}
TEST_ADDRESS = "TEMPORAL_TEST_ADDRESS"
STARTUP_TIMEOUT = 30
POLL_INTERVAL = 0.2
_termination_signal = 0


def cancel_on_sigterm(signum: int, _frame: object) -> None:
    # Defer cancellation until the launcher has registered its child processes.
    global _termination_signal
    _termination_signal = signum


def check_cancellation() -> None:
    if _termination_signal:
        raise SystemExit(128 + _termination_signal)


def download_cli() -> Path:
    """Install just the verified executable into a per-user cache."""
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(machine, machine)
    target = f"{platform.system().lower()}_{arch}"
    digest = CLI_DIGESTS.get(target)
    if digest is None:
        raise ValueError(
            f"No pinned Temporal CLI {CLI_VERSION} for {target}. "
            "Run the test runner directly with TEMPORAL_TEST_ADDRESS to use a running server."
        )
    cache_base = os.environ.get("LOCALAPPDATA" if os.name == "nt" else "XDG_CACHE_HOME")
    cache = Path(cache_base or Path.home() / ".cache") / "ai-integrations" / "temporal-cli" / digest
    executable = "temporal.exe" if os.name == "nt" else "temporal"
    destination = cache / executable
    if destination.is_file():
        return destination

    filename = f"temporal_cli_{CLI_VERSION}_{target}.tar.gz"
    url = f"https://github.com/temporalio/cli/releases/download/v{CLI_VERSION}/{filename}"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        archive = Path(directory) / filename
        for attempt in range(1, 4):
            check_cancellation()
            try:
                with urllib.request.urlopen(url, timeout=30) as response, archive.open("wb") as output:
                    while chunk := response.read(64 * 1024):
                        check_cancellation()
                        output.write(chunk)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.IncompleteRead) as error:
                check_cancellation()
                if attempt == 3 or (
                    isinstance(error, urllib.error.HTTPError)
                    and error.code not in (408, 429) and error.code < 500
                ):
                    raise
                print(f"Temporal CLI download failed ({error}). Retrying.", flush=True)
                time.sleep(1)
        with archive.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                raise ValueError(f"Temporal CLI archive checksum mismatch for {filename}")
        staged = Path(directory) / executable
        with tarfile.open(archive) as package:
            member = next((entry for entry in package if entry.name == executable and entry.isfile()), None)
            if member is None:
                raise ValueError(f"Expected a regular file named {executable} in {filename}")
            binary = package.extractfile(member)
            assert binary is not None
            with binary, staged.open("wb") as output:
                shutil.copyfileobj(binary, output)
        staged.chmod(0o755)
        try:
            staged.replace(destination)
        except PermissionError:
            # Windows may lock the binary already installed by another invocation.
            if not destination.is_file():
                raise
    return destination


def server_healthy(cli: Path, address: str) -> bool:
    try:
        result = subprocess.run(
            [str(cli), "operator", "cluster", "health", "--address", address,
             "--disable-config-file", "--disable-config-env"],
            capture_output=True, timeout=2, check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    finally:
        check_cancellation()
    return result.returncode == 0 and result.stdout.strip() == b"SERVING"


def ready(cli: Path, server: subprocess.Popen, host: str, port: int, address: str) -> bool:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while server.poll() is None and time.monotonic() < deadline:
        check_cancellation()
        try:
            with socket.create_connection((host, port), timeout=POLL_INTERVAL):
                pass
        except OSError:
            pass
        else:
            if server_healthy(cli, address) and server.poll() is None:
                return True
        time.sleep(POLL_INTERVAL)
    return False


def stop_tests(process: subprocess.Popen) -> None:
    """Stop the test runner and its workers before starting the next attempt."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/f", "/t", "/pid", str(process.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_tests(cli: Path, server: subprocess.Popen, address: str, command: list[str]) -> tuple[int, str | None]:
    """Return the test status and any independently confirmed outage."""
    env = {**os.environ, TEST_ADDRESS: address}
    with subprocess.Popen(command, env=env, start_new_session=os.name != "nt") as tests:
        try:
            while (status := tests.poll()) is None:
                check_cancellation()
                if (server_status := server.poll()) is not None:
                    return 1, f"Temporal dev server exited with status {server_status} during tests"
                time.sleep(POLL_INTERVAL)
            check_cancellation()
            if status > 0 and status not in (130, 0xC000013A) and not server_healthy(cli, address):
                time.sleep(1)
                if not server_healthy(cli, address):
                    return status, f"Temporal dev server at {address} became unreachable during tests"
            return status, None
        finally:
            stop_tests(tests)


def test_attempt(cli: Path, host: str, flags: list[str], command: list[str], log: Path) -> tuple[int, str | None]:
    for startup_attempt in range(1, 4):
        check_cancellation()
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.create_server((host, 0), family=family) as listener:
            port = listener.getsockname()[1]
        address = f"[{host}]:{port}" if family == socket.AF_INET6 else f"{host}:{port}"
        with log.open("wb") as output, subprocess.Popen(
            [str(cli), "server", "start-dev", "--headless", "--ip", host, "--port", str(port),
             "--log-level", "warn", "--disable-config-file", "--disable-config-env", *flags],
            stdout=output, stderr=subprocess.STDOUT,
        ) as server:
            try:
                server_ready = ready(cli, server, host, port, address)
                check_cancellation()
                if server_ready:
                    print(f"Temporal dev server ready at {address} (PID {server.pid})", flush=True)
                    return run_tests(cli, server, address, command)
                timed_out = server.poll() is None
                reason = "readiness timed out" if timed_out else f"startup exited with status {server.returncode}"
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill()
        check_cancellation()
        log_text = log.read_text(encoding="utf-8", errors="replace")
        print(f"Temporal dev server {reason} (startup attempt {startup_attempt}/3)", file=sys.stderr)
        bind_collision = "address already in use" in log_text or "Only one usage of each socket address" in log_text
        if startup_attempt == 3 or not (timed_out or bind_collision):
            return 1, None
        print(log_text, file=sys.stderr, end="", flush=True)
        time.sleep(1)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--typescript-helper-args", type=json.loads)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide the test command after --")

    cli = download_cli()
    host = os.environ.get("TEMPORAL_TEST_IP") or "127.0.0.1"
    flags = args.server_arg + (args.typescript_helper_args or [])
    with tempfile.TemporaryDirectory(prefix="temporal-test-") as directory:
        for attempt in (1, 2):
            log = Path(directory) / f"server-{attempt}.log"
            status, outage = test_attempt(cli, host, flags, command, log)
            check_cancellation()
            if status == 0:
                return 0
            if outage:
                print(outage, file=sys.stderr, flush=True)
            print(log.read_text(encoding="utf-8", errors="replace"), file=sys.stderr, end="", flush=True)
            if not outage or attempt == 2:
                return status if status >= 0 else 128 - status
            print("Retrying tests once with a fresh Temporal dev server.", file=sys.stderr, flush=True)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    if os.name == "posix":
        signal.signal(signal.SIGTERM, cancel_on_sigterm)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, http.client.IncompleteRead, tarfile.TarError, ValueError) as exc:
        sys.exit(f"Temporal test launcher: {exc}")
