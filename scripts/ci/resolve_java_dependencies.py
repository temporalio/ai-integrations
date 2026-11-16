#!/usr/bin/env python3
"""Resolve Java dependencies before offline lint/test/build, retrying transient downloads only."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

RETRY_DELAYS = (10, 30)
RESOLUTION_FAILURE = re.compile(r"Could not (?:resolve|download|get resource|GET|HEAD)\b")
HTTP_STATUS = re.compile(r"(?:status code|HTTP(?:/\d(?:\.\d)?)?)\s*[:=]?\s*(\d{3})\b", re.IGNORECASE)
TRANSIENT_NETWORK = re.compile(
    r"UnknownHostException|SocketTimeoutException|ConnectException|SocketException|"
    r"NoHttpResponseException|ConnectionClosedException|"
    r"nodename nor servname provided, or not known|Name or service not known|"
    r"Temporary failure in name resolution|Read timed out|connect(?:ion)? timed out|"
    r"Connection reset|Connection refused",
    re.IGNORECASE,
)
PERMANENT_FAILURE = re.compile(
    r"Could not find |SSLHandshakeException|PKIX|certificate|checksum|"
    r"Dependency verification failed|Dependency lock state .*out of date",
    re.IGNORECASE,
)


def is_transient_download_failure(output: str) -> bool:
    """Keep auth, missing artifacts, TLS, integrity and lock failures terminal."""
    if not RESOLUTION_FAILURE.search(output) or PERMANENT_FAILURE.search(output):
        return False
    statuses = [int(status) for status in HTTP_STATUS.findall(output)]
    if any(status not in (408, 429) and not 500 <= status < 600 for status in statuses):
        return False
    return bool(statuses or TRANSIENT_NETWORK.search(output))


def resolve(plugin_dir: Path, dependency_mode: str, spring_boot_version: str, version: str) -> int:
    plugin_dir = plugin_dir.resolve()
    command = [
        "bash", str(plugin_dir / "gradlew"), "--no-daemon", "--console=plain", "--stacktrace",
        f"-PdependencyMode={dependency_mode}", f"-PspringBootVersion={spring_boot_version}",
        f"-PreleaseVersion={version}",
    ]
    if dependency_mode == "latest":
        command += ["resolveAndLockAll", "--write-locks", "--refresh-dependencies"]
    else:
        command += ["resolveDependencies"]
    for attempt in range(len(RETRY_DELAYS) + 1):
        # A fresh JVM also clears cached DNS failures and failed Gradle resolutions.
        with subprocess.Popen(command, cwd=plugin_dir, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            assert process.stdout is not None
            lines = []
            for line in process.stdout:
                print(line, end="", flush=True)
                lines.append(line)
            returncode = process.wait()
        if returncode <= 0 or attempt == len(RETRY_DELAYS) or not is_transient_download_failure("".join(lines)):
            return returncode
        delay = RETRY_DELAYS[attempt]
        print(f"::warning::Java dependency download failed; retrying in {delay}s "
              f"(attempt {attempt + 2}/{len(RETRY_DELAYS) + 1})", flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--dependency-mode", choices=("locked", "latest"), default="locked")
    parser.add_argument("--spring-boot-version", default="")
    parser.add_argument("--version", default="0.0.0")
    args = parser.parse_args()
    return resolve(args.plugin_dir, args.dependency_mode, args.spring_boot_version, args.version)


if __name__ == "__main__":
    sys.exit(main())
