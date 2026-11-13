#!/usr/bin/env python3
"""Enforce the dependency cooldown on a .NET plugin's committed packages.lock.json files.

NuGet ships no client-side cooldown yet (NuGet/Home#14983 is accepted, update-time only and
unreleased), so this script enforces the policy the plugin's nuget.config declares in that spec's
syntax: `minPublishAgeHours` on the nuget.org source and `<minPublishAgeExceptions>` patterns.

For every `resolved` version in every committed lock, the registration leaf on nuget.org must say
the version is listed and was published at least `minPublishAgeHours` ago. Exempt patterns skip
the age check but not the listing check: an unlisted version is the yanked analogue. Anything but a
200 answer fails closed. NuGet resolves a reference to the lowest version that satisfies it, so a
lock that passes today still passes tomorrow; the check matters when someone bumps a version.

Usage:
    check_dotnet_deps.py --plugin-dir dotnet/<name> [--registry-dir DIR]

`--registry-dir` reads `<id>/<version>.json` registration leaves from disk instead of nuget.org
(tests); a missing file is the 404 case.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

NUGET_ORG = "https://api.nuget.org/v3/index.json"
REGISTRATION = "https://api.nuget.org/v3/registration5-gz-semver2/{id}/{version}.json"
LOCK_NAME = "packages.lock.json"


class CooldownError(ValueError):
    pass


def read_policy(config_path: Path) -> tuple[int, list[str]]:
    """Return (minPublishAgeHours, lower-cased exception patterns) from the plugin's nuget.config."""
    if not config_path.is_file():
        raise CooldownError(f"{config_path} is missing")
    root = ET.parse(config_path).getroot()
    sources = root.find("packageSources")
    if sources is None:
        raise CooldownError(f"{config_path}: no <packageSources> section")
    hours: int | None = None
    for source in sources.findall("add"):
        if source.get("value") != NUGET_ORG:
            continue
        raw = source.get("minPublishAgeHours", "")
        if not raw.isdigit() or int(raw) <= 0:
            raise CooldownError(f"{config_path}: the nuget.org source must set minPublishAgeHours to a positive integer")
        hours = int(raw)
    if hours is None:
        raise CooldownError(f"{config_path}: the nuget.org source {NUGET_ORG} with minPublishAgeHours is required")
    exceptions = root.find("minPublishAgeExceptions")
    patterns = [] if exceptions is None else [entry.get("pattern", "") for entry in exceptions.findall("package")]
    return hours, [pattern.lower() for pattern in patterns if pattern]


def lock_files(plugin_dir: Path) -> list[Path]:
    """Committed locks under the plugin; outside git, every lock on disk."""
    try:
        out = subprocess.run(["git", "-C", str(plugin_dir), "ls-files", "--", "."],
                             check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(plugin_dir.rglob(LOCK_NAME))
    return sorted(plugin_dir / line for line in out.splitlines() if line.strip().rsplit("/", 1)[-1] == LOCK_NAME)


def locked_versions(lock_path: Path) -> set[tuple[str, str]]:
    """Every (id, resolved version) a lock pins from a registry; project references have no version."""
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    found: set[tuple[str, str]] = set()
    for packages in data.get("dependencies", {}).values():
        for package_id, entry in packages.items():
            if entry.get("type") != "Project" and entry.get("resolved"):
                found.add((package_id, str(entry["resolved"])))
    return found


def registration_leaf(package_id: str, version: str, registry_dir: Path | None = None) -> dict:
    """Fetch one version's registration leaf; any answer but 200 (or a missing fixture) fails closed."""
    if registry_dir is not None:
        path = registry_dir / package_id.lower() / f"{version.lower()}.json"
        if not path.is_file():
            raise CooldownError(f"{package_id} {version}: registry answered HTTP 404; refusing to guess")
        return json.loads(path.read_text(encoding="utf-8"))
    url = REGISTRATION.format(id=package_id.lower(), version=version.lower())
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise CooldownError(f"{package_id} {version}: registry answered HTTP {exc.code}; refusing to guess") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise CooldownError(f"{package_id} {version}: registry unreachable ({exc}); refusing to guess") from exc
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return json.loads(payload)


def parse_time(raw: object) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise CooldownError("registration leaf has no published time")
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)  # nuget.org writes seven fractional digits
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def check_version(package_id: str, version: str, leaf: dict, hours: int, patterns: list[str], now: datetime) -> str:
    """One version against the policy: listed always; old enough unless an exception pattern matches."""
    if leaf.get("listed") is False:
        raise CooldownError(f"{package_id} {version} is unlisted on nuget.org (the yanked analogue); move to a listed version")
    if any(fnmatch.fnmatchcase(package_id.lower(), pattern) for pattern in patterns):
        return f"OK: {package_id} {version} is listed and exempt from the cooldown"
    age = now - parse_time(leaf.get("published"))
    if age < timedelta(hours=hours):
        raise CooldownError(
            f"{package_id} {version} was published {age.total_seconds() / 3600:.0f} hours ago; "
            f"the cooldown is {hours} hours (nuget.config minPublishAgeHours)"
        )
    return f"OK: {package_id} {version} is listed and {age.days} days old"


def check(plugin_dir: Path, registry_dir: Path | None = None, now: datetime | None = None) -> list[str]:
    """Return every violation across the plugin's committed locks (empty means the policy holds)."""
    hours, patterns = read_policy(plugin_dir / "nuget.config")
    locks = lock_files(plugin_dir)
    if not locks:
        raise CooldownError(f"{plugin_dir}: no committed {LOCK_NAME}; run `make lock` and commit the result")
    versions = sorted(set().union(*(locked_versions(lock) for lock in locks)))
    print(f"cooldown {hours} hours; exempt {patterns or 'none'}; {len(versions)} locked versions in "
          + ", ".join(str(lock.relative_to(plugin_dir)) for lock in locks))
    now = now or datetime.now(timezone.utc)

    def one(item: tuple[str, str]) -> str | None:
        package_id, version = item
        try:
            print(check_version(package_id, version, registration_leaf(package_id, version, registry_dir), hours, patterns, now))
            return None
        except CooldownError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return [failure for failure in pool.map(one, versions) if failure]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, default=None, help="read registration leaves from <id>/<version>.json files (tests)")
    args = parser.parse_args(argv)
    try:
        failures = check(args.plugin_dir.resolve(), args.registry_dir)
    except (CooldownError, OSError, ET.ParseError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    if failures:
        print(f"FAIL: {len(failures)} locked version(s) violate the dependency cooldown:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("OK: every locked version is listed and older than the cooldown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
