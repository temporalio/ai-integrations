"""Go version, vanity-path and public module verification for release_tool.py."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from functools import total_ordering
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

from semver import Version

PROXY = "https://proxy.golang.org"


class GoReleaseError(ValueError):
    pass


@total_ordering
@dataclass(frozen=True)
class GoVersion:
    """Canonical Go release SemVer, without the tag's leading v or build metadata."""

    raw: str
    _version: Version = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            version = Version.parse(self.raw)
        except ValueError as exc:
            raise GoReleaseError(f"Go version {self.raw!r} must be canonical SemVer, e.g. 0.1.0 or 0.1.0-rc.1") from exc
        if version.build is not None or str(version) != self.raw:
            raise GoReleaseError(f"Go version {self.raw!r} must be canonical SemVer, e.g. 0.1.0 or 0.1.0-rc.1")
        object.__setattr__(self, "_version", version)

    @property
    def release(self) -> tuple[int, ...]:
        return self._version.major, self._version.minor, self._version.patch

    @property
    def is_prerelease(self) -> bool:
        return self._version.prerelease is not None

    def __str__(self) -> str:
        return self.raw

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, GoVersion):
            return NotImplemented

        return self._version < other._version


def _fetch(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410):
            return None
        raise GoReleaseError(f"HTTP {exc.code} fetching {url}; refusing to guess") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise GoReleaseError(f"cannot fetch {url}: {exc}; refusing to guess") from exc


def fetch_versions(coordinate: str, local: Path | None = None) -> list[GoVersion]:
    escaped = "".join("!" + c.lower() if c.isupper() else c for c in coordinate)
    text = (local.read_text() if local.is_file() else None) if local is not None else _fetch(f"{PROXY}/{quote(escaped, safe='/!')}/@v/list")
    versions = []
    for raw in (text or "").splitlines():
        if not raw.startswith("v"):
            raise GoReleaseError(f"Go proxy returned an invalid version {raw!r}")
        versions.append(GoVersion(raw[1:]))
    return versions


class GoImports(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.imports: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta" and attributes.get("name") == "go-import":
            self.imports.append((attributes.get("content") or "").split())


def check_vanity(coordinate: str, repo: str, plugin_dir: str) -> None:
    url = f"https://{coordinate}?go-get=1"
    parser = GoImports()
    parser.feed(_fetch(url) or "")
    expected = [coordinate, "git", f"https://github.com/{repo}", plugin_dir]
    for record in parser.imports:
        if len(record) == 4 and record[:2] == expected[:2] and record[2].removesuffix(".git") == expected[2] and record[3] == plugin_dir:
            return
    raise GoReleaseError(f"{url} must serve go-import {' '.join(expected)!r}; got {parser.imports}")


def verify_zip(repo_root: Path, plugin_dir: str, coordinate: str, version: str, archive: Path) -> None:
    """Compare the public module's complete file set and bytes with the tested checkout."""
    module_dir = repo_root / plugin_dir
    tracked = subprocess.run(["git", "ls-files", "-z", "--", plugin_dir], cwd=repo_root, check=True, capture_output=True, text=True).stdout
    files = [repo_root / name for name in tracked.split("\0") if name]
    nested = [p.parent for p in files if p.name == "go.mod" and p.parent != module_dir]
    expected = {
        str(p.relative_to(module_dir)): p.read_bytes() for p in files
        if p.is_file() and not p.is_symlink()
        and not set(p.relative_to(module_dir).parts[:-1]).intersection({"vendor", ".git", ".hg", ".svn", ".bzr"})
        and not any(p.is_relative_to(directory) for directory in nested)
    }
    prefix = f"{coordinate}@v{version}/"
    with zipfile.ZipFile(archive) as z:
        names = [entry.filename for entry in z.infolist() if not entry.is_dir()]
        if len(names) != len(set(names)) or any(not name.startswith(prefix) for name in names):
            raise GoReleaseError("Go proxy archive has duplicate files or an unexpected module prefix")
        served = {name[len(prefix):]: z.read(name) for name in names}
    if served.keys() != expected.keys():
        raise GoReleaseError(f"Go proxy archive file set differs: missing {sorted(expected.keys() - served.keys())}, extra {sorted(served.keys() - expected.keys())}")
    changed = [name for name in expected if served[name] != expected[name]]
    if changed:
        raise GoReleaseError(f"Go proxy archive differs from the tested source: {changed}")


def verify_module(repo_root: Path, plugin_dir: str, coordinate: str, version: str, attempts: int = 10, delay: float = 30) -> None:
    if attempts < 1 or delay < 0:
        raise GoReleaseError("module verification needs attempts >= 1 and delay >= 0")
    with tempfile.TemporaryDirectory(prefix="go-release-") as directory:
        root = Path(directory)
        consumer = root / "consumer"
        consumer.mkdir()
        env = dict(os.environ, GOPROXY=PROXY, GOSUMDB="sum.golang.org", GOPRIVATE="", GONOPROXY="", GONOSUMDB="",
                   GOWORK="off", GOTOOLCHAIN="local", GOFLAGS="", GOMODCACHE=str(root / "modcache"))
        query = f"{coordinate}@v{version}"
        for attempt in range(attempts):
            result = subprocess.run(["go", "mod", "download", "-json", query], cwd=consumer, env=env, capture_output=True, text=True, timeout=180)
            if result.returncode == 0:
                break
            if attempt == attempts - 1:
                raise GoReleaseError(f"cannot download {query} from the public proxy: {result.stdout}\n{result.stderr}")
            print(f"::warning::public module is not available yet; retry {attempt + 1}/{attempts}", flush=True)
            time.sleep(delay)
        data = json.loads(result.stdout)
        if data.get("Path") != coordinate or data.get("Version") != f"v{version}" or data.get("Error"):
            raise GoReleaseError(f"Go proxy resolved a different module/version: {data}")
        if not all(data.get(key, "").startswith("h1:") for key in ("Sum", "GoModSum")):
            raise GoReleaseError("public module download is missing checksum database verification")
        verify_zip(repo_root, plugin_dir, coordinate, version, Path(data["Zip"]))
        if Path(data["GoMod"]).read_bytes() != (repo_root / plugin_dir / "go.mod").read_bytes():
            raise GoReleaseError("Go proxy's go.mod differs from the tested source")
        (consumer / "main.go").write_text(f'package main\n\nimport _ "{coordinate}"\n\nfunc main() {{}}\n')
        for command in (["go", "mod", "init", "release-smoke"], ["go", "get", query], ["go", "mod", "tidy"],
                        ["go", "mod", "verify"], ["go", "build", "-mod=readonly", "./..."]):
            subprocess.run(command, cwd=consumer, env=env, check=True, timeout=300)
        print(f"OK: {query} matches the tested source and builds in a clean consumer with {data['Sum']}")
