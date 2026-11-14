"""npm SemVer and verification of immutable, tested npm tarballs."""

from __future__ import annotations

import base64
import hashlib
import json
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from go_release import GoReleaseError, GoVersion

REGISTRY = "https://registry.npmjs.org"


class NpmReleaseError(ValueError):
    pass


class NpmVersion(GoVersion):
    """Apply npm release policy using the shared semver-backed implementation."""

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except GoReleaseError as exc:
            raise NpmReleaseError(str(exc).replace("Go version", "npm version")) from exc


def fetch_package(coordinate: str, local: Path | None = None) -> dict | None:
    if local is not None:
        return json.loads(local.read_text()) if local.is_file() else None
    try:
        with urllib.request.urlopen(f"{REGISTRY}/{quote(coordinate, safe='@')}", timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise NpmReleaseError(f"npm returned HTTP {exc.code}; refusing to guess") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise NpmReleaseError(f"npm is unreachable ({exc}); refusing to guess") from exc


def fetch_versions(coordinate: str, local: Path | None = None) -> list[NpmVersion]:
    package = fetch_package(coordinate, local)
    if package is None:
        return []
    if package.get("name") != coordinate or not isinstance(package.get("versions"), dict):
        raise NpmReleaseError("npm returned invalid package metadata")
    # Deprecated versions still participate in ordering.
    return [NpmVersion(version) for version in package["versions"]]


def artifact(dist: Path, coordinate: str, version: str) -> Path:
    tarballs = sorted(dist.glob("*.tgz"))
    if len(tarballs) != 1:
        raise NpmReleaseError("expected exactly one tested npm tarball")
    with tarfile.open(tarballs[0], "r:gz") as archive:
        members = [member for member in archive if member.name == "package/package.json"]
        if len(members) != 1 or not members[0].isfile():
            raise NpmReleaseError("tarball must have exactly one package/package.json")
        package = json.load(archive.extractfile(members[0]))
    if package.get("name") != coordinate or package.get("version") != version or package.get("private"):
        raise NpmReleaseError("tarball name, version or private flag does not match the release")
    return tarballs[0]


def verify_artifact(metadata: dict, tarball: Path, coordinate: str, version: str) -> None:
    if metadata.get("name") != coordinate or metadata.get("version") != version or metadata.get("deprecated"):
        raise NpmReleaseError("npm release identity differs or the release is deprecated")
    payload = tarball.read_bytes()
    integrity = "sha512-" + base64.b64encode(hashlib.sha512(payload).digest()).decode()
    dist = metadata.get("dist", {})
    # The clean consumer's npm install verifies downloaded bytes against this integrity.
    if dist.get("integrity") != integrity:
        raise NpmReleaseError("npm integrity differs from the tested tarball; never replace an uploaded version")


def registry_state(coordinate: str, version: str, dist: Path) -> bool:
    tarball = artifact(dist, coordinate, version)
    package = fetch_package(coordinate)
    if package is None:
        return False
    if package.get("name") != coordinate or not isinstance(package.get("versions"), dict):
        raise NpmReleaseError("npm returned invalid package metadata")
    metadata = package["versions"].get(version)
    if metadata is None:
        return False
    verify_artifact(metadata, tarball, coordinate, version)
    return True
