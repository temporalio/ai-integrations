"""NuGet SemVer and verification of immutable, tested NuGet packages against nuget.org."""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from go_release import GoReleaseError, GoVersion

FLAT_CONTAINER = "https://api.nuget.org/v3-flatcontainer"
REGISTRATION = "https://api.nuget.org/v3/registration5-gz-semver2"
SIGNATURE = ".signature.p7s"


class NuGetReleaseError(ValueError):
    pass


class NuGetMismatchError(NuGetReleaseError):
    """nuget.org serves bytes that are not the tested package; retrying cannot change that."""


class NuGetVersion(GoVersion):
    """Canonical SemVer without build metadata, the form nuget.org normalises every version to."""

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except GoReleaseError as exc:
            raise NuGetReleaseError(str(exc).replace("Go version", "NuGet version")) from exc


def _get(url: str) -> bytes | None:
    """GET a nuget.org resource; 404 -> None, anything else fails closed. Registration blobs are gzip-encoded."""
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            payload = response.read()
            encoding = response.headers.get("Content-Encoding", "")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise NuGetReleaseError(f"nuget.org returned HTTP {exc.code} for {url}; refusing to guess") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise NuGetReleaseError(f"nuget.org is unreachable ({exc}); refusing to guess") from exc
    if encoding.lower() == "gzip" or (url.endswith(".json") and payload[:2] == b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    return payload


def fetch_versions(coordinate: str, local: Path | None = None) -> list[NuGetVersion]:
    """Every version in the flat container index, unlisted ones included; 404 means no release yet."""
    if local is not None:
        payload = local.read_bytes() if local.is_file() else None
    else:
        payload = _get(f"{FLAT_CONTAINER}/{coordinate.lower()}/index.json")
    if payload is None:
        return []
    data = json.loads(payload)
    if not isinstance(data, dict) or not isinstance(data.get("versions"), list):
        raise NuGetReleaseError("nuget.org returned an invalid flat container index")
    # A version under our coordinate that is not canonical SemVer is a policy breach, not something to skip.
    return [NuGetVersion(str(version)) for version in data["versions"]]


def unsigned_entries(payload: bytes) -> list[tuple[str, int, int, bytes]]:
    """Zip entries as (name, size, crc, bytes) without the trailing stored repository signature.

    nuget.org appends `.signature.p7s` as the last stored entry of every upload and rewrites nothing
    else, so this is the part of a served package that must equal the tested package. A signature
    anywhere else is not that case and is an error. scripts/ci/smoke_dotnet.py carries the same rule.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        if infos and infos[-1].filename == SIGNATURE and infos[-1].compress_type == zipfile.ZIP_STORED:
            infos = infos[:-1]
        if any(info.filename == SIGNATURE for info in infos):
            raise NuGetReleaseError("package has a signature entry that is not the trailing stored repository signature")
        return [(info.filename, info.file_size, info.CRC, archive.read(info)) for info in infos]


def served_matches(served: bytes, tested: bytes) -> bool:
    return unsigned_entries(served) == unsigned_entries(tested)


def nuspec_identity(payload: bytes) -> tuple[str, str]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".nuspec") and "/" not in name]
        if len(names) != 1:
            raise NuGetReleaseError("package must have exactly one root nuspec")
        metadata = ET.fromstring(archive.read(names[0])).find("{*}metadata")
    if metadata is None:
        raise NuGetReleaseError("nuspec has no metadata element")
    return (metadata.findtext("{*}id") or "").strip(), (metadata.findtext("{*}version") or "").strip()


def artifact(dist: Path, coordinate: str, version: str) -> Path:
    """The one tested .nupkg in dist/, named and described as the release."""
    packages = sorted(dist.glob("*.nupkg"))
    if len(packages) != 1:
        raise NuGetReleaseError("expected exactly one tested .nupkg")
    package_id, package_version = nuspec_identity(packages[0].read_bytes())
    if (package_id.lower() != coordinate.lower() or package_version.lower() != version.lower()
            or packages[0].name.lower() != f"{coordinate}.{version}.nupkg".lower()):
        raise NuGetReleaseError("tested package id or version does not match the release")
    return packages[0]


def listed(coordinate: str, version: str) -> bool:
    """The registration leaf's `listed` flag; an absent flag means listed, an absent leaf has not propagated."""
    payload = _get(f"{REGISTRATION}/{coordinate.lower()}/{version.lower()}.json")
    if payload is None:
        raise NuGetReleaseError(f"nuget.org has no registration leaf for {coordinate} {version} yet")
    return json.loads(payload).get("listed", True) is not False


def registry_state(coordinate: str, version: str, dist: Path) -> bool:
    """False while nuget.org does not serve the version; True once it serves the tested entries, listed.

    Different entries are a mismatch (an upload is immutable and is never replaced); an unlisted
    version is treated as yanked. Both raise instead of returning.
    """
    tested = artifact(dist, coordinate, version).read_bytes()
    if NuGetVersion(version) not in fetch_versions(coordinate):
        return False
    served = _get(f"{FLAT_CONTAINER}/{coordinate.lower()}/{version.lower()}/{coordinate.lower()}.{version.lower()}.nupkg")
    if served is None:
        return False
    if not served_matches(served, tested):
        raise NuGetMismatchError(f"nuget.org serves different package entries for {coordinate} {version} than the tested "
                                 "package; uploads are immutable, fix forward with the next version")
    if not listed(coordinate, version):
        raise NuGetMismatchError(f"{coordinate} {version} is unlisted on nuget.org; an unlisted version is not what users get")
    return True
