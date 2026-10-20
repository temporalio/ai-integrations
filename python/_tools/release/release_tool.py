"""Python release policy, immutable-artifact verification, notes and recovery.

Run with bazel run //python/_tools:release -- <operation>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import tomllib
from packaging.version import InvalidVersion, Version

TAG_RE = re.compile(
    "^(?P<language>python)/(?P<plugin>[a-z0-9][a-z0-9_.-]*)/v(?P<version>.+)\\Z"
)
JAVA_VERSION_RE = re.compile(
    "(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)(?:-RC[1-9][0-9]*)?\\Z"
)
MAVEN_CENTRAL = "https://repo.maven.apache.org/maven2"
REGISTRY_JSON = {
    "pypi": "https://pypi.org/pypi/{coordinate}/json",
    "testpypi": "https://test.pypi.org/pypi/{coordinate}/json",
}
RELEASE_JSON = {
    "pypi": "https://pypi.org/pypi/{coordinate}/{version}/json",
    "testpypi": "https://test.pypi.org/pypi/{coordinate}/{version}/json",
}
FIRST_VERSION = {
    "generally-available": Version("1.0.0"),
    "public-preview": Version("0.1.0"),
    "pre-release": Version("0.0.1"),
}
DEFAULT_REPO = "temporalio/ai-integrations"


class PolicyError(Exception):
    pass


def _write_outputs(path: str | None, values: dict[str, str]) -> None:
    for k, v in values.items():
        if "\n" in v or "\r" in v:
            raise PolicyError(
                f"output {k!r} contains a line break; refusing to write it to GITHUB_OUTPUT"
            )
        print(f"{k}={v}")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            for k, v in values.items():
                fh.write(f"{k}={v}\n")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, cwd=cwd
    ).stdout


def _repo() -> str:
    return os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO


def release_version(language: str, raw: str) -> Version:
    if language != "python":
        raise PolicyError("Expected a Python release")
    try:
        version = Version(raw)
    except InvalidVersion as exc:
        raise PolicyError(str(exc)) from exc
    if version.local is not None or str(version) != raw:
        raise PolicyError("Version must be canonical PEP 440 without a local segment")
    return version


def parse_tag(tag: str) -> dict[str, str]:
    m = TAG_RE.match(tag)
    if not m:
        raise PolicyError(f"tag {tag!r} does not match <language>/<plugin>/v<version>")
    if ".." in m.group("plugin"):
        raise PolicyError(f"tag {tag!r} has an invalid plugin name")
    raw = m.group("version")
    version = release_version(m.group("language"), raw)
    return {
        "tag": tag,
        "language": m.group("language"),
        "plugin": m.group("plugin"),
        "version": raw,
        "prerelease": "true" if version.is_prerelease else "false",
        "plugin_dir": f"{m.group('language')}/{m.group('plugin')}",
    }


def cmd_parse_tag(args: argparse.Namespace) -> int:
    out = parse_tag(args.tag)
    plugin_toml = Path(args.repo_root) / out["plugin_dir"] / "plugin.toml"
    if plugin_toml.is_file():
        meta = _load(plugin_toml)
        out["coordinate"] = meta["plugin"]["coordinate"]
        out["root_api"] = meta["plugin"]["root-api"]
        out["smoke_imports"] = ",".join(meta.get("smoke", {}).get("imports", []))
        out["maturity"] = meta["plugin"]["maturity"]
        out["github_prerelease"] = (
            "true"
            if github_prerelease(
                release_version(out["language"], out["version"]), out["maturity"]
            )
            else "false"
        )
        if out["language"] == "python":
            out["requires_python"] = _load(plugin_toml.parent / "pyproject.toml")[
                "project"
            ]["requires-python"]
    _write_outputs(args.github_output, out)
    return 0


def github_prerelease(version: Version, maturity: str) -> bool:
    """Classify GitHub releases independently of which package index receives the version."""
    if maturity not in FIRST_VERSION:
        raise PolicyError(f"unknown maturity {maturity!r}")
    return version.is_prerelease or maturity == "pre-release"


def _fetch_json(url: str, local: Path | None = None) -> dict | None:
    """GET a registry JSON document; 404 -> None; anything else fails closed. `local` replaces the network (tests)."""
    if local is not None:
        return (
            json.loads(local.read_text(encoding="utf-8")) if local.is_file() else None
        )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise PolicyError(
            f"registry returned HTTP {exc.code} for {url}; refusing to guess (fail closed)"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PolicyError(
            f"registry unreachable ({exc}); refusing to guess (fail closed)"
        ) from exc


def fetch_published_versions(
    coordinate: str, registry: str, registry_json: Path | None = None
) -> list[Version] | list[Version]:
    """Return every published version (yanked included). 404 -> []. Anything else fails closed."""
    if registry not in REGISTRY_JSON:
        raise PolicyError(f"unsupported release registry {registry!r}")
    data = _fetch_json(
        REGISTRY_JSON[registry].format(coordinate=coordinate), registry_json
    )
    if not data:
        return []
    versions: list[Version] = []
    for raw in data.get("releases", {}):
        try:
            versions.append(Version(raw))
        except InvalidVersion:
            continue
    return versions


def check_policy(
    version: Version, maturity: str, published: list[Version] | list[Version]
) -> str | None:
    """Raise on a policy violation; return a warning when this looks like a re-run of the newest release."""
    if maturity not in FIRST_VERSION:
        raise PolicyError(f"unknown maturity {maturity!r}")
    if not published:
        first = FIRST_VERSION[maturity]
        if (
            version.release != first.release
            or getattr(version, "post", None) is not None
        ):
            raise PolicyError(
                f"first release of a {maturity} coordinate must be {first} (or a pre-release of it); got {version}"
            )
        return None
    newest = max(published)
    if version == newest:
        return f"{version} is already the newest published version of this coordinate; treating this run as a re-run of that release"
    if version < newest:
        raise PolicyError(
            f"{version} is not greater than the newest published version {newest}; versions never reset"
        )
    return None


def check_staging_policy(version: Version, staged: list[Version]) -> str | None:
    """Require TestPyPI versions to move forward; allow an exact newest-version re-run."""
    if not staged:
        return None
    newest = max(staged)
    if version == newest:
        return f"{version} is already the newest version staged on TestPyPI; treating this run as a re-run"
    if version < newest:
        raise PolicyError(
            f"{version} is not greater than the newest version staged on TestPyPI {newest}; staged versions never move backwards"
        )
    return None


def cmd_check_version_policy(args: argparse.Namespace) -> int:
    plugin_dir = Path(args.plugin_dir)
    meta = _load(plugin_dir / "plugin.toml")
    coordinate = meta["plugin"]["coordinate"]
    maturity = meta["plugin"]["maturity"]
    registry = meta["plugin"].get("registry", "pypi")
    language = "python"
    version = release_version(language, args.version)
    published = fetch_published_versions(
        coordinate, registry, Path(args.registry_json) if args.registry_json else None
    )
    rerun = check_policy(version, maturity, published)
    if rerun:
        print(f"::warning::{rerun}")
    print(
        f"OK: {coordinate} {args.version} satisfies the version policy (published: {[str(v) for v in sorted(published)] or 'none'})"
    )
    if registry == "pypi" and (args.check_testpypi or args.testpypi_json):
        staged = fetch_published_versions(
            coordinate,
            "testpypi",
            Path(args.testpypi_json) if args.testpypi_json else None,
        )
        staged_rerun = check_staging_policy(version, staged)
        if staged_rerun:
            print(
                f"::warning::{coordinate} {staged_rerun}: the TestPyPI upload will be skipped and smoke-testpypi verifies that the served files are this run's artifacts"
            )
        else:
            print(f"OK: {coordinate} {version} is not yet on TestPyPI")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_release_files(
    coordinate: str, version: str, registry: str, index_json: Path | None = None
) -> dict[str, dict]:
    """Return {filename: {"sha256", "yanked"}} for one version on the index (empty when the version is absent)."""
    data = _fetch_json(
        RELEASE_JSON[registry].format(coordinate=coordinate, version=version),
        index_json,
    )
    if not data:
        return {}
    return {
        entry["filename"]: {
            "sha256": entry["digests"]["sha256"],
            "yanked": bool(entry.get("yanked")),
        }
        for entry in data.get("urls", [])
    }


def verify_index_files(
    coordinate: str,
    version: str,
    registry: str,
    dist: Path,
    *,
    attempts: int,
    delay: float,
    index_json: Path | None = None,
) -> None:
    """Fail unless the index serves exactly the local distribution files, byte for byte, none yanked.

    Uploads are immutable and the publish step skips files that already exist, so this is what
    makes "the published bytes are the tested bytes" true on a re-run as well as on the first run.
    Index propagation and transient registry errors are retried; a real mismatch is not.
    """
    if attempts < 1:
        raise PolicyError("attempts must be at least 1")
    local = {
        path.name: _sha256(path) for path in sorted(dist.iterdir()) if path.is_file()
    }
    if not local:
        raise PolicyError(f"no distribution files in {dist}")
    remote: dict[str, dict] = {}
    for attempt in range(1, attempts + 1):
        try:
            remote = index_release_files(coordinate, version, registry, index_json)
        except PolicyError as exc:
            if attempt == attempts:
                raise
            print(f"{exc}; retrying (attempt {attempt}/{attempts})")
        else:
            missing = sorted(set(local) - set(remote))
            if not missing:
                break
            if attempt == attempts:
                raise PolicyError(
                    f"{registry} does not serve {missing} for {coordinate} {version} after {attempts} attempts"
                )
            print(
                f"{missing} not yet on {registry}; waiting for index propagation (attempt {attempt}/{attempts})"
            )
        time.sleep(delay)
    extra = sorted(set(remote) - set(local))
    if extra:
        raise PolicyError(
            f"{registry} serves files for {coordinate} {version} that this run did not build: {extra}. Only the tested artifacts may be published; fix forward with the next version"
        )
    yanked = sorted((name for name, entry in remote.items() if entry["yanked"]))
    if yanked:
        raise PolicyError(
            f"{registry} has yanked {yanked} for {coordinate} {version}; a yanked release is not what users get"
        )
    mismatched = sorted(
        (name for name, digest in local.items() if remote[name]["sha256"] != digest)
    )
    if mismatched:
        raise PolicyError(
            f"{registry} serves different bytes than this run built for {mismatched}. Either an earlier upload of {coordinate} {version} is what users get, or this is a fresh run of a tag whose upload already happened and the rebuild used a different uv version (uv_build stamps its version into the wheel). Use 'Re-run failed jobs' on the run that uploaded, or fix forward with the next version; uploads are immutable"
        )
    for name in sorted(local):
        print(
            f"OK: {name} on {registry} matches the tested artifact (sha256 {local[name][:12]}...)"
        )


def cmd_verify_index_files(args: argparse.Namespace) -> int:
    verify_index_files(
        args.coordinate,
        args.version,
        args.index,
        Path(args.dist),
        attempts=args.attempts,
        delay=args.delay,
        index_json=Path(args.index_json) if args.index_json else None,
    )
    return 0


PR_REF = re.compile("(?<![\\w/])#(\\d+)\\b")


def previous_tag(
    repo_root: Path, language: str, plugin: str, current: Version
) -> tuple[str, Version] | None:
    prefix = f"{language}/{plugin}/v"
    candidates: list[tuple[Version, str]] = []
    for tag in _git("tag", "--list", f"{prefix}*", cwd=repo_root).splitlines():
        tag = tag.strip()
        if not tag:
            continue
        try:
            v = release_version(language, tag[len(prefix) :])
        except PolicyError:
            continue
        if not current.is_prerelease and v.is_prerelease:
            continue
        if v < current:
            candidates.append((v, tag))
    if not candidates:
        return None
    v, tag = max(candidates)
    return (tag, v)


def release_notes(
    repo_root: Path, plugin_dir: str, tag: str, repo: str, *, ref: str | None = None
) -> str:
    parts = parse_tag(tag)
    version = release_version(parts["language"], parts["version"])
    meta_path = repo_root / plugin_dir / "plugin.toml"
    if ref is not None:
        ref = _git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=repo_root).strip()
        meta = tomllib.loads(
            _git("show", f"{ref}:{plugin_dir}/plugin.toml", cwd=repo_root)
        )
    else:
        meta = _load(meta_path) if meta_path.is_file() else {"plugin": {}}
    coordinate = meta["plugin"].get("coordinate", parts["plugin"])
    prev = previous_tag(repo_root, parts["language"], parts["plugin"], version)
    source = ref or tag
    revision = f"{prev[0]}..{source}" if prev else source
    log = _git(
        "log",
        "--no-decorate",
        "--format=%h%x1f%s",
        revision,
        "--",
        plugin_dir,
        cwd=repo_root,
    )
    lines: list[str] = []
    if prev is None:
        lines += [f"First standalone release of `{coordinate}`.", ""]
    lines += ["## What's changed", ""]
    entries = [line for line in log.splitlines() if line.strip()]
    if not entries:
        scope = "through this release" if prev is None else "since the previous release"
        lines.append(f"_No commits touched this plugin {scope}._")
    for entry in entries:
        sha, _, subject = entry.partition("\x1f")
        subject = PR_REF.sub(
            lambda m: f"[#{m.group(1)}](https://github.com/{repo}/pull/{m.group(1)})",
            subject,
        )
        lines.append(f"- {subject} ({sha})")
    if prev is None:
        lines += [
            "",
            f"**Full changelog**: https://github.com/{repo}/commits/refs/tags/{tag}/{plugin_dir}",
            "",
            f"**Source**: https://github.com/{repo}/tree/refs/tags/{tag}/{plugin_dir}",
            "",
        ]
    else:
        prev_tag, _ = prev
        lines += [
            "",
            f"**Full changelog**: https://github.com/{repo}/compare/{prev_tag}...{tag}",
            "",
        ]
    if version.is_prerelease and parts["language"] == "go":
        lines += [
            "## Pre-release notes",
            "",
            f"Install this Go pre-release with `go get {coordinate}@v{version}`.",
            "",
        ]
    elif version.is_prerelease and parts["language"] == "typescript":
        lines += [
            "## Pre-release notes",
            "",
            f"Install this npm pre-release with `npm install {coordinate}@{version}`.",
            "",
        ]
    elif version.is_prerelease:
        lines += [
            "## Pre-release notes",
            "",
            "- This candidate is privately staged and validated in **Maven Central Portal**; it is not published to Maven Central."
            if parts["language"] == "java"
            else "- This pre-release is published to **TestPyPI only** to validate the release pipeline.",
        ]
        lines.append("")
    return "\n".join(lines)


def cmd_release_notes(args: argparse.Namespace) -> int:
    notes = release_notes(
        Path(args.repo_root).resolve(),
        args.plugin_dir.rstrip("/"),
        args.tag,
        args.repo or _repo(),
        ref=args.ref,
    )
    Path(args.output).write_text(notes, encoding="utf-8")
    print(notes)
    return 0


def _gh(*args: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True, input=input_text
    ).stdout


def check_recovery_run(repo_root: Path, repo: str, tag: str, run_id: int) -> None:
    """Require an existing main-reachable tag and a completed run with its full matrix tested."""
    parsed = parse_tag(tag)
    language = parsed["language"]
    if language != "python" or run_id <= 0:
        raise PolicyError(
            "recovery requires a Python or TypeScript release tag and a positive run ID"
        )
    ref = f"refs/tags/{tag}"
    sha = _git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=repo_root).strip()
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=repo_root
        ).returncode
        != 0
    ):
        raise PolicyError("recovery tag must be reachable from origin/main")
    plugin_dir = parsed["plugin_dir"]
    filenames = ("plugin.toml", "pyproject.toml")
    for filename in filenames:
        relative = f"{plugin_dir}/{filename}"
        tagged = _git("show", f"{ref}:{relative}", cwd=repo_root)
        if (repo_root / relative).read_text(encoding="utf-8") != tagged:
            raise PolicyError(f"recovery requires {relative} to match the release tag")
    meta = _load(repo_root / plugin_dir / "plugin.toml")
    run = json.loads(_gh("api", f"repos/{repo}/actions/runs/{run_id}"))
    source_matches = (
        run.get("event") in {"release", "push"} and run.get("head_branch") == tag
    )
    source_description = "release-python.yml release or legacy push run"
    if (
        run.get("repository", {}).get("full_name") != repo
        or run.get("path") != f".github/workflows/release-{language}.yml"
        or (not source_matches)
        or (run.get("head_sha") != sha)
        or (run.get("status") != "completed")
    ):
        raise PolicyError(
            f"recovery source must be a completed {source_description} for this exact tag and commit"
        )
    pages = json.loads(
        _gh(
            "api",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
            "--paginate",
            "--slurp",
        )
    )
    jobs = {job["name"]: job for page in pages for job in page["jobs"]}
    versions = meta["ci"]["runtime-versions"]
    cells = {
        ("ubuntu-latest", versions[0]),
        ("ubuntu-latest", versions[-1]),
        ("macos-latest", versions[-1]),
        ("windows-latest", versions[-1]),
    }
    validation_job = "Validate tag and version policy"
    runtime = "py"
    required = {validation_job, "Test / matrix"}
    required.update(
        (
            f"Test / {parsed['plugin']} ({runner}, {runtime}{version})"
            for runner, version in cells
        )
    )
    for name in sorted(required):
        job = jobs.get(name, {})
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            raise PolicyError(
                f"recovery source has no successful completed job: {name}"
            )
    artifact_name = f"dist-{parsed['plugin']}-locked"
    artifacts = json.loads(
        _gh(
            "api",
            f"repos/{repo}/actions/runs/{run_id}/artifacts?name={artifact_name}&per_page=100",
        )
    )
    matching = [
        item
        for item in artifacts["artifacts"]
        if item["name"] == artifact_name and (not item["expired"])
    ]
    if len(matching) != 1:
        raise PolicyError(
            f"recovery source must have exactly one unexpired {artifact_name} artifact"
        )


def cmd_check_recovery_run(args: argparse.Namespace) -> int:
    check_recovery_run(
        Path(args.repo_root).resolve(), args.repo or _repo(), args.tag, args.run_id
    )
    _write_outputs(
        args.github_output, {"recovery": "true", "artifact_run_id": str(args.run_id)}
    )
    return 0


def cmd_check_published_release(args: argparse.Namespace) -> int:
    repo = args.repo or _repo()
    parsed = parse_tag(args.tag)
    meta = _load(Path(args.repo_root) / parsed["plugin_dir"] / "plugin.toml")
    prerelease = github_prerelease(
        release_version(parsed["language"], parsed["version"]),
        meta["plugin"]["maturity"],
    )
    release = json.loads(
        _gh("api", f"repos/{repo}/releases/tags/{quote(args.tag, safe='')}")
    )
    if release.get("tag_name") != args.tag or release.get("draft") is not False:
        raise PolicyError(f"a published GitHub release is required for {args.tag}")
    if release.get("prerelease") is not prerelease:
        raise PolicyError(
            f"GitHub release prerelease must be {prerelease} for {args.tag} and its plugin maturity"
        )
    if not release.get("html_url"):
        raise PolicyError("published GitHub release has no URL for the deployment")
    _write_outputs(args.github_output, {"url": release["html_url"]})
    return 0


def find_releases(repo: str, tag: str) -> list[dict]:
    """Every release, draft or published, whose tag_name is `tag` (newest first)."""
    out = _gh(
        "api",
        f"repos/{repo}/releases?per_page=100",
        "--paginate",
        "--jq",
        f'.[] | select(.tag_name == "{tag}") | "\\(.id)\\t\\(.draft)"',
    )
    releases = []
    for line in out.splitlines():
        if line.strip():
            release_id, _, draft = line.partition("\t")
            releases.append(
                {"id": release_id.strip(), "draft": draft.strip() == "true"}
            )
    return releases


def wait_for_release(
    repo: str, tag: str, attempts: int = 5, delay: float = 1.0
) -> list[dict]:
    """Wait briefly for a newly created draft to become visible through the releases API."""
    for attempt in range(attempts):
        releases = find_releases(repo, tag)
        if releases:
            return releases
        if attempt + 1 < attempts:
            time.sleep(delay)
    return []


def upload_release_asset(repo: str, release_id: str, asset: str) -> None:
    """Replace an asset on a draft by stable release ID, avoiding tag lookup in gh."""
    filename = Path(asset).name
    existing = json.loads(
        _gh("api", f"repos/{repo}/releases/{release_id}/assets?per_page=100") or "[]"
    )
    for item in existing:
        if item.get("name") == filename:
            _gh("api", "-X", "DELETE", f"repos/{repo}/releases/assets/{item['id']}")
    name = quote(filename, safe="")
    url = f"https://uploads.github.com/repos/{repo}/releases/{release_id}/assets?name={name}"
    _gh(
        "api",
        "--method",
        "POST",
        "-H",
        "Content-Type: application/octet-stream",
        "--input",
        asset,
        url,
    )


def cmd_draft_release(args: argparse.Namespace) -> int:
    repo = args.repo or _repo()
    releases = find_releases(repo, args.tag)
    if any((not release["draft"] for release in releases)):
        raise PolicyError(
            f"a published release already exists for {args.tag}; refusing to modify it"
        )
    if not releases:
        cmd = [
            "release",
            "create",
            args.tag,
            "--repo",
            repo,
            "--draft",
            "--verify-tag",
            "--title",
            args.title,
            "--notes-file",
            args.notes,
        ]
        if args.prerelease:
            cmd.append("--prerelease")
        _gh(*cmd)
        releases = wait_for_release(repo, args.tag)
        if not releases:
            raise PolicyError("release was created but could not be found afterwards")
        print(f"created draft release {releases[0]['id']} for {args.tag}")
    else:
        release_id = releases[0]["id"]
        body = Path(args.notes).read_text(encoding="utf-8")
        _gh(
            "api",
            "-X",
            "PATCH",
            f"repos/{repo}/releases/{release_id}",
            "-f",
            f"tag_name={args.tag}",
            "-f",
            f"name={args.title}",
            "-F",
            "draft=true",
            "-F",
            f"prerelease={('true' if args.prerelease else 'false')}",
            "-f",
            f"body={body}",
        )
        print(f"updated existing draft release {release_id} for {args.tag}")
    assets = [str(path) for path in sorted(Path(args.dist).iterdir()) if path.is_file()]
    if assets:
        for asset in assets:
            upload_release_asset(repo, releases[0]["id"], asset)
            print(f"uploaded {Path(asset).name}")
    release_url = _gh(
        "api", f"repos/{repo}/releases/{releases[0]['id']}", "--jq", ".html_url"
    ).strip()
    if not release_url:
        raise PolicyError("draft release has no URL for approval")
    _write_outputs(
        args.github_output, {"release_id": releases[0]["id"], "url": release_url}
    )
    print(f"OK: draft release ready: {release_url}")
    return 0


def cmd_publish_release(args: argparse.Namespace) -> int:
    repo = args.repo or _repo()
    parsed = parse_tag(args.tag)
    meta = _load(Path(args.repo_root) / parsed["plugin_dir"] / "plugin.toml")
    prerelease = github_prerelease(
        release_version(parsed["language"], parsed["version"]),
        meta["plugin"]["maturity"],
    )
    endpoint = f"repos/{repo}/releases/{args.release_id}"
    release = json.loads(_gh("api", endpoint))
    untagged_draft = release["draft"] and release["tag_name"].startswith("untagged-")
    if release["tag_name"] != args.tag and (not untagged_draft):
        raise PolicyError(f"release {args.release_id} does not belong to {args.tag}")
    if release["draft"]:
        release = json.loads(
            _gh(
                "api",
                "-X",
                "PATCH",
                endpoint,
                "--input",
                "-",
                input_text=json.dumps(
                    {"tag_name": args.tag, "draft": False, "prerelease": prerelease}
                ),
            )
        )
    else:
        print(f"release {args.release_id} is already published")
    _write_outputs(args.github_output, {"url": release["html_url"]})
    print(f"OK: published release: {release['html_url']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import os

    if os.environ.get("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(os.environ["BUILD_WORKSPACE_DIRECTORY"])
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-root", default=os.environ.get("GITHUB_WORKSPACE", "."))
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse-tag", help="validate and split a release tag")
    p.add_argument("tag")
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    p.set_defaults(func=cmd_parse_tag)
    p = sub.add_parser("check-version-policy", help="enforce the version policy")
    p.add_argument("--plugin-dir", required=True)
    p.add_argument("--version", required=True)
    p.add_argument(
        "--registry-json",
        default=None,
        help="read registry JSON (or Maven metadata XML) from this file instead of the network (tests)",
    )
    p.add_argument(
        "--check-testpypi",
        action="store_true",
        help="also enforce TestPyPI ordering and allow its newest exact version as a re-run",
    )
    p.add_argument(
        "--testpypi-json",
        default=None,
        help="read TestPyPI versions from this file instead of the registry (tests)",
    )
    p.set_defaults(func=cmd_check_version_policy)
    p = sub.add_parser(
        "verify-index-files",
        help="fail unless the index serves exactly the local distribution files",
    )
    p.add_argument("--coordinate", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--dist", required=True)
    p.add_argument("--index", choices=sorted(RELEASE_JSON), default="testpypi")
    p.add_argument(
        "--attempts",
        type=int,
        default=10,
        help="index propagation retries (default 10)",
    )
    p.add_argument(
        "--delay", type=float, default=30.0, help="seconds between retries (default 30)"
    )
    p.add_argument(
        "--index-json",
        default=None,
        help="read the release JSON from this file instead of the index (tests)",
    )
    p.set_defaults(func=cmd_verify_index_files)
    p = sub.add_parser(
        "check-recovery-run",
        help="validate tested artifacts for recovery of an existing tag",
    )
    p.add_argument("--tag", required=True)
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument("--repo", default=None)
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    p.set_defaults(func=cmd_check_recovery_run)
    p = sub.add_parser(
        "check-published-release",
        help="verify a published GitHub release and emit its URL",
    )
    p.add_argument("--tag", required=True)
    p.add_argument("--repo", default=None)
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    p.set_defaults(func=cmd_check_published_release)
    p = sub.add_parser("release-notes", help="generate release notes from history")
    p.add_argument("--plugin-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument(
        "--ref",
        help="tested commit to generate notes from before the release tag exists (default: --tag)",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--repo", default=None)
    p.set_defaults(func=cmd_release_notes)
    p = sub.add_parser(
        "draft-release",
        help="create or update a draft GitHub release and upload assets",
    )
    p.add_argument("--tag", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--notes", required=True)
    p.add_argument("--dist", required=True)
    p.add_argument("--prerelease", action="store_true")
    p.add_argument("--repo", default=None)
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    p.set_defaults(func=cmd_draft_release)
    p = sub.add_parser("publish-release", help="publish the reviewed draft")
    p.add_argument("--tag", required=True)
    p.add_argument(
        "--release-id",
        required=True,
        type=int,
        help="stable release ID captured by draft-release",
    )
    p.add_argument("--repo", default=None)
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    p.set_defaults(func=cmd_publish_release)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PolicyError as exc:
        print(f"FAIL: {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"FAIL: {' '.join(exc.cmd)} exited {exc.returncode}: {exc.stderr}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
