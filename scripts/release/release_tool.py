#!/usr/bin/env python3
"""Release tooling shared by every release-<language>.yml workflow.

Subcommands:
  parse-tag TAG                 validate `<language>/<plugin>/v<version>` and emit its parts
  check-version-policy          enforce production and staging registry version ordering (with
                                exact-version re-runs allowed for recovery)
  verify-index-files            prove the files an index serves for a version are the local artifacts
  release-notes                 generate release notes from commits touching the plugin dir
  draft-release                 create/update an idempotent draft GitHub Release with assets

Policy (AGENTS.md, "Release runbook"):
  * a coordinate with no published release starts at exactly 1.0.0 (maturity ga)
    or 0.1.0 (preview/experimental); pre-releases of that version are allowed
  * an existing coordinate only ever moves strictly forward (max published incl. yanked)
  * TestPyPI versions move forward too, except that its newest version may be re-run
  * final (non pre-release) versions additionally require plugin.toml
    [release] allow-final = true and no TRANSITION(sdk-cutover) markers in the plugin
  * versions are canonical PEP 440 and never local (+...)
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
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from packaging.version import InvalidVersion, Version

# \Z (not $) so a trailing newline cannot ride along into GITHUB_OUTPUT.
TAG_RE = re.compile(r"^(?P<language>python|typescript|java|go)/(?P<plugin>[a-z0-9][a-z0-9_.-]*)/v(?P<version>.+)\Z")
REGISTRY_JSON = {
    "pypi": "https://pypi.org/pypi/{coordinate}/json",
    "testpypi": "https://test.pypi.org/pypi/{coordinate}/json",
}
RELEASE_JSON = {
    "pypi": "https://pypi.org/pypi/{coordinate}/{version}/json",
    "testpypi": "https://test.pypi.org/pypi/{coordinate}/{version}/json",
}
FIRST_VERSION = {"ga": Version("1.0.0"), "preview": Version("0.1.0"), "experimental": Version("0.1.0")}
TRANSITION_MARKER = "TRANSITION(sdk-cutover)"
DEFAULT_REPO = "temporalio/ai-integrations"


class PolicyError(Exception):
    pass


def _write_outputs(path: str | None, values: dict[str, str]) -> None:
    for k, v in values.items():
        if "\n" in v or "\r" in v:
            raise PolicyError(f"output {k!r} contains a line break; refusing to write it to GITHUB_OUTPUT")
        print(f"{k}={v}")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            for k, v in values.items():
                fh.write(f"{k}={v}\n")


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=cwd).stdout


def _repo() -> str:
    return os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO


# --------------------------------------------------------------------------- parse-tag


def parse_tag(tag: str) -> dict[str, str]:
    m = TAG_RE.match(tag)
    if not m:
        raise PolicyError(f"tag {tag!r} does not match <language>/<plugin>/v<version>")
    if ".." in m.group("plugin"):
        raise PolicyError(f"tag {tag!r} has an invalid plugin name")
    raw = m.group("version")
    try:
        version = Version(raw)
    except InvalidVersion as exc:
        raise PolicyError(f"tag version {raw!r} is not PEP 440: {exc}") from exc
    if version.local is not None:
        raise PolicyError(f"tag version {raw!r} must not have a local segment")
    if str(version) != raw:
        raise PolicyError(f"tag version {raw!r} is not canonical PEP 440 (expected {version})")
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
        out["allow_final"] = "true" if meta.get("release", {}).get("allow-final") else "false"
    _write_outputs(args.github_output, out)
    return 0


# --------------------------------------------------------------------------- policy


def _fetch_json(url: str, local: Path | None = None) -> dict | None:
    """GET a registry JSON document; 404 -> None; anything else fails closed. `local` replaces the network (tests)."""
    if local is not None:
        return json.loads(local.read_text(encoding="utf-8")) if local.is_file() else None
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise PolicyError(f"registry returned HTTP {exc.code} for {url}; refusing to guess (fail closed)") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PolicyError(f"registry unreachable ({exc}); refusing to guess (fail closed)") from exc


def fetch_published_versions(coordinate: str, registry: str, registry_json: Path | None = None) -> list[Version]:
    """Return every published version (yanked included). 404 -> []. Anything else fails closed."""
    data = _fetch_json(REGISTRY_JSON[registry].format(coordinate=coordinate), registry_json)
    if not data:
        return []
    versions: list[Version] = []
    for raw in data.get("releases", {}):
        try:
            versions.append(Version(raw))
        except InvalidVersion:
            continue
    return versions


def check_policy(version: Version, maturity: str, published: list[Version]) -> str | None:
    """Raise on a policy violation; return a warning when this looks like a re-run of the newest release."""
    if maturity not in FIRST_VERSION:
        raise PolicyError(f"unknown maturity {maturity!r}")
    if not published:
        first = FIRST_VERSION[maturity]
        if version.release != first.release or version.post is not None:
            raise PolicyError(
                f"first release of a {maturity} coordinate must be {first} (or a pre-release of it); got {version}"
            )
        return None
    newest = max(published)
    if version == newest:
        # A fresh run of the tag that already produced the newest release (a failure after the PyPI
        # upload, for instance). Tags never move, uploads are skipped when present, and the smoke
        # jobs verify the served files, so this is safe to continue.
        return f"{version} is already the newest published version of this coordinate; treating this run as a re-run of that release"
    if version < newest:
        raise PolicyError(f"{version} is not greater than the newest published version {newest}; versions never reset")
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
            f"{version} is not greater than the newest version staged on TestPyPI {newest}; "
            "staged versions never move backwards"
        )
    return None


def transition_markers(repo_root: Path, plugin_dir: str) -> list[str]:
    """List plugin files containing the transition marker. Fails closed on any git error."""
    try:
        out = subprocess.run(
            ["git", "grep", "-l", TRANSITION_MARKER, "--", plugin_dir],
            capture_output=True, text=True, cwd=repo_root,
        )
    except FileNotFoundError as exc:
        raise PolicyError(f"git is required to check for {TRANSITION_MARKER} markers: {exc}") from exc
    # git grep exits 1 for "no match"; anything else (not a repo, bad pathspec, ...) is an error
    # and must not be mistaken for "no markers".
    if out.returncode not in (0, 1):
        raise PolicyError(f"git grep for {TRANSITION_MARKER} failed in {repo_root} ({out.returncode}): {out.stderr.strip()}")
    return [line for line in out.stdout.splitlines() if line.strip()]


def cmd_check_version_policy(args: argparse.Namespace) -> int:
    plugin_dir = Path(args.plugin_dir)
    meta = _load(plugin_dir / "plugin.toml")
    coordinate = meta["plugin"]["coordinate"]
    maturity = meta["plugin"]["maturity"]
    registry = meta["plugin"].get("registry", "pypi")
    version = Version(args.version)
    published = fetch_published_versions(coordinate, registry, Path(args.registry_json) if args.registry_json else None)
    rerun = check_policy(version, maturity, published)
    if rerun:
        print(f"::warning::{rerun}")
    print(f"OK: {coordinate} {version} satisfies the version policy (published: {[str(v) for v in sorted(published)] or 'none'})")
    if registry == "pypi" and (args.check_testpypi or args.testpypi_json):
        # Every release is staged on TestPyPI first and uploads are immutable, so a re-run finds the
        # version already there and skip-existing keeps the upload from failing. That is the normal
        # recovery path (a rejected environment approval, a flaky smoke), so allow an exact newest
        # version re-run; the smoke job proves the served files are this run's artifacts.
        staged = fetch_published_versions(coordinate, "testpypi", Path(args.testpypi_json) if args.testpypi_json else None)
        staged_rerun = check_staging_policy(version, staged)
        if staged_rerun:
            print(f"::warning::{coordinate} {staged_rerun}: the TestPyPI upload will be skipped and "
                  "smoke-testpypi verifies that the served files are this run's artifacts")
        else:
            print(f"OK: {coordinate} {version} is not yet on TestPyPI")
    if not version.is_prerelease:
        if not meta.get("release", {}).get("allow-final"):
            raise PolicyError(
                f"{coordinate} {version} is a final release but plugin.toml [release] allow-final is false "
                "(the SDK cutover has not happened; publish a pre-release instead)"
            )
        repo_root = Path(args.repo_root).resolve()
        markers = transition_markers(repo_root, os.path.relpath(plugin_dir.resolve(), repo_root))
        if markers:
            raise PolicyError(f"final release blocked: {TRANSITION_MARKER} markers remain in {markers}")
        print("OK: final-release gates satisfied (allow-final = true, no transition markers)")
    return 0


# --------------------------------------------------------------------------- index files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_release_files(coordinate: str, version: str, registry: str, index_json: Path | None = None) -> dict[str, dict]:
    """Return {filename: {"sha256", "yanked"}} for one version on the index (empty when the version is absent)."""
    data = _fetch_json(RELEASE_JSON[registry].format(coordinate=coordinate, version=version), index_json)
    if not data:
        return {}
    return {
        entry["filename"]: {"sha256": entry["digests"]["sha256"], "yanked": bool(entry.get("yanked"))}
        for entry in data.get("urls", [])
    }


def verify_index_files(coordinate: str, version: str, registry: str, dist: Path, *, attempts: int, delay: float, index_json: Path | None = None) -> None:
    """Fail unless the index serves exactly the local distribution files, byte for byte, none yanked.

    Uploads are immutable and the publish step skips files that already exist, so this is what
    makes "the published bytes are the tested bytes" true on a re-run as well as on the first run.
    Index propagation and transient registry errors are retried; a real mismatch is not.
    """
    if attempts < 1:
        raise PolicyError("attempts must be at least 1")
    local = {path.name: _sha256(path) for path in sorted(dist.iterdir()) if path.is_file()}
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
                raise PolicyError(f"{registry} does not serve {missing} for {coordinate} {version} after {attempts} attempts")
            print(f"{missing} not yet on {registry}; waiting for index propagation (attempt {attempt}/{attempts})")
        time.sleep(delay)
    extra = sorted(set(remote) - set(local))
    if extra:
        raise PolicyError(
            f"{registry} serves files for {coordinate} {version} that this run did not build: {extra}. "
            "Only the tested artifacts may be published; fix forward with the next version"
        )
    yanked = sorted(name for name, entry in remote.items() if entry["yanked"])
    if yanked:
        raise PolicyError(f"{registry} has yanked {yanked} for {coordinate} {version}; a yanked release is not what users get")
    mismatched = sorted(name for name, digest in local.items() if remote[name]["sha256"] != digest)
    if mismatched:
        raise PolicyError(
            f"{registry} serves different bytes than this run built for {mismatched}. Either an earlier upload of "
            f"{coordinate} {version} is what users get, or this is a fresh run of a tag whose upload already happened "
            "and the rebuild used a different uv version (uv_build stamps its version into the wheel). Use "
            "'Re-run failed jobs' on the run that uploaded, or fix forward with the next version; uploads are immutable"
        )
    for name in sorted(local):
        print(f"OK: {name} on {registry} matches the tested artifact (sha256 {local[name][:12]}...)")


def cmd_verify_index_files(args: argparse.Namespace) -> int:
    verify_index_files(
        args.coordinate, args.version, args.index, Path(args.dist),
        attempts=args.attempts, delay=args.delay, index_json=Path(args.index_json) if args.index_json else None,
    )
    return 0


# --------------------------------------------------------------------------- release notes

PR_REF = re.compile(r"(?<![\w/])#(\d+)\b")


def previous_tag(repo_root: Path, language: str, plugin: str, current: Version) -> tuple[str, Version] | None:
    prefix = f"{language}/{plugin}/v"
    candidates: list[tuple[Version, str]] = []
    for tag in _git("tag", "--list", f"{prefix}*", cwd=repo_root).splitlines():
        tag = tag.strip()
        if not tag:
            continue
        try:
            v = Version(tag[len(prefix):])
        except InvalidVersion:
            continue
        # A final release summarizes everything since the previous final. Its
        # release candidates are staging points, not stable changelog boundaries.
        if not current.is_prerelease and v.is_prerelease:
            continue
        if v < current:
            candidates.append((v, tag))
    if not candidates:
        return None
    v, tag = max(candidates)
    return tag, v


def release_notes(repo_root: Path, plugin_dir: str, tag: str, repo: str) -> str:
    parts = parse_tag(tag)
    version = Version(parts["version"])
    meta_path = repo_root / plugin_dir / "plugin.toml"
    meta = _load(meta_path) if meta_path.is_file() else {"plugin": {}}
    coordinate = meta["plugin"].get("coordinate", parts["plugin"])
    root_api = meta["plugin"].get("root-api", "")
    prev = previous_tag(repo_root, parts["language"], parts["plugin"], version)
    revision = f"{prev[0]}..{tag}" if prev else tag
    log = _git("log", "--no-decorate", "--format=%h%x1f%s", revision, "--", plugin_dir, cwd=repo_root)
    lines = [f"# {coordinate} {version}", ""]
    if prev is None:
        lines += [
            f"First standalone release of `{coordinate}`.",
            "",
        ]
        upstream = meta["plugin"].get("upstream")
        if upstream and ":" in upstream:
            up_repo, up_path = upstream.split(":", 1)
            lines += [
                f"This code previously shipped inside the Temporal SDK; its full history was imported with authorship preserved "
                f"from https://github.com/{up_repo}/commits/main/{up_path}.",
                "",
            ]
    lines += ["## What's changed", ""]
    entries = [line for line in log.splitlines() if line.strip()]
    if not entries:
        scope = "through this release" if prev is None else "since the previous release"
        lines.append(f"_No commits touched this plugin {scope}._")
    for entry in entries:
        sha, _, subject = entry.partition("\x1f")
        subject = PR_REF.sub(lambda m: f"[#{m.group(1)}](https://github.com/{repo}/pull/{m.group(1)})", subject)
        lines.append(f"- {subject} ({sha})")
    if prev is None:
        # refs/tags/ keeps GitHub from reading the slash-separated tag as a ref plus a path.
        lines += [
            "",
            f"**Full changelog**: https://github.com/{repo}/commits/refs/tags/{tag}/{plugin_dir}",
            "",
            f"**Source**: https://github.com/{repo}/tree/refs/tags/{tag}/{plugin_dir}",
            "",
        ]
    else:
        prev_tag, _ = prev
        lines += ["", f"**Full changelog**: https://github.com/{repo}/compare/{prev_tag}...{tag}", ""]
    if version.is_prerelease:
        lines += [
            "## Pre-release notes",
            "",
            "- This pre-release is published to **TestPyPI only** to validate the release pipeline.",
        ]
        if not meta.get("release", {}).get("allow-final", False):
            # Only a plugin the SDK still bundles shares files with it.
            lines += [
                f"- Do **not** install it alongside a `temporalio` release that still embeds `{root_api or coordinate}`; both distributions "
                "write the same files and whichever installs last wins. Wait for the SDK cutover release.",
                "- docs.temporal.io still describes the SDK-embedded install until the cutover.",
            ]
        lines.append("")
    return "\n".join(lines)


def cmd_release_notes(args: argparse.Namespace) -> int:
    notes = release_notes(Path(args.repo_root).resolve(), args.plugin_dir.rstrip("/"), args.tag, args.repo or _repo())
    Path(args.output).write_text(notes, encoding="utf-8")
    print(notes)
    return 0


# --------------------------------------------------------------------------- draft release


def _gh(*args: str, input_text: str | None = None) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True, input=input_text).stdout


def find_releases(repo: str, tag: str) -> list[dict]:
    """Every release, draft or published, whose tag_name is `tag` (newest first)."""
    out = _gh("api", f"repos/{repo}/releases?per_page=100", "--paginate", "--jq",
              f'.[] | select(.tag_name == "{tag}") | "\\(.id)\\t\\(.draft)"')
    releases = []
    for line in out.splitlines():
        if line.strip():
            release_id, _, draft = line.partition("\t")
            releases.append({"id": release_id.strip(), "draft": draft.strip() == "true"})
    return releases


def wait_for_release(repo: str, tag: str, attempts: int = 5, delay: float = 1.0) -> list[dict]:
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
    existing = json.loads(_gh("api", f"repos/{repo}/releases/{release_id}/assets?per_page=100") or "[]")
    for item in existing:
        if item.get("name") == filename:
            _gh("api", "-X", "DELETE", f"repos/{repo}/releases/assets/{item['id']}")
    name = quote(filename, safe="")
    url = f"https://uploads.github.com/repos/{repo}/releases/{release_id}/assets?name={name}"
    _gh("api", "--method", "POST", "-H", "Content-Type: application/octet-stream", "--input", asset, url)


def cmd_draft_release(args: argparse.Namespace) -> int:
    repo = args.repo or _repo()
    releases = find_releases(repo, args.tag)
    if any(not release["draft"] for release in releases):
        # Runbook step 5 publishes the draft by hand. A published release owns the tag, and a later
        # re-run must neither un-publish it nor swap its assets; refusing here also keeps
        # `gh release upload <tag>` below from ever resolving to a published release.
        raise PolicyError(f"a published release already exists for {args.tag}; refusing to modify it")
    if not releases:
        cmd = ["release", "create", args.tag, "--repo", repo, "--draft", "--verify-tag", "--title", args.title, "--notes-file", args.notes]
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
        _gh("api", "-X", "PATCH", f"repos/{repo}/releases/{release_id}", "-f", f"tag_name={args.tag}",
            "-f", f"name={args.title}",
            "-F", "draft=true", "-F", f"prerelease={'true' if args.prerelease else 'false'}", "-f", f"body={body}")
        print(f"updated existing draft release {release_id} for {args.tag}")
    assets = [str(path) for path in sorted(Path(args.dist).iterdir()) if path.is_file()]
    if assets:
        # Address the draft by its stable API ID. `gh release upload <tag>` cannot reliably resolve
        # slash-containing tags for drafts. Replacing a same-name draft asset preserves idempotency.
        for asset in assets:
            upload_release_asset(repo, releases[0]["id"], asset)
            print(f"uploaded {Path(asset).name}")
    print(f"OK: draft release ready: https://github.com/{repo}/releases/tag/{args.tag}")
    return 0


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=os.environ.get("GITHUB_WORKSPACE", "."))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse-tag", help="validate and split a release tag")
    p.add_argument("tag")
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    p.set_defaults(func=cmd_parse_tag)

    p = sub.add_parser("check-version-policy", help="enforce the version policy")
    p.add_argument("--plugin-dir", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--registry-json", default=None, help="read published versions from this file instead of the registry (tests)")
    p.add_argument(
        "--check-testpypi",
        action="store_true",
        help="also enforce TestPyPI ordering and allow its newest exact version as a re-run",
    )
    p.add_argument("--testpypi-json", default=None, help="read TestPyPI versions from this file instead of the registry (tests)")
    p.set_defaults(func=cmd_check_version_policy)

    p = sub.add_parser("verify-index-files", help="fail unless the index serves exactly the local distribution files")
    p.add_argument("--coordinate", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--dist", required=True)
    p.add_argument("--index", choices=sorted(RELEASE_JSON), default="testpypi")
    p.add_argument("--attempts", type=int, default=10, help="index propagation retries (default 10)")
    p.add_argument("--delay", type=float, default=30.0, help="seconds between retries (default 30)")
    p.add_argument("--index-json", default=None, help="read the release JSON from this file instead of the index (tests)")
    p.set_defaults(func=cmd_verify_index_files)

    p = sub.add_parser("release-notes", help="generate release notes from history")
    p.add_argument("--plugin-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repo", default=None)
    p.set_defaults(func=cmd_release_notes)

    p = sub.add_parser("draft-release", help="create or update a draft GitHub release and upload assets")
    p.add_argument("--tag", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--notes", required=True)
    p.add_argument("--dist", required=True)
    p.add_argument("--prerelease", action="store_true")
    p.add_argument("--repo", default=None)
    p.set_defaults(func=cmd_draft_release)

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
