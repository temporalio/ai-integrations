#!/usr/bin/env python3
"""Release tooling shared by every release-<language>.yml workflow.

Subcommands:
  parse-tag TAG                 validate `<language>/<plugin>/v<version>` and emit its parts
  check-version-policy          enforce the version policy against the production registry, and
                                refuse a version that already exists on TestPyPI (uploads are immutable)
  release-notes                 generate release notes from commits touching the plugin dir
  draft-release                 create/update an idempotent draft GitHub Release with assets

Policy (AGENTS.md, "Release runbook"):
  * a coordinate with no published release starts at exactly 1.0.0 (maturity ga)
    or 0.1.0 (preview/experimental); pre-releases of that version are allowed
  * an existing coordinate only ever moves strictly forward (max published incl. yanked)
  * final (non pre-release) versions additionally require plugin.toml
    [release] allow-final = true and no TRANSITION(sdk-cutover) markers in the plugin
  * versions are canonical PEP 440 and never local (+...)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from packaging.version import InvalidVersion, Version

TAG_RE = re.compile(r"^(?P<language>python|typescript|java|go)/(?P<plugin>[a-z0-9_.-]+)/v(?P<version>.+)$")
REGISTRY_JSON = {
    "pypi": "https://pypi.org/pypi/{coordinate}/json",
    "testpypi": "https://test.pypi.org/pypi/{coordinate}/json",
}
FIRST_VERSION = {"ga": Version("1.0.0"), "preview": Version("0.1.0"), "experimental": Version("0.1.0")}
TRANSITION_MARKER = "TRANSITION(sdk-cutover)"
DEFAULT_REPO = "temporalio/ai-integrations"


class PolicyError(Exception):
    pass


def _write_outputs(path: str | None, values: dict[str, str]) -> None:
    for k, v in values.items():
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


def fetch_published_versions(coordinate: str, registry: str, registry_json: Path | None = None) -> list[Version]:
    """Return every published version (yanked included). 404 -> []. Anything else fails closed."""
    if registry_json is not None:
        data = json.loads(registry_json.read_text(encoding="utf-8")) if registry_json.is_file() else None
    else:
        url = REGISTRY_JSON[registry].format(coordinate=coordinate)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                data = None
            else:
                raise PolicyError(f"registry returned HTTP {exc.code} for {url}; refusing to guess (fail closed)") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PolicyError(f"registry unreachable ({exc}); refusing to guess (fail closed)") from exc
    if not data:
        return []
    versions: list[Version] = []
    for raw in data.get("releases", {}):
        try:
            versions.append(Version(raw))
        except InvalidVersion:
            continue
    return versions


def check_policy(version: Version, maturity: str, published: list[Version]) -> None:
    if maturity not in FIRST_VERSION:
        raise PolicyError(f"unknown maturity {maturity!r}")
    if not published:
        first = FIRST_VERSION[maturity]
        if version.release != first.release or version.post is not None:
            raise PolicyError(
                f"first release of a {maturity} coordinate must be {first} (or a pre-release of it); got {version}"
            )
        return
    newest = max(published)
    if version <= newest:
        raise PolicyError(f"{version} is not strictly greater than the newest published version {newest}; versions never reset")


def transition_markers(repo_root: Path, plugin_dir: str) -> list[str]:
    """List plugin files containing the transition marker. Fails closed on any git error."""
    out = subprocess.run(
        ["git", "grep", "-l", TRANSITION_MARKER, "--", plugin_dir],
        capture_output=True, text=True, cwd=repo_root,
    )
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
    check_policy(version, maturity, published)
    print(f"OK: {coordinate} {version} satisfies the version policy (published: {[str(v) for v in sorted(published)] or 'none'})")
    if registry == "pypi":
        # Every release is uploaded to TestPyPI first, and uploads are immutable: a version that is
        # already there would be skipped (skip-existing) and the smoke test would validate stale
        # bytes while reporting success. Fix forward with the next rcN instead.
        staged = fetch_published_versions(coordinate, "testpypi", Path(args.testpypi_json) if args.testpypi_json else None)
        if version in staged:
            raise PolicyError(f"{coordinate} {version} already exists on TestPyPI; uploads are immutable, use the next pre-release number")
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
        # refs/tags/ keeps GitHub from reading the slash-separated tag as a ref plus a path.
        lines += [f"**Source**: https://github.com/{repo}/tree/refs/tags/{tag}/{plugin_dir}", ""]
    else:
        prev_tag, _ = prev
        log = _git("log", "--no-decorate", "--format=%h%x1f%s", f"{prev_tag}..{tag}", "--", plugin_dir, cwd=repo_root)
        lines += ["## What's changed", ""]
        entries = [line for line in log.splitlines() if line.strip()]
        if not entries:
            lines.append("_No commits touched this plugin since the previous release._")
        for entry in entries:
            sha, _, subject = entry.partition("\x1f")
            subject = PR_REF.sub(lambda m: f"[#{m.group(1)}](https://github.com/{repo}/pull/{m.group(1)})", subject)
            lines.append(f"- {subject} ({sha})")
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


def _json_documents(text: str) -> list:
    """Parse the concatenated JSON documents that `gh api --paginate` emits (one per page)."""
    decoder = json.JSONDecoder()
    docs: list = []
    idx = 0
    text = text.strip()
    while idx < len(text):
        doc, end = decoder.raw_decode(text, idx)
        docs.append(doc)
        idx = end
        while idx < len(text) and text[idx].isspace():
            idx += 1
    return docs


def find_release_id(repo: str, tag: str) -> str | None:
    out = _gh("api", f"repos/{repo}/releases?per_page=100", "--paginate", "--jq", f'.[] | select(.tag_name == "{tag}") | .id')
    ids = [line.strip() for line in out.splitlines() if line.strip()]
    return ids[0] if ids else None


def cmd_draft_release(args: argparse.Namespace) -> int:
    repo = args.repo or _repo()
    release_id = find_release_id(repo, args.tag)
    if release_id is None:
        cmd = ["release", "create", args.tag, "--repo", repo, "--draft", "--verify-tag", "--title", args.title, "--notes-file", args.notes]
        if args.prerelease:
            cmd.append("--prerelease")
        _gh(*cmd)
        release_id = find_release_id(repo, args.tag)
        if release_id is None:
            raise PolicyError("release was created but could not be found afterwards")
        print(f"created draft release {release_id} for {args.tag}")
    else:
        body = Path(args.notes).read_text(encoding="utf-8")
        _gh("api", "-X", "PATCH", f"repos/{repo}/releases/{release_id}", "-f", f"name={args.title}",
            "-F", "draft=true", "-F", f"prerelease={'true' if args.prerelease else 'false'}", "-f", f"body={body}")
        print(f"updated existing release {release_id} for {args.tag}")
    assets = [str(path) for path in sorted(Path(args.dist).iterdir()) if path.is_file()]
    if assets:
        # gh streams the binaries itself and --clobber replaces same-name assets atomically,
        # instead of a delete-then-POST through `gh api --input`.
        _gh("release", "upload", args.tag, *assets, "--repo", repo, "--clobber")
        for asset in assets:
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
    p.add_argument("--testpypi-json", default=None, help="read TestPyPI versions from this file instead of the registry (tests)")
    p.set_defaults(func=cmd_check_version_policy)

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
