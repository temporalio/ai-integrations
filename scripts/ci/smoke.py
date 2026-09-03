#!/usr/bin/env python3
"""Smoke-test a plugin's built distributions in isolated environments.

Two modes:

Orchestrator (default; needs uv on PATH and the scripts project's Python 3.11+):
    smoke.py --plugin python/<name> --dist python/<name>/dist
  For the wheel and the sdist: create a fresh venv, install the artifact with
  its dependencies, then reinstall the artifact alone (`--no-deps --reinstall`)
  so our files are written last, then run this script in `--in-env` mode inside
  that venv.

In-env (stdlib only; runs inside the target environment, Python 3.10+):
    smoke.py --in-env --coordinate C --root-api M [--imports a,b,c]
  Imports the root API (and extra modules), checks the installed version equals
  $EXPECTED_VERSION (PEP 440 comparison via `packaging` when available, else a
  small normaliser), and runs the provenance guard shared with the plugins'
  tests: not an editable install, every file in our RECORD matches its hash,
  no files under our package directory that we do not own, and no other
  distribution ships our module path. $ALLOW_OVERLAP_WITH_CORE=1 downgrades the
  last two checks to warnings while the SDK still ships the same module
  (TRANSITION(sdk-cutover)); the orchestrator sets it automatically while
  plugin.toml has [release] allow-final = false.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Provenance guard.  # keep in sync with python/_template/tests/helpers/provenance.py
# ---------------------------------------------------------------------------


class ProvenanceError(RuntimeError):
    pass


def check_provenance(dist_name: str, pkg_rel: str, allow_overlap: bool = False, warn=print) -> None:
    """Assert the installed files for `dist_name` are exactly the ones its wheel shipped.

    Raises ProvenanceError with a remediation hint on failure.
    """
    try:
        dist = md.distribution(dist_name)
    except md.PackageNotFoundError as exc:
        raise ProvenanceError(f"{dist_name} is not installed: {exc}; run `make sync`") from exc

    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        try:
            info = json.loads(direct_url)
        except json.JSONDecodeError:
            info = {}
        if info.get("dir_info", {}).get("editable"):
            raise ProvenanceError(
                f"{dist_name} is installed in editable mode; `{pkg_rel}` would resolve to the SDK's copy. "
                "Run `make sync` (installs non-editable with UV_NO_EDITABLE=1)."
            )

    owned: set[str] = set()
    for f in dist.files or []:
        if f.hash is None or f.name.endswith(".pyc") or f.hash.mode != "sha256":
            continue
        path = Path(str(f.locate()))
        owned.add(os.path.normpath(str(path)))
        if not path.is_file():
            raise ProvenanceError(f"{f} listed in RECORD is missing on disk; run `make sync` to reinstall {dist_name}")
        digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()
        if digest != f.hash.value:
            raise ProvenanceError(
                f"{f} differs from the hash recorded by {dist_name} (overwritten by another distribution, e.g. the SDK's embedded copy). "
                "Run `make sync` so our files are written last."
            )

    pkg_dir = Path(str(dist.locate_file(pkg_rel)))
    if not pkg_dir.is_dir():
        raise ProvenanceError(f"{pkg_dir} does not exist; run `make sync`")
    extras = {
        os.path.normpath(str(p))
        for p in pkg_dir.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
    } - owned
    if allow_overlap:
        extras.discard(os.path.normpath(str(pkg_dir / "README.md")))  # temporalio<=1.32 ships it inside the package
    if extras:
        raise ProvenanceError(
            f"files under {pkg_rel} not owned by {dist_name}: {sorted(extras)}. "
            "Another distribution installs into our package directory; run `make sync` or remove the stray files."
        )

    prefix = pkg_rel.rstrip("/") + "/"
    others: list[str] = []
    for other in md.distributions():
        try:
            name = other.metadata["Name"]
        except Exception:  # noqa: BLE001
            continue
        if not name or name.lower().replace("_", "-") == dist_name.lower().replace("_", "-"):
            continue
        if any(str(f).replace(os.sep, "/").startswith(prefix) for f in (other.files or [])):
            others.append(name)
    if others:
        msg = f"{pkg_rel} is also shipped by {sorted(others)}"
        if allow_overlap:
            warn(f"WARN: {msg} (expected until the SDK cutover release drops the embedded module)")
        else:
            raise ProvenanceError(msg + "; the temporalio floor must be a release that no longer embeds this module")


# ---------------------------------------------------------------------------
# Version comparison (packaging when available, tiny PEP 440 normaliser otherwise)
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(
    r"^v?(?P<release>\d+(?:\.\d+)*)"
    r"(?:[-_.]?(?P<pre_l>a|b|c|rc|alpha|beta|pre|preview)[-_.]?(?P<pre_n>\d*))?"
    r"(?:[-_.]?(?P<post_l>post|rev|r)[-_.]?(?P<post_n>\d*))?"
    r"(?:[-_.]?(?P<dev_l>dev)[-_.]?(?P<dev_n>\d*))?$",
    re.IGNORECASE,
)


def normalize_version(text: str) -> str:
    """Best-effort PEP 440 canonical form (no `packaging` dependency)."""
    m = _VERSION_RE.match(text.strip().lower())
    if not m:
        return text.strip().lower()
    release = ".".join(str(int(p)) for p in m.group("release").split("."))
    out = release
    if m.group("pre_l"):
        label = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc", "preview": "rc"}.get(m.group("pre_l"), m.group("pre_l"))
        out += f"{label}{int(m.group('pre_n') or 0)}"
    if m.group("post_l"):
        out += f".post{int(m.group('post_n') or 0)}"
    if m.group("dev_l"):
        out += f".dev{int(m.group('dev_n') or 0)}"
    return out


def versions_equal(a: str, b: str) -> bool:
    try:
        from packaging.version import Version  # type: ignore[import-not-found]

        return Version(a) == Version(b)
    except Exception:  # noqa: BLE001  packaging missing or unparsable: fall back
        return normalize_version(a) == normalize_version(b)


# ---------------------------------------------------------------------------
# In-env mode
# ---------------------------------------------------------------------------


def run_in_env(coordinate: str, root_api: str, imports: list[str], expected_version: str | None, allow_overlap: bool) -> int:
    failures: list[str] = []
    for module in [root_api, *imports]:
        try:
            importlib.import_module(module)
            print(f"OK: import {module}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"import {module} failed: {exc!r}")
    try:
        installed = md.version(coordinate)
        print(f"OK: {coordinate} {installed} is installed")
        if expected_version and not versions_equal(installed, expected_version):
            failures.append(f"installed version {installed} != expected {expected_version}")
        elif expected_version:
            print(f"OK: version matches expected {expected_version}")
    except md.PackageNotFoundError:
        failures.append(f"{coordinate} is not installed")
    try:
        check_provenance(coordinate, root_api.replace(".", "/"), allow_overlap=allow_overlap)
        print("OK: provenance guard passed")
    except ProvenanceError as exc:
        failures.append(str(exc))
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK: smoke test passed for {coordinate}")
    return 0


# ---------------------------------------------------------------------------
# Orchestrator mode
# ---------------------------------------------------------------------------


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run_isolated(plugin_dir: Path, dist: Path, keep: bool = False) -> int:
    import tomllib  # 3.11+ (scripts project)

    meta = tomllib.loads((plugin_dir / "plugin.toml").read_text(encoding="utf-8"))
    project = tomllib.loads((plugin_dir / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    coordinate = meta["plugin"]["coordinate"]
    root_api = meta["plugin"]["root-api"]
    imports = list(meta.get("smoke", {}).get("imports", []))
    expected_version = project["version"]
    allow_overlap = os.environ.get("ALLOW_OVERLAP_WITH_CORE")
    if allow_overlap is None:
        allow_overlap = "0" if meta.get("release", {}).get("allow-final") else "1"
        print(f"ALLOW_OVERLAP_WITH_CORE not set; defaulting to {allow_overlap} from plugin.toml [release] allow-final")

    artifacts = sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz"))
    if not artifacts:
        print(f"FAIL: no artifacts in {dist}")
        return 1
    uv = shutil.which("uv")
    if not uv:
        print("FAIL: uv not found on PATH")
        return 1
    rc = 0
    for artifact in artifacts:
        tmp = Path(tempfile.mkdtemp(prefix="smoke-"))
        venv = tmp / "venv"
        try:
            print(f"=== {artifact.name} ===")
            subprocess.run([uv, "venv", "--quiet", str(venv)], check=True)
            py = _venv_python(venv)
            env = {**os.environ, "VIRTUAL_ENV": str(venv), "UV_LINK_MODE": "copy"}
            subprocess.run([uv, "pip", "install", "--quiet", "--python", str(py), str(artifact)], check=True, env=env)
            # Install ours last so its files win any overlap with the SDK's embedded copy.
            subprocess.run([uv, "pip", "install", "--quiet", "--python", str(py), "--no-deps", "--reinstall", str(artifact)], check=True, env=env)
            cmd = [str(py), str(Path(__file__).resolve()), "--in-env", "--coordinate", coordinate, "--root-api", root_api]
            if imports:
                cmd += ["--imports", ",".join(imports)]
            result = subprocess.run(
                cmd, env={**env, "EXPECTED_VERSION": expected_version, "ALLOW_OVERLAP_WITH_CORE": allow_overlap}, cwd=str(tmp)
            )
            if result.returncode != 0:
                rc = 1
        except subprocess.CalledProcessError as exc:
            print(f"FAIL: {artifact.name}: {exc}")
            rc = 1
        finally:
            if not keep:
                shutil.rmtree(tmp, ignore_errors=True)
    print("OK: all artifacts passed" if rc == 0 else "FAIL: see above")
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-env", action="store_true", help="run inside the current environment (stdlib only)")
    parser.add_argument("--plugin", type=Path, help="plugin directory (orchestrator mode; optional in-env when tomllib exists)")
    parser.add_argument("--dist", type=Path, help="directory containing the wheel and sdist (orchestrator mode)")
    parser.add_argument("--coordinate", help="distribution name (in-env mode)")
    parser.add_argument("--root-api", help="root import path (in-env mode)")
    parser.add_argument("--imports", default="", help="comma-separated extra modules to import (in-env mode)")
    parser.add_argument("--keep", action="store_true", help="keep the temporary environments")
    args = parser.parse_args(argv)

    if args.in_env:
        coordinate, root_api, imports = args.coordinate, args.root_api, [i for i in args.imports.split(",") if i]
        if (not coordinate or not root_api) and args.plugin:
            try:
                import tomllib
            except ModuleNotFoundError:
                parser.error("--coordinate and --root-api are required on Python < 3.11 (no tomllib)")
            meta = tomllib.loads((args.plugin / "plugin.toml").read_text(encoding="utf-8"))
            coordinate = coordinate or meta["plugin"]["coordinate"]
            root_api = root_api or meta["plugin"]["root-api"]
            imports = imports or list(meta.get("smoke", {}).get("imports", []))
        if not coordinate or not root_api:
            parser.error("--in-env requires --coordinate and --root-api (or --plugin on Python >= 3.11)")
        allow = os.environ.get("ALLOW_OVERLAP_WITH_CORE", "0").lower() in {"1", "true", "yes"}
        return run_in_env(coordinate, root_api, imports, os.environ.get("EXPECTED_VERSION"), allow)

    if not args.plugin or not args.dist:
        parser.error("orchestrator mode requires --plugin and --dist")
    return run_isolated(args.plugin.resolve(), args.dist.resolve(), keep=args.keep)


if __name__ == "__main__":
    sys.exit(main())
