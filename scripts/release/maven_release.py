#!/usr/bin/env python3
"""Sign tested Maven artifacts and stage/publish them through Central Portal.

Candidates stop at VALIDATED. The workflow calls publish only for gated finals.
The saved deployment ID and signed bundle make failed-job reruns recoverable
without rebuilding, re-signing, or uploading a second deployment.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

from release_tool import JAVA_VERSION_RE, MAVEN_CENTRAL, PolicyError, _write_outputs

PORTAL = "https://central.sonatype.com/api/v1/publisher"
HASHES = ("md5", "sha1", "sha256")
SUFFIXES = (".jar", "-sources.jar", "-javadoc.jar", ".pom", ".module")


def digest(payload: bytes, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, payload, usedforsecurity=False).hexdigest()


def artifacts(dist: Path, coordinate: str, version: str) -> dict[str, bytes]:
    if not JAVA_VERSION_RE.fullmatch(version):
        raise PolicyError("Maven release versions must be X.Y.Z or X.Y.Z-RCN")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*:[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*", coordinate):
        raise PolicyError("invalid Maven coordinate")
    group, artifact = coordinate.split(":")
    expected = {f"{artifact}-{version}{suffix}" for suffix in SUFFIXES}
    found = {p.name for p in dist.iterdir()}
    if found != expected or any(not p.is_file() or p.is_symlink() for p in dist.iterdir()):
        raise PolicyError(f"expected exactly the five tested Maven artifacts; missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    prefix = f"{group.replace('.', '/')}/{artifact}/{version}"
    return {f"{prefix}/{name}": (dist / name).read_bytes() for name in sorted(expected)}


def bundle_names(raw: dict[str, bytes]) -> set[str]:
    return {name + suffix for name in raw for suffix in ("", ".asc", *[f".{h}" for h in HASHES], *[f".asc.{h}" for h in HASHES])}


def validate_bundle(bundle: Path, raw: dict[str, bytes]) -> dict[str, bytes]:
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != bundle_names(raw):
            raise PolicyError("signed bundle does not contain exactly the expected Maven files")
        contents = {name: archive.read(name) for name in names}
    for name, payload in raw.items():
        if contents[name] != payload:
            raise PolicyError(f"saved bundle has different bytes than the tested artifact {name}")
        if not contents[name + ".asc"].startswith(b"-----BEGIN PGP SIGNATURE-----"):
            raise PolicyError(f"missing armored signature for {name}")
        for signed_name in (name, name + ".asc"):
            for algorithm in HASHES:
                if contents[f"{signed_name}.{algorithm}"].decode().strip() != digest(contents[signed_name], algorithm):
                    raise PolicyError(f"incorrect {algorithm} checksum for {signed_name}")
    return contents


def write_bundle(bundle: Path, contents: dict[str, bytes]) -> None:
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)


@contextmanager
def signing_home():
    with tempfile.TemporaryDirectory(prefix="maven-signing-") as directory:
        home = Path(directory)
        home.chmod(0o700)
        try:
            yield home
        finally:
            subprocess.run(["gpgconf", "--homedir", str(home), "--kill", "gpg-agent"],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sign_bundle(bundle: Path, raw: dict[str, bytes]) -> None:
    key = os.environ.get("GPG_PRIVATE_KEY")
    passphrase = os.environ.get("GPG_PASSPHRASE")
    fingerprint = os.environ.get("GPG_FINGERPRINT", "").upper()
    if not key or passphrase is None or not re.fullmatch(r"(?:[0-9A-F]{40}|[0-9A-F]{64})", fingerprint):
        raise PolicyError("set GPG_PRIVATE_KEY, GPG_PASSPHRASE and the full GPG_FINGERPRINT environment variable")
    key_payload = private_key_payload(key)
    with signing_home() as home:
        base = ["gpg", "--homedir", str(home), "--batch", "--no-tty"]
        # Neither key nor passphrase enters CLI arguments, logs, artifacts, or the checkout.
        subprocess.run([*base, "--import"], input=key_payload, check=True, capture_output=True)
        listing = subprocess.run([*base, "--with-colons", "--list-secret-keys"], check=True, capture_output=True, text=True).stdout
        fingerprints = {line.split(":")[9] for line in listing.splitlines() if line.startswith("fpr:")}
        if fingerprint not in fingerprints:
            raise PolicyError("GPG_FINGERPRINT does not match the imported signing key")
        contents: dict[str, bytes] = {}
        for name, payload in raw.items():
            source = home / Path(name).name
            source.write_bytes(payload)
            signature = source.with_name(source.name + ".asc")
            subprocess.run(
                [*base, "--pinentry-mode", "loopback", "--passphrase-fd", "0", "--local-user", fingerprint,
                 "--armor", "--detach-sign", "--output", str(signature), str(source)],
                input=passphrase.encode() + b"\n", check=True, capture_output=True,
            )
            subprocess.run([*base, "--verify", str(signature), str(source)], check=True, capture_output=True)
            contents[name] = payload
            contents[name + ".asc"] = signature.read_bytes()
            for signed_name in (name, name + ".asc"):
                for algorithm in HASHES:
                    contents[f"{signed_name}.{algorithm}"] = digest(contents[signed_name], algorithm).encode()
        write_bundle(bundle, contents)


def private_key_payload(value: str) -> bytes:
    if value.lstrip().startswith("-----BEGIN PGP PRIVATE KEY BLOCK-----"):
        return value.encode()
    try:
        # sdk-java stores a base64-encoded binary secret keyring; accept it directly.
        return base64.b64decode("".join(value.split()), validate=True)
    except ValueError as exc:
        raise PolicyError("GPG_PRIVATE_KEY must be armored OpenPGP or a base64-encoded secret keyring") from exc


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Storage redirects may use presigned URLs; never forward Portal credentials."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise PolicyError("refusing a non-HTTPS registry redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


class Portal:
    def __init__(self) -> None:
        username = os.environ.get("CENTRAL_USERNAME")
        password = os.environ.get("CENTRAL_PASSWORD")
        if not username or not password:
            raise PolicyError("Central Portal user-token credentials are required")
        self.token = base64.b64encode(f"{username}:{password}".encode()).decode()
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::add-mask::{self.token}")
        self.opener = urllib.request.build_opener(SafeRedirect())

    def request(self, path: str, *, method: str = "GET", data: bytes | None = None, content_type: str | None = None, absent_ok: bool = False) -> bytes | None:
        headers = {"Authorization": f"Bearer {self.token}"}
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(PORTAL + path, data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and absent_ok:
                return None
            # Do not print the response body, which may reflect credentials.
            raise PolicyError(f"Central Portal {method} returned HTTP {exc.code} for {path}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PolicyError(f"Central Portal {method} failed; inspect Portal before retrying an upload") from exc

    def status(self, deployment: str) -> dict:
        result = json.loads(self.request(f"/status?id={deployment}", method="POST") or b"{}")
        if result.get("deploymentId") != deployment or "deploymentState" not in result:
            raise PolicyError("Portal returned an invalid deployment status")
        return result

    def wait(self, deployment: str, *, published: bool = False, attempts: int = 60, delay: float = 10) -> str:
        for attempt in range(attempts):
            status = self.status(deployment)
            state = status["deploymentState"]
            print(f"Central deployment {deployment}: {state}")
            if state == "FAILED":
                raise PolicyError(f"Central validation failed for {deployment}; inspect its validation report in Portal")
            if state == "PUBLISHED" or (state == "VALIDATED" and not published):
                return state
            if state not in {"PENDING", "VALIDATING", "VALIDATED", "PUBLISHING"}:
                raise PolicyError(f"unexpected Portal state {state!r}")
            if attempt + 1 < attempts:
                time.sleep(delay)
        raise PolicyError(f"timed out waiting for {deployment}; rerun the failed job using the saved deployment ID")

    def upload(self, bundle: Path, coordinate: str, version: str) -> str:
        boundary = "central-" + uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="bundle"; filename="bundle.zip"\r\n'
                'Content-Type: application/octet-stream\r\n\r\n').encode() + bundle.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        query = urllib.parse.urlencode({"name": f"{coordinate}:{version}", "publishingType": "USER_MANAGED"})
        # A POST is never retried: an ambiguous failure may already have created a deployment.
        response = self.request(f"/upload?{query}", method="POST", data=body, content_type=f"multipart/form-data; boundary={boundary}")
        deployment = deployment_id((response or b"").decode().strip())
        print(f"Created private Central deployment {deployment}")
        return deployment

    def download(self, deployment: str, name: str) -> bytes:
        result = self.request(f"/deployment/{deployment}/download/{name}")
        assert result is not None
        return result


def deployment_id(value: str) -> str:
    try:
        result = str(uuid.UUID(value))
    except ValueError as exc:
        raise PolicyError("deployment ID must be a UUID from Central Portal") from exc
    if value != result:
        raise PolicyError("deployment ID must be a canonical UUID")
    return result


def save_state(state_dir: Path, deployment: str, coordinate: str, version: str, raw: dict[str, bytes]) -> dict:
    state_dir.mkdir(parents=True, exist_ok=True)
    state = {"deployment_id": deployment, "coordinate": coordinate, "version": version,
             "artifacts": {name: digest(payload) for name, payload in raw.items()},
             "bundle_sha256": digest((state_dir / "bundle.zip").read_bytes())}
    path = state_dir / "deployment.json"
    temporary = state_dir / "deployment.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)
    return state


def load_state(state_dir: Path, coordinate: str, version: str, raw: dict[str, bytes]) -> dict | None:
    path = state_dir / "deployment.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    expected = {name: digest(payload) for name, payload in raw.items()}
    if state.get("coordinate") != coordinate or state.get("version") != version or state.get("artifacts") != expected:
        raise PolicyError("saved deployment does not match this coordinate, version and tested artifacts")
    deployment_id(state.get("deployment_id", ""))
    bundle = state_dir / "bundle.zip"
    if not bundle.is_file() or digest(bundle.read_bytes()) != state.get("bundle_sha256"):
        raise PolicyError("saved signed bundle is missing or has changed")
    validate_bundle(bundle, raw)
    return state


def public_file(name: str) -> bytes | None:
    try:
        with urllib.request.urlopen(f"{MAVEN_CENTRAL}/{name}", timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise PolicyError(f"Maven Central returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PolicyError("Maven Central is unreachable; refusing to guess") from exc


def verify_remote(contents: dict[str, bytes], fetch, *, attempts: int = 1, delay: float = 10) -> None:
    for name, payload in contents.items():
        for attempt in range(attempts):
            remote = fetch(name)
            if remote is not None:
                if remote != payload:
                    raise PolicyError(f"registry serves different bytes for {name}; uploads are immutable, fix forward")
                break
            if attempt + 1 == attempts:
                raise PolicyError(f"registry does not serve {name}")
            time.sleep(delay)
    print(f"OK: registry serves all {len(contents)} tested artifacts, signatures and checksum files byte for byte")


def smoke(plugin_dir: Path, version: str, repository: str, portal: Portal | None = None) -> None:
    env = os.environ.copy()
    for name in ("CENTRAL_USERNAME", "CENTRAL_PASSWORD", "GPG_PRIVATE_KEY", "GPG_PASSPHRASE"):
        env.pop(name, None)
    if portal:
        env["CENTRAL_BEARER"] = portal.token
    else:
        env.pop("CENTRAL_BEARER", None)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "ci/smoke_java.py"),
                    "--plugin-dir", str(plugin_dir), "--version", version, "--repository", repository], check=True, env=env)


def verify_deployment(portal: Portal, deployment: str, coordinate: str, version: str) -> None:
    group, artifact = coordinate.split(":")
    expected = {f"pkg:maven/{group}/{artifact}@{version}"}
    if set(portal.status(deployment).get("purls", [])) != expected:
        raise PolicyError("deployment must contain only the requested coordinate and version; refusing to publish unrelated artifacts")


def stage(args: argparse.Namespace, coordinate: str, raw: dict[str, bytes]) -> dict:
    portal = Portal()
    state = load_state(args.state_dir, coordinate, args.version, raw)
    bundle = args.state_dir / "bundle.zip"
    if state:
        if args.deployment_id and args.deployment_id != state["deployment_id"]:
            raise PolicyError("requested deployment ID differs from the saved deployment")
        deployment = state["deployment_id"]
    elif args.deployment_id:
        deployment = deployment_id(args.deployment_id)
        # Fresh dispatch recovery: fetch the existing signatures instead of re-signing.
        if portal.wait(deployment) == "PUBLISHED":
            contents = {name: public_file(name) for name in bundle_names(raw)}
            if any(payload is None for payload in contents.values()):
                raise PolicyError("published deployment files have not propagated; retry recovery later")
        else:
            contents = {name: portal.download(deployment, name) for name in bundle_names(raw)}
        write_bundle(bundle, contents)
        validate_bundle(bundle, raw)
        state = save_state(args.state_dir, deployment, coordinate, args.version, raw)
    else:
        intent = args.state_dir / "upload-started.json"
        if intent.exists():
            raise PolicyError("a previous upload may have succeeded without returning its ID; inspect Portal and supply --deployment-id")
        if args.recover_only:
            raise PolicyError("recovery requires saved deployment state or --deployment-id; inspect Portal before retrying")
        name = next(name for name in raw if name.endswith(f"{coordinate.split(':')[1]}-{args.version}.jar"))
        if public_file(name) is not None or portal.request(f"/deployments/download/{name}", absent_ok=True) is not None:
            raise PolicyError("this Maven version already exists publicly or in Portal; recover with its deployment ID instead of uploading again")
        sign_bundle(bundle, raw)
        validate_bundle(bundle, raw)
        # Persist intent before the POST. If its response is lost, recovery must
        # require a human-provided deployment ID rather than guess and upload twice.
        intent.write_text(json.dumps({"coordinate": coordinate, "version": args.version}) + "\n")
        deployment = portal.upload(bundle, coordinate, args.version)
        # Save before polling: a validation/consumer failure must not trigger another upload.
        state = save_state(args.state_dir, deployment, coordinate, args.version, raw)
    _write_outputs(args.github_output, {"deployment_id": deployment})
    state_name = portal.wait(deployment)
    verify_deployment(portal, deployment, coordinate, args.version)
    contents = validate_bundle(bundle, raw)
    if state_name == "PUBLISHED":
        if "-RC" in args.version:
            raise PolicyError("this candidate has already been published publicly; candidates must remain privately staged")
        verify_remote(contents, public_file, attempts=30)
        repository = MAVEN_CENTRAL
        smoke_portal = None
    else:
        verify_remote(contents, lambda name: portal.download(deployment, name))
        repository = f"{PORTAL}/deployment/{deployment}/download/"
        smoke_portal = portal
    if args.smoke:
        smoke(args.plugin_dir, args.version, repository, smoke_portal)
    return state


def publish(args: argparse.Namespace, coordinate: str, raw: dict[str, bytes]) -> dict:
    if "-RC" in args.version:
        raise PolicyError("candidates cannot be published to Maven Central")
    # Defense in depth: a direct CLI invocation has the same final/cutover policy as the workflow.
    subprocess.run([sys.executable, str(Path(__file__).with_name("release_tool.py")), "check-version-policy",
                    "--plugin-dir", str(args.plugin_dir), "--version", args.version], check=True)
    state = load_state(args.state_dir, coordinate, args.version, raw)
    if not state:
        raise PolicyError("publication requires the saved, verified staging deployment")
    portal = Portal()
    deployment = state["deployment_id"]
    state_name = portal.wait(deployment)
    verify_deployment(portal, deployment, coordinate, args.version)
    if state_name == "VALIDATED":
        portal.request(f"/deployment/{deployment}", method="POST")
    portal.wait(deployment, published=True)
    verify_remote(validate_bundle(args.state_dir / "bundle.zip", raw), public_file, attempts=30)
    if args.smoke:
        smoke(args.plugin_dir, args.version, MAVEN_CENTRAL)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bundle", "stage", "publish"))
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=Path("release-state"))
    parser.add_argument("--deployment-id", default="")
    parser.add_argument("--recover-only", action="store_true",
                        help="reuse saved state or an explicit deployment ID; never start another upload")
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        meta = tomllib.loads((args.plugin_dir / "plugin.toml").read_text())
        if meta["plugin"]["language"] != "java" or meta["plugin"]["registry"] != "maven":
            raise PolicyError("expected a Java plugin with the Maven registry")
        coordinate = meta["plugin"]["coordinate"]
        raw = artifacts(args.dist, coordinate, args.version)
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "ci/check_java_dist.py"),
                        "--plugin-dir", str(args.plugin_dir), "--version", args.version, "--dist", str(args.dist)], check=True)
        if args.command == "bundle":
            sign_bundle(args.state_dir / "bundle.zip", raw)
            validate_bundle(args.state_dir / "bundle.zip", raw)
        else:
            (stage if args.command == "stage" else publish)(args, coordinate, raw)
        return 0
    except (PolicyError, OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        # subprocess stderr is deliberately omitted: signing tools may reflect key material.
        print(f"FAIL: {exc}" if not isinstance(exc, subprocess.CalledProcessError) else "FAIL: signing, policy, artifact validation or consumer check failed; see the preceding safe diagnostics")
        return 1


if __name__ == "__main__":
    sys.exit(main())
