from __future__ import annotations

import io
import base64
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import maven_release as maven
from release_tool import PolicyError

COORDINATE = "io.temporal:spring-ai"
VERSION = "0.1.0-RC1"
DEPLOYMENT = "12345678-1234-1234-1234-123456789abc"


def raw_files(tmp_path: Path, version: str = VERSION) -> dict[str, bytes]:
    dist = tmp_path / "dist"
    dist.mkdir()
    for suffix in maven.SUFFIXES:
        (dist / f"spring-ai-{version}{suffix}").write_bytes(f"tested artifact {suffix}".encode())
    return maven.artifacts(dist, COORDINATE, version)


def fake_bundle(bundle: Path, raw: dict[str, bytes]) -> dict[str, bytes]:
    contents = {}
    for name, payload in raw.items():
        contents[name] = payload
        contents[name + ".asc"] = b"-----BEGIN PGP SIGNATURE-----\nfixture signature\n"
        for signed in (name, name + ".asc"):
            for algorithm in maven.HASHES:
                contents[f"{signed}.{algorithm}"] = maven.digest(contents[signed], algorithm).encode()
    maven.write_bundle(bundle, contents)
    return contents


def args(tmp_path: Path, version: str = VERSION, **overrides) -> SimpleNamespace:
    values = dict(state_dir=tmp_path / "state", version=version, deployment_id="", github_output=None,
                  smoke=False, recover_only=False, plugin_dir=tmp_path / "plugin")
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bundle_and_state_preserve_tested_bytes_and_detect_tampering(tmp_path: Path) -> None:
    raw = raw_files(tmp_path)
    state_dir = tmp_path / "state"
    contents = fake_bundle(state_dir / "bundle.zip", raw)
    assert len(contents) == 40 and maven.validate_bundle(state_dir / "bundle.zip", raw) == contents
    state = maven.save_state(state_dir, DEPLOYMENT, COORDINATE, VERSION, raw)
    assert maven.load_state(state_dir, COORDINATE, VERSION, raw) == state
    with pytest.raises(PolicyError, match="tested artifacts"):
        maven.load_state(state_dir, COORDINATE, "0.1.0-RC2", raw)
    changed = dict(raw)
    changed[next(iter(raw))] = b"rebuilt different bytes"
    with pytest.raises(PolicyError, match="tested artifacts"):
        maven.load_state(state_dir, COORDINATE, VERSION, changed)
    (state_dir / "bundle.zip").write_bytes(b"changed signed bundle")
    with pytest.raises(PolicyError, match="bundle is missing or has changed"):
        maven.load_state(state_dir, COORDINATE, VERSION, raw)


def test_bundle_rejects_duplicate_extra_changed_and_bad_checksum_files(tmp_path: Path) -> None:
    raw = raw_files(tmp_path)
    bundle = tmp_path / "bundle.zip"
    contents = fake_bundle(bundle, raw)
    name = next(iter(raw))
    for changed, message in [({**contents, "foreign.jar": b"unexpected"}, "exactly"),
                             ({**contents, name: b"different"}, "different bytes"),
                             ({**contents, name + ".sha1": b"incorrect"}, "checksum")]:
        maven.write_bundle(bundle, changed)
        with pytest.raises(PolicyError, match=message):
            maven.validate_bundle(bundle, raw)
    maven.write_bundle(bundle, contents)
    with zipfile.ZipFile(bundle, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr(name, raw[name])
    with pytest.raises(PolicyError, match="exactly"):
        maven.validate_bundle(bundle, raw)


def test_artifact_set_and_identifiers_reject_unexpected_inputs(tmp_path: Path) -> None:
    raw_files(tmp_path)
    (tmp_path / "dist/extra.jar").write_bytes(b"extra")
    with pytest.raises(PolicyError, match="five tested"):
        maven.artifacts(tmp_path / "dist", COORDINATE, VERSION)
    for value in ("../something", DEPLOYMENT + "\n", DEPLOYMENT.upper()):
        with pytest.raises(PolicyError, match="UUID"):
            maven.deployment_id(value)


def test_signing_accepts_the_existing_sdk_java_key_encoding() -> None:
    secret_keyring = b"binary OpenPGP secret keyring fixture"
    encoded = base64.b64encode(secret_keyring).decode()
    assert maven.private_key_payload(encoded) == secret_keyring
    assert maven.private_key_payload(encoded[:10] + "\n" + encoded[10:]) == secret_keyring
    armored = "-----BEGIN PGP PRIVATE KEY BLOCK-----\nfixture\n"
    assert maven.private_key_payload(armored) == armored.encode()
    with pytest.raises(PolicyError, match="base64-encoded"):
        maven.private_key_payload("invalid!encoding")


class FakePortal:
    def __init__(self, contents=None, state="VALIDATED", version=VERSION):
        self.contents = contents or {}
        self.state = state
        self.uploads = 0
        self.requests = []
        self.version = version

    def status(self, deployment):
        return {"deploymentId": deployment, "deploymentState": self.state,
                "purls": [f"pkg:maven/io.temporal/spring-ai@{self.version}"]}

    def wait(self, deployment, **kwargs):
        assert deployment == DEPLOYMENT
        return "PUBLISHED" if kwargs.get("published") else self.state

    def download(self, deployment, name):
        assert deployment == DEPLOYMENT
        return self.contents[name]

    def request(self, path, **kwargs):
        self.requests.append((path, kwargs))
        return None

    def upload(self, bundle, coordinate, version):
        self.uploads += 1
        self.contents = maven.validate_bundle(bundle, self.raw)
        return DEPLOYMENT


def test_failed_staging_preserves_id_and_rerun_never_resigns_or_uploads(tmp_path: Path, monkeypatch) -> None:
    raw = raw_files(tmp_path)
    portal = FakePortal()
    portal.raw = raw
    monkeypatch.setattr(maven, "Portal", lambda: portal)
    monkeypatch.setattr(maven, "public_file", lambda name: None)
    monkeypatch.setattr(maven, "sign_bundle", fake_bundle)
    real_wait = portal.wait
    portal.wait = lambda *a, **k: (_ for _ in ()).throw(PolicyError("transient validation timeout"))
    invocation = args(tmp_path)
    with pytest.raises(PolicyError, match="timeout"):
        maven.stage(invocation, COORDINATE, raw)
    assert portal.uploads == 1
    assert maven.load_state(invocation.state_dir, COORDINATE, VERSION, raw)["deployment_id"] == DEPLOYMENT
    portal.wait = real_wait
    invocation.recover_only = True
    monkeypatch.setattr(maven, "sign_bundle", lambda *a: pytest.fail("rerun re-signed artifacts"))
    maven.stage(invocation, COORDINATE, raw)
    assert portal.uploads == 1


def test_new_upload_refuses_a_version_already_staged_or_published(tmp_path: Path, monkeypatch) -> None:
    raw = raw_files(tmp_path)
    portal = FakePortal()
    monkeypatch.setattr(maven, "Portal", lambda: portal)
    monkeypatch.setattr(maven, "sign_bundle", lambda *a: pytest.fail("must check before signing/uploading"))
    monkeypatch.setattr(maven, "public_file", lambda name: b"published")
    with pytest.raises(PolicyError, match="recover with its deployment ID"):
        maven.stage(args(tmp_path), COORDINATE, raw)
    monkeypatch.setattr(maven, "public_file", lambda name: None)
    portal.request = lambda *a, **k: b"privately staged"
    with pytest.raises(PolicyError, match="recover with its deployment ID"):
        maven.stage(args(tmp_path), COORDINATE, raw)
    assert portal.uploads == 0


def test_ambiguous_upload_failure_cannot_trigger_a_second_post(tmp_path: Path, monkeypatch) -> None:
    raw = raw_files(tmp_path)
    invocation = args(tmp_path)
    portal = FakePortal()
    monkeypatch.setattr(maven, "Portal", lambda: portal)
    monkeypatch.setattr(maven, "public_file", lambda name: None)
    monkeypatch.setattr(maven, "sign_bundle", fake_bundle)
    def uncertain(*a):
        portal.uploads += 1
        raise PolicyError("response lost")
    portal.upload = uncertain
    with pytest.raises(PolicyError, match="response lost"):
        maven.stage(invocation, COORDINATE, raw)
    with pytest.raises(PolicyError, match="previous upload may have succeeded"):
        maven.stage(invocation, COORDINATE, raw)
    assert portal.uploads == 1


def test_deployment_recovery_rejects_additional_coordinates() -> None:
    portal = FakePortal()
    portal.status = lambda _: {"purls": [f"pkg:maven/io.temporal/spring-ai@{VERSION}", "pkg:maven/io.temporal/other@1.0.0"]}
    with pytest.raises(PolicyError, match="unrelated artifacts"):
        maven.verify_deployment(portal, DEPLOYMENT, COORDINATE, VERSION)


@pytest.mark.parametrize("state", ["PENDING", "VALIDATING"])
def test_fresh_dispatch_after_ambiguous_upload_never_posts_again(tmp_path: Path, monkeypatch, state: str) -> None:
    raw = raw_files(tmp_path)
    portal = FakePortal(state=state)
    monkeypatch.setattr(maven, "Portal", lambda: portal)
    monkeypatch.setattr(maven, "public_file", lambda name: None)
    monkeypatch.setattr(maven, "sign_bundle", fake_bundle)
    def uncertain(*a):
        portal.uploads += 1
        raise PolicyError("response lost after Portal accepted the upload")
    portal.upload = uncertain
    with pytest.raises(PolicyError, match="response lost"):
        maven.stage(args(tmp_path), COORDINATE, raw)
    # A new workflow run cannot retrieve the previous run's state artifact, and
    # the shared download endpoint does not expose a deployment still validating.
    with pytest.raises(PolicyError, match="recovery requires saved deployment state or --deployment-id"):
        maven.stage(args(tmp_path, state_dir=tmp_path / "fresh-run", recover_only=True), COORDINATE, raw)
    assert portal.uploads == 1


@pytest.mark.parametrize("ref_type,skip_publish,deployment,expected", [
    ("tag", "false", "", 1),
    ("tag", "true", "", 0),
    ("tag", "false", DEPLOYMENT, 0),
    ("branch", "false", "", 0),
    ("branch", "false", DEPLOYMENT, 1),
])
def test_release_dispatch_recovery_gate(ref_type: str, skip_publish: str, deployment: str, expected: int) -> None:
    workflow_path = Path(__file__).resolve().parents[2] / ".github/workflows/release-java.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    gate = next(step for step in workflow["jobs"]["prepare"]["steps"]
                if step.get("name") == "Dispatch inputs are consistent with the ref")
    tag = "java/spring-ai/v0.1.0-RC1"
    result = subprocess.run(["bash", "-c", gate["run"]], capture_output=True, text=True,
                            env={**os.environ, "REF_TYPE": ref_type, "REF_NAME": tag,
                                 "INPUT_TAG": tag, "DEPLOYMENT_ID": deployment, "SKIP_PUBLISH": skip_publish})
    assert result.returncode == expected, result.stdout + result.stderr


def test_fresh_dispatch_recovers_existing_signatures_and_verifies_raw_bytes(tmp_path: Path, monkeypatch) -> None:
    raw = raw_files(tmp_path)
    original = tmp_path / "original.zip"
    contents = fake_bundle(original, raw)
    portal = FakePortal(contents)
    monkeypatch.setattr(maven, "Portal", lambda: portal)
    monkeypatch.setattr(maven, "sign_bundle", lambda *a: pytest.fail("recovery must not re-sign"))
    state = maven.stage(args(tmp_path, deployment_id=DEPLOYMENT, recover_only=True), COORDINATE, raw)
    assert state["deployment_id"] == DEPLOYMENT and portal.uploads == 0
    assert (tmp_path / "state/bundle.zip").read_bytes() == original.read_bytes()


def test_candidates_never_publish_and_recovery_rejects_a_public_candidate(tmp_path: Path, monkeypatch) -> None:
    raw = raw_files(tmp_path)
    invocation = args(tmp_path)
    fake_bundle(invocation.state_dir / "bundle.zip", raw)
    maven.save_state(invocation.state_dir, DEPLOYMENT, COORDINATE, VERSION, raw)
    monkeypatch.setattr(maven, "Portal", lambda: FakePortal(state="PUBLISHED"))
    with pytest.raises(PolicyError, match="already been published publicly"):
        maven.stage(invocation, COORDINATE, raw)
    with pytest.raises(PolicyError, match="candidates cannot"):
        maven.publish(invocation, COORDINATE, raw)


def test_final_publishes_only_verified_deployment_and_rechecks_policy(tmp_path: Path, monkeypatch) -> None:
    version = "0.1.0"
    raw = raw_files(tmp_path, version)
    invocation = args(tmp_path, version, smoke=True)
    contents = fake_bundle(invocation.state_dir / "bundle.zip", raw)
    maven.save_state(invocation.state_dir, DEPLOYMENT, COORDINATE, version, raw)
    portal = FakePortal(contents, version=version)
    monkeypatch.setattr(maven, "Portal", lambda: portal)
    monkeypatch.setattr(maven, "public_file", contents.__getitem__)
    policy_calls = []
    monkeypatch.setattr(maven.subprocess, "run", lambda command, **kwargs: policy_calls.append(command))
    consumers = []
    monkeypatch.setattr(maven, "smoke", lambda *a: consumers.append(a))
    maven.publish(invocation, COORDINATE, raw)
    assert "check-version-policy" in policy_calls[0]
    assert portal.requests == [(f"/deployment/{DEPLOYMENT}", {"method": "POST"})]
    assert consumers == [(invocation.plugin_dir, version, maven.MAVEN_CENTRAL)]
    portal.state = "PUBLISHED"
    portal.requests.clear()
    maven.publish(invocation, COORDINATE, raw)
    assert portal.requests == [], "rerun must not request publication again"


def test_verify_remote_never_retries_a_real_mismatch(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(maven.time, "sleep", lambda *_: pytest.fail("mismatch should fail immediately"))
    def changed(name):
        calls.append(name)
        return b"different"
    with pytest.raises(PolicyError, match="different bytes"):
        maven.verify_remote({"file.jar": b"tested"}, changed, attempts=30)
    assert calls == ["file.jar"]


def test_portal_api_uses_user_managed_staging_and_masks_token(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("CENTRAL_USERNAME", "test-username")
    monkeypatch.setenv("CENTRAL_PASSWORD", "test-password")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    portal = maven.Portal()
    assert capsys.readouterr().out == f"::add-mask::{portal.token}\n"
    requests = []
    def opened(request, **kwargs):
        requests.append(request)
        return io.BytesIO(DEPLOYMENT.encode())
    monkeypatch.setattr(portal.opener, "open", opened)
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"signed ZIP")
    assert portal.upload(bundle, COORDINATE, VERSION) == DEPLOYMENT
    request = requests[0]
    assert request.method == "POST" and "publishingType=USER_MANAGED" in request.full_url
    assert request.get_header("Authorization") == "Bearer " + portal.token
    assert b"signed ZIP" in request.data
    def unavailable(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, None)
    monkeypatch.setattr(portal.opener, "open", unavailable)
    with pytest.raises(PolicyError, match="HTTP 503"):
        portal.upload(bundle, COORDINATE, VERSION)
    assert "test-password" not in capsys.readouterr().out


def test_portal_waits_through_publishing_and_fails_validation(monkeypatch) -> None:
    portal = object.__new__(maven.Portal)
    monkeypatch.setattr(maven.time, "sleep", lambda *_: None)
    states = iter(["PUBLISHING", "PUBLISHED"])
    portal.status = lambda _: {"deploymentState": next(states)}
    assert portal.wait(DEPLOYMENT, attempts=2) == "PUBLISHED"
    portal.status = lambda _: {"deploymentState": "FAILED"}
    with pytest.raises(PolicyError, match="validation failed"):
        portal.wait(DEPLOYMENT, attempts=1)


def test_redirects_strip_portal_credentials_to_storage_hosts() -> None:
    request = urllib.request.Request(maven.PORTAL + "/download/file", headers={"Authorization": "Bearer secret"})
    redirected = maven.SafeRedirect().redirect_request(request, None, 302, "redirect", {}, "https://storage.example/file")
    assert redirected.get_header("Authorization") is None
    with pytest.raises(PolicyError, match="non-HTTPS"):
        maven.SafeRedirect().redirect_request(request, None, 302, "redirect", {}, "http://storage.example/file")


@pytest.mark.skipif(shutil.which("gpg") is None, reason="GnuPG is installed on the release/CI Ubuntu runners")
def test_real_gpg_signatures_and_checksum_bundle(tmp_path: Path, monkeypatch) -> None:
    raw = raw_files(tmp_path)
    with maven.signing_home() as home:
        base = ["gpg", "--homedir", str(home), "--batch", "--pinentry-mode", "loopback", "--passphrase-fd", "0"]
        subprocess.run([*base, "--quick-generate-key", "Release Test <release-test@example.invalid>", "rsa2048", "sign", "1d"],
                       input=b"test-passphrase\n", check=True, capture_output=True)
        listing = subprocess.run(["gpg", "--homedir", str(home), "--with-colons", "--list-secret-keys"], check=True, capture_output=True, text=True).stdout
        fingerprint = next(line.split(":")[9] for line in listing.splitlines() if line.startswith("fpr:"))
        key = subprocess.run([*base, "--armor", "--export-secret-keys", fingerprint], input=b"test-passphrase\n", check=True, capture_output=True).stdout
        monkeypatch.setenv("GPG_PRIVATE_KEY", key.decode())
        monkeypatch.setenv("GPG_PASSPHRASE", "test-passphrase")
        monkeypatch.setenv("GPG_FINGERPRINT", fingerprint)
        bundle = tmp_path / "signed.zip"
        maven.sign_bundle(bundle, raw)
        contents = maven.validate_bundle(bundle, raw)
        assert len(contents) == 40
        for name, payload in raw.items():
            source = home / Path(name).name
            source.write_bytes(payload)
            signature = source.with_name(source.name + ".asc")
            signature.write_bytes(contents[name + ".asc"])
            subprocess.run(["gpg", "--homedir", str(home), "--batch", "--verify", str(signature), str(source)], check=True, capture_output=True)
