"""Exercise pnpm cooldown checks against a local registry without changing the project."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not all(shutil.which(tool) for tool in ("pnpm", "node", "make")), reason="pnpm, node and make are required")
@pytest.mark.parametrize("recent,exemption", [
    (False, None), (True, None), (True, "cooldown-*"), (True, "cooldown-test@1.0.0 || 2.0.0"),
])
def test_check_deps_preserves_the_project_and_honors_release_age(
    tmp_path: Path, recent: bool, exemption: str | None,
) -> None:
    versions = {}
    for version in ("1.0.0", "2.0.0"):
        filename = f"cooldown-test-{version}.tgz"
        data = json.dumps({"name": "cooldown-test", "version": version}).encode()
        with tarfile.open(tmp_path / filename, "w:gz") as archive:
            info = tarfile.TarInfo("package/package.json")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        versions[version] = {
            "name": "cooldown-test", "version": version,
            "dist": {"integrity": "sha512-" + base64.b64encode(hashlib.sha512((tmp_path / filename).read_bytes()).digest()).decode()},
        }
    upload_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat() if recent else "2000-01-01T00:00:00.000Z"
    registry = {
        "name": "cooldown-test", "dist-tags": {"latest": "1.0.0"},
        "versions": {"1.0.0": versions["1.0.0"]},
        "time": {version: upload_time for version in versions},
    }

    class RegistryHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/cooldown-test":
                content = json.dumps(registry).encode()
                content_type = "application/json"
            elif self.path in {f"/cooldown-test-{version}.tgz" for version in versions}:
                content = (tmp_path / self.path.lstrip("/")).read_bytes()
                content_type = "application/octet-stream"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), RegistryHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for version, metadata in versions.items():
                metadata["dist"]["tarball"] = f"http://127.0.0.1:{server.server_port}/cooldown-test-{version}.tgz"
            project = tmp_path / "project"
            project.mkdir()
            (project / "package.json").write_text(json.dumps({
                "name": "cooldown-fixture", "version": "0.0.0", "private": True,
                "packageManager": "pnpm@10.27.0",
                "dependencies": {"cooldown-test": ">=1 <3"},
                "devDependencies": {"test-helpers": "link:tests/helpers"},
                "scripts": {"preinstall": "node -e \"require('node:fs').writeFileSync('lifecycle-ran', '')\""},
            }))
            helpers = project / "tests/helpers"
            helpers.mkdir(parents=True)
            (helpers / "package.json").write_text('{"name":"test-helpers","version":"0.0.0","private":true}')
            workspace = project / "pnpm-workspace.yaml"
            exceptions = f"minimumReleaseAgeExclude: [{json.dumps(exemption)}]\n" if exemption else ""
            workspace.write_text("minimumReleaseAge: 0\n" + exceptions)
            (project / ".npmrc").write_text(f"registry=http://127.0.0.1:{server.server_port}/\n")
            temporary = tmp_path / "temporary"
            temporary.mkdir()
            env = dict(os.environ, CI="true", TMPDIR=str(temporary),
                       XDG_CACHE_HOME=str(tmp_path / "cache"), XDG_DATA_HOME=str(tmp_path / "data"),
                       XDG_STATE_HOME=str(tmp_path / "state"))

            def make(target: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["make", "-f", str(REPO / "typescript/_shared/typescript.mk"), target],
                    cwd=project, env=env, capture_output=True, text=True, timeout=30,
                )

            sync = make("sync-latest")
            assert sync.returncode == 0, sync.stdout + sync.stderr
            # Model a committed lock created before the cooldown was applied.
            workspace.write_text("minimumReleaseAge: 20160\n" + exceptions)
            registry["versions"] = versions
            registry["dist-tags"]["latest"] = "2.0.0"
            sync = make("sync")
            assert sync.returncode == 0, sync.stdout + sync.stderr
            (project / "lifecycle-ran").unlink()

            def snapshot() -> dict[str, bytes]:
                return {str(path.relative_to(project)): path.read_bytes() for path in project.rglob("*") if path.is_file()}

            original = snapshot()
            temporary_entries = set(temporary.iterdir())
            check = make("check-deps")
            assert (check.returncode == 0) == (not recent or exemption is not None), check.stdout + check.stderr
            assert snapshot() == original
            assert set(temporary.iterdir()) == temporary_entries, "the temporary lockfile must be removed on success and failure"
            if recent and exemption is None:
                assert "ERR_PNPM_NO_MATURE_MATCHING_VERSION" in check.stdout + check.stderr
        finally:
            server.shutdown()
            thread.join(timeout=5)
