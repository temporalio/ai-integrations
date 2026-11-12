"""Exercise the shared dependency check against a local index with upload timestamps."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not shutil.which("uv") or not shutil.which("make"), reason="uv and make are required")
@pytest.mark.parametrize("recent,exempt", [(False, False), (True, False), (True, True)])
def test_check_deps_applies_cooldown_without_changing_the_lock(tmp_path: Path, recent: bool, exempt: bool) -> None:
    files = []
    for version in ("1.0.0", "2.0.0"):
        filename = f"cooldown_test-{version}-py3-none-any.whl"
        info = f"cooldown_test-{version}.dist-info"
        with ZipFile(tmp_path / filename, "w") as wheel:
            wheel.writestr(f"{info}/METADATA", f"Metadata-Version: 2.3\nName: cooldown-test\nVersion: {version}\n")
            wheel.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            wheel.writestr(f"{info}/RECORD", "")
        files.append({
            "filename": filename,
            "url": f"/{filename}",
            "hashes": {"sha256": hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()},
            "upload-time": "2000-01-01T00:00:00Z",
        })
    index = {"meta": {"api-version": "1.1"}, "name": "cooldown-test", "files": files[:1]}

    class IndexHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            if self.path == "/simple/cooldown-test/":
                content = json.dumps(index).encode()
                content_type = "application/vnd.pypi.simple.v1+json"
            elif self.path in {f"/{file['filename']}" for file in files}:
                content = (tmp_path / self.path.lstrip("/")).read_bytes()
                content_type = "application/octet-stream"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(content)

        def log_message(self, *_args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), IndexHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            manifest = tmp_path / "pyproject.toml"
            exemption = ', "cooldown-test" = false' if exempt else ""
            manifest.write_text(
                '[project]\nname = "cooldown-fixture"\nversion = "0.0.0"\n'
                'requires-python = ">=3.11"\ndependencies = ["cooldown-test>=1,<3"]\n'
                '[tool.uv]\npackage = false\nexclude-newer = "2 weeks"\n'
                f'exclude-newer-package = {{ temporalio = false{exemption} }}\n'
                f'[[tool.uv.index]]\nurl = "http://127.0.0.1:{server.server_port}/simple"\ndefault = true\n'
            )
            env = dict(os.environ, UV_CACHE_DIR=str(tmp_path / "cache"), UV_PYTHON=sys.executable,
                       UV_PYTHON_DOWNLOADS="never")

            def make(target: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["make", "-f", str(REPO / "python/_shared/python.mk"), "DIST=cooldown-fixture", target],
                    cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
                )

            # Create a valid committed lock, then model a too-young release already in it.
            sync = make("sync")
            assert sync.returncode == 0, sync.stdout + sync.stderr
            lock = tmp_path / "uv.lock"
            original = lock.read_bytes()
            if recent:
                upload_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
                for file in files:
                    file["upload-time"] = upload_time
            # A successful dry run would upgrade to 2.0.0; the tested lock must stay at 1.0.0.
            index["files"] = files
            sync = make("sync")
            assert sync.returncode == 0, sync.stdout + sync.stderr
            assert lock.read_bytes() == original
            check = make("check-deps")
            assert (check.returncode == 0) == (not recent or exempt), check.stdout + check.stderr
            assert lock.read_bytes() == original
            if recent and not exempt:
                assert "exclude-newer" in check.stderr
            else:
                assert "2.0.0" in check.stderr
        finally:
            server.shutdown()
            thread.join(timeout=5)
