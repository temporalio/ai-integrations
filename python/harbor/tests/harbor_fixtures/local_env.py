"""A harbor environment that runs a trial's commands on the host.

Real environments are containers or remote sandboxes; this one lets the suite
run genuine harbor trials with no container runtime and no credentials. It
honors the same contract a bind-mounting environment gives harbor: ``/logs``
is the trial directory itself, so harbor reads rewards and logs straight from
it. ``/tests``, ``/solution`` and ``/harbor`` live under a private temporary
root. Harbor's own commands are rewritten onto those locations; task scripts
find ``/logs`` through ``$HARBOR_LOGS``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

_ROOTS = ("/logs", "/tests", "/solution", "/harbor")
_ROOT_RE = re.compile(
    r"(?<![\w./])(" + "|".join(re.escape(r) for r in _ROOTS) + r")(?=/|\b)"
)


class LocalEnvironment(BaseEnvironment):
    """Runs commands with bash on the host, under a per-trial root."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._root = Path(tempfile.mkdtemp(prefix="harbor-local-"))

    @staticmethod
    def type() -> str:
        return "local"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self) -> None:
        pass

    def _map(self, path: str) -> Path:
        path = str(path)
        if path == "/logs" or path.startswith("/logs/"):
            logs = Path(self.trial_paths.trial_dir).resolve()
            return logs / path[len("/logs") :].lstrip("/")
        return self._root / path.lstrip("/")

    def _rewrite(self, command: str) -> str:
        return _ROOT_RE.sub(lambda m: str(self._map(m.group(1))), command)

    async def start(self, force_build: bool) -> None:
        for root in _ROOTS:
            self._map(root).mkdir(parents=True, exist_ok=True)

    async def stop(self, delete: bool) -> None:
        if delete:
            shutil.rmtree(self._root, ignore_errors=True)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            self._rewrite(command),
            cwd=str(self._map(cwd)) if cwd else str(self._root),
            env={**os.environ, **(env or {}), "HARBOR_LOGS": str(self._map("/logs"))},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except BaseException:
            # Timed out or cancelled: never leave the command running.
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            raise
        assert proc.returncode is not None  # communicate() waited for exit
        return ExecResult(
            stdout=out.decode(), stderr=err.decode(), return_code=proc.returncode
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        target = self._map(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        shutil.copytree(source_dir, self._map(target_dir), dirs_exist_ok=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._map(source_path), target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source = self._map(source_dir)
        if source.exists():
            shutil.copytree(source, target_dir, dirs_exist_ok=True)
