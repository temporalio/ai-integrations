"""Choose the Claude Code executable a turn runs, once per worker.

An explicit ``cli_path`` wins. Otherwise the CLI bundled in the
``claude-agent-sdk`` wheel is used. That lookup is private SDK API, called in
one place; a change there is an error, never a guessed path. Native-tool turns
never search ``PATH``. The reported ``--version`` is what the native-tool gate
checks.
"""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from temporalio.claude_agent_sdk._sdk import load_sdk

VERSION_PROBE_TIMEOUT_SECONDS = 15
_VERSION = re.compile(r"\s*([0-9]+\.[0-9]+\.[0-9]+)")


@dataclass(frozen=True)
class ResolvedCLI:
    """The executable a turn runs and the version it reported."""

    path: str
    """Absolute path of the executable the turn runs."""
    version: str | None
    """What it reported for ``--version`` (``2.1.294``), or None when that failed."""


class CLIResolutionError(Exception):
    """No CLI the turn may run; the message says why."""


def resolve_cli(cli_path: str | None, *, native: bool) -> ResolvedCLI:
    """The CLI a turn runs and its version. ``native`` turns never search ``PATH``."""
    if cli_path is not None:
        found = shutil.which(cli_path)
        if found is None:
            raise CLIResolutionError(f"cli_path {cli_path!r} is not an executable")
    elif (found := _sdk_lookup("_find_bundled_cli")) is None and native:
        raise CLIResolutionError(
            f"claude-agent-sdk {load_sdk().__version__} has no bundled Claude Code CLI, and native "
            "tools never fall back to a `claude` on PATH; pass cli_path to choose one"
        )
    elif found is None:
        found = _sdk_lookup("_find_cli")
    if found is None:
        raise CLIResolutionError(
            "claude-agent-sdk has no Claude Code CLI; pass cli_path to choose one"
        )
    path = str(Path(found).resolve())
    return ResolvedCLI(path=path, version=cli_version(path))


def cli_version(path: str) -> str | None:
    """``<path> --version``, the probe the SDK itself makes, parsed to ``X.Y.Z``."""
    try:
        out = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=VERSION_PROBE_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = _VERSION.match(out)
    return match.group(1) if match else None


def _sdk_lookup(method: str) -> str | None:
    """Run one of the SDK's own CLI lookups (``_find_bundled_cli`` or ``_find_cli``)."""
    try:
        return getattr(_sdk_transport(), method)()
    except AttributeError as exc:
        raise CLIResolutionError(
            f"claude-agent-sdk {load_sdk().__version__} no longer finds its CLI the way 0.2.165 "
            f"did ({exc}); pass cli_path to choose one"
        ) from exc
    except load_sdk().CLINotFoundError as exc:
        raise CLIResolutionError(str(exc)) from exc


def _sdk_transport() -> Any:
    # Private SDK API (0.2.165): the subprocess transport is the only code that knows where the
    # wheel keeps its CLI. Constructing it spawns nothing.
    sdk = load_sdk()
    module = importlib.import_module(
        "claude_agent_sdk._internal.transport.subprocess_cli"
    )
    return module.SubprocessCLITransport(prompt="", options=sdk.ClaudeAgentOptions())
