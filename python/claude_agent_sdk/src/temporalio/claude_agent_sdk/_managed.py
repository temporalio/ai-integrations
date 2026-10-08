"""SDK subprocess adapters that require supervision before engine startup."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk._internal.session_resume import (
    _copy_auth_files,
    apply_materialized_options,
)
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from . import _process


class SupervisedTransport(SubprocessCLITransport):
    """Start the CLI through a supervisor; never fall back to a direct launch."""

    def __init__(self, prompt: Any, options: ClaudeAgentOptions, lock: Path) -> None:
        """Configure the SDK transport and its exclusive engine lock."""
        # The SDK normally adds this flag when it constructs its transport.
        # With a supplied transport it configures only its control protocol.
        if options.can_use_tool and options.permission_prompt_tool_name is None:
            options = replace(options, permission_prompt_tool_name="stdio")
        super().__init__(prompt=prompt, options=options)
        self.lock = lock

    def _build_command(self) -> list[str]:
        launcher = Path(_process.__file__)
        if not launcher.is_file():
            raise RuntimeError("Claude engine supervisor is missing")
        return [
            sys.executable,
            "-I",
            "-S",
            str(launcher),
            str(os.getpid()),
            str(self.lock),
            *super()._build_command(),
        ]


class SupervisedClient(ClaudeSDKClient):
    """Retain SDK recovery/materialization and supervise the resulting subprocess."""

    def __init__(self, options: ClaudeAgentOptions, lock: Path) -> None:
        """Configure a live SDK client and its workspace lock."""
        super().__init__(options=options)
        self.lock = lock
        self._private_config: tempfile.TemporaryDirectory[str] | None = None

    async def _connect_inner(self, prompt: Any, actual_prompt: Any) -> None:
        # SDK connect has already recovered pending tools and materialized the
        # store. Install the transport only now, so neither step is bypassed.
        options = self.options
        if self._materialized is not None:
            options = apply_materialized_options(options, self._materialized)
        state = options.env.get("TCA_NATIVE_STATE")
        if state:
            if self._materialized is None:
                self._private_config = tempfile.TemporaryDirectory(prefix="tca-native-")
                config = Path(self._private_config.name)
                _copy_auth_files(config, options.env)
                options = replace(
                    options, env={**options.env, "CLAUDE_CONFIG_DIR": str(config)}
                )
            else:
                config = self._materialized.config_dir
            root = Path(state)
            for name in ("tasks", "todos", "plans"):
                target = root / name
                target.mkdir(parents=True, mode=0o700, exist_ok=True)
                (config / name).symlink_to(target, target_is_directory=True)
        self._custom_transport = SupervisedTransport(actual_prompt, options, self.lock)
        original_options = self.options
        self.options = options
        try:
            await super()._connect_inner(prompt, actual_prompt)
        finally:
            self.options = original_options
            self._custom_transport = None

    async def disconnect(self) -> None:
        """Stop the engine before removing its private config and workspace links."""
        try:
            await super().disconnect()
        finally:
            if self._private_config is not None:
                self._private_config.cleanup()
                self._private_config = None
