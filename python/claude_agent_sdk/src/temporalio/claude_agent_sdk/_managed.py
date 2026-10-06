"""SDK subprocess adapters that require supervision before engine startup."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, Message, query
from claude_agent_sdk._internal.session_resume import (
    apply_materialized_options,
    materialize_resume_session,
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

    async def _connect_inner(self, prompt: Any, actual_prompt: Any) -> None:
        # SDK connect has already recovered pending tools and materialized the
        # store. Install the transport only now, so neither step is bypassed.
        options = self.options
        if self._materialized is not None:
            options = apply_materialized_options(options, self._materialized)
        self._custom_transport = SupervisedTransport(actual_prompt, options, self.lock)
        try:
            await super()._connect_inner(prompt, actual_prompt)
        finally:
            self._custom_transport = None


async def supervised_query(
    *, prompt: str | AsyncIterable[dict[str, Any]], options: ClaudeAgentOptions
) -> AsyncIterator[Message]:
    """Run a supervised segment, retaining store-backed resume and mirror cleanup.

    Args:
        prompt: The SDK prompt or user-message stream.
        options: The segment's SDK options.

    Yields:
        Original SDK messages.
    """
    identity = f"{options.cwd}:{options.resume or options.session_id}"
    digest = hashlib.sha256(identity.encode()).hexdigest()
    lock = Path(tempfile.gettempdir()) / f"tca-session-{digest}.lock"
    materialized = await materialize_resume_session(options)
    messages = None
    transport = None
    try:
        configured = (
            apply_materialized_options(options, materialized)
            if materialized is not None
            else options
        )
        transport = SupervisedTransport(prompt, configured, lock)
        messages = cast(
            AsyncGenerator[Message, None],
            query(prompt=prompt, options=configured, transport=transport),
        )
        async for message in messages:
            yield message
    finally:
        try:
            if messages is not None:
                await messages.aclose()
        finally:
            try:
                if transport is not None:
                    await transport.close()
            finally:
                if materialized is not None:
                    await materialized.cleanup()
