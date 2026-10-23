"""Register the Claude turn Activity and make payloads Pydantic-capable.

Place this plugin before any other plugin that replaces the data converter, so
that plugin runs last. This package installs no sandbox passthrough and no
interceptor.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from temporalio.claude_agent_sdk._activity import (
    CANCEL_LATENCY,
    ClaudeTurnActivity,
    PartialSink,
)
from temporalio.claude_agent_sdk._lease import LeaseStore
from temporalio.claude_agent_sdk._models import RUN_CLAUDE_TURN_ACTIVITY
from temporalio.claude_agent_sdk._native import DEFAULT_NATIVE_ENV_DENYLIST
from temporalio.claude_agent_sdk._sdk import load_sdk
from temporalio.claude_agent_sdk._workspace import WorkspaceStore
from temporalio.claude_agent_sdk.sandbox import SandboxRuntime
from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkerConfig

if TYPE_CHECKING:
    from claude_agent_sdk import SessionStore


class ClaudeAgentPlugin(SimplePlugin):
    """Temporal plugin for the Claude Agent SDK integration.

    >>> from temporalio.claude_agent_sdk import ClaudeAgentPlugin
    >>> plugins = [ClaudeAgentPlugin()]

    Every argument except ``register_activities`` is passed to :class:`ClaudeTurnActivity`;
    see its docstring. Set ``register_activities=False`` on workflow-only workers.
    """

    def __init__(
        self,
        *,
        cli_path: str | None = None,
        session_store: SessionStore | None = None,
        workspace_root: str | Path | None = None,
        enable_native_tools: bool = False,
        interrupt_timeout: timedelta = timedelta(seconds=5),
        drain_timeout: timedelta = timedelta(seconds=10),
        disconnect_timeout: timedelta = timedelta(seconds=15),
        register_activities: bool = True,
        lease_store: LeaseStore | None = None,
        workspace_store: WorkspaceStore | None = None,
        native_env_denylist: Sequence[str] = DEFAULT_NATIVE_ENV_DENYLIST,
        sandbox: SandboxRuntime | None = None,
        partial_sink: PartialSink | None = None,
        require_seccomp_helper: bool = True,
    ) -> None:
        """Register the turn Activity. See the class docstring for parameters."""
        if register_activities:
            # Fail at worker startup, not on the first turn, when the extra is missing.
            load_sdk()
        turn = ClaudeTurnActivity(
            cli_path=cli_path,
            session_store=session_store,
            workspace_root=workspace_root,
            enable_native_tools=enable_native_tools,
            interrupt_timeout=interrupt_timeout,
            drain_timeout=drain_timeout,
            disconnect_timeout=disconnect_timeout,
            lease_store=lease_store,
            workspace_store=workspace_store,
            native_env_denylist=native_env_denylist,
            sandbox=sandbox,
            partial_sink=partial_sink,
            require_seccomp_helper=require_seccomp_helper,
        )

        def add_activities(
            activities: Sequence[Callable[..., Any]] | None,
        ) -> Sequence[Callable[..., Any]]:
            existing = list(activities or [])
            if not register_activities or any(
                _activity_name(fn) == RUN_CLAUDE_TURN_ACTIVITY for fn in existing
            ):
                return existing
            return [*existing, turn.run_turn]

        super().__init__(
            name="ClaudeAgentPlugin",
            data_converter=_data_converter,
            activities=add_activities,
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Cap the worker's heartbeat throttle at ``CANCEL_LATENCY`` so a turn hears of a
        cancellation within about a second instead of at its next throttled heartbeat.
        """
        config = super().configure_worker(config)
        config["max_heartbeat_throttle_interval"] = min(
            config.get("max_heartbeat_throttle_interval", CANCEL_LATENCY),
            CANCEL_LATENCY,
        )
        return config


def _data_converter(converter: DataConverter | None) -> DataConverter:
    """Upgrade only the SDK's default payload converter to Pydantic, the rule
    other plugins follow; a converter someone chose is left alone.
    """
    if converter is None:
        return DataConverter(payload_converter_class=PydanticPayloadConverter)
    if converter.payload_converter_class is DefaultPayloadConverter:
        return dataclasses.replace(
            converter, payload_converter_class=PydanticPayloadConverter
        )
    return converter


def _activity_name(fn: Callable[..., Any]) -> str | None:
    return getattr(getattr(fn, "__temporal_activity_definition", None), "name", None)
