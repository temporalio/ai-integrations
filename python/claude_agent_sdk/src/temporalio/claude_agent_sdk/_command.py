"""Claude Code's command line and environment, built from the Claude Agent SDK's options."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import __version__ as sdk_version
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from ._payloads import transcripts

# Makes Claude Code emit the session_state_changed messages the Claude Agent SDK
# uses to tell when a run is over.
_READS_SESSION_STATE = "CLAUDE_CODE_SDK_READS_SESSION_STATE"
# Default cleanupPeriodDays (100 years), so Claude Code never deletes an idle
# session's transcript. The settings can override it.
_CLEANUP_PERIOD_DAYS = 36500


class ClaudeCodeCommand:
    """Builds Claude Code's command line as the Claude Agent SDK's own transport does."""

    def __init__(self, options: ClaudeAgentOptions) -> None:
        """Build from the options passed to ``ClaudeSDKClient``.

        Raises:
            ValueError: If ``options.sandbox`` is set with ``options.settings``
                as a file path. The Claude Agent SDK merges the two by reading the file,
                which is in the sandbox, not the Workflow.
        """
        if options.sandbox and not _is_json(options.settings):
            raise ValueError(
                "TemporalTransport can't merge options.sandbox into a settings file; "
                "pass options.settings as a JSON string instead"
            )
        self._options = options

    def env(self) -> dict[str, str]:
        """The CLI's environment: the variables the Claude Agent SDK's subprocess transport would set.

        TemporalTransport replaces that transport, so it sets them itself. The
        connect Activity adds ``CLAUDE_CONFIG_DIR``.
        """
        options = self._options
        env = {
            "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
            **options.env,
            "CLAUDE_AGENT_SDK_VERSION": sdk_version,
        }
        # Like the Claude Agent SDK: set unless options.env has it, in any case.
        if not any(key.upper() == _READS_SESSION_STATE for key in env):
            env[_READS_SESSION_STATE] = "1"
        if options.enable_file_checkpointing:
            env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"
        return env

    def args(self, config_dir: str) -> list[str]:
        """The CLI's flags."""
        options = self._options
        options = dataclasses.replace(
            options,
            cli_path="claude",
            # The plugins are copied into the sandbox, and the launch adds their
            # paths there: the paths in the options are on the Worker.
            plugins=[],
            # As ClaudeSDKClient sets it for its own transport.
            permission_prompt_tool_name="stdio"
            if options.can_use_tool
            else options.permission_prompt_tool_name,
            settings=self._settings(config_dir),
        )
        # The Claude Agent SDK builds Claude Code's command line only inside its
        # subprocess transport, so that code is borrowed here. Replace this once
        # the Python Claude Agent SDK has a hook like the TypeScript one's
        # spawnClaudeCodeProcess.
        builder = SubprocessCLITransport(prompt="", options=options)
        return builder._build_command()[1:]

    def _settings(self, config_dir: str) -> str | None:
        """The options' settings, keeping the session's files from the agent and from cleanup."""
        settings = self._options.settings
        # A settings file is in the sandbox, out of the Workflow's reach.
        if not _is_json(settings):
            return settings
        merged: dict[str, Any] = json.loads(settings or "{}")
        transcript_rule = f"Edit(//{transcripts(config_dir).lstrip('/')})"
        permissions = merged.setdefault("permissions", {})
        permissions["deny"] = [*permissions.get("deny", []), transcript_rule]
        merged.setdefault("cleanupPeriodDays", _CLEANUP_PERIOD_DAYS)
        return json.dumps(merged)


def _is_json(settings: str | None) -> bool:
    """Whether ``settings`` is unset or JSON, rather than a file path, told apart as the Claude Agent SDK does."""
    text = (settings or "{}").strip()
    return text.startswith("{") and text.endswith("}")
