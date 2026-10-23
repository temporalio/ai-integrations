"""The errors the Activities fail with to tell the Workflow what happened in the sandbox."""

from __future__ import annotations

from temporalio.exceptions import ApplicationError

CLAUDE_CODE_EXITED = "ClaudeCodeExited"
"""The CLI exited with a nonzero status."""
UNKNOWN_SANDBOX = "UnknownSandbox"
"""No sandbox has the name the transport asked for."""


def claude_code_exited(status: int | None, stderr: bytes) -> ApplicationError:
    """The failure that tells the Workflow the CLI exited, with its status and stderr."""
    return ApplicationError(
        f"Claude Code exited with status {status}: {stderr.decode(errors='replace')}",
        type=CLAUDE_CODE_EXITED,
        non_retryable=True,
    )


def unknown_sandbox(name: str) -> ApplicationError:
    """The failure for a sandbox name no factory is registered under."""
    return ApplicationError(
        f"No sandbox named {name!r} is registered on ClaudeAgentSDKPlugin",
        type=UNKNOWN_SANDBOX,
        non_retryable=True,
    )
