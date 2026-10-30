"""Carry the Claude Agent SDK's errors from the Activities to the transport, which raises them as in the Claude Agent SDK."""

from __future__ import annotations

from claude_agent_sdk import (
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)

from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    TimeoutError,
    TimeoutType,
)

from ._constants import CONNECTION_LOST


def activity_error(error: ClaudeSDKError) -> ApplicationError:
    """The Activity failure for a Claude Agent SDK error. No retry fixes one, so it fails for good."""
    details: tuple[object, ...]
    match error:
        case ProcessError():
            details = (error.exit_code, error.stderr)
        case CLIJSONDecodeError():
            details = (error.line, str(error.original_error))
        case _:
            details = ()
    return ApplicationError(
        str(error), *details, type=type(error).__name__, non_retryable=True
    )


def sdk_error(error: ActivityError) -> ClaudeSDKError:
    """The Claude Agent SDK error for an Activity that failed for good."""
    sdk: ClaudeSDKError
    match error.cause:
        case TimeoutError(type=TimeoutType.SCHEDULE_TO_START):
            # No Worker started the Activity, and only the Worker connected to
            # Claude Code polls its task queue.
            sdk = CLIConnectionError(CONNECTION_LOST)
        case ApplicationError(
            type=ProcessError.__name__, message=message, details=(exit_code, stderr)
        ):
            sdk = ProcessError(message)
            sdk.exit_code, sdk.stderr = exit_code, stderr
        case ApplicationError(
            type=CLIJSONDecodeError.__name__, details=(line, original)
        ):
            sdk = CLIJSONDecodeError(line, ValueError(original))
        case ApplicationError(type=CLINotFoundError.__name__, message=message):
            sdk = CLINotFoundError(message)
        case ApplicationError(message=message):
            sdk = CLIConnectionError(message)
        case cause:
            sdk = CLIConnectionError(str(cause or error))
    sdk.__cause__ = error
    return sdk
