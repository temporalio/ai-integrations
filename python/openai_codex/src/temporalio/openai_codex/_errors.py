"""Preserve SDK errors across Activity boundaries."""

from openai_codex import CodexError, TransportClosedError
from openai_codex.errors import JsonRpcError, map_jsonrpc_error
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    TimeoutError,
    TimeoutType,
)

from ._constants import CONNECTION_LOST


def activity_error(error: Exception) -> ApplicationError:
    """Encode permanent SDK failures without retrying side effects."""
    details = (
        (error.code, error.message, error.data)
        if isinstance(error, JsonRpcError)
        else ()
    )
    return ApplicationError(
        str(error), *details, type=type(error).__name__, non_retryable=True
    )


def sdk_error(error: ActivityError) -> Exception:
    """Restore SDK error types and identify a lost connection Worker."""
    cause = error.cause
    if isinstance(cause, TimeoutError) and cause.type == TimeoutType.SCHEDULE_TO_START:
        result: Exception = TransportClosedError(CONNECTION_LOST)
    elif isinstance(cause, ApplicationError):
        if len(cause.details) == 3:
            result = map_jsonrpc_error(*cause.details)
        elif cause.type == "TransportClosedError":
            result = TransportClosedError(cause.message)
        elif cause.type == "FileNotFoundError":
            result = FileNotFoundError(cause.message)
        elif cause.type == "PermissionError":
            result = PermissionError(cause.message)
        else:
            result = CodexError(cause.message)
    else:
        result = TransportClosedError(str(cause or error))
    result.__cause__ = error
    return result
