"""Error types for the OpenAI Agents SDK Temporal integration."""

from temporalio.exceptions import ApplicationError, TemporalError


class AgentsWorkflowError(TemporalError):
    """Error that terminates the calling workflow or update.

    Raised when the agents SDK raises an error which should terminate, or when
    the plugin rejects an unsupported configuration.
    """


class NonRetryableAgentsWorkflowError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    AgentsWorkflowError, ApplicationError
):
    """An :py:class:`AgentsWorkflowError` that a Workflow retry policy does not retry.

    It fails the workflow as an application failure of type ``AgentsWorkflowError`` with
    ``non_retryable`` set.
    """

    def __init__(self, message: str) -> None:
        """Initialize with the error message."""
        ApplicationError.__init__(
            self,
            message,
            type=AgentsWorkflowError.__name__,
            non_retryable=True,
        )
