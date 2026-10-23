"""Failure converter for Strands-specific exceptions."""

from strands.interrupt import InterruptException
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
    SessionException,
    StructuredOutputException,
)

import temporalio.api.failure.v1
from temporalio.converter import (
    DefaultFailureConverter,
    DefaultFailureConverterWithEncodedAttributes,
    FailureConverter,
    PayloadConverter,
)
from temporalio.exceptions import ApplicationError

# Activity-side: when a Strands ``InterruptException`` would otherwise be
# serialized by the default converter, the ``Interrupt`` payload on
# ``exc.interrupt`` is dropped (it lives on the instance, not in the
# serialized ApplicationError). We translate to a typed ApplicationError so
# the interrupt data survives the activity boundary and the workflow side
# can rebuild a real ``Interrupt``.
STRANDS_INTERRUPT_TYPE = "StrandsInterrupt"

# Strands' model/session exceptions that are deterministic failures (token
# limits, context overflow, structured-output validation, session I/O). They
# won't succeed on retry, so they cross the boundary as non-retryable typed
# ApplicationErrors. TemporalAgent.invoke_async rewraps these as
# StrandsWorkflowError on the workflow side so users can `except` cleanly.
_TERMINAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    MaxTokensReachedException,
    ContextWindowOverflowException,
    StructuredOutputException,
    SessionException,
)


class StrandsFailureConverter(DefaultFailureConverter):
    """Failure converter that preserves Strands exception payloads and retryability."""

    def to_failure(
        self,
        exception: BaseException,
        payload_converter: PayloadConverter,
        failure: temporalio.api.failure.v1.Failure,
    ) -> None:
        """Translate Strands exceptions to typed ApplicationErrors."""
        if isinstance(exception, InterruptException):
            super().to_failure(
                ApplicationError(
                    f"interrupt:{exception.interrupt.name}",
                    exception.interrupt.to_dict(),
                    type=STRANDS_INTERRUPT_TYPE,
                    non_retryable=True,
                ),
                payload_converter,
                failure,
            )
            return
        if isinstance(exception, _TERMINAL_EXCEPTIONS):
            super().to_failure(
                ApplicationError(
                    str(exception),
                    type=type(exception).__name__,
                    non_retryable=True,
                ),
                payload_converter,
                failure,
            )
            return
        super().to_failure(exception, payload_converter, failure)


class StrandsFailureConverterWithEncodedAttributes(StrandsFailureConverter):
    """``StrandsFailureConverter`` that moves message and stack trace to encoded attributes.

    It encodes like ``DefaultFailureConverterWithEncodedAttributes`` and decodes like
    ``StrandsFailureConverter``.
    """

    def __init__(self) -> None:
        """Create the converter with encoded attributes."""
        super().__init__(encode_common_attributes=True)


def strands_failure_converter_class(
    base: type[FailureConverter],
) -> type[StrandsFailureConverter]:
    """Return the Strands failure converter class to use in place of ``base``.

    Only ``DefaultFailureConverterWithEncodedAttributes`` is kept, as
    ``StrandsFailureConverterWithEncodedAttributes``. Every other class is replaced
    with ``StrandsFailureConverter``, so the failures that Workflow code reads stay
    the same.
    """
    if base in (
        DefaultFailureConverterWithEncodedAttributes,
        StrandsFailureConverterWithEncodedAttributes,
    ):
        return StrandsFailureConverterWithEncodedAttributes
    return StrandsFailureConverter
