"""Worker plugin registering TypeSafe Activities.

:class:`TypeSafePlugin` holds a caller-configured SDK client worker-side, so
credentials stay out of workflow history, and appends the ``system_one`` Activity to
every Worker built.
Workflow code talks to it through
:class:`temporalio.typesafe.TemporalTypeSafe`.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import sys
from collections.abc import Mapping
from typing import Any

from typesafe_sdk import (
    AsyncTypeSafeClient,
    SystemOneResponse,
    TypeSafeError,
)

from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.plugin import SimplePlugin
from temporalio.typesafe._activities import TypeSafeActivities
from temporalio.typesafe._types import register_response_model
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

_SDK_RETRY_MESSAGE = (
    "The SDK must not retry inside one Activity attempt; Temporal owns "
    "retries between attempts and the two loops would compound. Configure "
    "the supplied client's retry policy with max_retries=0 and Temporal's "
    "policy through TemporalTypeSafe(activity_config={'retry_policy': ...})."
)

_UNBOUNDED = sys.maxsize
"""Attempt-count stance of a tenacity stop node that caps by time, not count."""


def _reject_sdk_retries(client: AsyncTypeSafeClient) -> None:
    """Turn an SDK retry policy that retries into a construction error."""
    if _client_retries_enabled(client):
        raise TypeSafeError(_SDK_RETRY_MESSAGE)


def _client_retries_enabled(client: AsyncTypeSafeClient) -> bool:
    """Whether the client's built retry loop performs more than one attempt.

    The SDK collapses the configured :class:`~typesafe_sdk.RetryPolicy` into a
    tenacity stop policy at construction, so the policy itself is gone by the
    time a client is injected.
    """
    return _attempt_cap(client._retry.stop) > 1


def _attempt_cap(stop: Any) -> int:
    """How many attempts one tenacity stop node allows; ``_UNBOUNDED`` is no cap.

    Combined stops trigger when any member fires, so choose the smallest cap.
    """
    stops = getattr(stop, "stops", None)
    if stops is not None:
        return min(_attempt_cap(one) for one in stops)
    attempts = getattr(stop, "max_attempt_number", None)
    if attempts is not None:
        return attempts
    # stop_before_delay and anything unfamiliar cap by time, not by count.
    return _UNBOUNDED


def _data_converter(converter: DataConverter | None) -> DataConverter:
    """Upgrade the default payload converter to the Pydantic one, keeping the rest."""
    if converter is None:
        return DataConverter(payload_converter_class=PydanticPayloadConverter)
    if converter.payload_converter_class is DefaultPayloadConverter:
        return dataclasses.replace(
            converter, payload_converter_class=PydanticPayloadConverter
        )
    return converter


class TypeSafePlugin(SimplePlugin):
    """Register the TypeSafe ``system_one`` Activity on a Worker.

    Args:
        client: A fully configured ``typesafe_sdk.AsyncTypeSafeClient``.
            The caller owns credentials, endpoint, HTTP options, retries, and
            closing the client. Configure the SDK retry policy with
            ``max_retries=0`` so Temporal owns retries between Activity
            attempts.
        response_models: Registry for ``TemporalTypeSafe(...,
            response_model="name")`` calls: names mapped to subclasses of
            ``typesafe_sdk.SystemOneResponse``, validated in the Activity and
            used to decode workflow results.

    Raises:
        TypeSafeError: The supplied client enables SDK-level retries or a
            response model is invalid.
    """

    def __init__(
        self,
        client: AsyncTypeSafeClient,
        *,
        response_models: Mapping[str, type[SystemOneResponse]] | None = None,
    ) -> None:
        """Register Activities that use a caller-configured SDK client."""
        _reject_sdk_retries(client)
        self._register_response_models(response_models)
        support = TypeSafeActivities(client)
        self._support = support

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if runner is None:
                raise ValueError("No WorkflowRunner provided to the TypeSafe plugin")
            if isinstance(runner, SandboxedWorkflowRunner):
                # httpx2 subclasses urllib.request.Request at import time,
                # which the sandbox's __mro_entries__ rule rejects. The SDK's
                # pydantic question/response models run in the workflow too;
                # annotated_types is outside the SDK's default passthrough,
                # and pydantic's compiled core is not in any default
                # passthrough, so the whole chain passes through.
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "typesafe_sdk",
                        "pydantic",
                        "pydantic_core",
                        "httpx2",
                        "annotated_types",
                    ),
                )
            return runner

        super().__init__(
            name="TypeSafePlugin",
            data_converter=_data_converter,
            activities=support.activities,
            workflow_runner=workflow_runner,
        )

    def version(self) -> str:
        """Return this package's version, which worker heartbeats report."""
        try:
            return importlib.metadata.version("temporalio-typesafe")
        except importlib.metadata.PackageNotFoundError:
            return ""

    @staticmethod
    def _register_response_models(
        response_models: Mapping[str, type[SystemOneResponse]] | None,
    ) -> None:
        """Validate and register into the module-level registry.

        Validation runs at Worker startup; the Activities and workflow code
        then decode by ``lookup_response_model``. A re-registered class
        overwrites nothing usable.
        """
        if response_models is None:
            return
        for name, model in response_models.items():
            if not (isinstance(model, type) and issubclass(model, SystemOneResponse)):
                raise TypeSafeError(
                    f"response_models[{name!r}] must subclass typesafe_sdk.SystemOneResponse"
                )
            register_response_model(name, model)
