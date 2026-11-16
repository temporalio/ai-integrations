"""Worker plugin that runs harbor trials as Activities."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import weakref
from typing import Any

from harbor.auth import tokens as harbor_tokens
from harbor.utils.logger import logger as harbor_logger
from pydantic_core import to_json

from temporalio.api.common.v1 import Payload
from temporalio.contrib.pydantic import (
    PydanticJSONPlainPayloadConverter,
    PydanticPayloadConverter,
)
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.harbor._activity import HarborActivities
from temporalio.harbor._hooks import TrialHooks
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


class _UnmaskedJSONPayloadConverter(PydanticJSONPlainPayloadConverter):
    def to_payload(self, value: Any) -> Payload | None:
        # Harbor masks agent env values whose names look secret unless told
        # not to, which would hand the trial "****" in their place.
        return Payload(
            metadata={"encoding": self.encoding.encode()},
            data=to_json(value, context={"redact_sensitive_env": False}),
        )


class _PayloadConverter(PydanticPayloadConverter):
    def __init__(self) -> None:
        super().__init__()
        self._set_converters(
            *(
                _UnmaskedJSONPayloadConverter()
                if isinstance(c, PydanticJSONPlainPayloadConverter)
                else c
                for c in self.converters.values()
            )
        )


class _ForwardHarborLogs(logging.Handler):
    """Pass INFO and above from harbor's logger on to the worker's handlers."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)

    def emit(self, record: logging.LogRecord) -> None:
        if harbor_logger.parent is not None:
            harbor_logger.parent.handle(record)


def _quiet_harbor_logs() -> None:
    if any(isinstance(h, _ForwardHarborLogs) for h in harbor_logger.handlers):
        return
    harbor_logger.addHandler(_ForwardHarborLogs())
    harbor_logger.propagate = False


class _LockPerLoop:
    """An ``asyncio.Lock`` for each event loop that enters it."""

    def __init__(self) -> None:
        self._locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Lock
        ] = weakref.WeakKeyDictionary()

    def _lock(self) -> asyncio.Lock:
        return self._locks.setdefault(asyncio.get_running_loop(), asyncio.Lock())

    async def __aenter__(self) -> None:
        await self._lock().acquire()

    async def __aexit__(self, *_: object) -> None:
        self._lock().release()


def _lock_harbor_tokens_per_loop() -> None:
    # Harbor guards its auth token with one asyncio.Lock for the process, and
    # planning runs harbor on loops of its own: a lock used on two loops raises.
    cache = harbor_tokens._cache
    if not isinstance(cache._lock, _LockPerLoop):
        cache._lock = _LockPerLoop()  # type: ignore[reportAttributeAccessIssue]


class HarborPlugin(SimplePlugin):
    """Run harbor evaluation jobs durably, one Activity per trial.

    This class is experimental and may change in future versions.

    Registers the Activities that :func:`temporalio.harbor.plan_job`,
    :func:`temporalio.harbor.execute_trial` and
    :func:`temporalio.harbor.aggregate_job` schedule, and passes harbor
    through the workflow sandbox. Harbor's configuration and results are
    pydantic models, so the plugin makes a pydantic payload converter the
    data converter's payload converter, keeping any payload codec and
    failure converter. A custom payload converter is left in place.

    On each Worker, harbor's logger, which logs at DEBUG, passes only INFO
    and above on to the worker's handlers; each trial's ``trial.log`` keeps
    its DEBUG records.

    Register it on the Client, so both workflows and the starter use the same
    converter, and it is applied to every Worker built from that Client.
    """

    def __init__(self, *, trial_hooks: TrialHooks | None = None) -> None:
        """Create the plugin.

        Args:
            trial_hooks: How this worker takes part in running each trial.
                Defaults to :class:`temporalio.harbor.TrialHooks`, which runs
                each trial as harbor's own ``Trial`` runs it.
        """
        activities = HarborActivities(
            hooks=trial_hooks if trial_hooks is not None else TrialHooks()
        )

        def data_converter(converter: DataConverter | None) -> DataConverter:
            if converter is None:
                return DataConverter(payload_converter_class=_PayloadConverter)
            if converter.payload_converter_class is DefaultPayloadConverter:
                return dataclasses.replace(
                    converter, payload_converter_class=_PayloadConverter
                )
            return converter

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if runner is None:
                raise ValueError("No WorkflowRunner provided to the harbor plugin")
            if isinstance(runner, SandboxedWorkflowRunner):
                # Workflow code builds and reads harbor's pydantic models; they
                # must be the same classes the data converter produces.
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules("harbor"),
                )
            return runner

        super().__init__(
            name="HarborPlugin",
            data_converter=data_converter,
            activities=[
                activities.plan_job,
                activities.run_trial,
                activities.aggregate_job,
            ],
            workflow_runner=workflow_runner,
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """See base class."""
        _quiet_harbor_logs()
        _lock_harbor_tokens_per_loop()
        return super().configure_worker(config)
