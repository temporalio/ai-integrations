"""Points at which a worker can take part in running a trial."""

from __future__ import annotations

import contextlib
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import JsonValue

from temporalio.harbor import _compat


@dataclass(frozen=True)
class TrialContext:
    """The trial an attempt is running, as :class:`TrialHooks` sees it.

    This class is experimental and may change in future versions.
    """

    config: TrialConfig
    """The trial, as the workflow passed it to :func:`execute_trial`."""

    retry: RetryConfig
    """Harbor's retry configuration for the trial."""

    attempt: int
    """The Activity attempt, counting from 1."""

    data: JsonValue
    """What the workflow passed as ``data`` to :func:`execute_trial`."""

    @property
    def trial_dir(self) -> Path:
        """Where harbor writes this trial's logs, artifacts and result."""
        return Path(self.config.trials_dir) / self.config.trial_name


@dataclass(frozen=True)
class TrialRetry:
    """Run the trial again, after ``delay``.

    This class is experimental and may change in future versions.
    """

    delay: timedelta

    def __post_init__(self) -> None:
        """Reject a negative delay."""
        if self.delay < timedelta(0):
            raise ValueError("delay must not be negative")


class TrialHooks:
    """Take part in how the trial Activity runs each trial.

    This class is experimental and may change in future versions.

    Subclass it and pass an instance to :class:`HarborPlugin`. Every method
    has a default, so override only what the worker needs. For each attempt,
    the Activity:

    1. enters :meth:`scope`, and inside it
    2. creates harbor's ``Trial`` and calls :meth:`trial_created` with it,
    3. runs the trial,
    4. if the trial recorded an exception, asks :meth:`retry` whether to run
       it again, raising out of the scope if so, and otherwise
    5. calls :meth:`output` and returns what it gives beside the result.

    Hooks run on the worker, so they may do I/O.
    """

    def scope(self, context: TrialContext) -> AbstractAsyncContextManager[None]:  # type: ignore[reportUnusedParameter]
        """Wrap everything the attempt does, from before the trial exists.

        Use it for setup the trial depends on, and for cleanup that must run
        however the attempt ends. An exception leaving the scope is how the
        attempt ends early: an ``ApplicationError`` when :meth:`retry` asks
        for another attempt, ``asyncio.CancelledError`` when the trial is
        cancelled, or whatever made the trial fail before harbor could record
        a result. Re-raise it; a scope that suppresses it fails the attempt.

        The default does nothing.
        """
        return contextlib.nullcontext()

    async def trial_created(self, trial: Trial, context: TrialContext) -> None:  # type: ignore[reportUnusedParameter]
        """Adjust harbor's ``Trial`` before it runs.

        For example, add lifecycle hooks with ``trial.add_hook``. The default
        does nothing.
        """

    def retry(self, result: TrialResult, context: TrialContext) -> TrialRetry | None:
        """Decide whether to run a trial that recorded an exception again.

        Called only when ``result.exception_info`` is set and the Activity has
        attempts left under its retry policy. On the last attempt the result
        is returned instead, so a recorded failure is never lost.

        The default is harbor's own decision for ``context.retry``: the same
        ``include_exceptions`` and ``exclude_exceptions``, the same
        ``max_retries``, and the same backoff, from harbor's ``TrialQueue``.
        Override it to change the decision for some failures and call
        ``super().retry(result, context)`` for the rest.

        Returns:
            When to run the trial again, or ``None`` to return this result.
        """
        exc = result.exception_info
        if exc is None or context.attempt > context.retry.max_retries:
            return None
        if not _compat.should_retry_exception(context.retry, exc.exception_type):
            return None
        # Harbor counts attempts from 0; this is the delay after that attempt.
        delay = _compat.backoff_delay_sec(context.retry, context.attempt - 1)
        return TrialRetry(delay=timedelta(seconds=delay))

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:  # type: ignore[reportUnusedParameter]
        """Produce what the Activity returns beside the trial's result.

        Called with the complete result, before it is slimmed, for the
        attempt whose result is returned. What it returns reaches the workflow
        as :attr:`TrialOutcome.output`, through workflow history, so keep it
        small. The default returns ``None``.
        """
        return None
