"""Points at which a worker can take part in running a trial."""

from __future__ import annotations

import contextlib
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import JsonValue


@dataclass(frozen=True)
class TrialContext:
    """The trial an attempt is running, as :class:`TrialHooks` sees it.

    This class is experimental and may change in future versions.
    """

    config: TrialConfig
    """The ``config`` of the :class:`temporalio.harbor.TrialPlan` the workflow
    passed to :func:`temporalio.harbor.execute_trial`.

    After entering :meth:`TrialHooks.scope` and before calling
    :meth:`TrialHooks.trial_created`, the attempt replaces ``agent.skills``, in
    place, with the local paths its skill sources resolve to.
    """

    attempt: int
    """The Activity attempt, counting from 1."""

    data: JsonValue
    """What the workflow passed as ``data`` to
    :func:`temporalio.harbor.execute_trial`."""

    @property
    def trial_dir(self) -> Path:
        """Where harbor writes this trial's logs, artifacts and result."""
        return Path(self.config.trials_dir) / self.config.trial_name


class TrialHooks:
    """Take part in how the trial Activity runs each trial.

    This class is experimental and may change in future versions.

    Subclass it and pass an instance to
    :class:`temporalio.harbor.HarborPlugin`. Every method has a default, so
    override only what the worker needs. For each attempt, the Activity:

    1. waits for any earlier attempt of the same trial on this worker to
       finish, then deletes the trial's directory,
    2. enters :meth:`scope`, and inside it
    3. creates harbor's ``Trial`` and calls :meth:`trial_created` with it,
    4. runs the trial, and if harbor produced a result,
    5. passes its text through :meth:`redact` and calls :meth:`output`.

    The Activity returns the redacted result with what :meth:`output` gave.
    It raises instead, and Temporal retries the trial, when the trial could
    not run for a reason other than its configuration, failed before its
    agent started, or its agent's run ended in a transient provider error
    (see :func:`temporalio.harbor.execute_trial`).

    Hooks run on the worker's event loop, beside every other trial the worker
    is running, so do I/O with ``async`` libraries or ``asyncio.to_thread``.
    Together, the async hooks of one attempt may spend the ``extra_time``
    given to :func:`temporalio.harbor.execute_trial`; a hook that runs past
    it fails the attempt for Temporal to retry.
    """

    def scope(self, context: TrialContext) -> AbstractAsyncContextManager[None]:  # type: ignore[reportUnusedParameter]
        """Wrap everything the attempt does, from before the trial exists.

        Use it for setup the trial depends on, and for cleanup that must run
        however the attempt ends. An exception leaving the scope is how the
        attempt ends early: an ``ApplicationError`` from :meth:`output`, or
        from the Activity when Temporal is to run the trial again or when
        :meth:`output` or :meth:`redact` failed; ``asyncio.CancelledError``
        when the attempt is cancelled, including when the worker shuts down
        or the Activity is paused; or whatever made the trial fail before
        harbor could record a result. It reaches Temporal as it leaves the
        scope, so a scope may raise its own exception from it. A scope that
        suppresses an exception fails the attempt anyway when the trial
        produced no result.

        The default does nothing.
        """
        return contextlib.nullcontext()

    async def trial_created(self, trial: Trial, context: TrialContext) -> None:  # type: ignore[reportUnusedParameter]
        """Adjust harbor's ``Trial`` before it runs.

        For example, add lifecycle hooks with ``trial.add_hook``. The default
        does nothing.
        """

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:  # type: ignore[reportUnusedParameter]
        """Produce what the Activity returns beside the trial's result.

        Called with harbor's complete result, unredacted, for every attempt
        that produced one, including one that records an exception. What it
        returns reaches the workflow as
        :attr:`temporalio.harbor.TrialOutcome.output`, through workflow
        history and without passing through :meth:`redact`, so keep it small
        and free of secrets.

        To have Temporal run the trial again, raise an ``ApplicationError``;
        its type, message, ``non_retryable`` and ``next_retry_delay`` are
        yours to choose. The Activity attaches the attempt's outcome to its
        details, so when Temporal stops retrying,
        :func:`temporalio.harbor.execute_trial` returns the most recent
        outcome any attempt produced. Any other exception, or a return value
        that cannot be converted to JSON or that brings the outcome past
        about 2 MB, fails the attempt without a retry, and
        :func:`temporalio.harbor.execute_trial` returns the attempt's outcome
        without an ``output``.

        The default returns ``None``.
        """
        return None

    def redact(self, text: str, context: TrialContext) -> str:  # type: ignore[reportUnusedParameter]
        """Rewrite text before it leaves the worker for Temporal history.

        Called on every string value in a trial's result, before long
        exception text is shortened, and on the message and stack trace of
        every error the attempt raises, causes included. Use it to remove
        credentials or paths that harbor recorded. It runs on the event loop
        and must not block.

        It does not see dictionary keys, what :meth:`output` returns, the
        details of an ``ApplicationError`` a hook raises, or the text
        :func:`temporalio.harbor.aggregate_job` reads from a result, which
        reaches the workflow as harbor recorded it: the result's
        ``trial_name``, ``task_name``, ``source`` and ``started_at``, its
        agent's name and model name, and the type of its recorded exception.
        A value in the result that is not free text, such as a UUID, a
        timestamp or a host the agent is allowed to reach, keeps the value
        harbor recorded when the rewritten text is not valid for it.

        If it raises or returns something other than a ``str``, the attempt
        fails without a retry and sends none of its error's text, and
        :func:`temporalio.harbor.execute_trial` returns the most recent
        outcome any attempt produced, if there is one.

        The default returns ``text`` unchanged.
        """
        return text
