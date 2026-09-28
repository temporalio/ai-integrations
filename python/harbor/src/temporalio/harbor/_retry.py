"""Harbor's trial retry policy, expressed as a Temporal RetryPolicy."""

from __future__ import annotations

from datetime import timedelta

from harbor.models.job.config import RetryConfig

from temporalio.common import RetryPolicy


def retry_policy_from_harbor(
    config: RetryConfig, *, infrastructure_retries: int = 3
) -> RetryPolicy:
    """Build the RetryPolicy that runs a trial the way ``harbor run`` retries it.

    Harbor waits ``min_wait_sec * wait_multiplier ** n`` after the ``n``-th
    failed attempt (counting from zero), capped at ``max_wait_sec``. Temporal's
    schedule has the same shape, so the backoff carries over exactly.

    Which failures are retried is decided inside the trial Activity against the
    same ``config``, because a Temporal RetryPolicy cannot see the exception a
    trial recorded. Temporal also spends an attempt when a worker is lost or a
    trial fails before harbor can record anything, which harbor never sees;
    ``infrastructure_retries`` is the extra budget for those.

    Args:
        config: Harbor's retry configuration, usually ``JobConfig.retry``.
        infrastructure_retries: Attempts allowed beyond harbor's own
            ``max_retries`` for failures harbor does not record.

    Raises:
        ValueError: If ``infrastructure_retries`` is negative, or if
            ``wait_multiplier`` is below 1, which Temporal cannot express.
    """
    if infrastructure_retries < 0:
        raise ValueError("infrastructure_retries must not be negative")
    if config.wait_multiplier < 1:
        raise ValueError(
            f"wait_multiplier {config.wait_multiplier} shrinks the delay between "
            "retries; Temporal's backoff coefficient must be at least 1"
        )
    # Harbor clamps every delay to max_wait_sec, including the first one.
    first = min(config.min_wait_sec, config.max_wait_sec)
    return RetryPolicy(
        initial_interval=timedelta(seconds=first),
        backoff_coefficient=config.wait_multiplier,
        maximum_interval=timedelta(seconds=config.max_wait_sec),
        maximum_attempts=config.max_retries + 1 + infrastructure_retries,
    )
