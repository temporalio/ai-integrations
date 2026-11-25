"""Parameters for configuring Temporal activity execution for model calls."""

import warnings
from dataclasses import dataclass
from datetime import timedelta

from temporalio.common import Priority, RetryPolicy
from temporalio.openai_agents._model_options import ModelSummaryProvider
from temporalio.workflow import ActivityCancellationType, VersioningIntent


@dataclass
class ModelActivityParameters:
    """Parameters for configuring Temporal activity execution for model calls.

    This class encapsulates all the parameters that can be used to configure
    how Temporal activities are executed when making model calls through the
    OpenAI Agents integration.

    .. deprecated:: 2.x
        Use ActivityConfig via activity_config and the separate execution options
        on OpenAIAgentsPlugin and AgentEnvironment instead.
    """

    task_queue: str | None = None
    """Specific task queue to use for model activities."""

    schedule_to_close_timeout: timedelta | None = None
    """Maximum time from scheduling to completion."""

    schedule_to_start_timeout: timedelta | None = None
    """Maximum time from scheduling to starting."""

    start_to_close_timeout: timedelta | None = timedelta(seconds=60)
    """Maximum time for the activity to complete."""

    heartbeat_timeout: timedelta | None = timedelta(seconds=10)
    """Maximum time between heartbeats. The model activities heartbeat on a
    timer at half this timeout, so a cancel reaches a running model call, and a
    lost worker is noticed, within about this long. It doesn't catch a slow
    model call. Set it to ``None`` to turn heartbeats off; a cancel then doesn't
    reach a running model call. For streaming, that includes the cancel when a
    parallel input guardrail trips, so the activity keeps publishing to
    ``streaming_topic``."""

    retry_policy: RetryPolicy | None = None
    """Policy for retrying failed activities."""

    cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL
    """How the activity handles cancellation. For streaming, ``TRY_CANCEL``
    lets the run go on while the activity stops, so the ``error`` event it
    publishes can come after the workflow closes or among later events on
    ``streaming_topic``. ``WAIT_CANCELLATION_COMPLETED`` waits until the
    activity has published it. ``ABANDON`` sends no cancel, so the activity
    keeps publishing."""

    versioning_intent: VersioningIntent | None = None
    """Versioning intent for the activity."""

    summary_override: None | (str | ModelSummaryProvider) = None
    """Summary for the activity execution."""

    priority: Priority = Priority.default
    """Priority for the activity execution."""

    use_local_activity: bool = False
    """Whether to use a local activity. If changed during a workflow execution, that would break determinism."""

    streaming_topic: str | None = None
    """Stream topic to publish raw model stream events to when the workflow
    calls ``Runner.run_streamed``. Required for ``Runner.run_streamed``;
    if left as ``None``, ``run_streamed`` raises before scheduling any
    activity. The workflow must host a
    :class:`temporalio.contrib.workflow_streams.WorkflowStream` to receive
    the publishes; otherwise the signals are unhandled and dropped.

    Streaming is incompatible with ``use_local_activity`` (local activities
    do not support heartbeats or the workflow stream signal channel).

    .. warning::
        Streaming support is experimental and may change in future
        versions."""

    streaming_batch_interval: timedelta = timedelta(milliseconds=100)
    """Interval between automatic flushes for the stream publisher used
    by the streaming activity.

    .. warning::
        Streaming support is experimental and may change in future
        versions."""

    def __post_init__(self) -> None:
        """Warn when constructing the deprecated configuration type."""
        warnings.warn(
            "ModelActivityParameters is deprecated; use activity_config=ActivityConfig(...) "
            "and separate execution options instead",
            DeprecationWarning,
            stacklevel=2,
        )
