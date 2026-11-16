"""An upstream Agent with per-agent Temporal Activity configuration."""

from dataclasses import dataclass, field
from datetime import timedelta

from agents import Agent, TContext

from temporalio.openai_agents._model_options import (
    ModelSummaryProvider,
    _ModelOptions,
)
from temporalio.openai_agents._temporal_model import TemporalModel
from temporalio.workflow import ActivityConfig


@dataclass
class TemporalAgent(Agent[TContext]):
    """Run an OpenAI agent with durable model calls.

    Inherits the upstream Agent arguments, hooks, tools, handoffs, and cloning.
    String model names remain visible to the SDK for its model-specific defaults.
    Temporal options are keyword-only and inherit from the Worker and model.
    """

    activity_config: ActivityConfig | None = field(default=None, kw_only=True)
    """Per-field Activity overrides. Explicit None values override inherited fields."""

    use_local_activity: bool | None = field(default=None, kw_only=True)
    """Use local Activities; None inherits the Worker's or model's choice."""

    streaming_topic: str | None = field(default=None, kw_only=True)
    """Workflow Stream topic for streamed model calls (experimental)."""

    streaming_batch_interval: timedelta | None = field(default=None, kw_only=True)
    """Stream publishing interval; defaults to 100 milliseconds (experimental)."""

    summary_provider: ModelSummaryProvider | None = field(default=None, kw_only=True)
    """Deterministic dynamic summary; cannot be combined with a static summary."""

    def __post_init__(self) -> None:
        """Preserve upstream initialization and snapshot Activity configuration."""
        super().__post_init__()
        if self.model is not None and not isinstance(self.model, (str, TemporalModel)):
            raise ValueError(
                "TemporalAgent.model must be a string, None, or TemporalModel"
            )
        if self.activity_config is not None:
            self.activity_config = self.activity_config.copy()
        self._model_options()

    def _model_options(self) -> _ModelOptions:
        return _ModelOptions(
            activity_config=self.activity_config or ActivityConfig(),
            use_local_activity=self.use_local_activity,
            streaming_topic=self.streaming_topic,
            streaming_batch_interval=self.streaming_batch_interval,
            summary_provider=self.summary_provider,
        )
