"""Configuration shared by durable agents, models, and the Worker plugin."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, cast

from agents import Agent, TResponseInputItem

from temporalio.openai_agents._errors import AgentsWorkflowError
from temporalio.workflow import ActivityConfig, LocalActivityConfig


class ModelSummaryProvider(ABC):
    """Provide deterministic summaries for model Activities."""

    @abstractmethod
    def provide(
        self,
        agent: Agent[Any] | None,
        instructions: str | None,
        input: str | list[TResponseInputItem],
    ) -> str:
        """Produce a summary for the model invocation Activity."""
        pass


@dataclass
class _ModelOptions:
    """Partial options; omitted fields inherit from the preceding layer."""

    activity_config: ActivityConfig = field(default_factory=ActivityConfig)
    use_local_activity: bool | None = None
    streaming_topic: str | None = None
    streaming_batch_interval: timedelta | None = None
    summary_provider: ModelSummaryProvider | None = None

    def __post_init__(self) -> None:
        self.activity_config = self.activity_config.copy()
        self._validate_summary()

    def _validate_summary(self) -> None:
        if (
            self.activity_config.get("summary") is not None
            and self.summary_provider is not None
        ):
            raise ValueError(
                "Cannot combine activity_config['summary'] with summary_provider"
            )

    @classmethod
    def merge(cls, *layers: "_ModelOptions") -> "_ModelOptions":
        """Merge partial options without modifying any caller's configuration."""
        result = cls()
        for layer in layers:
            result.activity_config.update(layer.activity_config)
            for name in (
                "use_local_activity",
                "streaming_topic",
                "streaming_batch_interval",
                "summary_provider",
            ):
                value = getattr(layer, name)
                if value is not None:
                    setattr(result, name, value)
        result._validate_summary()
        return result

    def config(self, *, streaming: bool = False) -> ActivityConfig:
        """Resolve defaults and validate before scheduling an Activity."""
        if streaming and self.streaming_topic is None:
            raise AgentsWorkflowError(
                "Runner.run_streamed requires streaming_topic to be set."
            )
        if streaming and self.use_local_activity:
            raise AgentsWorkflowError(
                "Runner.run_streamed is incompatible with use_local_activity "
                "(local activities do not support heartbeats or the workflow stream signal channel)."
            )
        config = ActivityConfig(start_to_close_timeout=timedelta(seconds=60))
        if self.use_local_activity:
            unsupported = sorted(
                set(self.activity_config) - set(LocalActivityConfig.__annotations__)
            )
            if unsupported:
                raise AgentsWorkflowError(
                    "Local model Activities do not support activity_config options: "
                    + ", ".join(unsupported)
                )
        else:
            config["heartbeat_timeout"] = timedelta(seconds=10)
        config.update(self.activity_config)
        if not (
            config.get("start_to_close_timeout")
            or config.get("schedule_to_close_timeout")
        ):
            raise AgentsWorkflowError(
                "Model activity_config requires start_to_close_timeout or schedule_to_close_timeout"
            )
        return config

    def local_config(self, config: ActivityConfig) -> LocalActivityConfig:
        """Narrow a validated local Activity configuration for the SDK."""
        return cast(LocalActivityConfig, cast(object, config))
