"""Configuration shared by durable agents, models, and the Worker plugin."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from agents import Agent, TResponseInputItem

from temporalio.openai_agents._errors import AgentsWorkflowError
from temporalio.workflow import ActivityConfig, LocalActivityConfig

if TYPE_CHECKING:
    from temporalio.openai_agents._model_parameters import ModelActivityParameters


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
    legacy_fields: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.activity_config = self.activity_config.copy()
        self.legacy_fields = self.legacy_fields.copy()
        self._validate_summary(ValueError)

    def _validate_summary(self, error_type: type[Exception]) -> None:
        if (
            self.activity_config.get("summary") is not None
            and self.summary_provider is not None
        ):
            raise error_type(
                "Cannot combine activity_config['summary'] with summary_provider"
            )

    @classmethod
    def merge(cls, *layers: "_ModelOptions") -> "_ModelOptions":
        """Merge partial options without modifying any caller's configuration."""
        result = cls()
        for layer in layers:
            result.activity_config.update(layer.activity_config)
            result.legacy_fields.difference_update(layer.activity_config)
            result.legacy_fields.update(layer.legacy_fields)
            for name in (
                "use_local_activity",
                "streaming_topic",
                "streaming_batch_interval",
                "summary_provider",
            ):
                value = getattr(layer, name)
                if value is not None:
                    setattr(result, name, value)
        result._validate_summary(AgentsWorkflowError)
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
        ignored: set[str] = set()
        if self.use_local_activity:
            ignored = self.legacy_fields - set(LocalActivityConfig.__annotations__)
            unsupported = sorted(
                set(self.activity_config)
                - set(LocalActivityConfig.__annotations__)
                - ignored
            )
            if unsupported:
                raise AgentsWorkflowError(
                    "Local model Activities do not support activity_config options: "
                    + ", ".join(unsupported)
                )
        else:
            config["heartbeat_timeout"] = timedelta(seconds=10)
        config.update(self.activity_config)
        for key in ignored:
            cast(dict[str, object], cast(object, config)).pop(key, None)
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


def resolve_model_options(
    model_params: ModelActivityParameters | None,
    *,
    activity_config: ActivityConfig | None = None,
    use_local_activity: bool | None = None,
    streaming_topic: str | None = None,
    streaming_batch_interval: timedelta | None = None,
    summary_provider: ModelSummaryProvider | None = None,
    custom_provider: bool = False,
) -> _ModelOptions:
    """Translate the deprecated constructor argument without changing its defaults."""
    if model_params is None:
        return _ModelOptions(
            activity_config=activity_config
            if activity_config is not None
            else ActivityConfig(),
            use_local_activity=use_local_activity,
            streaming_topic=streaming_topic,
            streaming_batch_interval=streaming_batch_interval,
            summary_provider=summary_provider,
        )
    warnings.warn(
        "model_params is deprecated; use activity_config and separate execution options instead",
        DeprecationWarning,
        stacklevel=3,
    )
    if any(
        value is not None
        for value in (
            activity_config,
            use_local_activity,
            streaming_topic,
            streaming_batch_interval,
            summary_provider,
        )
    ):
        raise ValueError(
            "Cannot combine model_params with activity_config or separate execution options"
        )
    config = cast(
        ActivityConfig,
        cast(
            object,
            {
                item.name: getattr(model_params, item.name)
                for item in fields(model_params)
                if item.name in ActivityConfig.__annotations__
            },
        ),
    )
    if (
        config.get("start_to_close_timeout") is None
        and config.get("schedule_to_close_timeout") is None
    ):
        if custom_provider:
            raise ValueError(
                "When configuring a custom provider, the model activity must have "
                "start_to_close_timeout or schedule_to_close_timeout"
            )
        config["start_to_close_timeout"] = timedelta(seconds=60)
    summary = model_params.summary_override
    if isinstance(summary, str):
        if summary:
            config["summary"] = summary
    elif summary is not None:
        summary_provider = summary
    return _ModelOptions(
        activity_config=config,
        use_local_activity=model_params.use_local_activity,
        streaming_topic=model_params.streaming_topic,
        streaming_batch_interval=model_params.streaming_batch_interval,
        summary_provider=summary_provider,
        # Legacy local Activities ignore regular-only fields. Retain them for
        # per-run overrides that explicitly switch back to regular Activities.
        legacy_fields=set(config),
    )
