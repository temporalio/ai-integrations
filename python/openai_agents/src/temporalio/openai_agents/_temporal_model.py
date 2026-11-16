"""Public durable implementation of the OpenAI Agents Model interface."""

from datetime import timedelta

from agents import ModelSettings
from agents.models import get_default_model_settings

from temporalio.openai_agents._model_parameters import (
    ModelSummaryProvider,
    _ModelOptions,
)
from temporalio.openai_agents._temporal_model_stub import _TemporalModelStub
from temporalio.workflow import ActivityConfig


class TemporalModel(_TemporalModelStub):
    """Execute model calls in Activities using the provider configured on the Worker.

    May be used on TemporalAgent, upstream Agent (including SandboxAgent), or
    RunConfig.model. No provider clients or credentials belong on this object.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        activity_config: ActivityConfig | None = None,
        use_local_activity: bool | None = None,
        streaming_topic: str | None = None,
        streaming_batch_interval: timedelta | None = None,
        summary_provider: ModelSummaryProvider | None = None,
    ) -> None:
        """Configure durable model calls.

        Args:
            model_name: Name resolved by the Worker's model provider; None uses
                that provider's default model.
            activity_config: Per-field Activity overrides. Explicit None values
                override inherited fields. Static summaries use the summary key.
            use_local_activity: Whether to use local Activities; None inherits.
                Explicit options unsupported by local Activities are refused.
            streaming_topic: Workflow Stream topic, required for streaming
                (experimental). Streaming cannot use local Activities.
            streaming_batch_interval: Publishing interval (experimental);
                defaults to 100 milliseconds.
            summary_provider: Deterministic dynamic summary provider. Cannot be
                combined with a non-None static summary in effective configuration.
        """
        super().__init__(
            model_name,
            model_params=_ModelOptions(
                activity_config=activity_config or ActivityConfig(),
                use_local_activity=use_local_activity,
                streaming_topic=streaming_topic,
                streaming_batch_interval=streaming_batch_interval,
                summary_provider=summary_provider,
            ),
            agent=None,
        )

    def get_default_model_settings(self) -> ModelSettings | None:
        """Expose upstream model-name defaults without constructing a provider client."""
        return get_default_model_settings(self.model_name)

    def _effective_model_options(self) -> _ModelOptions:
        from temporalio.openai_agents._temporal_model_provider import (
            _current_agent,
            effective_model_options,
        )

        return effective_model_options(_current_agent.get(), self)
