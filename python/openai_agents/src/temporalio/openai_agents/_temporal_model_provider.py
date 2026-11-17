"""Model provider that resolves model names to Temporal activity stubs."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Awaitable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from agents import Agent, Model, ModelProvider, ModelSettings, RunConfig
from agents.run_config import CallModelData, CallModelInputFilter, ModelInputData

from temporalio.openai_agents._model_options import _ModelOptions
from temporalio.openai_agents._temporal_agent import TemporalAgent
from temporalio.openai_agents._temporal_model import TemporalModel

_current_agent: ContextVar[Agent[Any] | None] = ContextVar(
    "temporal_openai_agents_current_agent", default=None
)
# The run-level ``RunConfig.model_settings`` of the run in progress. Model stubs receive
# settings already merged with these, so they are carried separately to the activity.
_current_run_model_settings: ContextVar[ModelSettings | None] = ContextVar(
    "temporal_openai_agents_run_model_settings", default=None
)
_current_model_defaults: ContextVar[_ModelOptions | None] = ContextVar(
    "temporal_openai_agents_model_defaults", default=None
)
_current_run_model: ContextVar[TemporalModel | None] = ContextVar(
    "temporal_openai_agents_run_model", default=None
)


def effective_model_options(
    agent: Agent[Any] | None,
    model: TemporalModel | None = None,
) -> _ModelOptions:
    """Resolve Worker, model, agent, then run overrides per field."""
    layers = [_current_model_defaults.get() or _ModelOptions()]
    if agent is not None and isinstance(agent.model, TemporalModel):
        layers.append(agent.model._options)
    elif agent is None and model is not None:
        layers.append(model._options)
    if isinstance(agent, TemporalAgent):
        layers.append(agent._model_options())
    run_model = _current_run_model.get()
    if run_model is not None:
        layers.append(run_model._options)
    return _ModelOptions.merge(*layers)


def validate_agent(agent: Agent[Any]) -> None:
    """Refuse models that would run provider code inside the Workflow."""
    if agent.model is not None and not isinstance(agent.model, (str, TemporalModel)):
        raise ValueError(
            "Temporal workflows require a model name to be a string or TemporalModel in the agent."
        )
    if not isinstance(agent, TemporalAgent) and not isinstance(
        agent.model, TemporalModel
    ):
        warnings.warn(
            "Using Agent with a string or implicit model is deprecated; use TemporalAgent "
            "or Agent(model=TemporalModel(...)) instead",
            DeprecationWarning,
            stacklevel=3,
        )


@contextmanager
def capture_scope(
    run_model_settings: ModelSettings | None = None,
    *,
    defaults: _ModelOptions | None = None,
    run_model: Model | str | None = None,
) -> Generator[None, None, None]:
    """Scope one run: expose its run-level model settings and discard captured agents on exit.

    Must be entered and exited in the same task. Entering removes any agent captured by an
    enclosing run, and exiting restores the caller's context so the variables are unset again.
    """
    token = _current_agent.set(None)
    settings_token = _current_run_model_settings.set(run_model_settings)
    defaults_token = _current_model_defaults.set(defaults)
    model_token = _current_run_model.set(
        run_model if isinstance(run_model, TemporalModel) else None
    )
    try:
        yield
    finally:
        _current_run_model.reset(model_token)
        _current_model_defaults.reset(defaults_token)
        _current_run_model_settings.reset(settings_token)
        _current_agent.reset(token)


class _CaptureAgentFilter:
    """A ``call_model_input_filter`` that records the active agent, then defers to the user's."""

    def __init__(
        self, wrapped: CallModelInputFilter | None, *, streaming: bool
    ) -> None:
        self.wrapped = wrapped
        self.streaming = streaming

    def __call__(
        self, data: CallModelData[Any]
    ) -> ModelInputData | Awaitable[ModelInputData]:
        """Record ``data.agent`` and return the wrapped filter's result unchanged.

        Raises:
            ValueError: If the agent could call a provider inside the Workflow.
        """
        validate_agent(data.agent)
        effective_model_options(data.agent).config(streaming=self.streaming)
        _current_agent.set(data.agent)
        if self.wrapped is None:
            return data.model_data
        return self.wrapped(data)


class _TemporalModelProvider(ModelProvider):
    """Resolves string and ``None`` model names to stubs that run as Temporal activities.

    ``ModelProvider.get_model`` only receives the model name, so the agent used for activity
    summaries is captured by :meth:`install` from ``CallModelData`` and read when the stub
    builds the activity input. The agent lives in a ``ContextVar`` so that concurrent runs,
    such as agents used as tools, each see their own agent. :func:`capture_scope` clears it
    when a run ends.
    """

    def __init__(self, *, streaming: bool = False) -> None:
        self.streaming = streaming

    def get_model(self, model_name: str | None) -> Model:
        """Return the durable model for a name or the Worker's default model."""
        model = TemporalModel(model_name)
        model._is_provider_default = model_name is None
        return model

    def install(self, run_config: RunConfig) -> RunConfig:
        """Return ``run_config`` using this provider and an agent-capturing input filter."""
        user_filter = run_config.call_model_input_filter
        if isinstance(user_filter, _CaptureAgentFilter):
            user_filter = user_filter.wrapped
        capture_filter = _CaptureAgentFilter(user_filter, streaming=self.streaming)
        return dataclasses.replace(
            run_config, model_provider=self, call_model_input_filter=capture_filter
        )


def install_model_provider(
    run_config: RunConfig,
    *,
    streaming: bool = False,
) -> RunConfig:
    """Route model resolution in ``run_config`` through the Temporal model provider.

    A nested run inherits the parent's provider when the execution mode agrees.
    """
    provider = run_config.model_provider
    if (
        not isinstance(provider, _TemporalModelProvider)
        or provider.streaming != streaming
    ):
        provider = _TemporalModelProvider(streaming=streaming)
    return provider.install(run_config)
