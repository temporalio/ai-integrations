"""Model provider that resolves model names to Temporal activity stubs."""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable
from contextvars import ContextVar
from typing import Any

from agents import Agent, Model, ModelProvider, RunConfig
from agents.run_config import CallModelData, CallModelInputFilter, ModelInputData

from temporalio.openai_agents._model_parameters import ModelActivityParameters
from temporalio.openai_agents._temporal_model_stub import _TemporalModelStub


class _CaptureAgentFilter:
    """A ``call_model_input_filter`` that records the active agent, then defers to the user's."""

    def __init__(
        self,
        current_agent: ContextVar[Agent[Any] | None],
        wrapped: CallModelInputFilter | None,
    ) -> None:
        self.current_agent = current_agent
        self.wrapped = wrapped

    def __call__(
        self, data: CallModelData[Any]
    ) -> ModelInputData | Awaitable[ModelInputData]:
        """Record ``data.agent`` and return the wrapped filter's result unchanged."""
        self.current_agent.set(data.agent)
        if self.wrapped is None:
            return data.model_data
        return self.wrapped(data)


class _TemporalModelProvider(ModelProvider):
    """Resolves string and ``None`` model names to stubs that run as Temporal activities.

    ``ModelProvider.get_model`` only receives the model name, so the agent used for activity
    summaries is captured by :meth:`install` from ``CallModelData`` and read when the stub
    builds the activity input. The agent lives in a ``ContextVar`` so that concurrent runs,
    such as agents used as tools, each see their own agent.
    """

    def __init__(self, model_params: ModelActivityParameters) -> None:
        self._model_params = model_params
        self._current_agent: ContextVar[Agent[Any] | None] = ContextVar(
            "temporal_openai_agents_current_agent", default=None
        )

    def get_model(self, model_name: str | None) -> Model:
        """Return a stub that runs the model call as a Temporal activity."""
        return _TemporalModelStub(
            model_name,
            model_params=self._model_params,
            agent=None,
            current_agent=self._current_agent.get,
        )

    def install(self, run_config: RunConfig) -> RunConfig:
        """Return ``run_config`` using this provider and an agent-capturing input filter."""
        user_filter = run_config.call_model_input_filter
        if (
            isinstance(user_filter, _CaptureAgentFilter)
            and user_filter.current_agent is self._current_agent
        ):
            capture_filter = user_filter
        else:
            capture_filter = _CaptureAgentFilter(self._current_agent, user_filter)
        return dataclasses.replace(
            run_config, model_provider=self, call_model_input_filter=capture_filter
        )


def install_model_provider(
    model_params: ModelActivityParameters, run_config: RunConfig
) -> RunConfig:
    """Route model resolution in ``run_config`` through the Temporal model provider.

    A nested run, such as an agent used as a tool, inherits the parent's ``run_config`` and
    keeps its provider.
    """
    provider = run_config.model_provider
    if not isinstance(provider, _TemporalModelProvider):
        provider = _TemporalModelProvider(model_params)
    return provider.install(run_config)
