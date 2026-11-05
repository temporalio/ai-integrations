"""Support for using the OpenAI Agents SDK as part of Temporal workflows.

This module provides compatibility between the
`OpenAI Agents SDK <https://github.com/openai/openai-agents-python>`_ and Temporal workflows.
"""

# Swap the openai item types' Iterable fields for lists before any class that holds them is built.
from temporalio.openai_agents import _openai_types  # noqa: F401  # isort: skip  # pyright: ignore[reportUnusedImport]
from temporalio.openai_agents._errors import AgentsWorkflowError
from temporalio.openai_agents._mcp import (
    StatefulMCPServerProvider,
    StatelessMCPServerProvider,
)
from temporalio.openai_agents._model_parameters import (
    ModelActivityParameters,
    ModelSummaryProvider,
)
from temporalio.openai_agents._temporal_openai_agents import (
    OpenAIAgentsPlugin,
    OpenAIPayloadConverter,
)
from temporalio.openai_agents._temporal_worker_env_ref import (
    AllowAllWorkerEnvVars,
    temporal_worker_env_ref,
)
from temporalio.openai_agents.sandbox._sandbox_client_provider import (
    SandboxClientProvider,
)
from temporalio.openai_agents.sandbox._temporal_worker_env_value import (
    TemporalWorkerEnvValue,
)
from temporalio.openai_agents.session import WorkflowSession

from . import testing, workflow

__all__ = [
    "AgentsWorkflowError",
    "AllowAllWorkerEnvVars",
    "ModelActivityParameters",
    "ModelSummaryProvider",
    "OpenAIAgentsPlugin",
    "OpenAIPayloadConverter",
    "SandboxClientProvider",
    "StatelessMCPServerProvider",
    "StatefulMCPServerProvider",
    "TemporalWorkerEnvValue",
    "WorkflowSession",
    "temporal_worker_env_ref",
    "testing",
    "workflow",
]
