"""Agent workflows imported only by fresh interpreter processes.

Each case installs a stand-in for one package, written to a directory on the
process's path. A stand-in does at import what the real package does and the
workflow sandbox refuses.
"""

import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel

from temporalio import workflow
from temporalio.client import Client, WorkflowHistory
from temporalio.deepagents import DeepAgentsPlugin, create_temporal_deep_agent
from temporalio.deepagents.testing import mock_model_provider
from temporalio.worker import Replayer, Worker

# Each middleware appends its package name to the model's reply, so the
# workflow result shows that deepagents found it.
_MIDDLEWARE = """
from langchain.agents.middleware.types import AgentMiddleware

from {package}.chat_models import {model}


class {middleware}(AgentMiddleware):
    def __init__(self, unsupported_model_behavior="warn"):
        self.unsupported_model_behavior = unsupported_model_behavior

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        response.result[-1].content += " [{package}]"
        return response
"""

# httpx2 subclasses this at import. The OpenAI SDK imports httpx2, and so does
# langchain_openai itself.
_HTTPX2 = """
import urllib.request


class Request(urllib.request.Request):
    pass
"""

_STAND_INS: dict[str, dict[str, str]] = {
    "langchain_aws": {
        # The real package reads the environment at import, after it imports
        # its submodules.
        "langchain_aws/__init__.py": """
from langchain_aws._version import __version__
from langchain_aws.chat_models import ChatBedrock


def setup_logging():
    import os

    os.environ.get("LANGCHAIN_AWS_DEBUG", "FALSE")


setup_logging()
""",
        "langchain_aws/_version.py": '__version__ = "0.0.0"\n',
        # dateutil, which botocore imports, reads the local time at import.
        "langchain_aws/chat_models.py": """
import time

time.localtime()


class ChatBedrock:
    pass
""",
        "langchain_aws/middleware/__init__.py": """
from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware
""",
        "langchain_aws/middleware/prompt_caching.py": _MIDDLEWARE.format(
            package="langchain_aws",
            model="ChatBedrock",
            middleware="BedrockPromptCachingMiddleware",
        ),
    },
    "langchain_fireworks": {
        "langchain_fireworks/__init__.py": """
from langchain_fireworks.chat_models import ChatFireworks
""",
        # sniffio, which the Fireworks SDK imports, subclasses this at import.
        "langchain_fireworks/chat_models.py": """
import threading


class ThreadLocal(threading.local):
    pass


class ChatFireworks:
    pass
""",
        "langchain_fireworks/middleware/__init__.py": """
from langchain_fireworks.middleware.prompt_caching import (
    FireworksPromptCachingMiddleware,
)
""",
        "langchain_fireworks/middleware/prompt_caching.py": _MIDDLEWARE.format(
            package="langchain_fireworks",
            model="ChatFireworks",
            middleware="FireworksPromptCachingMiddleware",
        ),
    },
    "langchain_openai": {
        "langchain_openai/__init__.py": """
from langchain_openai.chat_models import ChatOpenAI
""",
        "langchain_openai/_compat.py": _HTTPX2,
        "langchain_openai/chat_models/__init__.py": """
from langchain_openai.chat_models.base import ChatOpenAI
""",
        "langchain_openai/chat_models/base.py": """
import openai

from langchain_openai import _compat


class BaseChatOpenAI:
    pass


class ChatOpenAI(BaseChatOpenAI):
    pass
""",
        "openai/__init__.py": _HTTPX2,
    },
}


class Capital(BaseModel):
    city: str
    country: str


@workflow.defn
class PlainAgentWorkflow:
    """The README's hello world agent.

    deepagents loads the AWS and Fireworks prompt caching middleware into
    every agent it builds, when their packages are installed.
    """

    @workflow.run
    async def run(self, question: str) -> str:
        agent = create_temporal_deep_agent(model="anthropic:claude-sonnet-5-5")
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        return str(result["messages"][-1].content)


@workflow.defn
class StructuredAgentWorkflow:
    """Provider-native structured output, for which LangChain loads langchain_openai."""

    @workflow.run
    async def run(self, question: str) -> str:
        agent = create_temporal_deep_agent(
            model="openai:gpt-5", response_format=ProviderStrategy(Capital)
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        capital: Capital = result["structured_response"]
        return f"{capital.city}, {capital.country}"


@dataclass(frozen=True)
class _Case:
    workflow: type
    reply: str
    result: str
    # What deepagents or LangChain loads with importlib.import_module.
    lazy_module: str


_CASES = {
    "langchain_aws": _Case(
        PlainAgentWorkflow,
        "Paris.",
        "Paris. [langchain_aws]",
        "langchain_aws.middleware.prompt_caching",
    ),
    "langchain_fireworks": _Case(
        PlainAgentWorkflow,
        "Paris.",
        "Paris. [langchain_fireworks]",
        "langchain_fireworks.middleware.prompt_caching",
    ),
    "langchain_openai": _Case(
        StructuredAgentWorkflow,
        '{"city": "Paris", "country": "France"}',
        "Paris, France",
        "langchain_openai.chat_models.base",
    ),
}


def install(package: str, directory: Path) -> None:
    """Write the stand-in for ``package`` and put it on this process's path."""
    for name, source in _STAND_INS[package].items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    sys.path.insert(0, str(directory))


async def execute(
    package: str, target_host: str, namespace: str, history_path: Path
) -> None:
    """Run the agent on a new Worker and save its history for cold replay."""
    case = _CASES[package]
    plugin = DeepAgentsPlugin(model_provider=mock_model_provider([case.reply]))
    client = await Client.connect(target_host, namespace=namespace, plugins=[plugin])
    assert package not in sys.modules
    task_queue = f"cold-provider-{uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[case.workflow],
        workflow_failure_exception_types=[Exception],
    ):
        # Read before the agent runs, which loads the module too.
        imported = case.lazy_module in sys.modules
        handle = await client.start_workflow(
            case.workflow.__name__,
            "What is the capital of France?",
            id=f"cold-provider-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
            result_type=str,
        )
        assert await handle.result() == case.result
        assert imported, f"{case.lazy_module} was not imported with the Worker"
        history_path.write_text((await handle.fetch_history()).to_json())


async def replay(package: str, history_path: Path) -> None:
    """Replay the saved history in a process that has not run the agent."""
    case = _CASES[package]
    await Replayer(
        workflows=[case.workflow], plugins=[DeepAgentsPlugin()]
    ).replay_workflow(
        WorkflowHistory.from_json("cold-provider", history_path.read_text())
    )
