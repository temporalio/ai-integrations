"""The Worker model is asked for its own name, as outside Temporal.

ADK puts the agent's model name in the request, which through this plugin is
the ``TemporalModel``'s name. A registry name with a class prefix, such as
``echo:echo-1``, builds a model named ``echo-1``; outside Temporal that model is
asked for ``echo-1``. A name a ``before_model_callback`` set still passes
through unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk.models import BaseLlm, LLMRegistry
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part

from temporalio.google_adk._model import (
    TemporalLlmRequest,
    _new_llm,  # pyright: ignore[reportPrivateUsage]
    _request_for_llm,  # pyright: ignore[reportPrivateUsage]
    invoke_model,
)
from temporalio.testing import ActivityEnvironment


class EchoLlm(BaseLlm):
    """Answers with the model name the request carries."""

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["echo-.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        text = f"{self.model} asked for {llm_request.model}"
        yield LlmResponse(content=Content(role="model", parts=[Part(text=text)]))


def _request(model: str) -> LlmRequest:
    return LlmRequest(
        model=model, contents=[Content(role="user", parts=[Part(text="hi")])]
    )


async def _ask(request_model: str, temporal_model_name: str | None) -> str | None:
    LLMRegistry.register(EchoLlm)
    request = TemporalLlmRequest.wrap(_request(request_model), "unused")
    request.temporal_model_name = temporal_model_name
    responses = await ActivityEnvironment().run(invoke_model, request)
    content = responses[0].content
    return content.parts[0].text if content and content.parts else None


@pytest.mark.asyncio
async def test_prefixed_registry_name_reaches_model_without_prefix() -> None:
    assert await _ask("echo:echo-1", "echo:echo-1") == "echo-1 asked for echo-1"


@pytest.mark.asyncio
async def test_callback_model_name_passes_through() -> None:
    assert (
        await _ask("echo-callback", "echo:echo-1") == "echo-1 asked for echo-callback"
    )


@pytest.mark.asyncio
async def test_request_scheduled_without_model_name_is_renamed() -> None:
    assert await _ask("echo:echo-1", None) == "echo-1 asked for echo-1"


def test_request_for_llm_leaves_matching_name_alone() -> None:
    LLMRegistry.register(EchoLlm)
    request = _request("echo-1")
    llm = _new_llm(request, "echo-1")
    assert _request_for_llm(request, llm, "echo-1") is request
