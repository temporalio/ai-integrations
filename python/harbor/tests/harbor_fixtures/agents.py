"""Harbor agents, and a verifier, that the suite runs trials with."""

from __future__ import annotations

import os
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

TRANSCRIPT_AGENT = "tests.harbor_fixtures.agents:TranscriptAgent"
SKILLS_AGENT = "tests.harbor_fixtures.agents:SkillsAgent"
FAILING_AGENT = "tests.harbor_fixtures.agents:FailingAgent"
NAN_VERIFIER = "tests.harbor_fixtures.agents:NanVerifier"
TOKENS = 4_000


class TranscriptAgent(BaseAgent):
    """Writes the expected answer, then reports tokens, rollouts and metadata."""

    @staticmethod
    def name() -> str:
        return "transcript"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await environment.exec(
            "mkdir -p /logs/artifacts && echo hello > /logs/artifacts/out.txt"
        )
        context.n_input_tokens = TOKENS
        context.n_output_tokens = TOKENS // 10
        context.cost_usd = 0.25
        context.rollout_details = [
            {
                "prompt_token_ids": [list(range(TOKENS))],
                "completion_token_ids": [list(range(TOKENS))],
                "logprobs": [[-0.5] * TOKENS],
            }
        ]
        context.metadata = {"transcript": "x" * 50_000}


class SkillsAgent(TranscriptAgent):
    """A ``TranscriptAgent`` that declares skills, which harbor requires to give it any."""

    capabilities = TranscriptAgent.capabilities.model_copy(update={"skills": True})


class FailingAgent(BaseAgent):
    """Ends every run with an exception of the type ``error`` names.

    Harbor classifies an exception by its type's name alone. The message
    includes the value of the worker's environment variable ``leak`` names.
    """

    def __init__(
        self, *args: Any, error: str, leak: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._error = type(error, (Exception,), {})
        self._leak = f" ({os.environ[leak]})" if leak else ""

    @staticmethod
    def name() -> str:
        return "failing"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        raise self._error(
            f"the agent's run ended in {self._error.__name__}{self._leak}"
        )


class NanVerifier(BaseVerifier):
    """Grades every trial with a reward that is not a number."""

    async def verify(self) -> VerifierResult:
        return VerifierResult(rewards={"reward": float("nan")})
