"""A harbor agent that answers correctly and records a large transcript."""

from __future__ import annotations

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

TRANSCRIPT_AGENT = "tests.harbor_fixtures.agents:TranscriptAgent"
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
