"""Agent workflow probes imported only by fresh interpreter processes."""

from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

# Match conftest's preload of ADK's optional model SDK to avoid an unrelated
# cold-import deadlock on Python 3.10. Do not import any MCP or auth modules.
import openai  # noqa: F401  # pyright: ignore[reportUnusedImport]
from google.adk.agents import Agent
from google.adk.models import BaseLlm, LLMRegistry
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from temporalio import workflow
from temporalio.client import Client, WorkflowHistory
from temporalio.google_adk import GoogleAdkPlugin, TemporalModel
from temporalio.worker import Replayer, Worker


class LocalModel(BaseLlm):
    @classmethod
    def supported_models(cls) -> list[str]:
        return ["cold_process_model"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="Hello")])
        )


@workflow.defn
class ColdAgentWorkflow:
    @workflow.run
    async def run(self) -> str:
        runner = InMemoryRunner(
            agent=Agent(name="cold_agent", model=TemporalModel("cold_process_model")),
            app_name="cold_app",
        )
        session = await runner.session_service.create_session(
            app_name="cold_app", user_id="user"
        )
        result = ""
        async with aclosing(
            runner.run_async(
                user_id="user",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="Say hello")]
                ),
            )
        ) as events:
            async for event in events:
                if event.content and event.content.parts:
                    result += "".join(part.text or "" for part in event.content.parts)
        return result


async def execute(target_host: str, namespace: str, history_path: Path) -> None:
    """Execute an agent with a local model and save its history for cold replay."""
    LLMRegistry.register(LocalModel)
    client = await Client.connect(
        target_host, namespace=namespace, plugins=[GoogleAdkPlugin()]
    )
    task_queue = f"cold-adk-{uuid4()}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ColdAgentWorkflow],
        workflow_failure_exception_types=[Exception],
    ):
        handle = await client.start_workflow(
            ColdAgentWorkflow.run,
            id=f"cold-adk-{uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=30),
        )
        assert await handle.result() == "Hello"
        history_path.write_text((await handle.fetch_history()).to_json())


async def replay(history_path: Path) -> None:
    """Replay the recorded execution without running the model activity."""
    await Replayer(
        workflows=[ColdAgentWorkflow], plugins=[GoogleAdkPlugin()]
    ).replay_workflow(WorkflowHistory.from_json("cold-adk", history_path.read_text()))
