import sys
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp.mcp_client import MCPClient

from temporalio import workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.strands_agents import StrandsPlugin, TemporalAgent, TemporalMCPClient
from temporalio.worker import Replayer, Worker
from tests.common import get_activities
from tests.mock_model import MockModel

# Same bytes as ``image_mcp_server.screenshot`` returns; not valid UTF-8.
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


@workflow.defn
class MCPImageWorkflow:
    def __init__(self) -> None:
        screenshots = TemporalMCPClient(
            server="image",
            cache_tools=True,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        self.agent = TemporalAgent(
            model="mock",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[screenshots],
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        await self.agent.invoke_async(prompt)
        # The tool result in the agent's history is what the next model call receives.
        return " ".join(
            f"{content['image']['format']} {type(content['image']['source'].get('bytes')).__name__}"
            for message in self.agent.messages
            for block in message["content"]
            if "toolResult" in block
            for content in block["toolResult"]["content"]
            if "image" in content
        )


async def test_mcp_image_tool_result(client: Client, tmp_path: Path):
    """An MCP tool that returns an image runs once and hands the model its bytes."""
    task_queue = f"test_mcp_image-{uuid4()}"
    calls_file = tmp_path / "calls.txt"
    model_saw: list[object] = []

    class RecordingModel(MockModel):
        async def stream(self, messages, *args, **kwargs):  # type: ignore[no-untyped-def, override]
            for message in messages:
                for block in message["content"]:
                    if "toolResult" in block:
                        model_saw.extend(block["toolResult"]["content"])
            async for event in super().stream(messages, *args, **kwargs):
                yield event

    script: list[str | dict[str, object]] = [
        {"name": "screenshot", "input": {}},
        "Done!",
    ]
    plugin = StrandsPlugin(
        models={"mock": lambda: RecordingModel(script)},
        mcp_clients={
            "image": lambda: MCPClient(
                lambda: stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=[str(Path(__file__).parent / "image_mcp_server.py")],
                        env={"IMAGE_MCP_CALLS_FILE": str(calls_file)},
                    )
                )
            ),
        },
    )

    # The plugin goes on the Client, where the data converter is configured, so the
    # call-tool activity result is encoded with the Pydantic JSON converter.
    config = client.config()
    config["plugins"] = [*config["plugins"], plugin]
    client = Client(**config)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[MCPImageWorkflow],
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            MCPImageWorkflow.run,
            "take a screenshot",
            id=f"test_mcp_image_{uuid4()}",
            task_queue=task_queue,
        )
        result = await handle.result()

    # The tool ran exactly once, and both the workflow and the model saw the PNG.
    assert calls_file.read_text().splitlines() == ["call"]
    assert result == "png bytes"
    assert model_saw == [{"image": {"format": "png", "source": {"bytes": PNG}}}]

    history = await handle.fetch_history()
    assert get_activities(history) == [
        "image-list-tools",
        "invoke_model",
        "image-call-tool",
        "invoke_model",
    ]

    await Replayer(
        workflows=[MCPImageWorkflow],
        plugins=[plugin],
    ).replay_workflow(history)
