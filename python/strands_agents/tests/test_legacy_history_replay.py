"""Histories recorded before bytes support still replay.

The fixtures in ``histories/`` were recorded on ``5e91c26`` (upstream main
before the bytes change) against a local dev server, by running the workflows
below unchanged. Each shape is recorded twice, because the wire format of
``bytes`` depends on the data converter in use:

* plugin attached to the ``Client`` (pydantic converter): ``bytes`` are written
  as a UTF-8 string, so ``redactedContent=b"ASCII-redacted"`` is the JSON string
  ``"ASCII-redacted"``;
* plugin attached only to the ``Worker`` (default converter): ``bytes`` are
  written as a list of ints.

``invoke_model`` was typed ``list[StreamEvent]``, so the type hint turned
either form back into ``bytes`` for the workflow. ``{server}-call-tool`` was read
without a type hint, so the workflow saw the string (or the list of ints) as is.

The workflows decode these results with the code under test: a result shape it
cannot read makes the replay fail.
"""

from datetime import timedelta
from pathlib import Path

import pytest

from temporalio import workflow
from temporalio.client import WorkflowHistory
from temporalio.strands_agents import StrandsPlugin, TemporalAgent, TemporalMCPClient
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner
from tests.mock_model import MockModel

HISTORIES = Path(__file__).parent / "histories"

# Reply that contains a ``redactedContent`` block, followed by text.
LEGACY_REDACTED = b"ASCII-redacted"
LEGACY_IMAGE = b"ASCII-image"


class RedactedContentModel(MockModel):
    """Answers with a ``redactedContent`` reasoning block, then text."""

    async def stream(self, *_args, **_kwargs):  # type: ignore[no-untyped-def, override]
        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"redactedContent": LEGACY_REDACTED}}
            }
        }
        yield {"contentBlockStop": {}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": "ok"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


@workflow.defn
class LegacyRedactedContentWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="legacy", start_to_close_timeout=timedelta(seconds=15)
        )

    @workflow.run
    async def run(self) -> str:
        await self.agent.invoke_async("hello")
        return "done"


@workflow.defn
class LegacyMCPImageWorkflow:
    def __init__(self) -> None:
        self.agent = TemporalAgent(
            model="legacy",
            start_to_close_timeout=timedelta(seconds=30),
            tools=[
                TemporalMCPClient(
                    server="image",
                    cache_tools=True,
                    start_to_close_timeout=timedelta(seconds=30),
                )
            ],
        )

    @workflow.run
    async def run(self) -> str:
        await self.agent.invoke_async("take a screenshot")
        return "done"


def _legacy_history(name: str) -> WorkflowHistory:
    return WorkflowHistory.from_json(
        "legacy-replay", (HISTORIES / f"{name}.json").read_text()
    )


CONVERTERS = ["pydantic", "default_converter"]


def _replayer(workflow_class: type, converter: str) -> Replayer:
    """Replay with the converter the history was recorded under.

    A worker that has the plugin only on ``Worker(plugins=...)`` keeps the
    client's default converter; the plugin on the ``Client`` swaps in pydantic.
    """
    if converter == "pydantic":
        plugin = StrandsPlugin(models={"legacy": lambda: MockModel([])})
        return Replayer(workflows=[workflow_class], plugins=[plugin])
    return Replayer(
        workflows=[workflow_class], workflow_runner=UnsandboxedWorkflowRunner()
    )


@pytest.mark.parametrize("converter", CONVERTERS)
async def test_replay_recorded_redacted_content(converter: str):
    """A recorded ``redactedContent`` result decodes to the bytes the model sent."""
    await _replayer(LegacyRedactedContentWorkflow, converter).replay_workflow(
        _legacy_history(f"legacy_redacted_content_{converter}")
    )


@pytest.mark.parametrize("converter", CONVERTERS)
async def test_replay_recorded_mcp_image(converter: str):
    """A recorded MCP image result, whose bytes were ASCII, still replays."""
    await _replayer(LegacyMCPImageWorkflow, converter).replay_workflow(
        _legacy_history(f"legacy_mcp_ascii_image_{converter}")
    )
