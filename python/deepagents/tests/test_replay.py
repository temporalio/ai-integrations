"""Recorded histories replay cleanly through the plugin.

The Deep Agents control loop runs in the workflow, so replay determinism is the
core safety property. This records the history of a real run (a plain fake agent
driven by ``run_deep_agent``, no LangChain needed) and feeds it back through a
``Replayer`` configured with the plugin. A nondeterministic seam would raise on
replay; a clean pass proves the in-workflow dispatch is deterministic.
"""

from __future__ import annotations

import importlib
import sys
import uuid
from collections.abc import Sequence
from typing import Any

import pytest

from temporalio.testing import WorkflowEnvironment

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)
from temporalio import workflow
from temporalio.api.common.v1 import Payload
from temporalio.api.enums.v1 import EventType
from temporalio.converter import DataConverter, PayloadCodec
from temporalio.deepagents import DeepAgentsPlugin, run_deep_agent
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from tests.helpers import deepagents as helpers


class FakeAgent:
    async def ainvoke(self, input: Any) -> dict:
        messages = list(input.get("messages", [])) if isinstance(input, dict) else []
        return {"messages": [*messages, "answered"], "todos": []}


@workflow.defn
class ReplayWorkflow:
    @workflow.run
    async def run(self, input: dict) -> dict:
        return await run_deep_agent(FakeAgent(), input)


@pytest.mark.asyncio
async def test_replay_with_plugin(env: WorkflowEnvironment) -> None:
    plugin = DeepAgentsPlugin()
    async with Worker(
        env.client,
        task_queue="da-replay",
        workflows=[ReplayWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ReplayWorkflow.run,
            {"messages": ["question"]},
            id=f"da-replay-{uuid.uuid4()}",
            task_queue="da-replay",
        )
        await handle.result()
        history = await handle.fetch_history()

    # A fresh replayer (new worker identity) must replay the recorded history
    # without a nondeterminism error.
    replayer = Replayer(
        workflows=[ReplayWorkflow],
        plugins=[DeepAgentsPlugin()],
    )
    await replayer.replay_workflow(history)


# ---------------------------------------------------------------------------
# Replay does not rebuild model inputs
# ---------------------------------------------------------------------------
#
# A replayed model call is answered from history, so its input is dropped.
# Building it anyway (dumping the whole transcript and passing it through the
# payload codec) made a cold replay cost grow with calls x transcript length.

_LONG_PROMPT = "Summarize this. " + "x" * 20_000


class _SizeCodec(PayloadCodec):
    """Pass payloads through, recording the size of each one encoded."""

    def __init__(self) -> None:
        self.encoded: list[int] = []

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        self.encoded.extend(p.ByteSize() for p in payloads)
        return list(payloads)

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return list(payloads)


@workflow.defn
class LongPromptWorkflow:
    @workflow.run
    async def run(self) -> str:
        create_deep_agent = importlib.import_module("deepagents").create_deep_agent
        agent = create_deep_agent(model="fake:model")
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": _LONG_PROMPT}]}
        )
        return str(result["messages"][-1].content)


def _ls_twice_then_answer() -> Any:
    from langchain_core.messages import AIMessage

    from temporalio.deepagents.testing import mock_model_provider

    ls = {"name": "ls", "args": {"path": "/"}}
    return mock_model_provider(
        [
            AIMessage(content="", tool_calls=[{**ls, "id": "ls-1"}]),
            AIMessage(content="", tool_calls=[{**ls, "id": "ls-2"}]),
            "Done.",
        ]
    )


@pytest.mark.asyncio
async def test_replay_does_not_rebuild_model_inputs(env: WorkflowEnvironment) -> None:
    pytest.importorskip("deepagents")
    async with Worker(
        env.client,
        task_queue="da-replay-inputs",
        workflows=[LongPromptWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_ls_twice_then_answer())],
    ):
        handle = await env.client.start_workflow(
            LongPromptWorkflow.run,
            id=f"da-replay-inputs-{uuid.uuid4()}",
            task_queue="da-replay-inputs",
        )
        assert await handle.result() == "Done."
        history = await handle.fetch_history()

    codec = _SizeCodec()
    replayer = Replayer(
        workflows=[LongPromptWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_ls_twice_then_answer())],
        data_converter=DataConverter(payload_codec=codec),
    )
    await replayer.replay_workflow(history)
    # Three model calls each carried the 20 KB prompt when they ran; replaying
    # them encodes no payload that size.
    assert codec.encoded, "the codec saw no payloads"
    assert max(codec.encoded) < len(_LONG_PROMPT)


async def _model_input_sizes(handle: Any) -> list[int]:
    """Bytes of each scheduled model Activity's input, from history."""
    sizes = []
    async for event in handle.fetch_history_events():
        if event.event_type != EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            continue
        attrs = event.activity_task_scheduled_event_attributes
        if attrs.activity_type.name == "deepagents.invoke_model":
            sizes.append(sum(p.ByteSize() for p in attrs.input.payloads))
    return sizes


def _fail_second_model_call_once() -> Any:
    """Middleware that fails the workflow task making the second model call, once per run.

    Built lazily, like ``create_deep_agent`` below, so collecting this module
    does not import LangChain.
    """
    from langchain.agents.middleware import AgentMiddleware

    class _FailSecondModelCallOnce(AgentMiddleware):
        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            run = workflow.info().run_id
            turns = sum(1 for m in request.messages if m.type == "ai")
            if turns == 1 and run not in helpers.FAILED_ONCE:
                helpers.FAILED_ONCE.add(run)
                raise RuntimeError("fail this workflow task once")
            return await handler(request)

    return _FailSecondModelCallOnce()


@workflow.defn
class FrontierWorkflow:
    @workflow.run
    async def run(self, fail_once: bool) -> str:
        create_deep_agent = importlib.import_module("deepagents").create_deep_agent
        middleware = [_fail_second_model_call_once()] if fail_once else []
        agent = create_deep_agent(model="fake:model", middleware=middleware)
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": _LONG_PROMPT}]}
        )
        return str(result["messages"][-1].content)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_once", [False, True])
async def test_calls_after_replay_get_full_inputs(
    env: WorkflowEnvironment, fail_once: bool
) -> None:
    """Calls made right after replaying history are new and carry the transcript.

    With no workflow cache every workflow task replays the history before it.
    With ``fail_once`` the task that would make the second model call fails
    after the first call's result arrived, so the retry replays up to that
    result and makes the call there. Every recorded model input must hold the
    full prompt.
    """
    pytest.importorskip("deepagents")
    async with Worker(
        env.client,
        task_queue="da-replay-frontier",
        workflows=[FrontierWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_ls_twice_then_answer())],
        max_cached_workflows=0,
    ):
        handle = await env.client.start_workflow(
            FrontierWorkflow.run,
            fail_once,
            id=f"da-replay-frontier-{uuid.uuid4()}",
            task_queue="da-replay-frontier",
        )
        assert await handle.result() == "Done."
    failed = [
        e
        async for e in handle.fetch_history_events()
        if e.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED
    ]
    assert len(failed) == (1 if fail_once else 0)
    sizes = await _model_input_sizes(handle)
    assert len(sizes) == 3
    assert min(sizes) > len(_LONG_PROMPT)


@workflow.defn
class LongPromptRunWorkflow:
    @workflow.run
    async def run(self) -> str:
        create_deep_agent = importlib.import_module("deepagents").create_deep_agent
        agent = create_deep_agent(model="fake:model")
        result = await run_deep_agent(
            agent, {"messages": [{"role": "user", "content": _LONG_PROMPT}]}
        )
        return str(result["messages"][-1].content)


@pytest.mark.asyncio
async def test_legacy_history_still_builds_inputs(
    env: WorkflowEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A history recorded under the legacy result cache replays.

    ``run_deep_agent`` is what turns the legacy cache on.

    The legacy cache keys on the model input, so replaying it with empty
    inputs would answer the second call from the first one's cache entry and
    schedule one Activity fewer than history holds.
    """
    pytest.importorskip("deepagents")
    from temporalio.deepagents import workflow as plugin_workflow

    monkeypatch.setattr(plugin_workflow, "_legacy_result_cache", lambda: True)
    async with Worker(
        env.client,
        task_queue="da-replay-legacy-inputs",
        workflows=[LongPromptRunWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_ls_twice_then_answer())],
    ):
        handle = await env.client.start_workflow(
            LongPromptRunWorkflow.run,
            id=f"da-replay-legacy-inputs-{uuid.uuid4()}",
            task_queue="da-replay-legacy-inputs",
        )
        assert await handle.result() == "Done."
        history = await handle.fetch_history()
    monkeypatch.undo()
    assert len(await _model_input_sizes(handle)) == 3

    codec = _SizeCodec()
    replayer = Replayer(
        workflows=[LongPromptRunWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=_ls_twice_then_answer())],
        data_converter=DataConverter(payload_codec=codec),
    )
    await replayer.replay_workflow(history)
    # Legacy replay still builds (and encodes) the full model inputs.
    assert max(codec.encoded) > len(_LONG_PROMPT)
