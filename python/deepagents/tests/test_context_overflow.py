"""A provider's context overflow reaches Deep Agents' summarization.

Deep Agents' summarization middleware recovers from a model call the provider
rejects as too long: it summarizes the older messages and retries once. It
recognizes the overflow by the exception the model call raises, which under the
plugin is the model Activity's ``ActivityError``. The model Activity classifies
the provider's exception on the worker, and ``TemporalModel`` raises
``ContextOverflowError`` from it in the Workflow, so the agent recovers as it
does without Temporal.
"""

from __future__ import annotations

import importlib
import sys
import uuid
from datetime import timedelta
from typing import Any

import pytest

from temporalio.testing import ActivityEnvironment, WorkflowEnvironment

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)

pytest.importorskip("deepagents")
pytest.importorskip("langchain_core")

from temporalio import workflow  # noqa: E402
from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner  # noqa: E402
from tests.helpers.deepagents import count_scheduled_activities  # noqa: E402

create_deep_agent = pytest.importorskip("deepagents").create_deep_agent

with workflow.unsafe.imports_passed_through():
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.exceptions import ContextOverflowError
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from temporalio.deepagents import DeepAgentsPlugin, _activity
    from temporalio.deepagents._activity import DeepAgentActivities, ModelActivityInput
    from temporalio.deepagents._serde import dump_messages

INVOKE_MODEL = "deepagents.invoke_model"
OVERFLOW_TYPE = "deepagents.ContextOverflowError"
LIMIT = 8
SEEN: list[int] = []


class _ProviderOverflow(ContextOverflowError):
    """Shaped like langchain-anthropic's and langchain-openai's overflow errors."""

    status_code = 400


class _StreamedOverflow(ContextOverflowError):
    """Shaped like langchain-openai's overflow reported in a stream, with no status."""


class _OpenAICompatibleError(Exception):
    """The 400 an OpenAI-compatible server such as vLLM returns for a long prompt."""

    status_code = 400


OVERFLOWS = {
    "langchain": lambda: _ProviderOverflow(
        "Error code: 400 - prompt is too long: 213000 tokens > 200000 maximum"
    ),
    "openai-compatible": lambda: _OpenAICompatibleError(
        "Error code: 400 - This model's maximum context length is 4096 tokens. "
        "However, your request has 5120 input tokens."
    ),
    "streamed": lambda: _StreamedOverflow("Input exceeds the context window"),
}


class SmallContextModel(BaseChatModel):
    """Rejects requests with more than ``limit`` non-system messages."""

    provider: str = "langchain"
    limit: int = LIMIT

    @property
    def _llm_type(self) -> str:
        return "small-context"

    def bind_tools(self, tools: Any, **kwargs: Any) -> SmallContextModel:
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        count = sum(1 for m in messages if m.type != "system")
        SEEN.append(count)
        if count > self.limit:
            raise OVERFLOWS[self.provider]()
        reply = "summary of the earlier turns" if count == 1 else "done"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(reply))])


# Deep Agents resends a rejected request only if the summarized one is smaller,
# so each turn must outweigh its share of the summary message that replaces it.
_DETAILS = "details " * 25


def _history() -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for i in range(5):
        messages += [
            HumanMessage(f"question {i}: {_DETAILS}"),
            AIMessage(f"answer {i}: {_DETAILS}"),
        ]
    return [*messages, HumanMessage("final question")]


@workflow.defn
class OverflowWorkflow:
    @workflow.run
    async def run(self) -> str:
        agent = create_deep_agent(model="fake:small")
        result = await agent.ainvoke({"messages": _history()})
        return result["messages"][-1].content


# -- worker side (no server) --------------------------------------------------


@pytest.fixture(params=["installed", "0.7.0-0.7.13"])
def deepagents_release(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run with the installed Deep Agents, then as on 0.7.0 to 0.7.13.

    The plugin allows ``deepagents>=0.7``; before 0.7.14 the summarization
    middleware has no ``_is_context_overflow``.
    """
    if request.param != "installed":
        summarization = importlib.import_module("deepagents.middleware.summarization")
        monkeypatch.delattr(summarization, "_is_context_overflow", raising=False)


class _StatusError(Exception):
    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize(
    ("status", "message", "overflow"),
    [
        (413, "Request too large: input tokens exceed the configured limit", True),
        (422, "Error code: 422 - context_length_exceeded", True),
        (400, "Error code: 400 - invalid tool schema", False),
        (500, "maximum context length is 4096 tokens", False),
        (None, "prompt is too long", False),
    ],
)
def test_overflow_check(status: int | None, message: str, overflow: bool) -> None:
    assert _activity._is_context_overflow(_StatusError(status, message)) is overflow


@pytest.mark.usefixtures("deepagents_release")
@pytest.mark.parametrize("provider", sorted(OVERFLOWS))
async def test_model_activity_marks_provider_overflow(provider: str) -> None:
    activities = DeepAgentActivities(
        model_provider=lambda _: SmallContextModel(provider=provider)
    )
    input = ModelActivityInput(
        model_name="fake:small", messages=dump_messages(_history())
    )
    with pytest.raises(ApplicationError) as raised:
        await ActivityEnvironment().run(activities.invoke_model, input)
    assert raised.value.type == OVERFLOW_TYPE
    assert raised.value.non_retryable


@pytest.mark.usefixtures("deepagents_release")
async def test_other_bad_request_keeps_its_type() -> None:
    class BadRequest(Exception):
        status_code = 400

    class RejectingModel(SmallContextModel):
        async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
            raise BadRequest("Error code: 400 - invalid tool schema")

    activities = DeepAgentActivities(model_provider=lambda _: RejectingModel())
    input = ModelActivityInput(
        model_name="fake:small", messages=dump_messages(_history())
    )
    with pytest.raises(ApplicationError) as raised:
        await ActivityEnvironment().run(activities.invoke_model, input)
    assert raised.value.type == "BadRequest"
    assert raised.value.non_retryable


def test_unrecoverable_overflow_fails_the_workflow_not_its_task() -> None:
    config: Any = DeepAgentsPlugin().configure_worker(
        {"task_queue": "q", "workflow_runner": SandboxedWorkflowRunner()}
    )
    assert ContextOverflowError in config["workflow_failure_exception_types"]


# -- workflows (local dev server) ----------------------------------------------


@pytest.mark.parametrize("provider", sorted(OVERFLOWS))
async def test_agent_summarizes_after_provider_overflow(
    env: WorkflowEnvironment, provider: str
) -> None:
    SEEN.clear()
    native = await create_deep_agent(
        model=SmallContextModel(provider=provider)
    ).ainvoke({"messages": _history()})
    native_seen = list(SEEN)
    assert native["messages"][-1].content == "done"

    SEEN.clear()
    plugin = DeepAgentsPlugin(
        model_provider=lambda _: SmallContextModel(provider=provider),
        model_activity_options={"start_to_close_timeout": timedelta(seconds=30)},
    )
    task_queue = f"da-overflow-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[OverflowWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            OverflowWorkflow.run,
            id=f"da-overflow-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        assert await handle.result() == "done"
    assert SEEN == native_seen == [11, 1, 7]
    assert (await count_scheduled_activities(handle))[INVOKE_MODEL] == 3

    replayer = Replayer(workflows=[OverflowWorkflow], plugins=[plugin])
    await replayer.replay_workflow(await handle.fetch_history())


class AlwaysOverflowModel(SmallContextModel):
    limit: int = 1


async def test_irreducible_overflow_fails_the_workflow(
    env: WorkflowEnvironment,
) -> None:
    plugin = DeepAgentsPlugin(
        model_provider=lambda _: AlwaysOverflowModel(),
        model_activity_options={"start_to_close_timeout": timedelta(seconds=30)},
    )
    task_queue = f"da-overflow-{uuid.uuid4()}"
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[OverflowWorkflow],
        plugins=[plugin],
    ):
        with pytest.raises(WorkflowFailureError) as raised:
            await env.client.execute_workflow(
                OverflowWorkflow.run,
                id=f"da-overflow-{uuid.uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=30),
            )
    cause = raised.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "ContextOverflowError"
