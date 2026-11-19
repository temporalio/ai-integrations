"""The model seam: every ``TemporalModel`` generation is one activity.

These exercise the seam directly through ``TemporalModel`` (no full agent
needed), so they depend only on LangChain, not on deepagents.
"""

from __future__ import annotations

import asyncio
import time
import uuid
import warnings
from collections.abc import Sequence
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from temporalio.common import RetryPolicy
from temporalio.testing import WorkflowEnvironment

pytest.importorskip("langchain_core")

from temporalio import workflow  # noqa: E402
from temporalio.worker import Worker  # noqa: E402
from tests.helpers.deepagents import (  # noqa: E402
    count_scheduled_activities,
    scheduled_attributes,
)

with workflow.unsafe.imports_passed_through():
    from langchain_core.messages import HumanMessage

    from temporalio.deepagents import (
        DeepAgentsPlugin,
        TemporalModel,
        create_temporal_deep_agent,
        get_model_profile,
    )
    from temporalio.deepagents._activity import (
        DeepAgentActivities,
        ModelActivityInput,
    )
    from temporalio.deepagents.testing import FakeModel, mock_model_provider

INVOKE_MODEL = "deepagents.invoke_model"


class _ToolBindingModel(FakeModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self.bind(tools=tools, **kwargs)


@pytest.mark.parametrize(
    "bind_kwargs",
    [
        {"response_format": {"type": "json_schema", "json_schema": {"name": "Answer"}}},
        {"tool_choice": "any"},
    ],
)
def test_model_binding_preserves_kwargs_with_tools(bind_kwargs: dict[str, Any]) -> None:
    tool_schemas = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    activities = DeepAgentActivities(
        model_provider=lambda _: _ToolBindingModel(responses=["answer"])
    )

    bound = activities._build_bound_model(
        ModelActivityInput(
            model_name="fake:model",
            messages=[],
            tool_schemas=tool_schemas,
            bind_kwargs=bind_kwargs,
        )
    )

    assert bound.kwargs == {"tools": tool_schemas, **bind_kwargs}


@pytest.mark.parametrize(
    "tool_kwargs",
    [
        {"tool_choice": "any"},
        {"parallel_tool_calls": False},
        {"tool_choice": "any", "parallel_tool_calls": False},
    ],
)
def test_tool_options_go_to_bind_tools(tool_kwargs: dict[str, Any]) -> None:
    # Providers convert these options only inside bind_tools, so the activity
    # must pass them there; other options still go through .bind().
    tool_schemas = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    other_kwargs = {"stop": ["Observation:"]}
    model = Mock(wraps=_ToolBindingModel(responses=["answer"]))
    activities = DeepAgentActivities(model_provider=lambda _: model)

    bound = activities._build_bound_model(
        ModelActivityInput(
            model_name="fake:model",
            messages=[],
            tool_schemas=tool_schemas,
            bind_kwargs={**tool_kwargs, **other_kwargs},
        )
    )

    model.bind_tools.assert_called_once_with(tool_schemas, **tool_kwargs)
    assert bound.kwargs == {"tools": tool_schemas, **tool_kwargs, **other_kwargs}


@pytest.mark.parametrize(
    ("tool_kwargs", "expected"),
    [
        ({"tool_choice": "any"}, {"tool_choice": {"type": "any"}}),
        (
            {"parallel_tool_calls": False},
            {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}},
        ),
    ],
)
def test_tool_options_use_provider_format(
    tool_kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:
    # create_agent binds tool_choice="any" for structured output; the provider's
    # bind_tools must convert it, as it does without Temporal.
    langchain_anthropic = pytest.importorskip("langchain_anthropic")
    chat_model = langchain_anthropic.ChatAnthropic(model="test-model", api_key="test")
    tool = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object"}},
    }
    messages = [HumanMessage(content="hi")]
    binding = TemporalModel(model="provider:model").bind_tools([tool], **tool_kwargs)
    activity_input = binding.bound._build_input(messages, None, **binding.kwargs)
    activities = DeepAgentActivities(model_provider=lambda _: chat_model)

    model = activities._build_bound_model(activity_input)
    payload = model.bound._get_request_payload(messages, **model.kwargs)

    keys = ("tool_choice", "parallel_tool_calls")
    assert {k: payload[k] for k in keys if k in payload} == expected


@workflow.defn
class ModelWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        model = TemporalModel(model="fake:model")
        message = await model.ainvoke([HumanMessage(content=prompt)])
        return str(message.content)


@workflow.defn
class ExplicitTimeoutWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        model = TemporalModel(
            model="fake:model",
            activity_options={"start_to_close_timeout": timedelta(seconds=20)},
        )
        message = await model.ainvoke([HumanMessage(content=prompt)])
        return str(message.content)


@pytest.mark.asyncio
async def test_model_call_is_activity(env: WorkflowEnvironment) -> None:
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["The capital of France is Paris."]),
        model_activity_options={"start_to_close_timeout": timedelta(seconds=30)},
    )
    async with Worker(
        env.client,
        task_queue="da-model",
        workflows=[ModelWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ModelWorkflow.run,
            "What is the capital of France?",
            id=f"da-model-{uuid.uuid4()}",
            task_queue="da-model",
        )
        out = await handle.result()
    assert "Paris" in out
    counts = await count_scheduled_activities(handle)
    assert counts[INVOKE_MODEL] == 1, counts


@pytest.mark.asyncio
async def test_model_call_warns_nothing_about_allowed_objects(
    env: WorkflowEnvironment,
) -> None:
    """Neither side of the seam relies on ``load``'s default ``allowed_objects``.

    One model call loads its input messages in the activity and its reply in
    the workflow; ``langchain_core.load.load`` warns on each such call that
    relies on the default (scheduled to narrow, which would change what the
    plugin's own payloads revive). Assert neither warning fires.
    """
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["The capital of France is Paris."]),
        model_activity_options={"start_to_close_timeout": timedelta(seconds=30)},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        async with Worker(
            env.client,
            task_queue="da-model-warn",
            workflows=[ModelWorkflow],
            plugins=[plugin],
        ):
            handle = await env.client.start_workflow(
                ModelWorkflow.run,
                "What is the capital of France?",
                id=f"da-model-warn-{uuid.uuid4()}",
                task_queue="da-model-warn",
            )
            out = await handle.result()
    assert "Paris" in out
    offenders = [
        w
        for w in caught
        if issubclass(w.category, PendingDeprecationWarning)
        and "allowed_objects" in str(w.message)
    ]
    assert not offenders, [f"{w.filename}: {w.message}" for w in offenders]


@pytest.mark.asyncio
async def test_temporal_model_explicit(env: WorkflowEnvironment) -> None:
    # The explicit escape hatch routes through the same activity, with a
    # per-model timeout override rather than the plugin default.
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["Bonjour."]),
    )
    async with Worker(
        env.client,
        task_queue="da-model-explicit",
        workflows=[ExplicitTimeoutWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ExplicitTimeoutWorkflow.run,
            "hi",
            id=f"da-model-x-{uuid.uuid4()}",
            task_queue="da-model-explicit",
        )
        out = await handle.result()
    assert out == "Bonjour."
    counts = await count_scheduled_activities(handle)
    assert counts[INVOKE_MODEL] == 1, counts


@workflow.defn
class HeartbeatOverrideWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        model = TemporalModel(
            model="fake:model",
            activity_options={"heartbeat_timeout": timedelta(seconds=7)},
        )
        message = await model.ainvoke([HumanMessage(content=prompt)])
        return str(message.content)


@pytest.mark.asyncio
async def test_model_activity_heartbeats_by_default(
    env: WorkflowEnvironment,
) -> None:
    """The model activity is scheduled with a default heartbeat timeout.

    Without one the plugin's auto-heartbeater never beats, so the server can
    only detect a crashed worker at start-to-close (five minutes), and a
    cancelled workflow's model call never learns it was cancelled — a Python
    activity observes cancellation solely through a heartbeat response.
    """
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["Still there."]),
    )
    async with Worker(
        env.client,
        task_queue="da-model-heartbeat",
        workflows=[ModelWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ModelWorkflow.run,
            "Are you there?",
            id=f"da-model-hb-{uuid.uuid4()}",
            task_queue="da-model-heartbeat",
        )
        await handle.result()
    # The literal 10 s (not the source constant) so this test speaks against
    # any code that resolves the schedule, main's included.
    attrs = await scheduled_attributes(handle, INVOKE_MODEL)
    assert attrs.heartbeat_timeout.ToTimedelta() == timedelta(seconds=10)
    assert attrs.start_to_close_timeout.ToTimedelta() == timedelta(minutes=5)


@pytest.mark.asyncio
async def test_model_activity_heartbeat_caps_to_start_to_close(
    env: WorkflowEnvironment,
) -> None:
    """The default never outlives the schedule's own close timeout: a plugin
    ``start_to_close_timeout`` shorter than 10 s caps the heartbeat timeout
    with it, in plugin code rather than by platform normalization."""
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["Brief."]),
        model_activity_options={"start_to_close_timeout": timedelta(seconds=4)},
    )
    async with Worker(
        env.client,
        task_queue="da-model-hb-cap",
        workflows=[ModelWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ModelWorkflow.run,
            "hi",
            id=f"da-model-hb-cap-{uuid.uuid4()}",
            task_queue="da-model-hb-cap",
        )
        await handle.result()
    attrs = await scheduled_attributes(handle, INVOKE_MODEL)
    assert attrs.heartbeat_timeout.ToTimedelta() == timedelta(seconds=4)
    assert attrs.start_to_close_timeout.ToTimedelta() == timedelta(seconds=4)


@pytest.mark.asyncio
async def test_model_activity_heartbeat_override_wins(env: WorkflowEnvironment) -> None:
    """A per-model ``heartbeat_timeout`` replaces the default."""
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["Punctual."]),
    )
    async with Worker(
        env.client,
        task_queue="da-model-heartbeat-override",
        workflows=[HeartbeatOverrideWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            HeartbeatOverrideWorkflow.run,
            "hi",
            id=f"da-model-hb-ov-{uuid.uuid4()}",
            task_queue="da-model-heartbeat-override",
        )
        await handle.result()
    attrs = await scheduled_attributes(handle, INVOKE_MODEL)
    assert attrs.heartbeat_timeout.ToTimedelta() == timedelta(seconds=7)


@pytest.mark.asyncio
async def test_model_activity_heartbeat_opt_out(env: WorkflowEnvironment) -> None:
    """A plugin-wide ``heartbeat_timeout=None`` opts out of the default."""
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["Unhurried."]),
        model_activity_options={"heartbeat_timeout": None},
    )
    async with Worker(
        env.client,
        task_queue="da-model-heartbeat-optout",
        workflows=[ModelWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ModelWorkflow.run,
            "hi",
            id=f"da-model-hb-out-{uuid.uuid4()}",
            task_queue="da-model-heartbeat-optout",
        )
        await handle.result()
    attrs = await scheduled_attributes(handle, INVOKE_MODEL)
    assert attrs.heartbeat_timeout.ToTimedelta() == timedelta(0)


_SLOW_BUILD_OPTIONS = {
    "start_to_close_timeout": timedelta(seconds=20),
    "heartbeat_timeout": timedelta(seconds=1),
    "retry_policy": RetryPolicy(maximum_attempts=1),
}


@workflow.defn
class SlowProviderWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        profile = await get_model_profile(
            "fake:slow", activity_options=_SLOW_BUILD_OPTIONS
        )
        model = TemporalModel(
            model="fake:slow", profile=profile, activity_options=_SLOW_BUILD_OPTIONS
        )
        message = await model.ainvoke([HumanMessage(content=prompt)])
        return str(message.content)


@pytest.mark.asyncio
async def test_slow_model_provider_keeps_heartbeating(
    env: WorkflowEnvironment,
) -> None:
    """A ``model_provider`` that blocks longer than the heartbeat timeout
    still succeeds on its first attempt, for the model call and for
    ``get_model_profile`` when the plugin opts in with
    ``model_provider_in_thread``: the activities build the model on a worker
    thread, so the auto-heartbeater keeps beating. Built on the event loop
    (the default), the same provider starves the heartbeater and the activity
    heartbeat-times out."""
    provide = mock_model_provider(["Loaded."])

    def slow_provider(name: str) -> Any:
        time.sleep(2.5)
        return provide(name)

    async with Worker(
        env.client,
        task_queue="da-model-slow-provider",
        workflows=[SlowProviderWorkflow],
        plugins=[
            DeepAgentsPlugin(
                model_provider=slow_provider, model_provider_in_thread=True
            )
        ],
    ):
        out = await env.client.execute_workflow(
            SlowProviderWorkflow.run,
            "hi",
            id=f"da-model-slow-provider-{uuid.uuid4()}",
            task_queue="da-model-slow-provider",
        )
    assert out == "Loaded."


@pytest.mark.asyncio
async def test_loop_bound_model_provider_builds_on_the_loop(
    env: WorkflowEnvironment,
) -> None:
    """A synchronous ``model_provider`` that needs the activity's running event
    loop (an ``aiohttp`` session, say) works by default, for the model call and
    for ``get_model_profile``: construction stays on the loop unless the plugin
    opts in to a thread, where ``get_running_loop`` raises."""
    provide = mock_model_provider(["Bound."])

    def loop_bound_provider(name: str) -> Any:
        asyncio.get_running_loop()  # RuntimeError off the loop
        return provide(name)

    async with Worker(
        env.client,
        task_queue="da-model-loop-bound",
        workflows=[SlowProviderWorkflow],
        plugins=[DeepAgentsPlugin(model_provider=loop_bound_provider)],
    ):
        out = await env.client.execute_workflow(
            SlowProviderWorkflow.run,
            "hi",
            id=f"da-model-loop-bound-{uuid.uuid4()}",
            task_queue="da-model-loop-bound",
        )
    assert out == "Bound."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "zero_field, other_field",
    [
        ("start_to_close_timeout", "schedule_to_close_timeout"),
        ("schedule_to_close_timeout", "start_to_close_timeout"),
    ],
)
async def test_model_activity_heartbeat_ignores_zero_close_timeout(
    env: WorkflowEnvironment, zero_field: str, other_field: str
) -> None:
    """A zero close timeout is unset to the server, so it must not zero the
    default heartbeat: the cap is the other, active close timeout."""
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["Zero."]),
        model_activity_options={
            zero_field: timedelta(0),
            other_field: timedelta(seconds=4),
        },
    )
    task_queue = f"da-model-hb-zero-{zero_field}"
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[ModelWorkflow],
        plugins=[plugin],
    ):
        handle = await env.client.start_workflow(
            ModelWorkflow.run,
            "hi",
            id=f"da-model-hb-zero-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        await handle.result()
    attrs = await scheduled_attributes(handle, INVOKE_MODEL)
    assert attrs.heartbeat_timeout.ToTimedelta() == timedelta(seconds=4)


class _ResampleAgent:
    """ainvoke-shaped driver: the SAME prompt sent twice through a
    TemporalModel — deliberate resampling — under run_deep_agent so the
    continue-as-new result cache is active."""

    def __init__(self) -> None:
        self.model = TemporalModel(model="fake:model")

    async def ainvoke(self, _input: Any) -> dict:
        first = await self.model.ainvoke([HumanMessage(content="same prompt")])
        second = await self.model.ainvoke([HumanMessage(content="same prompt")])
        return {
            "messages": [f"{first.content}|{second.content}"],
            "todos": [],
        }


@workflow.defn
class ResampleWorkflow:
    @workflow.run
    async def run(self, input: dict, state_snapshot: dict | None = None) -> str:
        from temporalio.deepagents import run_deep_agent

        result = await run_deep_agent(
            _ResampleAgent(), input, state_snapshot=state_snapshot
        )
        return str(result["messages"][-1])


@pytest.mark.asyncio
async def test_repeated_identical_model_call_resamples(
    env: WorkflowEnvironment,
) -> None:
    """Two identical live model calls each run their own Activity and can
    return different responses (deliberate resampling) — under the active
    CAN cache, the old input-only key served the FIRST response twice."""
    plugin = DeepAgentsPlugin(
        model_provider=mock_model_provider(["first answer", "second answer"]),
    )
    async with Worker(
        env.client,
        task_queue="da-model-resample",
        workflows=[ResampleWorkflow],
        plugins=[plugin],
        max_cached_workflows=0,
    ):
        handle = await env.client.start_workflow(
            ResampleWorkflow.run,
            {"messages": []},
            id=f"da-model-resample-{uuid.uuid4()}",
            task_queue="da-model-resample",
        )
        out = await handle.result()

    assert out == "first answer|second answer", out
    counts = await count_scheduled_activities(handle)
    assert counts["deepagents.invoke_model"] == 2, counts


@workflow.defn
class ModelProfileWorkflow:
    @workflow.run
    async def run(self, model_name: str) -> Any:
        profile = await get_model_profile(model_name)
        return TemporalModel(model=model_name, profile=profile).profile


@pytest.mark.asyncio
async def test_get_model_profile_reads_the_workers_model(
    env: WorkflowEnvironment,
) -> None:
    # Deep Agents sizes summarization from model.profile (#374).
    profiles = {"fake:profiled": {"max_input_tokens": 200_000}}
    plugin = DeepAgentsPlugin(
        model_provider=lambda name: FakeModel(["unused"], profile=profiles.get(name)),
    )
    async with Worker(
        env.client,
        task_queue="da-model-profile",
        workflows=[ModelProfileWorkflow],
        plugins=[plugin],
    ):
        results = []
        for name in ("fake:profiled", "fake:model"):
            handle = await env.client.start_workflow(
                ModelProfileWorkflow.run,
                name,
                id=f"da-model-profile-{uuid.uuid4()}",
                task_queue="da-model-profile",
            )
            results.append(await handle.result())
            counts = await count_scheduled_activities(handle)
            assert counts["deepagents.get_model_profile"] == 1, counts
    assert results == [{"max_input_tokens": 200_000}, None]


def test_activity_options_keep_the_models_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(
        sys.modules,
        "deepagents",
        SimpleNamespace(create_deep_agent=lambda **kwargs: kwargs),
    )
    options = {"start_to_close_timeout": timedelta(seconds=30)}
    built = create_temporal_deep_agent(
        model=TemporalModel(model="fake:model", profile={"max_input_tokens": 200_000}),
        activity_options=options,
    )
    assert built["model"].activity_options == options
    assert built["model"].profile == {"max_input_tokens": 200_000}
