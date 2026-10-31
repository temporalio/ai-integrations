"""Tests for ToolContextSnapshot injection into activity-backed tools.

Covers https://github.com/temporalio/sdk-python/issues/1470: activities
wrapped with activity_as_tool can read the serializable subset of the ADK
ToolContext (session state and function-call id) without the live,
non-serializable ToolContext ever crossing the activity boundary, and
without the parameter leaking into the LLM-facing tool schema.
"""

import uuid
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import (
    Annotated,
    Any,
    ClassVar,
    Optional,  # pyright: ignore[reportDeprecated]
)

import pytest
from google.adk import Agent
from google.adk.features import FeatureName, override_feature_enabled
from google.adk.models import BaseLlm, LLMRegistry
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from google.genai.types import Content, FunctionCall, Part

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.google_adk import GoogleAdkPlugin, TemporalModel
from temporalio.google_adk.workflow import (
    ToolContextSnapshot,
    activity_as_tool,
)
from temporalio.worker import Worker
from tests.helpers.postponed_annotations import PostponedSnapshotActivity

TASK_QUEUE = "adk-tool-context-task-queue"

SESSION_STATE: dict[str, Any] = {
    "db_url": "postgres://config",
    "retries": 3,
    "regions": {"primary": "us-east1", "replicas": ["eu-west1"]},
}


@activity.defn
async def lookup_weather(
    city: str, tool_context: ToolContextSnapshot, units: str = "celsius"
) -> str:
    """Activity that reads tool configuration from session state.

    Mirrors the shape reported in issue #1470, with tool_context deliberately
    in the middle of the parameter list to prove positional slotting. The
    returned string also records whether the code ran inside a real activity,
    so tests can tell the activity boundary was actually crossed (or not).
    """
    db_url = tool_context.state.get("db_url", "<missing>")
    retries = tool_context.state.get("retries", -1)
    region = tool_context.state.get("regions", {}).get("primary", "<missing>")
    has_function_call_id = "yes" if tool_context.function_call_id else "no"
    in_act = "yes" if activity.in_activity() else "no"
    return f"{city}|{units}|{db_url}|{retries}|{region}|fc={has_function_call_id}|act={in_act}"


@activity.defn
class LookupWeather:
    """Callable-class form of lookup_weather, with an Annotated snapshot."""

    async def __call__(
        self,
        city: str,
        tool_context: Annotated[ToolContextSnapshot, "snapshot"],
        units: str = "celsius",
    ) -> str:
        """Activity that reads tool configuration from session state."""
        return await lookup_weather(city, tool_context, units)


def weather_agent(model_name: str) -> Agent:
    return Agent(
        name="state_agent",
        model=TemporalModel(model_name),
        tools=[
            activity_as_tool(
                lookup_weather, start_to_close_timeout=timedelta(seconds=30)
            ),
            activity_as_tool(
                LookupWeather, start_to_close_timeout=timedelta(seconds=30)
            ),
        ],
    )


class StateToolModel(BaseLlm):
    """Scripted model: call lookup_weather once, then echo its response."""

    tool_name: ClassVar[str] = "lookup_weather"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        tool_response = None
        for content in llm_request.contents:
            for part in content.parts or []:
                if (
                    part.function_response is not None
                    and part.function_response.name == self.tool_name
                ):
                    tool_response = part.function_response
        if tool_response is None:
            yield LlmResponse(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                name=self.tool_name, args={"city": "NYC"}
                            )
                        )
                    ],
                )
            )
        else:
            yield LlmResponse(
                content=Content(
                    role="model",
                    parts=[Part(text=f"done:{tool_response.response}")],
                )
            )

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["state_tool_model"]


class ClassStateToolModel(StateToolModel):
    """Scripted model that calls the LookupWeather class tool instead."""

    tool_name: ClassVar[str] = "LookupWeather"

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["class_state_tool_model"]


async def run_state_agent(model_name: str) -> str:
    """Runs the agent against a session seeded with state; returns final text."""
    runner = InMemoryRunner(agent=weather_agent(model_name), app_name="test_app")
    session = await runner.session_service.create_session(
        app_name="test_app",
        user_id="test",
        state=SESSION_STATE,
    )
    final_text = ""
    async with Aclosing(
        runner.run_async(
            user_id="test",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part(text="weather in NYC?")]
            ),
        )
    ) as agen:
        async for event in agen:
            if event.content and event.content.parts and event.content.parts[0].text:
                final_text = event.content.parts[0].text
    return final_text


@workflow.defn
class StateToolWorkflow:
    @workflow.run
    async def run(self, model_name: str) -> str:
        return await run_state_agent(model_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", [StateToolModel, ClassStateToolModel], ids=["function", "class"]
)
async def test_activity_as_tool_receives_tool_context_snapshot(
    client: Client, model: type[StateToolModel]
):
    new_config = client.config()
    new_config["plugins"] = [GoogleAdkPlugin()]
    client = Client(**new_config)

    async with Worker(
        client,
        task_queue=f"{TASK_QUEUE}-{model.tool_name}",
        activities=[lookup_weather, LookupWeather()],
        workflows=[StateToolWorkflow],
        max_cached_workflows=0,
    ):
        LLMRegistry.register(model)
        result = await client.execute_workflow(
            StateToolWorkflow.run,
            model.supported_models()[0],
            id=f"tool-context-{uuid.uuid4()}",
            task_queue=f"{TASK_QUEUE}-{model.tool_name}",
            execution_timeout=timedelta(seconds=30),
        )
    # The activity saw mixed-type session state (string, int, nested dict),
    # the default parameter value, and a populated function-call id — none of
    # which came from the LLM — and act=yes proves the snapshot crossed a
    # real activity boundary rather than running inline in the workflow.
    assert "NYC|celsius|postgres://config|3|us-east1|fc=yes|act=yes" in result


@pytest.mark.asyncio
async def test_activity_as_tool_snapshot_outside_workflow():
    """Local ADK runs (no Temporal) receive the same snapshot."""
    LLMRegistry.register(StateToolModel)
    result = await run_state_agent("state_tool_model")
    assert "NYC|celsius|postgres://config|3|us-east1|fc=yes|act=no" in result


def _declared_properties(tool: FunctionTool) -> dict[str, Any]:
    """Property names in the LLM-facing declaration, across schema styles."""
    declaration = tool._get_declaration()
    assert declaration is not None
    if declaration.parameters_json_schema is not None:
        return declaration.parameters_json_schema.get("properties", {})
    assert declaration.parameters is not None
    return declaration.parameters.properties or {}


def test_tool_schema_excludes_tool_context():
    """The tool_context parameter never appears in the LLM-facing schema."""
    tool = FunctionTool(
        func=activity_as_tool(
            lookup_weather, start_to_close_timeout=timedelta(seconds=30)
        )
    )
    properties = _declared_properties(tool)
    assert "city" in properties
    assert "units" in properties
    assert "tool_context" not in properties


def test_tool_schema_excludes_tool_context_legacy_declaration():
    """Exclusion also holds on the legacy (non-JSON-schema) declaration path."""
    override_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False)
    try:
        tool = FunctionTool(
            func=activity_as_tool(
                lookup_weather, start_to_close_timeout=timedelta(seconds=30)
            )
        )
        declaration = tool._get_declaration()
        assert declaration is not None
        assert declaration.parameters is not None
        properties = declaration.parameters.properties or {}
        assert "city" in properties
        assert "units" in properties
        assert "tool_context" not in properties
    finally:
        # The flag is default-on across the supported google-adk range.
        override_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True)


def test_tool_schema_context_only_parameter():
    """A tool whose only parameter is tool_context exposes no LLM arguments."""

    @activity.defn
    async def ctx_only_tool(tool_context: ToolContextSnapshot) -> str:
        return str(tool_context.state)

    tool = FunctionTool(
        func=activity_as_tool(
            ctx_only_tool, start_to_close_timeout=timedelta(seconds=30)
        )
    )
    declaration = tool._get_declaration()
    if declaration is not None:
        json_properties = (declaration.parameters_json_schema or {}).get(
            "properties", {}
        )
        legacy_properties = (
            (declaration.parameters.properties or {}) if declaration.parameters else {}
        )
        assert not json_properties
        assert not legacy_properties


def test_activity_as_tool_accepts_optional_snapshot_annotation():
    """ToolContextSnapshot | None is accepted and still excluded from the schema."""

    @activity.defn
    async def optional_tool(
        query: str,
        tool_context: ToolContextSnapshot | None = None,  # pyright: ignore[reportUnusedParameter]
    ) -> str:
        return query

    tool = FunctionTool(
        func=activity_as_tool(
            optional_tool, start_to_close_timeout=timedelta(seconds=30)
        )
    )
    properties = _declared_properties(tool)
    assert set(properties) == {"query"}

    # Also for a callable-class instance whose annotations are strings: its
    # types are read from its class's __call__.
    tool = FunctionTool(
        func=activity_as_tool(
            PostponedSnapshotActivity(), start_to_close_timeout=timedelta(seconds=30)
        )
    )
    assert set(_declared_properties(tool)) == {"query"}


def test_activity_as_tool_rejects_adk_tool_context_annotation():
    """Annotating with the live ADK ToolContext gives an actionable error."""

    @activity.defn
    async def bad_tool(query: str, tool_context: ToolContext) -> str:  # pyright: ignore[reportUnusedParameter]
        return query

    with pytest.raises(ValueError, match="ToolContextSnapshot"):
        activity_as_tool(bad_tool, start_to_close_timeout=timedelta(seconds=30))


def test_activity_as_tool_rejects_optional_adk_tool_context_annotation():
    """Optional[ToolContext] is rejected with the ADK-specific message."""

    @activity.defn
    async def optional_bad_tool(
        query: str,
        tool_context: Optional[ToolContext] = None,  # pyright: ignore[reportUnusedParameter, reportDeprecated]
    ) -> str:
        return query

    with pytest.raises(ValueError, match="not serializable"):
        activity_as_tool(
            optional_bad_tool, start_to_close_timeout=timedelta(seconds=30)
        )


def test_activity_as_tool_rejects_adk_context_under_any_name():
    """ADK injects into any param annotated with a context type, so all are rejected."""

    @activity.defn
    async def sneaky_tool(query: str, ctx: ToolContext) -> str:  # pyright: ignore[reportUnusedParameter]
        return query

    with pytest.raises(ValueError, match="not serializable"):
        activity_as_tool(sneaky_tool, start_to_close_timeout=timedelta(seconds=30))


def test_tool_schema_excludes_annotated_tool_context_of_class():
    """A class's Annotated snapshot is accepted and excluded like a function's."""
    tool = FunctionTool(
        func=activity_as_tool(
            LookupWeather, start_to_close_timeout=timedelta(seconds=30)
        )
    )
    assert set(_declared_properties(tool)) == {"city", "units"}


@activity.defn
async def annotated_context_tool(
    query: str,
    ctx: Annotated[ToolContext, "metadata"],  # pyright: ignore[reportUnusedParameter]
) -> str:
    return query


@activity.defn
class AnnotatedContextActivity:
    async def __call__(
        self,
        query: str,
        ctx: Annotated[ToolContext, "metadata"],  # pyright: ignore[reportUnusedParameter]
    ) -> str:
        return query


@activity.defn
class OptionalAnnotatedContextActivity:
    async def __call__(
        self,
        query: str,
        tool_context: Annotated[ToolContext, "metadata"] | None = None,  # pyright: ignore[reportUnusedParameter]
    ) -> str:
        return query


# Optional[Alias] nests the alias's own Optional inside Annotated.
LiveContextAlias = Annotated[Optional[ToolContext], "metadata"]  # pyright: ignore[reportDeprecated]
SnapshotAlias = Annotated[Optional[ToolContextSnapshot], "metadata"]  # pyright: ignore[reportDeprecated]


@activity.defn
async def nested_alias_context_tool(
    query: str,
    ctx: Optional[LiveContextAlias] = None,  # pyright: ignore[reportUnusedParameter, reportDeprecated]
) -> str:
    return query


@activity.defn
class NestedAliasContextActivity:
    async def __call__(
        self,
        query: str,
        ctx: Optional[LiveContextAlias] = None,  # pyright: ignore[reportUnusedParameter, reportDeprecated]
    ) -> str:
        return query


@activity.defn
async def nested_alias_snapshot_tool(
    query: str,
    tool_context: Optional[SnapshotAlias] = None,  # pyright: ignore[reportUnusedParameter, reportDeprecated]
) -> str:
    return query


@activity.defn
class NestedAliasSnapshotActivity:
    async def __call__(
        self,
        query: str,
        tool_context: Optional[SnapshotAlias] = None,  # pyright: ignore[reportUnusedParameter, reportDeprecated]
    ) -> str:
        return query


@pytest.mark.parametrize(
    "activity_def",
    [nested_alias_snapshot_tool, NestedAliasSnapshotActivity],
    ids=["function", "class"],
)
def test_activity_as_tool_accepts_nested_snapshot_alias(activity_def: Any):
    """A snapshot inside nested Annotated and Optional is accepted and excluded."""
    tool = FunctionTool(
        func=activity_as_tool(
            activity_def, start_to_close_timeout=timedelta(seconds=30)
        )
    )
    assert set(_declared_properties(tool)) == {"query"}


@pytest.mark.parametrize(
    "activity_def",
    [
        annotated_context_tool,
        AnnotatedContextActivity,
        AnnotatedContextActivity(),
        OptionalAnnotatedContextActivity,
        nested_alias_context_tool,
        NestedAliasContextActivity,
    ],
)
def test_activity_as_tool_rejects_annotated_adk_context(activity_def: Any):
    """Annotated metadata doesn't hide a live ADK context from the check."""
    with pytest.raises(ValueError, match="not serializable"):
        activity_as_tool(activity_def, start_to_close_timeout=timedelta(seconds=30))


def test_activity_as_tool_rejects_snapshot_under_other_name():
    """ToolContextSnapshot on a differently-named param would leak into the schema."""

    @activity.defn
    async def misnamed_tool(query: str, snap: ToolContextSnapshot) -> str:  # pyright: ignore[reportUnusedParameter]
        return query

    with pytest.raises(ValueError, match="named 'tool_context'"):
        activity_as_tool(misnamed_tool, start_to_close_timeout=timedelta(seconds=30))


def test_activity_as_tool_rejects_unannotated_tool_context():
    """The reserved name without an annotation gives an actionable error."""

    @activity.defn
    async def untyped_tool(query: str, tool_context) -> str:  # type: ignore[no-untyped-def] # pyright: ignore[reportUnusedParameter, reportMissingParameterType]
        return query

    with pytest.raises(ValueError, match="unannotated 'tool_context'"):
        activity_as_tool(untyped_tool, start_to_close_timeout=timedelta(seconds=30))


def test_activity_as_tool_rejects_other_tool_context_annotation():
    """The reserved name with an unrelated annotation gives an actionable error."""

    @activity.defn
    async def confused_tool(query: str, tool_context: dict[str, Any]) -> str:  # pyright: ignore[reportUnusedParameter]
        return query

    with pytest.raises(ValueError, match="reserved by ADK"):
        activity_as_tool(confused_tool, start_to_close_timeout=timedelta(seconds=30))


def test_activity_as_tool_without_tool_context_unchanged():
    """Activities without a tool_context parameter keep their exact schema."""

    @activity.defn
    async def plain_tool(query: str) -> str:
        return query

    tool = FunctionTool(
        func=activity_as_tool(plain_tool, start_to_close_timeout=timedelta(seconds=30))
    )
    assert set(_declared_properties(tool)) == {"query"}
