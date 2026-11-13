"""Preserve ADK mutations and defaults across Temporal payloads."""

import json
from dataclasses import dataclass
from typing import Any

import pytest
from google.adk.events import Event, EventActions
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from temporalio.api.common.v1 import WorkflowExecution
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.converter import DataConverter
from temporalio.google_adk import GoogleAdkPlugin
from temporalio.google_adk._mcp import TemporalToolContext


@pytest.fixture
def converter() -> DataConverter:
    return GoogleAdkPlugin()._configure_data_converter(None)


async def test_appended_session_events_survive_roundtrip(converter: DataConverter):
    service = InMemorySessionService()
    session = await service.create_session(app_name="app", user_id="user")
    event = Event(
        author="agent",
        content=types.Content(role="model", parts=[types.Part(text="answer")]),
    )
    await service.append_event(session=session, event=event)
    assert session.events == [event]

    payloads = await converter.encode([session])
    [restored] = await converter.decode(payloads, [Session])

    assert restored.events == [event]
    assert restored.last_update_time == session.last_update_time


async def test_event_factory_timestamp_survives_roundtrip(converter: DataConverter):
    event = Event(author="agent")
    assert "timestamp" not in event.model_fields_set

    payloads = await converter.encode([event])
    [restored] = await converter.decode(payloads, [Event])

    assert json.loads(payloads[0].data)["timestamp"] == event.timestamp
    assert restored.timestamp == event.timestamp
    assert restored.id == event.id


async def test_model_content_default_role_survives_llm_response(
    converter: DataConverter,
):
    response = LlmResponse(
        content=types.ModelContent(parts=[types.Part(text="answer")])
    )
    assert response.content is not None
    assert response.content.role == "model"
    assert "role" not in response.content.model_fields_set

    payloads = await converter.encode([response])
    [restored] = await converter.decode(payloads, [LlmResponse])

    assert restored.content is not None
    assert restored.content.role == "model"


async def test_in_place_tool_confirmation_survives_roundtrip(converter: DataConverter):
    context = TemporalToolContext(None, "call-id", EventActions())
    context.request_confirmation(hint="Approve lookup", payload={"query": "value"})
    assert "requested_tool_confirmations" not in context.event_actions.model_fields_set

    payloads = await converter.encode([context])
    [restored] = await converter.decode(payloads, [TemporalToolContext])

    confirmation = restored.event_actions.requested_tool_confirmations["call-id"]
    assert confirmation.hint == "Approve lookup"
    assert confirmation.payload == {"query": "value"}
    assert confirmation.confirmed is False


async def test_in_place_event_state_delta_survives_roundtrip(converter: DataConverter):
    event = Event(author="agent")
    event.actions.state_delta["answer"] = 42
    assert "actions" not in event.model_fields_set
    assert "state_delta" not in event.actions.model_fields_set

    payloads = await converter.encode([event])
    [restored] = await converter.decode(payloads, [Event])

    assert restored.actions.state_delta == {"answer": 42}


class _OptionalFields(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    unset_none: str | None = Field(default=None, alias="optionalUnsetNone")
    explicit_none: str | None = Field(default=None, alias="explicitNone")
    default_value: str = Field(default="kept", alias="defaultValue")
    values: list[str] = Field(default_factory=list)


@dataclass
class _NestedValues:
    items: list[_OptionalFields]
    mapping: dict[str, _OptionalFields]
    plain: dict[str, Any]


async def test_nested_none_fields_are_omitted_without_losing_defaults_or_dict_entries(
    converter: DataConverter,
):
    model = _OptionalFields(explicitNone=None)
    model.values.append("in-place")
    value = _NestedValues(
        items=[model], mapping={"model": model}, plain={"keep_none": None}
    )
    original_dump = model.model_dump()
    original_fields = model.model_fields_set.copy()

    payloads = await converter.encode([value])
    [restored] = await converter.decode(payloads, [_NestedValues])
    wire = json.loads(payloads[0].data)

    for nested in (wire["items"][0], wire["mapping"]["model"]):
        assert "optionalUnsetNone" not in nested
        assert "explicitNone" not in nested
        assert nested["defaultValue"] == "kept"
        assert nested["values"] == ["in-place"]
    assert wire["plain"] == {"keep_none": None}
    assert restored.items[0] == restored.mapping["model"] == model
    assert restored.plain == {"keep_none": None}
    assert model.model_dump() == original_dump
    assert model.model_fields_set == original_fields
    assert value.items[0] is value.mapping["model"] is model


@pytest.mark.parametrize("value", [b"binary-data", WorkflowExecution(workflow_id="wf")])
async def test_sdk_binary_and_protobuf_payloads_are_unchanged(
    converter: DataConverter, value: bytes | WorkflowExecution
):
    payloads = await converter.encode([value])
    standard_payloads = await pydantic_data_converter.encode([value])

    assert payloads == standard_payloads
    assert await converter.decode(payloads, [type(value)]) == [value]
