"""Nested tool values survive converters that include or omit unset fields."""

from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from pydantic import Field, field_serializer

from temporalio.contrib.pydantic import PydanticPayloadConverter, ToJsonOptions
from temporalio.converter import DataConverter
from temporalio.deepagents import _serde
from temporalio.deepagents._activity import DeepAgentActivities, ToolActivityInput
from temporalio.deepagents._tools import register_tool
from temporalio.testing import ActivityEnvironment


class OmitUnsetPayloadConverter(PydanticPayloadConverter):
    def __init__(self) -> None:
        super().__init__(ToJsonOptions(exclude_unset=True))


class DefaultArtifactMessage(ToolMessage):
    artifact: Any = Field(default_factory=lambda: Document(page_content="source"))
    sources: Any = Field(
        default_factory=lambda: [{"document": Document(page_content="nested")}]
    )
    hidden: Any = Field(
        default_factory=lambda: {"document": Document(page_content="hidden")},
        exclude=True,
    )


class SerializedArtifactMessage(DefaultArtifactMessage):
    @field_serializer("artifact", "sources")
    def omit_artifacts(self, _value: Any) -> None:
        return None


@tool
async def read_default_artifacts(message: DefaultArtifactMessage) -> str:
    """Read default-filled retrieval artifacts from an injected message."""
    return (
        f"{message.artifact.page_content}:{message.sources[0]['document'].page_content}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("include_defaults", [False, True])
async def test_wrapped_tool_rebuilds_serialized_default_values(
    include_defaults: bool,
) -> None:
    message = DefaultArtifactMessage(content="match", tool_call_id="call-1")
    message.artifact.page_content = "updated"
    message.sources[0]["document"].page_content = "updated nested"
    assert "artifact" not in message.model_fields_set
    assert "sources" not in message.model_fields_set

    register_tool(read_default_artifacts)
    args = {"message": message}
    input = ToolActivityInput(
        tool_name=read_default_artifacts.name,
        tool_call_id="call-2",
        args=args,
        arg_messages=_serde.find_tool_arg_messages(
            args, _serde.tool_arg_types(read_default_artifacts)
        ),
    )
    converter = (
        DataConverter(payload_converter_class=PydanticPayloadConverter)
        if include_defaults
        else DataConverter(payload_converter_class=OmitUnsetPayloadConverter)
    )
    [sent] = await converter.decode(
        await converter.encode([input]), [ToolActivityInput]
    )
    assert "hidden" not in sent.args["message"]
    output = await ActivityEnvironment().run(DeepAgentActivities().invoke_tool, sent)
    assert _serde.load_object(output.message).content == (
        "updated:updated nested" if include_defaults else "source:nested"
    )


@pytest.mark.asyncio
async def test_default_value_revival_preserves_field_serializer_nulls() -> None:
    message = SerializedArtifactMessage(content="match", tool_call_id="call-1")
    args = {"message": message}
    types = {f"{type(message).__module__}:{type(message).__qualname__}": type(message)}
    converter = DataConverter(payload_converter_class=PydanticPayloadConverter)
    [sent] = await converter.decode(await converter.encode([args]))
    rebuilt = _serde.load_tool_arg_messages(
        sent, _serde.find_tool_arg_messages(args, types), types
    )
    assert isinstance(rebuilt["message"], SerializedArtifactMessage)
    assert rebuilt["message"].artifact is None
    assert rebuilt["message"].sources is None
