"""Tests for worker environment references in hosted tool secrets."""

import dataclasses
import gc
import json
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Collection
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Generic, Literal, TypeVar, cast

import pydantic
import pydantic.dataclasses
import pytest
from agents import (
    Agent,
    AgentOutputSchemaBase,
    CodeInterpreterTool,
    Handoff,
    HostedMCPTool,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    Runner,
    Tool,
    TResponseInputItem,
    Usage,
    UserError,
)
from agents.items import TResponseStreamEvent
from agents.tool import ApplyPatchTool, ShellTool, ShellToolEnvironment
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
)

from temporalio import workflow
from temporalio.client import Client
from temporalio.openai_agents import (
    ModelActivityParameters,
    OpenAIAgentsPlugin,
    OpenAIPayloadConverter,
    temporal_worker_env_ref,
)
from temporalio.openai_agents._invoke_model_activity import (
    ActivityModelInput,
    ApplyPatchToolInput,
    FunctionToolInput,
    HandoffInput,
    ModelActivity,
    ModelTracingInput,
    ShellToolInput,
    StreamingActivityModelInput,
    _build_tool,
    _build_tools_and_handoffs,
)
from temporalio.openai_agents._temporal_model_stub import _TemporalModelStub
from temporalio.openai_agents._temporal_worker_env_ref import (
    AllowAllWorkerEnvVars,
    _WorkerEnvRefResolver,
)
from temporalio.openai_agents.testing import (
    AgentEnvironment,
    TestModelProvider,
)
from temporalio.testing import ActivityEnvironment
from tests.helpers import new_worker

SENTINEL = "sk-test-sentinel-4f1a9c7e2b"
ENV_NAME = "TEMPORAL_TEST_TOOL_SECRET"
OTHER_SENTINEL = "sk-test-other-8c3d5e0a1f"
OTHER_ENV_NAME = "TEMPORAL_TEST_OTHER_TOOL_SECRET"

_RESOLVER_ALLOWING_TEST_NAMES = _WorkerEnvRefResolver([ENV_NAME, OTHER_ENV_NAME])


def test_lenient_activity_input_reconstructs_tools_and_handoffs():
    activity_input: ActivityModelInput = {
        "model_name": "gpt-5",
        "input": "hi",
        "model_settings": ModelSettings(),
        "tools": [
            FunctionToolInput(
                name="get_weather",
                description="Get the weather",
                params_json_schema={"type": "object"},
            ),
            ApplyPatchToolInput(name="apply_patch"),
            ShellToolInput(name="apply_patch", environment={"type": "local"}),
        ],
        "handoffs": [
            HandoffInput(
                tool_name="transfer_to_agent",
                tool_description="Transfer to the agent",
                input_json_schema={"type": "object"},
                agent_name="helper",
            )
        ],
        "tracing": ModelTracingInput.DISABLED,
    }
    converter = OpenAIPayloadConverter()
    payload = converter.to_payload(activity_input)
    raw = json.loads(payload.data)
    raw["tracing"] = "invalid-enum-value"
    payload.data = json.dumps(raw).encode()

    received = converter.from_payload(payload, ActivityModelInput)

    received_tools = received.get("tools")
    received_handoffs = received.get("handoffs")
    assert received_tools and isinstance(received_tools[0], FunctionToolInput)
    assert isinstance(received_tools[1], ApplyPatchToolInput)
    assert isinstance(received_tools[2], ShellToolInput)
    assert received_handoffs and isinstance(received_handoffs[0], HandoffInput)
    tools, handoffs = _build_tools_and_handoffs(received, _WorkerEnvRefResolver(()))
    assert tools[0].name == "get_weather"
    assert isinstance(tools[1], ApplyPatchTool)
    assert isinstance(tools[2], ShellTool)
    assert handoffs[0].agent_name == "helper"


def test_lenient_activity_input_with_unvalidated_input_item():
    # A function call item with an explicit None the next model input does not allow, as
    # some model implementations write it. The tools must still decode as dataclasses.
    activity_input: ActivityModelInput = {
        "model_name": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
            }
        ],
        "model_settings": ModelSettings(),
        "tools": [
            FunctionToolInput(
                name="get_weather",
                description="Get the weather",
                params_json_schema={"type": "object"},
            )
        ],
        "tracing": ModelTracingInput.DISABLED,
    }
    converter = OpenAIPayloadConverter()
    payload = converter.to_payload(activity_input)
    raw = json.loads(payload.data)
    raw["input"][0]["namespace"] = None
    payload.data = json.dumps(raw).encode()

    received = converter.from_payload(payload, ActivityModelInput)

    assert isinstance(received["input"], list)
    first_item: Any = received["input"][0]
    assert first_item["namespace"] is None
    received_tools = received.get("tools")
    assert received_tools and isinstance(received_tools[0], FunctionToolInput)


class _CallerResult(pydantic.BaseModel):
    message: ResponseOutputMessage
    responses: list[ModelResponse]
    note: str | None = None


def test_lenient_caller_model_keeps_its_type():
    # An output_text part without annotations fails strict validation, as a drifted
    # live response does. A caller's model holding it must still decode as that model.
    message = ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text="done", annotations=[])],
    )
    value = _CallerResult(
        message=message,
        responses=[ModelResponse(output=[message], usage=Usage(), response_id="r1")],
        note="kept",
    )
    converter = OpenAIPayloadConverter()
    payload = converter.to_payload(value)
    raw = json.loads(payload.data)
    del raw["message"]["content"][0]["annotations"]
    del raw["responses"][0]["output"][0]["content"][0]["annotations"]
    payload.data = json.dumps(raw).encode()

    received = converter.from_payload(payload, _CallerResult)

    assert isinstance(received, _CallerResult)
    assert isinstance(received.message, ResponseOutputMessage)
    assert isinstance(received.message.content[0], ResponseOutputText)
    assert received.message.content[0].text == "done"
    assert isinstance(received.responses[0], ModelResponse)
    assert isinstance(received.responses[0].output[0], ResponseOutputMessage)
    assert received.note == "kept"


class _ExtraResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    message: ResponseOutputMessage


class _NamedResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(validate_by_alias=False, validate_by_name=True)
    text: str = pydantic.Field(alias="original")
    original: str
    message: ResponseOutputMessage


_T = TypeVar("_T")


class _WrappedResult(pydantic.BaseModel, Generic[_T]):
    item: _T


class _DescribedResult(pydantic.BaseModel):
    record: Annotated[ModelResponse, pydantic.Field(description="record")]


class _MappedResult(pydantic.BaseModel):
    records: dict[str, ModelResponse]


_RootResult = pydantic.RootModel[dict[str, ResponseOutputMessage]]


class _UnionResult(pydantic.BaseModel):
    call: ResponseOutputItem


class _TimedResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        ser_json_temporal="milliseconds", val_temporal_unit="milliseconds"
    )
    at: datetime
    record: ModelResponse


class _NormalizedResult(pydantic.BaseModel):
    record: ModelResponse

    @pydantic.field_validator("record", mode="before")
    @classmethod
    def _lower_id(cls, value: Any) -> Any:
        # Reads the response as the mapping it was sent as.
        return {**value, "response_id": value["response_id"].lower()}


class _StrictResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        strict=True, ser_json_bytes="base64", val_json_bytes="base64"
    )
    id: uuid.UUID
    blob: bytes
    record: ModelResponse


@dataclasses.dataclass
class _Holder:
    record: Annotated[ModelResponse | None, pydantic.Field(description="response")] = (
        None
    )


class _HeldResult(pydantic.BaseModel):
    holder: _Holder


class _ListRoot(pydantic.RootModel[list[ModelResponse]]):
    pass


class _ListedResult(pydantic.BaseModel):
    records: _ListRoot | None = None


class _Initialized(pydantic.BaseModel):
    count: int

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)


class _InitializedPairResult(pydantic.BaseModel):
    first: _Initialized
    second: _Initialized
    record: ModelResponse


@pydantic.dataclasses.dataclass(config=pydantic.ConfigDict(extra="allow"))
class _ExtraSide:
    name: str

    @pydantic.model_serializer(mode="wrap")
    def _with_extra(self, handler: Any) -> Any:
        # Sends the extra field it was validated with.
        return {**handler(self), "sentinel": cast(Any, self).sentinel}


class _ExtraSideResult(pydantic.BaseModel):
    side: _ExtraSide
    record: ModelResponse


class _TaggedA(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(serialize_by_alias=True)
    kind: Literal["a"] = pydantic.Field(alias="Kind")
    record: ModelResponse


class _TaggedB(pydantic.BaseModel):
    kind: Literal["b"]
    record: ModelResponse


@pydantic.dataclasses.dataclass
class _TaggedC:
    kind: Literal["c"]
    record: ModelResponse


class _TaggedResult(pydantic.BaseModel):
    item: _TaggedA | _TaggedB | _TaggedC


@pydantic.dataclasses.dataclass(config=pydantic.ConfigDict(extra="allow"))
class _Labelled:
    note: str

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        # Sent with the fields, and read back as an extra field of the same name.
        return self.note.upper()


class _LabelledResult(pydantic.BaseModel):
    side: _Labelled
    record: ModelResponse


class _CheckedA(pydantic.BaseModel):
    kind: Literal["a"]
    record: ModelResponse

    @pydantic.model_validator(mode="before")
    @classmethod
    def _check(cls, value: Any) -> Any:
        return value


class _CheckedB(pydantic.BaseModel):
    kind: Literal["b"]
    record: ModelResponse

    @pydantic.model_validator(mode="before")
    @classmethod
    def _check(cls, value: Any) -> Any:
        return value


class _ChoicesA(pydantic.BaseModel):
    kind: Literal["a"] = pydantic.Field(
        validation_alias=pydantic.AliasChoices("kind", "Kind")
    )
    record: ModelResponse


class _PathB(pydantic.BaseModel):
    kind: Literal["b"] = pydantic.Field(validation_alias=pydantic.AliasPath("kind"))
    record: ModelResponse


class _CheckedResult(pydantic.BaseModel):
    item: _CheckedA | _CheckedB


class _AliasFormResult(pydantic.BaseModel):
    item: _ChoicesA | _PathB


class _UpperA(pydantic.BaseModel):
    kind: Literal["a"]
    record: ModelResponse

    @pydantic.field_serializer("kind")
    def _upper(self, kind: str) -> str:
        return kind.upper()

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _lower(cls, value: Any, handler: Any) -> Any:
        # Reads the tag it sent in upper case.
        if isinstance(value, dict):
            value = {**value, "kind": value["kind"].lower()}
        return handler(value)


class _NormalizedTagResult(pydantic.BaseModel):
    item: _UpperA | _TaggedB


def _lower_type(value: Any) -> Any:
    if isinstance(value, dict):
        return {**value, "type": value["type"].lower()}
    return value


def _passthrough(value: Any, handler: Any) -> Any:
    return handler(value)


class _LoweredMessageResult(pydantic.BaseModel):
    item: (
        Annotated[ResponseOutputMessage, pydantic.BeforeValidator(_lower_type)]
        | _TaggedB
    )

    @pydantic.model_serializer(mode="wrap")
    def _upper_type(self, handler: Any) -> Any:
        # Sends the message's type in upper case.
        data = handler(self)
        return {**data, "item": {**data["item"], "type": "MESSAGE"}}


class _WrappedMessageResult(pydantic.BaseModel):
    item: (
        Annotated[ResponseOutputMessage, pydantic.WrapValidator(_passthrough)]
        | _TaggedB
    )


def _retry(value: Any, handler: Any) -> Any:
    try:
        return handler(value)
    except pydantic.ValidationError:
        # Calls its handler again, as a validator that normalizes on error does.
        return handler(value)


class _RetriedMessageResult(pydantic.BaseModel):
    item: Annotated[ResponseOutputMessage, pydantic.WrapValidator(_retry)] | _TaggedB


class _TextResult(pydantic.BaseModel):
    text: str
    record: ModelResponse


def _upper_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {**value, "text": value["text"].upper()}
    return value


def _upper_result(result: _TextResult) -> _TextResult:
    return result.model_copy(update={"text": result.text.upper()})


_UpperedResult: Any = Annotated[_TextResult, pydantic.AfterValidator(_upper_result)]


class _TextA(pydantic.BaseModel):
    kind: Literal["a"]
    text: str
    record: ModelResponse


class _TextB(pydantic.BaseModel):
    kind: Literal["b"]
    text: str
    record: ModelResponse


_UpperedA: Any = Annotated[_TextA, pydantic.BeforeValidator(_upper_text)]
_UpperedB: Any = Annotated[_TextB, pydantic.BeforeValidator(_upper_text)]
_UpperedUnion: Any = _UpperedA | _UpperedB


@dataclasses.dataclass
class _Box(Generic[_T]):
    note: str
    record: _T


def _from_list(value: Any) -> Any:
    return {"record": value[0]} if isinstance(value, list) else value


class _Listed(pydantic.BaseModel):
    record: ModelResponse

    @pydantic.model_serializer
    def _serialize(self) -> Any:
        return [self.record]

    @pydantic.model_validator(mode="before")
    @classmethod
    def _restore(cls, value: Any) -> Any:
        return _from_list(value)


_ListedOrInts: Any = _Listed | list[int]


class _ListedEnvelope(pydantic.BaseModel):
    item: _Listed | list[int]


@pydantic.dataclasses.dataclass
class _ListedBox(Generic[_T]):
    record: _T

    @pydantic.model_serializer
    def _serialize(self) -> Any:
        return [self.record]

    @pydantic.model_validator(mode="before")
    @classmethod
    def _restore(cls, value: Any) -> Any:
        return _from_list(value)


class _Leaf(pydantic.BaseModel):
    count: int


class _ReturnedLeafResult(pydantic.BaseModel):
    leaf: Annotated[_Leaf, pydantic.BeforeValidator(_Leaf.model_validate)]
    record: ModelResponse


@pydantic.dataclasses.dataclass(
    config=pydantic.ConfigDict(revalidate_instances="always")
)
class _Revalidated:
    tag: str

    @pydantic.field_validator("tag")
    @classmethod
    def _mark(cls, value: str) -> str:
        return value + "!"


class _ReturnedDataclassResult(pydantic.BaseModel):
    side: Annotated[
        _Revalidated, pydantic.BeforeValidator(lambda value: _Revalidated(**value))
    ]
    record: ModelResponse


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and len(value) == 1 else value


_UnwrappedMessageOrInts: Any = (
    Annotated[ResponseOutputMessage, pydantic.BeforeValidator(_first)] | list[int]
)


def _extra_side() -> _ExtraSide:
    side = _ExtraSide(name="n")
    cast(Any, side).sentinel = "keep-me"
    return side


_AT = datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
_BLOB = b"\x00\xffbytes"


def _message() -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text="done", annotations=[])],
    )


def _call() -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call", call_id="c1", name="f", arguments="{}"
    )


def _response() -> ModelResponse:
    return ModelResponse(output=[_message()], usage=Usage(), response_id="r1")


def _is_message(value: Any) -> bool:
    return isinstance(value, ResponseOutputMessage) and isinstance(
        value.content[0], ResponseOutputText
    )


def _is_response(value: Any) -> bool:
    return isinstance(value, ModelResponse) and _is_message(value.output[0])


@pytest.mark.parametrize(
    "value,type_,path,check",
    [
        (
            _ExtraResult(message=_message(), sentinel="keep-me"),  # type: ignore[call-arg]
            _ExtraResult,
            ["message", "content", 0, "annotations"],
            lambda got: (
                _is_message(got.message)
                and got.model_extra == {"sentinel": "keep-me"}
                and json.loads(got.model_dump_json())["sentinel"] == "keep-me"
            ),
        ),
        (
            _NamedResult.model_validate(
                {"text": "A", "original": "B", "message": _message()}
            ),
            _NamedResult,
            ["message", "content", 0, "annotations"],
            lambda got: (
                got.text == "A" and got.original == "B" and _is_message(got.message)
            ),
        ),
        (
            _WrappedResult[ModelResponse](item=_response()),
            _WrappedResult[ModelResponse],
            ["item", "output", 0, "content", 0, "annotations"],
            lambda got: _is_response(got.item),
        ),
        (
            _DescribedResult(record=_response()),
            _DescribedResult,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: _is_response(got.record),
        ),
        (
            _MappedResult(records={"a": _response()}),
            _MappedResult,
            ["records", "a", "output", 0, "content", 0, "annotations"],
            lambda got: _is_response(got.records["a"]),
        ),
        (
            _RootResult({"a": _message()}),
            _RootResult,
            ["a", "content", 0, "annotations"],
            lambda got: _is_message(got.root["a"]),
        ),
        (
            _UnionResult(call=_call()),
            _UnionResult,
            ["call", "arguments"],
            lambda got: isinstance(got.call, ResponseFunctionToolCall),
        ),
        (
            _TimedResult(at=_AT, record=_response()),
            _TimedResult,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                got.at == _AT
                and _is_response(got.record)
                and json.loads(got.model_dump_json())["at"] == 1000.0
            ),
        ),
        (
            _NormalizedResult.model_construct(
                record=ModelResponse(
                    output=[_message()], usage=Usage(), response_id="R1"
                )
            ),
            _NormalizedResult,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: _is_response(got.record) and got.record.response_id == "r1",
        ),
        (
            _StrictResult(id=_ID, blob=_BLOB, record=_response()),
            _StrictResult,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                got.id == _ID and got.blob == _BLOB and _is_response(got.record)
            ),
        ),
        (
            _HeldResult(holder=_Holder(record=_response())),
            _HeldResult,
            ["holder", "record", "output", 0, "content", 0, "annotations"],
            lambda got: _is_response(got.holder.record),
        ),
        (
            _Holder(record=_response()),
            _Holder,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: _is_response(got.record),
        ),
        (
            _ListedResult(records=_ListRoot([_response()])),
            _ListedResult,
            ["records", 0, "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.records, _ListRoot) and _is_response(got.records.root[0])
            ),
        ),
        (
            _ListRoot([_response()]),
            _ListRoot | None,
            [0, "output", 0, "content", 0, "annotations"],
            lambda got: isinstance(got, _ListRoot) and _is_response(got.root[0]),
        ),
        (
            _InitializedPairResult(
                first=_Initialized(count=1),
                second=_Initialized(count=2),
                record=_response(),
            ),
            _InitializedPairResult,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.first, _Initialized)
                and (got.first.count, got.second.count) == (1, 2)
                and _is_response(got.record)
            ),
        ),
        (
            _ExtraSideResult(side=_extra_side(), record=_response()),
            _ExtraSideResult,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                cast(Any, got.side).sentinel == "keep-me"
                and json.loads(got.model_dump_json())["side"]["sentinel"] == "keep-me"
                and _is_response(got.record)
            ),
        ),
        (
            _TaggedResult(item=_TaggedA(Kind="a", record=_response())),
            _TaggedResult,
            ["item", "record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.item, _TaggedA) and _is_response(got.item.record)
            ),
        ),
        (
            _TaggedResult(item=_TaggedC(kind="c", record=_response())),
            _TaggedResult,
            ["item", "record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.item, _TaggedC) and _is_response(got.item.record)
            ),
        ),
        (
            _TaggedB(kind="b", record=_response()),
            _TaggedA | _TaggedB,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: isinstance(got, _TaggedB) and _is_response(got.record),
        ),
        (
            _LabelledResult(side=_Labelled(note="n"), record=_response()),
            _LabelledResult,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.side, _Labelled)
                and got.side.label == "N"
                and json.loads(got.model_dump_json())["side"]
                == {"note": "n", "label": "N"}
                and _is_response(got.record)
            ),
        ),
        (
            _CheckedResult(item=_CheckedB(kind="b", record=_response())),
            _CheckedResult,
            ["item", "record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.item, _CheckedB) and _is_response(got.item.record)
            ),
        ),
        (
            _AliasFormResult(item=_ChoicesA(kind="a", record=_response())),
            _AliasFormResult,
            ["item", "record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.item, _ChoicesA) and _is_response(got.item.record)
            ),
        ),
        (
            _AliasFormResult(item=_PathB(kind="b", record=_response())),
            _AliasFormResult,
            ["item", "record", "output", 0, "content", 0, "annotations"],
            lambda got: isinstance(got.item, _PathB) and _is_response(got.item.record),
        ),
        (
            _NormalizedTagResult(item=_UpperA(kind="a", record=_response())),
            _NormalizedTagResult,
            ["item", "record", "output", 0, "content", 0, "annotations"],
            lambda got: isinstance(got.item, _UpperA) and _is_response(got.item.record),
        ),
        (
            _LoweredMessageResult(item=_message()),
            _LoweredMessageResult,
            ["item", "content", 0, "annotations"],
            lambda got: _is_message(got.item),
        ),
        (
            _WrappedMessageResult(item=_TaggedB(kind="b", record=_response())),
            _WrappedMessageResult,
            ["item", "record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.item, _TaggedB) and _is_response(got.item.record)
            ),
        ),
        (
            _RetriedMessageResult(item=_TaggedB(kind="b", record=_response())),
            _RetriedMessageResult,
            ["item", "record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.item, _TaggedB) and _is_response(got.item.record)
            ),
        ),
        (
            _TextResult(text="done", record=_response()),
            Annotated[_TextResult, pydantic.BeforeValidator(_upper_text)],
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got, _TextResult)
                and got.text == "DONE"
                and _is_response(got.record)
            ),
        ),
        (
            _TextResult(text="done", record=_response()),
            _UpperedResult | None,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got, _TextResult)
                and got.text == "DONE"
                and _is_response(got.record)
            ),
        ),
        (
            _TextB(kind="b", text="done", record=_response()),
            _UpperedUnion,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got, _TextB)
                and got.text == "DONE"
                and _is_response(got.record)
            ),
        ),
        (
            _Box(note="n", record=_response()),
            _Box[ModelResponse],
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got, _Box) and got.note == "n" and _is_response(got.record)
            ),
        ),
        (
            _Listed(record=_response()),
            _ListedOrInts,
            [0, "output", 0, "content", 0, "annotations"],
            lambda got: isinstance(got, _Listed) and _is_response(got.record),
        ),
        (
            _ListedEnvelope(item=_Listed(record=_response())),
            _ListedEnvelope,
            ["item", 0, "output", 0, "content", 0, "annotations"],
            lambda got: isinstance(got.item, _Listed) and _is_response(got.item.record),
        ),
        (
            _ListedBox(record=_response()),
            _ListedBox[ModelResponse],
            [0, "output", 0, "content", 0, "annotations"],
            lambda got: isinstance(got, _ListedBox) and _is_response(got.record),
        ),
        (
            _ReturnedLeafResult(leaf=_Leaf(count=3), record=_response()),
            _ReturnedLeafResult,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.leaf, _Leaf)
                and got.leaf.count == 3
                and _is_response(got.record)
            ),
        ),
        (
            {"side": {"tag": "x"}, "record": _response()},
            _ReturnedDataclassResult,
            ["record", "output", 0, "content", 0, "annotations"],
            lambda got: (
                isinstance(got.side, _Revalidated)
                and got.side.tag == "x!!"
                and _is_response(got.record)
            ),
        ),
        (
            [_message()],
            _UnwrappedMessageOrInts,
            [0, "content", 0, "annotations"],
            _is_message,
        ),
    ],
    ids=[
        "extra",
        "alias-is-a-field-name",
        "generic",
        "annotated",
        "dict",
        "root-model",
        "discriminated-union",
        "millisecond-datetime",
        "before-validator",
        "strict-json",
        "annotated-optional-in-dataclass",
        "dataclass-with-annotated-optional",
        "optional-root-model-field",
        "optional-root-model",
        "custom-init-fields-beside-a-response",
        "dataclass-extra-beside-a-response",
        "literal-tagged-union",
        "literal-tagged-dataclass-in-union",
        "literal-tagged-declared-union",
        "dataclass-computed-property-beside-a-response",
        "literal-tagged-union-behind-model-validators",
        "literal-tag-with-alias-choices",
        "literal-tag-with-alias-path",
        "literal-tag-normalized-by-wrap-validator",
        "openai-variant-behind-a-normalizing-validator",
        "openai-variant-behind-a-wrap-validator",
        "openai-variant-behind-a-retrying-wrap-validator",
        "declared-annotated-before-validator",
        "declared-optional-annotated-after-validator",
        "declared-union-of-annotated-variants",
        "declared-generic-dataclass",
        "declared-union-with-list-restoring-validator",
        "union-with-list-restoring-validator-in-a-model",
        "declared-generic-dataclass-with-list-restoring-validator",
        "model-returned-by-a-validator-beside-a-response",
        "revalidated-dataclass-returned-by-a-validator-beside-a-response",
        "declared-openai-variant-unwrapping-a-list",
    ],
)
def test_lenient_caller_model_shapes(
    value: Any, type_: Any, path: list[Any], check: Callable[[Any], bool]
):
    converter = OpenAIPayloadConverter()
    payload = converter.to_payload(value)
    raw = json.loads(payload.data)
    node = raw
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    payload.data = json.dumps(raw).encode()

    received = converter.from_payload(payload, type_)

    if isinstance(type_, type):
        assert isinstance(received, type_)
    assert check(received)


def test_lenient_fallback_releases_retired_caller_types():
    # A Workflow sandbox imports a new copy of a caller's class for each run.
    payload = OpenAIPayloadConverter().to_payload(_DescribedResult(record=_response()))
    raw = json.loads(payload.data)
    del raw["record"]["output"][0]["content"][0]["annotations"]
    payload.data = json.dumps(raw).encode()
    retired = []
    for _ in range(100):
        type_ = pydantic.create_model("_Retired", record=(ModelResponse, ...))
        received = OpenAIPayloadConverter().from_payload(payload, type_)
        assert isinstance(received, type_)
        assert _is_response(cast(Any, received).record)
        retired.append(weakref.ref(type_))
        del type_, received
    gc.collect()

    assert sum(ref() is not None for ref in retired) <= 32


def test_build_tool_reports_unreconstructed_dict():
    with pytest.raises(UserError, match="not reconstructed.*dict.*name"):
        _build_tool(cast(Any, {"name": "get_weather"}), _WorkerEnvRefResolver(()))


def _round_trip_activity_input(tool: Tool) -> tuple[bytes, ActivityModelInput]:
    stub = _TemporalModelStub(
        model_name="gpt-5",
        model_params=ModelActivityParameters(),
        agent=None,
    )
    activity_input, _summary = stub._build_activity_input(
        system_instructions=None,
        input="hi",
        model_settings=ModelSettings(),
        tools=[tool],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )
    converter = OpenAIPayloadConverter()
    payload = converter.to_payload(activity_input)
    return payload.data, converter.from_payload(payload, ActivityModelInput)


def _activity_input_payload_and_tool(tool: Tool) -> tuple[bytes, Any]:
    payload, received = _round_trip_activity_input(tool)
    tools = received.get("tools") or []
    assert len(tools) == 1
    return payload, tools[0]


def _hosted_mcp_tool(authorization: str, header_value: str) -> HostedMCPTool:
    return HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "test_server",
            "server_url": "https://example.com/mcp",
            "authorization": authorization,
            "headers": {"X-Token": header_value, "X-Plain": "not-a-secret"},
        }
    )


def _domain_secret(name: str, value: str) -> dict[str, str]:
    return {"domain": "example.com", "name": name, "value": value}


def _network_policy(domain_secrets: tuple[Any, ...]) -> Any:
    return {
        "type": "allowlist",
        "allowed_domains": ["example.com"],
        "domain_secrets": list(domain_secrets),
    }


def _shell_tool(*domain_secrets: Any) -> ShellTool:
    environment: Any = {
        "type": "container_auto",
        "network_policy": _network_policy(domain_secrets),
    }
    return ShellTool(environment=environment)


def _code_interpreter_tool(*domain_secrets: Any) -> CodeInterpreterTool:
    tool_config: Any = {
        "type": "code_interpreter",
        "container": {
            "type": "auto",
            "network_policy": _network_policy(domain_secrets),
        },
    }
    return CodeInterpreterTool(tool_config=tool_config)


def _as_dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value)


def _secrets_in(network_policy: Any) -> list[Any]:
    return list(_as_dict(network_policy)["domain_secrets"])


def _shell_secrets(built: Tool) -> list[Any]:
    assert isinstance(built, ShellTool)
    assert built.environment is not None
    return _secrets_in(_as_dict(built.environment)["network_policy"])


def _code_interpreter_secrets(built: Tool) -> list[Any]:
    assert isinstance(built, CodeInterpreterTool)
    container = _as_dict(built.tool_config)["container"]
    return _secrets_in(_as_dict(container)["network_policy"])


def test_hosted_mcp_secrets_stay_out_of_activity_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)

    payload, received = _activity_input_payload_and_tool(_hosted_mcp_tool(ref, ref))

    assert SENTINEL.encode() not in payload
    assert payload.count(ref.encode()) == 2
    assert received.tool_config["authorization"] == ref
    assert received.tool_config["headers"]["X-Token"] == ref


def test_hosted_shell_domain_secret_stays_out_of_activity_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)

    payload, received = _activity_input_payload_and_tool(
        _shell_tool(_domain_secret("TOKEN", ref))
    )

    assert SENTINEL.encode() not in payload
    assert ref.encode() in payload
    secrets = _secrets_in(received.environment["network_policy"])
    assert secrets[0]["value"] == ref


def test_code_interpreter_domain_secret_stays_out_of_activity_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)

    payload, received = _activity_input_payload_and_tool(
        _code_interpreter_tool(_domain_secret("TOKEN", ref))
    )

    assert SENTINEL.encode() not in payload
    assert ref.encode() in payload
    secrets = _secrets_in(received.tool_config["container"]["network_policy"])
    assert secrets[0]["value"] == ref


def test_hosted_mcp_secrets_resolve_for_the_model_call(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(_hosted_mcp_tool(ref, ref))

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == SENTINEL
    assert _as_dict(built.tool_config)["headers"] == {
        "X-Token": SENTINEL,
        "X-Plain": "not-a-secret",
    }
    assert received.tool_config["authorization"] == ref
    assert received.tool_config["headers"]["X-Token"] == ref


@pytest.mark.parametrize(
    ("resolvable", "resolves"),
    [
        ([ENV_NAME], True),
        (AllowAllWorkerEnvVars(), True),
        ([OTHER_ENV_NAME], False),
    ],
    ids=["the_name", "allow_all", "another_name"],
)
def test_a_reference_resolves_only_from_a_variable_the_worker_allows(
    monkeypatch: pytest.MonkeyPatch,
    resolvable: Collection[str] | AllowAllWorkerEnvVars,
    resolves: bool,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(_hosted_mcp_tool(ref, ref))

    built = _build_tool(received, _WorkerEnvRefResolver(resolvable))

    assert isinstance(built, HostedMCPTool)
    expected = SENTINEL if resolves else ref
    assert _as_dict(built.tool_config)["authorization"] == expected
    assert _as_dict(built.tool_config)["headers"]["X-Token"] == expected


def test_a_star_in_the_resolvable_names_is_an_ordinary_name(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(_hosted_mcp_tool(ref, ref))

    built = _build_tool(received, _WorkerEnvRefResolver([OTHER_ENV_NAME, "*"]))

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == ref
    assert _as_dict(built.tool_config)["headers"]["X-Token"] == ref


def test_a_glob_in_the_resolvable_names_matches_no_name(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(_hosted_mcp_tool(ref, ref))

    built = _build_tool(received, _WorkerEnvRefResolver(["TEMPORAL_TEST_*"]))

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == ref
    assert _as_dict(built.tool_config)["headers"]["X-Token"] == ref


def test_hosted_shell_domain_secret_resolves_for_the_model_call(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(
        _shell_tool(_domain_secret("TOKEN", ref))
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert _shell_secrets(built) == [_domain_secret("TOKEN", SENTINEL)]


def test_code_interpreter_domain_secret_resolves_for_the_model_call(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(
        _code_interpreter_tool(_domain_secret("TOKEN", ref))
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert _code_interpreter_secrets(built) == [_domain_secret("TOKEN", SENTINEL)]


def test_code_interpreter_domain_secret_survives_a_second_build(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(
        _code_interpreter_tool(_domain_secret("TOKEN", ref))
    )

    first = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)
    second = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert _code_interpreter_secrets(first) == [_domain_secret("TOKEN", SENTINEL)]
    assert _code_interpreter_secrets(second) == [_domain_secret("TOKEN", SENTINEL)]
    assert _secrets_in(received.tool_config["container"]["network_policy"]) == [
        _domain_secret("TOKEN", ref)
    ]


def test_only_domain_secrets_holding_a_worker_env_ref_are_resolved(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)
    literal = _domain_secret("PLAIN", "plain-token-value")
    _payload, received = _activity_input_payload_and_tool(
        _code_interpreter_tool(literal, _domain_secret("TOKEN", ref))
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert _code_interpreter_secrets(built) == [
        literal,
        _domain_secret("TOKEN", SENTINEL),
    ]


def test_two_domain_secrets_resolve_to_their_own_secrets(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    monkeypatch.setenv(OTHER_ENV_NAME, OTHER_SENTINEL)
    _payload, received = _activity_input_payload_and_tool(
        _code_interpreter_tool(
            _domain_secret("TOKEN", temporal_worker_env_ref(ENV_NAME)),
            _domain_secret("OTHER", temporal_worker_env_ref(OTHER_ENV_NAME)),
        )
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert _code_interpreter_secrets(built) == [
        _domain_secret("TOKEN", SENTINEL),
        _domain_secret("OTHER", OTHER_SENTINEL),
    ]


def test_shell_domain_secrets_resolve_to_their_own_secrets_in_order(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    monkeypatch.setenv(OTHER_ENV_NAME, OTHER_SENTINEL)
    literal = _domain_secret("PLAIN", "plain-token-value")
    _payload, received = _activity_input_payload_and_tool(
        _shell_tool(
            literal,
            _domain_secret("TOKEN", temporal_worker_env_ref(ENV_NAME)),
            _domain_secret("OTHER", temporal_worker_env_ref(OTHER_ENV_NAME)),
        )
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert _shell_secrets(built) == [
        literal,
        _domain_secret("TOKEN", SENTINEL),
        _domain_secret("OTHER", OTHER_SENTINEL),
    ]


def test_a_shell_domain_secret_naming_a_denied_variable_is_passed_through(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(OTHER_ENV_NAME, OTHER_SENTINEL)
    denied = temporal_worker_env_ref(OTHER_ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(
        _shell_tool(_domain_secret("TOKEN", denied))
    )

    built = _build_tool(received, _WorkerEnvRefResolver([ENV_NAME]))

    assert _shell_secrets(built) == [_domain_secret("TOKEN", denied)]


def test_a_code_interpreter_domain_secret_naming_a_denied_variable_is_passed_through(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(OTHER_ENV_NAME, OTHER_SENTINEL)
    denied = temporal_worker_env_ref(OTHER_ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(
        _code_interpreter_tool(_domain_secret("TOKEN", denied))
    )

    built = _build_tool(received, _WorkerEnvRefResolver([ENV_NAME]))

    assert _code_interpreter_secrets(built) == [_domain_secret("TOKEN", denied)]


def test_two_worker_env_refs_in_one_mcp_config_resolve_to_their_own_secrets(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    monkeypatch.setenv(OTHER_ENV_NAME, OTHER_SENTINEL)
    _payload, received = _activity_input_payload_and_tool(
        _hosted_mcp_tool(
            temporal_worker_env_ref(ENV_NAME), temporal_worker_env_ref(OTHER_ENV_NAME)
        )
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == SENTINEL
    assert _as_dict(built.tool_config)["headers"] == {
        "X-Token": OTHER_SENTINEL,
        "X-Plain": "not-a-secret",
    }


def test_a_worker_env_ref_in_a_header_name_is_passed_through(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    ref = temporal_worker_env_ref(ENV_NAME)
    tool_config: Any = {
        "type": "mcp",
        "server_label": "test_server",
        "server_url": "https://example.com/mcp",
        "headers": {ref: "not-a-secret"},
    }

    payload, received = _activity_input_payload_and_tool(
        HostedMCPTool(tool_config=tool_config)
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["headers"] == {ref: "not-a-secret"}
    assert SENTINEL.encode() not in payload


def test_an_mcp_config_without_an_authorization_resolves_its_headers(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    tool_config: Any = {
        "type": "mcp",
        "server_label": "test_server",
        "server_url": "https://example.com/mcp",
        "headers": {"X-Token": temporal_worker_env_ref(ENV_NAME)},
    }

    _payload, received = _activity_input_payload_and_tool(
        HostedMCPTool(tool_config=tool_config)
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, HostedMCPTool)
    assert "authorization" not in _as_dict(built.tool_config)
    assert _as_dict(built.tool_config)["headers"] == {"X-Token": SENTINEL}


def test_an_mcp_config_without_headers_resolves_its_authorization(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    tool_config: Any = {
        "type": "mcp",
        "server_label": "test_server",
        "server_url": "https://example.com/mcp",
        "authorization": temporal_worker_env_ref(ENV_NAME),
    }

    _payload, received = _activity_input_payload_and_tool(
        HostedMCPTool(tool_config=tool_config)
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == SENTINEL
    assert "headers" not in _as_dict(built.tool_config)


def test_local_shell_environment_keeps_its_executor():
    _payload, received = _activity_input_payload_and_tool(
        ShellTool(environment={"type": "local"}, executor=lambda _request: "")
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, ShellTool)
    assert built.executor is not None
    assert _as_dict(built.environment) == {"type": "local"}


@pytest.mark.parametrize(
    "environment",
    [
        {"type": "container_auto"},
        {
            "type": "container_auto",
            "network_policy": {"type": "disabled"},
        },
        {"type": "container_reference", "container_id": "cntr_abc"},
    ],
    ids=["container_auto", "container_auto_disabled_policy", "container_reference"],
)
def test_hosted_shell_environment_gets_no_executor(
    environment: ShellToolEnvironment,
):
    _payload, received = _activity_input_payload_and_tool(
        ShellTool(environment=environment)
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, ShellTool)
    assert built.executor is None
    assert _as_dict(built.environment) == environment


def test_code_interpreter_container_id_is_passed_through():
    _payload, received = _activity_input_payload_and_tool(
        CodeInterpreterTool(
            tool_config={"type": "code_interpreter", "container": "cntr_abc"}
        )
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, CodeInterpreterTool)
    assert _as_dict(built.tool_config)["container"] == "cntr_abc"


@pytest.mark.parametrize(
    "container",
    [
        {"type": "auto"},
        {"type": "auto", "network_policy": {"type": "disabled"}},
    ],
    ids=["no_policy", "disabled_policy"],
)
def test_code_interpreter_container_without_domain_secrets_is_passed_through(
    container: Any,
):
    _payload, received = _activity_input_payload_and_tool(
        CodeInterpreterTool(
            tool_config={"type": "code_interpreter", "container": container}
        )
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, CodeInterpreterTool)
    assert _as_dict(built.tool_config)["container"] == container


def test_an_unset_environment_variable_resolves_to_an_empty_value(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(ENV_NAME, raising=False)
    ref = temporal_worker_env_ref(ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(_hosted_mcp_tool(ref, ref))

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == ""
    assert _as_dict(built.tool_config)["headers"]["X-Token"] == ""


def test_a_policy_with_no_domain_secrets_is_passed_through():
    policy: Any = {"type": "allowlist", "allowed_domains": ["example.com"]}
    _payload, received = _activity_input_payload_and_tool(
        CodeInterpreterTool(
            tool_config={
                "type": "code_interpreter",
                "container": {"type": "auto", "network_policy": policy},
            }
        )
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, CodeInterpreterTool)
    assert _as_dict(built.tool_config)["container"] == {
        "type": "auto",
        "network_policy": policy,
    }


def test_plain_values_are_passed_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    plain = "temporal.worker_env_ref-but-not-quite"
    _payload, received = _activity_input_payload_and_tool(
        _hosted_mcp_tool(plain, plain)
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == plain
    assert _as_dict(built.tool_config)["headers"] == {
        "X-Token": plain,
        "X-Plain": "not-a-secret",
    }


def test_a_worker_env_ref_inside_a_larger_value_is_substituted_in_place(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    composed = "Bearer " + temporal_worker_env_ref(ENV_NAME)
    _payload, received = _activity_input_payload_and_tool(
        _hosted_mcp_tool(composed, composed)
    )

    built = _build_tool(received, _RESOLVER_ALLOWING_TEST_NAMES)

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == f"Bearer {SENTINEL}"
    assert _as_dict(built.tool_config)["headers"]["X-Token"] == f"Bearer {SENTINEL}"


def test_one_value_holding_two_refs_resolves_only_the_allowed_name(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    monkeypatch.setenv(OTHER_ENV_NAME, OTHER_SENTINEL)
    denied = temporal_worker_env_ref(OTHER_ENV_NAME)
    composed = f"{temporal_worker_env_ref(ENV_NAME)} {denied}"
    _payload, received = _activity_input_payload_and_tool(
        _hosted_mcp_tool(composed, composed)
    )

    built = _build_tool(received, _WorkerEnvRefResolver([ENV_NAME]))

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == f"{SENTINEL} {denied}"
    assert _as_dict(built.tool_config)["headers"]["X-Token"] == f"{SENTINEL} {denied}"


def test_a_worker_env_ref_with_no_closing_brace_is_passed_through(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    truncated = "temporal.worker_env_ref:{" + ENV_NAME
    _payload, received = _activity_input_payload_and_tool(
        _hosted_mcp_tool(truncated, truncated)
    )

    built = _build_tool(received, _WorkerEnvRefResolver(AllowAllWorkerEnvVars()))

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == truncated
    assert _as_dict(built.tool_config)["headers"]["X-Token"] == truncated


def test_a_value_packed_with_unclosed_references_does_not_stall_the_resolver():
    opener = "temporal.worker_env_ref:{"
    packed = opener * (1024 * 1024 // len(opener))
    _payload, received = _activity_input_payload_and_tool(
        _hosted_mcp_tool(packed, "not-a-secret")
    )

    start = time.monotonic()
    built = _build_tool(received, _WorkerEnvRefResolver([]))
    elapsed = time.monotonic() - start

    assert isinstance(built, HostedMCPTool)
    assert _as_dict(built.tool_config)["authorization"] == packed
    assert elapsed < 5.0


def test_a_bare_string_is_rejected_as_the_resolvable_variable_names():
    with pytest.raises(TypeError, match="resolvable_worker_env_vars"):
        _WorkerEnvRefResolver(ENV_NAME)


def test_the_allow_all_class_itself_is_rejected_as_the_resolvable_variable_names():
    with pytest.raises(TypeError, match=r"AllowAllWorkerEnvVars\(\)"):
        _WorkerEnvRefResolver(AllowAllWorkerEnvVars)  # type: ignore[arg-type]


def test_the_plugin_rejects_a_bare_string_as_the_resolvable_variable_names():
    with pytest.raises(TypeError, match="resolvable_worker_env_vars"):
        OpenAIAgentsPlugin(resolvable_worker_env_vars=ENV_NAME)


async def _no_stream_events() -> AsyncIterator[TResponseStreamEvent]:
    """Publishing an event here makes the flusher retry for ten minutes against a workflow that does not exist."""
    events: list[TResponseStreamEvent] = []
    for event in events:
        yield event


class _ToolRecordingModel(Model):
    def __init__(self) -> None:
        self.tools: list[Tool] = []

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> ModelResponse:
        self.tools = tools
        return ModelResponse(output=[], usage=Usage(), response_id=None)

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> AsyncIterator[TResponseStreamEvent]:
        self.tools = tools
        return _no_stream_events()


def _hosted_mcp_config_the_model_received(model: _ToolRecordingModel) -> dict[str, Any]:
    assert len(model.tools) == 1
    tool = model.tools[0]
    assert isinstance(tool, HostedMCPTool)
    return _as_dict(tool.tool_config)


async def test_invoke_model_activity_resolves_tool_secrets(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    sent = _hosted_mcp_tool(temporal_worker_env_ref(ENV_NAME), "not-a-secret")
    _payload, activity_input = _round_trip_activity_input(sent)
    model = _ToolRecordingModel()

    await ActivityEnvironment().run(
        ModelActivity(
            TestModelProvider(model), resolvable_worker_env_vars=[ENV_NAME]
        ).invoke_model_activity,
        activity_input,
    )

    assert _hosted_mcp_config_the_model_received(model) == {
        **_as_dict(sent.tool_config),
        "authorization": SENTINEL,
    }


async def test_invoke_model_activity_resolves_nothing_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    sent = _hosted_mcp_tool(temporal_worker_env_ref(ENV_NAME), "not-a-secret")
    _payload, activity_input = _round_trip_activity_input(sent)
    model = _ToolRecordingModel()

    await ActivityEnvironment().run(
        ModelActivity(TestModelProvider(model)).invoke_model_activity,
        activity_input,
    )

    assert _hosted_mcp_config_the_model_received(model) == _as_dict(sent.tool_config)


async def test_invoke_model_activity_streaming_resolves_tool_secrets(
    monkeypatch: pytest.MonkeyPatch, client: Client
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    sent = _hosted_mcp_tool(temporal_worker_env_ref(ENV_NAME), "not-a-secret")
    _payload, activity_input = _round_trip_activity_input(sent)
    streaming_input: StreamingActivityModelInput = {
        **activity_input,
        "streaming_topic": "events",
    }
    model = _ToolRecordingModel()

    await ActivityEnvironment(client).run(
        ModelActivity(
            TestModelProvider(model), resolvable_worker_env_vars=[ENV_NAME]
        ).invoke_model_activity_streaming,
        streaming_input,
    )

    assert _hosted_mcp_config_the_model_received(model) == {
        **_as_dict(sent.tool_config),
        "authorization": SENTINEL,
    }


@workflow.defn
class WorkerEnvRefAgentWorkflow:
    @workflow.run
    async def run(self) -> None:
        tool_config: Any = {
            "type": "mcp",
            "server_label": "test_server",
            "server_url": "https://example.com/mcp",
            "authorization": temporal_worker_env_ref(ENV_NAME),
            "headers": {"X-Token": temporal_worker_env_ref(OTHER_ENV_NAME)},
        }
        agent = Agent[None](
            name="Worker env ref agent",
            tools=[HostedMCPTool(tool_config=tool_config)],
        )
        await Runner.run(starting_agent=agent, input="hi")


async def test_a_worker_resolves_only_the_variables_its_plugin_names(
    monkeypatch: pytest.MonkeyPatch, client: Client
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    monkeypatch.setenv(OTHER_ENV_NAME, OTHER_SENTINEL)
    model = _ToolRecordingModel()
    config = client.config()
    config["plugins"] = [
        *config.get("plugins", []),
        OpenAIAgentsPlugin(
            model_provider=TestModelProvider(model),
            resolvable_worker_env_vars=[ENV_NAME],
        ),
    ]
    client = Client(**config)

    async with new_worker(client, WorkerEnvRefAgentWorkflow) as worker:
        await client.execute_workflow(
            WorkerEnvRefAgentWorkflow.run,
            id=f"worker-env-ref-allowlist-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )

    tool_config = _hosted_mcp_config_the_model_received(model)
    assert tool_config["authorization"] == SENTINEL
    assert tool_config["headers"] == {
        "X-Token": temporal_worker_env_ref(OTHER_ENV_NAME)
    }


async def test_an_agent_environment_forwards_the_variables_it_names_to_the_worker(
    monkeypatch: pytest.MonkeyPatch, client: Client
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    monkeypatch.setenv(OTHER_ENV_NAME, OTHER_SENTINEL)
    model = _ToolRecordingModel()

    async with AgentEnvironment(
        model=model, resolvable_worker_env_vars=[ENV_NAME]
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(client, WorkerEnvRefAgentWorkflow) as worker:
            await client.execute_workflow(
                WorkerEnvRefAgentWorkflow.run,
                id=f"agent-environment-env-ref-allowlist-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )

    tool_config = _hosted_mcp_config_the_model_received(model)
    assert tool_config["authorization"] == SENTINEL
    assert tool_config["headers"] == {
        "X-Token": temporal_worker_env_ref(OTHER_ENV_NAME)
    }


async def test_re_entering_an_agent_environment_keeps_the_variables_it_names(
    monkeypatch: pytest.MonkeyPatch, client: Client
):
    monkeypatch.setenv(ENV_NAME, SENTINEL)
    model = _ToolRecordingModel()
    names = [ENV_NAME]
    env = AgentEnvironment(model=model, resolvable_worker_env_vars=names)

    async with env:
        pass
    names.clear()

    async with env:
        client = env.applied_on_client(client)
        async with new_worker(client, WorkerEnvRefAgentWorkflow) as worker:
            await client.execute_workflow(
                WorkerEnvRefAgentWorkflow.run,
                id=f"agent-environment-re-entry-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )

    assert _hosted_mcp_config_the_model_received(model)["authorization"] == SENTINEL
