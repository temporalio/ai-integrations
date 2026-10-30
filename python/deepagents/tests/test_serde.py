"""The LangChain object (de)serialization that crosses the activity boundary.

Model inputs, model outputs (streaming chunks included) and tool results are
the LangChain values shipped between workflow and activity, and they are all
chat messages. These tests pin the load side of that seam: every message class
``langchain_core.messages`` exports round-trips with its subtype intact, the
payload shape already sitting in recorded histories (legacy
``langchain.schema.messages.*`` ids) keeps loading, and no call relies on — or
warns about — ``load``'s default ``allowed_objects``.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="deepagents requires Python >= 3.11"
)

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    RemoveMessage,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)

from temporalio.deepagents import _serde  # noqa: E402

_MESSAGE_CLASSES: tuple[type[Any], ...] = (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    RemoveMessage,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)

# Literal ``dumpd`` payloads captured from main's dump side (``dump_object``
# is untouched there and here): the shapes the agent loop produces, plus the
# legacy-only ``ChatMessage`` / ``FunctionMessage`` that LangChain's suggested
# ``allowed_objects="messages"`` spelling refuses — histories recorded before
# this change must keep loading both.
_RECORDED_PAYLOADS: dict[str, tuple[type[Any], dict[str, Any]]] = {
    "human": (
        HumanMessage,
        {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "messages", "HumanMessage"],
            "kwargs": {"content": "find the docs", "type": "human"},
        },
    ),
    "ai_with_tool_call": (
        AIMessage,
        {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "messages", "AIMessage"],
            "kwargs": {
                "content": "looking",
                "type": "ai",
                "tool_calls": [
                    {
                        "name": "search",
                        "args": {"q": "temporal"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
                "invalid_tool_calls": [],
            },
        },
    ),
    "tool": (
        ToolMessage,
        {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "messages", "ToolMessage"],
            "kwargs": {
                "content": "42",
                "type": "tool",
                "tool_call_id": "c1",
                "status": "success",
            },
        },
    ),
    "chat_legacy": (
        ChatMessage,
        {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "messages", "ChatMessage"],
            "kwargs": {
                "content": "find the docs",
                "type": "chat",
                "role": "assistant",
            },
        },
    ),
    "function_legacy": (
        FunctionMessage,
        {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "messages", "FunctionMessage"],
            "kwargs": {"content": "42", "type": "function", "name": "calc"},
        },
    ),
}


def _instantiate(cls: type[Any]) -> Any:
    """Build one instance of each message class ``langchain_core`` exports."""
    if cls in (ChatMessage, ChatMessageChunk):
        return cls(role="assistant", content="find the docs")
    if cls in (FunctionMessage, FunctionMessageChunk):
        return cls(content="42", name="calc")
    if cls in (ToolMessage, ToolMessageChunk):
        return cls(content="42", tool_call_id="c1")
    if cls is RemoveMessage:
        return cls(id="m1")
    if cls in (BaseMessage, BaseMessageChunk):
        return cls(content="find the docs", type="human")
    if issubclass(cls, SystemMessage):
        return cls(content="be terse")
    return cls(content="find the docs")


def _allowed_objects_warnings(
    caught: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    """Return the ``allowed_objects`` pending-deprecation warnings among ``caught``.

    ``langchain_core.load.load`` emits one on every call that relies on its
    default ``allowed_objects``, and that default is scheduled to narrow,
    which would change what the plugin's own payloads revive. Matched on the
    category plus the ``allowed_objects`` message text so unrelated
    deprecations (including ``load``'s own beta warning) are not implicated.
    """
    return [
        w
        for w in caught
        if issubclass(w.category, PendingDeprecationWarning)
        and "allowed_objects" in str(w.message)
    ]


def _load_without_allowed_objects_warning(data: Any) -> Any:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        revived = _serde.load_object(data)
    assert not _allowed_objects_warnings(caught), _allowed_objects_warnings(caught)
    return revived


@pytest.mark.parametrize("cls", _MESSAGE_CLASSES, ids=lambda c: c.__name__)
def test_message_round_trip_preserves_subtype(cls: type[Any]) -> None:
    """Every message class round-trips with its subtype, warning-clean."""
    message = _instantiate(cls)
    revived = _load_without_allowed_objects_warning(_serde.dump_object(message))
    assert type(revived) is cls
    assert revived == message


def test_message_list_round_trip_preserves_subtypes() -> None:
    """``load_messages`` revives the whole input state, warning-clean."""
    messages = [_instantiate(cls) for cls in _MESSAGE_CLASSES]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        revived = _serde.load_messages(_serde.dump_messages(messages))
    assert [type(m) for m in revived] == [type(m) for m in messages]
    assert not _allowed_objects_warnings(caught)


@pytest.mark.parametrize("name", sorted(_RECORDED_PAYLOADS))
def test_recorded_payload_loads(name: str) -> None:
    """A payload as ``dumpd`` wrote it before this change still revives."""
    expected_type, payload = _RECORDED_PAYLOADS[name]
    revived = _load_without_allowed_objects_warning(payload)
    assert type(revived) is expected_type
    assert revived.content == payload["kwargs"]["content"]


def test_recorded_payloads_load_as_a_message_list() -> None:
    """``load_messages`` revives a whole recorded input state."""
    revived = _serde.load_messages(
        [payload for _, payload in _RECORDED_PAYLOADS.values()]
    )
    assert [type(m) for m in revived] == [t for t, _ in _RECORDED_PAYLOADS.values()]


def test_load_keeps_current_default_semantics() -> None:
    """Pinning ``allowed_objects`` changes nothing about what revives.

    A message from caller state can embed any core serializable in
    ``additional_kwargs`` (``dumpd`` encodes it as a nested constructor), and
    today's default revive set includes those. This pins that the explicit
    allowlist equals main's behavior; narrowing it would be its own change.
    """
    message = AIMessage(
        content="hi", additional_kwargs={"doc": Document(page_content="d")}
    )
    revived = _load_without_allowed_objects_warning(_serde.dump_object(message))
    assert isinstance(revived.additional_kwargs["doc"], Document)
