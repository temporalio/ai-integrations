"""Test helpers for agents that run through ``AgentFrameworkPlugin``.

Register a scripted client under the model name your workflow uses and the
model Activity runs offline::

    plugin = AgentFrameworkPlugin(
        chat_clients={"gpt-4o": fake_chat_client(["Paris."])}
    )

``FakeChatClient`` is a full Agent Framework chat client (function invocation,
chat middleware and the base client), so it behaves like a provider client on
the worker: with declaration-only tools it returns the scripted function calls
unexecuted and assigns their occurrence ids, as a real provider's client does.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any, ClassVar, cast

from agent_framework import (
    BaseChatClient,
    ChatAndFunctionMiddlewareTypes,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    Message,
    ResponseStream,
)

__all__ = [
    "FakeChatClient",
    "fake_chat_client",
    "function_call_response",
    "text_response",
]

ScriptEntry = str | ChatResponse | Exception
"""One scripted model call: a text reply, a full response, or an error to raise."""

_STREAM_CHUNK_CHARS = 12


def text_response(text: str, *, response_id: str | None = None) -> ChatResponse:
    """Return an assistant text reply."""
    return ChatResponse(
        messages=[Message(role="assistant", contents=[Content.from_text(text)])],
        response_id=response_id,
    )


def function_call_response(
    name: str,
    arguments: Mapping[str, Any],
    *,
    call_id: str,
    id: str | None = None,
) -> ChatResponse:
    """Return an assistant reply that calls ``name`` with ``arguments``.

    ``call_id`` is the provider's call id that the result answers. ``id`` is
    the occurrence id; leave it ``None`` to script a provider that omits it,
    so the client's function-invocation layer assigns one.
    """
    return ChatResponse(
        messages=[
            Message(
                role="assistant",
                contents=[
                    Content.from_function_call(
                        call_id, name, arguments=dict(arguments), id=id
                    )
                ],
            )
        ]
    )


class FakeChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):  # type: ignore[type-arg]
    """A scripted chat client for tests.

    Each model call takes the next entry of ``responses``; the last entry
    repeats once the script runs out. A ``str`` becomes a text reply, a
    ``ChatResponse`` is returned as is (``text_response`` and
    ``function_call_response`` build the common shapes) and an ``Exception`` is
    raised from the call. A streamed call splits text into updates of 12
    characters and sends every other content item as one update.

    ``calls`` records every call as a dict with ``messages``, ``options`` and
    ``stream``, so a test can assert what reached the model.
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "temporal-fake"

    def __init__(
        self,
        responses: Sequence[ScriptEntry],
        *,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    ) -> None:
        """Script the client; ``responses`` must hold at least one entry."""
        if not responses:
            raise ValueError("FakeChatClient needs at least one scripted response.")
        super().__init__(middleware=middleware)
        self._responses = list(responses)
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []

    def _next(self) -> ScriptEntry:
        entry = self._responses[min(self._cursor, len(self._responses) - 1)]
        self._cursor += 1
        return entry

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        self.calls.append(
            {"messages": list(messages), "options": dict(options), "stream": stream}
        )
        entry = self._next()
        if stream:
            return ResponseStream(
                _updates(entry),
                finalizer=lambda updates: ChatResponse.from_updates(
                    updates, output_format_type=options.get("response_format")
                ),
            )
        return _respond(entry)


async def _respond(entry: ScriptEntry) -> ChatResponse:
    return _response(entry)


def _response(entry: ScriptEntry) -> ChatResponse:
    if isinstance(entry, Exception):
        raise entry
    if isinstance(entry, str):
        return text_response(entry)
    # The loop mutates responses (ids, author names); keep the script intact.
    return copy.deepcopy(entry)


async def _updates(entry: ScriptEntry) -> AsyncIterator[ChatResponseUpdate]:
    response = _response(entry)
    for message in response.messages:
        for content in message.contents:
            if content.type == "text" and content.text:
                text = content.text
                for start in range(0, len(text), _STREAM_CHUNK_CHARS):
                    yield ChatResponseUpdate(
                        role=cast(Any, message.role),
                        contents=[
                            Content.from_text(text[start : start + _STREAM_CHUNK_CHARS])
                        ],
                        response_id=response.response_id,
                        message_id=message.message_id,
                    )
            else:
                yield ChatResponseUpdate(
                    role=cast(Any, message.role),
                    contents=[content],
                    response_id=response.response_id,
                    message_id=message.message_id,
                )


def fake_chat_client(responses: Sequence[ScriptEntry]) -> Callable[[], FakeChatClient]:
    """Return a ``chat_clients`` factory for one scripted ``FakeChatClient``.

    The factory hands out the same instance every time, so a test can call it
    to read ``calls`` after the workflow finishes.
    """
    client = FakeChatClient(responses)
    return lambda: client
