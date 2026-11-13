"""Dict forms of what crosses the model Activity boundary.

Agent Framework's messages, responses and updates are ``SerializationMixin``
objects whose ``to_dict`` and ``from_dict`` produce JSON-native dicts and leave
out ``raw_representation`` (the provider SDK's objects), so the plugin needs no
payload converter of its own. Tools cross as their JSON schema specs and come
back as declaration-only ``FunctionTool`` objects, so the provider advertises
them and returns the model's calls without executing anything. A Pydantic
``response_format`` crosses as its JSON schema.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_framework import ChatResponse, ChatResponseUpdate, FunctionTool, Message
from pydantic import BaseModel

from temporalio.microsoft_agent_framework.workflow import (
    MicrosoftAgentFrameworkWorkflowError,
)


def dump_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Serialize the messages of one model call."""
    return [message.to_dict() for message in messages]


def load_messages(data: Sequence[dict[str, Any]]) -> list[Message]:
    """Rebuild the messages of one model call."""
    return [Message.from_dict(item) for item in data]


def dump_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize the chat options of one model call.

    Raises:
        MicrosoftAgentFrameworkWorkflowError: If a tool or an option value cannot be
            recorded in workflow history.
    """
    dumped: dict[str, Any] = {}
    for key, value in options.items():
        if key == "tools" and value is not None:
            value = [_dump_tool(tool) for tool in value]
        elif key == "response_format":
            value = _dump_response_format(value)
        _require_json(key, value)
        dumped[key] = value
    return dumped


HostedMCPHeaders = Mapping[str, str] | Callable[[], Mapping[str, str]]
"""Headers for one hosted MCP server: a mapping, or a callable that returns one per call."""

# Keys of a hosted MCP tool mapping that carry credentials for the provider's
# calls to the MCP server. Workflow code must not set them: the mapping is
# recorded in workflow history.
_HOSTED_MCP_CREDENTIAL_KEYS = ("headers", "authorization")
# Where a hosted MCP mapping names its server, in the shapes the providers build.
_HOSTED_MCP_NAME_KEYS = ("server_label", "name", "server_url", "url")


def load_options(
    options: Mapping[str, Any],
    hosted_mcp_headers: Mapping[str, HostedMCPHeaders] | None = None,
) -> dict[str, Any]:
    """Rebuild the chat options of one model call for the provider client.

    ``hosted_mcp_headers`` supplies the headers of hosted MCP tools, keyed by
    the server's label or URL; they are added to the tool here, on the worker.
    """
    loaded = dict(options)
    tools = loaded.get("tools")
    if tools is not None:
        loaded["tools"] = [_load_tool(tool, hosted_mcp_headers) for tool in tools]
    return loaded


# Header names that carry credentials. ``client_kwargs`` are recorded in
# workflow history, so an ``extra_headers`` entry with one of these names is
# refused; the header belongs on the provider client built in the factory.
_CREDENTIAL_HEADER_NAMES = frozenset({"authorization", "x-api-key", "api-key"})


def dump_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize the ``client_kwargs`` of one model call.

    Raises:
        MicrosoftAgentFrameworkWorkflowError: If a value cannot be recorded in workflow
            history, or ``extra_headers`` carries a credential header
            (``authorization``, ``x-api-key`` or ``api-key``, in any case).
    """
    _require_json("client_kwargs", kwargs)
    headers = kwargs.get("extra_headers")
    if isinstance(headers, Mapping):
        present = [
            key
            for key in headers
            if isinstance(key, str) and key.lower() in _CREDENTIAL_HEADER_NAMES
        ]
        if present:
            raise _credentials_in_workflow_error(
                "client_kwargs extra_headers",
                present,
                "Set the header on the provider client built in the chat client "
                "factory instead.",
            )
    return dict(kwargs)


def _credentials_in_workflow_error(
    what: str, keys: Sequence[str], fix: str
) -> MicrosoftAgentFrameworkWorkflowError:
    return MicrosoftAgentFrameworkWorkflowError(
        f"{what} sets {', '.join(keys)} in workflow code, which would record the "
        f"credentials in workflow history; credentials stay on the worker. {fix}"
    )


def dump_response(response: ChatResponse) -> dict[str, Any]:
    """Serialize a provider response."""
    data = response.to_dict()
    for message in data.get("messages") or []:
        _drop_declaration_markers(message.get("contents") or [])
    return data


def load_response(data: dict[str, Any], response_format: Any = None) -> ChatResponse:
    """Rebuild a provider response; ``response_format`` is what ``value`` parses with."""
    if response_format is None:
        return ChatResponse.from_dict(data)
    return ChatResponse.from_dict({**data, "response_format": response_format})


def dump_update(update: ChatResponseUpdate) -> dict[str, Any]:
    """Serialize one streamed update."""
    data = update.to_dict()
    _drop_declaration_markers(data.get("contents") or [])
    return data


def _drop_declaration_markers(contents: list[dict[str, Any]]) -> None:
    # Every tool reaches the provider as declaration-only, so its function
    # invocation layer marks each returned call as needing caller input. The
    # workflow's own loop decides that for the real tools, so the marker from
    # the worker side must not reach history.
    for content in contents:
        if content.get("type") == "function_call":
            content.pop("user_input_request", None)


def load_update(data: dict[str, Any]) -> ChatResponseUpdate:
    """Rebuild one streamed update."""
    return ChatResponseUpdate.from_dict(data)


class StreamRecorder:
    """Builds the Activity result of one streamed call from the provider's updates.

    Two things differ from the raw update list. The provider's function
    invocation layer, which sees every tool as declaration-only, follows each
    function call with a marked copy of it (the framework's pause for caller
    input); the copy repeats a call already streamed, so it is dropped. Text
    and tool-call arguments arrive one token per update, and a long answer or
    a large argument would put megabytes of update envelopes into history, so
    adjacent text-only updates that share their envelope are merged into one
    entry, and so are adjacent ``merge_mode="append"`` argument deltas of one
    call (matched by ``call_id``, or by the provider's ``tool_call_index`` when
    a continuation chunk carries no id). A ``replace`` update with the full
    arguments stays its own entry. ``ChatResponse.from_updates`` rebuilds the
    same response either way. ``add`` returns the serialized update to publish,
    before merging, or ``None`` for a dropped copy; ``updates`` is the record.
    """

    def __init__(self) -> None:
        """Start an empty record."""
        self._updates: list[dict[str, Any]] = []
        self._streamed_calls: set[str] = set()
        # The open run of deltas: the content dict of the last entry that
        # receives them, the key under it, and the pieces so far. They are
        # joined once when the run ends; appending to the string on every
        # delta would copy it each time, quadratic in the length of the stream.
        self._run_content: dict[str, Any] | None = None
        self._run_key = ""
        self._run_pieces: list[str] = []

    @property
    def updates(self) -> list[dict[str, Any]]:
        """The recorded entries, with the open run of deltas joined."""
        self._close_run()
        return self._updates

    def add(self, update: ChatResponseUpdate) -> dict[str, Any] | None:
        """Record ``update``; return its serialized form, or ``None`` if dropped."""
        if self._is_pause_copy(update):
            return None
        for content in update.contents:
            if content.type == "function_call" and content.call_id:
                self._streamed_calls.add(content.call_id)
        serialized = dump_update(update)
        previous = self._updates[-1] if self._updates else None
        delta = _argument_delta(serialized)
        if previous is not None and same_envelope(previous, serialized):
            if _mergeable_text(serialized) and _mergeable_text(previous):
                self._extend_run(
                    previous["contents"][0], "text", serialized["contents"][0]["text"]
                )
                return serialized
            earlier = _argument_delta(previous)
            if delta is not None and earlier is not None and _same_call(earlier, delta):
                self._extend_run(earlier, "arguments", delta["arguments"])
                return serialized
        self._close_run()
        if _mergeable_text(serialized) or delta is not None:
            # The caller publishes ``serialized`` as is; the record gets its own
            # copy, which later deltas may be merged into.
            self._updates.append(
                {**serialized, "contents": [dict(c) for c in serialized["contents"]]}
            )
        else:
            self._updates.append(serialized)
        return serialized

    def _extend_run(self, content: dict[str, Any], key: str, piece: str) -> None:
        if self._run_content is not content or self._run_key != key:
            self._close_run()
            self._run_content = content
            self._run_key = key
            self._run_pieces = [content[key]]
        self._run_pieces.append(piece)

    def _close_run(self) -> None:
        if self._run_content is not None:
            self._run_content[self._run_key] = "".join(self._run_pieces)
            self._run_content = None
            self._run_pieces = []

    def _is_pause_copy(self, update: ChatResponseUpdate) -> bool:
        if not update.contents:
            return False
        return all(
            content.type == "function_call"
            and content.user_input_request is True
            and content.call_id in self._streamed_calls
            for content in update.contents
        )


_TEXT_CONTENT_KEYS = {"type", "text", "additional_properties"}


def _mergeable_text(update: dict[str, Any]) -> bool:
    """Whether ``update`` is text only, with nothing a merge would lose."""
    contents = update.get("contents") or []
    if len(contents) != 1 or update.get("finish_reason") is not None:
        return False
    (content,) = contents
    return (
        content.get("type") == "text"
        and isinstance(content.get("text"), str)
        and set(content) <= _TEXT_CONTENT_KEYS
        and not content.get("additional_properties")
    )


def _argument_delta(update: dict[str, Any]) -> dict[str, Any] | None:
    """Return the content of a single append-mode tool-call argument delta, or ``None``."""
    contents = update.get("contents") or []
    if len(contents) != 1 or update.get("finish_reason") is not None:
        return None
    (content,) = contents
    if (
        content.get("type") == "function_call"
        and content.get("merge_mode") == "append"
        and isinstance(content.get("arguments"), str)
    ):
        return content
    return None


def _same_call(earlier: dict[str, Any], delta: dict[str, Any]) -> bool:
    """Whether ``delta`` continues the call ``earlier`` started."""
    call_id = delta.get("call_id")
    if call_id:
        return earlier.get("call_id") == call_id
    # The Chat Completions client leaves the id off continuation chunks and
    # numbers the call instead.
    index = (delta.get("additional_properties") or {}).get("tool_call_index")
    return index is not None and (
        (earlier.get("additional_properties") or {}).get("tool_call_index") == index
    )


def same_envelope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Whether two serialized updates agree on everything but their contents."""
    return {k: v for k, v in left.items() if k != "contents"} == {
        k: v for k, v in right.items() if k != "contents"
    }


def _dump_tool(tool: Any) -> dict[str, Any]:
    if isinstance(tool, FunctionTool):
        return tool.to_json_schema_spec()
    if isinstance(tool, Mapping):
        if tool.get("type") == "mcp":
            present = [key for key in _HOSTED_MCP_CREDENTIAL_KEYS if tool.get(key)]
            if present:
                raise _credentials_in_workflow_error(
                    f"Hosted MCP tool {_hosted_mcp_name(tool)!r}",
                    present,
                    "Remove them from the tool and pass them with "
                    "MicrosoftAgentFrameworkPlugin(hosted_mcp_headers=...) keyed by "
                    "the server label or URL.",
                )
        return dict(tool)
    raise MicrosoftAgentFrameworkWorkflowError(
        f"Tool {tool!r} of type {type(tool).__qualname__} cannot be sent to the "
        "model Activity. Use a @tool function, a FunctionTool, activity_as_tool(...), "
        "a provider tool mapping, or a TemporalMCPToolProvider in context_providers."
    )


def _hosted_mcp_name(tool: Mapping[str, Any]) -> str | None:
    for key in _HOSTED_MCP_NAME_KEYS:
        value = tool.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _with_hosted_mcp_headers(
    spec: Mapping[str, Any], hosted_mcp_headers: Mapping[str, HostedMCPHeaders] | None
) -> dict[str, Any]:
    tool = dict(spec)
    if not hosted_mcp_headers:
        return tool
    for key in _HOSTED_MCP_NAME_KEYS:
        value = spec.get(key)
        if isinstance(value, str) and value in hosted_mcp_headers:
            headers = hosted_mcp_headers[value]
            tool["headers"] = dict(headers() if callable(headers) else headers)
            return tool
    return tool


def _load_tool(
    spec: Any, hosted_mcp_headers: Mapping[str, HostedMCPHeaders] | None = None
) -> Any:
    if not isinstance(spec, Mapping):
        return spec
    if spec.get("type") == "mcp":
        return _with_hosted_mcp_headers(spec, hosted_mcp_headers)
    if spec.get("type") != "function":
        return spec
    function = spec.get("function")
    if not isinstance(function, Mapping):
        return spec
    parameters = function.get("parameters")
    return FunctionTool(
        name=str(function["name"]),
        description=str(function.get("description") or ""),
        func=None,
        input_model=dict(parameters) if isinstance(parameters, Mapping) else None,
    )


def _dump_response_format(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, type) and issubclass(value, BaseModel):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": value.__name__,
                "schema": value.model_json_schema(),
            },
        }
    if isinstance(value, Mapping):
        return dict(value)
    raise MicrosoftAgentFrameworkWorkflowError(
        f"response_format {value!r} is not a Pydantic model class or a mapping, "
        "so it cannot be sent to the model Activity."
    )


def _require_json(key: str, value: Any) -> None:
    offender = _first_non_json(value)
    if offender is not None:
        raise MicrosoftAgentFrameworkWorkflowError(
            f"Option {key!r} holds a {type(offender).__qualname__} value, which "
            "cannot be recorded in workflow history; pass JSON-compatible option values."
        )


def _first_non_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            offender = _first_non_json(item)
            if offender is not None:
                return offender
        return None
    if isinstance(value, Mapping):
        for item_key, item in value.items():
            if not isinstance(item_key, str):
                return item_key
            offender = _first_non_json(item)
            if offender is not None:
                return offender
        return None
    return value
