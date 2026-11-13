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

import copy
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
    adjacent text-only updates that share their envelope, their ``merge_mode``
    (none or ``append``, as the OpenAI clients emit) and their other
    properties are merged into one entry, with their ``logprobs`` lists
    concatenated and empty ones dropped; so are adjacent
    ``merge_mode="append"`` argument deltas of one call (matched by
    ``call_id``, or by the provider's ``tool_call_index`` when a continuation
    chunk carries no id). A ``replace`` update with the full text or
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
            if (
                _mergeable_text(serialized)
                and _mergeable_text(previous)
                and _same_text_run(previous["contents"][0], serialized["contents"][0])
            ):
                target = previous["contents"][0]
                self._extend_run(target, "text", serialized["contents"][0]["text"])
                _merge_logprobs(target, serialized["contents"][0])
                return serialized
            earlier = _argument_delta(previous)
            if delta is not None and earlier is not None and _same_call(earlier, delta):
                self._extend_run(earlier, "arguments", delta["arguments"])
                return serialized
        self._close_run()
        if _mergeable_text(serialized) or delta is not None:
            # The caller publishes ``serialized`` as is, possibly later in a
            # batch; the record gets its own copy, which later deltas are
            # merged into.
            self._updates.append(
                {
                    **serialized,
                    "contents": [_copy_content(c) for c in serialized["contents"]],
                }
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
            if self._run_key == "text":
                _drop_empty_logprobs(self._run_content)
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


_TEXT_CONTENT_KEYS = {"type", "text", "merge_mode", "additional_properties"}


def _mergeable_text(update: dict[str, Any]) -> bool:
    """Whether ``update`` is one text delta, with nothing a merge would lose."""
    contents = update.get("contents") or []
    if len(contents) != 1 or update.get("finish_reason") is not None:
        return False
    (content,) = contents
    return (
        content.get("type") == "text"
        and isinstance(content.get("text"), str)
        and set(content) <= _TEXT_CONTENT_KEYS
        and content.get("merge_mode") in (None, "append")
        and isinstance(content.get("additional_properties", {}), dict)
    )


def _same_text_run(earlier: dict[str, Any], content: dict[str, Any]) -> bool:
    """Whether two text deltas agree on everything a merge keeps once: mode and properties."""
    return earlier.get("merge_mode") == content.get("merge_mode") and (
        _without_logprobs(earlier.get("additional_properties"))
        == _without_logprobs(content.get("additional_properties"))
    )


def _without_logprobs(properties: Any) -> dict[str, Any]:
    if not isinstance(properties, dict):
        return {}
    rest = {k: v for k, v in properties.items() if k != "logprobs"}
    openai = rest.get("openai")
    if isinstance(openai, dict):
        openai = {k: v for k, v in openai.items() if k != "logprobs"}
        if openai:
            rest["openai"] = openai
        else:
            rest.pop("openai")
    return rest


def _merge_logprobs(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Append the incoming delta's ``logprobs`` (top level and under ``openai``) to the record's."""
    source = incoming.get("additional_properties")
    if not isinstance(source, dict):
        return
    properties = target.setdefault("additional_properties", {})
    for namespace in (None, "openai"):
        holder = source if namespace is None else source.get(namespace)
        if not isinstance(holder, dict) or not isinstance(holder.get("logprobs"), list):
            continue
        destination = (
            properties if namespace is None else properties.setdefault(namespace, {})
        )
        existing = destination.get("logprobs")
        if isinstance(existing, list):
            existing.extend(holder["logprobs"])
        else:
            destination["logprobs"] = list(holder["logprobs"])


def _drop_empty_logprobs(content: dict[str, Any]) -> None:
    properties = content.get("additional_properties")
    if not isinstance(properties, dict):
        return
    if properties.get("logprobs") == []:
        properties.pop("logprobs")
    openai = properties.get("openai")
    if isinstance(openai, dict):
        if openai.get("logprobs") == []:
            openai.pop("logprobs")
        if not openai:
            properties.pop("openai")


def _copy_content(content: dict[str, Any]) -> dict[str, Any]:
    """Copy a content dict so merging into the record never touches the published one."""
    copied = dict(content)
    properties = content.get("additional_properties")
    if isinstance(properties, dict):
        copied["additional_properties"] = copy.deepcopy(properties)
    return copied


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
        # The form the OpenAI clients send natively for a Pydantic class: the
        # strict schema under the class name, with strict=True, so the
        # provider holds the model to it.
        schema = value.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": value.__name__,
                "schema": _strict_json_schema(schema, root=schema),
                "strict": True,
            },
        }
    if isinstance(value, Mapping):
        return dict(value)
    raise MicrosoftAgentFrameworkWorkflowError(
        f"response_format {value!r} is not a Pydantic model class or a mapping, "
        "so it cannot be sent to the model Activity."
    )


def _strict_json_schema(
    schema: dict[str, Any], *, root: dict[str, Any]
) -> dict[str, Any]:
    """Rewrite ``schema`` in place into the providers' strict form, as the OpenAI SDK does.

    Every object gets ``additionalProperties: false`` and all of its
    properties required; ``$defs``, ``items``, ``anyOf`` and ``allOf`` are
    walked, a single-entry ``allOf`` is flattened, a ``None`` default is
    dropped, and a ``$ref`` with sibling keys is inlined.
    """
    for key in ("$defs", "definitions"):
        definitions = schema.get(key)
        if isinstance(definitions, dict):
            for definition in definitions.values():
                if isinstance(definition, dict):
                    _strict_json_schema(definition, root=root)
    if schema.get("type") == "object" and "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["required"] = list(properties)
        for prop in properties.values():
            if isinstance(prop, dict):
                _strict_json_schema(prop, root=root)
    items = schema.get("items")
    if isinstance(items, dict):
        _strict_json_schema(items, root=root)
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        for variant in any_of:
            if isinstance(variant, dict):
                _strict_json_schema(variant, root=root)
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        if len(all_of) == 1 and isinstance(all_of[0], dict):
            schema.pop("allOf")
            schema.update(_strict_json_schema(all_of[0], root=root))
        else:
            for entry in all_of:
                if isinstance(entry, dict):
                    _strict_json_schema(entry, root=root)
    if "default" in schema and schema["default"] is None:
        schema.pop("default")
    ref = schema.get("$ref")
    if isinstance(ref, str) and len(schema) > 1:
        resolved = _resolve_ref(root, ref)
        if isinstance(resolved, dict):
            # The schema's own keys take priority over the referenced ones.
            schema.update({**resolved, **schema})
            schema.pop("$ref")
            return _strict_json_schema(schema, root=root)
    return schema


def _resolve_ref(root: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


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
