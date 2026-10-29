"""Serialization helpers, a result cache, and worker-runtime configuration.

The Deep Agents control loop runs *inside* the Temporal workflow, so the values
that actually cross the workflow⇄activity boundary are a small set of LangChain
types: chat messages, tool-call descriptors, and the ``RunnableConfig`` metadata
attached to each model / tool call. LangChain messages are polymorphic
``Serializable`` models (an ``AIMessage`` must not be rehydrated as a
``ToolMessage``) and ``RunnableConfig`` carries live callback / checkpointer
references, so neither survives a naive round-trip. This module owns:

* message (de)serialization via ``langchain_core.load.dumpd`` / ``load`` — the
  round-trip that preserves message subtype and tool-call structure;
* the strip → ship → rebuild dance for ``RunnableConfig``;
* tool → JSON-schema advertisement (full name + description + argument schema,
  never ``{name, description}`` alone — without the argument schema the model
  picks the right tool but hallucinates its arguments);
* the Pydantic data converter (``exclude_unset=True``) the plugin installs on
  the client and replayer, so message payloads stay small and round-trip
  cleanly;
* a workflow-scoped result cache so model / tool results computed before a
  ``continue_as_new`` are reused rather than recomputed after it;
* the sandbox passthrough module list covering LangChain's transitive
  eager-import tree.

LangChain imports are deferred into the functions that need them, so importing
this module — and constructing the plugin — does not require LangChain to be
installed on the machine assembling the worker.
"""

from __future__ import annotations

import base64
import contextvars
import copy
import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.runnables import RunnableConfig
    from pydantic import BaseModel

from temporalio.contrib.pydantic import PydanticPayloadConverter, ToJsonOptions
from temporalio.converter import DataConverter, DefaultPayloadConverter

# ---------------------------------------------------------------------------
# Worker-wide dispatch settings
# ---------------------------------------------------------------------------
#
# These are fixed for the worker's lifetime (not per-workflow state), so a module
# global is correct. This module lives under ``temporalio``, which the workflow
# sandbox passes through, so the object the in-workflow model stub reads is the
# same one the plugin configured. They live here (not in ``_model``) so that
# ``DeepAgentsPlugin`` can push them down without importing ``_model`` — which
# would drag in LangChain at plugin-construction time.


@dataclasses.dataclass
class Settings:
    """Dispatch defaults shared by every ``TemporalModel`` on the worker."""

    model_activity_options: Any = None
    """``ActivityConfig`` or ``Mapping[model_name, ActivityConfig]``."""
    streaming_topic: str | None = None


_settings = Settings()


def set_settings(
    *,
    model_activity_options: Any = None,
    streaming_topic: str | None = None,
) -> None:
    """Install the worker-wide model dispatch defaults (called by the plugin)."""
    _settings.model_activity_options = model_activity_options
    _settings.streaming_topic = streaming_topic


def get_settings() -> Settings:
    """Return the active dispatch settings."""
    return _settings


# ---------------------------------------------------------------------------
# Data converter
# ---------------------------------------------------------------------------


class DeepAgentsPayloadConverter(PydanticPayloadConverter):
    """Pydantic payload converter pinned to ``exclude_unset=True``.

    LangChain request/response types (and ``DeepAgentState``) are deeply nested
    with many ``Optional[...] = None`` fields. Shipping every unset default
    inflates payloads several-fold and some peers reject the explicit nulls on
    round-trip, so we exclude unset fields by convention.
    """

    def __init__(self) -> None:
        """Construct the converter with ``exclude_unset`` serialization."""
        super().__init__(ToJsonOptions(exclude_unset=True))


data_converter = DataConverter(payload_converter_class=DeepAgentsPayloadConverter)
"""The plugin's default data converter (LangChain messages are shipped as their
``dumpd`` JSON form, so the Pydantic converter only ever sees plain containers)."""


def build_data_converter(
    user_converter: DataConverter | None,
) -> DataConverter:
    """Compose the plugin's converter with whatever the caller already set.

    * ``None`` — install the plugin default.
    * the SDK default payload converter — swap in the LangChain-aware Pydantic
      converter while preserving the codec and other data converter settings.
    * a compatible payload converter — preserve the supplied data converter.
    * any other payload converter — refuse rather than silently clobber it.
    """
    if user_converter is None:
        return data_converter
    if user_converter.payload_converter_class is DefaultPayloadConverter:
        return dataclasses.replace(
            user_converter, payload_converter_class=DeepAgentsPayloadConverter
        )
    if issubclass(user_converter.payload_converter_class, DeepAgentsPayloadConverter):
        return user_converter
    raise ValueError(
        "DeepAgentsPlugin cannot compose with a custom data_converter "
        "automatically. Set payload_converter_class=DeepAgentsPayloadConverter "
        "on your own DataConverter (so LangChain messages serialize with "
        "exclude_unset=True)."
    )


# ---------------------------------------------------------------------------
# LangChain object (de)serialization
# ---------------------------------------------------------------------------


def dump_object(obj: Any) -> Any:
    """Serialize a single LangChain ``Serializable`` (message, tool call, …)."""
    from langchain_core.load import dumpd

    return dumpd(obj)


def load_object(data: Any) -> Any:
    """Rehydrate a value produced by :func:`dump_object`, preserving subtype."""
    from langchain_core.load import load

    return load(data)


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _tool_arg_classes() -> tuple[type[BaseModel], ...]:
    """The built-in classes :func:`find_tool_arg_messages` records.

    Every class in ``langchain_core.messages``, including ``ChatMessage``,
    ``FunctionMessage`` and the chunks, and ``Document``. Other classes are
    recorded only when the tool's annotations name them; see
    :func:`tool_arg_types`.
    """
    from langchain_core.documents import Document
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
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

    return (
        AIMessage,
        AIMessageChunk,
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
        Document,
    )


def tool_arg_types(tool: Any) -> dict[str, type[BaseModel]]:
    """The Pydantic classes a tool's argument schema checks its arguments against.

    For example ``CitedMessage`` in
    ``Annotated[list[CitedMessage], InjectedState("notes")]``, also inside a
    state schema or another such class. They come from the schema Pydantic
    built, so they are the classes LangChain validates the arguments with, even
    for a tool registered from inside the workflow sandbox. Each is keyed by
    module and qualified name, which also matches the sandbox's copy of the
    class. The built-in classes are left out; they are always rebuilt.
    """
    from pydantic import BaseModel

    schema = getattr(tool, "args_schema", None)
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return {}
    skip = (*_tool_arg_classes(), schema)
    found: dict[str, type[BaseModel]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            cls = node.get("cls")
            if (
                node.get("type") == "model"
                and isinstance(cls, type)
                and issubclass(cls, BaseModel)
                and cls not in skip
            ):
                found[_qualified_name(cls)] = cls
            for value in node.values():
                visit(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(schema.__pydantic_core_schema__)
    return found


def _tool_arg_class_name(cls: type, types: Mapping[str, type[BaseModel]]) -> str | None:
    """The name :func:`find_tool_arg_messages` records for ``cls``, if any.

    A built-in class is recorded by its name, a class from ``types`` by its
    module and qualified name.
    """
    if cls in _tool_arg_classes():
        return cls.__name__
    name = _qualified_name(cls)
    return name if name in types else None


def find_tool_arg_messages(
    value: Any,
    types: Mapping[str, type[BaseModel]],
    path: tuple[str | int, ...] = (),
) -> list[tuple[list[str | int], str]]:
    """Find the LangChain messages in a wrapped tool's arguments.

    A tool can read the conversation from injected state, so its arguments can
    hold messages. The Pydantic converter writes each one as a plain dict, so
    the activity cannot tell it was a message. This returns the path and class
    name of each message, ``Document`` and ``types`` value (see
    :func:`tool_arg_types`) in nested lists, tuples, string-keyed dicts and the
    fields of those values, for :func:`load_tool_arg_messages` to rebuild. The
    arguments themselves are sent unchanged.
    """
    found: list[tuple[list[str | int], str]] = []
    name = _tool_arg_class_name(type(value), types)
    if name is not None:
        found.append((list(path), name))
        # The converter writes only the fields that were set, extras included.
        fields = {**vars(value), **(value.model_extra or {})}
        for key, item in fields.items():
            if key in value.model_fields_set:
                found.extend(find_tool_arg_messages(item, types, (*path, key)))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            found.extend(find_tool_arg_messages(item, types, (*path, i)))
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                found.extend(find_tool_arg_messages(item, types, (*path, key)))
    return found


def load_tool_arg_messages(
    args: dict[str, Any],
    messages: list[tuple[list[str | int], str]],
    types: Mapping[str, type[BaseModel]],
) -> dict[str, Any]:
    """Rebuild the values :func:`find_tool_arg_messages` found in ``args``.

    Each recorded value is validated as its class, innermost first, so a
    ``Document`` artifact is an object before its ``ToolMessage`` is rebuilt.
    Every other value is returned unchanged.
    """
    classes = {cls.__name__: cls for cls in _tool_arg_classes()}
    classes.update(types)
    args = copy.deepcopy(args)
    for path, name in sorted(messages, key=lambda m: -len(m[0])):
        cls = classes.get(name)
        if cls is None:
            raise ValueError(f"Unknown tool argument class {name!r}")
        parent: Any = args
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = cls.model_validate(parent[path[-1]])
    return args


# ---------------------------------------------------------------------------
# Backend protocol value (de)serialization
# ---------------------------------------------------------------------------

_BACKEND_DATACLASS_KEY = "__deepagents_dataclass__"
_BACKEND_BYTES_KEY = "__deepagents_bytes__"


def dump_backend_result(value: Any) -> Any:
    """Encode a backend op's arguments or return value for the activity boundary.

    Backend protocol results (``WriteResult`` / ``ReadResult`` / ``GrepResult``
    and their nested ``FileInfo`` / ``GrepMatch`` items, …) are plain
    dataclasses — not LangChain ``Serializable`` objects — and the filesystem
    middleware reads their ATTRIBUTES in-workflow, so a plain JSON round-trip
    (which decays them to dicts) breaks the seam at the first real backend op.
    Tag deepagents dataclasses with their import path so
    :func:`load_backend_result` rebuilds the real type; anything else (str,
    dict, a custom backend's own types) passes through with today's
    plain-JSON behavior. ``bytes`` (``upload_files`` content,
    ``FileDownloadResponse.content``) is tagged and base64-encoded: JSON has no
    bytes type, and file content need not be valid UTF-8.
    """
    import dataclasses

    if isinstance(value, bytes):
        return {_BACKEND_BYTES_KEY: base64.b64encode(value).decode("ascii")}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        dumped_fields = {
            f.name: dump_backend_result(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
        if cls.__module__.split(".", 1)[0] == "deepagents":
            return {
                _BACKEND_DATACLASS_KEY: f"{cls.__module__}:{cls.__qualname__}",
                "fields": dumped_fields,
            }
        return dumped_fields
    if isinstance(value, (list, tuple)):
        return [dump_backend_result(v) for v in value]
    if isinstance(value, dict):
        return {k: dump_backend_result(v) for k, v in value.items()}
    return value


def load_backend_result(value: Any) -> Any:
    """Rebuild a value produced by :func:`dump_backend_result`.

    Only ``deepagents.*`` dataclasses are reconstructed (the tag is written
    exclusively for them); anything else would mean a forged payload, so
    refuse rather than import arbitrary types. Reconstruction suppresses
    ``DeprecationWarning``: this is transport, not user code — the backend
    already constructed the object once on the activity side, and required
    deprecated fields (e.g. ``WriteResult.files_update``) would otherwise
    warn on every op. Field values equal to a declared default are omitted
    from the constructor call.
    """
    import dataclasses
    import importlib
    import warnings

    if isinstance(value, dict) and _BACKEND_BYTES_KEY in value:
        return base64.b64decode(value[_BACKEND_BYTES_KEY])
    if isinstance(value, dict) and _BACKEND_DATACLASS_KEY in value:
        path = value[_BACKEND_DATACLASS_KEY]
        module_name, _, qualname = path.partition(":")
        if module_name.split(".", 1)[0] != "deepagents":
            raise ValueError(f"refusing to rebuild non-deepagents type {path!r}")
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            raise ValueError(f"{path!r} is not a dataclass")
        loaded = {k: load_backend_result(v) for k, v in value["fields"].items()}
        kwargs = {}
        for f in dataclasses.fields(obj):
            if f.name not in loaded:
                continue
            if f.default is not dataclasses.MISSING and loaded[f.name] == f.default:
                continue
            kwargs[f.name] = loaded[f.name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return obj(**kwargs)
    if isinstance(value, list):
        return [load_backend_result(v) for v in value]
    if isinstance(value, dict):
        return {k: load_backend_result(v) for k, v in value.items()}
    return value


def dump_messages(messages: Any) -> list[Any]:
    """Serialize a sequence of LangChain messages to their ``dumpd`` form."""
    from langchain_core.load import dumpd

    return [dumpd(m) for m in messages]


def load_messages(dumped: list[Any]) -> list[Any]:
    """Rehydrate messages serialized by :func:`dump_messages`."""
    from langchain_core.load import load

    return [load(d) for d in dumped]


def tool_to_schema(tool: Any) -> dict[str, Any]:
    """Advertise a tool to the model as a full OpenAI tool schema.

    Carries name + description + argument JSON schema so the model can build
    valid arguments, not just select the tool by name.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return convert_to_openai_tool(tool)


# ---------------------------------------------------------------------------
# RunnableConfig strip / rebuild
# ---------------------------------------------------------------------------


def _is_jsonish(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_jsonish(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_jsonish(v) for k, v in value.items())
    return False


def strip_runnable_config(config: Any) -> dict[str, Any]:
    """Reduce a live ``RunnableConfig`` to its JSON-safe subset for shipping.

    Keeps ``tags``, ``run_name``, ``run_id``, ``recursion_limit``, JSON-safe
    ``metadata`` and the JSON-safe (non-dunder) ``configurable`` keys. Drops
    callbacks, checkpointer / store / cache handles and every other live
    reference — those are reconstructed activity-side.
    """
    if not config:
        return {}
    out: dict[str, Any] = {}
    if config.get("tags"):
        out["tags"] = list(config["tags"])
    for key in ("run_id", "run_name"):
        if config.get(key) is not None:
            out[key] = str(config[key])
    if config.get("recursion_limit") is not None:
        out["recursion_limit"] = config["recursion_limit"]
    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        out["metadata"] = {k: v for k, v in metadata.items() if _is_jsonish(v)}
    configurable = config.get("configurable")
    if isinstance(configurable, dict):
        kept = {
            k: v
            for k, v in configurable.items()
            if not k.startswith("__") and _is_jsonish(v)
        }
        if kept:
            out["configurable"] = kept
    return out


def rebuild_runnable_config(data: dict[str, Any]) -> "RunnableConfig":
    """Reconstruct a minimal ``RunnableConfig`` from :func:`strip_runnable_config`."""
    config: dict[str, Any] = {"metadata": dict(data.get("metadata", {}))}
    if data.get("tags"):
        config["tags"] = list(data["tags"])
    for key in ("run_id", "run_name"):
        if data.get(key) is not None:
            config[key] = data[key]
    if data.get("recursion_limit") is not None:
        config["recursion_limit"] = data["recursion_limit"]
    if data.get("configurable"):
        config["configurable"] = dict(data["configurable"])
    # Double cast: a TypedDict and dict[str, Any] "insufficiently overlap"
    # for basedpyright's reportInvalidCast; object is the sanctioned bridge.
    return cast("RunnableConfig", cast(object, config))


# ---------------------------------------------------------------------------
# Result cache (continue-as-new dedup)
# ---------------------------------------------------------------------------

# Per-workflow state: set at the top of the workflow run, read by the model and
# tool dispatch paths, snapshotted for continue-as-new. A ContextVar (not a
# module-global keyed by run id) so update / signal handler tasks spawned by the
# workflow inherit the same cache automatically.
_result_cache: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_deepagents_result_cache", default=None
)


def set_result_cache(cache: dict[str, Any] | None) -> None:
    """Seed the workflow-scoped result cache (e.g. carried across CAN)."""
    _result_cache.set(dict(cache) if cache else {})


def result_cache_snapshot() -> dict[str, Any] | None:
    """Return a serializable copy of the cache, or ``None`` when empty."""
    cache = _result_cache.get()
    return dict(cache) if cache else None


def cache_key(kind: str, call_id: str, args: Any) -> str:
    """Stable key over ``(kind, call_id, args)`` for cache lookups."""
    blob = json.dumps([kind, call_id, args], default=str, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_lookup(key: str) -> tuple[bool, Any]:
    """Return ``(hit, value)`` for ``key`` in the active cache."""
    cache = _result_cache.get()
    if cache is not None and key in cache:
        return True, cache[key]
    return False, None


def cache_put(key: str, value: Any) -> None:
    """Record ``value`` under ``key`` when a cache is active for this run."""
    cache = _result_cache.get()
    if cache is not None:
        cache[key] = value


# ---------------------------------------------------------------------------
# Sandbox passthrough
# ---------------------------------------------------------------------------

# The workflow sandbox re-imports modules per run; LangChain / LangGraph /
# deepagents build large class hierarchies with eager import side effects, and
# LangSmith pulls in numpy. Passing them through means "import once in the host
# and share", which is both faster and required for identity checks
# (isinstance across the sandbox boundary) to hold.
_DEFAULT_PASSTHROUGH: tuple[str, ...] = (
    "langchain",
    "langchain_core",
    "langchain_anthropic",
    "langgraph",
    "deepagents",
    "langsmith",
    "numpy",
    "pydantic",
    "pydantic_core",
    "anthropic",
    "tiktoken",
    "jsonpatch",
    "jsonpointer",
    "tenacity",
    "orjson",
    "httpx",
    "httpcore",
)


def default_passthrough_modules() -> tuple[str, ...]:
    """The LangChain / deepagents transitive import tree passed through the sandbox."""
    return _DEFAULT_PASSTHROUGH


def resolve_passthrough_modules(user: Any) -> tuple[str, ...]:
    """Merge caller-supplied passthrough modules with the plugin defaults."""
    merged = [*_DEFAULT_PASSTHROUGH, *(user or ())]
    # dict.fromkeys preserves order while de-duplicating.
    return tuple(dict.fromkeys(merged))
