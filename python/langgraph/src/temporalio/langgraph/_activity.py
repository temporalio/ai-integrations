"""Activity wrappers for executing LangGraph nodes and tasks."""

import asyncio
import dataclasses
import typing
from collections.abc import Awaitable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import cache
from inspect import iscoroutinefunction, signature
from types import UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

from langchain_core.messages import BaseMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, Interrupt, Send, TimeoutPolicy
from pydantic import (
    BaseModel,
    PlainSerializer,
    RootModel,
    TypeAdapter,
    WrapSerializer,
)
from pydantic.fields import FieldInfo
from typing_extensions import NotRequired, Required, is_typeddict

from temporalio import activity, workflow
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.converter import PayloadConverter
from temporalio.langgraph._langgraph_config import (
    context_schema_id,
    get_langgraph_config,
    is_context_schema,
    set_langgraph_config,
    strip_runnable_config,
)
from temporalio.langgraph._task_cache import (
    cache_key,
    cache_lookup,
    cache_put,
)

# Per-run dedupe so we only warn once when a user passes a Store via
# graph.compile(store=...) / @entrypoint(store=...). Cleared by
# LangGraphInterceptor.execute_workflow on workflow exit.
_warned_store_runs: set[str] = set()


def clear_store_warning(run_id: str) -> None:
    """Drop the store-warning dedupe entry for a workflow run."""
    _warned_store_runs.discard(run_id)


@dataclass
class ActivityInput:
    """Input for a LangGraph activity, containing args, kwargs, and config."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    langgraph_config: dict[str, Any]
    value_types: list[Any] | None = None


@dataclass
class ActivityOutput:
    """Output from an Activity, containing result, command, or interrupts."""

    result: Any = None
    langgraph_command: Any = None
    langgraph_interrupts: tuple[Interrupt] | None = None
    value_types: list[Any] | None = None


def _command_to_payload(command: Command) -> dict[str, Any]:
    """Convert a ``Command`` to a dict the data converter can serialize.

    ``Send`` objects in ``goto`` become ``{"node", "arg", "timeout"}`` dicts.
    """

    def send_to_dict(item: Any) -> Any:
        if isinstance(item, Send):
            return {"node": item.node, "arg": item.arg, "timeout": item.timeout}
        return item

    goto = command.goto
    if isinstance(goto, Sequence) and not isinstance(goto, str):
        goto = [send_to_dict(item) for item in goto]
    else:
        goto = send_to_dict(goto)
    return {
        "graph": command.graph,
        "update": command.update,
        "resume": command.resume,
        "goto": goto,
    }


def _command_from_payload(payload: dict[str, Any]) -> Command:
    """Rebuild a ``Command`` from ``_command_to_payload`` output.

    Node names are the only other values ``goto`` can hold, so a dict in
    ``goto`` is always a serialized ``Send``. Payloads written before ``Send``
    support only hold node names and pass through unchanged.
    """

    def dict_to_send(item: Any) -> Any:
        if isinstance(item, dict):
            timeout = item.get("timeout")
            return Send(
                item["node"],
                item["arg"],
                timeout=TimeoutPolicy(**timeout) if timeout else None,
            )
        return item

    goto = payload["goto"]
    if isinstance(goto, list):
        goto = [dict_to_send(item) for item in goto]
    else:
        goto = dict_to_send(goto)
    return Command(
        graph=payload["graph"],
        update=payload["update"],
        resume=payload["resume"],
        goto=goto,
    )


_COMMAND_TYPE_ID = "langgraph.types.Command"
# Keys in the tree ValueTypes.restore builds from paths: a node's type ID and
# the value sent on its own for it.
_TYPE_ID = object()
_SENT = object()
_MISSING = object()
_UNIONS = (typing.Union, UnionType)  # pyright: ignore[reportDeprecated]


def _subclasses(cls: type) -> Iterator[type]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _subclasses(subclass)


def _type_hints(obj: Any) -> list[Any]:
    try:
        return list(get_type_hints(obj, include_extras=True).values())
    except Exception:
        return []


@cache
def _class_hints(cls: Any) -> dict[str, Any]:
    try:
        return get_type_hints(cls, include_extras=True)
    except Exception:
        return {}


def _unwrap(hint: Any) -> Any:
    while get_origin(hint) in (Annotated, Required, NotRequired):
        hint = get_args(hint)[0]
    return hint


@cache
def _dataclass_fields(cls: Any) -> dict[str, FieldInfo]:
    fields = getattr(cls, "__pydantic_fields__", None)  # Pydantic dataclasses
    if fields is not None:
        return fields
    hints = _class_hints(cls)
    return {
        f.name: FieldInfo.from_annotated_attribute(hints.get(f.name, Any), f.default)
        for f in dataclasses.fields(cls)
    }


def _field_info(owner: Any, name: str | int) -> FieldInfo | None:
    """Return Pydantic's settings for field ``name`` of a model or dataclass."""
    cls: Any = type(owner)
    if isinstance(owner, BaseModel):
        return cls.model_fields.get(str(name))
    if dataclasses.is_dataclass(owner):
        return _dataclass_fields(cls).get(str(name))
    return None


def _field_hint(owner: Any, name: str) -> Any:
    if isinstance(owner, BaseModel):
        field = type(owner).model_fields.get(name)
        return field.annotation if field else Any
    cls: Any = type(owner)
    return _class_hints(cls).get(name, Any)


def _is_instance(value: Any, cls: Any) -> bool:
    try:
        return isinstance(value, cls)
    except TypeError:  # Protocols that aren't runtime checkable, TypedDicts, Any
        return False


def _holds(hint: Any, value: Any) -> bool:
    """Whether the class of ``value`` fits ``hint``, ignoring type arguments."""
    hint = _unwrap(hint)
    return _is_instance(value, dict if is_typeddict(hint) else get_origin(hint) or hint)


def _item_hint(hint: Any, owner: Any, key: str | int) -> Any:
    """Return the declared type of item ``key`` of ``owner``, of type ``hint``.

    A tuple holds the item's type under each union member the serializer may
    write ``owner`` with.
    """
    if isinstance(hint, tuple):
        return tuple(_item_hint(member, owner, key) for member in hint)
    hint = _unwrap(hint)
    origin = get_origin(hint)
    args = [arg for arg in get_args(hint) if arg is not type(None)]
    if origin in _UNIONS:
        # Pydantic writes the whole container with one member its class fits,
        # picked from all of its items, so keep the item's type under each.
        hints = tuple(_item_hint(arg, owner, key) for arg in args if _holds(arg, owner))
        return hints[0] if len(hints) == 1 else hints or Any
    if is_typeddict(hint):
        return _class_hints(hint).get(str(key), Any)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        return args[key] if isinstance(key, int) and key < len(args) else Any
    if isinstance(origin, type) and issubclass(origin, Mapping):
        return args[1] if len(args) == 2 else Any
    if isinstance(origin, type) and issubclass(origin, Iterable) and args:
        return args[0]
    return Any


def _written_as_base(value: Any, hint: Any) -> bool:
    """Whether a serializer following ``hint`` writes ``value`` as a base class.

    Pydantic writes a value of a subclass of the declared class with the
    declared class's fields only. For a tuple of types from ``_item_hint``,
    it may write ``value`` following any of them.
    """
    if isinstance(hint, tuple):
        return any(_written_as_base(value, member) for member in hint)
    hint = _unwrap(hint)
    members = get_args(hint) if get_origin(hint) in _UNIONS else (hint,)
    classes = [m for m in map(_unwrap, members) if isinstance(m, type) and m is not Any]
    return type(value) not in classes and any(_is_instance(value, c) for c in classes)


def _excluded(field: FieldInfo | None, value: Any) -> bool:
    """Whether Pydantic leaves a field holding ``value`` out of its owner's JSON."""
    if field is None:
        return False
    exclude_if = getattr(field, "exclude_if", None)  # Newer Pydantic versions
    return bool(field.exclude) or (exclude_if is not None and bool(exclude_if(value)))


def _has_serializer(hint: Any) -> bool:
    """Whether ``hint``, or a tuple of hints and metadata, sets a custom serializer."""
    if isinstance(hint, (PlainSerializer, WrapSerializer)):
        return True
    if isinstance(hint, tuple):
        return any(map(_has_serializer, hint))
    if get_origin(hint) in (Annotated, Required, NotRequired, *_UNIONS):
        return any(map(_has_serializer, get_args(hint)))  # Includes metadata
    return False


def _custom_serialized(owner: Any, key: str | int, hint: Any) -> bool:
    """Whether Pydantic writes ``owner``'s item ``key`` with a custom serializer."""
    decorators = getattr(type(owner), "__pydantic_decorators__", None)
    if decorators is not None:
        if decorators.model_serializers:
            return True
        for serializer in decorators.field_serializers.values():
            if key in serializer.info.fields or "*" in serializer.info.fields:
                return True
    field = _field_info(owner, key)
    return _has_serializer((hint, *field.metadata) if field else hint)


@cache
def _adapter(cls: Any) -> TypeAdapter[Any]:
    return TypeAdapter(cls)


def _dump(value: Any) -> Any:
    """Return ``value`` the way Pydantic writes its class to JSON, or ``_MISSING``.

    Unlike the payload converter, this applies every serializer Pydantic uses
    for JSON, including those on a dataclass's fields.
    """
    cls: Any = type(value)
    try:
        return _adapter(cls).dump_python(value, mode="json")
    except Exception:
        return _MISSING


def _path_key(key: Any) -> bool:
    """Whether dict key ``key`` can go in a recorded path."""
    return isinstance(key, (str, int)) and not isinstance(key, bool)


def _plain_key(plain: dict[Any, Any], key: str | int) -> Any:
    """Return ``key`` as it appears in ``plain``. JSON writes int keys as strings."""
    return str(key) if key not in plain else key


def _children(node: dict[Any, Any]) -> list[Any]:
    return [key for key in node if key is not _TYPE_ID and key is not _SENT]


def _get(owner: Any, key: str | int) -> Any:
    try:
        if isinstance(owner, (dict, list, tuple)):
            return owner[key]  # type: ignore[index]
        return getattr(owner, str(key))
    except (KeyError, IndexError, AttributeError, TypeError):
        return _MISSING


def _put(owner: Any, key: str | int, value: Any) -> Any:
    if isinstance(owner, tuple):
        index = int(key)
        return (*owner[:index], value, *owner[index + 1 :])
    if isinstance(owner, (dict, list)):
        owner[key] = value  # type: ignore[index]
    elif (
        isinstance(owner, BaseModel) and owner.model_extra and key in owner.model_extra
    ):
        owner.model_extra[str(key)] = value
    else:
        object.__setattr__(owner, str(key), value)
    return owner


def _plain_item(plain: Any, owner: Any, key: str | int) -> Any:
    """Return the plain JSON the converter wrote for ``owner``'s item ``key``."""
    if isinstance(owner, RootModel) and key == "root":
        return plain  # A RootModel is written as its root value.
    if isinstance(plain, list):
        return plain[key] if isinstance(key, int) and key < len(plain) else _MISSING
    if not isinstance(plain, dict):
        return _MISSING
    field = _field_info(owner, key)
    aliases = (field.serialization_alias, field.alias) if field else ()
    for name in (_plain_key(plain, key), *aliases):
        if name in plain:
            return plain[name]
    return _MISSING


class ValueTypes:
    """Keep the types of node and task values across the Activity boundary.

    Activity inputs and outputs hold these values as ``Any``, so the payload
    converter decodes them as plain JSON. The sender records where values of
    known classes sit, by class ID, and the receiver rebuilds them with the
    payload converter, as for an Activity parameter of that class. Values are
    encoded as before, so older payloads and readers are unaffected. Values
    inside a rebuilt value are recorded too and put back when the converter
    rebuilds them with another class, such as an ``AIMessage`` in a
    ``list[BaseMessage]`` field. Pydantic writes such a value with the declared
    class's fields only, so it is also sent on its own.

    Known classes are ``bytes``, LangChain messages, and the Pydantic models
    and dataclasses reachable from the given schemas and function signatures,
    with their subclasses. Only these are rebuilt.
    """

    def __init__(
        self, schemas: Iterable[Any] = (), functions: Iterable[Any] = ()
    ) -> None:
        """Collect the classes reachable from ``schemas`` and ``functions``."""
        classes: dict[str, type] = {}
        conflicts: set[str] = set()
        # Keyed by id() because some hints are unhashable; keeps them alive.
        seen: dict[int, Any] = {}
        stack = [bytes, *_subclasses(BaseMessage), *schemas]
        for function in functions:
            stack.extend(_type_hints(function))
        while stack:
            hint = stack.pop()
            if id(hint) in seen:
                continue
            seen[id(hint)] = hint
            origin = get_origin(hint)
            if origin is Literal:
                continue
            if origin is not None:
                args = get_args(hint)
                stack.extend(args[:1] if origin is Annotated else args)
                continue
            if not isinstance(hint, type):
                continue
            if issubclass(hint, BaseModel):
                stack.extend(field.annotation for field in hint.model_fields.values())
                if hint is not BaseModel:
                    stack.extend(_subclasses(hint))
            elif dataclasses.is_dataclass(hint):
                stack.extend(_type_hints(hint))
                stack.extend(_subclasses(hint))
            elif hint is not bytes:
                if is_typeddict(hint):
                    stack.extend(_type_hints(hint))
                continue
            type_id = context_schema_id(hint)
            if classes.setdefault(type_id, hint) is not hint:
                conflicts.add(type_id)
        for type_id in conflicts:
            del classes[type_id]
        self._classes = classes
        self._ids = {cls: type_id for type_id, cls in classes.items()}

    def _type_id(self, cls: type) -> str | None:
        type_id = self._ids.get(cls)
        # Workflow code can build values from the sandbox's copy of a class.
        if type_id is None and (
            issubclass(cls, BaseModel) or dataclasses.is_dataclass(cls)
        ):
            candidate = context_schema_id(cls)
            if candidate in self._classes:
                type_id = candidate
        return type_id

    def encode(self, value: Any, path: list[str | int], value_types: list[Any]) -> Any:
        """Record known values in ``value`` and put its Commands in payload form."""
        if isinstance(value, Command):
            payload = self.encode(_command_to_payload(value), path, value_types)
            value_types.append([path, _COMMAND_TYPE_ID])
            return payload
        type_id = self._type_id(type(value))
        if type_id is not None:
            value_types.append([path, type_id])
            self._record(value, path, value_types, Any)
            return value
        if isinstance(value, dict):
            items = {
                key: self.encode(item, [*path, key], value_types)
                if _path_key(key)
                else item
                for key, item in value.items()
            }
            changed = any(items[key] is not value[key] for key in value)
            return items if changed else value
        if isinstance(value, (list, tuple)):
            elements = [
                self.encode(item, [*path, index], value_types)
                for index, item in enumerate(value)
            ]
            changed = any(new is not old for new, old in zip(elements, value))
            return elements if changed else value
        return value

    def _record(
        self, value: Any, path: list[str | int], value_types: list[Any], hint: Any
    ) -> None:
        """Record known values inside ``value``, which the converter writes whole."""
        items: list[tuple[str | int, Any, Any]]
        if isinstance(value, BaseModel):
            items = [
                (name, getattr(value, name), _field_hint(value, name))
                for name, field in type(value).model_fields.items()
                if not _excluded(field, getattr(value, name))
            ]
            items += [
                (key, item, Any) for key, item in (value.model_extra or {}).items()
            ]
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            items = [
                (field.name, getattr(value, field.name), _field_hint(value, field.name))
                for field in dataclasses.fields(value)
                if not _excluded(
                    _field_info(value, field.name), getattr(value, field.name)
                )
            ]
        elif isinstance(value, dict):
            items = [
                (key, item, _item_hint(hint, value, key))
                for key, item in value.items()
                if _path_key(key)
            ]
        elif isinstance(value, (list, tuple)):
            items = [
                (index, item, _item_hint(hint, value, index))
                for index, item in enumerate(value)
            ]
        else:
            return
        for key, item, item_hint in items:
            # A custom serializer may leave data out, so record nothing in it.
            if _custom_serialized(value, key, item_hint):
                continue
            item_path = [*path, key]
            type_id = self._type_id(type(item))
            if type_id is not None:
                entry: list[Any] = [item_path, type_id]
                whole = _dump(item) if _written_as_base(item, item_hint) else _MISSING
                if whole is not _MISSING:
                    entry.append(whole)
                value_types.append(entry)
            self._record(item, item_path, value_types, item_hint)

    def restore(
        self, value: Any, value_types: list[Any], converter: PayloadConverter
    ) -> Any:
        """Rebuild the values ``encode`` recorded, leaving ``value`` unchanged."""
        tree: dict[Any, Any] = {}
        for path, type_id, *sent in value_types:
            node = tree
            for key in path:
                node = node.setdefault(key, {})
            node[_TYPE_ID] = type_id
            if sent:
                node[_SENT] = sent[0]

        def rebuild(value: Any, node: dict[Any, Any]) -> Any:
            """Rebuild a value from its plain JSON."""
            type_id = node.get(_TYPE_ID)
            if type_id is not None and type_id != _COMMAND_TYPE_ID:
                rebuilt = self._rebuild(
                    type_id, self._init_values(value, node), converter
                )
                return value if rebuilt is _MISSING else fix(rebuilt, value, node)
            children = _children(node)
            if children:
                value = dict(value) if isinstance(value, dict) else list(value)
                for key in children:
                    found = _plain_key(value, key) if isinstance(value, dict) else key
                    value[found] = rebuild(value[found], node[key])
            return value if type_id is None else _command_from_payload(value)

        def fix(current: Any, plain: Any, node: dict[Any, Any]) -> Any:
            """Put back values the converter rebuilt with another class."""
            plain = node.get(_SENT, plain)
            type_id = node.get(_TYPE_ID)
            if type_id is not None and type(current) is not self._classes.get(type_id):
                return rebuild(plain, node)
            if dataclasses.is_dataclass(current) and isinstance(plain, dict):
                # Restore saved init=False fields, as for run context.
                for field in dataclasses.fields(current):
                    if not field.init and field.name in plain:
                        object.__setattr__(current, field.name, plain[field.name])
            for key in _children(node):
                item = _get(current, key)
                item_plain = _plain_item(plain, current, key)
                if item is _MISSING or item_plain is _MISSING:
                    continue
                fixed = fix(item, item_plain, node[key])
                if fixed is not item:
                    current = _put(current, key, fixed)
            return current

        return rebuild(value, tree)

    def _init_values(self, plain: Any, node: dict[Any, Any]) -> Any:
        """Copy ``plain`` without the init=False fields of the dataclasses in ``node``.

        The default converter passes every field it finds to ``__init__``.
        """
        cls = self._classes.get(node.get(_TYPE_ID, ""))
        non_init = (
            {f.name for f in dataclasses.fields(cls) if not f.init}
            if dataclasses.is_dataclass(cls)
            else set()
        )
        if isinstance(plain, dict):
            # JSON writes int keys as strings.
            children = {str(key): node[key] for key in _children(node)}
            return {
                key: self._init_values(item, children[key]) if key in children else item
                for key, item in plain.items()
                if key not in non_init
            }
        if isinstance(plain, list):
            return [
                self._init_values(item, node[index]) if index in node else item
                for index, item in enumerate(plain)
            ]
        return plain

    def _rebuild(self, type_id: str, value: Any, converter: PayloadConverter) -> Any:
        if type_id == _COMMAND_TYPE_ID:
            return _command_from_payload(value)
        cls = self._classes.get(type_id)
        if cls is None:
            raise ValueError(
                f"Unknown value type ID: {type_id!r}. Workers must register "
                "the same graphs and tasks."
            )
        try:
            return converter.from_payload(converter.to_payload(value), cls)
        except Exception:
            # A value the converter can't rebuild, such as a dataclass with a
            # Protocol field, keeps its plain JSON form, as before.
            return _MISSING


def wrap_activity(
    func: Callable,
    *,
    context_schema: type[Any] | None = None,
    context_schemas: dict[str, type[Any]] | None = None,
    value_types: ValueTypes | None = None,
    streaming_topic: str | None = None,
    streaming_batch_interval: timedelta = timedelta(milliseconds=100),
) -> Callable[[ActivityInput], Awaitable[ActivityOutput]]:
    """Wrap a function as a Temporal activity that handles LangGraph config and interrupts."""
    accepts_runtime = "runtime" in signature(func).parameters
    known = value_types if value_types is not None else ValueTypes()

    async def wrapper(input: ActivityInput) -> ActivityOutput:
        async def run(stream_writer: Callable[[Any], None] | None) -> ActivityOutput:
            # Sync funcs run on a thread (so the loop keeps flushing the
            # stream client mid-execution); marshal writer calls back to
            # the loop thread because the client's flush event is an
            # asyncio.Event and isn't safe to set off-thread.
            effective_writer = stream_writer
            if not iscoroutinefunction(func) and stream_writer is not None:
                loop = asyncio.get_running_loop()
                inner_writer = stream_writer

                def thread_safe_writer(value: Any) -> None:
                    loop.call_soon_threadsafe(inner_writer, value)

                effective_writer = thread_safe_writer

            schema = context_schema
            schema_id = input.langgraph_config.get("context_schema_id")
            if schema_id is not None:
                schema = (context_schemas or {}).get(schema_id)
                if schema is None:
                    raise ValueError(f"Unknown context schema ID: {schema_id!r}")
            runtime = set_langgraph_config(
                input.langgraph_config,
                context_schema=schema,
                stream_writer=effective_writer,
            )
            values = {"args": list(input.args), "kwargs": input.kwargs}
            if input.value_types:
                values = known.restore(
                    values, input.value_types, activity.payload_converter()
                )
            args = values["args"]
            kwargs = dict(values["kwargs"])
            if accepts_runtime:
                kwargs["runtime"] = runtime

            try:
                if iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)
                # A top-level Command keeps its own field for older Workflows.
                output = (
                    {"langgraph_command": _command_to_payload(result)}
                    if isinstance(result, Command)
                    else {"result": result}
                )
                result_types: list[Any] = []
                output = known.encode(output, [], result_types)
                return ActivityOutput(**output, value_types=result_types or None)
            except GraphInterrupt as e:
                return ActivityOutput(langgraph_interrupts=e.args[0])

        if streaming_topic is None:
            return await run(stream_writer=None)
        async with WorkflowStreamClient.from_within_activity(
            batch_interval=streaming_batch_interval,
        ) as client:
            topic = client.topic(streaming_topic)
            return await run(stream_writer=topic.publish)

    return wrapper


def wrap_execute_activity(
    afunc: Callable[[ActivityInput], Awaitable[ActivityOutput]],
    task_id: str = "",
    summary_fn: Callable[[tuple[Any, ...], dict[str, Any]], str | None] | None = None,
    context_schemas: dict[str, type[Any]] | None = None,
    value_types: ValueTypes | None = None,
    **execute_activity_kwargs: Any,
) -> Callable[..., Any]:
    """Wrap an activity function to be called via workflow.execute_activity with caching."""
    known = value_types if value_types is not None else ValueTypes()

    def output_value(result: Any, command: Any, result_types: list[Any] | None) -> Any:
        output = {"result": result, "langgraph_command": command}
        if result_types:
            output = known.restore(output, result_types, workflow.payload_converter())
        if output["langgraph_command"] is not None:
            return _command_from_payload(output["langgraph_command"])
        return output["result"]

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # LangGraph may inject a RunnableConfig as the 'config' kwarg. Strip it
        # down to a serializable subset so it can cross the activity boundary;
        # callbacks, stores, etc. aren't serializable.
        if "config" in kwargs:
            kwargs["config"] = strip_runnable_config(kwargs["config"])

        # LangGraph may inject a Runtime as the 'runtime' kwarg. It's
        # reconstructed on the activity side from the serialized langgraph
        # config, so drop the live Runtime from the kwargs that cross the
        # activity boundary (it holds non-serializable stream_writer, store).
        runtime = kwargs.pop("runtime", None)
        run_id = workflow.info().run_id
        if (
            getattr(runtime, "store", None) is not None
            and run_id not in _warned_store_runs
        ):
            _warned_store_runs.add(run_id)
            workflow.logger.warning(
                "LangGraph Store passed via compile(store=...) / @entrypoint(store=...) "
                "is not accessible inside activity-wrapped nodes and tasks: the Store "
                "object isn't serializable across the activity boundary, and activities "
                "may run on a different worker than the workflow. Use a backend-backed "
                "store (Postgres/Redis) configured on each worker if you need shared "
                "memory, or use workflow state for per-run memory."
            )

        langgraph_config = get_langgraph_config()
        if runtime is not None:
            langgraph_config["context"] = runtime.context
        context_type = type(langgraph_config.get("context"))
        if is_context_schema(context_type):
            schema_id = context_schema_id(context_type)
            if schema_id not in (context_schemas or {}):
                raise ValueError(
                    f"Context type {schema_id!r} is not registered. "
                    "Declare it as a graph or entrypoint context_schema."
                )
            langgraph_config["context_schema_id"] = schema_id

        # Check task result cache (for continue-as-new deduplication).
        key = (
            cache_key(task_id, args, kwargs, langgraph_config.get("context"))
            if task_id
            else ""
        )
        # The cache travels through workflow.continue_as_new(), so a Command is
        # stored in its payload form under a separate "<key>:command" entry and
        # rebuilt on lookup. cache_key() returns 32 hex characters, so this
        # namespace can't collide with a plain result key, and a result dict
        # is never mistaken for a Command. Value types go under "<key>:types",
        # so older Workflows still find the plain entry.
        if task_id:
            found_command, command = cache_lookup(f"{key}:command")
            found_result, result = cache_lookup(key)
            if found_command or found_result:
                _, result_types = cache_lookup(f"{key}:types")
                return output_value(
                    None if found_command else result, command, result_types
                )

        arg_types: list[Any] = []
        values = known.encode({"args": args, "kwargs": kwargs}, [], arg_types)
        input = ActivityInput(
            args=tuple(values["args"]),
            kwargs=values["kwargs"],
            langgraph_config=langgraph_config,
            value_types=arg_types or None,
        )
        # Compute a dynamic activity summary (if configured) on the schedule
        # path only; a cache hit above returns before reaching here, so no
        # activity is scheduled and no summary is needed.
        call_kwargs = dict(execute_activity_kwargs)
        if summary_fn is not None:
            summary = summary_fn(args, kwargs)
            if summary:
                call_kwargs["summary"] = summary
        output = await workflow.execute_activity(afunc, input, **call_kwargs)
        if output.langgraph_interrupts is not None:
            raise GraphInterrupt(output.langgraph_interrupts)

        # Store in cache for future continue-as-new cycles.
        if task_id:
            if output.langgraph_command is not None:
                cache_put(f"{key}:command", output.langgraph_command)
            else:
                cache_put(key, output.result)
            if output.value_types:
                cache_put(f"{key}:types", output.value_types)

        return output_value(output.result, output.langgraph_command, output.value_types)

    return wrapper
