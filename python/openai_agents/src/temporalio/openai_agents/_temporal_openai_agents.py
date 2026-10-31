"""Initialize Temporal OpenAI Agents overrides."""

import dataclasses
import functools
import itertools
import json
import threading
import types
import typing
import warnings
from collections.abc import (
    AsyncGenerator,
    Callable,
    Collection,
    Generator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from datetime import timedelta
from weakref import WeakKeyDictionary

import openai
import pydantic
import pydantic_core
from agents import ModelProvider, Trace, set_trace_provider
from agents.mcp import MCPServer
from agents.run import get_default_agent_runner, set_default_agent_runner
from agents.tracing import get_trace_provider
from agents.tracing.provider import DefaultTraceProvider

# construct_type is OpenAI's lenient (non-validating) model builder, the same
# one the SDK uses to parse live API responses. It is in a private module but
# has no public alias.
from openai._models import construct_type
from pydantic.fields import FieldInfo
from typing_extensions import NotRequired, Required, is_typeddict

import temporalio.api.common.v1
from temporalio.contrib.opentelemetry._tracer_provider import ReplaySafeTracerProvider
from temporalio.contrib.pydantic import (
    PydanticJSONPlainPayloadConverter,
    ToJsonOptions,
)
from temporalio.converter import (
    CompositePayloadConverter,
    DataConverter,
    DefaultPayloadConverter,
    JSONPlainPayloadConverter,
)
from temporalio.openai_agents._errors import AgentsWorkflowError
from temporalio.openai_agents._invoke_model_activity import ModelActivity
from temporalio.openai_agents._model_parameters import ModelActivityParameters
from temporalio.openai_agents._openai_runner import (
    TemporalOpenAIRunner,
)
from temporalio.openai_agents._temporal_trace_provider import (
    TemporalTraceProvider,
)
from temporalio.openai_agents._temporal_worker_env_ref import (
    AllowAllWorkerEnvVars,
    _snapshot_resolvable_env_vars,
)
from temporalio.openai_agents._trace_interceptor import (
    OpenAIAgentsContextPropagationInterceptor,
)
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

if typing.TYPE_CHECKING:
    from openinference.instrumentation.openai_agents._processor import (
        OpenInferenceTracingProcessor,
    )

    from temporalio.openai_agents import (
        SandboxClientProvider,
        StatefulMCPServerProvider,
        StatelessMCPServerProvider,
    )


_otel_trace_start_patch_lock = threading.RLock()
_otel_trace_start_patch_ref_count = 0
_otel_trace_start_original: Callable[..., typing.Any] | None = None
_otel_trace_end_original: (
    Callable[["OpenInferenceTracingProcessor", Trace], None] | None
) = None
_otel_trace_start_instrumentor: typing.Any | None = None


def _install_otel_instrumentation(tracer_provider: typing.Any) -> None:
    """Configure OpenInference while at least one tracing context is active."""
    global _otel_trace_start_instrumentor
    global _otel_trace_start_original
    global _otel_trace_end_original
    global _otel_trace_start_patch_ref_count

    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
    from openinference.instrumentation.openai_agents._processor import (
        OpenInferenceTracingProcessor,
    )
    from opentelemetry.context import Context, attach, detach
    from opentelemetry.trace import set_span_in_context

    with _otel_trace_start_patch_lock:
        if _otel_trace_start_patch_ref_count == 0:
            original_on_trace_start = OpenInferenceTracingProcessor.on_trace_start
            original_on_trace_end = OpenInferenceTracingProcessor.on_trace_end
            _otel_trace_start_original = original_on_trace_start
            _otel_trace_end_original = original_on_trace_end
            trace_tokens: WeakKeyDictionary[Trace, Token[Context]] = WeakKeyDictionary()

            def on_trace_start(
                self: OpenInferenceTracingProcessor, trace: Trace
            ) -> None:
                original_on_trace_start(self=self, trace=trace)
                # Older OpenTelemetry versions annotate context tokens as object.
                trace_tokens[trace] = typing.cast(
                    Token[Context],  # pyright: ignore[reportUnnecessaryCast]
                    attach(set_span_in_context(self._root_spans[trace.trace_id])),
                )

            def on_trace_end(self: OpenInferenceTracingProcessor, trace: Trace) -> None:
                try:
                    original_on_trace_end(self=self, trace=trace)
                finally:
                    if token := trace_tokens.pop(trace, None):
                        detach(token)

            setattr(OpenInferenceTracingProcessor, "on_trace_start", on_trace_start)
            setattr(OpenInferenceTracingProcessor, "on_trace_end", on_trace_end)
            try:
                _otel_trace_start_instrumentor = OpenAIAgentsInstrumentor()
                _otel_trace_start_instrumentor.instrument(
                    tracer_provider=tracer_provider
                )
            except BaseException:
                setattr(
                    OpenInferenceTracingProcessor,
                    "on_trace_start",
                    _otel_trace_start_original,
                )
                setattr(
                    OpenInferenceTracingProcessor,
                    "on_trace_end",
                    _otel_trace_end_original,
                )
                _otel_trace_start_original = None
                _otel_trace_end_original = None
                _otel_trace_start_instrumentor = None
                raise
        _otel_trace_start_patch_ref_count += 1


def _uninstall_otel_instrumentation() -> None:
    """Tear down OpenInference after the final tracing context exits."""
    global _otel_trace_start_instrumentor
    global _otel_trace_start_original
    global _otel_trace_end_original
    global _otel_trace_start_patch_ref_count

    from openinference.instrumentation.openai_agents._processor import (
        OpenInferenceTracingProcessor,
    )

    with _otel_trace_start_patch_lock:
        if _otel_trace_start_patch_ref_count == 0:
            raise RuntimeError("OpenInference instrumentation was not acquired")
        _otel_trace_start_patch_ref_count -= 1
        if _otel_trace_start_patch_ref_count == 0:
            try:
                if _otel_trace_start_instrumentor is not None:
                    _otel_trace_start_instrumentor.uninstrument()
            finally:
                if _otel_trace_start_original is not None:
                    setattr(
                        OpenInferenceTracingProcessor,
                        "on_trace_start",
                        _otel_trace_start_original,
                    )
                if _otel_trace_end_original is not None:
                    setattr(
                        OpenInferenceTracingProcessor,
                        "on_trace_end",
                        _otel_trace_end_original,
                    )
                _otel_trace_start_original = None
                _otel_trace_end_original = None
                _otel_trace_start_instrumentor = None


@contextmanager
def _set_open_ai_agent_temporal_overrides(
    model_params: ModelActivityParameters,
    start_spans_in_replay: bool = False,
):
    previous_runner = get_default_agent_runner()
    previous_trace_provider = get_trace_provider()
    provider = TemporalTraceProvider(
        start_spans_in_replay=start_spans_in_replay,
    )

    try:
        set_default_agent_runner(TemporalOpenAIRunner(model_params))
        set_trace_provider(provider)
        yield provider
    finally:
        set_default_agent_runner(previous_runner)
        set_trace_provider(previous_trace_provider or DefaultTraceProvider())


def _lenient_construct(type_: typing.Any, value: typing.Any) -> typing.Any:
    """Build ``value`` into ``type_`` without enforcing required fields.

    OpenAI's ``construct_type`` handles its own response models, but not the
    ``TypedDict``, dataclass and caller Pydantic model layers that wrap them (e.g.
    ``ActivityModelInput``, ``ModelResponse`` or a Workflow result holding
    ``raw_responses``). Walk those containers here, then delegate their leaves to
    ``construct_type``. ``include_extras`` preserves the ``Annotated``
    discriminators the unions rely on.
    """
    origin = typing.get_origin(type_)
    if origin in (Required, NotRequired):
        return _lenient_construct(typing.get_args(type_)[0], value)
    if origin is typing.Annotated and (
        _plugin_builds(typing.get_args(type_)[0]) or _changes_input(type_)
    ):
        # Validate with the metadata, so a caller's validators on it still run, also on an
        # OpenAI model (see ``_may_hold``). Other Annotated types, such as OpenAI's
        # discriminated unions, keep their metadata.
        try:
            return _lenient_validate(type_, value)
        except (pydantic.ValidationError, pydantic_core.SchemaError):
            return _lenient_construct(typing.get_args(type_)[0], value)
    if origin in (typing.Union, types.UnionType):  # pyright: ignore[reportDeprecated]
        # Optional fields and the ``str | list[...]`` input field: build the only
        # variant that fits.
        if value is None:
            return None
        variants = [
            arg
            for arg in typing.get_args(type_)
            if arg is not type(None) and _may_hold(arg, value)
        ]
        if len(variants) == 1:
            return _lenient_construct(variants[0], value)
        if _plugin_builds(type_) or any(map(_changes_input, variants)):
            # Variants told apart by their tags or validation (see ``_lenient_union``).
            try:
                return _lenient_validate(type_, value)
            except (pydantic.ValidationError, pydantic_core.SchemaError):
                pass
        try:
            return construct_type(type_=type_, value=value)
        except TypeError:
            # construct_type caches discriminated unions by weak reference, which
            # ``X | Y`` unions do not support. Keep the decoded value.
            return value
    if origin is not None and (
        is_typeddict(origin) or dataclasses.is_dataclass(origin)
    ):
        # A generic specialization such as ``Box[ModelResponse]``, whose field types only
        # its schema resolves: validate it as declared, from any value its validators take.
        try:
            return _lenient_validate(type_, value)
        except (pydantic.ValidationError, pydantic_core.SchemaError):
            return construct_type(type_=type_, value=value)
    if is_typeddict(type_) and isinstance(value, dict):
        hints = typing.get_type_hints(type_, include_extras=True)
        return {
            key: _lenient_construct(hints.get(key, object), item)
            for key, item in value.items()
        }
    if origin is list and isinstance(value, list):
        (item_type,) = typing.get_args(type_)
        return [_lenient_construct(item_type, item) for item in value]
    if origin is dict and isinstance(value, dict):
        _, item_type = typing.get_args(type_) or (typing.Any, typing.Any)
        return {key: _lenient_construct(item_type, item) for key, item in value.items()}
    if isinstance(type_, type) and dataclasses.is_dataclass(type_):
        # Any value: the dataclass's validators may turn it into an object first.
        try:
            return _lenient_validate(type_, value)
        except (pydantic.ValidationError, pydantic_core.SchemaError):
            pass
        if isinstance(value, dict):
            hints = typing.get_type_hints(type_, include_extras=True)
            return type_(
                **{
                    field.name: _lenient_construct(
                        hints.get(field.name, object), value[field.name]
                    )
                    for field in dataclasses.fields(type_)
                    if field.name in value
                }
            )
    if (
        isinstance(type_, type)
        and issubclass(type_, pydantic.BaseModel)
        and not issubclass(type_, openai.BaseModel)
    ):
        try:
            return _lenient_validate(type_, value)
        except (pydantic.ValidationError, pydantic_core.SchemaError):
            return construct_type(type_=type_, value=value)
    return construct_type(type_=type_, value=value)


def _lenient_validate(type_: typing.Any, value: typing.Any) -> typing.Any:
    """Validate a caller's model or dataclass whose OpenAI parts fail strict validation.

    The value is validated as JSON against the type's own schema, so its aliases, extra fields,
    root type, validators, ``Annotated`` metadata, strictness and JSON settings apply as they
    do when the value is valid. Only the OpenAI models in it fall back to ``construct_type``
    (see ``_lenient_schema``). Raises ``ValidationError`` when the value still does not
    validate, and ``SchemaError`` if Pydantic cannot build the lenient copy of the type's schema.
    """
    try:
        hash(type_)
    except TypeError:
        # Annotated metadata that cannot be a cache key.
        validator = _lenient_validator.__wrapped__(type_)
    else:
        validator = _lenient_validator(type_)
    return validator.validate_json(json.dumps(value))


# A validator holds its type through its schema and functions, so a weak cache would never
# release a type, such as a Workflow sandbox's copy of a caller's class. Keep only the most
# recently used ones.
@functools.lru_cache(maxsize=32)
def _lenient_validator(type_: typing.Any) -> pydantic_core.SchemaValidator:
    """A validator for the lenient copy of ``type_``'s Pydantic core schema."""
    schema = pydantic.TypeAdapter(type_).core_schema
    return pydantic_core.SchemaValidator(
        _lenient_schema(schema, _schema_refs(schema, {}))
    )


# The choice a lenient union validates next: its original strict union, or the index of a
# lenient variant.
_union_choice: ContextVar[str | int] = ContextVar("_union_choice", default="strict")

# Whether the next OpenAI model to validate is a lenient union variant reached through a
# validator that can change the value, so it must check the value's kind and tags itself.
_variant_check: ContextVar[bool] = ContextVar("_variant_check", default=False)

_LENIENT_REF = ":lenient"


def _lenient_schema(schema: typing.Any, refs: dict[str, typing.Any]) -> typing.Any:
    """A copy of a Pydantic core schema in which OpenAI models fall back to ``construct_type``.

    Each OpenAI model first validates strictly. A union first validates as the original strict
    union, and only then as the one variant whose shape and ``Literal`` tags fit the value and
    that validates, so the fallback never picks among several variants. Definitions are kept as they were for the
    strict unions and copied, under a new ref, for the lenient ones.
    """
    if isinstance(schema, list | tuple):
        return type(schema)(_lenient_schema(item, refs) for item in schema)
    if not isinstance(schema, dict):
        return schema
    kind = schema.get("type")
    if not isinstance(kind, str):
        # A mapping of names to schemas, such as a model's fields.
        return {key: _lenient_schema(item, refs) for key, item in schema.items()}
    if kind == "definitions":
        return {
            **schema,
            "schema": _lenient_schema(schema["schema"], refs),
            "definitions": [
                *schema["definitions"],
                *(_lenient_schema(item, refs) for item in schema["definitions"]),
            ],
        }
    if kind == "definition-ref":
        return {**schema, "schema_ref": schema["schema_ref"] + _LENIENT_REF}
    if kind == "union":
        return _lenient_union(schema, refs)
    copy: typing.Any = {
        key: item if key in _SCHEMA_DATA_KEYS else _lenient_schema(item, refs)
        for key, item in schema.items()
    }
    ref = copy.pop("ref", None)
    if ref is not None:
        ref += _LENIENT_REF
    cls: typing.Any = schema.get("cls")
    if kind == "typed-dict":
        # The class only selects a prebuilt validator, which would skip the lenient copy.
        copy.pop("cls", None)
    if kind == "dataclass" and isinstance(cls, type):
        return _lenient_dataclass(cls, copy, ref)
    if kind != "model" or not isinstance(cls, type):
        return {**copy, "ref": ref} if ref is not None else copy
    if issubclass(cls, openai.BaseModel):

        def validate(value: typing.Any, handler: typing.Any) -> typing.Any:
            check = _variant_check.get()
            # The models inside it are not the variant. The check holds again once this call
            # returns, for a validator that calls its handler again.
            token = _variant_check.set(False)
            try:
                return handler(value)
            except pydantic.ValidationError:
                if check and not _schema_may_hold(schema, value, refs):
                    raise
                return construct_type(type_=cls, value=value)
            finally:
                _variant_check.reset(token)

        return pydantic_core.core_schema.no_info_wrap_validator_function(
            validate, copy, ref=ref
        )
    if schema.get("custom_init"):
        # Pydantic would call the model's own __init__; keep its strict validator. Its refs are
        # renamed like the rest of the copy, so they stay unique beside the original definitions.
        return _lenient_refs(schema)
    # Pydantic-core reuses a complete model's own validator for a model schema, which would
    # skip the lenient copy of its fields. Validate them into a stand-in with the same config,
    # then move the state onto the model as pydantic-core does.
    post_init = copy.pop("post_init", None)

    model: typing.Any = cls

    def build(state: _ModelState) -> typing.Any:
        instance = model.__new__(model)
        for name in _ModelState.__slots__:
            try:
                object.__setattr__(instance, name, object.__getattribute__(state, name))
            except AttributeError:
                pass
        if post_init:
            getattr(instance, post_init)(None)
        return instance

    def state(value: typing.Any) -> _ModelState:
        copied = _ModelState()
        for name in _ModelState.__slots__:
            try:
                object.__setattr__(copied, name, object.__getattribute__(value, name))
            except AttributeError:
                pass
        return copied

    stand_in = {**copy, "cls": _ModelState, "revalidate_instances": "always"}
    return _keeping_instances(
        model,
        _revalidation(schema),
        state,
        pydantic_core.core_schema.no_info_after_validator_function(
            build, typing.cast(typing.Any, stand_in)
        ),
        ref,
    )


# Numbers the stand-ins' refs, which must be unique in a schema.
_stand_ins = itertools.count()


def _keeping_instances(
    cls: type,
    revalidate: str,
    state: Callable[[typing.Any], typing.Any],
    stand_in: typing.Any,
    ref: str | None,
) -> typing.Any:
    """A stand-in's schema that also takes instances of ``cls``, such as validators return.

    pydantic-core keeps such an instance or revalidates it by its ``revalidate`` setting. The
    stand-in, which always revalidates, revalidates a copy of its ``state``. A JSON value goes to
    the stand-in as it is, so JSON validation still applies.
    """
    cs = pydantic_core.core_schema
    stand_in_ref = f"{ref or cls.__qualname__}:{next(_stand_ins)}:stand-in"

    def accept(value: typing.Any, handler: typing.Any) -> typing.Any:
        return handler(state(value)) if _revalidates(revalidate, value, cls) else value

    # pydantic-core takes refs only on top-level schemas and definitions, so the stand-in is
    # defined here, and the ref, which a definitions list needs on its items, goes on this schema.
    schema = cs.definitions_schema(
        cs.union_schema(
            [
                cs.chain_schema(
                    [
                        cs.is_instance_schema(cls),
                        cs.no_info_wrap_validator_function(
                            accept, cs.definition_reference_schema(stand_in_ref)
                        ),
                    ]
                ),
                cs.definition_reference_schema(stand_in_ref),
            ],
            mode="left_to_right",
        ),
        [typing.cast(typing.Any, {**stand_in, "ref": stand_in_ref})],
    )
    return {**schema, "ref": ref} if ref is not None else schema


def _revalidation(schema: dict[str, typing.Any]) -> str:
    """A model or dataclass schema's ``revalidate_instances`` setting, as pydantic-core reads it."""
    return schema.get("revalidate_instances") or (schema.get("config") or {}).get(
        "revalidate_instances", "never"
    )


def _revalidates(revalidate: str, value: typing.Any, cls: type) -> bool:
    """Whether pydantic-core revalidates an instance of ``cls`` under ``revalidate``."""
    return revalidate == "always" or (
        revalidate == "subclass-instances" and type(value) is not cls
    )


class _ModelState:
    """The attributes pydantic-core sets on a model instance it validates."""

    # pydantic-core sets these slots itself.
    __slots__ = (
        "__dict__",
        "__pydantic_extra__",  # pyright: ignore[reportUninitializedInstanceVariable]
        "__pydantic_fields_set__",  # pyright: ignore[reportUninitializedInstanceVariable]
        "__pydantic_private__",  # pyright: ignore[reportUninitializedInstanceVariable]
    )


def _lenient_dataclass(
    cls: typing.Any, copy: typing.Any, ref: str | None
) -> typing.Any:
    """Validate a dataclass's lenient fields into a stand-in, as ``_lenient_schema`` does models."""
    post_init = copy.pop("post_init", False)
    slots = copy.get("slots", False)

    def build(state: _DataclassState) -> typing.Any:
        instance = cls.__new__(cls)
        # The fields and any extra fields the config allows, as pydantic-core sets them: one by
        # one for slots, else as the whole __dict__, which skips properties of the same name.
        if slots:
            for name, item in state.__dict__.items():
                object.__setattr__(instance, name, item)
        else:
            object.__setattr__(instance, "__dict__", state.__dict__)
        if post_init:
            instance.__post_init__(*state.init_vars)
        return instance

    # pydantic-core revalidates a dataclass instance from the fields its class declares.
    state_cls = type(
        "_DataclassState",
        (_DataclassState,),
        {"__slots__": (), "__dataclass_fields__": cls.__dataclass_fields__},
    )

    def state(value: typing.Any) -> _DataclassState:
        copied = state_cls()
        for field in dataclasses.fields(cls):
            copied.__dict__[field.name] = getattr(value, field.name)
        return copied

    stand_in = {
        **copy,
        "cls": state_cls,
        "post_init": True,
        "revalidate_instances": "always",
    }
    return _keeping_instances(
        cls,
        _revalidation(copy),
        state,
        pydantic_core.core_schema.no_info_after_validator_function(
            build, typing.cast(typing.Any, stand_in)
        ),
        ref,
    )


class _DataclassState:
    """The fields pydantic-core sets on a dataclass instance it validates, and its init vars."""

    __slots__ = ("__dict__", "init_vars")  # pyright: ignore[reportUninitializedInstanceVariable]

    def __post_init__(self, *init_vars: typing.Any) -> None:
        self.init_vars = init_vars


def _schema_refs(
    schema: typing.Any, refs: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    """Every schema in ``schema`` that has a ref, by ref."""
    if isinstance(schema, list | tuple):
        for item in schema:
            _schema_refs(item, refs)
    elif isinstance(schema, dict):
        is_schema = isinstance(schema.get("type"), str)
        if is_schema and isinstance(schema.get("ref"), str):
            refs[schema["ref"]] = schema
        for key, item in schema.items():
            if not is_schema or key not in _SCHEMA_DATA_KEYS:
                _schema_refs(item, refs)
    return refs


def _lenient_refs(schema: typing.Any) -> typing.Any:
    """A copy of ``schema`` whose refs are renamed as in the lenient copy, and nothing else."""
    if isinstance(schema, list | tuple):
        return type(schema)(_lenient_refs(item) for item in schema)
    if not isinstance(schema, dict):
        return schema
    is_schema = isinstance(schema.get("type"), str)
    copy = {
        key: item if is_schema and key in _SCHEMA_DATA_KEYS else _lenient_refs(item)
        for key, item in schema.items()
    }
    for key in ("ref", "schema_ref"):
        if is_schema and isinstance(copy.get(key), str):
            copy[key] += _LENIENT_REF
    return copy


# Values in a core schema that are data or callables, not nested schemas.
_SCHEMA_DATA_KEYS = frozenset(
    {"cls", "config", "default", "expected", "function", "metadata", "serialization"}
)


def _lenient_union(
    schema: dict[str, typing.Any], refs: dict[str, typing.Any]
) -> typing.Any:
    """A union that validates strictly first, then as the only variant that validates leniently."""
    choices = [
        choice[0] if isinstance(choice, tuple) else choice
        for choice in schema["choices"]
    ]
    lenient = [_lenient_schema(choice, refs) for choice in choices]
    strict = {key: item for key, item in schema.items() if key != "ref"}
    # A variant whose value a before, wrap or plain validator can change is judged only after
    # that validator runs: by its own validation, or for an OpenAI model, which builds from any
    # object, by the kind and tag check in its lenient copy (see ``_lenient_schema``).
    checks_itself = []
    for choice in choices:
        transformed, target = _variant_target(choice, refs)
        checks_itself.append(
            transformed
            and isinstance(target, dict)
            and target.get("type") == "model"
            and isinstance(target.get("cls"), type)
            and issubclass(target["cls"], openai.BaseModel)
        )

    def validate(value: typing.Any, handler: typing.Any) -> typing.Any:
        token = _union_choice.set("strict")
        try:
            return handler(value)
        except pydantic.ValidationError as error:
            strict_error = error
        finally:
            _union_choice.reset(token)
        # OpenAI models in a lenient variant build from any object, so only the variants whose
        # kind and tags fit are tried. Their own validators and aliases then decide.
        built: list[typing.Any] = []
        for i, choice in enumerate(choices):
            if len(built) > 1:
                break
            if not _schema_may_hold(choice, value, refs):
                continue
            token = _union_choice.set(i)
            check = _variant_check.set(checks_itself[i])
            try:
                built.append(handler(value))
            except pydantic.ValidationError:
                pass
            finally:
                _variant_check.reset(check)
                _union_choice.reset(token)
        if len(built) != 1:
            raise strict_error
        return built[0]

    return pydantic_core.core_schema.no_info_wrap_validator_function(
        validate,
        pydantic_core.core_schema.tagged_union_schema(
            typing.cast(typing.Any, {"strict": strict, **dict(enumerate(lenient))}),
            discriminator=lambda _: _union_choice.get(),
        ),
        ref=schema["ref"] + _LENIENT_REF if "ref" in schema else None,
    )


def _schema_may_hold(
    schema: dict[str, typing.Any], value: typing.Any, refs: dict[str, typing.Any]
) -> bool:
    """Whether a union variant could hold a JSON ``value``, judged by its kind and tags.

    A variant behind a validator that can change the value always may: the tags it is sent with
    need not be the ones it validates.
    """
    transformed, target = _variant_target(schema, refs)
    if transformed or target is None:
        return True
    schema = target
    kind = schema.get("type")
    if isinstance(value, list):
        return kind in ("list", "tuple", "set", "frozenset", "any")
    if isinstance(value, dict):
        if kind not in ("model", "dataclass", "typed-dict", "dict", "any"):
            return False
        config = schema.get("config") or {}
        return all(
            _tag_fits(name, field, value, config)
            for name, field in _schema_fields(schema)
        )
    return True


# Validator schemas whose function can change the value before the schema inside sees it.
_TRANSFORMS = frozenset({"function-before", "function-wrap", "function-plain"})


def _variant_target(
    schema: typing.Any, refs: dict[str, typing.Any]
) -> tuple[bool, typing.Any]:
    """The schema a union variant's value reaches past its wrappers, ``None`` if unknown.

    Also whether a before, wrap or plain validator on the way can change the value.
    """
    transformed = False
    while True:
        kind = schema.get("type")
        if kind == "definitions":
            schema = schema["schema"]
        elif kind == "definition-ref":
            schema = refs.get(schema["schema_ref"])
            if schema is None:
                return transformed, None
        elif kind in _TRANSFORMS:
            transformed = True
            if "schema" not in schema:
                return transformed, schema
            schema = schema["schema"]
        elif kind in ("nullable", "default", "function-after") or (
            kind == "model" and schema.get("root_model")
        ):
            schema = schema["schema"]
        else:
            # Pydantic puts a model's or dataclass's before validators inside its schema.
            inner = schema.get("schema") if kind in ("model", "dataclass") else None
            if isinstance(inner, dict) and inner.get("type") in _TRANSFORMS:
                transformed = True
            return transformed, schema


def _schema_fields(schema: dict[str, typing.Any]) -> list[tuple[str, typing.Any]]:
    """The fields of a model, dataclass or ``TypedDict`` core schema, by name."""
    if schema.get("type") in ("model", "dataclass"):
        inner = schema.get("schema")
        fields = inner.get("fields") if isinstance(inner, dict) else None
    else:
        fields = schema.get("fields")
    if isinstance(fields, dict):
        return list(fields.items())
    if isinstance(fields, list):
        return [(field["name"], field) for field in fields]
    return []


def _tag_fits(
    name: str,
    field: dict[str, typing.Any],
    value: dict[str, typing.Any],
    config: typing.Any,
) -> bool:
    """Whether a JSON object holds an allowed value for a ``Literal`` field, or may omit it.

    Other fields, and ``Literal`` fields of values that JSON does not keep as they are, always
    fit. The key is looked up as the validator would, by alias and then by name as the type's
    config allows.
    """
    tag = field.get("schema") or {}
    required = field.get("required", True)
    if tag.get("type") == "default":
        tag = tag.get("schema") or {}
        required = False
    expected = tag.get("expected") if tag.get("type") == "literal" else None
    if not expected or not all(
        item is None or isinstance(item, str | int | float) for item in expected
    ):
        return True
    alias = field.get("validation_alias")
    if alias is None:
        keys = [name]
    elif isinstance(alias, str):
        keys = [alias] if config.get("validate_by_alias", True) else []
        if config.get("validate_by_name", False):
            keys.append(name)
    else:
        # Alias paths and choices.
        return True
    for key in keys:
        if key in value:
            return value[key] in expected
    return not required


def _field_type(field: FieldInfo) -> typing.Any:
    """A field's resolved type with the metadata Pydantic moved off it, such as union discriminators."""
    if field.metadata:
        return typing.Annotated[(field.annotation, *field.metadata)]  # pyright: ignore[reportInvalidTypeArguments]
    return field.annotation


def _plugin_builds(type_: typing.Any) -> bool:
    """Whether the fallback builds ``type_`` itself rather than handing it to ``construct_type``."""
    origin = typing.get_origin(type_)
    if origin is typing.Annotated:
        # Built with its metadata (see ``_lenient_construct``).
        return _plugin_builds(typing.get_args(type_)[0])
    if origin in (typing.Union, types.UnionType):  # pyright: ignore[reportDeprecated]
        # An Optional of a type the plugin builds; OpenAI's own unions keep their metadata.
        return any(
            _plugin_builds(arg)
            for arg in typing.get_args(type_)
            if arg is not type(None)
        )
    if origin in (list, dict):
        return True
    # A generic specialization by its class.
    cls = origin or type_
    return is_typeddict(cls) or (
        isinstance(cls, type)
        and (
            dataclasses.is_dataclass(cls)
            or (
                issubclass(cls, pydantic.BaseModel)
                and not issubclass(cls, openai.BaseModel)
            )
        )
    )


def _may_hold(type_: typing.Any, value: typing.Any) -> bool:
    """Whether a union variant could hold ``value``, judged by its container type.

    A variant behind a validator that can change the value always may: its validation decides
    after the validator runs.
    """
    if _changes_input(type_):
        return True
    origin = typing.get_origin(type_) or type_
    if origin is typing.Annotated:
        return _may_hold(typing.get_args(type_)[0], value)
    if origin in (typing.Union, types.UnionType):  # pyright: ignore[reportDeprecated]
        return any(_may_hold(arg, value) for arg in typing.get_args(type_))
    if isinstance(origin, type) and issubclass(origin, pydantic.RootModel):
        return _may_hold(_field_type(origin.model_fields["root"]), value)
    if isinstance(value, list):
        return origin is list
    if isinstance(value, dict):
        return (
            origin is dict
            or is_typeddict(origin)
            or (
                isinstance(origin, type)
                and (
                    dataclasses.is_dataclass(origin)
                    or issubclass(origin, pydantic.BaseModel)
                )
            )
        )
    if isinstance(value, str):
        return origin is str or isinstance(origin, type) and issubclass(origin, str)
    return True


def _changes_input(type_: typing.Any) -> bool:
    """Whether ``type_`` has a validator that can change its value first.

    Only types the plugin builds or validates are judged: its own and any ``Annotated`` type,
    such as an OpenAI model behind a caller's validator. Lists, dicts and unions judge their
    items' and variants' kinds themselves.
    """
    origin = typing.get_origin(type_)
    if origin in (
        list,
        dict,
        typing.Union,  # pyright: ignore[reportDeprecated]
        types.UnionType,
    ) or not (origin is typing.Annotated or _plugin_builds(type_)):
        return False
    try:
        schema = pydantic.TypeAdapter(type_).core_schema
    except (pydantic.PydanticUserError, pydantic.PydanticUndefinedAnnotation):
        return False
    transformed, _ = _variant_target(schema, _schema_refs(schema, {}))
    return transformed


class _OpenAIJSONPlainPayloadConverter(PydanticJSONPlainPayloadConverter):
    """Strict pydantic deserialization with a lenient fallback.

    OpenAI's response models can drift from live API payloads (e.g. a
    deprecated-but-required field the API has stopped sending). The SDK tolerates
    this when parsing responses, but strict ``validate_json`` on the workflow
    side does not, so fall back to lenient construction when validation fails.
    """

    def from_payload(
        self,
        payload: temporalio.api.common.v1.Payload,
        type_hint: type | None = None,
    ) -> typing.Any:
        """See base class."""
        try:
            return super().from_payload(payload, type_hint)
        except pydantic.ValidationError:
            if type_hint is None:
                raise
            return _lenient_construct(type_hint, json.loads(payload.data))


class OpenAIPayloadConverter(CompositePayloadConverter):
    """PayloadConverter for OpenAI agents."""

    def __init__(self) -> None:
        """Initialize a payload converter."""
        json_payload_converter = _OpenAIJSONPlainPayloadConverter(
            ToJsonOptions(exclude_unset=True)
        )
        super().__init__(
            *(
                c
                if not isinstance(c, JSONPlainPayloadConverter)
                else json_payload_converter
                for c in DefaultPayloadConverter.default_encoding_payload_converters
            )
        )


def _data_converter(converter: DataConverter | None) -> DataConverter:
    if converter is None:
        return DataConverter(payload_converter_class=OpenAIPayloadConverter)
    elif converter.payload_converter_class is DefaultPayloadConverter:
        return dataclasses.replace(
            converter, payload_converter_class=OpenAIPayloadConverter
        )
    elif not isinstance(converter.payload_converter, OpenAIPayloadConverter):
        raise ValueError(
            "The payload converter must be of type OpenAIPayloadConverter."
        )
    return converter


class OpenAIAgentsPlugin(SimplePlugin):
    """Temporal plugin for integrating OpenAI agents with Temporal workflows.

    This plugin provides seamless integration between the OpenAI Agents SDK and
    Temporal workflows. It automatically configures the necessary interceptors,
    activities, and data converters to enable OpenAI agents to run within
    Temporal workflows with proper tracing and model execution.

    The plugin:
    1. Configures the Pydantic data converter for type-safe serialization
    2. Sets up tracing interceptors for OpenAI agent interactions
    3. Registers model execution activities
    4. Automatically registers MCP server activities and manages their lifecycles
    5. Manages the OpenAI agent runtime overrides during worker execution

    Example:
        >>> from temporalio.client import Client
        >>> from temporalio.worker import Worker
        >>> from agents.mcp import MCPServerStreamableHttp
        >>> from temporalio.openai_agents import OpenAIAgentsPlugin, ModelActivityParameters
        >>> from datetime import timedelta
        >>>
        >>> # Configure model parameters
        >>> model_params = ModelActivityParameters(
        ...     start_to_close_timeout=timedelta(seconds=30),
        ...     retry_policy=RetryPolicy(maximum_attempts=3)
        ... )
        >>>
        >>> plugin = OpenAIAgentsPlugin(
        ...     model_params=model_params,
        ...     mcp_servers={"filesystem": lambda: MCPServerStreamableHttp(
        ...         name="filesystem", params={"url": "https://example.com/mcp"}
        ...     )}
        ... )
        >>>
        >>> # Use with client and worker
        >>> client = await Client.connect(
        ...     "localhost:7233",
        ...     plugins=[plugin]
        ... )
        >>> worker = Worker(
        ...     client,
        ...     task_queue="my-task-queue",
        ...     workflows=[MyWorkflow],
        ... )
    """

    def __init__(
        self,
        model_params: ModelActivityParameters | None = None,
        model_provider: ModelProvider | None = None,
        mcp_server_providers: Sequence[
            "StatelessMCPServerProvider | StatefulMCPServerProvider"
        ] = (),
        sandbox_clients: Sequence["SandboxClientProvider"] = (),
        register_activities: bool = True,
        add_temporal_spans: bool = True,
        use_otel_instrumentation: bool = False,
        resolvable_worker_env_vars: Collection[str] | AllowAllWorkerEnvVars = (),
        *,
        mcp_servers: Mapping[str, Callable[..., MCPServer]] | None = None,
        mcp_connection_idle_timeout: timedelta | None = timedelta(minutes=5),
    ) -> None:
        """Initialize the OpenAI agents plugin.

        Args:
            model_params: Configuration parameters for Temporal activity execution
                of model calls. If None, default parameters will be used.
            model_provider: Optional model provider for custom model implementations.
                Useful for testing or custom model integrations.
            mcp_server_providers: Legacy MCP server providers to register with the
                worker. Deprecated; use ``mcp_servers`` for new MCP v2
                integrations.
            mcp_servers: Named OpenAI MCP server factories to register on the worker.
                Each factory may declare at most one positional parameter, which
                receives the workflow's ``factory_argument`` (``None`` when the
                workflow supplied none); a factory declaring more raises here. A
                parameterless factory's modern connection is cached between
                Activities. A factory called with a workflow ``factory_argument``
                always creates a fresh server for that Activity. A factory's server
                with a callable ``tool_filter`` or a ``require_approval`` policy
                that can ask for approval is rejected; pass those to
                ``workflow.temporal_mcp_server()``.
            mcp_connection_idle_timeout: How long an idle modern MCP connection
                remains cached. Defaults to five minutes. None disables idle
                eviction, and zero disables reuse.
            sandbox_clients: Sequence of named sandbox client providers to register
                on the worker.  Each provider pairs a unique name with a real
                ``BaseSandboxClient`` (e.g. ``DaytonaSandboxClient``,
                ``UnixLocalSandboxClient``).  On the workflow side, use
                ``temporal_sandbox_client``
                with the matching name to target the correct backend.
                Warning: sandbox_clients is experimental and behavior may change in future versions.
                Use with caution in production environments.
            register_activities: Whether to register activities during the worker execution.
                This can be disabled on some workers to allow a separation of workflows and activities
                but should not be disabled on all workers, or agents will not be able to progress.
            add_temporal_spans: Whether to add temporal spans to traces
            use_otel_instrumentation: If set to true, enable open telemetry instrumentation.
                Warning: use_otel_instrumentation is experimental and behavior may change in future versions.
                Use with caution in production environments.
            resolvable_worker_env_vars: Names of the environment variables that
                ``temporal_worker_env_ref()`` and ``TemporalWorkerEnvValue`` may
                read on this worker. Names are matched exactly, with no globbing;
                pass ``AllowAllWorkerEnvVars()`` in place of the names to allow
                every variable.
                Warning: resolvable_worker_env_vars is experimental and behavior may change in future versions.
                Use with caution in production environments.

        """
        if model_params is None:
            model_params = ModelActivityParameters()

        # For the default provider, we provide a default start_to_close_timeout of 60 seconds.
        # Other providers will need to define their own.
        if (
            model_params.start_to_close_timeout is None
            and model_params.schedule_to_close_timeout is None
        ):
            if model_provider is None:
                model_params.start_to_close_timeout = timedelta(seconds=60)
            else:
                raise ValueError(
                    "When configuring a custom provider, the model activity must have start_to_close_timeout or schedule_to_close_timeout"
                )

        self._use_otel_instrumentation = use_otel_instrumentation
        if mcp_server_providers:
            warnings.warn(
                "mcp_server_providers is deprecated; use mcp_servers with "
                "workflow.temporal_mcp_server() instead",
                DeprecationWarning,
                stacklevel=2,
            )

        mcp_activities = None
        if mcp_servers:
            # The shared Activity layer requires MCP v2, but the legacy OpenAI
            # provider path remains importable with MCP v1.
            try:
                from temporalio.mcp._activities import _MCPActivities
                from temporalio.openai_agents._mcp_backend import (
                    _mcp_server_backend_factory,
                )
            except ImportError as err:
                raise RuntimeError(
                    "mcp_servers requires the 'mcp' extra and MCP Python SDK v2; "
                    "install temporalio-openai-agents[mcp]"
                ) from err

            mcp_activities = _MCPActivities(
                {
                    name: _mcp_server_backend_factory(name, factory)
                    for name, factory in mcp_servers.items()
                },
                idle_timeout=mcp_connection_idle_timeout,
            )

        resolvable_env_vars = _snapshot_resolvable_env_vars(resolvable_worker_env_vars)

        # Delay activity construction until they are actually needed
        def add_activities(
            activities: Sequence[Callable] | None,
        ) -> Sequence[Callable]:
            if not register_activities:
                return activities or []

            model_activity = ModelActivity(
                model_provider, resolvable_worker_env_vars=resolvable_env_vars
            )
            new_activities = [
                model_activity.invoke_model_activity,
                model_activity.invoke_model_activity_streaming,
            ]

            if mcp_activities:
                new_activities.extend(mcp_activities.activities)

            server_names = [server.name for server in mcp_server_providers]
            if len(server_names) != len(set(server_names)):
                raise ValueError(
                    "More than one mcp server registered with the same name. Please provide unique names."
                )
            for mcp_server in mcp_server_providers:
                new_activities.extend(mcp_server._get_activities())

            sandbox_names = [sc.name for sc in sandbox_clients]
            if len(sandbox_names) != len(set(sandbox_names)):
                raise ValueError(
                    "More than one sandbox client registered with the same name. Please provide unique names."
                )

            for sandbox_provider in sandbox_clients:
                new_activities.extend(
                    sandbox_provider._get_activities(resolvable_env_vars)
                )

            return list(activities or []) + new_activities

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if not runner:
                raise ValueError("No WorkflowRunner provided to the OpenAI plugin.")

            # If in sandbox, add additional passthrough
            if isinstance(runner, SandboxedWorkflowRunner):
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "openai", "agents", "mcp", "mcp_types"
                    ),
                )
            return runner

        if not use_otel_instrumentation:
            interceptor = OpenAIAgentsContextPropagationInterceptor(
                add_temporal_spans=add_temporal_spans,
            )
        else:
            from opentelemetry import trace as otel_trace

            from ._otel_trace_interceptor import (
                OTelOpenAIAgentsContextPropagationInterceptor,
            )

            provider = otel_trace.get_tracer_provider()
            if not isinstance(provider, ReplaySafeTracerProvider):
                raise ValueError(
                    "Global tracer provider must a ReplaySafeTracerProvider. Use temporalio.contrib.opentelemtry.create_trace_provider to create one."
                )

            interceptor = OTelOpenAIAgentsContextPropagationInterceptor(
                add_temporal_spans=add_temporal_spans,
            )

        @asynccontextmanager
        async def run_context() -> AsyncGenerator[None, None]:
            if mcp_activities:
                async with mcp_activities.run_context():
                    with self.tracing_context():
                        with _set_open_ai_agent_temporal_overrides(
                            model_params,
                            start_spans_in_replay=use_otel_instrumentation,
                        ):
                            yield
            else:
                with self.tracing_context():
                    with _set_open_ai_agent_temporal_overrides(
                        model_params,
                        start_spans_in_replay=use_otel_instrumentation,
                    ):
                        yield

        super().__init__(
            name="OpenAIAgentsPlugin",
            data_converter=_data_converter,
            interceptors=[interceptor],
            activities=add_activities,
            workflow_runner=workflow_runner,
            workflow_failure_exception_types=[AgentsWorkflowError],
            run_context=lambda: run_context(),
        )

    @contextmanager
    def tracing_context(self) -> Generator[None, None, None]:
        """Context manager for setting up OpenAI Agents tracing instrumentation.

        This should be called if AgentsSDK traces and/or spans are started outside of the context of a worker.
        For example:

        .. code-block:: python

            with env.openai_agents_plugin.tracing_context():
                with trace("External trace"):
                    with custom_span("External span"):
                        workflow_handle = await new_client.start_workflow(
                            ...
                        )

        Yields:
            Context with tracing instrumentation enabled.
        """
        # Set up OTEL instrumentation if enabled
        otel_instrumentation_installed = False
        if self._use_otel_instrumentation:
            from opentelemetry import trace

            _install_otel_instrumentation(trace.get_tracer_provider())
            otel_instrumentation_installed = True
        try:
            yield
        finally:
            # Clean up OTEL instrumentation
            if otel_instrumentation_installed:
                _uninstall_otel_instrumentation()
