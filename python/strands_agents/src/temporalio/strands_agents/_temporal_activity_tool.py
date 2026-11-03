import functools
import inspect
import json
import sys
import typing
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Annotated, Any

if sys.version_info >= (3, 14):
    import annotationlib  # type: ignore[reportUnreachable]

from pydantic import BaseModel, Field, create_model
from strands.interrupt import Interrupt
from strands.tools.decorator import FunctionToolMetadata
from strands.types._events import ToolInterruptEvent, ToolResultEvent
from strands.types.tools import AgentTool, ToolGenerator, ToolResult, ToolSpec, ToolUse

from temporalio import activity, workflow
from temporalio.exceptions import ActivityError, ApplicationError, FailureError

from ._failure_converter import STRANDS_INTERRUPT_TYPE


class TemporalActivityTool(AgentTool):
    """Strands ``AgentTool`` whose body dispatches a Temporal activity."""

    def __init__(self, activity_fn: Callable, options: dict[str, Any]) -> None:
        """Capture the target activity and the options to invoke it with."""
        super().__init__()
        defn = activity._Definition.from_callable(activity_fn)
        if not defn or not defn.name:
            raise ValueError("activity_fn must be decorated with @activity.defn")
        if not inspect.isclass(activity_fn) and hasattr(
            type(activity_fn), "__temporal_activity_definition"
        ):
            raise ValueError(
                f"Pass the activity class {type(activity_fn).__name__}, not an"
                " instance of it"
            )
        self._activity_name = defn.name
        self._options = options
        if inspect.isclass(activity_fn):
            # The stand-in's parameters are already bound: none is a receiver.
            activity_fn = _call_stand_in(activity_fn)
        signature = inspect.signature(activity_fn)
        params = list(signature.parameters.values())
        # A first parameter named `self` of a method passed as Class.method is
        # its receiver, however it is annotated: the worker registers the bound
        # method, so its instance supplies it. A staticmethod has no receiver.
        if params and params[0].name == "self" and _defined_as_method(activity_fn):
            params = params[1:]
        # The model's input binds by name; the activity still gets it by position.
        self._signature = signature.replace(
            parameters=[
                p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
                if p.kind is inspect.Parameter.POSITIONAL_ONLY
                else p
                for p in params
            ]
        )
        spec = _ActivityToolMetadata(activity_fn, self._signature).extract_metadata()
        spec["name"] = self._activity_name
        self._spec: ToolSpec = spec

    @property
    def tool_name(self) -> str:
        """Name of the underlying Temporal activity."""
        return self._activity_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Strands ToolSpec derived from the activity's signature."""
        return self._spec

    @property
    def tool_type(self) -> str:
        """Tool kind identifier used by Strands."""
        return "temporal_activity"

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> ToolGenerator:
        """Execute the tool by dispatching to the bound Temporal activity."""
        bound = self._signature.bind(**tool_use["input"])
        bound.apply_defaults()
        positional = list(bound.arguments.values())
        try:
            if not positional:
                result = await workflow.execute_activity(
                    self._activity_name, **self._options
                )
            elif len(positional) == 1:
                result = await workflow.execute_activity(
                    self._activity_name, positional[0], **self._options
                )
            else:
                result = await workflow.execute_activity(
                    self._activity_name, args=positional, **self._options
                )
        except ActivityError as e:
            cause = e.__cause__
            if (
                isinstance(cause, ApplicationError)
                and cause.type == STRANDS_INTERRUPT_TYPE
            ):
                yield ToolInterruptEvent(tool_use, [Interrupt(**cause.details[0])])
                return
            yield _activity_error_event(tool_use["toolUseId"], e)
            return
        yield ToolResultEvent(
            ToolResult(
                toolUseId=tool_use["toolUseId"],
                status="success",
                content=[{"text": _to_text(result)}],
            )
        )


class _ActivityToolMetadata(FunctionToolMetadata):
    """Builds the tool spec from every parameter of the tool's binding signature.

    Strands leaves parameters named ``self``, ``cls`` and ``agent`` out of the
    spec because it supplies them to its own tools. The tool sends the activity
    only the model's input, so they are inputs too.
    """

    def __init__(self, func: Callable, signature: inspect.Signature) -> None:
        # Set before the base class builds the input model.
        self._inputs = signature.parameters
        super().__init__(func)

    def _create_input_model(self) -> type[BaseModel]:
        # Pydantic refuses some field names, such as ones with a leading
        # underscore, so each field is named after its position and aliased to
        # its parameter: the schema's properties keep the parameters' names.
        fields: dict[str, Any] = {}
        for i, (name, param) in enumerate(self._inputs.items()):
            annotation = param.annotation
            if isinstance(annotation, str):
                annotation = self.type_hints.get(name, annotation)
            if annotation is inspect.Parameter.empty:
                annotation = Any
            default = ... if param.default is inspect.Parameter.empty else param.default
            field_type, field = self._extract_annotated_metadata(
                annotation, name, default
            )
            fields[f"input_{i}"] = (Annotated[field_type, Field(alias=name)], field)
        return create_model(f"{self.func.__name__.capitalize()}Tool", **fields)


def _defined_as_method(fn: Any) -> bool:
    """Return whether ``fn`` is a function a class body defines as a method.

    The worker registers such a function bound to an instance, unlike one the
    class wraps in ``staticmethod``. Only the class tells them apart, so it is
    found by the function's module and qualname.

    Raises:
        ValueError: If ``fn`` is defined in a class that can't be reached by
            name, such as one a function defines.
    """
    if not inspect.isfunction(fn):
        return False
    scopes = fn.__qualname__.split(".")
    if len(scopes) < 2 or scopes[-2] == "<locals>":
        return False
    owner: Any = sys.modules.get(fn.__module__)
    for scope in scopes[:-1]:
        owner = getattr(owner, "__dict__", {}).get(scope)
    attr = vars(owner).get(scopes[-1]) if inspect.isclass(owner) else None
    if isinstance(attr, staticmethod) and attr.__func__ is fn:
        return False
    if attr is fn:
        return True
    raise ValueError(
        f"Cannot tell whether the first parameter 'self' of {fn.__qualname__} is"
        f" a receiver: its class can't be reached by name in {fn.__module__}."
        " Define the class at module level, pass a bound method, or rename the"
        " parameter"
    )


def _call_function(method: Any) -> Callable | None:
    """Return the function under a function, staticmethod, classmethod or partialmethod."""
    while isinstance(method, (staticmethod, classmethod, functools.partialmethod)):
        method = (
            method.func
            if isinstance(method, functools.partialmethod)
            else method.__func__
        )
    return method if inspect.isfunction(method) else None


def _docstring(fn: Any) -> str | None:
    """Return ``fn``'s docstring, unless it is missing or blank."""
    doc = getattr(fn, "__doc__", None) if fn is not None else None
    return doc if doc and doc.strip() else None


def _generated_class_doc(cls: type) -> str | None:
    """Return the docstring dataclasses or namedtuple writes for a class that has none."""
    if is_dataclass(cls):
        try:
            # Format it as dataclasses did when it made the class.
            if sys.version_info >= (3, 14):
                signature = inspect.signature(  # type: ignore[reportUnreachable]
                    cls, annotation_format=annotationlib.Format.FORWARDREF
                )
                # 3.14 left a forward reference in the __init__ dataclasses
                # generated unevaluated, and the field keeps the reference from
                # that time.
                annotate = getattr(cls.__init__, "__annotate__", None)
                if getattr(annotate, "__generated_by_dataclasses__", False):
                    field_types = {
                        n: f.type for n, f in cls.__dataclass_fields__.items()
                    }
                    signature = signature.replace(
                        parameters=[
                            p.replace(annotation=field_types.get(p.name, p.annotation))
                            for p in signature.parameters.values()
                        ]
                    )
            else:
                signature = inspect.signature(cls)  # type: ignore[reportUnreachable]
            return cls.__name__ + str(signature).replace(" -> None", "")
        except (TypeError, ValueError):
            return cls.__name__
    fields = getattr(cls, "_fields", None)
    if issubclass(cls, tuple) and isinstance(fields, tuple):
        names = ", ".join(fields) + ("," if len(fields) == 1 else "")
        return f"{cls.__name__}({names})"
    return None


def _class_doc(cls: type) -> str | None:
    """Return a class's own docstring, unless dataclasses or namedtuple wrote it."""
    doc = _docstring(cls)
    if doc is None or not doc.startswith(cls.__name__):
        return doc
    try:
        generated = _generated_class_doc(cls)
    except Exception:
        # dataclasses would have failed the same way, so it didn't write this
        # docstring.
        return doc
    return None if doc == generated else doc


def _call_stand_in(cls: type) -> Callable:
    """Return a stand-in for ``cls.__call__`` that Strands builds the tool spec from.

    Workflow code references a callable-class activity by its class, but the
    worker's registered instance runs ``__call__``, so the stand-in has
    ``__call__``'s parameters as an instance binds it: without the receiver of a
    method, classmethod or partialmethod, and with every parameter of a
    staticmethod. It carries the class's name, ``__call__``'s types and the
    tool's docstring. The tool never calls it.
    """
    calls = [vars(c)["__call__"] for c in cls.__mro__ if "__call__" in vars(c)]
    func = _call_function(calls[0]) if calls else None
    if func is None:
        raise ValueError(
            f"Activity class {cls.__name__} needs a __call__ that is a function,"
            " staticmethod, classmethod or functools.partialmethod"
        )
    try:
        call = calls[0].__get__(object(), cls)
        signature = inspect.signature(call)
        # Resolve the types in __call__'s own module: the stand-in has no globals.
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception as e:
        raise ValueError(
            f"Cannot read the parameters of {cls.__name__}.__call__: {e}"
        ) from e
    # @activity.defn checks the constructor's signature, not __call__'s, so
    # __call__ can take *args, keyword-only arguments or **kwargs, which the
    # tool can't pass: it sends one argument per parameter, by position.
    for p in signature.parameters.values():
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        ):
            message = (
                f"Activity class {cls.__name__} cannot take *args, keyword-only"
                f" arguments or **kwargs in __call__ ('{p.name}'): the tool passes"
                " one argument per parameter, by position"
            )
            if isinstance(calls[0], functools.partialmethod) and calls[0].keywords:
                message += (
                    ". Its functools.partialmethod binds an argument by keyword,"
                    " which makes the arguments after it keyword-only"
                )
            raise ValueError(message)
    stand_in = functools.partial(call)
    setattr(stand_in, "__name__", cls.__name__)
    stand_in.__annotations__ = hints
    # __call__'s own docstring, else the class's, else that of a __call__ it
    # inherits.
    stand_in.__doc__ = (
        _docstring(_call_function(vars(cls).get("__call__")))
        or _class_doc(cls)
        or next(filter(None, map(_docstring, map(_call_function, calls))), None)
    )
    return stand_in


def _activity_error_event(tool_use_id: str, error: ActivityError) -> ToolResultEvent:
    # ActivityError reports a generic message; its cause holds the activity's failure.
    cause = error.__cause__
    exception = cause if isinstance(cause, Exception) else error
    message = (
        exception.message if isinstance(exception, FailureError) else str(exception)
    )
    return ToolResultEvent(
        ToolResult(
            toolUseId=tool_use_id,
            status="error",
            content=[{"text": message}],
        ),
        exception=exception,
    )


def _to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return str(result)
