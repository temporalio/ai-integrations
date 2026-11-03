"""Workflow utilities for Google Gemini SDK integration with Temporal.

This module provides utilities for using the Google Gemini SDK within Temporal
workflows.  The key entry points are:

- :func:`activity_as_tool` — converts a Temporal activity into a Gemini tool
  callable for use with automatic function calling (AFC).
"""

from __future__ import annotations

import functools
import inspect
import sys
import typing
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any

if sys.version_info >= (3, 14):
    import annotationlib  # type: ignore[reportUnreachable]

from temporalio import activity
from temporalio import workflow as temporal_workflow
from temporalio.google_genai._errors import GoogleGenAIError
from temporalio.workflow import ActivityConfig


def _call_function(method: Any) -> Callable | None:
    """Returns the function under a function, staticmethod, classmethod or partialmethod."""
    while isinstance(method, (staticmethod, classmethod, functools.partialmethod)):
        method = (
            method.func
            if isinstance(method, functools.partialmethod)
            else method.__func__
        )
    return method if inspect.isfunction(method) else None


def _docstring(fn: Any) -> str | None:
    """Returns ``fn``'s docstring, unless it is missing or blank."""
    doc = getattr(fn, "__doc__", None) if fn is not None else None
    return doc if doc and doc.strip() else None


def _generated_class_doc(cls: type) -> str | None:
    """Returns the docstring dataclasses or namedtuple writes for a class that has none."""
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
    """Returns a class's own docstring, unless dataclasses or namedtuple wrote it."""
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


def _call_signature(
    cls: type,
) -> tuple[inspect.Signature, dict[str, Any], str | None]:
    """Returns the parameters, types and docstring of a callable class's tool."""
    # Workflow code references a callable-class activity by its class, but the
    # worker's registered instance runs __call__, so take __call__'s parameters
    # as an instance binds it: without the receiver of a method, classmethod or
    # partialmethod, and with every parameter of a staticmethod.
    calls = [vars(c)["__call__"] for c in cls.__mro__ if "__call__" in vars(c)]
    func = _call_function(calls[0]) if calls else None
    if func is None:
        raise ValueError(
            f"Activity class '{cls.__name__}' needs a __call__ that is a function,"
            " staticmethod, classmethod or functools.partialmethod."
        )
    try:
        signature = inspect.signature(calls[0].__get__(object(), cls))
        # Resolve the types in __call__'s own module: the wrapper has other
        # globals.
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception as e:
        raise ValueError(
            f"Cannot read the parameters of '{cls.__name__}.__call__': {e}"
        ) from e
    parameters = []
    for p in signature.parameters.values():
        # @activity.defn checks the constructor's signature, not __call__'s, so
        # __call__ can take keyword-only arguments or **kwargs, which the tool
        # can't pass: activities get their arguments by position.
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD):
            message = (
                f"Activity class '{cls.__name__}' cannot take keyword-only"
                f" arguments or **kwargs in __call__ ('{p.name}'): activities get"
                " their arguments by position."
            )
            if isinstance(calls[0], functools.partialmethod) and calls[0].keywords:
                message += (
                    " Its functools.partialmethod binds an argument by keyword,"
                    " which makes the arguments after it keyword-only."
                )
            raise ValueError(message)
        # Gemini passes every argument by name, so bind positional-only
        # parameters by name too.
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            p = p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
        parameters.append(p)
    # __call__'s own docstring, else the class's, else that of a __call__ it
    # inherits.
    doc = (
        _docstring(_call_function(vars(cls).get("__call__")))
        or _class_doc(cls)
        or next(filter(None, map(_docstring, map(_call_function, calls))), None)
    )
    return signature.replace(parameters=parameters), hints, doc


def activity_as_tool(
    fn: Callable,
    *,
    activity_config: ActivityConfig | None = None,
) -> Callable:
    """Convert a Temporal activity into a Gemini-compatible async tool callable.

    .. warning::
        This API is experimental and may change in future versions.
        Use with caution in production environments.

    Returns an async callable with the same name, docstring, and type signature as
    ``fn``. When Gemini's automatic function calling (AFC) invokes the returned
    callable from within a Temporal workflow, the call is executed as a Temporal
    activity via :func:`workflow.execute_activity`. Each tool invocation therefore
    appears as a separate, durable entry in the workflow event history.

    Because AFC is left **enabled**, the Gemini SDK owns the agentic loop — no
    manual ``while`` loop or ``run_agent()`` helper is required. Pass the returned
    callable directly to ``GenerateContentConfig(tools=[...])``.

    Args:
        fn: A Temporal activity function decorated with ``@activity.defn``, a
            method passed as ``Class.method``, or a callable-class activity
            passed as its class. A class's tool takes the parameters of its
            ``__call__``.
        activity_config: Configuration for the activity execution (timeouts,
            retry policy, etc.).  Must set ``start_to_close_timeout`` or
            ``schedule_to_close_timeout`` — Temporal requires one, and there is
            no default; otherwise the tool call raises when the activity is
            invoked.

    Returns:
        An async callable suitable for use as a Gemini tool.

    Raises:
        GoogleGenAIError: If ``fn`` is not decorated with ``@activity.defn`` or
            has no activity name.
        ValueError: If ``fn`` is a class whose ``__call__`` is not a function,
            staticmethod, classmethod or ``functools.partialmethod``, can't be
            read, or takes keyword-only arguments or ``**kwargs``.
    """
    ret = activity._Definition.from_callable(fn)
    if not ret:
        raise GoogleGenAIError(
            "Bare function without @activity.defn decorator is not supported",
            "invalid_tool",
        )
    if ret.name is None:
        raise GoogleGenAIError(
            "Activity must have a name to be used as a Gemini tool",
            "invalid_tool",
        )

    config: ActivityConfig = {**(activity_config or {})}
    if "summary" not in config:
        config["summary"] = "tool_call"

    schema_fn: Callable = fn
    if inspect.isclass(fn):
        sig, annotations, doc = _call_signature(fn)
    else:
        # For a method passed as Class.method the first parameter is 'self'.
        # Partially apply it so that Gemini inspects only the user-facing
        # parameters when building the function-call schema, while the worker
        # resolves the real instance at execution time.
        params = list(inspect.signature(fn).parameters.keys())
        if params and params[0] == "self":
            partial = functools.partial(fn, None)
            setattr(partial, "__name__", fn.__name__)
            partial.__annotations__ = getattr(fn, "__annotations__", {})
            setattr(
                partial,
                "__temporal_activity_definition",
                getattr(fn, "__temporal_activity_definition", None),
            )
            partial.__doc__ = fn.__doc__
            schema_fn = partial
        sig = inspect.signature(schema_fn)
        annotations = getattr(schema_fn, "__annotations__", {})
        doc = schema_fn.__doc__

    activity_name: str = ret.name

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        activity_args = list(bound.arguments.values())
        return await temporal_workflow.execute_activity(
            activity_name,
            args=activity_args,
            **config,
        )

    wrapper.__name__ = schema_fn.__name__  # type: ignore
    wrapper.__doc__ = doc
    setattr(wrapper, "__signature__", sig)
    wrapper.__annotations__ = annotations

    return wrapper
