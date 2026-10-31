"""Workflow utilities for Google ADK agents integration with Temporal."""

import functools
import inspect
import sys
import types
import typing
from dataclasses import dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

if sys.version_info >= (3, 14):
    import annotationlib  # type: ignore[reportUnreachable]

import temporalio.workflow
from temporalio import workflow

if TYPE_CHECKING:
    from google.adk.workflow import FunctionNode

_TOOL_CONTEXT_PARAM = "tool_context"


@dataclass(frozen=True)
class ToolContextSnapshot:
    """Serializable snapshot of the ADK ``ToolContext`` for activity-backed tools.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    ADK's ``ToolContext`` holds live, non-serializable objects, so it cannot
    cross the activity boundary: activity inputs are sent to the server and
    may run on a different worker than the workflow. This snapshot carries the
    serializable subset instead.

    Declare a parameter named ``tool_context`` annotated with this type (or
    ``ToolContextSnapshot | None``) on an activity wrapped by
    :func:`activity_as_tool`:

    .. code-block:: python

        @activity.defn
        async def get_weather(query: str, tool_context: ToolContextSnapshot) -> dict:
            db_url = tool_context.state.get("url", "")
            ...

    Exactly like a native ADK function tool's ``tool_context`` parameter, it
    is excluded from the LLM-facing tool schema and filled in at invocation
    time — here with a snapshot taken from the live ``ToolContext`` before the
    activity is scheduled.

    When running under Temporal, the entire session state crosses the
    activity boundary: every value in it must be serializable by the
    configured data converter and the total size must fit within payload
    limits, even for keys the tool never reads. A non-serializable value
    fails the workflow task when the activity is scheduled. Local ADK runs
    pass the snapshot in memory and have no such constraint.

    The snapshot is one-way and should be treated as read-only: mutating it
    inside the activity does not propagate back to the session (and in local
    runs nested values may alias the live session state, so mutating them can
    corrupt the session). To modify session state, return the needed
    information from the activity and apply it in workflow-side code (for
    example an ADK callback or a plain tool function).

    Attributes:
        state: The session state visible to this tool call, as a plain dict.
        function_call_id: The id of the function call being handled, when
            available.
    """

    state: dict[str, Any] = field(default_factory=dict)
    function_call_id: str | None = None


def _strip_annotated(annotation: Any) -> Any:
    """Returns the type under ``Annotated[...]``, as ADK sees it."""
    if typing.get_origin(annotation) is typing.Annotated:
        return typing.get_args(annotation)[0]
    return annotation


def _annotation_members(annotation: Any) -> tuple[Any, ...]:
    """Returns a union annotation's members, or the annotation itself.

    ``Annotated`` metadata is dropped and nested unions are flattened, at any
    depth, as ``typing.get_type_hints`` does for a function's hints.
    """
    annotation = _strip_annotated(annotation)
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:  # pyright: ignore[reportDeprecated]
        members: list[Any] = []
        for arg in typing.get_args(annotation):
            members += [m for m in _annotation_members(arg) if m not in members]
        return tuple(members)
    return (annotation,)


def _annotation_display(members: tuple[Any, ...]) -> str:
    """Renders annotation members for error messages."""
    return " | ".join(
        "None" if member is type(None) else getattr(member, "__name__", str(member))
        for member in members
    )


def _activity_name(activity_def: Callable) -> str:
    """Returns the activity's name for messages; an instance has its class's."""
    return getattr(activity_def, "__name__", type(activity_def).__name__)


def _adk_context_error(
    activity_def: Callable, name: str, annotation_name: str
) -> ValueError:
    """Builds the error for a parameter annotated with a live ADK context."""
    return ValueError(
        f"Activity '{_activity_name(activity_def)}' declares '{name}:"
        f" {annotation_name}', but ADK context objects are not serializable"
        " and cannot be activity arguments. Declare a parameter named"
        " 'tool_context' annotated with ToolContextSnapshot instead to receive"
        " the serializable subset (session state and function-call id)."
    )


def _validated_tool_context_parameter(
    activity_def: Callable, parameter: inspect.Parameter, members: tuple[Any, ...]
) -> inspect.Parameter:
    """Returns the ``tool_context`` parameter after validating its annotation.

    The parameter must be annotated with :class:`ToolContextSnapshot` or
    ``ToolContextSnapshot | None``.
    """
    if parameter.annotation is inspect.Parameter.empty:
        raise ValueError(
            f"Activity '{_activity_name(activity_def)}' has an unannotated"
            " 'tool_context' parameter. Annotate it with ToolContextSnapshot"
            " to receive the serializable subset of the ADK tool context"
            " (session state and function-call id)."
        )
    non_none = [member for member in members if member is not type(None)]
    if non_none == [ToolContextSnapshot]:
        return parameter
    annotation_name = _annotation_display(members)
    if any(
        getattr(member, "__module__", "").startswith("google.adk")
        for member in non_none
    ):
        raise _adk_context_error(activity_def, _TOOL_CONTEXT_PARAM, annotation_name)
    raise ValueError(
        f"Activity '{_activity_name(activity_def)}' has a 'tool_context' parameter"
        f" annotated with {annotation_name}. The name 'tool_context' is"
        " reserved by ADK for context injection; annotate the parameter with"
        " ToolContextSnapshot to receive the serializable subset of the tool"
        " context."
    )


def _tool_context_parameter(
    activity_def: Callable, signature: inspect.Signature, hints: dict[str, Any]
) -> inspect.Parameter | None:
    """Validates context-related parameters and returns the ``tool_context`` one.

    ADK injects the live context into the first parameter annotated with an
    ADK context type (regardless of name), or failing that into one named
    ``tool_context``, and excludes that parameter from the LLM-facing tool
    schema. Live context objects cannot cross the activity boundary, so the
    only supported declaration is a parameter named ``tool_context`` annotated
    with :class:`ToolContextSnapshot` (or ``ToolContextSnapshot | None``);
    anything else ADK would treat as a context parameter is rejected at wrap
    time, as is a misplaced ToolContextSnapshot annotation that would leak
    into the tool schema.
    """
    adk_context: type[Any] | None
    try:
        from google.adk.tools.tool_context import ToolContext

        adk_context = ToolContext
    except ImportError:
        adk_context = None
    tool_context_parameter: inspect.Parameter | None = None
    for name, parameter in signature.parameters.items():
        members = _annotation_members(hints.get(name, parameter.annotation))
        if name == _TOOL_CONTEXT_PARAM:
            tool_context_parameter = _validated_tool_context_parameter(
                activity_def, parameter, members
            )
        elif adk_context is not None and any(
            member is adk_context for member in members
        ):
            raise _adk_context_error(activity_def, name, _annotation_display(members))
        elif any(member is ToolContextSnapshot for member in members):
            raise ValueError(
                f"Activity '{_activity_name(activity_def)}' annotates parameter"
                f" '{name}' with ToolContextSnapshot, but ADK only injects the"
                " tool context into a parameter named 'tool_context'; under"
                " any other name it would appear in the LLM-facing tool"
                " schema. Rename the parameter to 'tool_context'."
            )
    return tool_context_parameter


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


def _mangled(owner: Any, name: str) -> str:
    """Returns the attribute ``name`` in a class body of ``owner`` is stored as."""
    prefix = owner.__name__.lstrip("_") if inspect.isclass(owner) else ""
    if prefix and name.startswith("__") and not name.endswith("__"):
        return f"_{prefix}{name}"
    return name


def _is_method(fn: Callable) -> bool:
    """Returns whether a function taken from a class is a method it binds.

    A module-level or nested function, or a staticmethod, is called with every
    parameter. A function defined in a class body is found in that class to
    tell a method from a staticmethod; a class that can't be found, or whose
    attribute isn't ``fn``, raises ``ValueError``.
    """
    qualname = fn.__qualname__.split(".")
    if len(qualname) < 2 or qualname[-2] == "<locals>":
        return False
    owner: Any = None
    if "<locals>" not in qualname:
        # Look the class up in the globals of the function's own module, not in
        # sys.modules, which differs inside the Workflow sandbox. A
        # functools.wraps decorator's wrapper has its own module's globals, so
        # unwrap it first.
        owner = getattr(inspect.unwrap(fn), "__globals__", {}).get(qualname[0])
        for name in qualname[1:-1]:
            owner = getattr(owner, _mangled(owner, name), None)
    try:
        attr = inspect.getattr_static(owner, _mangled(owner, qualname[-1]))
    except AttributeError:
        attr = None
    if not inspect.isclass(owner) or attr is None:
        raise ValueError(
            f"Cannot tell whether '{fn.__qualname__}' is a method or a"
            f" staticmethod: no class '{'.'.join(qualname[:-1])}' is reachable"
            " from the function's globals. Define the class at module level"
            " (wrap decorators with functools.wraps), pass a bound method for a"
            " method, or rename a staticmethod's parameter."
        )
    static = isinstance(attr, staticmethod)
    if (attr.__func__ if static else attr) is not fn:
        # The class's name or attribute was rebound after fn was taken from it,
        # so the attribute's binding isn't fn's.
        raise ValueError(
            f"Cannot tell whether '{fn.__qualname__}' is a method or a"
            f" staticmethod: '{'.'.join(qualname)}' reachable from the"
            " function's globals is another object. Pass a bound method for a"
            " method, or rename a staticmethod's parameter."
        )
    return not static


def _tool_signature(
    activity_def: Callable,
) -> tuple[inspect.Signature, dict[str, Any], str | None]:
    """Returns the parameters the activity is called with, their types and its docstring."""
    if inspect.isclass(activity_def):
        # Workflow code references a callable-class activity by its class, but
        # the worker's registered instance runs __call__, so take __call__'s
        # parameters as an instance binds it: without the receiver of a method,
        # classmethod or partialmethod, and with every parameter of a
        # staticmethod.
        calls = [
            vars(c)["__call__"] for c in activity_def.__mro__ if "__call__" in vars(c)
        ]
        func = _call_function(calls[0]) if calls else None
        if func is None:
            raise ValueError(
                f"Activity class '{activity_def.__name__}' needs a __call__ that is"
                " a function, staticmethod, classmethod or functools.partialmethod."
            )
        try:
            signature = inspect.signature(calls[0].__get__(object(), activity_def))
            # Resolve the types in __call__'s own module: the wrapper has other
            # globals.
            hints = typing.get_type_hints(func, include_extras=True)
        except Exception as e:
            raise ValueError(
                f"Cannot read the parameters of '{activity_def.__name__}.__call__': {e}"
            ) from e
        # @activity.defn checks the constructor's signature, not __call__'s, so
        # __call__ can take keyword-only arguments or **kwargs, which the tool
        # can't pass: activities get their arguments by position.
        for p in signature.parameters.values():
            if p.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            ):
                message = (
                    f"Activity class '{activity_def.__name__}' cannot take keyword-only"
                    f" arguments or **kwargs in __call__ ('{p.name}'): activities get"
                    " their arguments by position."
                )
                if isinstance(calls[0], functools.partialmethod) and calls[0].keywords:
                    message += (
                        " Its functools.partialmethod binds an argument by keyword,"
                        " which makes the arguments after it keyword-only."
                    )
                raise ValueError(message)
        # __call__'s own docstring, else the class's, else that of a __call__ it
        # inherits.
        doc = (
            _docstring(_call_function(vars(activity_def).get("__call__")))
            or _class_doc(activity_def)
            or next(filter(None, map(_docstring, map(_call_function, calls))), None)
        )
        return signature, hints, doc
    signature = inspect.signature(activity_def)
    params = list(signature.parameters.values())
    # Drop the receiver of a method passed as Class.method: a first parameter
    # named `self`, however it is annotated. The worker's registered instance
    # supplies it.
    if (
        inspect.isfunction(activity_def)
        and params
        and params[0].name == "self"
        and _is_method(activity_def)
    ):
        signature = signature.replace(parameters=params[1:])
    try:
        # A callable instance's types are those of its class's __call__.
        hints = typing.get_type_hints(
            activity_def
            if inspect.isroutine(activity_def)
            else _call_function(
                inspect.getattr_static(type(activity_def), "__call__", None)
            )
        )
    except Exception:
        hints = {}
    return signature, hints, activity_def.__doc__


def _snapshot_tool_context(tool_context: Any) -> ToolContextSnapshot:
    """Builds the serializable snapshot from a live ADK ``ToolContext``."""
    state: dict[str, Any] = {}
    state_object = getattr(tool_context, "state", None)
    if state_object is not None:
        to_dict = getattr(state_object, "to_dict", None)
        state = dict(cast(Any, to_dict() if callable(to_dict) else state_object))
    return ToolContextSnapshot(
        state=state,
        function_call_id=getattr(tool_context, "function_call_id", None),
    )


def activity_as_tool(activity_def: Callable, **kwargs: Any) -> Callable:
    """Decorator/Wrapper to wrap a Temporal Activity as an ADK Tool.

    .. warning::
        This function is experimental and may change in future versions.
        Use with caution in production environments.

    This ensures the activity's signature is preserved for ADK's tool schema generation
    while marking it as a tool that executes via 'workflow.execute_activity'.

    ``activity_def`` can be an activity function, a bound method, a
    callable-class instance, a method passed as ``Class.method``, or a
    callable-class activity passed as its class. The tool offers the parameters
    the activity is called with: without a method's ``self``, and from
    ``__call__`` for a class. A class whose ``__call__`` is not a function,
    staticmethod, classmethod or ``functools.partialmethod``, can't be read, or
    takes keyword-only arguments or ``**kwargs`` raises ``ValueError``. So does
    a method taking ``self`` whose class can't be reached from the function's
    globals, such as a class defined in a function, or whose class's name or
    attribute now holds another object: its ``self`` could be a staticmethod's
    input.

    Outside a Workflow, as in local ADK runs, the tool calls ``activity_def``
    itself. A method passed as ``Class.method`` and a class passed as itself
    need the Worker's instance, so their tools run only in a Workflow.

    If the activity declares a parameter named ``tool_context``, it must be
    annotated with :class:`ToolContextSnapshot` (or ``ToolContextSnapshot |
    None``). ADK excludes the parameter from the tool schema and injects the
    live ``ToolContext`` into the wrapper, which passes the activity a
    serializable snapshot of it (session state and function-call id) in that
    parameter's position. Annotating any parameter with a live ADK context
    type raises ``ValueError`` at wrap time, since ADK would inject the
    non-serializable context into it regardless of its name.
    """
    signature, hints, doc = _tool_signature(activity_def)
    tool_context_param = _tool_context_parameter(activity_def, signature, hints)
    # ADK passes every argument by name, and the activity gets them by
    # position, so bind positional-only parameters by name too.
    binding = signature.replace(
        parameters=[
            p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            if p.kind is inspect.Parameter.POSITIONAL_ONLY
            else p
            for p in signature.parameters.values()
        ]
    )

    @functools.wraps(activity_def)
    async def wrapper(*args: Any, **kw: Any):
        # ADK injects the live ToolContext by name; replace it with the
        # serializable snapshot the activity actually declares.
        if tool_context_param is not None and _TOOL_CONTEXT_PARAM in kw:
            kw[_TOOL_CONTEXT_PARAM] = _snapshot_tool_context(kw[_TOOL_CONTEXT_PARAM])

        # Bind arguments to the parameters the activity is called with
        bound = binding.bind(*args, **kw)
        bound.apply_defaults()

        # Convert to positional args for Temporal
        activity_args = list(bound.arguments.values())

        # Decorator kwargs are defaults.
        options = kwargs.copy()

        if not temporalio.workflow.in_workflow():
            # If executed outside a workflow, like when doing local adk runs, use the function directly
            result = activity_def(*bound.args, **bound.kwargs)
            if inspect.isawaitable(result):
                return await result
            else:
                return result

        if not activity_args:
            return await workflow.execute_activity(activity_def, **options)
        if len(activity_args) == 1:
            return await workflow.execute_activity(
                activity_def, activity_args[0], **options
            )
        return await workflow.execute_activity(
            activity_def, args=activity_args, **options
        )

    # functools.wraps copies name/doc/module/annotations/qualname/dict.
    # Signature must be set explicitly since the wrapper uses *args/**kw.
    setattr(wrapper, "__signature__", signature)
    if inspect.isclass(activity_def):
        # ADK reads the tool's types and description from the wrapper, so give
        # it __call__'s instead of the class's.
        wrapper.__annotations__ = hints
        wrapper.__doc__ = doc

    return wrapper


def _bind_node_input(
    activity_def: Callable, params: list[inspect.Parameter], node_input: dict[str, Any]
) -> list[Any]:
    """Maps a dict node input onto the activity's positional parameters by name."""
    unknown = sorted(set(node_input) - {p.name for p in params})
    if unknown:
        raise TypeError(
            f"Activity node '{activity_def.__name__}' got unexpected input"
            f" key(s) {unknown}."
        )
    missing = [
        p.name
        for p in params
        if p.name not in node_input and p.default is inspect.Parameter.empty
    ]
    if missing:
        raise TypeError(
            f"Activity node '{activity_def.__name__}' is missing required input"
            f" key(s) {missing}."
        )
    return [node_input.get(p.name, p.default) for p in params]


def activity_node(
    activity_def: Callable,
    *,
    name: str | None = None,
    rerun_on_resume: bool = False,
    **kwargs: Any,
) -> "FunctionNode":
    """Wraps a Temporal Activity as a node for an ADK workflow graph.

    .. warning::
        This function is experimental and may change in future versions.
        Use with caution in production environments.

    The returned ``google.adk.workflow.FunctionNode`` can be placed in a
    ``Workflow(edges=[...])`` graph or invoked from a dynamic node via
    ``ctx.run_node(...)``. The previous node's output (``node_input``) is
    passed to the activity: directly for a single-parameter activity, or bound
    by parameter name for a multi-parameter activity (``node_input`` must then
    be a dict). Outside a workflow (local ADK runs) the activity function is
    invoked directly.

    Args:
        activity_def: The ``@activity.defn`` function to run.
        name: Node name; defaults to the activity function's name.
        rerun_on_resume: Passed through to ``FunctionNode``. Keep the default
            ``False`` so that on a human-in-the-loop resume the node is
            fast-forwarded from the session instead of re-executing the
            activity.
        **kwargs: Activity execution options for
            ``workflow.execute_activity`` (e.g. ``start_to_close_timeout``).
            Prefer configuring retries here via ``retry_policy`` rather than
            wrapping the node with an ADK ``RetryConfig``, which would retry
            on top of Temporal's own activity retries.
    """
    from google.adk.workflow import FunctionNode

    sig = inspect.signature(activity_def)
    if any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in sig.parameters.values()
    ):
        raise TypeError(
            f"Activity '{activity_def.__name__}' has *args or **kwargs; activity_node"
            " needs named parameters to bind node_input."
        )
    # Activities are invoked positionally; activity.defn rejects keyword-only parameters.
    params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    # Resolve hints against the activity's own module so that string
    # annotations (e.g. under `from __future__ import annotations`) still
    # resolve once copied onto the wrapper, whose globals differ.
    try:
        resolved_hints = typing.get_type_hints(activity_def)
    except Exception:
        resolved_hints = {}

    async def wrapper(node_input: Any = None) -> Any:
        options = kwargs.copy()

        if not params:
            activity_args: list[Any] = []
        elif len(params) == 1:
            activity_args = [node_input]
        else:
            if not isinstance(node_input, dict):
                raise TypeError(
                    f"Activity node '{activity_def.__name__}' takes"
                    f" {len(params)} parameters, so its input must be a dict"
                    f" of parameter names to values, got"
                    f" {type(node_input).__name__}."
                )
            activity_args = _bind_node_input(activity_def, params, node_input)

        if not temporalio.workflow.in_workflow():
            # Outside a workflow, like local adk runs, use the function directly.
            result = activity_def(*activity_args)
            if inspect.isawaitable(result):
                return await result
            return result

        if not activity_args:
            return await workflow.execute_activity(activity_def, **options)
        if len(activity_args) == 1:
            return await workflow.execute_activity(
                activity_def, activity_args[0], **options
            )
        return await workflow.execute_activity(
            activity_def, args=activity_args, **options
        )

    # ADK's FunctionNode binds parameters from the wrapper's signature and
    # type hints: a single `node_input` parameter receives the previous
    # node's output directly. Set metadata explicitly rather than via
    # functools.wraps: copying the activity's multi-parameter signature (or a
    # `__wrapped__` link, which ADK's type-hint resolution follows) would make
    # ADK bind the activity's own parameters from workflow state instead.
    wrapper.__name__ = name or activity_def.__name__
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = activity_def.__doc__
    input_annotation = (
        resolved_hints.get(params[0].name, Any) if len(params) == 1 else Any
    )
    return_annotation = resolved_hints.get("return", Any)
    wrapper.__annotations__ = {
        "node_input": input_annotation,
        "return": return_annotation,
    }
    setattr(
        wrapper,
        "__signature__",
        inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "node_input",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=input_annotation,
                )
            ],
            return_annotation=return_annotation,
        ),
    )

    return FunctionNode(
        func=wrapper, name=wrapper.__name__, rerun_on_resume=rerun_on_resume
    )
