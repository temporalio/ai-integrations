"""Workflow-specific primitives for working with the OpenAI Agents SDK in a workflow context"""

import dataclasses
import functools
import inspect
import json
import sys
import typing
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import timedelta
from typing import Any

if sys.version_info >= (3, 14):
    import annotationlib  # type: ignore[reportUnreachable]

import nexusrpc
from agents import (
    ModelBehaviorError,
    RunContextWrapper,
    Tool,
    _debug,
    default_tool_error_function,
)
from agents.function_schema import FuncSchema, function_schema
from agents.tool import (
    FunctionTool,
    ToolErrorFunction,
)
from pydantic import BaseModel, ValidationError

from temporalio import activity
from temporalio import workflow as temporal_workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.exceptions import ApplicationError, TemporalError
from temporalio.openai_agents._errors import (
    AgentsWorkflowError as AgentsWorkflowError,
)
from temporalio.openai_agents.sandbox._temporal_sandbox_client import (
    TemporalSandboxClient,
)
from temporalio.workflow import (
    ActivityCancellationType,
    ActivityConfig,
    VersioningIntent,
)

if typing.TYPE_CHECKING:
    from agents.mcp import MCPServer
    from agents.mcp.server import RequireApprovalSetting


def _parse_tool_arguments(schema: FuncSchema, input: str | None) -> BaseModel:
    """Parse and validate model-supplied tool arguments.

    Mirrors the Agents SDK's ``function_tool``, including leaving the input and
    validation details out of the error unless ``DONT_LOG_TOOL_DATA`` is disabled.

    Raises:
        ModelBehaviorError: If the input is not a JSON object matching the tool's parameters.
    """
    message = f"Invalid JSON input for tool {schema.name}"
    json_data: Any = {}
    json_error: json.JSONDecodeError | None = None
    try:
        if input:
            json_data = json.loads(input)
    except json.JSONDecodeError as e:
        json_error = e
    # Raised outside the except blocks so redacted errors do not chain the input
    if json_error is not None:
        if _debug.DONT_LOG_TOOL_DATA:
            raise ModelBehaviorError(message)
        raise ModelBehaviorError(f"{message}: {input}") from json_error
    if not isinstance(json_data, dict):
        raise ModelBehaviorError(f"{message}: expected a JSON object")
    validation_error: ValidationError | None = None
    try:
        return schema.params_pydantic_model(**json_data)
    except ValidationError as e:
        validation_error = e
    if _debug.DONT_LOG_TOOL_DATA:
        raise ModelBehaviorError(message)
    raise ModelBehaviorError(f"{message}: {validation_error}") from validation_error


async def _tool_argument_error(
    failure_error_function: ToolErrorFunction | None,
    ctx: RunContextWrapper[Any],
    error: ModelBehaviorError,
) -> str:
    """Format an argument error as the tool output returned to the model.

    Raises:
        ApplicationError: Non-retryable, if ``failure_error_function`` is None.
            Unlike ``ModelBehaviorError``, it fails the workflow instead of
            retrying the workflow task.
    """
    if failure_error_function is None:
        raise ApplicationError(
            str(error), type="ModelBehaviorError", non_retryable=True
        ) from error
    result = failure_error_function(ctx, error)
    if inspect.isawaitable(result):
        result = await result
    return result


def _invalid_tool(message: str) -> ApplicationError:
    """Return the error for an activity that can't be made into a tool."""
    return ApplicationError(message, type="invalid_tool")


def _call_function(method: Any) -> Callable | None:
    """Return the function under a function, staticmethod, classmethod or partialmethod."""
    while isinstance(method, (staticmethod, classmethod, functools.partialmethod)):
        method = (
            method.func
            if isinstance(method, functools.partialmethod)
            else method.__func__
        )
    return method if inspect.isfunction(method) else None


def _generated_class_doc(cls: type) -> str | None:
    """Return the docstring dataclasses or namedtuple writes for a class that has none."""
    if dataclasses.is_dataclass(cls):
        try:
            # Format it as dataclasses did when it made the class: 3.14 leaves forward
            # references unevaluated, so use each field's type from that time.
            if sys.version_info >= (3, 14):
                signature = inspect.signature(  # type: ignore[reportUnreachable]
                    cls, annotation_format=annotationlib.Format.FORWARDREF
                )
            else:
                signature = inspect.signature(cls)  # type: ignore[reportUnreachable]
            types = {name: f.type for name, f in cls.__dataclass_fields__.items()}
            signature = signature.replace(
                parameters=[
                    p.replace(annotation=types.get(p.name, p.annotation))
                    for p in signature.parameters.values()
                ]
            )
            return cls.__name__ + str(signature).replace(" -> None", "")
        except (TypeError, ValueError):
            return cls.__name__
    fields = getattr(cls, "_fields", None)
    if issubclass(cls, tuple) and isinstance(fields, tuple):
        names = ", ".join(fields) + ("," if len(fields) == 1 else "")
        return f"{cls.__name__}({names})"
    return None


def _docstring(fn: Callable | None) -> str | None:
    """Return ``fn``'s docstring, unless it is missing or blank."""
    doc = fn.__doc__ if fn is not None else None
    return doc if doc and doc.strip() else None


def _class_doc(cls: type) -> str | None:
    """Return a class's own docstring, unless dataclasses or namedtuple wrote it."""
    doc = _docstring(cls)
    if doc is None or not doc.startswith(cls.__name__):
        return doc
    try:
        generated = _generated_class_doc(cls)
    except Exception:
        # dataclasses would have failed the same way, so it didn't write this docstring.
        return doc
    return None if doc == generated else doc


def _tool_stand_in(
    call: Callable, name: str, hints: dict[str, Any], doc: str | None
) -> Callable:
    """Wrap ``call`` with the name, docstring and annotations function_schema should read.

    A partial has no module globals, so ``hints`` are resolved in the function's own module.
    """
    stand_in = functools.partial(call)
    setattr(stand_in, "__name__", name)
    stand_in.__annotations__ = hints
    stand_in.__doc__ = doc
    return stand_in


def activity_as_tool(
    fn: Callable,
    *,
    task_queue: str | None = None,
    schedule_to_close_timeout: timedelta | None = None,
    schedule_to_start_timeout: timedelta | None = None,
    start_to_close_timeout: timedelta | None = None,
    heartbeat_timeout: timedelta | None = None,
    retry_policy: RetryPolicy | None = None,
    cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL,
    activity_id: str | None = None,
    versioning_intent: VersioningIntent | None = None,
    summary: str | None = None,
    priority: Priority = Priority.default,
    strict_json_schema: bool = True,
    failure_error_function: ToolErrorFunction | None = default_tool_error_function,
) -> Tool:
    """Convert a single Temporal activity function to an OpenAI agent tool.

    This function takes a Temporal activity function and converts it into an
    OpenAI agent tool that can be used by the agent to execute the activity
    during workflow execution. The tool will automatically handle the conversion
    of inputs and outputs between the agent and the activity. Note that if you take a context,
    mutation will not be persisted, as the activity may not be running in the same location.

    For undocumented arguments, refer to :py:mod:`workflow` and :py:meth:`start_activity`

    Args:
        fn: A Temporal activity function to convert to a tool, a method passed as
            ``Class.method``, or a callable-class activity passed as its class. For a
            ``__call__`` set with ``functools.partialmethod``, the SDK runs an async one as
            sync before Python 3.13 and decodes its arguments without their types.
        strict_json_schema: Whether the tool should follow a strict schema.
            See https://openai.github.io/openai-agents-python/ref/tool/#agents.tool.FunctionTool.strict_json_schema
        failure_error_function: Formats the error text returned to the model
            when its arguments are not valid JSON or do not match the tool's
            parameters, so that it can retry. Defaults to the Agents SDK's
            ``default_tool_error_function``. If None, the workflow fails with a
            non-retryable ``ApplicationError`` of type ``ModelBehaviorError``.
            Errors raised by the activity itself are not handled by this
            function.

    Returns:
        An OpenAI agent tool that wraps the provided activity.

    Raises:
        ApplicationError: If the function is not properly decorated as a Temporal activity,
            or is a callable-class activity passed as an instance, or whose ``__call__`` is
            not a function, staticmethod, classmethod or ``functools.partialmethod``, can't
            be read, or takes keyword-only arguments or ``**kwargs``.

    Example:
        >>> @activity.defn
        >>> def process_data(input: str) -> str:
        ...     return f"Processed: {input}"
        >>>
        >>> # Create tool with custom activity options
        >>> tool = activity_as_tool(
        ...     process_data,
        ...     start_to_close_timeout=timedelta(seconds=30),
        ...     retry_policy=RetryPolicy(maximum_attempts=3),
        ...     heartbeat_timeout=timedelta(seconds=10)
        ... )
        >>> # Use tool with an OpenAI agent
    """
    ret = activity._Definition.from_callable(fn)
    if not ret:
        raise _invalid_tool(
            "Bare function without tool and activity decorators is not supported"
        )
    if ret.name is None:
        raise _invalid_tool("Input activity must have a name to be made into a tool")
    if not inspect.isclass(fn) and hasattr(type(fn), "__temporal_activity_definition"):
        raise _invalid_tool(
            f"Pass the activity class {type(fn).__name__}, not an instance of it"
        )
    if inspect.isclass(fn):
        # Workflow code references a callable-class activity by its class, but the worker's
        # registered instance runs __call__, so build the tool from __call__ as an instance binds
        # it: without the receiver of a method, classmethod or partialmethod, and with every
        # parameter of a staticmethod. The tool never calls this stand-in binding.
        calls = [vars(c)["__call__"] for c in fn.__mro__ if "__call__" in vars(c)]
        func = _call_function(calls[0]) if calls else None
        if func is None:
            raise _invalid_tool(
                f"Activity class {fn.__name__} needs a __call__ that is a function, "
                "staticmethod, classmethod or functools.partialmethod"
            )
        try:
            call = calls[0].__get__(object(), fn)
            hints = typing.get_type_hints(func, include_extras=True)
            parameters = inspect.signature(call).parameters.values()
        except Exception as e:
            raise _invalid_tool(
                f"Cannot build a tool from {fn.__name__}.__call__: {e}"
            ) from e
        # @activity.defn checks the constructor's signature, not __call__'s, so __call__ can
        # take keyword-only arguments, which run_activity can't pass.
        if any(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters):
            message = f"Activity class {fn.__name__} cannot take keyword-only arguments in __call__"
            if isinstance(calls[0], functools.partialmethod) and calls[0].keywords:
                message += (
                    ": its functools.partialmethod binds an argument by keyword, which makes"
                    " the arguments after it keyword-only"
                )
            raise _invalid_tool(message)
        # __call__'s own docstring, else the class's, else that of a __call__ it inherits.
        doc = (
            _docstring(_call_function(vars(fn).get("__call__")))
            or _class_doc(fn)
            or next(filter(None, map(_docstring, map(_call_function, calls))), None)
        )
        fn = _tool_stand_in(call, fn.__name__, hints, doc)
    else:
        # Drop the receiver of a method passed as Class.method the way the SDK does: a first
        # parameter named `self` that is unannotated or annotated `typing.Self`. The activity
        # execution picks up the actual instance, so this partial is never executed.
        first = next(iter(inspect.signature(fn).parameters.values()), None)
        if (
            first is not None
            and first.name == "self"
            and (
                first.annotation is inspect.Parameter.empty
                or str(first.annotation) == "typing.Self"
            )
        ):
            fn = _tool_stand_in(
                functools.partial(fn, None),
                fn.__name__,
                typing.get_type_hints(fn, include_extras=True),
                fn.__doc__,
            )
    # run_activity passes only positional arguments, so it would drop the model's **kwargs.
    if any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(fn).parameters.values()
    ):
        raise _invalid_tool(f"Activity {fn.__name__} cannot take **kwargs")
    schema = function_schema(fn)

    async def run_activity(ctx: RunContextWrapper[Any], input: str) -> Any:
        try:
            params = _parse_tool_arguments(schema, input)
        except ModelBehaviorError as e:
            return await _tool_argument_error(failure_error_function, ctx, e)

        # Activities don't support keyword only arguments, so we can ignore the kwargs_dict return
        args, _ = schema.to_call_args(params)

        # Add the context to the arguments if it takes that
        if schema.takes_context:
            args = [ctx] + args
        result = await temporal_workflow.execute_activity(
            ret.name,  # type: ignore
            args=args,
            task_queue=task_queue,
            schedule_to_close_timeout=schedule_to_close_timeout,
            schedule_to_start_timeout=schedule_to_start_timeout,
            start_to_close_timeout=start_to_close_timeout,
            heartbeat_timeout=heartbeat_timeout,
            retry_policy=retry_policy,
            cancellation_type=cancellation_type,
            activity_id=activity_id,
            versioning_intent=versioning_intent,
            summary=summary or schema.description,
            priority=priority,
        )
        try:
            return str(result)
        except Exception as e:
            raise ToolSerializationError(
                "You must return a string representation of the tool output, or something we can call str() on"
            ) from e

    return FunctionTool(
        name=schema.name,
        description=schema.description or "",
        params_json_schema=schema.params_json_schema,
        on_invoke_tool=run_activity,
        strict_json_schema=strict_json_schema,
    )


def nexus_operation_as_tool(
    operation: nexusrpc.Operation[Any, Any],
    *,
    service: type[Any],
    endpoint: str,
    schedule_to_close_timeout: timedelta | None = None,
    strict_json_schema: bool = True,
    failure_error_function: ToolErrorFunction | None = default_tool_error_function,
) -> Tool:
    """Convert a Nexus operation into an OpenAI agent tool.

    This function takes a Nexus operation and converts it into an
    OpenAI agent tool that can be used by the agent to execute the operation
    during workflow execution. The tool will automatically handle the conversion
    of inputs and outputs between the agent and the operation.

    Args:
        operation: A Nexus operation to convert into a tool.
        service: The Nexus service class that contains the operation.
        endpoint: The Nexus endpoint to use for the operation.
        strict_json_schema: Whether the tool should follow a strict schema
        failure_error_function: Formats the error text returned to the model
            when its arguments are not valid JSON or do not match the
            operation's input, so that it can retry. Defaults to the Agents
            SDK's ``default_tool_error_function``. If None, the workflow fails
            with a non-retryable ``ApplicationError`` of type
            ``ModelBehaviorError``. Errors raised by the operation itself are
            not handled by this function.

    Returns:
        An OpenAI agent tool that wraps the provided operation.

    Example:
        >>> @nexusrpc.service
        ... class WeatherService:
        ...     get_weather_object_nexus_operation: nexusrpc.Operation[WeatherInput, Weather]
        >>>
        >>> # Create tool with custom activity options
        >>> tool = nexus_operation_as_tool(
        ...     WeatherService.get_weather_object_nexus_operation,
        ...     service=WeatherService,
        ...     endpoint="weather-service",
        ... )
        >>> # Use tool with an OpenAI agent
    """

    def operation_callable(input: Any):  # type: ignore[reportUnusedParameter]
        raise NotImplementedError("This function definition is used as a type only")

    operation_callable.__annotations__ = {
        "input": operation.input_type,
        "return": operation.output_type,
    }
    operation_callable.__name__ = operation.name

    schema = function_schema(operation_callable)

    async def run_operation(ctx: RunContextWrapper[Any], input: str) -> Any:
        try:
            params = _parse_tool_arguments(schema, input)
        except ModelBehaviorError as e:
            return await _tool_argument_error(failure_error_function, ctx, e)

        nexus_client = temporal_workflow.create_nexus_client(
            service=service, endpoint=endpoint
        )
        args, _ = schema.to_call_args(params)
        assert len(args) == 1, "Nexus operations must have exactly one argument"
        [arg] = args
        result = await nexus_client.execute_operation(
            operation,
            arg,
            schedule_to_close_timeout=schedule_to_close_timeout,
        )
        try:
            return str(result)
        except Exception as e:
            raise ToolSerializationError(
                "You must return a string representation of the tool output, or something we can call str() on"
            ) from e

    return FunctionTool(
        name=schema.name,
        description=schema.description or "",
        params_json_schema=schema.params_json_schema,
        on_invoke_tool=run_operation,
        strict_json_schema=strict_json_schema,
    )


def temporal_sandbox_client(
    name: str,
    config: ActivityConfig | None = None,
) -> Any:
    """Create a sandbox client reference for use in a Temporal workflow ``RunConfig``.

    .. warning::
        This is experimental and may change in future versions.
        Use with caution in production environments.

    This returns a ``BaseSandboxClient`` that dispatches all sandbox operations
    as Temporal activities, targeting the ``SandboxClientProvider`` registered
    on the worker with the matching ``name``.

    Example::

        run_config = RunConfig(
            sandbox=SandboxRunConfig(
                client=temporal_sandbox_client("daytona"),
                options=DaytonaSandboxClientOptions(...),
            ),
        )

    Args:
        name: The name of the ``SandboxClientProvider`` registered on the
            worker.  Must match exactly.
        config: Optional activity configuration for controlling timeouts,
            retries, etc.  Defaults to a 5-minute ``start_to_close_timeout``.
    """
    return TemporalSandboxClient(name=name, config=config)


def stateless_mcp_server(
    name: str,
    config: ActivityConfig | None = None,
    cache_tools_list: bool = False,
    factory_argument: Any | None = None,
    require_approval: "RequireApprovalSetting" = None,
) -> "MCPServer":
    """A stateless MCP server implementation for Temporal workflows.

    .. deprecated:: 1.0.0
        Use :py:func:`temporal_mcp_server` instead.

    This uses a TemporalMCPServer of the same name registered with the OpenAIAgents plugin to implement
    durable MCP operations statelessly.

    This approach is suitable for simple use cases where connection overhead is acceptable
    and you don't need to maintain state between operations. It should be preferred to stateful when possible due to its
    superior durability guarantees.

    .. warning::
        Do not pass secrets, credentials, or API keys through ``factory_argument``. It is an
        activity argument, so it is recorded in workflow history and, without a payload codec,
        visible in the web UI. Resolve credentials worker-side inside the server factory
        instead.

    Args:
        name: A string name for the server. Should match that provided in the plugin.
        config: Optional activity configuration for MCP operation activities.
               Defaults to 1-minute start-to-close timeout.
        cache_tools_list: If true, the list of tools will be cached for the duration of the server
        factory_argument: Optional argument to be provided to the factory when producing an MCPServer.
               Must not contain secrets.
        require_approval: Optional OpenAI Agents approval policy for this server's tools, with the
               same values as ``MCPServerStdio(require_approval=...)``. A policy set on the
               factory's server is not used. A callable policy runs in the workflow and must be
               deterministic.
    """
    import warnings

    warnings.warn(
        "stateless_mcp_server() is deprecated; use temporal_mcp_server() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from temporalio.openai_agents._mcp import (
        _StatelessMCPServerReference,
    )

    return _StatelessMCPServerReference(
        name,
        config,
        cache_tools_list,
        factory_argument,
        require_approval=require_approval,
    )


def stateful_mcp_server(
    name: str,
    config: ActivityConfig | None = None,
    server_session_config: ActivityConfig | None = None,
    factory_argument: Any | None = None,
    require_approval: "RequireApprovalSetting" = None,
) -> AbstractAsyncContextManager["MCPServer"]:
    """A stateful MCP server implementation for Temporal workflows.

    .. deprecated:: 1.0.0
        Use :py:func:`temporal_mcp_server` instead.

    This wraps an MCP server to maintain a persistent connection throughout
    the workflow execution. It creates a dedicated worker that stays connected to
    the MCP server and processes operations on a dedicated task queue.

    This approach is more efficient for workflows that make multiple MCP calls,
    as it avoids connection overhead, but requires more resources to maintain
    the persistent connection and worker.

    The caller will have to handle cases where the dedicated worker fails, as Temporal is
    unable to seamlessly recreate any lost state in that case.

    .. warning::
        Do not pass secrets, credentials, or API keys through ``factory_argument``. It is an
        activity argument, so it is recorded in workflow history and, without a payload codec,
        visible in the web UI. Resolve credentials worker-side inside the server factory
        instead.

    Args:
        name: A string name for the server. Should match that provided in the plugin.
        config: Optional activity configuration for MCP operation activities.
               Defaults to 1-minute start-to-close and 30-second schedule-to-start timeouts.
        server_session_config: Optional activity configuration for the connection activity.
                       Defaults to 1-hour start-to-close timeout.
        factory_argument: Optional argument to be provided to the factory when producing an MCPServer.
               Must not contain secrets.
        require_approval: Optional OpenAI Agents approval policy for this server's tools, with the
               same values as ``MCPServerStdio(require_approval=...)``. A policy set on the
               factory's server is not used. A callable policy runs in the workflow and must be
               deterministic.
    """
    import warnings

    warnings.warn(
        "stateful_mcp_server() is deprecated; use temporal_mcp_server() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from temporalio.openai_agents._mcp import (
        _StatefulMCPServerReference,
    )

    return _StatefulMCPServerReference(
        name,
        config,
        server_session_config,
        factory_argument,
        require_approval=require_approval,
    )


def temporal_mcp_server(
    name: str,
    activity_config: ActivityConfig | None = None,
    cache_tools_list: bool = True,
    factory_argument: Any = None,
    **server_options: Any,
) -> "MCPServer":
    """Create a durable MCP v2 server proxy for an OpenAI agent in a workflow.

    MCP operations execute as Activities using the server factory registered as
    ``OpenAIAgentsPlugin(mcp_servers={name: factory})``. The tool list is cached
    in workflow state by default. Set ``cache_tools_list=False`` to list tools on
    every OpenAI Agents turn.

    .. warning::
        Do not pass secrets, credentials, or API keys through ``factory_argument``. It is an
        activity argument, so it is recorded in workflow history and, without a payload codec,
        visible in the web UI. Resolve credentials worker-side inside the server factory
        instead.

    .. warning::
        A callable ``tool_filter`` executes in the workflow and must be deterministic. Do not
        read external state, perform I/O, use the system clock, or use nondeterministic randomness
        from the filter.

    Args:
        name: Name matching a server factory registered on the plugin.
        activity_config: Activity options for MCP operations. Defaults to a
            one-minute start-to-close timeout.
        cache_tools_list: Whether to cache the tool list for this object.
        factory_argument: Non-secret value passed to the worker-side factory.
            A non-None value creates a fresh server for every Activity instead
            of using the worker connection cache.
        server_options: OpenAI Agents ``MCPServer`` options such as
            ``require_approval`` and ``failure_error_function``.
    """
    try:
        from temporalio.mcp import TemporalMCPClient
        from temporalio.openai_agents._temporal_mcp_server import (
            _TemporalMCPServer,
        )
    except ImportError as err:
        raise RuntimeError(
            "temporal_mcp_server() requires the 'mcp' extra and MCP Python SDK "
            "v2; install temporalio-openai-agents[mcp]"
        ) from err

    return _TemporalMCPServer(
        TemporalMCPClient(
            name,
            activity_config=activity_config,
            cache_tools_list=cache_tools_list,
            factory_argument=factory_argument,
        ),
        **server_options,
    )


class ToolSerializationError(TemporalError):
    """Error that occurs when a tool output could not be serialized.

    This exception is raised when a tool (created from an activity or Nexus operation)
    returns a value that cannot be properly serialized for use by the OpenAI agent.
    All tool outputs must be convertible to strings for the agent to process them.

    The error typically occurs when:
    - A tool returns a complex object that doesn't have a meaningful string representation
    - The returned object cannot be converted using str()
    - Custom serialization is needed but not implemented

    Example:
        >>> @activity.defn
        >>> def problematic_tool() -> ComplexObject:
        ...     return ComplexObject()  # This might cause ToolSerializationError

    To fix this error, ensure your tool returns string-convertible values or
    modify the tool to return a string representation of the result.
    """
