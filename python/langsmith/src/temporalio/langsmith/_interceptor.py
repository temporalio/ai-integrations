"""LangSmith interceptor implementation for Temporal SDK."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, ClassVar, NoReturn, Protocol
from urllib.parse import quote

import langsmith
import langsmith.utils
import nexusrpc.handler
from langsmith import tracing_context
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree

import temporalio.activity
import temporalio.client
import temporalio.converter
import temporalio.worker
import temporalio.workflow
from temporalio.api.common.v1 import Payload
from temporalio.exceptions import ApplicationError, ApplicationErrorCategory
from temporalio.langsmith._run_tree import (
    install_run_tree_overrides,
    workflow_run_tree_context,
)


def _env_destination(client: langsmith.Client) -> dict[str, Any] | None:
    """Return the destination LangSmith's env defaults name, or None.

    The env defaults are ``LANGSMITH_PROJECT``, and on LangSmith versions with
    agent addresses, ``LANGSMITH_AGENT_ID`` and ``LANGSMITH_AGENT_ENVIRONMENT``.
    The probe is a LangSmith ``RunTree`` that names no destination, so the
    installed version's own rules pick one; it is never posted. LangSmith
    raises ``EnvAddressError``, a ``LangSmithUserError``, for an env it can't
    use, and then this returns None.
    """
    try:
        probe = RunTree(name="temporal-langsmith-destination-probe", ls_client=client)
    except langsmith.utils.LangSmithUserError:
        return None
    if probe.session_name:
        return {"project_name": probe.session_name}
    address = getattr(probe, "address", None)
    return {"address": address} if address is not None else None


# Used by _end_run_on_exception outside workflows. Workflow code logs through
# temporalio.workflow.logger instead — the sandbox blocks logging I/O here.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER_KEY = "_temporal-langsmith-context"

_BUILTIN_QUERIES: frozenset[str] = frozenset(
    {
        "__stack_trace",
        "__enhanced_stack_trace",
    }
)


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

_payload_converter = temporalio.converter.PayloadConverter.default


class _InputWithHeaders(Protocol):
    headers: Mapping[str, Payload]


def _trace_headers(run_tree: RunTree) -> dict[str, str]:
    """Include credential-free replica destinations omitted by LangSmith 0.14.4."""
    headers = run_tree.to_headers()
    baggage = headers.get("baggage", "")
    if any(item.startswith("langsmith-replicas=") for item in baggage.split(",")):
        return headers
    replicas = getattr(run_tree, "replicas", None)
    if replicas:
        from langsmith import run_trees

        # Use LangSmith's own allow-list/sanitizer: credentials, client objects,
        # and ingestion URLs must never enter persisted Temporal headers.
        filter_replica = getattr(run_trees, "_filter_replica_for_headers", None)
        if filter_replica is not None:
            safe = [filter_replica(replica) for replica in replicas]
            encoded = quote(json.dumps(safe, separators=(",", ":"), default=str))
            headers["baggage"] = ",".join(
                part for part in (baggage, f"langsmith-replicas={encoded}") if part
            )
    return headers


def _inject_context(
    headers: Mapping[str, Payload],
    run_tree: RunTree,
) -> dict[str, Payload]:
    """Inject LangSmith context into Temporal payload headers.

    Serializes the run's trace context (trace ID, parent run ID, dotted order)
    into a Temporal header under ``_temporal-langsmith-context``, enabling parent-child
    trace nesting across process boundaries (client → worker, workflow → activity).
    """
    ls_headers = _trace_headers(run_tree)
    return {
        **headers,
        HEADER_KEY: _payload_converter.to_payloads([ls_headers])[0],
    }


def _inject_current_context(
    headers: Mapping[str, Payload],
) -> Mapping[str, Payload]:
    """Inject the current ambient LangSmith context into Temporal payload headers.

    Reads ``get_current_run_tree()`` and injects if present. Returns
    headers unchanged if no context is active. Called unconditionally so that
    context propagation is independent of the ``add_temporal_runs`` toggle.
    """
    current = get_current_run_tree()
    if current is not None:
        return _inject_context(headers, current)
    return headers


def _extract_context(
    headers: Mapping[str, Payload],
    ls_client: langsmith.Client,
) -> RunTree | None:
    """Extract LangSmith context from Temporal payload headers.

    Reconstructs the sender's run as a parent for inbound tracing.
    Returns ``None`` if no header is present.
    """
    header = headers.get(HEADER_KEY)
    if not header:
        return None
    ls_headers = _payload_converter.from_payloads([header])[0]
    run = RunTree.from_headers(ls_headers)
    if run is None:
        return None
    # This parent is never posted. Match LangSmith's "now" default using
    # workflow time so child timestamp clamping stays deterministic.
    if temporalio.workflow.in_workflow():
        run.start_time = temporalio.workflow.now()
    run.ls_client = ls_client
    return run


def _inject_nexus_context(
    headers: Mapping[str, str],
    run_tree: RunTree,
) -> dict[str, str]:
    """Inject LangSmith context into Nexus string headers."""
    ls_headers = _trace_headers(run_tree)
    return {
        **headers,
        HEADER_KEY: json.dumps(ls_headers),
    }


def _extract_nexus_context(
    headers: Mapping[str, str],
    ls_client: langsmith.Client,
) -> RunTree | None:
    """Extract LangSmith context from Nexus string headers."""
    raw = headers.get(HEADER_KEY)
    if not raw:
        return None
    ls_headers = json.loads(raw)
    run = RunTree.from_headers(ls_headers)
    if run is None:
        return None
    run.ls_client = ls_client
    return run


# ---------------------------------------------------------------------------
# Workflow event loop safety: override @traceable's aio_to_thread
# ---------------------------------------------------------------------------

_aio_to_thread_override_installed = False


async def _temporal_aio_to_thread(
    default_aio_to_thread: Callable[..., Any],
    ctx: Any,
    func: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run LangSmith's ``aio_to_thread`` synchronously inside Temporal workflows.

    The ``@traceable`` decorator on async functions uses ``aio_to_thread()`` →
    ``loop.run_in_executor()`` for run setup/teardown.  The Temporal workflow
    event loop does not support ``run_in_executor``.  This override runs those
    functions synchronously on the workflow thread when inside a workflow,
    and delegates to the default implementation outside workflows.

    Registered via ``langsmith.set_runtime_overrides(aio_to_thread=...)``.
    """
    if not temporalio.workflow.in_workflow():
        return await default_aio_to_thread(ctx, func, *args, **kwargs)
    with temporalio.workflow.unsafe.sandbox_unrestricted():
        return ctx.run(func, *args, **kwargs)


def _install_aio_to_thread_override() -> None:
    """Install the ``aio_to_thread`` override via LangSmith's official API.

    Safe to call multiple times; the override is only installed once.
    """
    global _aio_to_thread_override_installed  # noqa: PLW0603
    if _aio_to_thread_override_installed:
        return
    langsmith.set_runtime_overrides(aio_to_thread=_temporal_aio_to_thread)
    _aio_to_thread_override_installed = True


def _destination(
    project_name: str | None,
    address: str | None,
    parent: RunTree | None = None,
) -> dict[str, Any]:
    """Inherit the upstream destination before applying local defaults."""
    if parent is not None:
        parent_address = getattr(parent, "address", None)
        if isinstance(parent_address, str):
            return {"address": parent_address}
        if isinstance(parent.session_name, str):
            return {"project_name": parent.session_name}
    if address is not None:
        return {"address": address}
    if project_name is not None:
        return {"project_name": project_name}
    return {}


# ---------------------------------------------------------------------------
# _maybe_run context manager
# ---------------------------------------------------------------------------


def _is_benign_error(exc: Exception) -> bool:
    """Check if an exception is a benign ApplicationError."""
    return (
        isinstance(exc, ApplicationError)
        and getattr(exc, "category", None) == ApplicationErrorCategory.BENIGN
    )


def _end_run_on_exception(run_tree: RunTree, **kwargs: Any) -> None:
    """End and patch a run while an exception unwinds, without masking it."""
    try:
        run_tree.end(**kwargs)
        run_tree.patch()
    except Exception as exc:
        try:
            log = (
                temporalio.workflow.logger
                if temporalio.workflow.in_workflow()
                else logger
            )
            log.error("LangSmith run finalization error: %s", exc)
        except Exception:
            # A failing log handler (e.g. one restricted by the workflow
            # sandbox) must not replace the exception being raised either.
            pass


@contextmanager
def _maybe_run(
    client: langsmith.Client,
    name: str,
    *,
    add_temporal_runs: bool,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    parent: RunTree | None = None,
    destination: dict[str, Any] | None,
) -> Generator[None, None, None]:
    """Create a LangSmith run, handling errors.

    - If add_temporal_runs is False **or** ``langsmith.utils.tracing_is_enabled()``
      returns False, yields None (no run created).
      Context propagation is handled unconditionally by callers.
    - With no parent or ``destination`` (no plugin project and an env LangSmith
      can't address), yields None. In a workflow, tracing is also disabled for
      the body.
    - When a run IS created, sets it as ambient context via
      ``tracing_context(parent=run_tree)`` so ``get_current_run_tree()``
      returns it and ``_inject_current_context()`` can inject it.
    - On exception: marks run as errored (unless benign ApplicationError), re-raises.
    - On continue-as-new or cancellation: ends the run without an error, re-raises.

    Note on ``tracing_is_enabled()`` and cross-process traces:
    ``tracing_is_enabled()`` checks for an active run tree in context
    *before* consulting the ``LANGSMITH_TRACING`` env var (langsmith
    semantics).  If a parent run is propagated into this worker via
    headers from an upstream tracer, tracing continues regardless of
    ``LANGSMITH_TRACING=false``.  This matches langsmith's "continue
    mid-trace" model: the env var suppresses *new* local traces but
    does not break an inbound parent trace.

    Args:
        client: LangSmith client instance.
        name: Display name for the run.
        add_temporal_runs: Whether to create Temporal-level trace runs.
        run_type: LangSmith run type (default ``"chain"``).
        inputs: Input data to record on the run.
        metadata: Extra metadata to attach to the run.
        tags: Tags to attach to the run.
        parent: Parent run for nesting.
        destination: Fallback project or address for runs with no parent,
            resolved once at startup.
    """
    if not add_temporal_runs or not langsmith.utils.tracing_is_enabled():
        yield None
        return

    # If no explicit parent, inherit from ambient @traceable context
    if parent is None:
        parent = get_current_run_tree()

    run_tree_args: dict[str, Any] = dict(
        name=name,
        run_type=run_type,
        inputs=inputs or {},
        ls_client=client,
    )
    if parent is not None:
        # RunTree(parent_run=...) inherits IDs but not the destination.
        run_tree_args["project_name"] = parent.session_name
        run_tree_args["address"] = getattr(parent, "address", None)
        run_tree_args["parent_run"] = parent
        run_tree_args["replicas"] = parent.replicas
    elif destination is not None:
        run_tree_args.update(destination)
    else:
        if temporalio.workflow.in_workflow():
            # Preserve startup policy for an invalid environment destination.
            with tracing_context(enabled=False):
                yield None
        else:
            yield None
        return

    if metadata:
        run_tree_args["extra"] = {"metadata": metadata}
    if tags:
        run_tree_args["tags"] = tags
    run_tree = RunTree(**run_tree_args)
    run_tree.post()
    try:
        with tracing_context(parent=run_tree, client=client):
            yield None
    except Exception as exc:
        if _is_benign_error(exc):
            _end_run_on_exception(run_tree)
        else:
            _end_run_on_exception(run_tree, error=f"{type(exc).__name__}: {exc}")
        raise
    except (temporalio.workflow.ContinueAsNewError, asyncio.CancelledError) as exc:
        # BaseExceptions that end the run without failing it.
        status = (
            "continued_as_new"
            if isinstance(exc, temporalio.workflow.ContinueAsNewError)
            else "cancelled"
        )
        _end_run_on_exception(run_tree, outputs={"status": status})
        raise
    else:
        run_tree.end(outputs={"status": "ok"})
        run_tree.patch()


# ---------------------------------------------------------------------------
# LangSmithInterceptor
# ---------------------------------------------------------------------------


class LangSmithInterceptor(
    temporalio.client.Interceptor, temporalio.worker.Interceptor
):
    """Interceptor that supports client and worker LangSmith run creation
    and context propagation.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.
    """

    def __init__(
        self,
        *,
        client: langsmith.Client | None = None,
        project_name: str | None = None,
        address: str | None = None,
        add_temporal_runs: bool = False,
        default_metadata: dict[str, Any] | None = None,
        default_tags: list[str] | None = None,
    ) -> None:
        """Initialize the LangSmith interceptor with tracing configuration."""
        super().__init__()
        if project_name is not None and address is not None:
            raise ValueError("project_name and address are mutually exclusive")
        if address is not None:
            address_api = getattr(langsmith, "address", None)
            parse_address = getattr(address_api, "parse", None)
            if not callable(parse_address):
                raise ValueError("Agent addressing requires langsmith>=0.14.4")
            address = str(parse_address(address))
        if client is None:
            client = langsmith.Client()
        self._client = client
        # LangSmith treats an empty project name as unset, so the plugin does too.
        self._project_name = project_name or None
        self._address = address
        # The plugin project/address or env destination is a fallback for roots.
        # None when no destination is set and the env is invalid.
        self._destination = _destination(
            self._project_name, self._address
        ) or _env_destination(client)
        if self._destination is None:
            logger.warning(
                "LangSmith can't name a destination from the environment, so "
                "the Temporal plugin traces nothing that starts without a "
                "parent run."
            )
        self._add_temporal_runs = add_temporal_runs
        self._default_metadata = default_metadata or {}
        self._default_tags = default_tags or []
        self._executor = ThreadPoolExecutor(max_workers=1)

    @contextmanager
    def maybe_run(
        self,
        name: str,
        *,
        run_type: str = "chain",
        parent: RunTree | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Generator[None, None, None]:
        """Create a LangSmith run with this interceptor's config already applied."""
        metadata = {**self._default_metadata, **(extra_metadata or {})}
        with _maybe_run(
            self._client,
            name,
            add_temporal_runs=self._add_temporal_runs,
            run_type=run_type,
            metadata=metadata,
            tags=list(self._default_tags),
            parent=parent,
            destination=self._destination,
        ) as run:
            yield run

    def intercept_client(
        self, next: temporalio.client.OutboundInterceptor
    ) -> temporalio.client.OutboundInterceptor:
        """Create a client outbound interceptor for LangSmith tracing."""
        return _LangSmithClientOutboundInterceptor(next, self)

    def intercept_activity(
        self, next: temporalio.worker.ActivityInboundInterceptor
    ) -> temporalio.worker.ActivityInboundInterceptor:
        """Create an activity inbound interceptor for LangSmith tracing."""
        return _LangSmithActivityInboundInterceptor(next, self)

    def workflow_interceptor_class(
        self, input: temporalio.worker.WorkflowInterceptorClassInput
    ) -> type[_LangSmithWorkflowInboundInterceptor]:
        """Return the workflow interceptor class with config bound."""
        _install_aio_to_thread_override()
        install_run_tree_overrides()
        config = self

        class InterceptorWithConfig(_LangSmithWorkflowInboundInterceptor):
            _config = config

        return InterceptorWithConfig

    def intercept_nexus_operation(
        self, next: temporalio.worker.NexusOperationInboundInterceptor
    ) -> temporalio.worker.NexusOperationInboundInterceptor:
        """Create a Nexus operation inbound interceptor for LangSmith tracing."""
        return _LangSmithNexusOperationInboundInterceptor(next, self)


# ---------------------------------------------------------------------------
# Client Outbound Interceptor
# ---------------------------------------------------------------------------


class _LangSmithClientOutboundInterceptor(temporalio.client.OutboundInterceptor):
    """Instruments all client-side calls with LangSmith runs."""

    def __init__(
        self,
        next: temporalio.client.OutboundInterceptor,
        config: LangSmithInterceptor,
    ) -> None:
        super().__init__(next)
        self._config = config

    @contextmanager
    def _traced_call(
        self, name: str, input: _InputWithHeaders
    ) -> Generator[None, None, None]:
        """Wrap a client call with a LangSmith run and inject context into headers."""
        with self._config.maybe_run(name):
            input.headers = _inject_current_context(input.headers)
            yield

    @contextmanager
    def _traced_start(
        self, name: str, input: _InputWithHeaders
    ) -> Generator[None, None, None]:
        """Wrap a start operation, injecting ambient parent context before creating the run.

        Unlike ``_traced_call``, this injects headers *before* ``maybe_run``
        so the downstream ``RunFoo`` becomes a sibling of ``StartFoo`` rather
        than a child.
        """
        input.headers = _inject_current_context(input.headers)
        with self._config.maybe_run(name):
            yield

    async def start_workflow(
        self, input: temporalio.client.StartWorkflowInput
    ) -> temporalio.client.WorkflowHandle[Any, Any]:
        prefix = "SignalWithStartWorkflow" if input.start_signal else "StartWorkflow"
        with self._traced_start(f"{prefix}:{input.workflow}", input):
            return await super().start_workflow(input)

    async def query_workflow(self, input: temporalio.client.QueryWorkflowInput) -> Any:
        with self._traced_call(f"QueryWorkflow:{input.query}", input):
            return await super().query_workflow(input)

    async def signal_workflow(
        self, input: temporalio.client.SignalWorkflowInput
    ) -> None:
        with self._traced_call(f"SignalWorkflow:{input.signal}", input):
            return await super().signal_workflow(input)

    async def start_workflow_update(
        self, input: temporalio.client.StartWorkflowUpdateInput
    ) -> temporalio.client.WorkflowUpdateHandle[Any]:
        with self._traced_call(f"StartWorkflowUpdate:{input.update}", input):
            return await super().start_workflow_update(input)

    async def start_update_with_start_workflow(
        self, input: temporalio.client.StartWorkflowUpdateWithStartInput
    ) -> temporalio.client.WorkflowUpdateHandle[Any]:
        input.start_workflow_input.headers = _inject_current_context(
            input.start_workflow_input.headers
        )
        input.update_workflow_input.headers = _inject_current_context(
            input.update_workflow_input.headers
        )
        with self._config.maybe_run(
            f"StartUpdateWithStartWorkflow:{input.start_workflow_input.workflow}",
        ):
            return await super().start_update_with_start_workflow(input)


# ---------------------------------------------------------------------------
# Activity Inbound Interceptor
# ---------------------------------------------------------------------------


class _LangSmithActivityInboundInterceptor(
    temporalio.worker.ActivityInboundInterceptor
):
    """Instruments activity execution with LangSmith runs."""

    def __init__(
        self,
        next: temporalio.worker.ActivityInboundInterceptor,
        config: LangSmithInterceptor,
    ) -> None:
        super().__init__(next)
        self._config = config

    async def execute_activity(
        self, input: temporalio.worker.ExecuteActivityInput
    ) -> Any:
        parent = _extract_context(input.headers, self._config._client)
        info = temporalio.activity.info()
        extra_metadata = {
            "temporalWorkflowID": info.workflow_id or "",
            "temporalRunID": info.workflow_run_id or "",
            "temporalActivityID": info.activity_id or "",
        }
        tracing_args: dict[str, Any] = {
            "client": self._config._client,
            "parent": parent,
            **_destination(self._config._project_name, self._config._address, parent),
        }
        with tracing_context(**tracing_args):
            with self._config.maybe_run(
                f"RunActivity:{info.activity_type}",
                run_type="tool",
                parent=parent,
                extra_metadata=extra_metadata,
            ):
                return await super().execute_activity(input)


# ---------------------------------------------------------------------------
# Workflow Inbound Interceptor
# ---------------------------------------------------------------------------


class _LangSmithWorkflowInboundInterceptor(
    temporalio.worker.WorkflowInboundInterceptor
):
    """Instruments workflow execution with LangSmith runs."""

    _config: ClassVar[LangSmithInterceptor]

    def init(self, outbound: temporalio.worker.WorkflowOutboundInterceptor) -> None:
        super().init(_LangSmithWorkflowOutboundInterceptor(outbound, self._config))

    @contextmanager
    def _workflow_maybe_run(
        self,
        name: str,
        headers: Mapping[str, Payload] | None = None,
    ) -> Generator[None, None, None]:
        """Workflow-specific run creation with metadata.

        Extracts parent from headers (if provided) and sets up
        ``tracing_context`` so ``@traceable`` functions called from workflow
        code can discover the parent and LangSmith client, independent of the
        ``add_temporal_runs`` toggle.
        """
        with workflow_run_tree_context(self._config._executor, self._config._client):
            parent = (
                _extract_context(headers, self._config._client) if headers else None
            )
            tracing_args: dict[str, Any] = {
                "client": self._config._client,
                "parent": parent,
                **_destination(
                    self._config._project_name, self._config._address, parent
                ),
            }
            if parent is None and self._config._destination is None:
                tracing_args["enabled"] = False
            info = temporalio.workflow.info()
            extra_metadata = {
                "temporalWorkflowID": info.workflow_id,
                "temporalRunID": info.run_id,
            }
            with tracing_context(**tracing_args):
                with self._config.maybe_run(
                    name,
                    parent=parent,
                    extra_metadata=extra_metadata,
                ) as run:
                    yield run

    async def execute_workflow(
        self, input: temporalio.worker.ExecuteWorkflowInput
    ) -> Any:
        wf_type = temporalio.workflow.info().workflow_type
        with self._workflow_maybe_run(
            f"RunWorkflow:{wf_type}",
            input.headers,
        ):
            return await super().execute_workflow(input)

    async def handle_signal(self, input: temporalio.worker.HandleSignalInput) -> None:
        with self._workflow_maybe_run(f"HandleSignal:{input.signal}", input.headers):
            return await super().handle_signal(input)

    async def handle_query(self, input: temporalio.worker.HandleQueryInput) -> Any:
        if input.query.startswith("__temporal") or input.query in _BUILTIN_QUERIES:
            return await super().handle_query(input)
        with self._workflow_maybe_run(f"HandleQuery:{input.query}", input.headers):
            return await super().handle_query(input)

    def handle_update_validator(
        self, input: temporalio.worker.HandleUpdateInput
    ) -> None:
        with self._workflow_maybe_run(f"ValidateUpdate:{input.update}", input.headers):
            return super().handle_update_validator(input)

    async def handle_update_handler(
        self, input: temporalio.worker.HandleUpdateInput
    ) -> Any:
        with self._workflow_maybe_run(f"HandleUpdate:{input.update}", input.headers):
            return await super().handle_update_handler(input)


# ---------------------------------------------------------------------------
# Workflow Outbound Interceptor
# ---------------------------------------------------------------------------


class _LangSmithWorkflowOutboundInterceptor(
    temporalio.worker.WorkflowOutboundInterceptor
):
    """Instruments all outbound calls from workflow code."""

    def __init__(
        self,
        next: temporalio.worker.WorkflowOutboundInterceptor,
        config: LangSmithInterceptor,
    ) -> None:
        super().__init__(next)
        self._config = config

    @contextmanager
    def _traced_outbound(
        self, name: str, input: _InputWithHeaders
    ) -> Generator[None, None, None]:
        """Outbound workflow run creation with context injection into input.headers.

        Uses ambient context so ``@traceable`` step functions that wrap
        outbound calls correctly parent the outbound run under themselves.
        """
        context_source = get_current_run_tree()
        with self._config.maybe_run(name):
            if context_source:
                input.headers = _inject_context(input.headers, context_source)
            yield None

    def start_activity(
        self, input: temporalio.worker.StartActivityInput
    ) -> temporalio.workflow.ActivityHandle[Any]:
        with self._traced_outbound(f"StartActivity:{input.activity}", input):
            return super().start_activity(input)

    def start_local_activity(
        self, input: temporalio.worker.StartLocalActivityInput
    ) -> temporalio.workflow.ActivityHandle[Any]:
        with self._traced_outbound(f"StartActivity:{input.activity}", input):
            return super().start_local_activity(input)

    async def start_child_workflow(
        self, input: temporalio.worker.StartChildWorkflowInput
    ) -> temporalio.workflow.ChildWorkflowHandle[Any, Any]:
        with self._traced_outbound(f"StartChildWorkflow:{input.workflow}", input):
            return await super().start_child_workflow(input)

    async def signal_child_workflow(
        self, input: temporalio.worker.SignalChildWorkflowInput
    ) -> None:
        with self._traced_outbound(f"SignalChildWorkflow:{input.signal}", input):
            return await super().signal_child_workflow(input)

    async def signal_external_workflow(
        self, input: temporalio.worker.SignalExternalWorkflowInput
    ) -> None:
        with self._traced_outbound(f"SignalExternalWorkflow:{input.signal}", input):
            return await super().signal_external_workflow(input)

    def continue_as_new(self, input: temporalio.worker.ContinueAsNewInput) -> NoReturn:
        # No trace created, but inject context from ambient run
        current_run = get_current_run_tree()
        if current_run:
            input.headers = _inject_context(input.headers, current_run)
        super().continue_as_new(input)

    async def start_nexus_operation(
        self, input: temporalio.worker.StartNexusOperationInput[Any, Any]
    ) -> temporalio.workflow.NexusOperationHandle[Any]:
        context_source = get_current_run_tree()
        with self._config.maybe_run(
            f"StartNexusOperation:{input.service}/{input.operation_name}",
        ):
            if context_source:
                input.headers = _inject_nexus_context(
                    input.headers or {}, context_source
                )
            return await super().start_nexus_operation(input)


# ---------------------------------------------------------------------------
# Nexus Operation Inbound Interceptor
# ---------------------------------------------------------------------------


class _LangSmithNexusOperationInboundInterceptor(
    temporalio.worker.NexusOperationInboundInterceptor
):
    """Instruments Nexus operations with LangSmith runs."""

    def __init__(
        self,
        next: temporalio.worker.NexusOperationInboundInterceptor,
        config: LangSmithInterceptor,
    ) -> None:
        super().__init__(next)
        self._config = config

    async def execute_nexus_operation_start(
        self, input: temporalio.worker.ExecuteNexusOperationStartInput
    ) -> (
        nexusrpc.handler.StartOperationResultSync[Any]
        | nexusrpc.handler.StartOperationResultAsync
    ):
        parent = _extract_nexus_context(input.ctx.headers, self._config._client)
        tracing_args: dict[str, Any] = {
            "client": self._config._client,
            "parent": parent,
            **_destination(self._config._project_name, self._config._address, parent),
        }
        with tracing_context(**tracing_args):
            with self._config.maybe_run(
                f"RunStartNexusOperationHandler:{input.ctx.service}/{input.ctx.operation}",
                run_type="tool",
                parent=parent,
            ):
                return await self.next.execute_nexus_operation_start(input)

    async def execute_nexus_operation_cancel(
        self, input: temporalio.worker.ExecuteNexusOperationCancelInput
    ) -> None:
        parent = _extract_nexus_context(input.ctx.headers, self._config._client)
        tracing_args: dict[str, Any] = {
            "client": self._config._client,
            "parent": parent,
            **_destination(self._config._project_name, self._config._address, parent),
        }
        with tracing_context(**tracing_args):
            with self._config.maybe_run(
                f"RunCancelNexusOperationHandler:{input.ctx.service}/{input.ctx.operation}",
                run_type="tool",
                parent=parent,
            ):
                return await self.next.execute_nexus_operation_cancel(input)
