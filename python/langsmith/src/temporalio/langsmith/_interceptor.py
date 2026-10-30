"""LangSmith interceptor implementation for Temporal SDK."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from typing import Any, ClassVar, NoReturn, Protocol

import langsmith
import langsmith.utils
import nexusrpc.handler
from langsmith import tracing_context
from langsmith.run_helpers import get_current_run_tree, get_tracing_context
from langsmith.run_trees import RunTree, WriteReplica

import temporalio.activity
import temporalio.client
import temporalio.converter
import temporalio.worker
import temporalio.workflow
from temporalio.api.common.v1 import Payload
from temporalio.exceptions import ApplicationError, ApplicationErrorCategory


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


# Used by _log_future_exception, which runs on the executor thread, and by
# _end_run_on_exception outside workflows.  Workflow code logs through
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


def _inject_context(
    headers: Mapping[str, Payload],
    run_tree: RunTree,
) -> dict[str, Payload]:
    """Inject LangSmith context into Temporal payload headers.

    Serializes the run's trace context (trace ID, parent run ID, dotted order)
    into a Temporal header under ``_temporal-langsmith-context``, enabling parent-child
    trace nesting across process boundaries (client → worker, workflow → activity).
    """
    ls_headers = _run_headers(run_tree)
    return {
        **headers,
        HEADER_KEY: _payload_converter.to_payloads([ls_headers])[0],
    }


def _inject_current_context(
    headers: Mapping[str, Payload],
) -> Mapping[str, Payload]:
    """Inject the current ambient LangSmith context into Temporal payload headers.

    Reads ``_get_current_run_for_propagation()`` and injects if present. Returns
    headers unchanged if no context is active. Called unconditionally so that
    context propagation is independent of the ``add_temporal_runs`` toggle.
    """
    current = _get_current_run_for_propagation()
    if current is not None:
        return _inject_context(headers, current)
    return headers


def _extract_context(
    headers: Mapping[str, Payload],
    executor: ThreadPoolExecutor,
    ls_client: langsmith.Client,
) -> _ReplaySafeRunTree | None:
    """Extract LangSmith context from Temporal payload headers.

    Reconstructs a ``RunTree`` from the ``_temporal-langsmith-context`` header on
    the receiving side, wrapped in a :class:`_ReplaySafeRunTree` so inbound
    interceptors can establish a parent-child relationship with the sender's
    run. Returns ``None`` if no header is present.
    """
    header = headers.get(HEADER_KEY)
    if not header:
        return None
    ls_headers = _payload_converter.from_payloads([header])[0]
    run = RunTree.from_headers(ls_headers)
    if run is None:
        return None
    run.ls_client = ls_client
    return _ReplaySafeRunTree(run, executor=executor)


def _inject_nexus_context(
    headers: Mapping[str, str],
    run_tree: RunTree,
) -> dict[str, str]:
    """Inject LangSmith context into Nexus string headers."""
    ls_headers = _run_headers(run_tree)
    return {
        **headers,
        HEADER_KEY: json.dumps(ls_headers),
    }


def _extract_nexus_context(
    headers: Mapping[str, str],
    executor: ThreadPoolExecutor,
    ls_client: langsmith.Client,
) -> _ReplaySafeRunTree | None:
    """Extract LangSmith context from Nexus string headers."""
    raw = headers.get(HEADER_KEY)
    if not raw:
        return None
    ls_headers = json.loads(raw)
    run = RunTree.from_headers(ls_headers)
    if run is None:
        return None
    run.ls_client = ls_client
    return _ReplaySafeRunTree(run, executor=executor)


def _get_current_run_for_propagation() -> RunTree | None:
    """Get the current ambient run for context propagation.

    Filters out ``_RootReplaySafeRunTreeFactory``, which is internal
    scaffolding that should never be serialized into headers or used as
    parent runs.
    """
    run = get_current_run_tree()
    if isinstance(run, _RootReplaySafeRunTreeFactory):
        return None
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


# ---------------------------------------------------------------------------
# Replay safety
# ---------------------------------------------------------------------------


def _is_replaying() -> bool:
    """Check if we're currently replaying workflow history."""
    return (
        temporalio.workflow.in_workflow()
        and temporalio.workflow.unsafe.is_replaying_history_events()
    )


def _get_workflow_random() -> random.Random | None:
    """Get a deterministic random generator for the current workflow.

    Creates a workflow-safe random generator once via
    ``workflow.new_random()`` and stores it on the workflow instance so
    subsequent calls return the same generator.  The generator is seeded
    from the workflow's deterministic seed, so it produces identical UUIDs
    across replays and worker restarts.

    Returns ``None`` outside a workflow, in read-only (query) contexts, or
    when workflow APIs are mocked (unit tests).
    """
    try:
        if not temporalio.workflow.in_workflow():
            return None
        if temporalio.workflow.unsafe.is_read_only():
            return None
        inst = temporalio.workflow.instance()
        rng = getattr(inst, "__temporal_langsmith_random", None)
        if rng is None:
            rng = temporalio.workflow.new_random()
            setattr(inst, "__temporal_langsmith_random", rng)
        return rng
    except Exception:
        return None


def _uuid_from_random(rng: random.Random) -> uuid.UUID:
    """Generate a deterministic UUID4 from a workflow-bound random generator."""
    return uuid.UUID(int=rng.getrandbits(128), version=4)


# ---------------------------------------------------------------------------
# Replay safety for LangChain tracer runs
# ---------------------------------------------------------------------------


class _LangChainRuns:
    """Stable IDs for the runs LangChain's tracer makes in one workflow.

    LangChain gives each run a random ID and a wall-clock start time, so a
    replay would make the same runs again as new ones.  In workflow code,
    the IDs LangChain generates come from ``new_id``, a workflow-seeded
    generator (see ``_install_langchain_run_id_hook``), so callbacks and
    LangSmith see the same ID.  A run ID the caller chose, such as
    ``config["run_id"]``, is kept.  Every run's start time becomes
    ``workflow.now()``.  The generator is separate from
    ``_get_workflow_random`` so the plugin's own run IDs do not change.
    """

    def __init__(self) -> None:
        self._rng = random.Random()
        self._ids: dict[str, tuple[uuid.UUID, datetime]] = {}
        # Writes held back during replay, for runs that are still open.
        self.pending: dict[uuid.UUID, list[_HeldWrite]] = {}

        def reseed(seed: int) -> None:
            self._rng.seed(f"temporal-langsmith-langchain-{seed}")

        reseed(temporalio.workflow.random_seed())
        temporalio.workflow.register_random_seed_callback(reseed)

    def new_id(self, version: int) -> uuid.UUID:
        """Generate a run ID, a UUID of ``version`` 4 or 7, like LangChain's."""
        if version == 4:
            return _uuid_from_random(self._rng)
        # UUIDv7: 48 bits of Unix milliseconds, then version, variant and
        # random bits.  Python before 3.14 rejects version=7 in UUID().
        millis = int(temporalio.workflow.now().timestamp() * 1000)
        value = (millis & (1 << 48) - 1) << 80 | self._rng.getrandbits(80)
        value = value & ~(0xF << 76) | 0x7 << 76
        value = value & ~(0x3 << 62) | 0x2 << 62
        return uuid.UUID(int=value)

    def add(self, run_id: Any, *, keep_id: bool = False) -> uuid.UUID:
        """Give a run a stable ID and start time, once, and return the ID.

        With ``keep_id``, the run keeps its own ID.
        """
        entry = self._ids.get(str(run_id))
        if entry is None:
            stable = uuid.UUID(str(run_id)) if keep_id else _uuid_from_random(self._rng)
            entry = (stable, temporalio.workflow.now())
            self._ids[str(run_id)] = entry
        return entry[0]

    def stable_id(self, run_id: Any) -> Any:
        """Get a run's stable ID, or ``run_id`` for a run this class does not know."""
        entry = self._ids.get(str(run_id))
        return run_id if entry is None else entry[0]

    def dotted_order(self, dotted_order: str) -> str:
        """Rewrite each ``<start time><run ID>`` segment this class knows."""
        segments = []
        for segment in dotted_order.split("."):
            entry = self._ids.get(segment[-36:])
            if entry is not None:
                segment = entry[1].strftime("%Y%m%dT%H%M%S%fZ") + str(entry[0])
            segments.append(segment)
        return ".".join(segments)

    def relink(self, run: RunTree) -> None:
        """Point a run at the stable IDs of the runs above it."""
        run.parent_run_id = self.stable_id(run.parent_run_id)
        run.trace_id = self.stable_id(run.trace_id)
        run.dotted_order = self.dotted_order(run.dotted_order)

    def remap(self, kwargs: dict[str, Any], id_key: str) -> uuid.UUID | None:
        """Rewrite a run write's IDs in place and return the run's stable ID.

        Returns ``None``, leaving ``kwargs`` alone, for a run this class
        does not know.
        """
        entry = self._ids.get(str(kwargs.get(id_key)))
        if entry is None:
            return None
        for key in (id_key, "parent_run_id", "trace_id"):
            if kwargs.get(key) is not None:
                kwargs[key] = self.stable_id(kwargs[key])
        if kwargs.get("start_time") is not None:
            kwargs["start_time"] = entry[1]
        if kwargs.get("dotted_order"):
            kwargs["dotted_order"] = self.dotted_order(kwargs["dotted_order"])
        return entry[0]


def _get_langchain_runs() -> _LangChainRuns | None:
    """Get the current workflow's ``_LangChainRuns``.

    Returns ``None`` outside a workflow, in read-only (query) contexts, or
    when workflow APIs are mocked (unit tests).
    """
    try:
        if not temporalio.workflow.in_workflow():
            return None
        if temporalio.workflow.unsafe.is_read_only():
            return None
        inst = temporalio.workflow.instance()
        runs = getattr(inst, "__temporal_langsmith_langchain_runs", None)
        if runs is None:
            runs = _LangChainRuns()
            setattr(inst, "__temporal_langsmith_langchain_runs", runs)
        return runs
    except Exception:
        return None


_langchain_run_id_hook_installed = False


def _stable_run_ids(generate: Callable[..., Any], version: int) -> Callable[..., Any]:
    """Wrap a LangChain run ID generator to use ``_LangChainRuns.new_id``.

    It calls ``generate`` outside the tracing context this plugin sets for
    workflow code, such as in a workflow on a worker without the plugin.
    """

    def generate_run_id(*args: Any, **kwargs: Any) -> Any:
        runs = None
        if not (args or kwargs) and isinstance(
            get_tracing_context().get("client"), _ReplaySafeClient
        ):
            runs = _get_langchain_runs()
        return generate(*args, **kwargs) if runs is None else runs.new_id(version)

    return generate_run_id


class _UuidModule:
    """The ``uuid`` module, with another ``uuid4``."""

    def __init__(self, uuid4: Callable[..., Any]) -> None:
        self.uuid4 = uuid4

    def __getattr__(self, name: str) -> Any:
        return getattr(uuid, name)


def _install_langchain_run_id_hook() -> None:
    """Make the run IDs LangChain generates in workflow code stable.

    LangChain's callback manager generates an ID for each run the caller
    gave none.  In workflow code these come from ``_LangChainRuns.new_id``
    instead, so the IDs LangChain's callbacks get are the IDs LangSmith
    stores, and a replay generates them again.  A caller's own ``run_id``
    is kept.  Safe to call more than once.
    """
    global _langchain_run_id_hook_installed  # noqa: PLW0603
    if _langchain_run_id_hook_installed:
        return
    try:
        import langchain_core.callbacks.manager as manager
    except ImportError:
        return
    # The names the manager calls; neither is in its __all__.
    if hasattr(manager, "uuid7"):
        # langchain-core 0.3.83 to 0.3.86, and 1.1.1 on
        setattr(manager, "uuid7", _stable_run_ids(getattr(manager, "uuid7"), 7))
    elif getattr(manager, "uuid", None) is uuid:
        # Earlier releases call uuid.uuid4()
        setattr(manager, "uuid", _UuidModule(_stable_run_ids(uuid.uuid4, 4)))
    else:
        logger.warning(
            "This langchain-core generates run IDs in a way the LangSmith plugin "
            "does not know; LangChain runs in workflows may be posted again on "
            "replay"
        )
    _langchain_run_id_hook_installed = True


def _run_headers(run_tree: RunTree) -> dict[str, str]:
    """Get a run's LangSmith headers, with stable IDs in workflow code."""
    ls_headers = run_tree.to_headers()
    runs = _get_langchain_runs()
    if runs is not None and "langsmith-trace" in ls_headers:
        ls_headers["langsmith-trace"] = runs.dotted_order(ls_headers["langsmith-trace"])
    return ls_headers


# ("post" or "patch", the run dict ``Client.create_run`` or ``update_run``
# built, that method's auth keyword arguments), as in the client's run ops
# buffer; or, for a client that is not a ``langsmith.Client``, (method name,
# its keyword arguments, {})
_HeldWrite = tuple[str, dict[str, Any], dict[str, Any]]

_holding_client_types: dict[type, type] = {}


def _holding_client_type(cls: type) -> type:
    """Get a subclass of a ``langsmith.Client`` class that holds back writes.

    An instance wraps a client and reads its state.  Its ``create_run`` and
    ``update_run`` are the client's own, so they sample, redact with the
    client's callbacks and add the runtime environment as the client does,
    and raise what it raises.  Only the write after those steps, the tracing
    queue item, HTTP request or entry in the client's buffer for
    ``process_buffered_run_ops``, is held back, in ``held``.
    """
    holding = _holding_client_types.get(cls)
    if holding is None:

        def __init__(self: Any, client: langsmith.Client) -> None:  # noqa: N807
            object.__setattr__(self, "_wrapped", client)
            object.__setattr__(self, "held", [])
            # A buffer of its own, never flushed, takes the buffered writes.
            object.__setattr__(self, "_run_ops_buffer", self.held)
            object.__setattr__(self, "_run_ops_buffer_lock", threading.Lock())

        def __getattr__(self: Any, name: str) -> Any:  # noqa: N807
            return getattr(self.__dict__["_wrapped"], name)

        def _create_run(self: Any, run: dict[str, Any], **auth: Any) -> None:
            self.held.append(("post", run, auth))

        def _update_run(self: Any, run: dict[str, Any], **auth: Any) -> None:
            self.held.append(("patch", run, auth))

        def _should_flush_run_ops_buffer(_self: Any) -> bool:
            return False

        holding = type(
            f"_Holding{cls.__name__}",
            (cls,),
            {
                "__init__": __init__,
                "__getattr__": __getattr__,
                "_create_run": _create_run,
                "_update_run": _update_run,
                "_should_flush_run_ops_buffer": _should_flush_run_ops_buffer,
            },
        )
        _holding_client_types[cls] = holding
    return holding


def _write_held(client: Any, write: _HeldWrite) -> None:
    """Make a held write as the client would have made it.

    A client with ``process_buffered_run_ops`` gets the write in its buffer,
    so the callback processes it before it is sent.
    """
    op, run, auth = write
    if op not in ("post", "patch"):
        getattr(client, op)(**run)
    elif getattr(client, "_process_buffered_run_ops", None):
        with client._run_ops_buffer_lock:
            client._run_ops_buffer.append(write)
            if client._should_flush_run_ops_buffer():
                client._flush_run_ops_buffer()
    elif op == "post":
        client._create_run(run, **auth)
    else:
        client._update_run(run, **auth)


def _prepare_write(
    client: Any, method: str, kwargs: dict[str, Any]
) -> list[_HeldWrite]:
    """Run ``client.<method>(**kwargs)`` without writing, and get its writes.

    Errors from the client's steps propagate.  A client that is not a
    ``langsmith.Client``, such as a mock, has no such steps; its write is
    held back as the call itself.
    """
    if not issubclass(type(client), langsmith.Client):
        return [(method, kwargs, {})]
    holding = _holding_client_type(type(client))(client)
    getattr(holding, method)(**kwargs)
    return holding.held


class _ReplaySafeClient:
    """Wraps the plugin's LangSmith client for use in workflow code.

    LangChain's tracer writes its runs through this client, not through
    ``_ReplaySafeRunTree``.  In workflow code, this gives those runs stable
    IDs (see ``_LangChainRuns``) and holds back their writes while history
    replays.  A replayed callback still runs the client's own steps before
    its write, synchronously, so errors those steps raise, such as from a
    redaction callback, reach LangChain's tracer as in the original run.  At
    the first write after a replay, it makes the held writes of the replayed
    runs that are still open and that this client has not posted, as after
    a worker restart.  Other calls, and writes from other threads such as
    ``_ReplaySafeRunTree``'s background I/O, go to the wrapped client as is.
    """

    def __init__(self, client: langsmith.Client) -> None:
        """Wrap a LangSmith client."""
        self._client = client
        # Runs this client posted from workflow code that have not ended.
        self._open: set[uuid.UUID] = set()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped client."""
        return getattr(self._client, name)

    def create_run(self, *args: Any, **kwargs: Any) -> None:
        """Post a run, replay-safe in workflow code."""
        runs = _get_langchain_runs()
        if runs is None or args or kwargs.get("id") is None:
            return self._client.create_run(*args, **kwargs)
        run_id = runs.add(kwargs["id"], keep_id=True)
        runs.remap(kwargs, "id")
        ended = kwargs.get("end_time") is not None
        if _is_replaying():
            held = _prepare_write(self._client, "create_run", kwargs)
            # An empty list: the client's sampling dropped the run.
            if not ended and held:
                runs.pending[run_id] = held
            return None
        self._post_pending(runs)
        self._client.create_run(**kwargs)
        if not ended:
            self._open.add(run_id)
        return None

    def update_run(self, *args: Any, **kwargs: Any) -> None:
        """Update a run, replay-safe in workflow code."""
        runs = _get_langchain_runs()
        run_id = None if runs is None or args else runs.remap(kwargs, "run_id")
        if runs is None or run_id is None:
            return self._client.update_run(*args, **kwargs)
        ended = kwargs.get("end_time") is not None
        if _is_replaying():
            _prepare_write(self._client, "update_run", kwargs)
            if ended:
                runs.pending.pop(run_id, None)
            return None
        self._post_pending(runs)
        self._client.update_run(**kwargs)
        if ended:
            self._open.discard(run_id)
        return None

    def _post_pending(self, runs: _LangChainRuns) -> None:
        # The held writes already went through the client's synchronous
        # steps at their replayed callbacks; only a process_buffered_run_ops
        # callback, which runs on the client's buffer, processes them here.
        # No error can reach workflow code from here; log it.
        pending = list(runs.pending.items())
        runs.pending.clear()
        for run_id, held in pending:
            if run_id in self._open:
                continue
            self._open.add(run_id)
            for write in held:
                try:
                    _write_held(self._client, write)
                except Exception:
                    temporalio.workflow.logger.warning(
                        "Failed to post LangSmith run %s after replay",
                        run_id,
                        exc_info=True,
                    )


# ---------------------------------------------------------------------------
# _ReplaySafeRunTree wrapper
# ---------------------------------------------------------------------------


class _ReplaySafeRunTree(RunTree):
    """Wrapper around a ``RunTree`` with replay-safe ``post``, ``end``, and ``patch``.

    Inherits from ``RunTree`` so ``isinstance`` checks pass, but does
    **not** call ``super().__init__()``—the wrapped ``_run`` is the real
    RunTree.  Attribute access is delegated via ``__getattr__``/``__setattr__``.

    During replay, ``post()``, ``end()``, and ``patch()`` become no-ops
    (I/O suppression), but ``create_child()`` still runs to maintain
    parent-child linkage so ``@traceable``'s ``_setup_run`` can build the
    run tree across the replay boundary.  In workflow context, ``post()``
    and ``patch()`` submit to a single-worker ``ThreadPoolExecutor`` for
    FIFO ordering, avoiding blocking on the workflow task thread.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        run_tree: RunTree,
        *,
        executor: ThreadPoolExecutor,
    ) -> None:
        """Wrap an existing RunTree with replay-safe overrides."""
        object.__setattr__(self, "_run", run_tree)
        object.__setattr__(self, "_executor", executor)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped RunTree."""
        return getattr(self._run, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to the wrapped RunTree."""
        setattr(self._run, name, value)

    def to_headers(self) -> dict[str, str]:
        """Delegate to the wrapped RunTree's to_headers."""
        return self._run.to_headers()

    def _inject_deterministic_ids(self, kwargs: dict[str, Any]) -> None:
        """Inject deterministic run_id and start_time in workflow context."""
        if temporalio.workflow.in_workflow():
            if kwargs.get("run_id") is None:
                rng = _get_workflow_random()
                if rng is not None:
                    kwargs["run_id"] = _uuid_from_random(rng)
            if kwargs.get("start_time") is None:
                kwargs["start_time"] = temporalio.workflow.now()

    def create_child(self, *args: Any, **kwargs: Any) -> _ReplaySafeRunTree:
        """Create a child run, returning another _ReplaySafeRunTree.

        In workflow context, injects deterministic ``run_id`` and ``start_time``
        unless they are passed in manually via ``kwargs``.
        """
        self._inject_deterministic_ids(kwargs)
        child_run = self._run.create_child(*args, **kwargs)
        return _ReplaySafeRunTree(child_run, executor=self._executor)

    def _submit(self, fn: Callable[..., object], *args: Any, **kwargs: Any) -> None:
        """Submit work to the background executor."""

        def _log_future_exception(future: Future[None]) -> None:
            exc = future.exception()
            if exc is not None:
                logger.error("LangSmith background I/O error: %s", exc)

        future = self._executor.submit(fn, *args, **kwargs)
        future.add_done_callback(_log_future_exception)

    def post(self, exclude_child_runs: bool = True) -> None:
        """Post the run to LangSmith, skipping during replay."""
        if temporalio.workflow.in_workflow():
            if _is_replaying():
                return
            self._submit(self._run.post, exclude_child_runs=exclude_child_runs)
        else:
            self._run.post(exclude_child_runs=exclude_child_runs)

    def end(self, **kwargs: Any) -> None:
        """End the run, skipping during replay.

        Pre-computes ``end_time`` via ``workflow.now()`` in workflow context
        so ``RunTree.end()`` doesn't call ``datetime.now()`` (non-deterministic
        and sandbox-restricted).
        """
        if _is_replaying():
            return
        if temporalio.workflow.in_workflow():
            kwargs.setdefault("end_time", temporalio.workflow.now())
        self._run.end(**kwargs)

    def patch(self, *, exclude_inputs: bool | None = None) -> None:
        """Patch the run to LangSmith, skipping during replay.

        ``exclude_inputs`` is forwarded untouched so LangSmith's own default
        applies (a plain ``False`` before 0.11; the
        ``LANGSMITH_EXCLUDE_INPUTS_ON_PATCH`` setting from 0.11 on).
        """
        if temporalio.workflow.in_workflow():
            if _is_replaying():
                return
            self._submit(self._run.patch, exclude_inputs=exclude_inputs)
        else:
            self._run.patch(exclude_inputs=exclude_inputs)


class _RootReplaySafeRunTreeFactory(_ReplaySafeRunTree):
    """Factory that produces independent root ``_ReplaySafeRunTree`` instances with no parent link.

    When ``add_temporal_runs=False`` and no parent was propagated via headers,
    ``@traceable`` functions still need *something* in the LangSmith
    ``tracing_context`` to call ``create_child()`` on — otherwise they
    cannot create ``_ReplaySafeRunTree`` children at all and instead default to
    creating generic ``RunTree``s, which are not replay safe. This class fills
    that role: it sits in the context as the nominal parent so
    ``@traceable`` has a ``create_child()`` target.

    However, ``create_child()`` deliberately creates fresh ``RunTree``
    instances with **no** ``parent_run_id``.  This means every child appears
    as an independent root run in LangSmith rather than being nested under
    a phantom parent that was never meant to be visible.

    ``post()``, ``patch()``, and ``end()`` all raise ``RuntimeError``
    because this object is purely internal scaffolding — it must never
    appear in LangSmith.  If any of these methods are called, it indicates
    a programming error.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        ls_client: langsmith.Client,
        executor: ThreadPoolExecutor,
        destination: dict[str, Any],
        replicas: Sequence[WriteReplica] | None = None,
    ) -> None:
        """Create a root factory with the given LangSmith client."""
        # Create a minimal RunTree for the factory — it will never be posted
        factory_run = RunTree(
            name="__root_factory__",
            run_type="chain",
            ls_client=ls_client,
            **destination,
        )
        if replicas is not None:
            factory_run.replicas = replicas
        # LangChain's tracer makes the factory its runs' parent.
        runs = _get_langchain_runs()
        if runs is not None:
            runs.add(factory_run.id)
        object.__setattr__(self, "_run", factory_run)
        object.__setattr__(self, "_executor", executor)

    def post(self, exclude_child_runs: bool = True) -> NoReturn:
        """Factory must never be posted."""
        raise RuntimeError("_RootReplaySafeRunTreeFactory must never be posted")

    def patch(self, *, exclude_inputs: bool | None = None) -> NoReturn:
        """Factory must never be patched."""
        raise RuntimeError("_RootReplaySafeRunTreeFactory must never be patched")

    def end(self, **kwargs: Any) -> NoReturn:
        """Factory must never be ended."""
        raise RuntimeError("_RootReplaySafeRunTreeFactory must never be ended")

    def create_child(self, *args: Any, **kwargs: Any) -> _ReplaySafeRunTree:
        """Create a root _ReplaySafeRunTree (no parent_run_id).

        Creates a fresh ``RunTree(...)`` directly (bypassing
        ``self._run.create_child``) so children are independent root runs
        with no link back to the factory.
        """
        self._inject_deterministic_ids(kwargs)

        # RunTree expects "id", but callers pass "run_id". RunTree.create_child
        # also does the same mapping internally.
        if "run_id" in kwargs:
            kwargs["id"] = kwargs.pop("run_id")

        # Inherit ls_client and the destination from factory; both must be
        # passed at construction time.
        kwargs.setdefault("ls_client", self._run.ls_client)
        address = getattr(self._run, "address", None)
        if address is not None:
            kwargs.setdefault("address", address)
        else:
            kwargs.setdefault("session_name", self._run.session_name)

        child_run = RunTree(*args, **kwargs)
        # Replicas must be set post-construction
        if self._run.replicas is not None:
            child_run.replicas = self._run.replicas
        return _ReplaySafeRunTree(child_run, executor=self._executor)


# ---------------------------------------------------------------------------
# _maybe_run context manager
# ---------------------------------------------------------------------------


def _is_benign_error(exc: Exception) -> bool:
    """Check if an exception is a benign ApplicationError."""
    return (
        isinstance(exc, ApplicationError)
        and getattr(exc, "category", None) == ApplicationErrorCategory.BENIGN
    )


def _end_run_on_exception(run_tree: _ReplaySafeRunTree, **kwargs: Any) -> None:
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
    executor: ThreadPoolExecutor,
) -> Generator[None, None, None]:
    """Create a LangSmith run, handling errors.

    - If add_temporal_runs is False **or** ``langsmith.utils.tracing_is_enabled()``
      returns False, yields None (no run created).
      Context propagation is handled unconditionally by callers.
    - With no parent or ``destination`` (no plugin project and an env LangSmith
      can't address), yields None. In a workflow, tracing is also disabled for
      the body.
    - When a run IS created, uses :class:`_ReplaySafeRunTree` for
      replay and event loop safety, then sets it as ambient context via
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
        executor: ThreadPoolExecutor for background I/O.
    """
    if not add_temporal_runs or not langsmith.utils.tracing_is_enabled():
        yield None
        return

    # If no explicit parent, inherit from ambient @traceable context
    if parent is None:
        parent = _get_current_run_for_propagation()

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
    elif destination is not None:
        run_tree_args.update(destination)
    else:
        if temporalio.workflow.in_workflow():
            # Without a replay-safe root, workflow @traceable calls would build
            # plain RunTrees, so the workflow runs untraced.
            with tracing_context(enabled=False):
                yield None
        else:
            yield None
        return

    # Deterministic IDs so replayed workflows produce identical runs
    # instead of duplicates (see _get_workflow_random for details).
    rng = _get_workflow_random()
    # In read-only contexts (queries, update validators), _get_workflow_random()
    # returns None. Deterministic IDs aren't needed — these aren't replayed.
    # LangSmith will auto-generate a random UUID.
    if rng is not None:
        run_tree_args["id"] = _uuid_from_random(rng)
        run_tree_args["start_time"] = temporalio.workflow.now()
    if metadata:
        run_tree_args["extra"] = {"metadata": metadata}
    if tags:
        run_tree_args["tags"] = tags
    run = RunTree(**run_tree_args)
    # The parent may be one of LangChain's runs, which have stable IDs.
    runs = _get_langchain_runs()
    if runs is not None:
        runs.relink(run)
    run_tree = _ReplaySafeRunTree(run, executor=executor)
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
        add_temporal_runs: bool = False,
        default_metadata: dict[str, Any] | None = None,
        default_tags: list[str] | None = None,
    ) -> None:
        """Initialize the LangSmith interceptor with tracing configuration."""
        super().__init__()
        if client is None:
            client = langsmith.Client()
        self._client = client
        # Workflow code uses this wherever it would use a langsmith.Client.
        self._workflow_client: Any = _ReplaySafeClient(client)
        # LangSmith treats an empty project name as unset, so the plugin does too.
        self._project_name = project_name or None
        # The plugin project or env destination is a fallback for roots.
        # None when no project is set and the env is invalid.
        self._destination = (
            {"project_name": self._project_name}
            if self._project_name is not None
            else _env_destination(client)
        )
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
            self._workflow_client
            if temporalio.workflow.in_workflow()
            else self._client,
            name,
            add_temporal_runs=self._add_temporal_runs,
            run_type=run_type,
            metadata=metadata,
            tags=list(self._default_tags),
            parent=parent,
            executor=self._executor,
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
        _install_langchain_run_id_hook()
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
        parent = _extract_context(
            input.headers, self._config._executor, self._config._client
        )
        info = temporalio.activity.info()
        extra_metadata = {
            "temporalWorkflowID": info.workflow_id or "",
            "temporalRunID": info.workflow_run_id or "",
            "temporalActivityID": info.activity_id or "",
        }
        tracing_args: dict[str, Any] = {
            "client": self._config._client,
            "project_name": self._config._project_name,
            "parent": parent,
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
        client = self._config._workflow_client
        parent = (
            _extract_context(headers, self._config._executor, client)
            if headers
            else None
        )
        tracing_args: dict[str, Any] = {
            "client": client,
            "project_name": self._config._project_name,
            "parent": parent,
        }
        # @traceable calls need a replay-safe root here; without one they run untraced.
        if parent is None and not self._config._add_temporal_runs:
            if self._config._destination is None:
                tracing_args["enabled"] = False
            else:
                tracing_args["parent"] = _RootReplaySafeRunTreeFactory(
                    ls_client=client,
                    executor=self._config._executor,
                    destination=self._config._destination,
                )
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
        context_source = _get_current_run_for_propagation()
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
        current_run = _get_current_run_for_propagation()
        if current_run:
            input.headers = _inject_context(input.headers, current_run)
        super().continue_as_new(input)

    async def start_nexus_operation(
        self, input: temporalio.worker.StartNexusOperationInput[Any, Any]
    ) -> temporalio.workflow.NexusOperationHandle[Any]:
        context_source = _get_current_run_for_propagation()
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
        parent = _extract_nexus_context(
            input.ctx.headers, self._config._executor, self._config._client
        )
        tracing_args: dict[str, Any] = {
            "client": self._config._client,
            "project_name": self._config._project_name,
            "parent": parent,
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
        parent = _extract_nexus_context(
            input.ctx.headers, self._config._executor, self._config._client
        )
        tracing_args: dict[str, Any] = {
            "client": self._config._client,
            "project_name": self._config._project_name,
            "parent": parent,
        }
        with tracing_context(**tracing_args):
            with self._config.maybe_run(
                f"RunCancelNexusOperationHandler:{input.ctx.service}/{input.ctx.operation}",
                run_type="tool",
                parent=parent,
            ):
                return await self.next.execute_nexus_operation_cancel(input)
