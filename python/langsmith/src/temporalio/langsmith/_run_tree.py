"""Make native LangSmith runs deterministic and nonblocking in workflows."""

from __future__ import annotations

import functools
import logging
import random
import types
import uuid
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, tzinfo
from typing import Any

import langchain_core.callbacks.manager
import langchain_core.tracers.core
import langchain_core.tracers.langchain
import langsmith.client
from langsmith.run_trees import RunTree
from langsmith.uuid import uuid7

import temporalio.workflow
from temporalio.langsmith._signature import bind, require, signature

logger = logging.getLogger(__name__)
_executor: ContextVar[ThreadPoolExecutor | None] = ContextVar(
    "temporal_langsmith_executor", default=None
)
_client: ContextVar[langsmith.Client | None] = ContextVar(
    "temporal_langsmith_client", default=None
)
_installed = False


def _get_workflow_random() -> random.Random | None:
    """Return this workflow's independent random stream, or None in read-only or non-workflow code."""
    if (
        not temporalio.workflow.in_workflow()
        or temporalio.workflow.unsafe.is_read_only()
    ):
        return None
    instance = temporalio.workflow.instance()
    rng = getattr(instance, "__temporal_langsmith_random", None)
    if rng is None:
        rng = temporalio.workflow.new_random("temporalio.langsmith")
        setattr(instance, "__temporal_langsmith_random", rng)
    return rng


def _workflow_executor() -> ThreadPoolExecutor | None:
    """Limit runtime changes to workflow code running under this plugin."""
    executor = _executor.get()
    return (
        executor if executor is not None and temporalio.workflow.in_workflow() else None
    )


@contextmanager
def workflow_run_tree_context(
    executor: ThreadPoolExecutor,
    client: langsmith.Client | None = None,
) -> Generator[None, None, None]:
    """Bind native run creation and I/O to the current workflow handler."""
    token = _executor.set(executor)
    client_token = _client.set(client)
    try:
        yield
    finally:
        _client.reset(client_token)
        _executor.reset(token)


def run_id() -> uuid.UUID:
    """Deterministic IDs in workflow code, and LangSmith's random IDs everywhere else."""
    rng = _get_workflow_random()
    if rng is None:
        return uuid7()

    # v7 layout: 48-bit time, version 7, 12 random bits, variant 0b10, 62 random bits.
    # Built by hand because uuid.UUID(version=7) needs Python 3.14.
    ms = int(temporalio.workflow.now().timestamp() * 1000)
    return uuid.UUID(
        int=ms << 80
        | 0x7 << 76
        | rng.getrandbits(12) << 64
        | 0x2 << 62
        | rng.getrandbits(62)
    )


def worker_or_server_time(tz: tzinfo | None = timezone.utc) -> datetime:
    """Worker clock live, and the Server's clock during replays."""
    if (
        temporalio.workflow.in_workflow()
        and temporalio.workflow.unsafe.is_replaying_history_events()
    ):
        return temporalio.workflow.now()

    return datetime.now(tz)


def _deterministic_defaults(kwargs: dict[str, Any], id_key: str) -> None:
    """Supply workflow IDs and time without changing explicit caller values."""
    if kwargs.get(id_key) is None:
        kwargs[id_key] = run_id()
    if kwargs.get("start_time") is None:
        kwargs["start_time"] = worker_or_server_time()


def _submit(
    executor: ThreadPoolExecutor, fn: Callable[..., None], /, *args: Any, **kwargs: Any
) -> None:
    """Keep LangSmith I/O off the workflow thread, in submission order."""

    def log_exception(future: Future[None]) -> None:
        if (exc := future.exception()) is not None:
            logger.error("LangSmith background I/O error: %s", exc)

    executor.submit(fn, *args, **kwargs).add_done_callback(log_exception)


def install_run_tree_overrides() -> None:
    """Adapt native RunTree lifecycle methods without changing trace setup.

    Patching the class also covers aliases imported before plugin registration.
    LangSmith retains its own root selection, destination/replica resolution,
    child creation, and serialization. Outside this plugin's workflow context,
    every method delegates directly to the original implementation.
    """
    global _installed  # noqa: PLW0603
    if _installed:
        return

    original_init = RunTree.__init__
    original_create_child = RunTree.create_child
    original_post = RunTree.post
    original_end = RunTree.end
    original_patch = RunTree.patch
    original_ensure_uuid = langsmith.client._ensure_uuid
    require(RunTree, "field", RunTree.model_fields, "id", "start_time", "ls_client")
    create_child_signature = signature(original_create_child, "run_id", "start_time")
    end_signature = signature(original_end, "end_time")
    ensure_uuid_signature = signature(original_ensure_uuid, "value", "accept_null")

    @functools.wraps(original_init)
    def init(*args: Any, **kwargs: Any) -> None:
        if _workflow_executor() is not None:
            _deterministic_defaults(kwargs, "id")
            # A nested tracing_context() may clear its ambient client. Keep
            # the worker-local client as a fallback, preserving explicit clients.
            if all(
                kwargs.get(key) is None for key in ("client", "ls_client", "_client")
            ):
                kwargs["client"] = _client.get()
        original_init(*args, **kwargs)

    @functools.wraps(original_create_child)
    def create_child(*args: Any, **kwargs: Any) -> RunTree:
        if _workflow_executor() is not None and (
            call := bind(create_child_signature, args, kwargs)
        ):
            # LangSmith resolves child defaults before constructing RunTree.
            _deterministic_defaults(call.arguments, "run_id")
            args, kwargs = call.args, call.kwargs
        return original_create_child(*args, **kwargs)

    @functools.wraps(original_post)
    def post(*args: Any, **kwargs: Any) -> None:
        executor = _workflow_executor()
        if executor is None:
            original_post(*args, **kwargs)
        elif not temporalio.workflow.unsafe.is_replaying_history_events():
            _submit(executor, original_post, *args, **kwargs)

    @functools.wraps(original_end)
    def end(*args: Any, **kwargs: Any) -> None:
        if _workflow_executor() is not None and (
            call := bind(end_signature, args, kwargs)
        ):
            if call.arguments["end_time"] is None:
                call.arguments["end_time"] = worker_or_server_time()
            args, kwargs = call.args, call.kwargs
        # Ending only changes the in-memory tree, including on replay.
        original_end(*args, **kwargs)

    @functools.wraps(original_patch)
    def patch(*args: Any, **kwargs: Any) -> None:
        executor = _workflow_executor()
        if executor is None:
            original_patch(*args, **kwargs)
        elif not temporalio.workflow.unsafe.is_replaying_history_events():
            _submit(executor, original_patch, *args, **kwargs)

    @functools.wraps(original_ensure_uuid)
    def ensure_uuid(*args: Any, **kwargs: Any) -> Any:
        # The trace() context manager generates its root ID before RunTree.__init__.
        if (
            (call := bind(ensure_uuid_signature, args, kwargs))
            and call.arguments["value"] is None
            and not call.arguments["accept_null"]
        ):
            return run_id()
        return original_ensure_uuid(*args, **kwargs)

    # LangChain passes its own run IDs and times into RunTree, so they must come
    # from the workflow where LangChain makes them.
    manager: Any = langchain_core.callbacks.manager
    if hasattr(manager, "uuid7"):  # langchain-core 0.3.83+ and 1.1.1+
        manager.uuid7 = run_id
    else:  # Older releases call uuid.uuid4() through the module.
        manager.uuid = types.SimpleNamespace(uuid4=run_id)

    RunTree.__init__ = init  # type: ignore[method-assign]
    RunTree.create_child = create_child  # type: ignore[method-assign]
    RunTree.post = post  # type: ignore[method-assign]
    RunTree.end = end  # type: ignore[method-assign]
    RunTree.patch = patch  # type: ignore[method-assign]
    langsmith.client._ensure_uuid = ensure_uuid
    # LangChain's tracer modules only call datetime.now().
    clock = types.SimpleNamespace(now=worker_or_server_time)
    langchain_core.tracers.core.datetime = clock  # type: ignore[misc,assignment]
    langchain_core.tracers.langchain.datetime = clock  # type: ignore[misc,assignment]
    _installed = True
