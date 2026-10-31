"""Make native LangSmith runs deterministic and nonblocking in workflows."""

from __future__ import annotations

import functools
import logging
import random
import uuid
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import langsmith.client
from langsmith.run_trees import RunTree

import temporalio.workflow

logger = logging.getLogger(__name__)
_executor: ContextVar[ThreadPoolExecutor | None] = ContextVar(
    "temporal_langsmith_executor", default=None
)
_client: ContextVar[langsmith.Client | None] = ContextVar(
    "temporal_langsmith_client", default=None
)
_installed = False


def _get_workflow_random() -> random.Random | None:
    """Return this workflow's independent random stream, except in read-only code."""
    if temporalio.workflow.unsafe.is_read_only():
        return None
    instance = temporalio.workflow.instance()
    rng = getattr(instance, "__temporal_langsmith_random", None)
    if rng is None:
        rng = temporalio.workflow.new_random()
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


def _deterministic_defaults(kwargs: dict[str, Any], id_key: str) -> None:
    """Supply workflow IDs and time without changing explicit caller values."""
    if kwargs.get(id_key) is None:
        rng = _get_workflow_random()
        if rng is not None:
            kwargs[id_key] = uuid.UUID(int=rng.getrandbits(128), version=4)
    if kwargs.get("start_time") is None:
        kwargs["start_time"] = temporalio.workflow.now()


def _submit(
    executor: ThreadPoolExecutor, fn: Callable[..., None], *args: Any, **kwargs: Any
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

    @functools.wraps(original_init)
    def init(self: RunTree, **kwargs: Any) -> None:
        if _workflow_executor() is not None:
            _deterministic_defaults(kwargs, "id")
            # A nested tracing_context() may clear its ambient client. Keep
            # the worker-local client as a fallback, preserving explicit clients.
            if all(
                kwargs.get(key) is None for key in ("client", "ls_client", "_client")
            ):
                kwargs["client"] = _client.get()
        original_init(self, **kwargs)

    @functools.wraps(original_create_child)
    def create_child(self: RunTree, *args: Any, **kwargs: Any) -> RunTree:
        if _workflow_executor() is not None:
            # LangSmith resolves child defaults before constructing RunTree.
            _deterministic_defaults(kwargs, "run_id")
        return original_create_child(self, *args, **kwargs)

    @functools.wraps(original_post)
    def post(self: RunTree, exclude_child_runs: bool = True) -> None:
        executor = _workflow_executor()
        if executor is None:
            original_post(self, exclude_child_runs=exclude_child_runs)
        elif not temporalio.workflow.unsafe.is_replaying_history_events():
            _submit(
                executor, original_post, self, exclude_child_runs=exclude_child_runs
            )

    @functools.wraps(original_end)
    def end(self: RunTree, **kwargs: Any) -> None:
        if _workflow_executor() is not None and kwargs.get("end_time") is None:
            kwargs["end_time"] = temporalio.workflow.now()
        # Ending only changes the in-memory tree, including on replay.
        original_end(self, **kwargs)

    @functools.wraps(original_patch)
    def patch(self: RunTree, *, exclude_inputs: bool | None = None) -> None:
        executor = _workflow_executor()
        if executor is None:
            original_patch(self, exclude_inputs=exclude_inputs)
        elif not temporalio.workflow.unsafe.is_replaying_history_events():
            _submit(executor, original_patch, self, exclude_inputs=exclude_inputs)

    @functools.wraps(original_ensure_uuid)
    def ensure_uuid(value: Any, *, accept_null: bool = False) -> Any:
        # The trace() context manager generates its root ID before RunTree.__init__.
        if value is None and not accept_null and _workflow_executor() is not None:
            rng = _get_workflow_random()
            if rng is not None:
                return uuid.UUID(int=rng.getrandbits(128), version=4)
        return original_ensure_uuid(value, accept_null=accept_null)

    RunTree.__init__ = init  # type: ignore[method-assign]
    RunTree.create_child = create_child  # type: ignore[method-assign]
    RunTree.post = post  # type: ignore[method-assign]
    RunTree.end = end  # type: ignore[method-assign]
    RunTree.patch = patch  # type: ignore[method-assign]
    langsmith.client._ensure_uuid = ensure_uuid
    _installed = True
