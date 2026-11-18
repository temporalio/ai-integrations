"""Task result cache for continue-as-new support.

Caches task results by (module.qualname, args, kwargs) hash and by how many
identical calls the same caller made before in the run, so that previously
completed calls are not re-executed after a continue-as-new when the run makes
them again in the same order, while a call repeated within a run executes
again. The cache state is a plain dict that can travel through
workflow.continue_as_new().
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections import Counter, deque
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, time, timedelta
from enum import Enum
from hashlib import sha256
from heapq import heapify, heappop, heappush
from json import dumps
from numbers import Number
from pathlib import PurePath
from typing import Any
from uuid import UUID

from langchain_core.runnables.config import var_child_runnable_config
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel

_task_cache: ContextVar[dict[str, Any] | None] = ContextVar(
    "_temporal_task_cache", default=None
)
# Per key, how many places calls in this run have taken, and the places given
# back by calls that stored no result. Neither is part of the cache, so they
# do not travel through continue-as-new.
_key_uses: ContextVar[dict[str, int] | None] = ContextVar(
    "_temporal_task_key_uses", default=None
)
_free_places: ContextVar[dict[str, list[int]] | None] = ContextVar(
    "_temporal_task_free_places", default=None
)
# The call of a workflow-side node or task that the current code runs in (see
# call_scope()), or "" outside one.
_scope: ContextVar[str] = ContextVar("_temporal_task_scope", default="")
# The keys that the calls made in that call took, its nested calls' included.
_taken: ContextVar[list[str] | None] = ContextVar("_temporal_task_taken", default=None)
# The keys whose places the calls of the current entrypoint invocation gave
# back (see invocation_scope()), or None outside one.
_returned: ContextVar[list[str] | None] = ContextVar(
    "_temporal_task_returned", default=None
)
# Cache entries written in this run by calls made in a node or task call that
# then failed (see call_scope()). Not part of the cache either.
_stale: ContextVar[set[str] | None] = ContextVar("_temporal_task_stale", default=None)
# Per scope whose last call failed: the LangGraph task that call ran as, and
# the keys the calls made in it took, those of earlier attempts resumed from
# the checkpoint included (see call_scope()). Not part of the cache either.
_failed: ContextVar[dict[str, tuple[str | None, list[str]]] | None] = ContextVar(
    "_temporal_task_failed", default=None
)
# Per LangGraph task of an entrypoint invocation that failed or was cancelled:
# the places its finished calls held, which it gave back, and the keys its
# calls took (see invocation_scope()). Not part of the cache either.
_failed_invocations: ContextVar[dict[str, tuple[list[str], list[str]]] | None] = (
    ContextVar("_temporal_task_failed_invocations", default=None)
)
# Keyword arguments LangGraph injects into a node function: the plugin keeps
# these two (see _plugin.py). They are not the caller's input.
_INJECTED_KWARGS = frozenset({"config", "runtime"})


def set_task_cache(cache: dict[str, Any] | None) -> None:
    """Set the task result cache for the current context and restart call counting."""
    _task_cache.set(cache)
    _key_uses.set({})
    _free_places.set({})
    _stale.set(set())
    _failed.set({})
    _failed_invocations.set({})


def occurrence_key(key: str) -> str:
    """Return the cache key for this occurrence of a call in the current run.

    Calls are counted per scope (see :func:`call_scope`). The first call with
    ``key`` from the entrypoint or a top-level node keeps it, so caches
    written by earlier versions still match; calls made in a workflow-side
    node or task get ``"<key>@<scope>"``. The n-th repeat adds ``"#<n>"``, so
    it does not find an earlier call's result and runs again, while the same
    calls made in the same order after continue-as-new find theirs. A call
    takes the first place that a failed or interrupted call gave back (see
    :func:`release`).
    """
    scope = _scope.get()
    base = f"{key}@{scope}" if scope else key
    uses, free = _key_uses.get(), _free_places.get()
    if uses is None or free is None:
        return base
    if free.get(base):
        n = heappop(free[base])
    else:
        n = uses.get(base, 0)
        uses[base] = n + 1
    key = base if n == 0 else f"{base}#{n}"
    taken = _taken.get()
    if taken is not None:
        taken.append(key)
    return key


def release(key: str) -> None:
    """Give back the place of the call under ``key``: it stored no result.

    The call failed or was interrupted, so the next identical call in its
    scope, such as a retry or the resumed call, takes its place, and after
    continue-as-new the first call gets that call's result.
    """
    free = _free_places.get()
    if free is not None:
        base, sep, n = key.rpartition("#")
        heappush(free.setdefault(base if sep else key, []), int(n) if sep else 0)
        returned = _returned.get()
        if returned is not None:
            returned.append(key)


def _abandoned(err: BaseException) -> bool:
    """Return whether ``err`` ends a call for good.

    A failure or a cancellation does; an interrupt (``GraphBubbleUp``) does
    not, since LangGraph resumes the call.
    """
    return isinstance(err, (Exception, asyncio.CancelledError)) and not isinstance(
        err, GraphBubbleUp
    )


@contextmanager
def invocation_scope() -> Generator[None, None, None]:
    """Track the calls made by one invocation of an entrypoint.

    Calls made directly by an entrypoint keep their keys (see
    :func:`occurrence_key`). Without this, a Workflow that invokes the
    entrypoint again after a failed or cancelled invocation would make the
    calls that had succeeded under their next places, while their earlier
    results kept the first ones for the next run to find. So if the
    invocation fails or is cancelled, the results of its calls become stale
    and the places they still hold are given back: invoked again, it makes
    them under the same keys, runs them again and replaces their results.

    When LangGraph resumes the invocation from its checkpoint instead, as the
    same task, the calls it restores are not made again, so they take their
    places back and keep their results for the next run, and the call that
    failed takes the place it gave back. If that attempt fails too, the
    restored calls still count as the invocation's own.

    An entrypoint invoked inside another invocation, or inside a
    workflow-side node or task, is tracked the same way, so a failure the
    enclosing code catches is cleaned up too; its calls also count as the
    enclosing call's.
    """
    task = _langgraph_task()
    failed = _failed_invocations.get()
    restored: list[str] = []
    if failed is not None and task is not None and task in failed:
        held, restored = failed.pop(task)
        _reclaim(held)
    outer_taken, outer_returned = _taken.get(), _returned.get()
    taken: list[str] = list(restored)
    returned: list[str] = []
    taken_token = _taken.set(taken)
    returned_token = _returned.set(returned)
    try:
        yield
    except BaseException as err:
        if _abandoned(err):
            _retire(taken)
            # A place given back may have been taken again since, by this
            # invocation or another, so only the places still held go back.
            held = list((Counter(taken) - Counter(returned)).elements())
            for key in held:
                release(key)
            if failed is not None and task is not None:
                failed[task] = (held, taken)
        raise
    finally:
        _returned.reset(returned_token)
        _taken.reset(taken_token)
        if outer_taken is not None:
            # The enclosing call has the restored keys from the earlier attempt.
            outer_taken.extend(taken[len(restored) :])
        if outer_returned is not None:
            outer_returned.extend(returned)


@contextmanager
def call_scope(
    name: str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    """Count the calls made in this block per call of the workflow-side ``name``.

    The scope is a hash of the enclosing scope, ``name`` with the caller's
    ``args`` and ``kwargs`` (see :func:`_stable`), and its place among the
    enclosing scope's identical calls, taken before the call awaits
    anything. So it is the same in every run and on every Worker that makes
    the same calls in the same order, whatever order parallel calls finish
    in, and calls with other input never share one. If the block raises,
    its place is given back, as a failed call's is.

    If it fails or is cancelled, rather than interrupted, the results of the
    calls made in it become stale. When LangGraph runs it again from its
    checkpoint, as the same task, those calls keep their places: one
    LangGraph restores from the checkpoint is not made again and keeps its
    result for the next run, and the one that failed takes the place it gave
    back. If that attempt fails too, the restored calls still count as the
    call's own. When it is called again as another task, such as by a retry,
    its calls count from the first place again, so they are made under the
    same keys, run again and replace their results.
    """
    caller = {k: v for k, v in (kwargs or {}).items() if k not in _INJECTED_KWARGS}
    key = occurrence_key(_scope_key(name, args, caller))
    scope = sha256(key.encode()).hexdigest()[:16]
    task = _langgraph_task()
    failed = _failed.get()
    restored: list[str] = []
    if failed is not None and scope in failed:
        failed_task, failed_keys = failed.pop(scope)
        if task is None or task != failed_task:
            _restart(failed_keys)
        else:
            restored = failed_keys
    taken: list[str] = list(restored)
    token = _scope.set(scope)
    taken_token = _taken.set(taken)
    try:
        yield
    except BaseException as err:
        if _abandoned(err):
            _retire(taken)
            if failed is not None:
                failed[scope] = (task, taken)
        release(key)
        raise
    finally:
        _taken.reset(taken_token)
        _scope.reset(token)
        outer = _taken.get()
        if outer is not None:
            # The enclosing call has the restored keys from the earlier attempt.
            outer.extend(taken[len(restored) :])


def _langgraph_task() -> str | None:
    """Return the checkpoint namespace of the LangGraph task the code runs in.

    It names the task's ID, which LangGraph keeps when it runs the task again
    from its checkpoint.
    """
    config = var_child_runnable_config.get() or {}
    return (config.get("metadata") or {}).get("langgraph_checkpoint_ns")


def _entries(key: str) -> tuple[str, str, str]:
    """Return the cache entries a call under ``key`` can write."""
    return key, f"{key}:command", f"{key}:types"


def _retire(keys: list[str]) -> None:
    """Mark the results of the calls under ``keys`` stale."""
    cache, stale = _task_cache.get(), _stale.get()
    if cache is not None and stale is not None:
        for key in keys:
            stale.update(entry for entry in _entries(key) if entry in cache)


def _reclaim(keys: list[str]) -> None:
    """Take back the places of the calls under ``keys`` that are still free.

    LangGraph restores those calls from a checkpoint without making them,
    so they keep their places, and their results stay for the next run.
    """
    free = _free_places.get()
    if free is None:
        return
    for key in keys:
        base, sep, n = key.rpartition("#")
        base, place = (base, int(n)) if sep else (key, 0)
        places = free.get(base)
        if places and place in places:
            places.remove(place)
            heapify(places)


def _restart(keys: list[str]) -> None:
    """Count the calls under ``keys`` from their first place again."""
    uses, free = _key_uses.get(), _free_places.get()
    for key in keys:
        base, sep, _ = key.rpartition("#")
        base = base if sep else key
        if uses is not None:
            uses.pop(base, None)
        if free is not None:
            free.pop(base, None)


def get_task_cache() -> dict[str, Any] | None:
    """Get the task result cache for the current context."""
    return _task_cache.get()


def task_id(func: Any) -> str:
    """Return the fully-qualified module.qualname for a function.

    Raises ValueError for functions that cannot be identified unambiguously
    (lambdas, closures, __main__ functions).
    """
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)

    if module is None or qualname is None:
        raise ValueError(
            f"Cannot identify task {func}: missing __module__ or __qualname__. "
            "Tasks must be defined at module level."
        )
    if module == "__main__":
        raise ValueError(
            f"Cannot identify task {qualname}: defined in __main__. "
            "Tasks must be importable from a named module."
        )
    if "<locals>" in qualname:
        raise ValueError(
            f"Cannot identify task {qualname}: closures/local functions are not supported. "
            "Tasks must be defined at module level."
        )
    return f"{module}.{qualname}"


def _canonical(value: Any) -> Any:
    """Return a JSON form of ``value`` that is the same in every process.

    ``str()`` can depend on set order or object identity, so models,
    dataclasses and sets give their class and contents instead. The class
    keeps them apart from a plain dict or list with the same contents.
    """
    cls = type(value)
    try:
        if isinstance(value, (set, frozenset)):
            contents: Any = sorted(
                value, key=lambda item: dumps(item, sort_keys=True, default=_canonical)
            )
        elif isinstance(value, BaseModel):
            contents = dict(value)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            contents = {
                f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)
            }
        else:
            return str(value)
    except Exception:
        return str(value)
    return {"__class__": f"{cls.__module__}.{cls.__qualname__}", "__value__": contents}


def _stable(value: Any, enclosing: frozenset[int] = frozenset()) -> Any:
    """Return a JSON form of ``value`` that is the same in every process.

    Containers, enums, dataclasses and Pydantic models give their class and
    contents; numbers, dates, UUIDs and paths their class and string form;
    functions and classes their qualified name. Any other object gives its
    class alone, since its ``str()`` can show its address, which differs on
    another Worker: calls that differ only in such values are told apart by
    their order. Unlike :func:`cache_key`, which keeps the form earlier
    versions wrote, this never falls back to ``str()``.
    """
    if value is None or (
        isinstance(value, (bool, int, float, str)) and not isinstance(value, Enum)
    ):
        return value
    cls = type(value)
    name = f"{cls.__module__}.{cls.__qualname__}"
    if isinstance(value, Enum):
        return [name, _stable(value.value, enclosing)]
    if isinstance(value, (bytes, bytearray)):
        return [name, value.hex()]
    if isinstance(value, (Number, date, time, timedelta, UUID, PurePath)):
        return [name, str(value)]
    if callable(value) and hasattr(value, "__qualname__"):
        return [name, f"{getattr(value, '__module__', None)}.{value.__qualname__}"]
    if id(value) in enclosing:
        return [name]
    enclosing |= {id(value)}
    if isinstance(value, BaseModel):
        return [name, [[k, _stable(v, enclosing)] for k, v in value]]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            name,
            [
                [f.name, _stable(getattr(value, f.name, None), enclosing)]
                for f in dataclasses.fields(value)
            ],
        ]
    if isinstance(value, Mapping):
        items = [
            [_stable(k, enclosing), _stable(v, enclosing)] for k, v in value.items()
        ]
        return [name, sorted(items, key=dumps)]
    if isinstance(value, (list, tuple, deque)):
        return [name, [_stable(item, enclosing) for item in value]]
    if isinstance(value, (set, frozenset)):
        return [name, sorted((_stable(item, enclosing) for item in value), key=dumps)]
    return [name]


def _scope_key(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Return the key that calls of the workflow-side ``name`` with this input share."""
    form = dumps(["call_scope", name, _stable(args), _stable(kwargs)])
    # Hex, like a cache key, so release() can split off the place. The form
    # never matches the one cache_key() hashes for an Activity call.
    return sha256(form.encode()).hexdigest()[:32]


def cache_key(
    task_id: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    context: Any = None,
) -> str:
    """Build a cache key from the full task identifier, arguments, and runtime context."""
    try:
        key_str = dumps(
            [task_id, args, kwargs, context], sort_keys=True, default=_canonical
        )
    except (TypeError, ValueError):
        key_str = repr([task_id, args, kwargs, context])
    return sha256(key_str.encode()).hexdigest()[:32]


def cache_lookup(key: str) -> tuple[bool, Any]:
    """Return (True, value) if cached, (False, None) otherwise.

    A stale entry (see :func:`call_scope`) misses: the call is being made
    again, so its old entries are dropped and it runs.
    """
    cache = _task_cache.get()
    if cache is None or key not in cache:
        return False, None
    stale = _stale.get()
    if stale and key in stale:
        for entry in _entries(key.split(":", 1)[0]):
            cache.pop(entry, None)
            stale.discard(entry)
        return False, None
    return True, cache[key]


def cache_put(key: str, value: Any) -> None:
    """Store a value in the task result cache."""
    cache = _task_cache.get()
    if cache is not None:
        cache[key] = value
