"""Task result cache for continue-as-new support.

Caches task results by (module.qualname, args, kwargs) hash and by how many
identical calls the same caller made before in the run, so that previously
completed calls are not re-executed after a continue-as-new when the run makes
them again in the same order, while a call repeated within a run executes
again. The cache state is a plain dict that can travel through
workflow.continue_as_new().
"""

from __future__ import annotations

import dataclasses
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from heapq import heappop, heappush
from json import dumps
from typing import Any

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


def set_task_cache(cache: dict[str, Any] | None) -> None:
    """Set the task result cache for the current context and restart call counting."""
    _task_cache.set(cache)
    _key_uses.set({})
    _free_places.set({})


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


@contextmanager
def call_scope(name: str) -> Generator[None, None, None]:
    """Count the calls made in this block per call of the workflow-side ``name``.

    The scope is a hash of the enclosing scope, ``name`` and its place among
    the enclosing scope's calls of ``name``, taken before the call awaits
    anything. So it is the same in every run that makes the same calls in the
    same order, whatever order parallel calls finish in. If the block raises,
    its place is given back, as a failed call's is. If it fails, rather than
    being interrupted, the results and places of the calls made in it are
    dropped too, so a retry makes them again under the same keys and, after
    continue-as-new, the first call gets the retry's results.
    """
    # Hex, like a cache key, so release() can split off the place.
    key = occurrence_key(sha256(name.encode()).hexdigest()[:32])
    taken: list[str] = []
    token = _scope.set(sha256(key.encode()).hexdigest()[:16])
    taken_token = _taken.set(taken)
    try:
        yield
    except BaseException as err:
        if isinstance(err, Exception) and not isinstance(err, GraphBubbleUp):
            _forget(taken)
        release(key)
        raise
    finally:
        _taken.reset(taken_token)
        _scope.reset(token)
        outer = _taken.get()
        if outer is not None:
            outer.extend(taken)


def _forget(keys: list[str]) -> None:
    """Drop the cached results and the places of the calls under ``keys``."""
    cache, uses, free = _task_cache.get(), _key_uses.get(), _free_places.get()
    for key in keys:
        base, sep, _ = key.rpartition("#")
        base = base if sep else key
        if uses is not None:
            uses.pop(base, None)
        if free is not None:
            free.pop(base, None)
        if cache is not None:
            for entry in (key, f"{key}:command", f"{key}:types"):
                cache.pop(entry, None)


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
    """Return (True, value) if cached, (False, None) otherwise."""
    cache = _task_cache.get()
    if cache is not None and key in cache:
        return True, cache[key]
    return False, None


def cache_put(key: str, value: Any) -> None:
    """Store a value in the task result cache."""
    cache = _task_cache.get()
    if cache is not None:
        cache[key] = value
