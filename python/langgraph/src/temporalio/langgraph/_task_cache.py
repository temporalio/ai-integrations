"""Task result cache for continue-as-new support.

Caches task results by (module.qualname, args, kwargs) hash and by how many
identical calls came before in the run, so that previously completed calls
are not re-executed after a continue-as-new when the run makes them again in
the same order, while a call repeated within a run executes again. The cache
state is a plain dict that can travel through workflow.continue_as_new().
"""

from __future__ import annotations

import dataclasses
from contextvars import ContextVar
from hashlib import sha256
from json import dumps
from typing import Any

from pydantic import BaseModel

_task_cache: ContextVar[dict[str, Any] | None] = ContextVar(
    "_temporal_task_cache", default=None
)
# How many times each cache key has been used since set_task_cache(). It is
# not part of the cache, so it does not travel through continue-as-new.
_key_uses: ContextVar[dict[str, int] | None] = ContextVar(
    "_temporal_task_key_uses", default=None
)


def set_task_cache(cache: dict[str, Any] | None) -> None:
    """Set the task result cache for the current context and restart call counting."""
    _task_cache.set(cache)
    _key_uses.set({})


def occurrence_key(key: str) -> str:
    """Return the cache key for this occurrence of a call in the current run.

    The first call with ``key`` keeps it, so caches written by earlier
    versions still match. The n-th repeat gets ``"<key>#<n>"``, so it does not
    find an earlier call's result and runs again, while the same calls made in
    the same order after continue-as-new find theirs.
    """
    uses = _key_uses.get()
    if uses is None:
        return key
    n = uses.get(key, 0)
    uses[key] = n + 1
    return key if n == 0 else f"{key}#{n}"


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
