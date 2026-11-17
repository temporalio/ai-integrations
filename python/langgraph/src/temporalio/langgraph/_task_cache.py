"""Task result cache for continue-as-new support.

Caches task results by (task id, args, kwargs, context) hash so that previously
completed tasks are not re-executed after a continue-as-new. The cache state
is a plain dict that can travel through workflow.continue_as_new(). Each
Workflow run has its own cache, and only what an earlier run handed over is
served, each entry once, so a call repeated in one run executes again.
"""

from __future__ import annotations

import asyncio
import dataclasses
from hashlib import sha256
from json import dumps
from typing import Any

from pydantic import BaseModel

# The attribute of a Workflow's event loop that holds its run's cache. The loop
# is the Workflow instance, so __init__, run and the handlers of one run share
# a cache, no other Workflow and no Replayer's instance of the same run sees
# it, and it is freed with the instance.
_RUN_CACHE = "_temporal_langgraph_task_cache"


class _RunCache:
    """The task cache of one Workflow run.

    Both dicts map a call's key to its entries in the cache dict: its result
    or Command, and its value types. ``carried`` holds what an earlier run
    handed over and this run has not served yet. ``results`` holds the served
    entries and this run's own results.
    """

    def __init__(self) -> None:
        self.carried: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any]] = {}


def _run_cache() -> _RunCache:
    loop = asyncio.get_running_loop()
    run: _RunCache | None = getattr(loop, _RUN_CACHE, None)
    if run is None:
        run = _RunCache()
        setattr(loop, _RUN_CACHE, run)
    return run


def set_task_cache(cache: dict[str, Any] | None) -> None:
    """Add a cache from an earlier run to the current run's task cache.

    Entries of a call the run already holds are skipped, so a cache passed
    again is not served twice and never replaces a newer result.
    """
    run = _run_cache()
    calls: dict[str, dict[str, Any]] = {}
    for name, value in (cache or {}).items():
        # A call's entries are named "<key>", "<key>:command" and "<key>:types".
        calls.setdefault(name.partition(":")[0], {})[name] = value
    for key, entries in calls.items():
        if key not in run.results:
            run.carried.setdefault(key, entries)


def get_task_cache() -> dict[str, Any]:
    """Get the current run's task cache as one dict for the next run."""
    run = _run_cache()
    return {
        name: value
        for entries in (*run.carried.values(), *run.results.values())
        for name, value in entries.items()
    }


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


def cache_lookup(key: str) -> dict[str, Any] | None:
    """Return the entries an earlier run cached for the call ``key``, or None.

    They are served once: they move to this run's results, so they still
    travel to the next run, and a repeat of the call executes.
    """
    run = _run_cache()
    entries = run.carried.pop(key, None)
    if entries is not None:
        run.results[key] = entries
    return entries


def cache_put(key: str, entries: dict[str, Any]) -> None:
    """Store this run's result of the call ``key``, replacing an earlier one."""
    run = _run_cache()
    run.carried.pop(key, None)
    run.results[key] = entries
