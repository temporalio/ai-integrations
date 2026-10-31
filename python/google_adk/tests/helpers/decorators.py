"""Decorator defined outside the modules of the functions it wraps."""

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any


def passthrough(
    fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Wraps ``fn`` with ``functools.wraps``; the wrapper has this module's globals."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    return wrapper


def unwrapped(
    fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Like ``passthrough``, but without ``__wrapped__``: ``fn``'s globals are out of reach."""
    wrapper = passthrough(fn)
    delattr(wrapper, "__wrapped__")
    setattr(wrapper, "__signature__", inspect.signature(fn))
    return wrapper
